# hermes-goal-loop

A goal loop for [Hermes Agent](https://hermes-agent.nousresearch.com) delivered as a
plugin. It makes a chat set its goals before the work, run to the end without a
babysitter, verify each goal with a fresh-context judge that reads the real state, keep a
confirmed goal done, and list open owner decisions as plain text.

The loop is built for the API server / `POST /v1/runs` surface, where the native `/goal`
slash loop does not run. The continuation itself is owned by a small, generic `pre_stop`
stop-gate hook added by a companion patch to Hermes (see **Requires** below); without that
patch the plugin's tools still work and the hook is simply never called.

## The five goals

1. **Set goals and wait for the go.** `goal_set` records each goal with one falsifiable
   check of the content, in the owner's language; the plugin shows the list and the go
   question one time. `goal_go` records the go. No loop runs before the go. See **Go rule**.
2. **Run to the end.** The generic `pre_stop` hook keeps the turn going while open goals
   remain, bounded per turn by `agent.max_stop_continuations` (default 20) and per goal by
   `budget_per_goal` over all turns and runs. Idle turns and repeated judge rejections first
   ask the model to plan again, then pause the goal with an owner decision. State is keyed by
   chat, not by session id, so it survives a context compaction; each owner turn gets the open
   goals as context.
3. **Judge with a fresh context.** `goal_claim` submits a structured completion claim; a
   read-only child agent on its own model task `goal_loop_judge` (fallback: a tool-less LLM
   call on the same task) checks the actual state, never trusts the worker's report, and gives
   the missing points for each rejected goal. A `pre_tool_call` guard vetoes writes of the
   judge child. A judge that does not answer keeps the goal status; `wait` keeps it open
   without nudges.
4. **Done stays done.** Only the judge sets a goal `done`. The chat shows the goal list
   quietly (see **What the chat shows** below).
5. **Decisions as text.** `goal_decision` records an owner decision (options + a
   recommendation); the chat shows the decisions as a numbered text list — no card, no
   modal.

## What the chat shows

The `transform_llm_output` hook appends a block to the final answer of a turn only when
there is news for the owner:

| Event in the turn | Block in the answer |
|---|---|
| `goal_set` set the list (before the go) | the full list with each check and status, then the go question as the last line; the model does not repeat the list |
| `goal_set` changed the list after the go (owner-approved) | the full list with each check and status |
| after the go: a goal became blocked or paused, or a new owner decision | only these items (`Goal status (needs the owner):`), each one time |
| progress, a goal in work, a judge rejection, a confirmed goal while others are open | nothing |
| the turn ends after the go while goals are still open | one line: a follow-up run continues the loop (`resume`), else the loop is paused, write to continue (`paused_hint`) |
| the judge confirmed the last goal | the full list with checkmarks, one time, with the final report |
| after the final list, until a new `goal_set` | only a new owner decision, one time |

Labels follow the owner language (`language`: `auto` from the owner messages, `de` or `en`).
Goal text and check are written by the model in the owner language (usage hint).

The state file records what the chat has shown (`list_changed`, `final_shown`,
`decisions_shown`, `block_shown` for each goal). A state file of an older plugin version
without these fields loads without error; the missing fields count as "already shown", so
an update does not repeat old items in a running chat.

## Show the list on request: `goal_status`

`goal_status` returns the full list (each goal with its check and status, then the open
decisions) as finished text. Its description tells the model to give this text to the
owner unchanged, word for word. Ask the chat for the goal list; the model calls
`goal_status` and passes the text through.

There is no `/goals` slash command. Hermes v0.23.2 has a plugin command API
(`PluginContext.register_command`, `hermes_cli/plugins_context.py`), but on the path
Open WebUI → pipe → `POST /v1/runs` a slash text never reaches a plugin command handler:

- The pipe handles only `/reset`, `/approve` and `/deny` itself (`SLASH_COMMANDS` in
  `functions/hermes_runs_pipe.py` of `hermes-openwebui-bridge`) and sends every other text
  as the run `input`.
- With the session authority, `gateway/session_ingress.py` sets
  `event.allow_gateway_control = False` for each API row, and
  `MessageEvent.is_command()` (`gateway/platforms/event.py`) is false without it, so
  `get_command()` gives `None` and `_hm_dispatch_plugin_command`
  (`gateway/run_inbound_hooks.py`) never runs.
- Without the session authority, `/v1/runs` creates the agent and calls
  `agent.run_conversation` directly (`gateway/platforms/api_server_runs.py`, `_execute_run`
  and `_run_agent_sync`), with no command dispatch.

## Go rule

`goal_go` is accepted only when an owner message came into the chat after the last
`goal_set` of the session. Otherwise the tool returns an error text, the go is not
recorded and the loop does not start. A new `goal_set` needs a new owner message.

The plugin counts owner messages with the `pre_llm_call` hook. Hermes fires it one time
for each incoming turn (`agent/turn_context.py`, `_collect_pre_llm_call_context`) and
never for a `pre_stop` continuation inside a turn. An API client turn counts also when
the session authority stores its user row as `internal_notification` with
`display_metadata.api_inbound` (`gateway/session_ingress.py`). These turns do not count: a
gateway notification (the text ends with the `[INTERNAL NOTIFICATION ...]` footer), a turn
whose user row has another synthesized `display_kind` (a typed `/steer` row counts), an
empty turn, and the `cron` and `subagent` platforms. The count is kept in
`gate_<chat>.json` in the state folder, keyed on the chat (see "Where the state lives").
After the go, a changed goal list needs one more owner message (`change_<chat>.json`), and a
paused goal resumes with `goal_go` after a new owner message.

## Usage hint

The plugin adds a short section to the system prompt of each new session
(`PluginContext.register_system_prompt_section`, rendered by
`agent/system_prompt.py` `_frozen_plugin_prompt_sections`): goal text and check in the
owner's language, content checks that the judge can examine at claim time, the live version
as the base, large goals split, no repeated list and no own go question, `goal_go` only after
the owner's go, progress with `goal_progress`. The section is empty, and
so not rendered, for the platforms in `EXCLUDED_PLATFORMS` (`cron`, `subagent`, which
includes the judge). Hermes renders the section one time per session, so a session that
started before the plugin update gets the hint only after a new session.

## Tools

`goal_set`, `goal_go`, `goal_status`, `goal_claim`, `goal_progress`, `goal_decision`, `goal_block`.

- `goal_set` after the go is refused until an owner message approves the change; judge-confirmed goals stay done.
- `goal_block` needs a reason and is accepted only when the same obstacle stays for `block_min_rounds` continuation rounds.
- `goal_progress` records the last step, its evidence and the next step; the goal context of each turn shows it.
- `goal_go` also resumes paused goals (budget, idle, rejections, stop word, restart).

## Hooks

- `pre_stop`: continuation with the open goals (text, check, progress), budget, idle and restart limits.
- `transform_llm_output`: status display in the owner language; one line when a turn ends with open goals.
- `pre_llm_call`: owner-message count (go and change gates), stop words, language votes, goal context for each owner turn.
- `post_tool_call`: tool-call count for the idle detection.
- `pre_tool_call`: read-only guard for the judge child.

Plus one system-prompt section (`goal-loop`) and the auxiliary task `goal_loop_judge`.

## Requires

- Hermes `>= 0.23.2`.
- The companion fork patch that adds the generic `pre_stop` stop-gate hook
  (`patch_stop_hook.py` on the `goal-loop-patches` branch of the Hermes fork). The plugin
  degrades to tools-only without it.

## Config

- `agent.max_stop_continuations` — bound on consecutive `pre_stop` continuations per turn
  (default 20).
- `auxiliary.goal_loop_judge` — provider and model of the judge (subagent and fallback path). Use a
  model family that is not the worker's (self-preference bias, arXiv 2404.13076).
- `plugins.entries.goal-loop.settings` (all optional):

  | Key | Default | Effect |
  |---|---|---|
  | `budget_per_goal` | 50 | continuations per goal over all turns and runs; then pause + decision (0 = off) |
  | `idle_replan_after` | 2 | continuations without a tool call before a request to plan again |
  | `idle_pause_after` | 3 | continuations without a tool call before a pause + decision (0 = off) |
  | `reject_replan_after` | 2 | similar judge rejections before a request to plan again |
  | `reject_pause_after` | 3 | similar judge rejections before a pause + decision (0 = off) |
  | `block_min_rounds` | 3 | rounds with the same obstacle before `goal_block` is accepted |
  | `judge_timeout` | 300 | seconds the judge child may run; a timeout keeps the goal status |
  | `language` | `auto` | `auto` (from the owner messages), `de` or `en` for the plugin labels |
  | `paused_hint` | true | one line when a turn ends with open goals after the go and no follow-up run is scheduled |
  | `stop_words` | stopp, stop, halt, pause, pausiere, abbrechen | an owner message that is one of these pauses the loop |
  | `resume` | true | follow-up runs (see **Follow-up runs**) |
  | `resume_delay` | 1m | delay of a follow-up run after the end of a run |
  | `resume_max` | 20 | follow-up runs per go; then pause + decision (0 = off) |
- `GOAL_LOOP_STATE_DIR` — directory for per-chat state and ledger files
  (default `$HERMES_HOME/goal-loop`).

## Follow-up runs

A run ends at the turn limit of Hermes (`agent.max_turns`) or after
`agent.max_stop_continuations`. When a run of an `api_server` chat ends after the go while goals
are still open, the plugin schedules one more run of the same session, without an owner message:

1. At the end of the turn (`transform_llm_output`) the plugin writes a request file
   `resume_<chat>.json` (session id, model, a nonce), a small wrapper script
   `$HERMES_HOME/scripts/goal-loop-resume-<hash>.py` and a one-shot `no_agent` cron job
   (`cron.jobs.create_job`, schedule `in <resume_delay>`, `repeat` 1). The answer gets one line:
   the loop continues in a follow-up run.
2. The cron run executes `goal_loop/resume_run.py`. It posts nothing when the go was taken back,
   no goal is open, Hermes started again after the request, or the request was used. Else it
   posts one turn to the local API server with `X-Hermes-Session-Id` and `API_SERVER_KEY` (the
   pattern of `gateway/wake.py` `_self_post_chat_completion`) and prints the answer.
3. The printed answer is the cron run output. A cron delivery adapter that binds the job to the
   chat where it was made shows it in that chat (homelab: `ki-delivery`, Open WebUI chat API).

The follow-up turn starts with `[goal-loop resume <nonce>]`. `pre_llm_call` never counts it as an
owner message, so it gives no go and no change approval; it gets the goal context. Limits:
`budget_per_goal`, the idle pause, the stop word, `resume_max`, one pending request per chat. A
new go resets the count. Hermes must have `API_SERVER_KEY` (file `/run/secrets/hermes_api_key`
or env) and cron scripts enabled.

## Where the state lives

Per-chat state and ledger files live under **`$HERMES_HOME/goal-loop`** of the Hermes
home that loads the plugin, so every home and profile has its own goal state — a test
home can never read or overwrite another home's goals. `GOAL_LOOP_STATE_DIR` overrides
the folder (tests, special setups); the folder is fixed once, when the plugin loads.

Hermes compacts a context **in place by default** (`compression.in_place: true`, see
`hermes_cli/config_defaults.py`): the session id does not change and the goal state
follows the chat either way. With the rotating path (`compression.in_place: false`)
the session id changes at every compaction; the plugin then keys on the
compression-stable `conversation_id` (with the session id as fallback), so the loop
continues in the rotated session. Keying on `conversation_id` is harmless with
`in_place: true`. The go gate is keyed on the chat as well: `pre_llm_call` gets no
`conversation_id` in the fork, so the plugin maps the session to its chat (each goal tool
call and each `pre_stop` call records the mapping). A rotation between `goal_set` and the go
needs one more owner message only when no tool call of the new session ran before it.

## Tests

```
python -m unittest discover -s tests -p 'test_*.py'
```

The integration test (`integration/`) needs a Hermes source tree with the companion patch
applied on `sys.path`; it is skipped by the unit run above. Run it from the fork root:

```
cd <hermes-agent goal-loop-patches>
PYTHONPATH=<this repo> GOAL_LOOP_STATE_DIR=<tmp> python -m unittest integration.test_goal_loop_integration -v
```

## Licence

Plugin code: see repository licence. Parts of the completion-claim validation are ported
from `william-ricchiuti/OpenCode-goal-plugin` (MIT) — see `THIRD_PARTY.md`.
