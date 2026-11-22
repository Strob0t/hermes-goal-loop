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
   check of the content, in the owner's language; the model shows the list and asks for the
   go in the owner's language (see **Owner language**). `goal_go` records the go. No loop
   runs before the go. See **Go rule**.
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

## Owner language (0.4.11)

The plugin appends no text to the answer: `transform_llm_output` returns None. All plugin
settings, prompts and tool texts are English. What the owner must see goes into
`state.display_note` at the end of the turn; the next `pre_llm_call` gives it to the model as
English context with the instruction: at the end of your answer, show this to the owner in the
owner's language, in your own words; keep the goal ids and decision numbers. Each item goes into
the context one time. While goals are set and the owner has not given the go, the context also
tells the model to ask the owner for `go` in the owner's language; the `goal_set` tool answer says
the same.

## What the chat shows

The model shows these items in the owner's language (context of the next turn, see
**Owner language**), only when there is news for the owner:

| Event in the turn | Display for the owner |
|---|---|
| `goal_set` set the list (before the go) | the full list with each check and status, and the question for the go |
| `goal_set` changed the list after the go (owner-approved) | the full list with each check and status |
| after the go: a goal became blocked or paused, or a new owner decision, while another goal is actionable | nothing; the item goes into the deferred list (0.4.10) |
| after the go: no goal is actionable any more | the deferred list (`Goal status (needs the owner):`, decisions with options, blocked goals with reason, the pending list change), each item one time |
| progress, a goal in work, a judge rejection, a confirmed goal while others are open | nothing |
| the turn ends after the go while goals are still open | nothing; the model gets a note for the next turn: a follow-up run continues the loop (`resume`), else the loop is paused (`paused_hint`) |
| the judge confirmed the last goal | the full list with checkmarks, one time, with the final report |
| after the final list, until a new `goal_set` | only a new owner decision, one time |

Goal text, check and decision text stay as the model wrote them (in the owner's language, usage
hint). Lines that explain the loop state (wait, pause, restart, follow-up run) also go to the model
as context, and the model writes them in the owner's language at the end of its answer.

## Owner messages (0.4.10)

- Each owner message pauses the loop while it has a go and open goals: `go` becomes false, a
  pending follow-up run is cancelled, the background and owner waits are cleared, ledger
  `paused_by_owner_message`. While paused, `pre_stop` does not continue, no follow-up run is
  scheduled or injected, and the background watchdog does not continue the chat.
- A message with the word `go` as a token (case-insensitive, alone or with other text, for example
  `4B 5C go`) keeps the go, or gives it again after a pause.
- The paused turn gets a context note: answer the message, put any question at the end, ask the
  owner to write `go`. `goal_go` is refused until an owner message with `go` comes in.
- Decisions close by number with an option (`4B`, `8. A`), by `go`, or by explicit consent
  (`ok`, `recommendation`, `all your recommendations`).
- Owner words in other languages: the optional file `<state dir>/vocabulary.json`, for example
  `{"go_words": ["onward"], "consent_patterns": ["^\\s*fine\\s*$"]}`. Without the file only the
  English defaults apply.

## Deferred list (0.4.10)

`goal_decision` takes an optional `goal_id`. A goal with an open linked decision, or with status
blocked or paused, waits for the owner. The other unfinished goals are actionable (a goal that waits
for a background process or a delegation is actionable later). While a goal is actionable, a new
decision, a `goal_block` and a refused `goal_set` change go into `state.deferred` (with time and
text); the tool answer names the goals to continue with, the goal display does not show the items,
and `pre_stop` continues, also after an answer with the stop cause `owner-question`. When no goal is
actionable, the goal display shows the deferred list one time and the loop waits for the owner.

## Evidence (0.4.10)

`goal_decision` requires `evidence`: the command or file that showed the live state, and its time.
Without it the tool records nothing. The prompt section and the pause note say: check each point
against the live state in this turn; do not rely on the chat history or memory.

The state file records what the chat has shown (`list_changed`, `final_shown`,
`decisions_shown`, `block_shown` for each goal). A state file of an older plugin version
without these fields loads without error; the missing fields count as "already shown", so
an update does not repeat old items in a running chat.

## Show the list on request: `goal_status`

`goal_status` returns the full list (each goal with its check and status, then the open
decisions) as English text. Its description tells the model to show the result to the
owner in the owner's language and keep each goal id. Ask the chat for the goal list; the
model calls `goal_status` and shows the list.

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
empty turn, a turn that starts with `[Your active task list was preserved across context compression]` (the task
list that Hermes restores after a compression, 0.4.4), and the `cron` and `subagent` platforms. An owner message
that interrupts a running turn counts through the `pre_api_request` hook (0.4.4). The count is kept in
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

- `goal_set` after the go is refused until an owner message approves the change (while a goal is actionable, the change goes into the deferred list); judge-confirmed goals stay done.
- `goal_decision` requires `evidence` and takes an optional `goal_id` (the goal that waits for it).
- `goal_block` needs a reason and is accepted only when the same obstacle stays for `block_min_rounds` continuation rounds.
- `goal_progress` records the last step, its evidence and the next step; the goal context of each turn shows it.
- `goal_go` also resumes paused goals (budget, idle, rejections, owner pause, restart); it needs an owner message with `go`.

## Hooks

- `pre_stop`: continuation with the open goals (text, check, progress), budget, idle and restart limits.
- `transform_llm_output`: status display (English labels); the loop-state note for the model when a turn ends with open goals.
- `pre_llm_call`: owner-message count (go and change gates), the owner pause, the vocabulary file, goal context and loop-state notes for each owner turn.
- `pre_api_request` (0.4.4): counts an owner message that interrupted a running turn (Hermes applies it as a
  user row with the api_content prefix `[Context from the interrupted assistant response]` and starts no new
  turn, so `pre_llm_call` does not run); each such row counts one time (ledger `owner_redirect_counted`).
- `post_tool_call`: tool-call count for the idle detection (`todo_list` does not count, 0.4.4).
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
  | `judge_timeout` | 300 | total seconds of all judge attempts; a judge without an answer keeps the goal status |
  | `judge_attempts` | 3 | judge attempts before `judge_unavailable` (0.4.3); an answer without a VERDICT line counts as no answer |
  | `judge_attempt_timeout` | 120 | seconds of one judge attempt; a retry gets a shortened claim |
  | `judge_backoff` | 5 | pause before attempt n+1 (`judge_backoff * n` seconds); ledger `judge_retry` |
  | `stale_days` | 1 | at plugin start a state with a go and open goals without activity for this many days is paused (`paused_stale`, one decision; 0 = off) |
  | `paused_hint` | true | a note for the model when a turn ends with open goals after the go and no follow-up run is scheduled |
  | `resume` | true | follow-up runs (see **Follow-up runs**) |
  | `resume_delay` | 1m | delay of a follow-up run after the end of a run |
  | `resume_max` | 20 | follow-up runs per go; then pause + decision (0 = off) |
  | `restart_resume` | true | a turn that a Hermes restart cut off gets one continuation turn (see **Restart resume**) |
  | `restart_resume_delay` | 2m | delay of that continuation turn after the plugin loads |
- `GOAL_LOOP_STATE_DIR` — directory for per-chat state and ledger files
  (default `$HERMES_HOME/goal-loop`).

## Restart resume

Hermes v0.23.2 resumes a turn that a restart cut off only on push platforms; an API server
(Open WebUI) or local turn ends `interrupted` and is not run again. The plugin closes that gap:

1. Each turn start (`pre_llm_call`) writes `running_<session>.json` (session id, platform, the
   Open WebUI key `owui:<user>:<chat>` that Hermes bound to the session, model, pid and process
   start time); the end of the turn (`transform_llm_output`) removes it.
2. When the plugin loads, each note of another process that does not run any more is a cut-off
   turn. The plugin claims it (atomic rename) and schedules one continuation turn through the
   follow-up path (request `resume_restart_<session>.json`, reason `restart`). An Open WebUI chat
   is resumed through hermes-models with `X-Hermes-Session-Key`, so the session map and the
   delivery bind the answer to that chat; other sessions go to the local API server.
3. The turn text starts with `[goal-loop resume <nonce>]` and asks the model to continue the
   cut-off step; it is no owner message (no go). Notes older than 6 h are dropped. Ledger events:
   `restart_resume_scheduled`, `restart_resume_dropped`, `restart_resume_not_scheduled`.

## Turn limit (issue #1)

A turn can end at `agent.max_turns` while goals are open after the go; `pre_stop` does not run on that
path. The end of the turn (`transform_llm_output`) still runs, so the loop goes on from there:
`api_server` chats get the cron follow-up run (**Follow-up runs**); local chats (CLI, TUI, desktop) get
one follow-up turn through `PluginContext.inject_message`, which Hermes commits to the durable FIFO of
the session (an owner message goes first). This needs `plugins.entries.goal-loop.allow_gateway_injection:
true`; without it the ledger has `resume_inject_refused` and the answer ends with the paused line. A
follow-up turn and an owner message start a new idle count, so a cut at the limit is no idle round.
Limits: `resume_max`, `budget_per_goal`, the owner pause.

Since 0.4.2 the follow-up turns themselves are counted: a follow-up turn without a tool call is idle,
and `idle_pause_after` idle follow-up turns in a row pause the goals (`paused_idle`); only an owner
message resets this count. So a chat whose open goals only wait for owner decisions stops after a few
runs, not after `resume_max`. Each `pre_stop` continuation writes the ledger event `continued`
(`attempt`, open goal ids). The goal context of `pre_llm_call` carries its time ("Goals of this chat at
HH:MM"): Hermes stores this context with the owner message and sends it again in later turns, so an
old copy is visible as old.

## Wait barriers (0.4.3)

The loop does not push while other work or the owner is awaited. `pre_stop`, the local follow-up turn and
the cron follow-up run are skipped when (checked in this order, ledger event once per change):

- `yield_builtin_goal`: Hermes' own `/goal` is active in the session (`hermes_cli.goals.load_goal`); it
  drives the session.
- `waiting_background`: a background process of the session with `notify_on_complete` or watch patterns
  runs (`tools.process_registry`); its result starts the next turn.
- `waiting_delegation`: an asynchronous delegation of the session is live (`tools.async_delegation`).
- `waiting_owner`: every open goal got a `goal_block` in the latest round and a `goal_decision` was
  recorded since the last continuation.

The answer then ends with one wait line. An owner message lifts every wait. A missing Hermes module
counts as "no wait" (the loop is not stopped by an import error).

## Boot id (0.4.3)

The id that marks the process (restart detection, `restart_hold`) is a hash of the process id and the
process start time, not a random id at import. A plugin reload keeps it, so a reload does not hold
chats with a go any more.

## Parent session (goal adoption)

A chat without its own goal state takes over the state of its nearest ancestor session with goals
(`parent_session_id` in state.db, or the note `parent_<session>.json` that an importer writes, because
the Hermes import keeps a parent only for imported history of the same owner). The old state gets
`adopted_by` and `go: false`; both ledgers record the take-over (`adopted_from`, `adopted_by`).

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
`budget_per_goal`, the idle pause, the owner pause, `resume_max`, one pending request per chat. A
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
