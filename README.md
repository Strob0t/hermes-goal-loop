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
   check; the model shows the list and asks the owner. `goal_go` records the go. No loop
   runs before the go. See **Go rule** below.
2. **Run to the end.** The generic `pre_stop` hook keeps the turn going while open goals
   remain, with no file edit, bounded by `agent.max_stop_continuations` (default 20). State
   is keyed by chat, not by session id, so it survives a context compaction.
3. **Judge with a fresh context.** `goal_claim` submits a structured completion claim; a
   read-only child agent (fallback: a tool-less LLM call) checks the actual state, never
   trusts the worker's report, and gives one actionable reason for each rejected goal.
   A judge call that fails rejects the claim (fail closed).
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
| `goal_set` set or changed the list (before or after the go) | the full list with each check and status |
| after the go: a goal became blocked (`goal_block`) or a new owner decision (`goal_decision`) | only these items (`Goal status (needs the owner):`), each one time |
| progress, a goal in work, a judge rejection, a confirmed goal while others are open | nothing |
| the judge confirmed the last goal | the full list with checkmarks, one time, with the final report |
| after the final list, until a new `goal_set` | nothing |

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
`gate_<session id>.json` in the state folder, because `pre_llm_call` gets the session id
only.

## Usage hint

The plugin adds a short section to the system prompt of each new session
(`PluginContext.register_system_prompt_section`, rendered by
`agent/system_prompt.py` `_frozen_plugin_prompt_sections`): for a multi-step task, call
`goal_set` first, then show the list and wait for the owner's go. The section is empty, and
so not rendered, for the platforms in `EXCLUDED_PLATFORMS` (`cron`, `subagent`, which
includes the judge). Hermes renders the section one time per session, so a session that
started before the plugin update gets the hint only after a new session.

## Tools

`goal_set`, `goal_go`, `goal_status`, `goal_claim`, `goal_decision`, `goal_block`.

## Hooks

`pre_stop` (continuation), `transform_llm_output` (status display) and `pre_llm_call`
(owner-message count for the go rule). Plus one system-prompt section (`goal-loop`).

## Requires

- Hermes `>= 0.23.2`.
- The companion fork patch that adds the generic `pre_stop` stop-gate hook
  (`patch_stop_hook.py` on the `goal-loop-patches` branch of the Hermes fork). The plugin
  degrades to tools-only without it.

## Config

- `agent.max_stop_continuations` — bound on consecutive `pre_stop` continuations per turn
  (default 20).
- `GOAL_LOOP_STATE_DIR` — directory for per-chat state and ledger files
  (default `$HERMES_HOME/goal-loop`).

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
`in_place: true`. The go gate is keyed on the session id: a rotation between `goal_set`
and the go needs one more owner message.

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
