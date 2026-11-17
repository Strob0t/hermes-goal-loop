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
   runs before the go.
2. **Run to the end.** The generic `pre_stop` hook keeps the turn going while open goals
   remain, with no file edit, bounded by `agent.max_stop_continuations` (default 20). State
   is keyed by chat, not by session id, so it survives a context compaction.
3. **Judge with a fresh context.** `goal_claim` submits a structured completion claim; a
   read-only child agent (fallback: a tool-less LLM call) checks the actual state, never
   trusts the worker's report, and gives one actionable reason for each rejected goal.
   A judge call that fails rejects the claim (fail closed).
4. **Done stays done.** Only the judge sets a goal `done`; a status block is appended to
   each answer so every report shows all goals.
5. **Decisions as text.** `goal_decision` records an owner decision (options + a
   recommendation); the status block renders the decisions as a numbered text list — no
   card, no modal.

## Tools

`goal_set`, `goal_go`, `goal_status`, `goal_claim`, `goal_decision`, `goal_block`.

## Hooks

`pre_stop` (continuation) and `transform_llm_output` (status block).

## Requires

- Hermes `>= 0.23.2`.
- The companion fork patch that adds the generic `pre_stop` stop-gate hook
  (`patch_stop_hook.py` on the `goal-loop-patches` branch of the Hermes fork). The plugin
  degrades to tools-only without it.

## Config

- `agent.max_stop_continuations` — bound on consecutive `pre_stop` continuations per turn
  (default 20).
- `GOAL_LOOP_STATE_DIR` — directory for per-chat state and ledger files
  (default `~/.hermes/goal-loop`).

## Tests

```
python -m unittest discover -s tests -p 'test_*.py'
```

The integration test (`integration/`) needs a Hermes source tree with the companion patch
applied on `sys.path`; it is skipped by the unit run above.

## Licence

Plugin code: see repository licence. Parts of the completion-claim validation are ported
from `william-ricchiuti/OpenCode-goal-plugin` (MIT) — see `THIRD_PARTY.md`.
