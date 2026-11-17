# Third-party code

## OpenCode-goal-plugin (MIT)

- Source: `william-ricchiuti/OpenCode-goal-plugin` @ commit `1ae8144a`
  (https://github.com/william-ricchiuti/OpenCode-goal-plugin), MIT licence.
- What was reused: the design of the structured completion claim and its validation
  (summary + criteria-with-evidence + checks-with-result/exitCode, a failed check rejects
  the claim, item/length caps) and the fresh-context, default-deny, fail-closed verifier
  pattern.
- How: re-implemented in Python for the Hermes plugin API (`goal_loop/claim.py`,
  `goal_loop/judge.py`). No source file was copied verbatim; the JavaScript was read as a
  reference for behaviour.

The MIT licence requires its notice to travel with substantial reuse; this file is that
notice. A full copy of the upstream MIT text is at the source repository above.
