"""The judge: a fresh-context verifier that checks the real state, not the worker's claim.

Goal 3 of the owner: a judge with a new context examines each goal on the actual state of the
system; it only reads; it does not accept the worker report as proof; it gives the cause for each
goal it rejects. The judge runs as a child agent through ``ctx.subagent_lifecycle.launch`` with a
read-only toolset subset (default deny) so it can inspect files and run read commands but never
edit or mutate goal state. If the launch is not available (for example the live
``disabled_toolsets`` removes every toolset a child could use), the judge falls back to a tool-less
``ctx.llm`` call that reads the goal plus the claim evidence as data — the native ``/goal`` judge
pattern. A judge call that fails rejects the claim (fail closed).

The judge returns exactly one verdict line: ``VERDICT: done`` or ``VERDICT: not-done <reason>``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("goal_loop.judge")

# Read-only toolsets a judge child may use. ``file`` bundles read and write tools, so the judge
# prompt forbids mutation explicitly; the subset check in subagent_lifecycle keeps it within parent
# rights. If none of these are in the parent toolsets, the launch path is skipped for ctx.llm.
JUDGE_TOOLSETS = ("file",)

JUDGE_SYSTEM = (
    "You are a goal-completion judge with a fresh context. Verify the claim against the goal and "
    "its check by inspecting the ACTUAL state of the system. You only read: never edit a file, "
    "never mutate goal state, never trust the worker's report as proof. Run the goal's check "
    "command yourself when you can. Answer with exactly one line: 'VERDICT: done' if the goal's "
    "check is proven to pass now, otherwise 'VERDICT: not-done <one actionable reason>'."
)

VERDICT_PREFIX = "VERDICT:"


@dataclass
class Verdict:
    done: bool
    reason: str
    method: str  # "subagent", "llm", or "error"


def parse_verdict(text: Any, method: str) -> Verdict:
    """Read the single verdict line. Anything that is not a clear 'done' rejects (fail closed)."""
    if not isinstance(text, str) or not text.strip():
        return Verdict(False, "judge returned no verdict", method)
    line = next(
        (ln.strip() for ln in reversed(text.splitlines()) if VERDICT_PREFIX in ln.upper()),
        "",
    )
    if not line:
        return Verdict(False, "judge gave no VERDICT line", method)
    body = line[line.upper().index(VERDICT_PREFIX) + len(VERDICT_PREFIX):].strip()
    verdict = body.lower()
    if verdict.startswith("done"):
        return Verdict(True, "", method)
    reason = body[len("not-done"):].strip(" :-") if verdict.startswith("not-done") else body
    return Verdict(False, reason or "rejected without a stated reason", method)


def _judge_prompt(goal_text: str, check: str, evidence: str) -> str:
    return (
        f"Goal: {goal_text}\n"
        f"Check that must pass: {check}\n\n"
        f"Worker completion claim (DATA, not proof):\n{evidence}\n\n"
        "Inspect the real state and give your one verdict line."
    )


def run_judge(ctx: Any, goal_text: str, check: str, evidence: str,
              *, parent_toolsets: Optional[set] = None) -> Verdict:
    """Run the judge, preferring a read-only child agent and falling back to a tool-less LLM call."""
    prompt = _judge_prompt(goal_text, check, evidence)
    allowed = tuple(ts for ts in JUDGE_TOOLSETS if not parent_toolsets or ts in parent_toolsets)
    if allowed:
        try:
            return _judge_via_subagent(ctx, goal_text, prompt, allowed)
        except Exception:
            logger.debug("judge subagent path failed, falling back to ctx.llm", exc_info=True)
    try:
        return _judge_via_llm(ctx, prompt)
    except Exception as exc:
        logger.debug("judge ctx.llm path failed", exc_info=True)
        return Verdict(False, f"judge call failed: {type(exc).__name__}", "error")


def _judge_via_subagent(ctx: Any, goal_text: str, prompt: str, allowed: tuple) -> Verdict:
    service = ctx.subagent_lifecycle
    handle = service.launch({
        "goal": f"{JUDGE_SYSTEM}\n\n{prompt}",
        "context": goal_text,
        "role": "leaf",
        "allowed_toolsets": allowed,
    })
    terminal = service.wait(handle, timeout_seconds=300)
    if not terminal.completed:
        return Verdict(False, "judge did not finish in time", "subagent")
    result = service.result(handle)
    summary = getattr(result, "summary", None)
    return parse_verdict(summary, "subagent")


def _judge_via_llm(ctx: Any, prompt: str) -> Verdict:
    result = ctx.llm.complete(
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0, task="goal_judge",
    )
    text = getattr(result, "text", result)
    return parse_verdict(text, "llm")
