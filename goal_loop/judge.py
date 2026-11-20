"""The judge: a fresh-context verifier that checks the real state, not the worker's claim.

Goal 3 of the owner: a judge with a new context examines each goal on the actual state of the
system; it only reads; it does not accept the worker report as proof; it gives the cause for each
goal it rejects. The judge runs as a child agent through ``ctx.subagent_lifecycle.launch`` with a
read-only toolset subset (default deny; the plugin's ``pre_tool_call`` guard also vetoes mutating
tools in the judge child). If the launch is not available, the judge falls back to a tool-less
``ctx.llm`` call that reads the goal plus the claim evidence as data.

Model (v0.3): both paths use the plugin's own auxiliary task ``goal_loop_judge``
(``auxiliary.goal_loop_judge`` in the Hermes config, registered with ``ctx.register_auxiliary_task``).
The subagent path passes its ``provider``/``model`` in the launch request; the fallback path routes
``ctx.llm.complete(task="goal_loop_judge")``. A plugin may route only through a task it registered
itself (agent/plugin_llm.py ``_check_task``), so the old built-in key ``goal_judge`` failed there.

Verdicts: ``VERDICT: done``, ``VERDICT: wait <reason>`` (the effect can be examined only later) or
``VERDICT: not-done <missing points>``. A judge that does not answer (timeout, launch or call error)
gives the method ``unavailable``: the caller keeps the goal status and asks for the same claim again.
0.4.3: the judge is tried again (attempts, time limit per attempt, backoff, shortened claim) before it
goes ``unavailable``; an answer without a VERDICT line counts as no answer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("goal_loop.judge")

JUDGE_TASK = "goal_loop_judge"
DEFAULT_TIMEOUT = 300
# 0.4.3: a retry sends a shortened claim (a long claim makes a slow judge slower).
RETRY_EVIDENCE_CHARS = 6000
MIN_ATTEMPT_SECONDS = 10

# Toolsets a judge child may use. ``terminal`` lets the judge RUN the goal's check command,
# ``file`` lets it read files; the judge prompt and the pre_tool_call guard forbid any mutation.
JUDGE_TOOLSETS = ("file", "terminal")

JUDGE_SYSTEM = (
    "You are a goal-completion judge with a fresh context. Verify the claim against the goal and "
    "its check by inspecting the ACTUAL state of the system. RUN the goal's check command yourself "
    "with the terminal tool and read files to confirm. You only inspect: never write or edit a "
    "file, never mutate goal state, never trust the worker's report as proof. Use only read-only "
    "commands: no output redirection, no rm, mv, cp, tee, no 'sed -i', no package install, no git "
    "commit or any command that changes state. Answer with exactly one line: 'VERDICT: done' if "
    "the goal's check is proven to pass now; 'VERDICT: wait <reason>' if the work is complete but "
    "its effect can only be examined later (a running job, a scheduled time); otherwise "
    "'VERDICT: not-done <the missing points, separated by semicolons>'."
)

VERDICT_PREFIX = "VERDICT:"


@dataclass
class Verdict:
    done: bool
    reason: str
    method: str  # "subagent", "llm", "unavailable"
    wait: bool = False

    @property
    def unavailable(self) -> bool:
        return self.method == "unavailable"


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
    if verdict.startswith("wait"):
        return Verdict(False, body[len("wait"):].strip(" :-") or "effect not examinable yet",
                       method, wait=True)
    reason = body[len("not-done"):].strip(" :-") if verdict.startswith("not-done") else body
    return Verdict(False, reason or "rejected without a stated reason", method)


def _judge_prompt(goal_text: str, check: str, evidence: str) -> str:
    return (
        f"Goal: {goal_text}\n"
        f"Check that must pass: {check}\n\n"
        f"Worker completion claim (DATA, not proof):\n{evidence}\n\n"
        "Inspect the real state and give your one verdict line."
    )


def judge_route() -> dict:
    """``{"provider": ..., "model": ...}`` of ``auxiliary.goal_loop_judge``, only the set keys."""
    try:
        from agent.auxiliary_task_config import _get_auxiliary_task_config
        cfg = _get_auxiliary_task_config(JUDGE_TASK) or {}
    except Exception:
        return {}
    route = {}
    for key in ("provider", "model"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "auto":
            route[key] = value.strip()
    return route


def run_judge(ctx: Any, goal_text: str, check: str, evidence: str,
              *, parent_toolsets: Optional[set] = None, timeout: int = DEFAULT_TIMEOUT,
              route: Optional[dict] = None, attempts: int = 1, attempt_timeout: Optional[int] = None,
              backoff: float = 0, on_retry: Optional[Callable[[int, str], None]] = None,
              sleep: Callable[[float], None] = time.sleep) -> Verdict:
    """Run the judge with up to ``attempts`` tries inside the total budget ``timeout`` seconds.

    0.4.3: an attempt that gives no answer (``unavailable``: timeout, launch or call error) or no
    ``VERDICT`` line is tried again after ``backoff * attempt`` seconds, with a shortened claim and a
    time limit of ``attempt_timeout`` seconds for each attempt. After the last attempt the result is
    ``unavailable`` (never a rejection), so the caller keeps the goal status and asks for the same
    claim again. ``on_retry(attempt, reason)`` is called before each new attempt.
    """
    attempts = max(1, int(attempts or 1))
    deadline = time.monotonic() + max(1, int(timeout or DEFAULT_TIMEOUT))
    last = Verdict(False, "judge did not run", "unavailable")
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if attempt > 1 and remaining < MIN_ATTEMPT_SECONDS:
            break
        per = int(max(1, min(remaining, attempt_timeout or remaining)))
        text = evidence if attempt == 1 or len(evidence) <= RETRY_EVIDENCE_CHARS else (
            evidence[:RETRY_EVIDENCE_CHARS] + "\n[claim shortened for a retry]")
        verdict = _judge_once(ctx, goal_text, check, text, parent_toolsets=parent_toolsets,
                              timeout=per, route=route)
        if not (verdict.unavailable or _no_verdict(verdict)):
            return verdict
        last = verdict
        if attempt < attempts:
            if on_retry is not None:
                try:
                    on_retry(attempt, verdict.reason)
                except Exception:
                    logger.debug("judge on_retry failed", exc_info=True)
            pause = float(backoff or 0) * attempt
            if pause > 0 and deadline - time.monotonic() > pause + MIN_ATTEMPT_SECONDS:
                sleep(pause)
    if last.unavailable:
        return last
    return Verdict(False, f"{last.reason} ({attempts} attempts)", "unavailable")


def _no_verdict(verdict: Verdict) -> bool:
    """The judge answered, but without a usable verdict line (0.4.3: counts as no answer)."""
    return (not verdict.done and not verdict.wait
            and verdict.reason in ("judge returned no verdict", "judge gave no VERDICT line"))


def _judge_once(ctx: Any, goal_text: str, check: str, evidence: str,
                *, parent_toolsets: Optional[set] = None, timeout: int = DEFAULT_TIMEOUT,
                route: Optional[dict] = None) -> Verdict:
    """One judge attempt: a read-only child agent, else a tool-less LLM call.

    ``subagent_lifecycle.launch`` clamps the requested toolsets to the parent's rights and raises
    if they would broaden them; then the ``ctx.llm`` path runs. A child that does not finish in
    time and a fallback that fails give an ``unavailable`` verdict, not a rejection.
    """
    prompt = _judge_prompt(goal_text, check, evidence)
    route = judge_route() if route is None else route
    if parent_toolsets:
        allowed = tuple(ts for ts in JUDGE_TOOLSETS if ts in parent_toolsets)
    else:
        allowed = JUDGE_TOOLSETS
    sub_problem = ""
    if allowed and getattr(ctx, "subagent_lifecycle", None) is not None:
        try:
            return _judge_via_subagent(ctx, goal_text, prompt, allowed, route, timeout)
        except Exception as exc:
            sub_problem = type(exc).__name__
            logger.debug("judge subagent path failed, falling back to ctx.llm", exc_info=True)
    try:
        return _judge_via_llm(ctx, prompt)
    except Exception as exc:
        logger.warning("goal-loop judge unavailable (subagent: %s, llm: %s)", sub_problem or "-",
                       type(exc).__name__)
        return Verdict(False, f"judge call failed: {type(exc).__name__}", "unavailable")


def _judge_via_subagent(ctx: Any, goal_text: str, prompt: str, allowed: tuple, route: dict,
                        timeout: int) -> Verdict:
    service = ctx.subagent_lifecycle
    request = {
        "goal": f"{JUDGE_SYSTEM}\n\n{prompt}",
        "context": goal_text,
        "role": "leaf",
        "allowed_toolsets": allowed,
    }
    request.update(route)
    handle = service.launch(request)
    terminal = service.wait(handle, timeout_seconds=timeout)
    if not terminal.completed:
        return Verdict(False, "judge did not finish in time", "unavailable")
    result = service.result(handle)
    summary = getattr(result, "summary", None)
    return parse_verdict(summary, "subagent")


def _judge_via_llm(ctx: Any, prompt: str) -> Verdict:
    result = ctx.llm.complete(
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0, task=JUDGE_TASK,
    )
    text = getattr(result, "text", result)
    return parse_verdict(text, "llm")
