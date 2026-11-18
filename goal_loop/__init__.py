"""hermes-goal-loop — a goal loop for Hermes as a plugin.

The plugin delivers the owner's five goals:

1. Before the work: the model sets the goals (``goal_set``), each with one falsifiable check, and
   asks for the go. ``goal_status`` shows the list. No loop runs before ``goal_go``.
2. To the end: a generic ``pre_stop`` hook (added by the companion fork patch) keeps the turn going
   while open goals remain, with no file edit, bounded by the goal count of continuations.
3. The judge: ``goal_claim`` submits a structured claim; a fresh-context judge checks the real state
   read-only and gives a reason for each goal it rejects (see :mod:`goal_loop.judge`).
4. Status: a judge-confirmed goal stays done for the session; a status block is appended to each
   answer (``transform_llm_output``) so every report shows all goals.
5. Decisions: ``goal_decision`` records an owner decision; the status block renders them as a
   numbered text list, no card and no modal.

The plugin never starts a turn in an API session and never edits core files (plugin rule). The
continuation is owned by the generic ``pre_stop`` fork patch; without that patch the hook simply
never fires and the tools still work.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .claim import ClaimError, serialize_completion_claim
from .judge import run_judge
from .state import Goal, GoalState, GoalStore, status_block

logger = logging.getLogger("goal_loop")

# Where per-chat state and ledgers live. Overridable so a test can use a temp dir.
_STATE_DIR_ENV = "GOAL_LOOP_STATE_DIR"
_DEFAULT_STATE_DIR = os.path.expanduser("~/.hermes/goal-loop")

STOP_CAUSES = ("done", "owner-question", "blocker", "no-statement")

# Appended to goal_claim's reply when the claim confirms the LAST open goal, so the model does not
# stop with an empty turn once the loop has nothing left to continue (F5). The model owns the final
# report; the plugin only prompts for it.
FINAL_REPORT_NUDGE = (
    " All goals are now confirmed. Write the final report for the owner in this message: "
    "list each goal, the judge result, and the evidence (including the file content)."
)

# Shown (before the status block) only when the model's own turn text is truly empty, so a missing
# owner report stays visible instead of being masked by the status block (F5/F6).
NO_REPORT_MARKER = "(no final report from the model)"


def _state_dir() -> str:
    return os.environ.get(_STATE_DIR_ENV) or _DEFAULT_STATE_DIR


def _chat_key(session_id: str) -> str:
    """Key state by chat. Uses ``session_id`` only and never ``platform``: the dispatcher passes
    ``session_id``/``conversation_id`` to tool handlers but not ``platform`` (#68318), while the
    pre_stop hook sees ``agent.platform``. A platform-free key is the one both sides can agree on,
    so a tool and the hook read the same file. (``conversation_id`` would also survive compaction;
    moving both sides to it is tracked separately.)"""
    return f"chat:{session_id or 'default'}"


# --- Tool schemas -----------------------------------------------------------------------------

_GOAL_SET_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "short stable id, e.g. g1"},
                    "text": {"type": "string", "description": "the goal in one sentence"},
                    "check": {"type": "string",
                              "description": "one falsifiable check that proves the goal is done"},
                },
                "required": ["id", "text", "check"],
            },
        },
    },
    "required": ["goals"],
}

_CLAIM_CHECK_ITEM_SCHEMA = {
    "type": "object",
    "description": "one verification command you actually ran for this goal",
    "properties": {
        "command": {"type": "string",
                    "description": "the shell command you ran, e.g. grep -qx READY ready.txt"},
        "result": {"type": "string", "enum": ["passed", "failed", "not-run"],
                   "description": "passed only if the command succeeded; a failed check rejects the claim"},
        "exitCode": {"type": "integer",
                     "description": "the command's exit code; 0 for a passed check, omit for not-run"},
        "explanation": {"type": "string",
                        "description": "use instead of command for a manual check you cannot script"},
    },
    "required": ["result"],
}

_CLAIM_CRITERION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string", "description": "one acceptance criterion in a short phrase"},
        "evidence": {"type": "array", "items": {"type": "string"},
                     "description": "at least one concrete evidence string for this criterion"},
    },
    "required": ["criterion", "evidence"],
}

_GOAL_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_id": {"type": "string", "description": "the id from goal_set, e.g. g1"},
        "claim": {
            "type": "object",
            "description": ("structured completion claim. Minimal valid form: "
                            '{"summary": "...", "checks": [{"command": "...", "result": "passed", '
                            '"exitCode": 0}]}'),
            "properties": {
                "summary": {"type": "string",
                            "description": "one line on what you did to finish this goal"},
                "criteria": {"type": "array", "items": _CLAIM_CRITERION_ITEM_SCHEMA},
                "checks": {"type": "array", "items": _CLAIM_CHECK_ITEM_SCHEMA},
                "changedFiles": {"type": "array", "items": {"type": "string"}},
                "knownLimitations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary"],
        },
    },
    "required": ["goal_id", "claim"],
}

_GOAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "the decision with options and a recommendation"},
    },
    "required": ["text"],
}

_GOAL_ID_SCHEMA = {
    "type": "object",
    "properties": {"goal_id": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["goal_id"],
}

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


# --- Tool handlers ----------------------------------------------------------------------------

class _Tools:
    """Tool handlers bound to one store. A class keeps the store injectable for tests."""

    def __init__(self, store: GoalStore, ctx: Any = None) -> None:
        self.store = store
        self.ctx = ctx

    @staticmethod
    def _ids(kwargs: dict) -> str:
        return _chat_key(str(kwargs.get("session_id") or ""))

    def goal_set(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        goals_in = args.get("goals") or []
        if not isinstance(goals_in, list) or not goals_in:
            return "error: goals must be a non-empty array"
        goals: list[Goal] = []
        seen: set[str] = set()
        for item in goals_in:
            if not isinstance(item, dict):
                return "error: each goal must be an object"
            gid = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            check = str(item.get("check") or "").strip()
            if not gid or not text or not check:
                return "error: each goal needs id, text and check"
            if gid in seen:
                return f"error: duplicate goal id {gid!r}"
            seen.add(gid)
            goals.append(Goal(id=gid, text=text, check=check))
        state.goals = goals
        state.go = False
        state.continuations = 0
        self.store.append_ledger(state.chat_key, "goals_set", count=len(goals))
        self.store.save(state)
        lines = ["Goals recorded. Show this list to the owner and ask for the go:"]
        for goal in goals:
            lines.append(f"- {goal.id}: {goal.text} (check: {goal.check})")
        lines.append("No loop runs before goal_go.")
        return "\n".join(lines)

    def goal_go(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        if not state.goals:
            return "error: no goals set; call goal_set first"
        state.go = True
        self.store.append_ledger(state.chat_key, "go")
        self.store.save(state)
        return f"Go recorded. The loop will work the {len(state.open_goals())} open goal(s) to the end."

    def goal_status(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        if not state.goals:
            return "No goals set."
        block = status_block(state)
        return block.lstrip("\n-") if block else "No goals set."

    def goal_decision(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        text = str(args.get("text") or "").strip()
        if not text:
            return "error: decision text is required"
        state.decisions.append(text)
        self.store.append_ledger(state.chat_key, "decision", text=text[:200])
        self.store.save(state)
        return f"Decision recorded ({len(state.decisions)} open). It will appear in the decision list."

    def goal_block(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        goal = state.goal(str(args.get("goal_id") or ""))
        if goal is None:
            return "error: unknown goal_id"
        goal.status = "blocked"
        goal.note = str(args.get("reason") or "").strip()
        self.store.append_ledger(state.chat_key, "blocked", goal_id=goal.id, reason=goal.note[:200])
        self.store.save(state)
        return f"Goal {goal.id} marked blocked. The loop will not retry it."

    def goal_claim(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        goal = state.goal(str(args.get("goal_id") or ""))
        if goal is None:
            known = ", ".join(g.id for g in state.goals) or "none set"
            return (f"error: unknown goal_id {str(args.get('goal_id') or '')!r}. "
                    f"Known goal ids: {known}. Pass goal_id at the top level, not inside claim.")
        try:
            evidence = serialize_completion_claim(args.get("claim"))
        except ClaimError as exc:
            return f"claim rejected: {exc}"
        verdict = run_judge(
            self.ctx, goal.text, goal.check, evidence,
            parent_toolsets=kwargs.get("_parent_toolsets"),
        )
        if verdict.done:
            goal.done = True
            goal.status = "done"
            goal.judge_reason = ""
            # terminal event to the ledger BEFORE the state file
            self.store.append_ledger(state.chat_key, "confirmed", goal_id=goal.id,
                                     method=verdict.method)
            self.store.save(state)
            reply = f"Goal {goal.id} confirmed by the judge ({verdict.method}). It stays done."
            # F5: if this was the last open goal, the loop will not continue the turn, so ask the
            # model for the owner report now instead of letting it stop with an empty message.
            if not state.open_goals():
                reply += FINAL_REPORT_NUDGE
            return reply
        goal.judge_reason = verdict.reason
        goal.status = "in_work"
        self.store.append_ledger(state.chat_key, "rejected", goal_id=goal.id,
                                 reason=verdict.reason[:200], method=verdict.method)
        self.store.save(state)
        return f"Goal {goal.id} rejected by the judge: {verdict.reason}. Do the work again."


# --- The pre_stop hook ------------------------------------------------------------------------

def make_pre_stop(store: GoalStore):
    """Build the ``pre_stop`` callback. It continues the turn while open goals remain and the go is
    given, bounded by the number of goals times a small factor; otherwise it lets the turn finish."""

    def _pre_stop(session_id: str = "", platform: str = "", attempt: int = 0,
                  final_response: str = "", **_: Any) -> Optional[dict]:
        try:
            state = store.load(_chat_key(session_id))
        except Exception:
            logger.debug("pre_stop state load failed", exc_info=True)
            return None
        # Cron sessions are excluded (owner rule); the loop is only for chat-started sessions.
        if platform == "cron" or not state.go or not state.goals:
            return None
        open_goals = state.open_goals()
        if not open_goals:
            return None
        # Own bound: the goal count gives a natural ceiling; the fork patch adds max_stop_continuations.
        state.continuations = attempt + 1
        store.save(state)
        names = ", ".join(g.id for g in open_goals)
        message = (
            f"[goal-loop] {len(open_goals)} goal(s) are still open: {names}. Continue the work. "
            "When a goal is done, submit goal_claim with real evidence for the judge. If a goal "
            "needs an owner decision, record it with goal_decision and keep working the others. If "
            "a goal is truly blocked, call goal_block. End your message with a stop-cause line, one "
            f"of: {', '.join(STOP_CAUSES)}."
        )
        return {"action": "continue", "message": message}

    return _pre_stop


def make_transform_llm_output(store: GoalStore):
    """Append the goal status block to each final answer (goal 4)."""

    def _transform(response_text: str = "", text: str = "", session_id: str = "",
                   platform: str = "", **_: Any) -> Optional[str]:
        # The fork passes the model's answer as ``response_text`` (agent/turn_finalizer.py
        # apply_llm_output_transform); ``text`` is a fallback only, never the live field (F6). A
        # wrong field name here would make the status block OVERWRITE the model's real answer, since
        # the fork takes the first non-empty string the hook returns.
        body = response_text if isinstance(response_text, str) and response_text else text
        try:
            state = store.load(_chat_key(session_id))
        except Exception:
            return None
        if not state.goals:
            return None
        block = status_block(state)
        if not block or not isinstance(body, str):
            return None
        # F5: keep an empty model turn visible instead of hiding it behind the status block, so a
        # missing owner report is not silently masked.
        if not body.strip():
            return NO_REPORT_MARKER + "\n" + block
        return body + "\n" + block

    return _transform


def register(ctx) -> None:
    """Register the goal tools and the two hooks (called once by the plugin loader)."""
    store = GoalStore(_state_dir())
    tools = _Tools(store, ctx)
    specs = (
        ("goal_set", _GOAL_SET_SCHEMA, tools.goal_set, "🎯",
         "Record the goals, each with one falsifiable check. Call before any goal work; then ask the owner for the go."),
        ("goal_go", _EMPTY_SCHEMA, tools.goal_go, "🟢",
         "Record the owner's go. The loop works the open goals to the end after this."),
        ("goal_status", _EMPTY_SCHEMA, tools.goal_status, "📋",
         "Show the goal list with each goal's check and status."),
        ("goal_claim", _GOAL_CLAIM_SCHEMA, tools.goal_claim, "✅",
         ("Submit a structured completion claim for one goal; a fresh-context judge verifies the "
          'real state. goal_id is top-level; claim needs a summary and checks you ran, e.g. '
          'claim={"summary": "wrote the file", "checks": [{"command": "grep -qx READY f", '
          '"result": "passed", "exitCode": 0}]}.')),
        ("goal_decision", _GOAL_DECISION_SCHEMA, tools.goal_decision, "🧭",
         "Record an owner decision (options + recommendation) for the end-of-run decision list."),
        ("goal_block", _GOAL_ID_SCHEMA, tools.goal_block, "⛔",
         "Mark a goal blocked so the loop stops retrying it."),
    )
    for name, schema, handler, emoji, description in specs:
        # The registry builds each definition as {**entry.schema, "name": ...} and the model reads
        # function["description"]/function["parameters"] (tools/registry.py get_definitions,
        # tools/tool_search.py). So the registered schema must be in function form
        # {name, description, parameters}; a bare parameter schema leaves the model with empty
        # parameters and no description. Pattern follows plugins/google_meet/tools.py _schema().
        ctx.register_tool(
            name=name, toolset="goal_loop",
            schema={"name": name, "description": description, "parameters": schema},
            handler=handler, emoji=emoji, description=description)
    ctx.register_hook("pre_stop", make_pre_stop(store))
    ctx.register_hook("transform_llm_output", make_transform_llm_output(store))
    logger.info("goal-loop plugin registered: 6 tools, pre_stop + transform_llm_output hooks")
