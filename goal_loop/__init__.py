"""hermes-goal-loop — a goal loop for Hermes as a plugin.

The plugin delivers the owner's five goals:

1. Before the work: the model sets the goals (``goal_set``), each with one falsifiable check, and
   asks for the go. No loop runs before ``goal_go``. **Go rule:** ``goal_go`` is accepted only when
   an owner message came into the chat after the last ``goal_set`` of the session; otherwise the tool
   returns an error text and the loop does not start. The ``pre_llm_call`` hook counts the owner
   messages (it fires once per incoming turn, never for a ``pre_stop`` continuation; gateway
   notifications, subagent and cron turns do not count).
2. To the end: a generic ``pre_stop`` hook (added by the companion fork patch) keeps the turn going
   while open goals remain, with no file edit, bounded by ``agent.max_stop_continuations``.
3. The judge: ``goal_claim`` submits a structured claim; a fresh-context judge checks the real state
   read-only and gives a reason for each goal it rejects (see :mod:`goal_loop.judge`).
4. Status, shown quietly (``transform_llm_output``):

   - The full goal list is appended to the answer of a turn in which ``goal_set`` set or changed
     the list, also after the go.
   - After that, an answer shows only what needs the owner: a goal that became blocked
     (``goal_block``) and a new owner decision (``goal_decision``), each one time. Progress and
     judge rejections add nothing.
   - When the judge has confirmed all goals, the full list with checkmarks is appended one time,
     to the answer with the final report. Then the chat appends nothing until a new ``goal_set``.

   ``goal_status`` returns the full list as finished text, and its description tells the model to
   give that text to the owner unchanged. There is no ``/goals`` slash command: on the Open WebUI →
   pipe → ``POST /v1/runs`` path a slash text never reaches a plugin command handler (see README).
5. Decisions: ``goal_decision`` records an owner decision; it is shown as a numbered text list, no
   card and no modal.

Usage hint: the plugin adds a short system-prompt section (``register_system_prompt_section``):
for a multi-step task call ``goal_set`` first, then show the list and wait for the owner's go. The
section is empty (so not rendered) in judge sessions and the other ``EXCLUDED_PLATFORMS``.

The plugin never starts a turn in an API session and never edits core files (plugin rule). The
continuation is owned by the generic ``pre_stop`` fork patch; without that patch the hook simply
never fires and the tools still work.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from .claim import ClaimError, serialize_completion_claim
from .judge import run_judge
from .state import Goal, GoalState, GoalStore, owner_block, status_block

logger = logging.getLogger("goal_loop")

# Where per-chat state and ledgers live: $HERMES_HOME/goal-loop, so each Hermes home or profile has
# its own goal state (F8). GOAL_LOOP_STATE_DIR overrides it (tests, special setups).
_STATE_DIR_ENV = "GOAL_LOOP_STATE_DIR"
_STATE_SUBDIR = "goal-loop"

STOP_CAUSES = ("done", "owner-question", "blocker", "no-statement")

# Platforms the loop must never drive. ``cron`` is the owner rule (no loop in scheduled runs).
# ``subagent`` matters because a delegated child (the judge) INHERITS the parent's conversation_id
# (tools/delegate_tool.py: child._inherited_conversation_id = conversation_id(parent_agent)), so it
# would otherwise share the parent's chat:<conversation_id> state key and the pre_stop hook would
# fire inside the judge, swallowing its verdict (F7). The loop belongs only to the parent session.
EXCLUDED_PLATFORMS = ("cron", "subagent")

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

# The usage hint for the model (system-prompt section, not for EXCLUDED_PLATFORMS).
PROMPT_SECTION_ID = "goal-loop"
USAGE_HINT = (
    "Goal loop: for a multi-step task, first call goal_set with the goals, each with one "
    "falsifiable check. Then show the goal list to the owner and wait for the owner's go. Call "
    "goal_go only after the owner gave the go in a new message; goal_go refuses a go without an "
    "owner message after the last goal_set."
)

GO_REFUSED = (
    "error: goal_go refused: no owner message came after the last goal_set. Show the goal list "
    "to the owner, ask for the go and end your turn. Call goal_go only in the turn that answers "
    "the owner's go. The loop does not start."
)

GOAL_STATUS_DESCRIPTION = (
    "Return the goal list with each goal's check and status as finished text. Give the returned "
    "text to the owner unchanged, word for word: do not summarize, shorten, reorder or reformat it."
)

# Gateway wake/notification turns end with this footer (gateway/run_notifications.py
# INTERNAL_NOTIFICATION_FOOTER); such a turn is machinery, not an owner message.
_INTERNAL_FOOTER_PREFIX = "[INTERNAL NOTIFICATION"
# A typed /steer row has full user authority (agent/prompt_builder.py STEER_DISPLAY_KIND); every
# other display_kind on the turn's user row marks a synthesized turn.
_STEER_DISPLAY_KIND = "steer"
# gateway/response_filters.py INTERNAL_NOTIFICATION_DISPLAY_KIND: also the kind of an API client turn
# under the session authority; the api_inbound metadata flag tells the two apart.
_API_TURN_DISPLAY_KIND = "internal_notification"


def _state_dir() -> str:
    """The goal-state folder, resolved at call time (not at import) from the active Hermes home.

    Order: GOAL_LOOP_STATE_DIR, then ``hermes_constants.get_hermes_home()`` (context-local override,
    then HERMES_HOME, then the platform default), then a plain HERMES_HOME / ~/.hermes fallback when
    the Hermes module is not importable (plain unit tests). A fixed ~/.hermes path would make every
    home and profile share one goal state, so a test home could read or overwrite a real one (F8).
    """
    override = os.environ.get(_STATE_DIR_ENV)
    if override:
        return override
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home() / _STATE_SUBDIR)
    except Exception:
        base = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        return os.path.join(base, _STATE_SUBDIR)


def _chat_key(chat_id: str) -> str:
    """Key state by chat. ``chat_id`` is ``conversation_id`` when the caller has it, else
    ``session_id``. ``conversation_id`` is the compression-lineage root
    (agent/prompt_cache_scope.py): a compaction rotates ``session_id`` but keeps
    ``conversation_id``, so state keyed on it survives compaction (F3). ``platform`` is never part
    of the key: the dispatcher gives tools no platform, the hooks get ``agent.platform`` (F2)."""
    return f"chat:{chat_id or 'default'}"


def _gate_key(session_id: Any) -> str:
    """The go gate is keyed by the session id: ``pre_llm_call`` gets only ``session_id``."""
    return str(session_id or "default")


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
        # F3: the dispatcher passes both ids; conversation_id is the compression-stable root
        # (agent/prompt_cache_scope.py), so state keyed on it survives a compaction, which rotates
        # session_id. Fall back to session_id for callers and tests that only have that.
        return _chat_key(str(kwargs.get("conversation_id") or kwargs.get("session_id") or ""))

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
        # Display: this turn shows the full list (g1); a new list can be finished (and shown) again.
        state.list_changed = True
        state.final_shown = False
        # Go rule: no owner message has come after this goal_set yet.
        self.store.open_gate(_gate_key(kwargs.get("session_id")))
        self.store.append_ledger(state.chat_key, "goals_set", count=len(goals))
        self.store.save(state)
        lines = ["Goals recorded. Show this list to the owner and ask for the go:"]
        for goal in goals:
            lines.append(f"- {goal.id}: {goal.text} (check: {goal.check})")
        lines.append("No loop runs before goal_go. Call goal_go only after the owner's go message.")
        return "\n".join(lines)

    def goal_go(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
        if not state.goals:
            return "error: no goals set; call goal_set first"
        if state.go:
            return f"Go already recorded. The loop works the {len(state.open_goals())} open goal(s)."
        gate = _gate_key(kwargs.get("session_id"))
        count = self.store.gate_count(gate)
        if not count:
            self.store.append_ledger(state.chat_key, "go_refused", owner_messages=count)
            return GO_REFUSED
        state.go = True
        self.store.append_ledger(state.chat_key, "go", owner_messages=count)
        self.store.save(state)
        self.store.close_gate(gate)
        return f"Go recorded. The loop will work the {len(state.open_goals())} open goal(s) to the end."

    def goal_status(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self.store.load(self._ids(kwargs))
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
        goal.block_shown = False
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
                  final_response: str = "", conversation_id: str = "",
                  **_: Any) -> Optional[dict]:
        # Exclude excluded platforms BEFORE touching the store: a judge child inherits the parent's
        # conversation_id (tools/delegate_tool.py), so without this guard it would read the parent's
        # state and even write continuations from inside the subagent (F7). The loop is the parent's.
        if platform in EXCLUDED_PLATFORMS:
            return None
        try:
            state = store.load(_chat_key(conversation_id or session_id))
        except Exception:
            logger.debug("pre_stop state load failed", exc_info=True)
            return None
        if not state.go or not state.goals:
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


# --- The status display (transform_llm_output) -------------------------------------------------

def _display_block(state: GoalState) -> str:
    """Decide what this turn's answer shows and record it as shown (the caller saves the state).

    1. ``goal_set`` set or changed the list in this turn: the full list.
    2. All goals judge-confirmed and the final list not shown yet: the full list, one time.
    3. Otherwise only the owner items not shown yet: newly blocked goals and new decisions.
    """
    if state.list_changed:
        state.mark_all_shown()
        state.final_shown = state.all_done()
        return status_block(state)
    if state.all_done():
        if state.final_shown:
            return ""
        state.final_shown = True
        state.mark_all_shown()
        return status_block(state)
    blocked = [g for g in state.goals if g.status == "blocked" and not g.done and not g.block_shown]
    start = min(max(state.decisions_shown, 0), len(state.decisions))
    decisions = [(index, text) for index, text in enumerate(state.decisions, start=1)
                 if index > start]
    if not blocked and not decisions:
        return ""
    for goal in blocked:
        goal.block_shown = True
    state.decisions_shown = len(state.decisions)
    return owner_block(blocked, decisions)


def make_transform_llm_output(store: GoalStore):
    """Append the goal display to the final answer of a turn, only when there is news (goal 4)."""

    def _transform(response_text: str = "", text: str = "", session_id: str = "",
                   platform: str = "", conversation_id: str = "", **_: Any) -> Optional[str]:
        # A judge child inherits the parent's conversation_id (tools/delegate_tool.py), so without
        # this guard the parent's status block would be appended to the judge's "VERDICT:" line and
        # the verdict parse would fail (F7). The status block belongs only to the parent session.
        if platform in EXCLUDED_PLATFORMS:
            return None
        # The fork passes the model's answer as ``response_text`` (agent/turn_finalizer.py
        # apply_llm_output_transform); ``text`` is a fallback only, never the live field (F6). A
        # wrong field name here would make the status block OVERWRITE the model's real answer, since
        # the fork takes the first non-empty string the hook returns.
        body = response_text if isinstance(response_text, str) and response_text else text
        if not isinstance(body, str):
            return None
        try:
            state = store.load(_chat_key(conversation_id or session_id))
        except Exception:
            return None
        if not state.goals:
            return None
        block = _display_block(state)
        if not block:
            return None
        try:
            store.save(state)
        except Exception:
            logger.debug("goal display state save failed", exc_info=True)
        # F5: keep an empty model turn visible instead of hiding it behind the status block, so a
        # missing owner report is not silently masked.
        if not body.strip():
            return NO_REPORT_MARKER + "\n" + block
        return body + "\n" + block

    return _transform


# --- The go gate (pre_llm_call) ----------------------------------------------------------------

def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(str(part.get("text") or "") for part in message
                         if isinstance(part, dict) and part.get("type") in (None, "text", "input_text"))
    return ""


def _is_owner_message(user_message: Any, conversation_history: Any) -> bool:
    """True for a human turn: not empty, no gateway notification footer, and the turn's user row
    has no synthesized ``display_kind``. Two typed rows count as human: a ``/steer`` row, and an
    API client turn. With the session authority, every ``/v1/runs`` turn is stored as
    ``internal_notification`` with ``display_metadata.api_inbound`` True
    (gateway/session_ingress.py, gateway/response_filters.py persisted_user_metadata_for_event)."""
    text = _message_text(user_message).strip()
    if not text or _INTERNAL_FOOTER_PREFIX in text[-400:]:
        return False
    if isinstance(conversation_history, list):
        row = next((m for m in reversed(conversation_history)
                    if isinstance(m, dict) and m.get("role") == "user"), None)
        kind = row.get("display_kind") if isinstance(row, dict) else None
        if kind and kind != _STEER_DISPLAY_KIND:
            metadata = row.get("display_metadata")
            api_inbound = (kind == _API_TURN_DISPLAY_KIND and isinstance(metadata, dict)
                           and metadata.get("api_inbound") is True)
            if not api_inbound:
                return False
    return True


def make_pre_llm_call(store: GoalStore):
    """Count owner messages for the go gate. ``pre_llm_call`` fires once per incoming turn
    (agent/turn_context.py _collect_pre_llm_call_context), never for a pre_stop continuation.
    Returns None: the hook injects no context."""

    def _pre_llm_call(session_id: str = "", platform: str = "", user_message: Any = None,
                      conversation_history: Any = None, **_: Any) -> None:
        if platform in EXCLUDED_PLATFORMS:
            return None
        try:
            if _is_owner_message(user_message, conversation_history):
                store.bump_gate(_gate_key(session_id))
        except Exception:
            logger.debug("goal-loop go gate update failed", exc_info=True)
        return None

    return _pre_llm_call


# --- The usage hint (system-prompt section) ----------------------------------------------------

def prompt_section(session_info: Mapping[str, Any]) -> str:
    """The usage hint for a normal session; empty (so not rendered) for EXCLUDED_PLATFORMS."""
    platform = str((session_info or {}).get("platform") or "")
    if platform in EXCLUDED_PLATFORMS:
        return ""
    return USAGE_HINT


def register(ctx) -> None:
    """Register the goal tools, the three hooks and the usage hint (called once by the loader)."""
    # The state folder is fixed ONCE here, when the plugin loads: $HERMES_HOME/goal-loop of the home
    # that loads the plugin (or GOAL_LOOP_STATE_DIR). A later change of HERMES_HOME in the same
    # process does not move an already registered store (F8).
    store = GoalStore(_state_dir())
    tools = _Tools(store, ctx)
    specs = (
        ("goal_set", _GOAL_SET_SCHEMA, tools.goal_set, "🎯",
         "Record the goals, each with one falsifiable check. Call before any goal work; then ask the owner for the go."),
        ("goal_go", _EMPTY_SCHEMA, tools.goal_go, "🟢",
         ("Record the owner's go. Call it only after the owner gave the go in a new message after "
          "goal_set; otherwise it is refused. The loop works the open goals to the end after this.")),
        ("goal_status", _EMPTY_SCHEMA, tools.goal_status, "📋", GOAL_STATUS_DESCRIPTION),
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
    ctx.register_hook("pre_llm_call", make_pre_llm_call(store))
    # hermes_cli/plugins_context.py register_system_prompt_section: rendered once per new session
    # (agent/system_prompt.py _frozen_plugin_prompt_sections); an empty string is not rendered.
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section):
        try:
            register_section(PROMPT_SECTION_ID, prompt_section)
        except Exception:
            logger.warning("goal-loop usage hint not registered", exc_info=True)
    logger.info("goal-loop plugin registered: 6 tools, pre_stop + transform_llm_output + "
                "pre_llm_call hooks, usage hint")
