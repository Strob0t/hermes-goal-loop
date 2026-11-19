"""hermes-goal-loop — a goal loop for Hermes as a plugin (v0.3).

1. Before the work: the model sets the goals (``goal_set``), each with one falsifiable check. The
   plugin appends the goal list and the go question to the answer of that turn (one time; the model
   does not repeat the list). No loop runs before ``goal_go``. **Go rule:** ``goal_go`` is accepted
   only when an owner message came into the chat after the last ``goal_set`` of the session. The
   ``pre_llm_call`` hook counts the owner messages (gateway notifications, subagent and cron turns
   do not count).
2. To the end: a generic ``pre_stop`` hook (companion fork patch) keeps the turn going while open
   goals remain. Limits (all in ``plugins.entries.goal-loop.settings``):

   - ``budget_per_goal`` (default 50): continuations per goal over all turns and runs. At the limit
     the goal is paused and an owner decision is recorded; the loop does not fail the goal.
   - ``idle_replan_after`` / ``idle_pause_after`` (2 / 3): continuations in a row without a tool
     call (counted with ``post_tool_call``): first a request to plan again, then a pause.
   - ``reject_replan_after`` / ``reject_pause_after`` (2 / 3): similar judge rejections in a row
     for one goal: first a request to plan again, then a pause with an owner decision.
   - ``block_min_rounds`` (3): ``goal_block`` needs a reason and is accepted only when the same
     obstacle stays for this number of continuation rounds (Codex rule).
   - ``stop_words``: an owner message that is one of these words pauses the loop.
   - After a Hermes restart the loop does not continue until an owner message comes.

3. The judge: ``goal_claim`` submits a structured claim; a fresh-context judge checks the real state
   read-only (see :mod:`goal_loop.judge`). Verdicts: done, wait (the goal waits without nudges),
   not-done (with the missing points). A judge that does not answer keeps the goal status.
4. Context: each owner turn with an active go gets the open goals (text, check, progress) as context
   (``pre_llm_call``), so a compression cannot remove them. ``goal_progress`` records the last step,
   its evidence and the next step of a goal.
5. Status, shown quietly (``transform_llm_output``): the full list on ``goal_set`` (with the go
   question as last line before the go), after that only goals that need the owner and new
   decisions, each one time, and the final list one time. Labels are in the owner's language
   (setting ``language``: ``auto``, ``de`` or ``en``). An answer that ends while goals are still open
   gets one line: the loop is paused, write to continue.
6. Changes after the go: ``goal_set`` is refused until an owner message approves the change;
   then the judge-confirmed goals stay done.

The plugin never starts a turn in an API session and never edits core files (plugin rule).
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Mapping, Optional

from .claim import ClaimError, serialize_completion_claim
from .judge import JUDGE_TASK, run_judge
from .state import Goal, GoalState, GoalStore, labels, owner_block, status_block

logger = logging.getLogger("goal_loop")

_STATE_DIR_ENV = "GOAL_LOOP_STATE_DIR"
_STATE_SUBDIR = "goal-loop"

# One id per process: a state that a different process drove is held after a restart (L13).
BOOT_ID = uuid.uuid4().hex

STOP_CAUSES = ("done", "owner-question", "blocker", "no-statement")

# ``cron`` is the owner rule (no loop in scheduled runs). ``subagent``: a delegated child (the judge)
# inherits the parent's conversation_id, so the loop must not drive it (F7).
EXCLUDED_PLATFORMS = ("cron", "subagent")

DEFAULTS = {
    "budget_per_goal": 50,
    "idle_replan_after": 2,
    "idle_pause_after": 3,
    "reject_replan_after": 2,
    "reject_pause_after": 3,
    "block_min_rounds": 3,
    "judge_timeout": 300,
    "language": "auto",
    "paused_hint": True,
    "stop_words": ["stopp", "stop", "halt", "pause", "pausiere", "abbrechen"],
    # L15 follow-up runs (owner decision 5): a run that ends after the go with open goals
    # schedules one more run of the same session; resume_max follow-up runs per go (0 = off).
    "resume": True,
    "resume_delay": "1m",
    "resume_max": 20,
}

FINAL_REPORT_NUDGE = (
    " All goals are now confirmed. Write the final report for the owner in this message, in the "
    "owner's language: list each goal, the judge result, and the evidence."
)

USAGE_HINT = (
    "Goal loop: for a multi-step task, first call goal_set with the goals, each with one "
    "falsifiable check. Write goal text and check in the language that the owner mostly uses in "
    "the chat; write internal texts (todo, notes, briefs) as usual. Rules for checks: (1) a check "
    "examines the content (evidence with a source, a real run, a value), not only the form (a "
    "file exists, a section count); (2) the judge must be able to examine the check when you "
    "submit the claim, so no check on a later chat report; (3) name the version that runs live as "
    "the base. Split a large goal into smaller goals. The plugin appends the goal list and the go "
    "question to your answer: do not repeat the list and do not ask for the go yourself. Call "
    "goal_go only after the owner gave the go in a new message. Record progress with "
    "goal_progress."
)

GO_REFUSED = (
    "error: goal_go refused: no owner message came after the last goal_set or pause. End your "
    "turn; the plugin shows the goal list and the go question. Call goal_go only in the turn that "
    "answers the owner's go. The loop does not start."
)

CHANGE_REFUSED = (
    "error: goal_set refused: the owner gave the go for the current goal list. Ask the owner to "
    "approve the change (name the goals that change), end your turn, and call goal_set again in "
    "the turn that answers the owner's approval. Judge-confirmed goals stay done."
)

GOAL_STATUS_DESCRIPTION = (
    "Return the goal list with each goal's check and status as finished text. Give the returned "
    "text to the owner unchanged, word for word: do not summarize, shorten, reorder or reformat it."
)

PROMPT_SECTION_ID = "goal-loop"
# Shown before the block when the model turn is empty (labels per language in state.LABELS).
NO_REPORT_MARKER = labels("en")["no_report"]

_INTERNAL_FOOTER_PREFIX = "[INTERNAL NOTIFICATION"
_STEER_DISPLAY_KIND = "steer"
_API_TURN_DISPLAY_KIND = "internal_notification"

# Tools that do not show progress on a goal (idle detection, L2).
_NO_PROGRESS_TOOLS = frozenset({"goal_status", "goal_decision", "goal_go", "goal_set"})

# L12: tools a judge child may call, and terminal commands it must not run.
_JUDGE_ALLOWED_TOOLS = frozenset({"read_file", "search_files", "terminal"})
_MUTATING_COMMAND = re.compile(
    r"(^|[;&|(`]\s*|\bsudo\s+|\bxargs\s+)(rm|mv|cp|tee|chmod|chown|ln|mkdir|rmdir|touch|truncate|dd|install|kill|"
    r"pkill|systemctl|docker|pip|apt|apt-get|npm|curl\s+[^|]*-X\s*(POST|PUT|DELETE|PATCH))\b"
    r"|sed\s+-i|\bgit\s+(commit|push|reset|checkout|merge|rebase|tag|branch\s+-[dD]|stash|clean)"
    r"|(^|[^0-9&>])>{1,2}(?!&)|\bpython[0-9.]*\s+-c\b"
)
# conversation_id -> parent session_id while a judge runs (L12)
_JUDGE_PARENTS: dict[str, str] = {}

_DE_WORDS = frozenset("und der die das ist nicht ich du mit für auf bitte sind wie was zu ein eine "
                      "dann noch auch schon oder wir ihr mir dir sollen soll kann muss wenn aber "
                      "jetzt hier gibt".split())
_EN_WORDS = frozenset("the and is not you with for please are what to a then also or this that "
                      "we they me can must should if but now here there".split())


def _state_dir() -> str:
    """The goal-state folder: GOAL_LOOP_STATE_DIR, else $HERMES_HOME/goal-loop (F8)."""
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
    """Key state by chat: ``conversation_id`` (compression-stable) when the caller has it, else
    ``session_id`` (F3)."""
    return f"chat:{chat_id or 'default'}"


def _gate_key(session_id: Any) -> str:
    return str(session_id or "default")


def _gate_for(store: "GoalStore", session_id: Any, conversation_id: Any = "") -> str:
    """The go gate is keyed on the chat (goal-loop-quiet decision 2a): the compression-stable
    ``conversation_id`` when the caller has it, else the chat that this session last used (tools
    and pre_stop record it), else the session id. ``pre_llm_call`` gets no conversation_id in fork
    5261bfd, so the session mapping is the fallback until the fork passes it."""
    if conversation_id:
        return _chat_key(str(conversation_id))
    sid = str(session_id or "")
    return store.chat_for_session(sid) or _chat_key(sid)


def detect_language(text: str) -> str:
    """'de', 'en' or '' (no clear vote) for one owner message."""
    words = re.findall(r"[a-zäöüß]+", (text or "").lower())
    de = sum(1 for w in words if w in _DE_WORDS) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    en = sum(1 for w in words if w in _EN_WORDS)
    if max(de, en) < 2 or de == en:
        return ""
    return "de" if de > en else "en"


def _similar(a: str, b: str) -> bool:
    """Two rejection reasons name the same problem (word overlap of at least one half)."""
    wa = set(re.findall(r"[a-z0-9äöüß_./-]{3,}", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9äöüß_./-]{3,}", (b or "").lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.5


DECISION_TEXT = {
    "de": {
        "budget": ("Ziel {gid} hat das Budget von {n} Fortsetzungen erreicht und ist pausiert. "
                   "Optionen: A) weitermachen mit {n} neuen Fortsetzungen (antworte mit 'go'); "
                   "B) das Ziel ändern; C) das Ziel streichen. Empfehlung: B, wenn die letzten "
                   "Ablehnungen denselben Grund nennen, sonst A."),
        "idle": ("Der Loop hat {n} Fortsetzungen in Folge ohne Werkzeugaufruf gemacht und ist "
                 "pausiert (Ziele: {gids}). Optionen: A) weitermachen (antworte mit 'go'); B) die "
                 "Ziele ändern; C) die Ziele streichen. Empfehlung: B, weil Leerlauf meist ein "
                 "unklares oder blockiertes Ziel zeigt."),
        "reject": ("Ziel {gid} wurde {n}-mal mit demselben Grund abgelehnt und ist pausiert: {why}. "
                   "Optionen: A) weitermachen (antworte mit 'go'); B) Ziel oder Prüfung ändern; "
                   "C) das Ziel streichen. Empfehlung: B, weil derselbe Weg {n}-mal gescheitert ist."),
        "paused_line": "Der Loop ist pausiert: {n} Ziel(e) sind offen ({gids}). Schreibe 'weiter', "
                       "damit er fortfährt.",
        "owner_stop": "Der Loop ist auf deinen Wunsch pausiert. Mit 'go' geht es weiter.",
        "resume_line": ("Der Loop läuft in {delay} in einem Folge-Run weiter: {n} Ziel(e) offen "
                        "({gids}). Die Antwort erscheint in diesem Chat. Mit 'stopp' hältst du ihn an."),
        "resume_max": ("Der Loop hat {n} Folge-Runs ohne Abschluss gemacht und ist pausiert "
                       "(Ziele: {gids}). Optionen: A) weitermachen (antworte mit 'go'); B) die Ziele "
                       "ändern; C) die Ziele streichen. Empfehlung: A, wenn die Ziele Fortschritt zeigen."),
    },
    "en": {
        "budget": ("Goal {gid} reached the budget of {n} continuations and is paused. Options: "
                   "A) continue with {n} new continuations (answer 'go'); B) change the goal; "
                   "C) remove the goal. Recommendation: B if the last rejections give the same "
                   "reason, else A."),
        "idle": ("The loop made {n} continuations in a row without a tool call and is paused "
                 "(goals: {gids}). Options: A) continue (answer 'go'); B) change the goals; "
                 "C) remove the goals. Recommendation: B, because idle turns usually show an "
                 "unclear or blocked goal."),
        "reject": ("Goal {gid} was rejected {n} times for the same reason and is paused: {why}. "
                   "Options: A) continue (answer 'go'); B) change the goal or its check; C) remove "
                   "the goal. Recommendation: B, because the same way failed {n} times."),
        "paused_line": "The loop is paused: {n} goal(s) are open ({gids}). Write 'continue' to go on.",
        "owner_stop": "The loop is paused as you asked. Answer 'go' to continue.",
        "resume_line": ("The loop continues in {delay} in a follow-up run: {n} goal(s) open "
                        "({gids}). The answer appears in this chat. Write 'stop' to halt it."),
        "resume_max": ("The loop made {n} follow-up runs without completion and is paused "
                       "(goals: {gids}). Options: A) continue (answer 'go'); B) change the goals; "
                       "C) remove the goals. Recommendation: A if the goals show progress."),
    },
}


def _dtext(language: str) -> dict:
    return DECISION_TEXT.get(language or "en", DECISION_TEXT["en"])


class Settings:
    """Plugin settings from ``plugins.entries.goal-loop.settings`` (``ctx.get_config``)."""

    def __init__(self, ctx: Any = None, overrides: Optional[dict] = None) -> None:
        self.ctx = ctx
        self.overrides = dict(overrides or {})

    def get(self, key: str) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        default = DEFAULTS[key]
        getter = getattr(self.ctx, "get_config", None)
        if callable(getter):
            try:
                value = getter(key, default)
            except Exception:
                value = default
        else:
            value = default
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = default
            return max(0, value)
        return value if value is not None else default


def _language(state: GoalState, store: GoalStore, settings: Settings, session_id: str = "") -> str:
    fixed = str(settings.get("language") or "auto").lower()
    if fixed in ("de", "en"):
        return fixed
    return state.language or store.session_language(session_id) or "en"


def _pause(store: GoalStore, state: GoalState, goals: list[Goal], reason: str, decision: str,
           gate: str, event: str) -> None:
    """Pause goals, record one owner decision, open the go gate of the chat (L1, L2, L17)."""
    for goal in goals:
        goal.status = "paused"
        goal.note = reason
        goal.block_shown = False
    state.decisions.append(decision)
    store.append_ledger(state.chat_key, event, goal_ids=[g.id for g in goals], reason=reason[:200])
    store.open_gate(gate)


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
                    "text": {"type": "string",
                             "description": "the goal in one sentence, in the owner's language"},
                    "check": {"type": "string",
                              "description": ("one falsifiable check of the content (evidence, "
                                              "source, real run) that the judge can examine at "
                                              "claim time, in the owner's language")},
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
        "text": {"type": "string",
                 "description": "the decision with options and a recommendation, in the owner's language"},
    },
    "required": ["text"],
}

_GOAL_BLOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_id": {"type": "string"},
        "reason": {"type": "string", "description": "the obstacle and what you tried (required)"},
    },
    "required": ["goal_id", "reason"],
}

_GOAL_PROGRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_id": {"type": "string"},
        "step": {"type": "string", "description": "the last completed step"},
        "evidence": {"type": "string", "description": "evidence for the step (command, file, value)"},
        "next": {"type": "string", "description": "the next step"},
    },
    "required": ["goal_id", "step", "next"],
}

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


# --- Tool handlers ----------------------------------------------------------------------------

class _Tools:
    """Tool handlers bound to one store. A class keeps the store injectable for tests."""

    def __init__(self, store: GoalStore, ctx: Any = None, settings: Optional[Settings] = None) -> None:
        self.store = store
        self.ctx = ctx
        self.settings = settings or Settings(ctx)

    def _load(self, kwargs: dict) -> GoalState:
        key = _chat_key(str(kwargs.get("conversation_id") or kwargs.get("session_id") or ""))
        self.store.remember_session(str(kwargs.get("session_id") or ""), key)
        return self.store.load(key)

    def goal_set(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        session = state.chat_key
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
        # a list with unfinished goals after the go is a change; a finished list starts fresh
        change = state.go and bool(state.unfinished())
        if change:
            # L3: after the go, a changed list needs the owner's approval in a new message.
            count = self.store.gate_count(session, "change")
            if not count:
                if count is None:
                    self.store.open_gate(session, "change")
                self.store.append_ledger(state.chat_key, "change_refused")
                return CHANGE_REFUSED
            self.store.close_gate(session, "change")
            old = {g.id: g for g in state.goals}
            for index, goal in enumerate(goals):
                prev = old.get(goal.id)
                if prev is not None and prev.done:
                    goals[index] = prev  # confirmed stays done, text and check unchanged
                elif prev is not None:
                    goal.progress = prev.progress
            kept = {g.id for g in goals}
            goals.extend(g for g in state.goals if g.done and g.id not in kept)
        state.goals = goals
        state.continuations = 0
        state.list_changed = True
        state.final_shown = False
        state.language = self.store.session_language(session) or state.language
        if not change:
            state.go = False
            state.idle_nudges = 0
            self.store.open_gate(session)
        self.store.append_ledger(state.chat_key, "goals_changed" if change else "goals_set",
                                 count=len(goals))
        self.store.save(state)
        if change:
            return ("Goal list changed with the owner's approval; confirmed goals stay done. The "
                    "plugin appends the new list to your answer. Continue the work.")
        return ("Goals recorded. The plugin appends the goal list and the go question to your "
                "answer of this turn. Do not repeat the list and do not ask for the go: write at "
                "most a short summary in the owner's language, then end your turn. No loop runs "
                "before goal_go.")

    def goal_go(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        if not state.goals:
            return "error: no goals set; call goal_set first"
        gate = state.chat_key
        paused = [g for g in state.goals if g.status == "paused" and not g.done]
        if state.go and not paused:
            return f"Go already recorded. The loop works the {len(state.open_goals())} open goal(s)."
        count = self.store.gate_count(gate)
        if not count:
            self.store.append_ledger(state.chat_key, "go_refused", owner_messages=count)
            return GO_REFUSED
        state.resumes = 0  # a new go allows resume_max new follow-up runs
        for goal in paused:
            goal.status = "in_work"
            goal.note = ""
            goal.nudges = 0
            goal.rejects = 0
            goal.last_reject = ""
        state.go = True
        state.idle_nudges = 0
        state.tool_calls_at_nudge = -1
        state.boot_id = BOOT_ID
        self.store.append_ledger(state.chat_key, "go", owner_messages=count,
                                 resumed=[g.id for g in paused])
        self.store.save(state)
        self.store.close_gate(gate)
        return f"Go recorded. The loop will work the {len(state.open_goals())} open goal(s) to the end."

    def goal_status(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        block = status_block(state, _language(state, self.store, self.settings, state.chat_key))
        return block.lstrip("\n-") if block else "No goals set."

    def goal_decision(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        text = str(args.get("text") or "").strip()
        if not text:
            return "error: decision text is required"
        state.decisions.append(text)
        self.store.append_ledger(state.chat_key, "decision", text=text[:200])
        self.store.save(state)
        return f"Decision recorded ({len(state.decisions)} open). It will appear in the decision list."

    def goal_block(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        goal = state.goal(str(args.get("goal_id") or ""))
        if goal is None:
            return "error: unknown goal_id"
        reason = str(args.get("reason") or "").strip()
        if not reason:
            return "error: goal_block needs a reason: the obstacle and what you tried (L11)"
        need = self.settings.get("block_min_rounds")
        if goal.block_seq != state.nudge_seq:
            same = bool(goal.block_reason) and _similar(goal.block_reason, reason)
            goal.block_tries = goal.block_tries + 1 if same else 1
            goal.block_seq = state.nudge_seq
            goal.block_reason = reason
        if goal.block_tries < need:
            self.store.append_ledger(state.chat_key, "block_noted", goal_id=goal.id,
                                     tries=goal.block_tries, reason=reason[:200])
            self.store.save(state)
            return (f"Block noted ({goal.block_tries}/{need}). A goal is blocked only when the same "
                    "obstacle stays for several rounds, not because the work is hard. Try a different "
                    "way now; call goal_block again in a later round if the same obstacle stays. If "
                    "the owner must decide or give something, use goal_decision.")
        goal.status = "blocked"
        goal.note = reason
        goal.block_shown = False
        self.store.append_ledger(state.chat_key, "blocked", goal_id=goal.id, reason=reason[:200])
        self.store.save(state)
        return f"Goal {goal.id} marked blocked. The loop will not retry it."

    def goal_progress(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        goal = state.goal(str(args.get("goal_id") or ""))
        if goal is None:
            return "error: unknown goal_id"
        step = str(args.get("step") or "").strip()
        nxt = str(args.get("next") or "").strip()
        if not step or not nxt:
            return "error: step and next are required"
        goal.progress = {"step": step[:500], "evidence": str(args.get("evidence") or "").strip()[:500],
                         "next": nxt[:500]}
        self.store.append_ledger(state.chat_key, "progress", goal_id=goal.id, step=step[:200])
        self.store.save(state)
        return f"Progress of {goal.id} recorded. It is shown in the goal context of the next turns."

    def goal_claim(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        goal = state.goal(str(args.get("goal_id") or ""))
        if goal is None:
            known = ", ".join(g.id for g in state.goals) or "none set"
            return (f"error: unknown goal_id {str(args.get('goal_id') or '')!r}. "
                    f"Known goal ids: {known}. Pass goal_id at the top level, not inside claim.")
        if goal.done:
            return f"Goal {goal.id} is already confirmed by the judge. It stays done."
        try:
            evidence = serialize_completion_claim(args.get("claim"))
        except ClaimError as exc:
            return f"claim rejected: {exc}"
        conv = str(kwargs.get("conversation_id") or kwargs.get("session_id") or "")
        _JUDGE_PARENTS[conv] = str(kwargs.get("session_id") or "")
        try:
            verdict = run_judge(self.ctx, goal.text, goal.check, evidence,
                                parent_toolsets=kwargs.get("_parent_toolsets"),
                                timeout=self.settings.get("judge_timeout"))
        finally:
            _JUDGE_PARENTS.pop(conv, None)
        state = self.store.load(state.chat_key)  # the judge run may take minutes
        goal = state.goal(goal.id) or goal
        if verdict.done:
            goal.done = True
            goal.status = "done"
            goal.judge_reason = ""
            self.store.append_ledger(state.chat_key, "confirmed", goal_id=goal.id,
                                     method=verdict.method)
            self.store.save(state)
            reply = f"Goal {goal.id} confirmed by the judge ({verdict.method}). It stays done."
            if not state.unfinished():
                reply += FINAL_REPORT_NUDGE
            return reply
        if verdict.unavailable:
            # L10: the judge did not answer; the goal keeps its status.
            self.store.append_ledger(state.chat_key, "judge_unavailable", goal_id=goal.id,
                                     reason=verdict.reason[:200])
            return (f"Judge unavailable for {goal.id} ({verdict.reason}). The goal keeps its status. "
                    "Submit the same claim again; do not redo the work.")
        if verdict.wait:
            goal.status = "waiting"
            goal.note = verdict.reason
            self.store.append_ledger(state.chat_key, "waiting", goal_id=goal.id,
                                     reason=verdict.reason[:200], method=verdict.method)
            self.store.save(state)
            return (f"Goal {goal.id} waits: {verdict.reason}. The loop does not nudge it. Submit the "
                    "claim again when the effect can be examined.")
        same = _similar(goal.last_reject, verdict.reason)
        goal.rejects = goal.rejects + 1 if same else 1
        goal.last_reject = verdict.reason
        goal.judge_reason = verdict.reason
        goal.status = "in_work"
        self.store.append_ledger(state.chat_key, "rejected", goal_id=goal.id,
                                 reason=verdict.reason[:200], method=verdict.method,
                                 similar=goal.rejects)
        reply = f"Goal {goal.id} rejected by the judge. Missing: {verdict.reason}."
        lang = _language(state, self.store, self.settings, state.chat_key)
        if goal.rejects >= self.settings.get("reject_pause_after") > 0:
            _pause(self.store, state, [goal], verdict.reason,
                   _dtext(lang)["reject"].format(gid=goal.id, n=goal.rejects, why=verdict.reason),
                   state.chat_key, "paused_rejects")
            reply += " The same reason came again; the goal is paused for an owner decision."
        elif goal.rejects >= self.settings.get("reject_replan_after") > 0:
            reply += (" The same reason came again: plan again. Name a different approach before "
                      "you continue, and fix each missing point.")
        else:
            reply += " Fix each missing point, then claim again."
        self.store.save(state)
        return reply


# --- The pre_stop hook ------------------------------------------------------------------------

def _goal_context(state: GoalState, lead: str) -> str:
    lines = [lead]
    for goal in state.unfinished():
        line = f"- {goal.id} [{goal.status}]: {goal.text} | check: {goal.check}"
        if goal.judge_reason and goal.status == "in_work":
            line += f" | last rejection: {goal.judge_reason}"
        if goal.status == "waiting" and goal.note:
            line += f" | waits: {goal.note}"
        if goal.progress:
            line += (f" | last step: {goal.progress.get('step', '')}"
                     f" | next: {goal.progress.get('next', '')}")
        lines.append(line)
    return "\n".join(lines)


def make_pre_stop(store: GoalStore, settings: Optional[Settings] = None):
    """Build the ``pre_stop`` callback: continue while open goals remain and the go is given."""
    settings = settings or Settings()

    def _pre_stop(session_id: str = "", platform: str = "", attempt: int = 0,
                  final_response: str = "", conversation_id: str = "",
                  **_: Any) -> Optional[dict]:
        if platform in EXCLUDED_PLATFORMS:
            return None
        try:
            key = _chat_key(conversation_id or session_id)
            store.remember_session(session_id, key)
            state = store.load(key)
        except Exception:
            logger.debug("pre_stop state load failed", exc_info=True)
            return None
        if not state.go or not state.goals:
            return None
        lang = _language(state, store, settings, key)
        # L8: the owner asked to stop.
        if store.gate_count(key, "stop") is not None:
            store.close_gate(key, "stop")
            state.go = False
            _cancel_follow_up(store, state.chat_key)
            state.decisions.append(_dtext(lang)["owner_stop"])
            store.append_ledger(state.chat_key, "paused_by_owner")
            store.open_gate(key)
            store.save(state)
            return None
        # L13: after a restart only an owner message of this process re-arms the loop.
        if state.boot_id != BOOT_ID:
            if store.gate_count(key, f"boot{BOOT_ID[:12]}") is None:
                store.append_ledger(state.chat_key, "restart_hold")
                return None
            state.boot_id = BOOT_ID
        open_goals = state.open_goals()
        if not open_goals:
            store.save(state)
            return None
        state.nudge_seq += 1
        state.continuations = attempt + 1
        # L2: idle detection by tool calls since the last continuation.
        if state.tool_calls_at_nudge >= 0 and state.tool_calls == state.tool_calls_at_nudge:
            state.idle_nudges += 1
        else:
            state.idle_nudges = 0
        state.tool_calls_at_nudge = state.tool_calls
        idle_pause = settings.get("idle_pause_after")
        if idle_pause and state.idle_nudges >= idle_pause:
            gids = ", ".join(g.id for g in open_goals)
            _pause(store, state, open_goals, "no tool call in a row",
                   _dtext(lang)["idle"].format(n=state.idle_nudges, gids=gids), key,
                   "paused_idle")
            store.save(state)
            return None
        # L1: budget per goal over all turns and runs.
        budget = settings.get("budget_per_goal")
        over = []
        for goal in open_goals:
            goal.nudges += 1
            if budget and goal.nudges >= budget:
                over.append(goal)
        for goal in over:
            _pause(store, state, [goal], f"budget of {budget} continuations reached",
                   _dtext(lang)["budget"].format(gid=goal.id, n=budget), key,
                   "paused_budget")
        open_goals = state.open_goals()
        store.save(state)
        if not open_goals:
            return None
        message = _goal_context(state, f"[goal-loop] {len(open_goals)} goal(s) are still open. "
                                       "Continue the work on these goals:")
        replan = settings.get("idle_replan_after")
        if replan and state.idle_nudges >= replan:
            message += ("\nThe last continuations had no tool call. Plan again: name a different "
                        "approach and do one concrete step with a tool now.")
        message += (
            "\nWhen a goal is done, submit goal_claim with real evidence for the judge; record each "
            "step with goal_progress. If a goal needs an owner decision, record it with "
            "goal_decision and keep working the others. Use goal_block only for an obstacle that "
            "stays. End your message with a stop-cause line, one of: "
            f"{', '.join(STOP_CAUSES)}."
        )
        return {"action": "continue", "message": message}

    return _pre_stop


# --- The status display (transform_llm_output) -------------------------------------------------

def _display_block(state: GoalState, language: str = "en") -> str:
    """Decide what this turn's answer shows and record it as shown (the caller saves the state)."""
    if state.list_changed:
        state.mark_all_shown()
        state.final_shown = state.all_done()
        return status_block(state, language, go_question=not state.go)
    if state.all_done():
        if not state.final_shown:
            state.final_shown = True
            state.mark_all_shown()
            return status_block(state, language)
        # goal-loop-quiet decision 4a: a decision recorded after the final list still shows once
    owner_goals = [g for g in state.goals
                   if g.status in ("blocked", "paused") and not g.done and not g.block_shown]
    start = min(max(state.decisions_shown, 0), len(state.decisions))
    decisions = [(index, text) for index, text in enumerate(state.decisions, start=1)
                 if index > start]
    if not owner_goals and not decisions:
        return ""
    for goal in owner_goals:
        goal.block_shown = True
    state.decisions_shown = len(state.decisions)
    return owner_block(owner_goals, decisions, language)


def _cancel_follow_up(store: GoalStore, chat_key: str) -> None:
    """The owner stopped the loop: a scheduled follow-up run posts nothing (resume_run.due)."""
    try:
        from . import resume
        path = resume.request_path(store.base_dir, chat_key)
        request = resume.load_request(path)
        if request and request.get("status") == "scheduled":
            resume.mark(path, "cancelled")
    except Exception:
        logger.debug("goal-loop follow-up run not cancelled", exc_info=True)


def _schedule_follow_up(store: GoalStore, state: GoalState, settings: Settings,
                        session_id: str) -> bool:
    """L15: schedule one follow-up run of this session (resume.py); False when off, at the
    limit, still pending, or not possible (no session, no cron module)."""
    limit = settings.get("resume_max")
    if not settings.get("resume") or not limit or state.resumes >= limit:
        return False
    try:
        from . import resume
        job = resume.schedule_resume(store.base_dir, state.chat_key, str(session_id or ""),
                                     state.last_model, str(settings.get("resume_delay") or "1m"))
    except Exception:
        logger.debug("goal-loop follow-up run not scheduled", exc_info=True)
        return False
    if not job:
        return False
    state.resumes += 1
    store.append_ledger(state.chat_key, "resume_scheduled", job_id=job, n=state.resumes)
    return True


def make_transform_llm_output(store: GoalStore, settings: Optional[Settings] = None):
    """Append the goal display to the final answer of a turn, only when there is news."""
    settings = settings or Settings()

    def _transform(response_text: str = "", text: str = "", session_id: str = "",
                   platform: str = "", conversation_id: str = "", **_: Any) -> Optional[str]:
        if platform in EXCLUDED_PLATFORMS:
            return None
        body = response_text if isinstance(response_text, str) and response_text else text
        if not isinstance(body, str):
            return None
        try:
            state = store.load(_chat_key(conversation_id or session_id))
        except Exception:
            return None
        if not state.goals:
            return None
        lang = _language(state, store, settings, state.chat_key)
        limit = settings.get("resume_max")
        waiting = state.open_goals() if state.go else []
        if waiting and settings.get("resume") and limit and state.resumes >= limit:
            gids = ", ".join(g.id for g in waiting)
            _pause(store, state, waiting, f"{state.resumes} follow-up runs without completion",
                   _dtext(lang)["resume_max"].format(n=state.resumes, gids=gids), state.chat_key,
                   "paused_resume_max")
        block = _display_block(state, lang)
        # L15: the turn ends while goals are still open after the go.
        open_goals = state.open_goals() if state.go else []
        tail = ""
        if open_goals and platform == "api_server" and _schedule_follow_up(store, state, settings,
                                                                          session_id):
            tail = "\n\n" + _dtext(lang)["resume_line"].format(
                n=len(open_goals), gids=", ".join(g.id for g in open_goals),
                delay=settings.get("resume_delay"))
        elif open_goals and settings.get("paused_hint"):
            tail = "\n\n" + _dtext(lang)["paused_line"].format(
                n=len(open_goals), gids=", ".join(g.id for g in open_goals))
        if not block and not tail:
            return None
        try:
            store.save(state)
        except Exception:
            logger.debug("goal display state save failed", exc_info=True)
        if not body.strip():
            body = labels(lang)["no_report"]
        return body + tail + ("\n" + block if block else "")

    return _transform


# --- The go gate, stop words, language and goal context (pre_llm_call) ------------------------

def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(str(part.get("text") or "") for part in message
                         if isinstance(part, dict) and part.get("type") in (None, "text", "input_text"))
    return ""


def _is_owner_message(user_message: Any, conversation_history: Any) -> bool:
    """True for a human turn: not empty, no gateway notification footer, and no synthesized
    ``display_kind`` (a ``/steer`` row and an API client turn under the session authority count)."""
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


def _is_follow_up(store: GoalStore, session_id: Any, conversation_id: Any, text: str) -> bool:
    """True for the turn of a follow-up run: it starts with the resume prefix (an owner who types
    the prefix is treated the same way, which only withholds a go)."""
    from . import resume
    return resume.is_resume_text(text)


def _take_request(store: GoalStore, key: str, text: str) -> None:
    """The follow-up turn of request N started: mark N as taken, so the end of this turn can
    schedule the next follow-up run (a request that is only "running" still blocks a new one)."""
    try:
        from . import resume
        path = resume.request_path(store.base_dir, key)
        request = resume.load_request(path)
        if request and request.get("nonce") == resume.nonce_of(text):
            resume.mark(path, "taken")
    except Exception:
        logger.debug("goal-loop follow-up request not marked", exc_info=True)


def _note_model(store: GoalStore, gate: str, model: Any) -> None:
    """Keep the model of the last owner turn, so a follow-up run uses the same model."""
    if not model or not store.exists(gate):
        return
    state = store.load(gate)
    if state.last_model != str(model):
        state.last_model = str(model)
        store.save(state)


def make_pre_llm_call(store: GoalStore, settings: Optional[Settings] = None):
    """For each owner message: count it for the go and change gates, mark it for this process
    (L13), detect a stop word (L8) and the language (O2), and return the goal context (L4)."""
    settings = settings or Settings()

    def _pre_llm_call(session_id: str = "", platform: str = "", user_message: Any = None,
                      conversation_history: Any = None, model: str = "", **_: Any) -> Optional[str]:
        if platform in EXCLUDED_PLATFORMS:
            return None
        try:
            text = _message_text(user_message).strip()
            if _is_follow_up(store, session_id, _.get("conversation_id"), text):
                # L15: a follow-up run of the plugin is never an owner message (no go, no
                # change approval, no restart re-arm); it gets the goal context.
                key = _gate_for(store, session_id, _.get("conversation_id"))
                _take_request(store, key, text)
                if store.exists(key):
                    state = store.load(key)
                    if state.go and state.unfinished():
                        return _goal_context(state, "[goal-loop] Goals of this chat (owner-approved; "
                                                    "do not change them without the owner):")
                return None
            if not _is_owner_message(user_message, conversation_history):
                return None
            gate = _gate_for(store, session_id, _.get("conversation_id"))
            store.bump_gate(gate)
            store.bump_gate(gate, "change")
            store.open_gate(gate, f"boot{BOOT_ID[:12]}")
            _note_model(store, gate, model)
            store.add_language_vote(gate, detect_language(text))
            words = [str(w).lower() for w in (settings.get("stop_words") or [])]
            if text.lower().strip(" .!") in words:
                store.open_gate(gate, "stop")
            key = gate
            if not store.exists(key):
                return None
            state = store.load(key)
            if state.go and state.unfinished() and text.lower().strip(" .!") not in words:
                return _goal_context(state, "[goal-loop] Goals of this chat (owner-approved; "
                                            "do not change them without the owner):")
        except Exception:
            logger.debug("goal-loop pre_llm_call failed", exc_info=True)
        return None

    return _pre_llm_call


# --- Tool counting (post_tool_call) and the judge guard (pre_tool_call) -----------------------

def make_post_tool_call(store: GoalStore):
    """Count tool calls per chat for the idle detection (L2)."""

    def _post_tool_call(tool_name: str = "", session_id: str = "", conversation_id: str = "",
                        **_: Any) -> None:
        if not tool_name or tool_name in _NO_PROGRESS_TOOLS:
            return None
        try:
            key = _chat_key(conversation_id or session_id)
            if not store.exists(key):
                return None
            state = store.load(key)
            if not state.go:
                return None
            state.tool_calls += 1
            state.idle_nudges = 0
            store.save(state)
        except Exception:
            logger.debug("goal-loop post_tool_call failed", exc_info=True)
        return None

    return _post_tool_call


def make_pre_tool_call():
    """Veto mutating tools in a judge child (L12). The child inherits the parent's conversation_id;
    while the parent waits in ``goal_claim``, a tool call with that conversation_id and another
    session id comes from the judge."""

    def _pre_tool_call(tool_name: str = "", args: Any = None, session_id: str = "",
                       conversation_id: str = "", **_: Any) -> Optional[dict]:
        parent = _JUDGE_PARENTS.get(str(conversation_id or ""))
        if parent is None or str(session_id or "") == parent:
            return None
        if tool_name not in _JUDGE_ALLOWED_TOOLS:
            return {"action": "block",
                    "message": f"goal-loop judge: tool {tool_name} is not allowed; the judge only reads."}
        if tool_name == "terminal":
            command = str((args or {}).get("command") or "") if isinstance(args, dict) else ""
            if _MUTATING_COMMAND.search(command):
                return {"action": "block",
                        "message": "goal-loop judge: this command can change state; use read-only commands."}
        return None

    return _pre_tool_call


# --- The usage hint (system-prompt section) ----------------------------------------------------

def prompt_section(session_info: Mapping[str, Any]) -> str:
    """The usage hint for a normal session; empty (so not rendered) for EXCLUDED_PLATFORMS."""
    platform = str((session_info or {}).get("platform") or "")
    if platform in EXCLUDED_PLATFORMS:
        return ""
    return USAGE_HINT


def register(ctx) -> None:
    """Register the goal tools, the hooks, the judge task and the usage hint."""
    store = GoalStore(_state_dir())
    settings = Settings(ctx)
    tools = _Tools(store, ctx, settings)
    register_task = getattr(ctx, "register_auxiliary_task", None)
    if callable(register_task):
        try:
            register_task(JUDGE_TASK, display_name="Goal loop judge",
                          description="Judge of the goal-loop plugin (use another model family than the worker).")
        except Exception:
            logger.warning("goal-loop judge task not registered", exc_info=True)
    specs = (
        ("goal_set", _GOAL_SET_SCHEMA, tools.goal_set, "🎯",
         ("Record the goals, each with one falsifiable check of the content, in the owner's language. "
          "Call before any goal work. The plugin shows the list and the go question.")),
        ("goal_go", _EMPTY_SCHEMA, tools.goal_go, "🟢",
         ("Record the owner's go (also to resume paused goals). Call it only after the owner gave "
          "the go in a new message; otherwise it is refused.")),
        ("goal_status", _EMPTY_SCHEMA, tools.goal_status, "📋", GOAL_STATUS_DESCRIPTION),
        ("goal_claim", _GOAL_CLAIM_SCHEMA, tools.goal_claim, "✅",
         ("Submit a structured completion claim for one goal; a fresh-context judge verifies the "
          'real state. goal_id is top-level; claim needs a summary and checks you ran, e.g. '
          'claim={"summary": "wrote the file", "checks": [{"command": "grep -qx READY f", '
          '"result": "passed", "exitCode": 0}]}.')),
        ("goal_progress", _GOAL_PROGRESS_SCHEMA, tools.goal_progress, "📍",
         "Record the last step, its evidence and the next step of a goal."),
        ("goal_decision", _GOAL_DECISION_SCHEMA, tools.goal_decision, "🧭",
         "Record an owner decision (options + recommendation) for the decision list."),
        ("goal_block", _GOAL_BLOCK_SCHEMA, tools.goal_block, "⛔",
         ("Mark a goal blocked by an obstacle that stays (reason required). Accepted only when the "
          "same obstacle stays for several rounds; hard work is no blocker.")),
    )
    for name, schema, handler, emoji, description in specs:
        ctx.register_tool(
            name=name, toolset="goal_loop",
            schema={"name": name, "description": description, "parameters": schema},
            handler=handler, emoji=emoji, description=description)
    ctx.register_hook("pre_stop", make_pre_stop(store, settings))
    ctx.register_hook("transform_llm_output", make_transform_llm_output(store, settings))
    ctx.register_hook("pre_llm_call", make_pre_llm_call(store, settings))
    ctx.register_hook("post_tool_call", make_post_tool_call(store))
    ctx.register_hook("pre_tool_call", make_pre_tool_call())
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section):
        try:
            register_section(PROMPT_SECTION_ID, prompt_section)
        except Exception:
            logger.warning("goal-loop usage hint not registered", exc_info=True)
    logger.info("goal-loop plugin registered: 7 tools, 5 hooks, judge task, usage hint")
