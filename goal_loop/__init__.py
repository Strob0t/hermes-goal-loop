"""hermes-goal-loop: a goal loop for Hermes as a plugin (v0.3).

0.4.10 (z13): three owner rules.
R1 Pause: each owner message pauses the loop (go False, a pending follow-up run is cancelled, the
   background and owner waits are cleared, ledger ``paused_by_owner_message``). Only a message with the
   word ``go`` as a token keeps the go ('4B 5C go', 'list ok go') or gives it again after such a pause.
   While paused, ``pre_stop`` does not continue, no follow-up run is scheduled or injected, and the
   background watchdog does not continue the chat. The paused turn gets a context note: answer the
   message, put any question at the end, ask for 'go'; ``goal_go`` stays refused without a 'go' message.
   Owner words in other languages come from the optional file ``<state dir>/vocabulary.json`` (keys
   ``go_words`` and ``consent_patterns``); without the file only the English defaults apply.
R2 Deferred list: ``goal_decision`` takes an optional ``goal_id``. A goal with an open linked decision,
   or with status blocked or paused, waits for the owner; the other unfinished goals are actionable
   (a goal that waits for a background process or a delegation is actionable later). While a goal is
   actionable, a new decision, a ``goal_block`` and a refused ``goal_set`` change go into
   ``state.deferred``; the answer block does not show them and the loop continues with the actionable
   goals (also after an answer with the stop cause owner-question). When no goal is actionable, the
   answer block shows the deferred list one time and the loop waits for the owner.
R3 English only: all plugin texts are English. Lines that explain the loop state (wait, pause,
   restart, follow-up run) go to the model as context, and the model writes them in the owner's
   language at the end of its answer. ``scripts/check_english.py`` checks the tracked files.
R4 Evidence: ``goal_decision`` requires ``evidence`` (the command or file that showed the live state,
   and its time); without it nothing is recorded. The prompt section and the pause note tell the model
   to check each point against the live state in this turn, and the judge accepts no evidence that
   comes only from the chat, memory or a worker's report.

0.4.9: the background watchdog does not continue while a turn of the session runs (a turn mark older
than turn_active_max does not count) and waits background_end_grace seconds after the first tick that
sees all awaited processes ended; '8. A', '8: A', '8 - A' close decision 8, also with a '?' in the message.
0.4.8: an owner message closes decisions only on a number with an option, a go word or explicit consent;
the goal block shows the numbers it closed.
0.4.7: an owner answer closes decisions by number ('5A' closes decision 5; 'go' alone closes all); the
first turn after a restart says that the loop is held; a watchdog continues a loop that waits for
background processes that do not run any more (or after background_wait_max seconds); each resume turn
writes one INFO line with its nonce and job id.

0.4.5 (r32): the stop-cause line of an answer (done, owner-question, blocker) holds the loop until
the next owner message.

1. Before the work: the model sets the goals (``goal_set``), each with one falsifiable check. The
   plugin appends the goal list and the go question to the answer of that turn (one time; the model
   does not repeat the list). No loop runs before ``goal_go``. **Go rule:** ``goal_go`` is accepted
   only when an owner message with the word 'go' came into the chat after the last ``goal_set`` or
   pause of the session. The ``pre_llm_call`` hook counts these messages (gateway notifications,
   subagent and cron turns do not count).
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
   - After a Hermes restart the loop does not continue until an owner message with 'go' comes.

3. The judge: ``goal_claim`` submits a structured claim; a fresh-context judge checks the real state
   read-only (see :mod:`goal_loop.judge`). Verdicts: done, wait (the goal waits without nudges),
   not-done (with the missing points). A judge that does not answer keeps the goal status.
4. Context: each owner turn with an active go gets the open goals (text, check, progress) as context
   (``pre_llm_call``), so a compression cannot remove them. ``goal_progress`` records the last step,
   its evidence and the next step of a goal.
5. Status, shown quietly (``transform_llm_output``): the full list on ``goal_set`` (with the go
   question as last line before the go), after that only goals that need the owner and new
   decisions, each one time, and the final list one time. The labels are English; goal and decision
   texts stay as the model wrote them.
6. Changes after the go: ``goal_set`` is refused until an owner message approves the change;
   then the judge-confirmed goals stay done.

The plugin never starts a turn in an API session and never edits core files (plugin rule).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from typing import Any, Mapping, Optional

from .claim import ClaimError, serialize_completion_claim
from .judge import JUDGE_TASK, run_judge
from .state import LABELS, OWNER_STATUSES, Goal, GoalState, GoalStore, owner_block, status_block

logger = logging.getLogger("goal_loop")

_STATE_DIR_ENV = "GOAL_LOOP_STATE_DIR"
_STATE_SUBDIR = "goal-loop"

_judge_sleep = time.sleep  # a test replaces it (no real pause between judge attempts)


def _process_boot_id() -> str:
    """One id per PROCESS, not per module import (0.4.3, F-reload-hold).

    A plugin reload imports this module again; an id made at import (uuid4) then changed and held
    every chat with a go like after a restart. The id is a hash of the process id and the process
    start time (``/proc/self/stat`` field 22), so a reload keeps it and a new process gets a new one.
    Without ``/proc`` it uses the module-level cache of an earlier import, else a random id."""
    import hashlib
    import sys
    cached = getattr(sys.modules.get("goal_loop_boot"), "ID", "") if "goal_loop_boot" in sys.modules else ""
    pid = os.getpid()
    start = ""
    try:
        with open("/proc/self/stat", encoding="ascii") as handle:
            start = handle.read().rsplit(")", 1)[1].split()[19]
    except Exception:
        start = ""
    if start:
        return hashlib.sha256(f"{pid}:{start}".encode()).hexdigest()[:32]
    if cached:
        return str(cached)
    import types
    holder = types.ModuleType("goal_loop_boot")
    holder.ID = uuid.uuid4().hex
    sys.modules["goal_loop_boot"] = holder
    return holder.ID


# One id per process: a state that a different process drove is held after a restart (L13).
BOOT_ID = _process_boot_id()

STOP_CAUSES = ("done", "owner-question", "blocker", "no-statement")
# 0.4.5 (r32): an answer whose stop-cause line names one of these causes ends the loop until the next
# owner message; "no-statement" lets the loop go on.
HOLD_CAUSES = ("done", "owner-question", "blocker")
# 0.4.10 (z13): an owner message with one of these words as a token keeps or gives the go; any other
# owner message pauses the loop. More words (other languages) come from <state dir>/vocabulary.json.
GO_WORDS = ("go",)
VOCABULARY_FILE = "vocabulary.json"
_STOP_CAUSE_LINE = re.compile(
    r"^\W*stop[- ]?cause\W*:\W*(%s)\b" % "|".join(re.escape(c) for c in STOP_CAUSES),
    re.IGNORECASE | re.MULTILINE)


def _stop_cause(text: Any) -> str:
    """The cause of the last stop-cause line in the last 12 lines of an answer, or ''."""
    if not isinstance(text, str) or not text:
        return ""
    found = _STOP_CAUSE_LINE.findall("\n".join(text.strip().splitlines()[-12:]))
    return found[-1].lower() if found else ""


# 0.4.7 (z3): a decision number with an option letter: '5A', '2b', '4 C' (not 'r4b', not 'z4 C').
# 0.4.9 (z11): also with a separator: '8. A', '8: A', '8 - A', '8) A' (not '0.4.8 A', not '20:15 A').
_DECISION_NUMBER = re.compile(r"(?<![\w.:])(\d{1,2})\s*(?:[.:)\-–]\s*)?[a-dA-D](?![\w])")
# 0.4.8 (z10): explicit consent closes all open decisions ('ok', 'all recommendations', ...).
_CONSENT = re.compile(r"^\s*(?:yes[\s,.!]+)?(?:all\s+)?(?:your\s+|the\s+)?"
                      r"(?:recommendations?|ok(?:ay)?|agreed|as\s+recommended)(?![\w])", re.IGNORECASE)


def _vocabulary(base_dir: str) -> tuple[tuple[str, ...], list]:
    """0.4.10 (z13): the go words and consent patterns: the English defaults plus the optional file
    ``<state dir>/vocabulary.json`` ({"go_words": [...], "consent_patterns": [regex, ...]}). A consent
    pattern matches at the start of a message. A missing or bad file gives the defaults only."""
    words, patterns = list(GO_WORDS), [_CONSENT]
    try:
        with open(os.path.join(base_dir, VOCABULARY_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return tuple(words), patterns
    if not isinstance(data, dict):
        return tuple(words), patterns
    for word in data.get("go_words") or []:
        if isinstance(word, str) and word.strip() and word.strip().lower() not in words:
            words.append(word.strip().lower())
    for pattern in data.get("consent_patterns") or []:
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        try:
            patterns.append(re.compile(r"^\s*(?:%s)(?![\w])" % pattern, re.IGNORECASE))
        except re.error:
            logger.debug("goal-loop vocabulary: bad consent pattern %r", pattern)
    return tuple(words), patterns


def _has_go(text: str, words: tuple[str, ...]) -> bool:
    """0.4.10 (z13): the message has a go word as a token ('go', '4B 5C go', 'list ok GO')."""
    return any(re.search(r"(?<!\w)%s(?!\w)" % re.escape(w), text or "", re.IGNORECASE) for w in words)


def _close_decisions(store: "GoalStore", state: "GoalState", text: str,
                      vocabulary: Optional[tuple] = None) -> list[int]:
    """0.4.8 (z10): an owner message closes decisions only on (1) a decision number with an option
    ('6A', '1c, 2b' close those numbers), (2) a message that is only a go word (closes all), or (3)
    explicit consent at the start ('ok', 'all recommendations'; closes all). A request, an order or a
    question without a number closes none. When decisions stay open, they are shown again
    (decisions_shown = 0). Returns the closed numbers; they are shown once in the next goal block
    (decisions_just_closed)."""
    words, patterns = vocabulary or _vocabulary(store.base_dir)
    open_numbers = [i for i, _ in state.open_decisions()]
    t = (text or "").strip()
    if not open_numbers or not t:
        return []
    numbers = {int(m.group(1)) for m in _DECISION_NUMBER.finditer(t)}
    if t.lower().strip(" .!") in words:
        close = open_numbers
    elif numbers:
        close = [i for i in open_numbers if i in numbers]
    elif any(p.match(t) for p in patterns):
        close = open_numbers
    else:
        return []
    if not close:
        return []
    state.decisions_closed = sorted(set(state.decisions_closed) | set(close))
    state.decisions_just_closed = sorted(close)
    store.append_ledger(state.chat_key, "decisions_closed", numbers=close)
    if state.open_decisions():
        state.decisions_shown = 0  # show the decisions that stay open once more
    return close


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
    # 0.4.3: the judge is tried again before it counts as unavailable. judge_timeout is the total
    # budget of all attempts; judge_attempt_timeout limits one attempt; judge_backoff is the pause
    # before attempt n+1 (backoff * n seconds).
    "judge_attempts": 3,
    "judge_attempt_timeout": 120,
    "judge_backoff": 5,
    # 0.4.3: at plugin start a state with a go and open goals, without activity for this many days,
    # is paused (0 = off).
    "stale_days": 1,
    "paused_hint": True,
    # L15 follow-up runs (owner decision 5): a run that ends after the go with open goals
    # schedules one more run of the same session; resume_max follow-up runs per go (0 = off).
    "resume": True,
    "resume_delay": "1m",
    "resume_max": 20,
    # r4: a turn that a Hermes restart cut off gets one continuation turn after the start.
    "restart_resume": True,
    "restart_resume_delay": "2m",
    # 0.4.7 (z5): the background watchdog checks every background_watch_every seconds; a wait for
    # background processes ends when none of them runs, or after background_wait_max seconds (0 = off).
    "background_watch_every": 60,
    "background_wait_max": 3600,
    # 0.4.9 (z11): no watchdog continuation while a turn of the session runs (a turn mark older than
    # turn_active_max seconds does not count: a hard-stopped turn never ends); and a grace of
    # background_end_grace seconds after the first tick that sees all awaited processes ended.
    "turn_active_max": 600,
    "background_end_grace": 120,
}

FINAL_REPORT_NUDGE = (
    " All goals are now confirmed. Write the final report for the owner in this message, in the "
    "owner's language: list each goal, the judge result, and the evidence."
)

# 0.4.10 (R4): evidence before claims and proposals.
EVIDENCE_RULE = ("Before you propose options, list open points or claim a goal, check each point against "
                 "the live state in this turn; do not rely on the chat history or memory.")
EVIDENCE_MIN = 12
EVIDENCE_REFUSED = ("error: goal_decision refused: examine the live state first and pass the result as "
                    "evidence (the command or file that showed it, and its time). Nothing was recorded.")

USAGE_HINT = (
    "Goal loop: for a multi-step task, first call goal_set with the goals, each with one "
    "falsifiable check. Write goal text and check in the language that the owner mostly uses in "
    "the chat; write internal texts (todo, notes, briefs) as usual. Rules for checks: (1) a check "
    "examines the content (evidence with a source, a real run, a value), not only the form (a "
    "file exists, a section count); (2) the judge must be able to examine the check when you "
    "submit the claim, so no check on a later chat report; (3) name the version that runs live as "
    "the base. Split a large goal into smaller goals. The plugin appends the goal list and the go "
    "question to your answer: do not repeat the list and do not ask for the go yourself. Call "
    "goal_go only after the owner wrote 'go' in a new message. Each other owner message pauses the "
    "loop: answer it and ask for 'go'. Record progress with goal_progress. If a goal needs the owner, "
    "record a goal_decision with its goal_id and continue with the other goals. " + EVIDENCE_RULE
)

GO_REFUSED = (
    "error: goal_go refused: no owner message with the word 'go' came after the last goal_set or "
    "pause. End your turn; the plugin shows the goal list and the go question. Call goal_go only in "
    "the turn that answers the owner's go. The loop does not start."
)

CHANGE_REFUSED = (
    "error: goal_set refused: the owner gave the go for the current goal list. Ask the owner to "
    "approve the change (name the goals that change), end your turn, and call goal_set again in "
    "the turn that answers the owner's approval. Judge-confirmed goals stay done."
)

CHANGE_DEFERRED = (
    "goal_set refused: the owner gave the go for the current goal list. The change is recorded in the "
    "deferred list; the owner sees it when no goal can be worked. Do not ask the owner now; continue "
    "with goal(s) {ids}."
)

GOAL_STATUS_DESCRIPTION = (
    "Return the goal list with each goal's check and status as finished text. Give the returned "
    "text to the owner unchanged, word for word: do not summarize, shorten, reorder or reformat it."
)

PROMPT_SECTION_ID = "goal-loop"
# Shown before the block when the model turn is empty.
NO_REPORT_MARKER = LABELS["no_report"]

_INTERNAL_FOOTER_PREFIX = "[INTERNAL NOTIFICATION"
_STEER_DISPLAY_KIND = "steer"
_API_TURN_DISPLAY_KIND = "internal_notification"

# Tools that do not show progress on a goal (idle detection, L2).
_NO_PROGRESS_TOOLS = frozenset({"goal_status", "goal_decision", "goal_go", "goal_set",
                                # 0.4.4 (C7): a task-list update alone is no progress on a goal
                                "todo_list", "todo"})
# 0.4.4 (C7): Hermes restores the task list after a context compression as a turn that starts with
# this line; it is no owner message (no go, no change approval).
_TASK_LIST_RESTORE_PREFIX = "[Your active task list was preserved across context compression]"
# 0.4.4: an owner message that interrupts a running turn (agent.redirect) does not start a new turn,
# so pre_llm_call does not see it; Hermes appends it as a user row with this api_content prefix.
_REDIRECT_PREFIX = "[Context from the interrupted assistant response]"

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


def _similar(a: str, b: str) -> bool:
    """Two rejection reasons name the same problem (word overlap of at least one half)."""
    wa = set(re.findall(r"[\w./-]{3,}", (a or "").lower()))
    wb = set(re.findall(r"[\w./-]{3,}", (b or "").lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.5


DECISION_TEXT = {
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
    "stale": ("The loop had no activity for {days} days and is paused (goals: {gids}). "
              "Options: A) continue (answer 'go'); B) change the goals; C) remove the goals. "
              "Recommendation: B if the goals can be out of date, else A."),
    "resume_max": ("The loop made {n} follow-up runs without completion and is paused "
                   "(goals: {gids}). Options: A) continue (answer 'go'); B) change the goals; "
                   "C) remove the goals. Recommendation: A if the goals show progress."),
}

# 0.4.10 (z13): the loop state after a turn, for the model (context of the next turn), not for the chat.
LOOP_NOTES = {
    "paused": "The loop is paused: {n} goal(s) are open ({gids}). The owner writes 'go' to continue it.",
    "resume": ("The loop continues in {delay} in a follow-up run: {n} goal(s) open ({gids}). The answer "
               "appears in this chat. Any owner message pauses it."),
    "wait": ("The loop waits ({why}): {n} goal(s) open ({gids}). It goes on when the result is there; "
             "any owner message pauses it."),
    "inject": "The loop continues in this chat now: {n} goal(s) open ({gids}). Any owner message pauses it.",
}
_LOOP_NOTE_LEAD = ("[goal-loop] Loop state (write this in the owner's language at the end of your "
                   "answer, in one short line): ")
RESTART_NOTE = ("[goal-loop] Hermes restarted: the goal loop is held ({n} goal(s) open: {gids}). Write "
                "this in the owner's language at the end of your answer: the loop is held after a "
                "restart, and the owner writes 'go' to continue it. Do not call goal_go yourself.")
PAUSE_NOTE = ("[goal-loop] This owner message paused the goal loop ({n} goal(s) open: {gids}). Only a "
              "message with the word 'go' continues it. Answer the owner's message now. Put any question "
              "to the owner at the end of your answer. Then, at the end of your answer, ask the owner to "
              "write 'go' to continue the loop; write this in the owner's language. Do not call goal_go "
              "yourself: it is refused until the owner writes 'go'.")


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


def _actionable(state: GoalState) -> list[Goal]:
    """0.4.10 (R2): the goals that the loop can still work on after the go (not owner-bound)."""
    return state.free_goals() if state.go else []


def _ids(goals: list[Goal]) -> str:
    return ", ".join(g.id for g in goals)


def _defer(state: GoalState, kind: str, text: str, goal_id: str = "", number: int = 0) -> None:
    """0.4.10 (R2): hold an owner item back until no goal can be worked (one entry per item)."""
    if any(e.get("kind") == kind and e.get("text") == text and not e.get("shown") for e in state.deferred):
        return
    state.deferred.append({"ts": round(time.time(), 3), "kind": kind, "text": text,
                           "goal_id": goal_id, "number": number, "shown": False})


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
        "goal_id": {"type": "string",
                    "description": "optional: the goal that waits for this decision (the loop goes on with the others)"},
        "evidence": {"type": "string",
                     "description": ("the command or file that showed the live state in this turn, and its "
                                     "time (required; the chat history or memory is no evidence)")},
    },
    "required": ["text", "evidence"],
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
        change = (state.go or state.owner_paused) and bool(state.unfinished())
        if change:
            # L3: after the go, a changed list needs the owner's approval in a new message.
            count = self.store.gate_count(session, "change")
            if not count:
                if count is None:
                    self.store.open_gate(session, "change")
                free = _actionable(state)
                self.store.append_ledger(state.chat_key, "change_refused", deferred=bool(free))
                if free:
                    _defer(state, "change", "; ".join(f"{g.id}: {g.text} (check: {g.check})" for g in goals))
                    self.store.save(state)
                    return CHANGE_DEFERRED.format(ids=_ids(free))
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
        if change:
            _lift_decision_wait(self.store, state, "change_approved")  # 0.4.5 (r32)
            for entry in state.deferred:
                if entry.get("kind") == "change":
                    entry["shown"] = True
        state.continuations = 0
        state.list_changed = True
        state.final_shown = False
        if not change:
            state.go = False
            state.owner_paused = False
            state.deferred = []
            state.decision_goals = {}
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
        state.owner_paused = False
        state.idle_nudges = 0
        state.tool_calls_at_nudge = -1
        state.decisions_at_nudge = len(state.decisions)  # 0.4.3: base of the owner wait
        _lift_decision_wait(self.store, state, "go")  # 0.4.5 (r32)
        state.boot_id = BOOT_ID
        self.store.append_ledger(state.chat_key, "go", owner_messages=count,
                                 resumed=[g.id for g in paused])
        self.store.save(state)
        self.store.close_gate(gate)
        return f"Go recorded. The loop will work the {len(state.open_goals())} open goal(s) to the end."

    def goal_status(self, args: dict[str, Any], **kwargs: Any) -> str:
        state = self._load(kwargs)
        block = status_block(state)
        return block.lstrip("\n-") if block else "No goals set."

    def goal_decision(self, args: dict[str, Any], **kwargs: Any) -> str:
        text = str(args.get("text") or "").strip()
        if not text:
            return "error: decision text is required"
        evidence = str(args.get("evidence") or "").strip()
        if len(evidence) < EVIDENCE_MIN:
            return EVIDENCE_REFUSED  # 0.4.10 (R4): nothing is recorded
        state = self._load(kwargs)
        gid = str(args.get("goal_id") or "").strip()
        goal = state.goal(gid) if gid else None
        if gid and (goal is None or goal.done):
            return f"error: unknown or finished goal_id {gid!r}"
        state.decisions.append(text)
        number = len(state.decisions)
        if goal is not None:
            state.decision_goals[str(number)] = goal.id
        free = _actionable(state)
        self.store.append_ledger(state.chat_key, "decision", text=text[:200], number=number, goal_id=gid,
                                 evidence=evidence[:200], deferred=bool(free))
        if free:
            _defer(state, "decision", text, goal_id=gid, number=number)
            self.store.save(state)
            return (f"Decision {number} recorded in the deferred list; the owner sees it when no goal can "
                    f"be worked. Do not ask the owner now; continue with goal(s) {_ids(free)}.")
        self.store.save(state)
        return (f"Decision {number} recorded ({len(state.open_decisions())} open). "
                "It will appear in the decision list.")

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
        # C1: a goal that the chat already showed as blocked for the same obstacle is not shown
        # again; only a new reason shows it again.
        shown_before = (goal.status == "blocked" and goal.block_shown
                        and bool(goal.note) and _similar(goal.note, reason))
        goal.status = "blocked"
        goal.note = reason
        goal.block_shown = shown_before
        free = _actionable(state)
        self.store.append_ledger(state.chat_key, "blocked", goal_id=goal.id, reason=reason[:200],
                                 shown_again=not shown_before, deferred=bool(free))
        if free:
            _defer(state, "block", reason, goal_id=goal.id)
            self.store.save(state)
            return (f"Goal {goal.id} marked blocked and recorded in the deferred list; the owner sees it "
                    f"when no goal can be worked. Do not ask the owner now; continue with goal(s) {_ids(free)}.")
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
            retry_key = state.chat_key

            def _note_retry(attempt: int, reason: str) -> None:
                self.store.append_ledger(retry_key, "judge_retry", goal_id=goal.id, attempt=attempt,
                                         reason=str(reason)[:120])
            verdict = run_judge(self.ctx, goal.text, goal.check, evidence,
                                parent_toolsets=kwargs.get("_parent_toolsets"),
                                timeout=self.settings.get("judge_timeout"),
                                attempts=self.settings.get("judge_attempts"),
                                attempt_timeout=self.settings.get("judge_attempt_timeout"),
                                backoff=self.settings.get("judge_backoff"), on_retry=_note_retry,
                                sleep=lambda seconds: _judge_sleep(seconds))
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
        if goal.rejects >= self.settings.get("reject_pause_after") > 0:
            _pause(self.store, state, [goal], verdict.reason,
                   DECISION_TEXT["reject"].format(gid=goal.id, n=goal.rejects, why=verdict.reason),
                   state.chat_key, "paused_rejects")
            reply += " The same reason came again; the goal is paused for an owner decision."
        elif goal.rejects >= self.settings.get("reject_replan_after") > 0:
            reply += (" The same reason came again: plan again. Name a different approach before "
                      "you continue, and fix each missing point.")
        else:
            reply += " Fix each missing point, then claim again."
        self.store.save(state)
        return reply


# --- Wait barriers (0.4.3): the loop does not push while Hermes work or the owner is awaited -------

def _background_wait(session_id: str) -> bool:
    """A background process of this session that reports at its end (notify_on_complete) or waits for
    a watch pattern is still running (tools.process_registry). Fail-safe: any error gives False."""
    return bool(_awaited_processes(session_id))


def _awaited_processes(session_id: str) -> list[dict]:
    """0.4.7 (z5): the running background processes of this session that the loop waits for, as
    {id, session_key}. Fail-safe: any error gives []."""
    out: list[dict] = []
    if not session_id:
        return out
    try:
        from tools.process_registry import process_registry
        with process_registry._lock:
            running = list(process_registry._running.values())
        for proc in running:
            if str(getattr(proc, "parent_session_id", "") or "") != str(session_id):
                continue
            if proc.exited or not (proc.notify_on_complete or proc.watch_patterns):
                continue
            if process_registry.is_session_waiting(proc.id):
                out.append({"id": str(proc.id), "session_key": str(getattr(proc, "session_key", "") or "")})
    except Exception:
        logger.debug("goal-loop background wait check failed", exc_info=True)
    return out


def _process_running(proc_id: str) -> bool:
    """0.4.7 (z5): the process is still in the live registry and has not exited."""
    try:
        from tools.process_registry import process_registry
        with process_registry._lock:
            proc = process_registry._running.get(proc_id)
        return proc is not None and not proc.exited
    except Exception:
        return True  # unknown: do not continue on a guess


def _note_background(state: "GoalState", session_id: str, platform: str) -> None:
    """0.4.7 (z5): remember the awaited processes (the first time of this wait keeps 'since')."""
    procs = _awaited_processes(session_id)
    if not procs:
        return
    ids = sorted(set(state.bg_wait.get("procs", [])) | {p["id"] for p in procs})
    keys = [p["session_key"] for p in procs if p["session_key"]]
    state.bg_wait = {"procs": ids, "since": state.bg_wait.get("since") or time.time(),
                     "session": str(session_id), "platform": str(platform or ""),
                     "session_key": keys[0] if keys else state.bg_wait.get("session_key", "")}


def _turn_marks() -> dict:
    """0.4.9 (z11): the running turns of this process as {session id or chat key: last activity}. The
    dict lives in a holder module, so a plugin reload keeps it (the watchdog thread calls the new module)."""
    holder = sys.modules.get("goal_loop_turns")
    if holder is None:
        import types
        holder = types.ModuleType("goal_loop_turns")
        holder.MARKS = {}
        sys.modules["goal_loop_turns"] = holder
    return holder.MARKS


def _mark_turn(*keys: Any, now: Optional[float] = None) -> None:
    marks = _turn_marks()
    t = time.time() if now is None else now
    for k in keys:
        if k:
            marks[str(k)] = t


def _touch_turn(*keys: Any) -> None:
    """Refresh the mark of a turn that runs (tool calls), so a long turn is not taken as stale."""
    marks = _turn_marks()
    t = time.time()
    for k in keys:
        if k and str(k) in marks:
            marks[str(k)] = t


def _end_turn(*keys: Any) -> None:
    marks = _turn_marks()
    for k in keys:
        if k:
            marks.pop(str(k), None)


def _turn_active(chat_key: str, session_id: str, now: float, settings: "Settings") -> bool:
    limit = float(settings.get("turn_active_max") or 0)
    marks = _turn_marks()
    for k in (chat_key, session_id):
        t = marks.get(str(k)) if k else None
        if t is not None and (not limit or now - float(t) < limit):
            return True
    return False


_WATCH_TEXT = ("[goal-loop resume {nonce}] Follow-up run of the goal loop (no owner message): the background "
               "process(es) {procs} that the loop waited for {why}. Read their results (process tool or "
               "their output files) and continue the work on the open goals of this chat.")


def background_watchdog_tick(store: GoalStore, settings: "Settings", now: Optional[float] = None,
                             running: Optional[Any] = None, inject: Optional[Any] = None,
                             schedule: Optional[Any] = None) -> list[str]:
    """0.4.7 (z5): continue each loop that waits for background processes when none of them runs any
    more (killed by a restart, or its completion turn was cut off) or when the wait is older than
    background_wait_max. A state of another boot stays held (restart rule, ledger only). Returns the chat
    keys that were continued."""
    now = time.time() if now is None else now
    running = running or _process_running
    limit = float(settings.get("background_wait_max") or 0)
    done = []
    try:
        names = sorted(os.listdir(store.base_dir))
    except OSError:
        return done
    for name in names:
        if not (name.startswith("chat_") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(store.base_dir, name), encoding="utf-8") as handle:
                state = GoalState.from_json(handle.read())
        except Exception:
            continue
        wait = state.bg_wait
        if not (state.go and wait and state.wait_reason == "waiting_background" and state.open_goals()):
            continue
        procs = list(wait.get("procs", []))
        alive = [p for p in procs if running(p)]
        age = now - float(wait.get("since") or now)
        if alive and not (limit and age >= limit):
            continue
        # 0.4.9 (z11): a turn of this chat runs (for example the completion-notice turn): that turn
        # decides; a mark older than turn_active_max does not count (a hard-stopped turn never ends).
        if _turn_active(state.chat_key, str(wait.get("session") or ""), now, settings):
            continue
        grace = float(settings.get("background_end_grace") or 0)
        if not alive and grace:
            seen = wait.get("ended_seen")
            if not seen:
                state.bg_wait = {**wait, "ended_seen": now}
                store.save(state)
                continue
            if now - float(seen) < grace:
                continue
        why = "ended without a completion turn" if not alive else f"still run after {int(age // 60)} min"
        if state.boot_id and state.boot_id != BOOT_ID:
            if not wait.get("held_noted"):
                state.bg_wait = {**wait, "held_noted": True}
                store.append_ledger(state.chat_key, "background_watchdog_held", procs=procs)
                store.save(state)
            continue
        import secrets
        nonce = secrets.token_hex(8)
        text = _WATCH_TEXT.format(nonce=nonce, procs=", ".join(procs) or "-", why=why)
        platform = str(wait.get("platform") or "")
        ok = False
        try:
            if platform == "api_server":
                ok = bool((schedule or _schedule_follow_up)(store, state, settings, str(wait.get("session") or "")))
            else:
                send = inject or getattr(getattr(sys.modules.get(__name__), "_CTX", None), "inject_message", None)
                key = str(wait.get("session_key") or "")
                ok = bool(callable(send) and key and send(text, session_key=key))
        except Exception:
            logger.debug("goal-loop background watchdog: continuation failed", exc_info=True)
        state.wait_reason = ""
        state.bg_wait = {}
        store.append_ledger(state.chat_key, "background_watchdog", procs=procs, alive=alive,
                            why="timeout" if alive else "ended", continued=ok, nonce=nonce)
        store.save(state)
        if ok:
            logger.info("goal-loop background watchdog: %s continued (%s %s)", state.chat_key,
                        ", ".join(procs), why)
            done.append(state.chat_key)
    return done


_WATCHDOG_NAME = "goal-loop-background-watchdog"


def _start_watchdog(settings: "Settings") -> None:
    """0.4.7 (z5): one daemon thread per process; after a plugin reload the thread calls the newest
    module (sys.modules), so a reload does not start a second thread."""
    if any(t.name == _WATCHDOG_NAME and t.is_alive() for t in threading.enumerate()):
        return
    name = __name__

    def loop() -> None:
        while True:
            mod = sys.modules.get(name)
            every = 60.0
            try:
                cfg = getattr(mod, "_SETTINGS", None) or settings
                every = float(cfg.get("background_watch_every") or 60)
                mod.background_watchdog_tick(mod.GoalStore(mod._state_dir()), cfg)
            except Exception:
                logger.debug("goal-loop background watchdog tick failed", exc_info=True)
            time.sleep(max(every, 10.0))

    threading.Thread(target=loop, name=_WATCHDOG_NAME, daemon=True).start()


def _delegation_wait(session_id: str) -> bool:
    """An asynchronous delegation of this session is still live (tools.async_delegation)."""
    if not session_id:
        return False
    try:
        from tools.async_delegation import _LIVE_STATES, _session_records
        return bool(_session_records(_LIVE_STATES, "", "", str(session_id)))
    except Exception:
        logger.debug("goal-loop delegation wait check failed", exc_info=True)
    return False


def _builtin_goal_active(session_id: str) -> bool:
    """Hermes' own /goal is active in this session: it drives the session, the plugin yields."""
    if not session_id:
        return False
    try:
        from hermes_cli.goals import load_goal
        goal = load_goal(str(session_id))
        return bool(goal is not None and getattr(goal, "status", "") == "active")
    except Exception:
        logger.debug("goal-loop builtin goal check failed", exc_info=True)
    return False


def _owner_wait(state: GoalState) -> bool:
    """All open goals wait for the owner: since the last owner message (or go) every open goal got a
    goal_block AND at least one goal_decision was recorded (a bare goal_block is only a try).
    The two calls may come in different rounds (live test 20:4x: decision in round 2, block in 6)."""
    open_goals = state.open_goals()
    return (bool(open_goals) and len(state.decisions) > state.decisions_at_nudge
            and all(g.block_tries >= 1 and g.block_seq >= 0 for g in open_goals))


def _wait_reason(state: GoalState, session_id: str) -> str:
    """Why the loop must not continue now, as a ledger event name; '' = continue."""
    if _builtin_goal_active(session_id):
        return "yield_builtin_goal"
    if _background_wait(session_id):
        return "waiting_background"
    if _delegation_wait(session_id):
        return "waiting_delegation"
    if state.stop_hold:
        return "waiting_stop_cause"
    if state.decision_wait:
        return "waiting_decision"
    if _owner_wait(state):
        return "waiting_owner"
    return ""


def _note_stop_cause(store: GoalStore, state: GoalState, text: Any) -> str:
    """0.4.5 (r32): read the stop-cause line of a final answer. A hold cause holds the loop until
    the next owner message; owner-question or blocker with a decision since the last owner answer
    also opens the decision wait (a later owner question does not lift it, only go or an answer)."""
    cause = _stop_cause(text)
    if cause not in HOLD_CAUSES or not state.open_goals():
        return cause
    if cause == "owner-question" and state.go:
        # 0.4.10 (R2): open goals can still be worked; the question goes to the deferred list.
        store.append_ledger(state.chat_key, "owner_question_not_held", open=[g.id for g in state.open_goals()])
        return cause
    if state.stop_hold != cause:
        state.stop_hold = cause
        store.append_ledger(state.chat_key, "stop_cause_hold", cause=cause,
                            open=[g.id for g in state.open_goals()])
    if (cause in ("owner-question", "blocker") and len(state.decisions) > state.decisions_at_nudge
            and not state.decision_wait):
        state.decision_wait = True
        store.append_ledger(state.chat_key, "decision_wait", decisions=len(state.decisions))
    return cause


def _lift_decision_wait(store: GoalStore, state: GoalState, why: str) -> None:
    if state.decision_wait or state.stop_hold:
        store.append_ledger(state.chat_key, "decision_wait_lifted", why=why)
    state.decision_wait = False
    state.stop_hold = ""
    state.decisions_at_nudge = len(state.decisions)


_WAIT_TEXT = {
    "yield_builtin_goal": "Hermes' own /goal is active",
    "waiting_background": "a background process is running",
    "waiting_delegation": "a delegated task is running",
    "waiting_owner": "all open goals wait for the owner's decision",
    "waiting_stop_cause": "the answer ended with a stop cause for the owner",
    "waiting_decision": "an open decision waits for the owner's answer with 'go'",
}


def _wait_note(store: GoalStore, state: GoalState, reason: str) -> None:
    """One ledger line per change of the wait reason (not one per turn)."""
    if reason != "waiting_background":
        state.bg_wait = {}
    if state.wait_reason != reason:
        state.wait_reason = reason
        store.append_ledger(state.chat_key, reason, open=[g.id for g in state.open_goals()])


# --- The pre_stop hook ------------------------------------------------------------------------

def _goal_context(state: GoalState, lead: str, goals: Optional[list[Goal]] = None) -> str:
    lines = [lead]
    for goal in (state.unfinished() if goals is None else goals):
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


def _pause_by_owner(store: GoalStore, state: GoalState, gate: str) -> None:
    """0.4.10 (R1): an owner message without 'go' takes the go back, cancels a pending follow-up run,
    clears the waits and opens the go gate (only a later 'go' message counts)."""
    state.go = False
    state.owner_paused = True
    _cancel_follow_up(store, state.chat_key)
    state.bg_wait = {}
    state.wait_reason = ""
    state.stop_hold = ""
    state.decision_wait = False
    state.decisions_at_nudge = len(state.decisions)
    store.append_ledger(state.chat_key, "paused_by_owner_message", open=[g.id for g in state.unfinished()])
    store.open_gate(gate)
    store.save(state)


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
        # 0.4.5 (r32): the stop-cause line of this answer.
        cause = _note_stop_cause(store, state, final_response)
        # 0.4.3: no push while Hermes work or the owner is awaited (the end of that work, or an owner
        # message, starts the next turn, and this hook runs again then).
        why = _wait_reason(state, session_id)
        if why == "waiting_owner" and not state.decision_wait:
            # 0.4.5 (r32): from now on an owner question does not lift this wait.
            state.decision_wait = True
            store.append_ledger(state.chat_key, "decision_wait", decisions=len(state.decisions))
        if why:
            _wait_note(store, state, why)
            if why == "waiting_background":
                _note_background(state, session_id, platform)  # 0.4.7 (z5)
            store.save(state)
            return None
        if state.wait_reason:
            state.wait_reason = ""
        state.bg_wait = {}
        state.nudge_seq += 1
        state.continuations = attempt + 1
        # 0.4.2: every continuation is visible in the ledger (also after a stop turn).
        store.append_ledger(state.chat_key, "continued", attempt=attempt,
                            open=[g.id for g in open_goals])
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
                   DECISION_TEXT["idle"].format(n=state.idle_nudges, gids=gids), key,
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
                   DECISION_TEXT["budget"].format(gid=goal.id, n=budget), key,
                   "paused_budget")
        open_goals = state.open_goals()
        store.save(state)
        if not open_goals:
            return None
        message = _goal_context(state, f"[goal-loop] {len(open_goals)} goal(s) are still open. "
                                       "Continue the work on these goals:", open_goals)
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
        free = _ids(open_goals)
        bound = [g for g in state.unfinished() if state.owner_bound(g)]
        if bound:
            message += (f"\nGoal(s) {_ids(bound)} wait for the owner (deferred list). Do not ask the owner "
                        f"now; continue with {free}.")
        if cause == "owner-question":
            message += (f"\nYour answer ended with the stop cause owner-question, but goal(s) {free} can "
                        f"still be worked: record the question with goal_decision and continue with {free}.")
        return {"action": "continue", "message": message}

    return _pre_stop


# --- The status display (transform_llm_output) -------------------------------------------------

def _display_block(state: GoalState) -> str:
    """Decide what this turn's answer shows and record it as shown (the caller saves the state).
    0.4.10 (R2): deferred items stay hidden while a goal can be worked; when none can, they are shown
    one time."""
    if state.list_changed:
        state.mark_all_shown()
        state.final_shown = state.all_done()
        return status_block(state, go_question=not state.go)
    if state.all_done():
        if not state.final_shown:
            state.final_shown = True
            state.mark_all_shown()
            return status_block(state)
        # goal-loop-quiet decision 4a: a decision recorded after the final list still shows once
    free = bool(_actionable(state))
    pending = [e for e in state.deferred if not e.get("shown")]
    held_numbers = {int(e.get("number") or 0) for e in pending if e.get("kind") == "decision"}
    held_goals = {str(e.get("goal_id") or "") for e in pending if e.get("kind") == "block"}
    owner_goals = [g for g in state.goals
                   if g.status in OWNER_STATUSES and not g.done and not g.block_shown
                   and not (free and g.id in held_goals)]
    start = min(max(state.decisions_shown, 0), len(state.decisions))
    decisions = [(index, text) for index, text in state.open_decisions()
                 if (index > start or (not free and index in held_numbers))
                 and not (free and index in held_numbers)]
    changes = [] if free else [str(e.get("text") or "") for e in pending if e.get("kind") == "change"]
    if not free:
        for entry in pending:
            entry["shown"] = True
    closed, state.decisions_just_closed = list(state.decisions_just_closed), []  # 0.4.8 (z10): show once
    if not owner_goals and not decisions and not closed and not changes:
        return ""
    for goal in owner_goals:
        goal.block_shown = True
    state.decisions_shown = len(state.decisions)
    return owner_block(owner_goals, decisions, closed=closed, changes=changes)


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


# Issue #1: the PluginContext, for inject_message (a follow-up turn of a local session).
_CTX: Any = None
_SETTINGS: Any = None
LOCAL_PLATFORMS = ("local", "cli", "tui", "desktop")


def _inject_follow_up(store: GoalStore, state: GoalState, settings: Settings,
                      platform: str, session_id: str = "") -> bool:
    """Issue #1: one follow-up turn of a local (CLI, TUI) session after the go while goals are
    open, through PluginContext.inject_message: Hermes commits it to the durable FIFO of the
    session the hook runs for (gateway/plugin_keyless_inject.py; a person's input goes first). It
    needs plugins.entries.goal-loop.allow_gateway_injection: true; False when refused."""
    limit = settings.get("resume_max")
    inject = getattr(_CTX, "inject_message", None)
    if (platform not in LOCAL_PLATFORMS or not settings.get("resume") or not limit
            or state.resumes >= limit or not callable(inject)):
        return False
    if _wait_reason(state, session_id):  # 0.4.3: handled in _transform (wait line), no follow-up
        return False
    # 0.4.2: a follow-up run without a tool call is idle; idle_pause_after idle runs in a row
    # pause the loop (only an owner message resets the count). A turn cut at the iteration
    # limit has tool calls, so issue #1 is not affected.
    if state.tool_calls_at_resume >= 0 and state.tool_calls == state.tool_calls_at_resume:
        state.idle_resumes += 1
    else:
        state.idle_resumes = 0
    state.tool_calls_at_resume = state.tool_calls
    idle_pause = settings.get("idle_pause_after")
    if idle_pause and state.idle_resumes >= idle_pause:
        open_goals = state.open_goals()
        gids = ", ".join(g.id for g in open_goals)
        _pause(store, state, open_goals, "follow-up runs in a row without a tool call",
               DECISION_TEXT["idle"].format(n=state.idle_resumes, gids=gids), state.chat_key,
               "paused_idle")
        store.save(state)
        return False
    from . import resume_run
    import secrets
    nonce = secrets.token_hex(8)
    try:
        ok = bool(inject(resume_run.TEXT.format(nonce=nonce)))
    except Exception:
        logger.debug("goal-loop follow-up inject failed", exc_info=True)
        ok = False
    if not ok:
        store.append_ledger(state.chat_key, "resume_inject_refused")
        return False
    state.resumes += 1
    store.append_ledger(state.chat_key, "resume_injected", n=state.resumes, nonce=nonce)
    return True


def make_transform_llm_output(store: GoalStore, settings: Optional[Settings] = None):
    """Append the goal display to the final answer of a turn, only when there is news."""
    settings = settings or Settings()

    def _transform(response_text: str = "", text: str = "", session_id: str = "",
                   platform: str = "", conversation_id: str = "", **_: Any) -> Optional[str]:
        if platform in EXCLUDED_PLATFORMS:
            return None
        _note_turn_end(store, session_id)
        _end_turn(session_id, _chat_key(conversation_id or session_id))  # 0.4.9 (z11)
        body = response_text if isinstance(response_text, str) and response_text else text
        if not isinstance(body, str):
            return None
        try:
            state = store.load(_chat_key(conversation_id or session_id))
        except Exception:
            return None
        if not state.goals:
            return None
        limit = settings.get("resume_max")
        waiting = state.open_goals() if state.go else []
        if waiting and settings.get("resume") and limit and state.resumes >= limit:
            _pause(store, state, waiting, f"{state.resumes} follow-up runs without completion",
                   DECISION_TEXT["resume_max"].format(n=state.resumes, gids=_ids(waiting)), state.chat_key,
                   "paused_resume_max")
        block = _display_block(state)
        # L15: the turn ends while goals are still open after the go. L13: a state that a
        # different process drove is held after a restart, so it gets no follow-up run either.
        held = bool(state.boot_id) and state.boot_id != BOOT_ID
        open_goals = state.open_goals() if state.go and not held else []
        if open_goals:
            _note_stop_cause(store, state, body)  # 0.4.5 (r32): also when pre_stop did not run
        why = _wait_reason(state, session_id) if open_goals else ""
        if why == "waiting_owner" and not state.decision_wait:
            state.decision_wait = True  # 0.4.5 (r32)
            store.append_ledger(state.chat_key, "decision_wait", decisions=len(state.decisions))
        gids, n = _ids(open_goals), len(open_goals)
        note = ""
        if why:
            _wait_note(store, state, why)
            if why == "waiting_background":
                _note_background(state, session_id, platform)  # 0.4.7 (z5)
            note = LOOP_NOTES["wait"].format(why=_WAIT_TEXT[why], n=n, gids=gids)
        elif open_goals and platform == "api_server" and _schedule_follow_up(store, state, settings,
                                                                            session_id):
            note = LOOP_NOTES["resume"].format(n=n, gids=gids, delay=settings.get("resume_delay"))
        elif open_goals and _inject_follow_up(store, state, settings, platform, session_id):
            # Issue #1: a local chat (also after a turn cut at the iteration limit) continues.
            note = LOOP_NOTES["inject"].format(n=n, gids=gids)
        elif open_goals and settings.get("paused_hint"):
            note = LOOP_NOTES["paused"].format(n=n, gids=gids)
        # 0.4.10 (R3): the loop state goes to the model as context of the next turn, not into this answer.
        changed = bool(block or open_goals) or note != state.loop_note
        state.loop_note = note
        if changed:
            try:
                store.save(state)
            except Exception:
                logger.debug("goal display state save failed", exc_info=True)
        if not block:
            return None
        # One goal block per answer: the model imitates the stored plugin blocks of earlier turns.
        body = strip_model_blocks(body, go_question=_GO_LINE.search(block) is not None)
        if not body.strip():
            body = LABELS["no_report"]
        return body + "\n" + block

    return _transform


# Lines of a goal block that the model wrote itself (labels of state.LABELS).
_TITLE_LINE = re.compile(r"^\s*(%s)\s*$" % "|".join(re.escape(LABELS[key]) for key in ("title", "title_owner")))
_DECISIONS_LINE = re.compile(r"^\s*(%s)\s*$" % re.escape(LABELS["decisions"]))
_GOAL_LINE = re.compile(r"^\s*- \[[ xX]\] \S+:")
_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s")
_GO_LINE = re.compile(r"^\s*(%s|Reply \W?go\W? to start\.?)\s*$" % re.escape(LABELS["go_question"]),
                      re.IGNORECASE | re.MULTILINE)


def strip_model_blocks(body: str, go_question: bool = False) -> str:
    """Remove goal blocks that the model wrote into its own text: a title line of a plugin label,
    then goal lines, a decisions heading with numbered lines, go lines and blank lines; the ``---``
    line before the title goes too. With ``go_question`` (the plugin block asks for the go), also
    remove go lines elsewhere, so that the go question is the last line of the answer, one time."""
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _TITLE_LINE.match(lines[i]):
            while out and not out[-1].strip():
                out.pop()
            if out and out[-1].strip() == "---":
                out.pop()
            i += 1
            in_decisions = False
            while i < len(lines):
                line = lines[i]
                if _DECISIONS_LINE.match(line):
                    in_decisions = True
                elif not (not line.strip() or _GOAL_LINE.match(line) or _GO_LINE.fullmatch(line)
                          or (in_decisions and _NUMBERED_LINE.match(line))):
                    break
                i += 1
            continue
        if go_question and _GO_LINE.fullmatch(lines[i]):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip()


# --- The go gate, the owner pause and the goal context (pre_llm_call) -------------------------

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
    if text.startswith(_TASK_LIST_RESTORE_PREFIX):
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


def _log_resume_turn(store: GoalStore, key: str, session_id: Any, text: str) -> None:
    """0.4.7 (z6): one INFO line in agent.log at the start of a resume turn, with the nonce of the turn
    and the cron job id of its request (follow-up run: request of the chat; restart resume: request
    'restart:<session>'). The Hermes log line of the turn shows only the first characters of the text."""
    try:
        from . import resume
        nonce = resume.nonce_of(text)
        sid = str(session_id or "")
        for chat in (key, "restart:" + sid):
            request = resume.load_request(resume.request_path(store.base_dir, chat))
            if request and request.get("nonce") == nonce:
                logger.info("goal-loop resume turn: nonce=%s job=%s reason=%s session=%s", nonce,
                            request.get("job_id") or "-", request.get("reason") or "follow-up", sid)
                return
        logger.info("goal-loop resume turn: nonce=%s job=- reason=inject session=%s", nonce, sid)
    except Exception:
        logger.debug("goal-loop resume log line failed", exc_info=True)


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


def _owui_key(session_id: str) -> str:
    """The Open WebUI key ``owui:<user>:<chat>`` that Hermes bound to this API session (state_meta
    ``gateway.api.conversation.v1.<key>`` = session id, gateway/session_api.py), or ''."""
    import sqlite3
    path = os.path.join(os.path.dirname(_state_dir()), "state.db")
    if not session_id or not os.path.exists(path):
        return ""
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            row = db.execute("SELECT key FROM state_meta WHERE key LIKE ? AND value = ? LIMIT 1",
                             ("gateway.api.conversation.v1.owui:%", session_id)).fetchone()
        finally:
            db.close()
    except Exception:
        return ""
    return row[0][len("gateway.api.conversation.v1."):] if row else ""


def _note_turn_start(store: GoalStore, session_id: Any, platform: str, conversation_id: Any,
                     model: str, settings: Settings) -> None:
    """r4: a turn of this session is running (restart_resume.py)."""
    if not settings.get("restart_resume"):
        return
    try:
        from . import restart_resume
        sid = str(session_id or "")
        owui = _owui_key(sid) if platform == "api_server" else ""
        restart_resume.note_start(store.base_dir, sid, platform, BOOT_ID,
                                  chat_key=_gate_for(store, sid, conversation_id),
                                  owui_key=owui, model=model)
    except Exception:
        logger.debug("goal-loop running note not written", exc_info=True)


def _note_turn_end(store: GoalStore, session_id: Any) -> None:
    try:
        from . import restart_resume
        restart_resume.note_end(store.base_dir, str(session_id or ""))
    except Exception:
        logger.debug("goal-loop running note not removed", exc_info=True)


def resume_after_restart(store: GoalStore, settings: Settings) -> list:
    """r4: one continuation turn for each turn that the last restart cut off."""
    if not settings.get("restart_resume"):
        return []
    from . import restart_resume
    return restart_resume.scan_and_schedule(
        store.base_dir, BOOT_ID, store.append_ledger,
        delay=str(settings.get("restart_resume_delay") or "2m"))


_GOALS_LEAD = ("[goal-loop] Goals of this chat at {time} (owner-approved; a newer goal list wins; "
               "do not change them without the owner):")
LOOP_HINT = ("[goal-loop] At the end of your answer, write one short line in the owner's language: the loop "
             "goes on by itself while goals are open, and any owner message without 'go' pauses it.")


def _restart_note(store: GoalStore, state: GoalState) -> str:
    """0.4.7 (z4): the first turn after a restart that finds the loop held gets the restart note."""
    held = bool(state.boot_id) and state.boot_id != BOOT_ID
    if not (held and state.go and state.unfinished() and state.restart_note_boot != BOOT_ID):
        return ""
    state.restart_note_boot = BOOT_ID
    store.append_ledger(state.chat_key, "restart_hold_note", open=[g.id for g in state.unfinished()])
    store.save(state)
    return "\n\n" + RESTART_NOTE.format(n=len(state.unfinished()), gids=_ids(state.unfinished()))


def _goals_context(state: GoalState) -> str:
    return _goal_context(state, _GOALS_LEAD.format(time=time.strftime("%H:%M")))


def make_pre_llm_call(store: GoalStore, settings: Optional[Settings] = None):
    """For each owner message: count a 'go' message for the go gate and each message for the change
    gate, mark it for this process (L13), pause the loop on a message without 'go' (R1), and return
    the goal context (L4)."""
    settings = settings or Settings()

    def _pre_llm_call(session_id: str = "", platform: str = "", user_message: Any = None,
                      conversation_history: Any = None, model: str = "", **_: Any) -> Optional[str]:
        if platform in EXCLUDED_PLATFORMS:
            return None
        try:
            text = _message_text(user_message).strip()
            _mark_turn(session_id, _chat_key(_.get("conversation_id") or session_id))  # 0.4.9 (z11)
            _note_turn_start(store, session_id, platform, _.get("conversation_id"), model, settings)
            if _is_follow_up(store, session_id, _.get("conversation_id"), text):
                # L15: a follow-up run of the plugin is never an owner message (no go, no
                # change approval, no restart re-arm); it gets the goal context.
                key = _gate_for(store, session_id, _.get("conversation_id"))
                _log_resume_turn(store, key, session_id, text)  # 0.4.7 (z6)
                _take_request(store, key, text)
                if store.exists(key):
                    state = store.load(key)
                    if state.idle_nudges or state.tool_calls_at_nudge >= 0:
                        state.idle_nudges = 0
                        state.tool_calls_at_nudge = -1
                        store.save(state)
                    if state.go and state.unfinished():
                        context = _goals_context(state) + "\n\n" + LOOP_HINT
                        if state.loop_note:
                            context += "\n" + _LOOP_NOTE_LEAD + state.loop_note
                        return context + _restart_note(store, state)
                return None
            if not _is_owner_message(user_message, conversation_history):
                return None
            gate = _gate_for(store, session_id, _.get("conversation_id"))
            vocabulary = _vocabulary(store.base_dir)
            go = _has_go(text, vocabulary[0])
            if go:
                store.bump_gate(gate)  # R1 c: only a 'go' message counts for goal_go
            store.bump_gate(gate, "change")
            store.open_gate(gate, f"boot{BOOT_ID[:12]}")
            _note_model(store, gate, model)
            if not store.exists(gate):
                return None
            state = store.load(gate)
            # Issue #1 and 0.4.2: an owner message starts a new count of idle turns and follow-up runs;
            # 0.4.3: it lifts the block tries and the waits (the owner answered).
            state.idle_nudges = 0
            state.tool_calls_at_nudge = -1
            state.idle_resumes = 0
            state.tool_calls_at_resume = -1
            for g in state.goals:
                g.block_seq = -1
            state.wait_reason = ""
            state.stop_hold = ""
            state.bg_wait = {}  # 0.4.7 (z5): the owner answered; the watchdog has nothing to do
            _close_decisions(store, state, text, vocabulary)  # 0.4.7 (z3)
            if go:
                _lift_decision_wait(store, state, "owner_go")
                if state.owner_paused:
                    state.owner_paused = False
                    if state.unfinished():
                        state.go = True
                        store.append_ledger(state.chat_key, "resumed_by_owner_go",
                                            open=[g.id for g in state.unfinished()])
                store.save(state)
                if state.go and state.unfinished():
                    return _goals_context(state) + "\n\n" + LOOP_HINT
                return None
            if state.go and state.unfinished():
                _pause_by_owner(store, state, gate)  # 0.4.10 (R1)
                return (_goals_context(state) + "\n\n" + PAUSE_NOTE.format(
                    n=len(state.unfinished()), gids=_ids(state.unfinished())) + " " + EVIDENCE_RULE)
            if not state.decision_wait:
                state.decisions_at_nudge = len(state.decisions)
            store.save(state)
        except Exception:
            logger.debug("goal-loop pre_llm_call failed", exc_info=True)
        return None

    return _pre_llm_call


def make_pre_api_request(store: GoalStore, settings: Optional[Settings] = None):
    """0.4.4: count an owner message that interrupted a running turn. Hermes applies it as a user row
    with the api_content prefix _REDIRECT_PREFIX and does not start a new turn, so pre_llm_call does
    not run for it. Each such row counts one time (go gate, change gate, owner pause, wait lift)."""
    settings = settings or Settings()
    inner = make_pre_llm_call(store, settings)

    def _pre_api_request(session_id: str = "", platform: str = "", conversation_history: Any = None,
                         model: str = "", **_: Any) -> None:
        if platform in EXCLUDED_PLATFORMS or not session_id or not isinstance(conversation_history, list):
            return None
        try:
            index, row = next(((i, m) for i, m in reversed(list(enumerate(conversation_history)))
                               if isinstance(m, dict) and m.get("role") == "user"), (-1, None))
            if row is None or not str(row.get("api_content") or "").startswith(_REDIRECT_PREFIX):
                return None
            text = _message_text(row.get("content")).strip()
            if not text:
                return None
            mark = hashlib.sha256(f"{index}\n{text}".encode("utf-8")).hexdigest()[:16]
            seen = (store._read_side("redir", str(session_id)) or {}).get("seen") or []
            if mark in seen:
                return None
            store._write_side("redir", str(session_id), {"seen": (seen + [mark])[-50:]})
            store.append_ledger(_gate_for(store, session_id, _.get("conversation_id")),
                                "owner_redirect_counted")
            inner(session_id=session_id, platform=platform, user_message=text,
                  conversation_history=conversation_history[:index + 1], model=model,
                  conversation_id=_.get("conversation_id"))
        except Exception:
            logger.debug("goal-loop pre_api_request failed", exc_info=True)
        return None

    return _pre_api_request


# --- Tool counting (post_tool_call) and the judge guard (pre_tool_call) -----------------------

def make_post_tool_call(store: GoalStore):
    """Count tool calls per chat for the idle detection (L2)."""

    def _post_tool_call(tool_name: str = "", session_id: str = "", conversation_id: str = "",
                        **_: Any) -> None:
        if not tool_name or tool_name in _NO_PROGRESS_TOOLS:
            return None
        _touch_turn(session_id, _chat_key(conversation_id or session_id))  # 0.4.9 (z11)
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


def _parent_session(session_id: str) -> Optional[str]:
    """E1 A: parent_session_id of a session in state.db (read only), or None. A compression edge
    is no parent here: the conversation id of a compressed chat already stays the same."""
    import sqlite3
    if not session_id:
        return None
    path = os.path.join(os.path.dirname(_state_dir()), "state.db")
    if not os.path.exists(path):
        return None
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    try:
        row = db.execute("SELECT s.parent_session_id, p.end_reason FROM sessions s "
                         "LEFT JOIN sessions p ON p.id = s.parent_session_id WHERE s.id = ?",
                         (session_id,)).fetchone()
    finally:
        db.close()
    if not row or not row[0] or row[1] == "compression":
        return None
    return str(row[0])


def pause_stale_states(store: GoalStore, settings: Settings, now: Optional[float] = None) -> list:
    """0.4.3 (decision 6B): a state with a go and unfinished goals (also blocked ones) whose file has not changed for
    ``stale_days`` days is paused (go False, open goals paused, ledger ``paused_stale``, one decision).
    Nothing is deleted; a new go of the owner continues it. Returns the paused chat keys."""
    days = settings.get("stale_days")
    if not days:
        return []
    now = time.time() if now is None else now
    limit = days * 86400
    paused = []
    for name in sorted(os.listdir(store.base_dir)):
        if not (name.startswith("chat_") and name.endswith(".json")):
            continue
        path = os.path.join(store.base_dir, name)
        try:
            if now - os.path.getmtime(path) < limit:
                continue
            with open(path, encoding="utf-8") as handle:
                state = GoalState.from_json(handle.read())
        except Exception:
            continue
        # every unfinished goal counts, also a blocked or paused one: the state still has a go
        open_goals = [g for g in state.unfinished() if g.status != "cancelled"]
        if not state.go or not open_goals:
            continue
        gids = ", ".join(g.id for g in open_goals)
        age = int((now - os.path.getmtime(path)) // 86400)
        store.append_ledger(state.chat_key, "paused_stale", goal_ids=[g.id for g in open_goals],
                            days=age)
        for goal in open_goals:
            goal.status = "paused"
            goal.note = f"no activity for {age} days"
            goal.block_shown = False
        state.decisions.append(DECISION_TEXT["stale"].format(days=age, gids=gids))
        state.go = False
        store.save(state)
        store.open_gate(state.chat_key)  # the next owner message counts for a new go
        paused.append(state.chat_key)
    return paused


def register(ctx) -> None:
    """Register the goal tools, the hooks, the judge task and the usage hint."""
    global _CTX, _SETTINGS
    _CTX = ctx
    store = GoalStore(_state_dir(), parent_of=_parent_session)
    settings = Settings(ctx)
    _SETTINGS = settings
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
         ("Record the owner's go (also to resume paused goals). Call it only after the owner wrote "
          "'go' in a new message; otherwise it is refused.")),
        ("goal_status", _EMPTY_SCHEMA, tools.goal_status, "📋", GOAL_STATUS_DESCRIPTION),
        ("goal_claim", _GOAL_CLAIM_SCHEMA, tools.goal_claim, "✅",
         ("Submit a structured completion claim for one goal; a fresh-context judge verifies the "
          'real state. goal_id is top-level; claim needs a summary and checks you ran, e.g. '
          'claim={"summary": "wrote the file", "checks": [{"command": "grep -qx READY f", '
          '"result": "passed", "exitCode": 0}]}.')),
        ("goal_progress", _GOAL_PROGRESS_SCHEMA, tools.goal_progress, "📍",
         "Record the last step, its evidence and the next step of a goal."),
        ("goal_decision", _GOAL_DECISION_SCHEMA, tools.goal_decision, "🧭",
         ("Record an owner decision (options + recommendation) with the evidence of the live state. "
          "Pass goal_id for the goal that waits for it; while other goals can be worked, it goes to the "
          "deferred list and the loop continues.")),
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
    ctx.register_hook("pre_api_request", make_pre_api_request(store, settings))
    ctx.register_hook("post_tool_call", make_post_tool_call(store))
    ctx.register_hook("pre_tool_call", make_pre_tool_call())
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section):
        try:
            register_section(PROMPT_SECTION_ID, prompt_section)
        except Exception:
            logger.warning("goal-loop usage hint not registered", exc_info=True)
    try:
        paused = pause_stale_states(store, settings)
        if paused:
            logger.info("goal-loop: %d stale state(s) paused: %s", len(paused), ", ".join(paused))
    except Exception:
        logger.warning("goal-loop stale pause failed", exc_info=True)
    try:
        jobs = resume_after_restart(store, settings)
        if jobs:
            logger.info("goal-loop: %d restart resume(s) scheduled: %s", len(jobs), ", ".join(jobs))
    except Exception:
        logger.warning("goal-loop restart resume scan failed", exc_info=True)
    try:
        _start_watchdog(settings)  # 0.4.7 (z5)
    except Exception:
        logger.warning("goal-loop background watchdog not started", exc_info=True)
    logger.info("goal-loop plugin registered: 7 tools, 6 hooks, judge task, usage hint, background watchdog")
