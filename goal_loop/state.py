"""Durable goal state and append-only ledger for one chat session.

The goal context must survive a context compression: Hermes rotates the session id on compaction
(agent/conversation_compression.py), so state is keyed by the conversation id (the compression-
lineage root, stable across rotation) with session id as fallback, and kept in a JSON file, not in
chat memory. The ledger is append-only and is written before the state file, so a terminal event
(a confirmed or blocked goal) is never lost if the process stops between the two writes (the rule
from the OpenCode plugin). Each goal keeps a boolean ``done`` that only the judge may set to True;
the worker changes progress notes, never the verdict.

Display bookkeeping (what the chat has already shown) lives in the same state file:
``list_changed`` (a ``goal_set`` in this turn: show the full list once), ``final_shown`` (the
all-confirmed list was shown once), ``decisions_shown`` (how many decisions the chat has shown) and
``Goal.block_shown`` (the chat has shown this blocked or paused goal). A state file written before
these fields existed loads without error; its missing fields default to "already shown", so an
upgrade never floods a running chat with old items.

Loop bookkeeping (v0.3): ``Goal.nudges`` (continuations while the goal was open, for the budget),
``Goal.rejects`` / ``Goal.last_reject`` (consecutive similar judge rejections), ``Goal.block_tries``
(``goal_block`` attempts in different continuation rounds), ``Goal.progress`` (last step, evidence,
next step), ``GoalState.tool_calls`` / ``tool_calls_at_nudge`` / ``idle_nudges`` (idle detection),
``nudge_seq`` (a counter of continuation rounds) and ``boot_id`` (the process that last drove the
loop).

Small side files per **session id**: ``gate_<session>.json`` (the go gate: ``goal_set`` opens it with
a count of 0, the ``pre_llm_call`` hook adds 1 for each owner message, ``goal_go`` needs at least 1),
``change_<session>.json`` (the same for a changed goal list after the go), ``sess_<session>.json``
(the chat key of the session, for hooks that get only ``session_id``) and ``lang_<session>.json``
(owner language votes).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

STATE_VERSION = 3
GOAL_STATUS = ("open", "in_work", "waiting", "paused", "blocked", "done")
# Goals that need the owner before the loop works on them again.
OWNER_STATUSES = ("blocked", "paused")

LABELS = {
    "en": {
        "title": "Goal status:", "title_owner": "Goal status (needs the owner):",
        "decisions": "Open decisions:", "check": "check", "go_question": "Do you give the go?",
        "no_report": "(no final report from the model)", "progress": "progress",
        "status": {"open": "open", "in_work": "in work", "waiting": "waiting", "paused": "paused",
                   "blocked": "blocked", "done": "done"},
    },
    "de": {
        "title": "Zielstatus:", "title_owner": "Zielstatus (braucht dich):",
        "decisions": "Offene Entscheidungen:", "check": "Prüfung", "go_question": "Gibst du das Go?",
        "no_report": "(kein Abschlussbericht vom Modell)", "progress": "Fortschritt",
        "status": {"open": "offen", "in_work": "in Arbeit", "waiting": "wartet", "paused": "pausiert",
                   "blocked": "blockiert", "done": "erledigt"},
    },
}


def labels(language: str) -> dict:
    return LABELS.get(language or "en", LABELS["en"])


@dataclass
class Goal:
    """One goal with its single falsifiable check. ``done`` is judge-owned."""

    id: str
    text: str
    check: str
    status: str = "open"
    done: bool = False
    note: str = ""
    judge_reason: str = ""
    block_shown: bool = False
    nudges: int = 0
    rejects: int = 0
    last_reject: str = ""
    block_tries: int = 0
    block_seq: int = -1
    block_reason: str = ""
    progress: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        known = {f.name for f in fields(cls)}
        goal = cls(**{k: v for k, v in data.items() if k in known})
        if "block_shown" not in data:
            # Old state file: a goal that was already blocked was shown by the old status block.
            goal.block_shown = goal.status == "blocked"
        if not isinstance(goal.progress, dict):
            goal.progress = {}
        return goal


@dataclass
class GoalState:
    """All goals of one chat plus the loop and display bookkeeping."""

    chat_key: str
    version: int = STATE_VERSION
    goals: list[Goal] = field(default_factory=list)
    go: bool = False
    continuations: int = 0
    decisions: list[str] = field(default_factory=list)
    # display bookkeeping (see module docstring)
    list_changed: bool = False
    final_shown: bool = False
    decisions_shown: int = 0
    # loop bookkeeping (v0.3)
    tool_calls: int = 0
    tool_calls_at_nudge: int = -1
    idle_nudges: int = 0
    nudge_seq: int = 0
    boot_id: str = ""
    language: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GoalState":
        data = json.loads(text)
        goals = [Goal.from_dict(g) for g in data.get("goals", [])]
        decisions = list(data.get("decisions", []))
        all_done = bool(goals) and all(g.done for g in goals)
        return cls(
            chat_key=data["chat_key"], version=STATE_VERSION,
            goals=goals, go=bool(data.get("go", False)),
            continuations=int(data.get("continuations", 0)),
            decisions=decisions,
            list_changed=bool(data.get("list_changed", False)),
            # Old file: a list that is already all done was shown by the old status block.
            final_shown=bool(data.get("final_shown", all_done)),
            decisions_shown=int(data.get("decisions_shown", len(decisions))),
            tool_calls=int(data.get("tool_calls", 0)),
            tool_calls_at_nudge=int(data.get("tool_calls_at_nudge", -1)),
            idle_nudges=int(data.get("idle_nudges", 0)),
            nudge_seq=int(data.get("nudge_seq", 0)),
            boot_id=str(data.get("boot_id", "")),
            language=str(data.get("language", "")),
        )

    def goal(self, goal_id: str) -> Optional[Goal]:
        return next((g for g in self.goals if g.id == goal_id), None)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def open_goals(self) -> list[Goal]:
        """Goals the loop must still work on: not confirmed, not waiting, not blocked or paused."""
        return [g for g in self.goals
                if not g.done and g.status not in OWNER_STATUSES and g.status != "waiting"]

    def unfinished(self) -> list[Goal]:
        return [g for g in self.goals if not g.done]

    def mark_all_shown(self) -> None:
        """The chat showed the full list: every owner item and every decision is now shown."""
        self.list_changed = False
        self.decisions_shown = len(self.decisions)
        for goal in self.goals:
            if goal.status in OWNER_STATUSES:
                goal.block_shown = True


class GoalStore:
    """File-backed store for :class:`GoalState` plus an append-only ledger, one pair per chat.

    ``base_dir`` holds ``<chat>.json`` (state) and ``<chat>.ledger`` (one JSON event per line).
    A terminal event is appended to the ledger BEFORE the state file is rewritten.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    @staticmethod
    def _safe(chat_key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in chat_key) or "default"

    def _state_path(self, chat_key: str) -> str:
        return os.path.join(self.base_dir, f"{self._safe(chat_key)}.json")

    def _ledger_path(self, chat_key: str) -> str:
        return os.path.join(self.base_dir, f"{self._safe(chat_key)}.ledger")

    def _side_path(self, kind: str, session_id: str) -> str:
        return os.path.join(self.base_dir, f"{kind}_{self._safe(session_id)}.json")

    def load(self, chat_key: str) -> GoalState:
        path = self._state_path(chat_key)
        if not os.path.exists(path):
            return GoalState(chat_key=chat_key)
        with open(path, encoding="utf-8") as handle:
            return GoalState.from_json(handle.read())

    def exists(self, chat_key: str) -> bool:
        return os.path.exists(self._state_path(chat_key))

    @staticmethod
    def _write_atomic(path: str, text: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)

    def save(self, state: GoalState) -> None:
        """Atomic rewrite of the state file (write temp, then replace)."""
        self._write_atomic(self._state_path(state.chat_key), state.to_json())

    def append_ledger(self, chat_key: str, event: str, **fields: Any) -> None:
        """Append one terminal/loop event as a JSON line. Call this BEFORE :meth:`save`."""
        record = {"ts": round(time.time(), 3), "event": event, **fields}
        with open(self._ledger_path(chat_key), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_ledger(self, chat_key: str) -> list[dict[str, Any]]:
        path = self._ledger_path(chat_key)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    # --- small side files per session id ---------------------------------------------------------

    def _read_side(self, kind: str, session_id: str) -> Optional[dict]:
        path = self._side_path(kind, session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _write_side(self, kind: str, session_id: str, data: dict) -> None:
        self._write_atomic(self._side_path(kind, session_id), json.dumps(data, ensure_ascii=False))

    # go gate (kind "gate") and change gate (kind "change")
    def open_gate(self, session_id: str, kind: str = "gate") -> None:
        """No owner message has come after this point yet."""
        self._write_side(kind, session_id, {"session_id": session_id, "owner_messages": 0})

    def gate_count(self, session_id: str, kind: str = "gate") -> Optional[int]:
        """Owner messages since the gate opened, or None without a gate."""
        data = self._read_side(kind, session_id)
        if data is None:
            return None
        try:
            return int(data.get("owner_messages", 0))
        except (TypeError, ValueError):
            return None

    def bump_gate(self, session_id: str, kind: str = "gate") -> bool:
        """Count one owner message for an open gate. No gate: nothing is written."""
        count = self.gate_count(session_id, kind)
        if count is None:
            return False
        self._write_side(kind, session_id, {"session_id": session_id, "owner_messages": count + 1})
        return True

    def close_gate(self, session_id: str, kind: str = "gate") -> None:
        path = self._side_path(kind, session_id)
        if os.path.exists(path):
            os.remove(path)

    # session -> chat key (hooks that get only the session id)
    def remember_session(self, session_id: str, chat_key: str) -> None:
        if not session_id or not chat_key:
            return
        data = self._read_side("sess", session_id)
        if data and data.get("chat_key") == chat_key:
            return
        self._write_side("sess", session_id, {"chat_key": chat_key})

    def chat_for_session(self, session_id: str) -> Optional[str]:
        data = self._read_side("sess", session_id) if session_id else None
        key = data.get("chat_key") if data else None
        return key if isinstance(key, str) and key else None

    # owner language votes per session
    def add_language_vote(self, session_id: str, language: str) -> None:
        if not session_id or not language:
            return
        data = self._read_side("lang", session_id) or {}
        data[language] = int(data.get(language, 0)) + 1
        self._write_side("lang", session_id, data)

    def session_language(self, session_id: str) -> str:
        data = self._read_side("lang", session_id) if session_id else None
        if not data:
            return ""
        best = max(sorted(data), key=lambda k: int(data.get(k, 0)))
        return best if int(data.get(best, 0)) > 0 else ""


def _goal_line(goal: Goal, language: str = "en") -> str:
    lab = labels(language)
    mark = "x" if goal.done else " "
    reason = f" — {goal.judge_reason}" if goal.judge_reason and not goal.done else ""
    if goal.status in OWNER_STATUSES and goal.note:
        reason = f" — {goal.note}"
    status = lab["status"].get(goal.status, goal.status)
    return f"- [{mark}] {goal.id}: {goal.text} ({lab['check']}: {goal.check}) [{status}]{reason}"


def status_block(state: GoalState, language: str = "en", go_question: bool = False) -> str:
    """Render the full goal list: every goal with its check and status, then the decisions.
    ``go_question`` adds the go question as the last line (goal list before the go)."""
    if not state.goals:
        return ""
    lab = labels(language)
    lines = ["", "---", lab["title"]]
    for goal in state.goals:
        lines.append(_goal_line(goal, language))
    if state.decisions:
        lines.append("")
        lines.append(lab["decisions"])
        for index, decision in enumerate(state.decisions, start=1):
            lines.append(f"{index}. {decision}")
    if go_question:
        lines.append("")
        lines.append(lab["go_question"])
    return "\n".join(lines)


def owner_block(blocked: list[Goal], decisions: list[tuple[int, str]], language: str = "en") -> str:
    """Render only the items that need the owner: newly blocked or paused goals and new decisions."""
    if not blocked and not decisions:
        return ""
    lab = labels(language)
    lines = ["", "---", lab["title_owner"]]
    for goal in blocked:
        lines.append(_goal_line(goal, language))
    if decisions:
        if blocked:
            lines.append("")
        lines.append(lab["decisions"])
        for index, decision in decisions:
            lines.append(f"{index}. {decision}")
    return "\n".join(lines)
