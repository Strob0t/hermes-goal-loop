"""Durable goal state and append-only ledger for one chat session.

The goal context must survive a context compression: Hermes rotates the session id on compaction
(agent/conversation_compression.py), so state is keyed by the conversation id (the compression-
lineage root, stable across rotation) with session id as fallback, and kept in a JSON file, not in
chat memory. The ledger is append-only and is
written before the state file, so a terminal event (a confirmed or blocked goal) is never lost if the
process stops between the two writes (the rule from the OpenCode plugin). Each goal keeps a boolean
``done`` that only the judge may set to True; the worker changes progress notes, never the verdict.

Display bookkeeping (what the chat has already shown) lives in the same state file:
``list_changed`` (a ``goal_set`` in this turn: show the full list once), ``final_shown`` (the
all-confirmed list was shown once), ``decisions_shown`` (how many decisions the chat has shown) and
``Goal.block_shown`` (the chat has shown this blocked goal). A state file written before these
fields existed loads without error; its missing fields default to "already shown", so an upgrade
never floods a running chat with old items.

The go gate is a separate small file per **session id**: ``goal_set`` opens it with a count of 0,
the ``pre_llm_call`` hook adds 1 for each owner message of that session, and ``goal_go`` is
accepted only when the count is at least 1.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

STATE_VERSION = 2
GOAL_STATUS = ("open", "in_work", "blocked", "done")


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

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        known = {f.name for f in fields(cls)}
        goal = cls(**{k: v for k, v in data.items() if k in known})
        if "block_shown" not in data:
            # Old state file: a goal that was already blocked was shown by the old status block.
            goal.block_shown = goal.status == "blocked"
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
            chat_key=data["chat_key"], version=int(data.get("version", STATE_VERSION)),
            goals=goals, go=bool(data.get("go", False)),
            continuations=int(data.get("continuations", 0)),
            decisions=decisions,
            list_changed=bool(data.get("list_changed", False)),
            # Old file: a list that is already all done was shown by the old status block.
            final_shown=bool(data.get("final_shown", all_done)),
            decisions_shown=int(data.get("decisions_shown", len(decisions))),
        )

    def goal(self, goal_id: str) -> Optional[Goal]:
        return next((g for g in self.goals if g.id == goal_id), None)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def open_goals(self) -> list[Goal]:
        """Goals the loop must still work on: not judge-confirmed and not blocked."""
        return [g for g in self.goals if not g.done and g.status != "blocked"]

    def mark_all_shown(self) -> None:
        """The chat showed the full list: every blocked goal and every decision is now shown."""
        self.list_changed = False
        self.decisions_shown = len(self.decisions)
        for goal in self.goals:
            if goal.status == "blocked":
                goal.block_shown = True


class GoalStore:
    """File-backed store for :class:`GoalState` plus an append-only ledger, one pair per chat.

    ``base_dir`` holds ``<chat>.json`` (state) and ``<chat>.ledger`` (one JSON event per line).
    A terminal event is appended to the ledger BEFORE the state file is rewritten.
    ``gate_<session>.json`` holds the go gate of one session id.
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

    def _gate_path(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"gate_{self._safe(session_id)}.json")

    def load(self, chat_key: str) -> GoalState:
        path = self._state_path(chat_key)
        if not os.path.exists(path):
            return GoalState(chat_key=chat_key)
        with open(path, encoding="utf-8") as handle:
            return GoalState.from_json(handle.read())

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

    # --- go gate (per session id) ---------------------------------------------------------------

    def open_gate(self, session_id: str) -> None:
        """``goal_set`` ran in this session: no owner message has come after it yet."""
        self._write_atomic(self._gate_path(session_id),
                           json.dumps({"session_id": session_id, "owner_messages": 0}))

    def gate_count(self, session_id: str) -> Optional[int]:
        """Owner messages since the last ``goal_set`` of this session, or None without a gate."""
        path = self._gate_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return int(json.load(handle).get("owner_messages", 0))
        except (OSError, ValueError, AttributeError):
            return None

    def bump_gate(self, session_id: str) -> bool:
        """Count one owner message for an open gate. No gate: nothing is written."""
        count = self.gate_count(session_id)
        if count is None:
            return False
        self._write_atomic(self._gate_path(session_id),
                           json.dumps({"session_id": session_id, "owner_messages": count + 1}))
        return True

    def close_gate(self, session_id: str) -> None:
        path = self._gate_path(session_id)
        if os.path.exists(path):
            os.remove(path)


def _goal_line(goal: Goal) -> str:
    mark = "x" if goal.done else " "
    reason = f" — {goal.judge_reason}" if goal.judge_reason and not goal.done else ""
    if goal.status == "blocked" and goal.note:
        reason = f" — {goal.note}"
    return f"- [{mark}] {goal.id}: {goal.text} (check: {goal.check}) [{goal.status}]{reason}"


def status_block(state: GoalState) -> str:
    """Render the full goal list: every goal with its check and status, then the decisions."""
    if not state.goals:
        return ""
    lines = ["", "---", "Goal status:"]
    for goal in state.goals:
        lines.append(_goal_line(goal))
    if state.decisions:
        lines.append("")
        lines.append("Open decisions:")
        for index, decision in enumerate(state.decisions, start=1):
            lines.append(f"{index}. {decision}")
    return "\n".join(lines)


def owner_block(blocked: list[Goal], decisions: list[tuple[int, str]]) -> str:
    """Render only the items that need the owner: newly blocked goals and new decisions."""
    if not blocked and not decisions:
        return ""
    lines = ["", "---", "Goal status (needs the owner):"]
    for goal in blocked:
        lines.append(_goal_line(goal))
    if decisions:
        if blocked:
            lines.append("")
        lines.append("Open decisions:")
        for index, decision in decisions:
            lines.append(f"{index}. {decision}")
    return "\n".join(lines)
