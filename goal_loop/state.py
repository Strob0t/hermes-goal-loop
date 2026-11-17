"""Durable goal state and append-only ledger for one chat session.

The goal context must survive a context compression: Hermes changes the session id on compaction, so
state is keyed by chat and kept in a JSON file, not in chat memory. The ledger is append-only and is
written before the state file, so a terminal event (a confirmed or blocked goal) is never lost if the
process stops between the two writes (the rule from the OpenCode plugin). Each goal keeps a boolean
``done`` that only the judge may set to True; the worker changes progress notes, never the verdict.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

STATE_VERSION = 1
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


@dataclass
class GoalState:
    """All goals of one chat plus the loop bookkeeping."""

    chat_key: str
    version: int = STATE_VERSION
    goals: list[Goal] = field(default_factory=list)
    go: bool = False
    continuations: int = 0
    decisions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GoalState":
        data = json.loads(text)
        goals = [Goal(**g) for g in data.get("goals", [])]
        return cls(
            chat_key=data["chat_key"], version=int(data.get("version", STATE_VERSION)),
            goals=goals, go=bool(data.get("go", False)),
            continuations=int(data.get("continuations", 0)),
            decisions=list(data.get("decisions", [])),
        )

    def goal(self, goal_id: str) -> Optional[Goal]:
        return next((g for g in self.goals if g.id == goal_id), None)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def open_goals(self) -> list[Goal]:
        """Goals the loop must still work on: not judge-confirmed and not blocked."""
        return [g for g in self.goals if not g.done and g.status != "blocked"]


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

    def load(self, chat_key: str) -> GoalState:
        path = self._state_path(chat_key)
        if not os.path.exists(path):
            return GoalState(chat_key=chat_key)
        with open(path, encoding="utf-8") as handle:
            return GoalState.from_json(handle.read())

    def save(self, state: GoalState) -> None:
        """Atomic rewrite of the state file (write temp, then replace)."""
        path = self._state_path(state.chat_key)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(state.to_json())
        os.replace(tmp, path)

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


def status_block(state: GoalState) -> str:
    """Render the goal status block appended to each answer (goal 4: every report shows all goals)."""
    if not state.goals:
        return ""
    lines = ["", "---", "Goal status:"]
    for goal in state.goals:
        mark = "x" if goal.done else " "
        reason = f" — {goal.judge_reason}" if goal.judge_reason and not goal.done else ""
        lines.append(f"- [{mark}] {goal.id}: {goal.text} (check: {goal.check}) [{goal.status}]{reason}")
    if state.decisions:
        lines.append("")
        lines.append("Open decisions:")
        for index, decision in enumerate(state.decisions, start=1):
            lines.append(f"{index}. {decision}")
    return "\n".join(lines)
