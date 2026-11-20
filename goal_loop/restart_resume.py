"""Turns that a Hermes restart cut off get a continuation turn after the next start (owner order r4).

Hermes v0.23.2 resumes a restart-interrupted session only for push platforms (gateway/run_shutdown.py
``_mark_running_sessions_resume_pending`` covers ``GatewayRunner._running_agents``; an API server run is
not in that map, and its adapter cannot push: gateway/platforms/api_server.py ``send``). A local or
authority turn settles ``interrupted`` and is never replayed (gateway/session_settlement_recovery.py
``owner_stop_notice``). So Open WebUI chats and local chats lose a running turn on each restart.

This module closes that gap without a fork change:

1. ``note_start`` (pre_llm_call of an owner message or a follow-up turn) writes
   ``running_<session>.json``: session id, platform, the Open WebUI key ``owui:<user>:<chat>`` when
   known, the boot id and the time. ``note_end`` (transform_llm_output, the end of the turn) removes it.
2. ``scan_and_schedule`` runs once per process start (register): each ``running_*`` file of another
   boot whose writer process does not run any more (pid + start time, ``writer_alive``) is a turn that
   the restart cut off; the scanning process claims it with an atomic rename. For each one it writes a request and a one-shot cron job
   (the follow-up-run path of resume.py), with ``reason = "restart"``. The job posts one continuation
   turn to the same session: through hermes-models with ``X-Hermes-Session-Key`` for an Open WebUI
   chat (so the session map and the delivery bind the answer to that chat), else to the local API
   server. Then the file is removed, so a turn is resumed at most one time.

The continuation text starts with ``RESUME_PREFIX``; ``pre_llm_call`` never counts it as an owner
message, so it never gives a go. A file older than ``max_age`` seconds is dropped without a turn.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PREFIX = "running_"


def _proc_start(pid: int) -> str:
    """Start time of a process (field 22 of /proc/<pid>/stat), or '' when it does not run."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return ""


def _me() -> tuple[int, str]:
    pid = os.getpid()
    return pid, _proc_start(pid)


def writer_alive(note: dict) -> bool:
    """The process that wrote the note still runs (same pid and same start time). Several processes
    load the plugin (gateway, CLI, TUI), so another boot id alone does not mean a restart."""
    pid, start = note.get("pid"), note.get("pid_start")
    if not pid or not start:
        return False
    return _proc_start(int(pid)) == str(start)


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def path_for(state_dir: str, session_id: str) -> str:
    return os.path.join(state_dir, PREFIX + _safe(session_id) + ".json")


def note_start(state_dir: str, session_id: str, platform: str, boot_id: str, chat_key: str = "",
               owui_key: str = "", model: str = "", now: Optional[float] = None) -> None:
    if not session_id or platform not in ("api_server", "local", "cli", "tui"):
        return
    pid, start = _me()
    data = {"session_id": session_id, "platform": platform, "boot_id": boot_id, "chat_key": chat_key,
            "owui_key": owui_key, "model": model, "started": time.time() if now is None else now,
            "pid": pid, "pid_start": start}
    path = path_for(state_dir, session_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)
    os.replace(tmp, path)


def note_end(state_dir: str, session_id: str) -> None:
    if not session_id:
        return
    try:
        os.remove(path_for(state_dir, session_id))
    except FileNotFoundError:
        pass


def interrupted(state_dir: str, boot_id: str) -> list[dict]:
    """The running notes that an earlier process wrote (turns that a restart cut off)."""
    out = []
    try:
        names = sorted(os.listdir(state_dir))
    except OSError:
        return out
    for name in names:
        if not (name.startswith(PREFIX) and name.endswith(".json")):
            continue
        path = os.path.join(state_dir, name)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if (isinstance(data, dict) and data.get("boot_id") != boot_id and data.get("session_id")
                and not writer_alive(data)):
            data["_path"] = path
            out.append(data)
    return out


def scan_and_schedule(state_dir: str, boot_id: str, ledger: Callable[..., Any],
                      create_job: Optional[Callable[..., dict]] = None,
                      hermes_home: Optional[str] = None, delay: str = "2m",
                      max_age: float = 6 * 3600, now: Optional[float] = None) -> list[str]:
    """Schedule one continuation turn per turn that a restart cut off; returns the job ids."""
    from . import resume
    now = time.time() if now is None else now
    jobs = []
    for note in interrupted(state_dir, boot_id):
        path = note.pop("_path")
        # Claim the note atomically: of several processes that start at the same time, one wins.
        claimed = f"{path}.claimed-{os.getpid()}"
        try:
            os.rename(path, claimed)
        except OSError:
            continue
        path = claimed
        sid = str(note["session_id"])
        chat_key = str(note.get("chat_key") or ("chat:" + sid))
        try:
            if now - float(note.get("started", 0)) > max_age:
                ledger(chat_key, "restart_resume_dropped", session=sid, why="older than max_age")
                continue
            job = resume.schedule_resume(
                state_dir, "restart:" + sid, sid, str(note.get("model") or ""), delay,
                create_job=create_job, hermes_home=hermes_home, now=now,
                extra={"reason": "restart", "owui_key": str(note.get("owui_key") or ""),
                       "platform": str(note.get("platform") or ""), "state_key": chat_key,
                       "interrupted_at": note.get("started")})
            if job:
                jobs.append(job)
                ledger(chat_key, "restart_resume_scheduled", session=sid, job_id=job,
                       owui=bool(note.get("owui_key")))
            else:
                ledger(chat_key, "restart_resume_not_scheduled", session=sid)
        except Exception:
            logger.warning("goal-loop restart resume of %s failed", sid, exc_info=True)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return jobs
