"""Visible follow-up runs (v0.3, owner decision 5): when a run ends after the go while goals are
still open, schedule one more run of the same Hermes session, without an owner message.

Path (all parts exist in the homelab; see README "Follow-up runs"):

1. ``schedule_resume`` (called by ``transform_llm_output`` at the end of the turn, while the Open
   WebUI turn is still open) writes a request file ``resume_<chat>.json`` (session id, nonce,
   model, time), a small wrapper script ``$HERMES_HOME/scripts/goal-loop-resume-<hash>.py`` and a
   one-shot ``no_agent`` cron job (``cron.jobs.create_job``) that runs the wrapper.
2. The cron delivery adapter of the homelab binds a job to the Open WebUI chat that had a turn
   open when the job was made (a job made inside a follow-up run inherits the chat of its parent
   job), and appends the run output of the job to that chat.
3. The wrapper runs ``resume_run.py``: it reads the request and the goal state, and when the go
   still stands and goals are open, it posts one continuation turn to the local API server with
   ``X-Hermes-Session-Id`` (the pattern of gateway/wake.py ``_self_post_chat_completion``) and
   prints the answer. The answer is the run output that the chat shows.

A follow-up turn starts with ``RESUME_PREFIX`` and its nonce; ``pre_llm_call`` never counts it as
an owner message, so it never gives a go. Limits: ``budget_per_goal``, the idle pause, the stop
word, ``resume_max`` follow-up runs per go, and one pending request per chat.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

RESUME_PREFIX = "[goal-loop resume "
# Test seam: a callable used instead of cron.jobs.create_job (tests set it, so no test makes a
# real cron job of the Hermes that runs the tests).
CREATE_JOB: Optional[Callable[..., dict]] = None
HERMES_HOME: Optional[str] = None  # test seam: the folder that gets scripts/
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_run.py")
WRAPPER = '''"""goal-loop follow-up run of one chat (written by the goal-loop plugin)."""
import runpy
import sys

sys.argv = ["goal-loop-resume", {request!r}, {state!r}]
runpy.run_path({runner!r}, run_name="__main__")
'''


def request_path(state_dir: str, chat_key: str) -> str:
    return os.path.join(state_dir, "resume_" + _safe(chat_key) + ".json")


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


def _cron_create_job() -> Optional[Callable[..., dict]]:
    """``cron.jobs.create_job`` of the running Hermes, or None (no Hermes on the path: tests)."""
    try:
        from cron.jobs import create_job
    except Exception:
        return None
    return create_job


def is_resume_text(text: str) -> bool:
    return text.lstrip().startswith(RESUME_PREFIX)


def nonce_of(text: str) -> str:
    rest = text.lstrip()[len(RESUME_PREFIX):]
    return rest.split("]", 1)[0].strip() if "]" in rest else ""


def load_request(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)
    os.replace(tmp, path)


def mark(path: str, status: str) -> None:
    data = load_request(path)
    if data is not None:
        data["status"] = status
        data["updated"] = time.time()
        _write(path, data)


BIND_DIR = "goal-loop-bind"


def write_bind_hint(home: str, script: str, owui_key: str) -> Optional[str]:
    """Write <home>/cron/goal-loop-bind/<script stem>.json with the Open WebUI key; returns the path or None."""
    folder = os.path.join(home, "cron", BIND_DIR)
    path = os.path.join(folder, os.path.splitext(script)[0] + ".json")
    try:
        os.makedirs(folder, exist_ok=True)
        _write(path, {"owui_key": owui_key, "written": time.time()})
    except OSError:
        logger.warning("goal-loop: bind hint not written", exc_info=True)
        return None
    return path


def schedule_resume(state_dir: str, chat_key: str, session_id: str, model: str, delay: str,
                    stale_after: float = 7200.0, create_job: Optional[Callable[..., dict]] = None,
                    hermes_home: Optional[str] = None, now: Optional[float] = None,
                    extra: Optional[dict] = None) -> Optional[str]:
    """Make one follow-up run for the chat; returns the cron job id, or None when a request is
    still pending (not older than ``stale_after``) or the job cannot be made."""
    create_job = create_job or CREATE_JOB or _cron_create_job()
    if create_job is None:
        return None
    now = time.time() if now is None else now
    path = request_path(state_dir, chat_key)
    old = load_request(path)
    if old and old.get("status") in ("scheduled", "running") and now - float(old.get("created", 0)) < stale_after:
        return None
    if not session_id:
        return None
    home = hermes_home or HERMES_HOME or _hermes_home()
    scripts = os.path.join(home, "scripts")
    os.makedirs(scripts, exist_ok=True)
    name = "goal-loop-resume-" + hashlib.sha256(chat_key.encode()).hexdigest()[:12] + ".py"
    wrapper = os.path.join(scripts, name)
    # the goal state of the chat (a restart resume names it in extra["state_key"])
    state_key = str((extra or {}).get("state_key") or chat_key)
    text = WRAPPER.format(request=path, state=os.path.join(state_dir, _safe(state_key) + ".json"),
                          runner=RUNNER)
    current = None
    if os.path.exists(wrapper):
        with open(wrapper, encoding="utf-8") as handle:
            current = handle.read()
    if current != text:
        with open(wrapper + ".tmp", "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(wrapper + ".tmp", wrapper)
    owui_key = str((extra or {}).get("owui_key") or "")
    if owui_key:
        # 0.4.5 (r34): the bind hint for the homelab delivery adapter. The resumed turn is not open when the
        # job is made, so the adapter cannot bind the job by its session map; it reads this file instead
        # (<HERMES_HOME>/cron/goal-loop-bind/<script stem>.json). Written before the job, so it is there at
        # the adapter's first sight of the job.
        write_bind_hint(home, name, owui_key)
    request = {"chat_key": chat_key, "session_id": session_id, "model": model or "",
               "nonce": secrets.token_hex(8), "created": now, "status": "scheduled", **(extra or {})}
    _write(path, request)
    try:
        name_ = "goal-loop restart resume" if (extra or {}).get("reason") == "restart" else "goal-loop follow-up run"
        job = create_job(prompt=None, schedule=f"in {delay}", name=name_,
                         repeat=1, deliver="local", script=name, no_agent=True)
    except Exception:
        logger.warning("goal-loop follow-up run: cron job not made", exc_info=True)
        mark(path, "failed")
        return None
    job_id = str((job or {}).get("id") or "")
    request["job_id"] = job_id
    _write(path, request)
    return job_id or None
