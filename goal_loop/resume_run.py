"""goal-loop follow-up run: one continuation turn of a Hermes session (stdlib only).

Usage (from the wrapper that ``resume.schedule_resume`` writes):
    resume_run.py <request file> <state file>

Prints the answer of the turn (the cron run output that the chat shows), or nothing (a silent run)
when the request is not valid any more: the go was taken back, no goal is open, Hermes started
again after the request, or the request was used. The API key is read from a file and never
printed.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

PREFIX = "[goal-loop resume "
KEY_FILES = ("/run/secrets/hermes_api_key",)
STARTS = "/opt/data/gateway-starts.log"
URL = os.environ.get("GOAL_LOOP_RESUME_URL", "http://127.0.0.1:8642/v1/chat/completions")
# An Open WebUI chat (restart resume with owui_key) goes through hermes-models, so its session map
# records the turn for the chat and the delivery binds the answer to that chat.
OWUI_URL = os.environ.get("GOAL_LOOP_RESUME_OWUI_URL", "http://hermes-models:8080/v1/chat/completions")
TIMEOUT = float(os.environ.get("GOAL_LOOP_RESUME_TIMEOUT", "3300"))
OPEN = ("open", "in_work")
TEXT = (PREFIX + "{nonce}] Follow-up run of the goal loop (no owner message). Continue the work on the "
        "open goals of this chat. Do not ask the owner for a go; record owner decisions with "
        "goal_decision.")
RESTART_TEXT = (PREFIX + "{nonce}] Hermes restarted while your last turn in this chat was running; that "
                "turn was cut off (no owner message now). Read the end of this chat, find the step that "
                "was cut off and continue the owner's last request from there. Check the real state "
                "before you repeat a step with side effects. Do not ask the owner for a go.")


def _load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _save(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True)
    os.replace(path + ".tmp", path)


def _last_start():
    try:
        with open(STARTS, encoding="utf-8") as handle:
            parts = handle.read().split()
        return float(parts[-1]) if parts else 0.0
    except (OSError, ValueError):
        return 0.0


def _key():
    for path in [os.environ.get("GOAL_LOOP_RESUME_KEY_FILE", "")] + list(KEY_FILES):
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read().strip()
    return os.environ.get("API_SERVER_KEY", "")


def due(request, state, last_start):
    """The reason why the follow-up run must not post, or '' when it may post."""
    if not request or request.get("status") != "scheduled":
        return "no scheduled request"
    if request.get("reason") == "restart":
        # Made at plugin registration, a few seconds before the start line of the same process; a
        # later start (another restart) makes a new request from the same running note.
        if last_start > float(request.get("created", 0)) + 120:
            return "Hermes started again after the request"
        return ""  # a turn that the restart cut off resumes with or without goals
    if last_start > float(request.get("created", 0)):
        return "Hermes started again after the request"
    if not state or not state.get("go"):
        return "no go"
    if not any(not g.get("done") and g.get("status", "open") in OPEN for g in state.get("goals", [])):
        return "no open goal"
    return ""


def post(request, key, opener=None):
    restart = request.get("reason") == "restart"
    text = (RESTART_TEXT if restart else TEXT).format(nonce=request["nonce"])
    body = {"model": request.get("model") or "hermes-agent", "stream": False,
            "messages": [{"role": "user", "content": text}]}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
               "X-Hermes-Session-Id": request["session_id"]}
    url = URL
    if restart and request.get("owui_key"):
        url = OWUI_URL
        headers["X-Hermes-Session-Key"] = request["owui_key"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    open_ = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    for attempt in range(3):
        try:
            with open_(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode() or "{}")
            choice = (data.get("choices") or [{}])[0]
            return str((choice.get("message") or {}).get("content") or "")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 2:
                return f"goal-loop: Folge-Run nicht gestartet (HTTP {exc.code})."
            time.sleep(10 * (attempt + 1))
    return ""


def main(argv):
    request_file, state_file = argv[1], argv[2]
    request = _load(request_file)
    why = due(request, _load(state_file), _last_start())
    if why:
        if request and request.get("status") == "scheduled":
            request.update(status="skipped", reason=why, updated=time.time())
            _save(request_file, request)
        return 0
    request.update(status="running", updated=time.time())
    _save(request_file, request)
    answer = post(request, _key())
    done = _load(request_file) or request
    if done.get("nonce") == request["nonce"] and done.get("status") == "running":
        done.update(status="done", updated=time.time())
        _save(request_file, done)
    if answer.strip():
        print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
