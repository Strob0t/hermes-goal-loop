"""Tests of r4: a turn that a Hermes restart cut off gets one continuation turn after the start."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from goal_loop import restart_resume, resume, resume_run
from tests.test_v03 import IDS, owner, setup


class Jobs:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"id": f"job{len(self.calls)}"}


def dead_note(state_dir, sid, **extra):
    data = {"session_id": sid, "platform": "api_server", "boot_id": "old-boot", "chat_key": "chat:" + sid,
            "owui_key": "", "model": "tier-medium", "started": 1000.0, "pid": 999999, "pid_start": "1"}
    data.update(extra)
    with open(restart_resume.path_for(state_dir, sid), "w", encoding="utf-8") as handle:
        json.dump(data, handle)


class NoteTest(unittest.TestCase):
    def test_a_turn_writes_a_note_at_start_and_removes_it_at_the_end(self):
        tools, store, _, s = setup()
        owner(store, "Please check the service and report to me", settings=s)
        path = restart_resume.path_for(store.base_dir, "s1")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as handle:
            note = json.load(handle)
        self.assertEqual(note["pid"], os.getpid())
        from goal_loop import make_transform_llm_output
        make_transform_llm_output(store, s)(response_text="complete", **IDS)
        self.assertFalse(os.path.exists(path))

    def test_the_setting_off_writes_no_note(self):
        tools, store, _, s = setup(restart_resume=False)
        owner(store, "Please check the service", settings=s)
        self.assertFalse(os.path.exists(restart_resume.path_for(store.base_dir, "s1")))


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="gl-rr-")
        self.home = tempfile.mkdtemp(prefix="gl-rr-home-")
        self.events = []
        self.jobs = Jobs()

    def scan(self, now=2000.0, boot="new-boot"):
        return restart_resume.scan_and_schedule(
            self.dir, boot, lambda key, event, **f: self.events.append((key, event, f)),
            create_job=self.jobs, hermes_home=self.home, now=now)

    def test_a_note_of_a_dead_process_gets_one_continuation_job(self):
        dead_note(self.dir, "sess-a", owui_key="owui:u1:c1")
        self.assertEqual(self.scan(), ["job1"])
        self.assertEqual(self.jobs.calls[0]["name"], "goal-loop restart resume")
        request = resume.load_request(resume.request_path(self.dir, "restart:sess-a"))
        self.assertEqual(request["reason"], "restart")
        self.assertEqual(request["owui_key"], "owui:u1:c1")
        self.assertEqual(request["session_id"], "sess-a")
        self.assertEqual(self.events[0][1], "restart_resume_scheduled")
        self.assertEqual(os.listdir(self.dir).count("running_sess-a.json"), 0)
        # the note is used: a second scan makes no second job
        self.assertEqual(self.scan(), [])
        self.assertEqual(len(self.jobs.calls), 1)

    def test_a_note_of_a_live_process_is_kept(self):
        dead_note(self.dir, "sess-b", pid=os.getpid(), pid_start=restart_resume._proc_start(os.getpid()))
        self.assertEqual(self.scan(), [])
        self.assertTrue(os.path.exists(restart_resume.path_for(self.dir, "sess-b")))

    def test_a_note_of_this_boot_is_kept(self):
        dead_note(self.dir, "sess-c", boot_id="new-boot")
        self.assertEqual(self.scan(), [])

    def test_an_old_note_is_dropped_without_a_turn(self):
        dead_note(self.dir, "sess-d", started=0.0)
        self.assertEqual(self.scan(now=10 * 3600.0), [])
        self.assertEqual(self.events[0][1], "restart_resume_dropped")
        self.assertFalse(os.path.exists(restart_resume.path_for(self.dir, "sess-d")))

    def test_the_wrapper_names_the_goal_state_of_the_chat(self):
        dead_note(self.dir, "sess-e", chat_key="chat:conv-root")
        self.scan()
        script = os.path.join(self.home, "scripts", self.jobs.calls[0]["script"])
        with open(script, encoding="utf-8") as handle:
            self.assertIn("chat_conv-root.json", handle.read())


class RunTest(unittest.TestCase):
    def request(self, **extra):
        return {"session_id": "sess-a", "nonce": "n1", "model": "tier-medium", "created": 1000.0,
                "status": "scheduled", **extra}

    def test_a_restart_request_is_due_without_goals(self):
        self.assertEqual(resume_run.due(self.request(reason="restart"), None, 1010.0), "")

    def test_a_restart_request_is_not_due_after_a_later_start(self):
        self.assertIn("started again", resume_run.due(self.request(reason="restart"), None, 2000.0))

    def test_a_follow_up_request_still_needs_the_go(self):
        self.assertEqual(resume_run.due(self.request(), None, 0.0), "no go")

    def _post(self, request):
        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"choices": [{"message": {"content": "onward"}}]}'

        def opener(req, timeout=None):
            seen["url"] = req.full_url
            seen["headers"] = dict(req.header_items())
            seen["body"] = json.loads(req.data)
            return Resp()
        self.assertEqual(resume_run.post(request, "k", opener=opener), "onward")
        return seen

    def test_an_open_webui_chat_resumes_through_hermes_models_with_its_key(self):
        seen = self._post(self.request(reason="restart", owui_key="owui:u1:c1"))
        self.assertEqual(seen["url"], resume_run.OWUI_URL)
        self.assertEqual(seen["headers"]["X-hermes-session-key"], "owui:u1:c1")
        self.assertEqual(seen["headers"]["X-hermes-session-id"], "sess-a")
        text = seen["body"]["messages"][0]["content"]
        self.assertTrue(text.startswith(resume.RESUME_PREFIX + "n1]"))
        self.assertIn("restarted", text)

    def test_a_local_session_resumes_on_the_local_api_server(self):
        seen = self._post(self.request(reason="restart"))
        self.assertEqual(seen["url"], resume_run.URL)
        self.assertNotIn("X-hermes-session-key", seen["headers"])

    def test_the_restart_turn_is_no_owner_message(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        owner(store, resume_run.RESTART_TEXT.format(nonce="n9"), settings=s)
        from goal_loop import GO_REFUSED
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)


if __name__ == "__main__":
    unittest.main()
