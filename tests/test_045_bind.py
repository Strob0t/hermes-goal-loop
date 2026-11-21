"""Tests of 0.4.5 (r34): a restart resume of an Open WebUI chat writes a bind hint for the delivery adapter before
the cron job is made; a resume without an Open WebUI key writes none."""
import json
import os
import tempfile
import unittest

from goal_loop import restart_resume, resume

try:  # discover -s tests -t . (package) or discover -s tests (top level)
    from tests.test_restart_resume import Jobs, dead_note
except ImportError:
    from test_restart_resume import Jobs, dead_note

KEY = "owui:d4a337c2-5bdd-4b19-8b6b-a950de76b389:440c325b-fa5a-4919-862d-c6c940d67fed"


class BindHintTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="gl-bh-")
        self.home = tempfile.mkdtemp(prefix="gl-bh-home-")
        self.seen = []

        def jobs(**kw):
            # what the delivery adapter sees at the first sight of the job: the hint must already be there
            folder = os.path.join(self.home, "cron", "goal-loop-bind")
            self.seen.append(sorted(os.listdir(folder)) if os.path.isdir(folder) else [])
            return {"id": "job1"}
        self.jobs = jobs

    def scan(self):
        return restart_resume.scan_and_schedule(self.dir, "new-boot", lambda *a, **f: None, create_job=self.jobs,
                                                hermes_home=self.home, now=2000.0)

    def test_a_restart_resume_with_an_owui_key_writes_the_hint_before_the_job(self):
        dead_note(self.dir, "run_0d784e613c4347ac9cb9b3cf33d52d52", owui_key=KEY)
        self.assertEqual(self.scan(), ["job1"])
        script = "goal-loop-resume-" + __import__("hashlib").sha256(
            b"restart:run_0d784e613c4347ac9cb9b3cf33d52d52").hexdigest()[:12]
        self.assertEqual(self.seen, [[script + ".json"]])
        with open(os.path.join(self.home, "cron", "goal-loop-bind", script + ".json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["owui_key"], KEY)
        self.assertTrue(os.path.exists(os.path.join(self.home, "scripts", script + ".py")))

    def test_a_resume_without_an_owui_key_writes_no_hint(self):
        dead_note(self.dir, "sess-local")
        self.assertEqual(self.scan(), ["job1"])
        self.assertEqual(self.seen, [[]])

    def test_the_hint_name_matches_the_job_script(self):
        job = resume.schedule_resume(self.dir, "restart:s9", "s9", "auto", "2m", create_job=lambda **kw: (
            self.seen.append(kw["script"]) or {"id": "j9"}), hermes_home=self.home, now=2000.0,
            extra={"reason": "restart", "owui_key": KEY})
        self.assertEqual(job, "j9")
        script = [s for s in self.seen if isinstance(s, str)][0]
        self.assertTrue(os.path.exists(os.path.join(self.home, "cron", "goal-loop-bind", script[:-3] + ".json")))


if __name__ == "__main__":
    unittest.main()
