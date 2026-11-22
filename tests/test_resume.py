"""Tests of the follow-up runs (L15, owner decision 5): resume.py, resume_run.py and the hooks."""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from goal_loop import Settings, _Tools, _chat_key, make_pre_llm_call, make_transform_llm_output, resume
from goal_loop import resume_run
from goal_loop.state import GoalStore

IDS = {"session_id": "s1", "conversation_id": "c1", "platform": "api_server"}
CHECKS = {"checks": [{"command": "c", "result": "passed", "exitCode": 0}]}


class Ctx:
    subagent_lifecycle = None

    def __init__(self):
        class Llm:
            def complete(self, messages, **kw):
                class R:
                    text = "VERDICT: done"
                return R()
        self.llm = Llm()

    def get_config(self, key, default=None):
        return default


class FakeJobs:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": f"job{len(self.calls)}"}


def owner(store, text="go", session="s1", **extra):
    return make_pre_llm_call(store)(session_id=session, platform="api_server", user_message=text,
                                    conversation_history=[{"role": "user", "content": text}],
                                    conversation_id="c1", model="tier-complex", **extra)


class Base(unittest.TestCase):
    def setUp(self):
        self.store = GoalStore(tempfile.mkdtemp(prefix="glr-"))
        self.home = tempfile.mkdtemp(prefix="glr-home-")
        self.jobs = FakeJobs()
        patcher = mock.patch.multiple(resume, CREATE_JOB=self.jobs, HERMES_HOME=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ctx = Ctx()
        self.tools = _Tools(self.store, self.ctx, Settings(self.ctx))

    def started(self, ids=("g1", "g2")):
        self.tools.goal_set({"goals": [{"id": g, "text": g, "check": "c"} for g in ids]}, **IDS)
        owner(self.store)
        self.assertTrue(self.tools.goal_go({}, **IDS).startswith("Go recorded"))


class ScheduleTest(Base):
    def test_a_run_that_ends_with_open_goals_schedules_one_follow_up_run(self):
        self.started()
        transform = make_transform_llm_output(self.store, Settings(self.ctx))
        transform(response_text="ANSWER", **IDS)
        self.assertIn("follow-up run", self.store.load(_chat_key("c1")).loop_note)  # 0.4.10 (R3): context
        self.assertEqual(len(self.jobs.calls), 1)
        call = self.jobs.calls[0]
        self.assertTrue(call["no_agent"])
        self.assertEqual(call["schedule"], "in 1m")
        self.assertEqual(call["repeat"], 1)
        wrapper = os.path.join(self.home, "scripts", call["script"])
        self.assertTrue(os.path.exists(wrapper))
        request = resume.load_request(resume.request_path(self.store.base_dir, _chat_key("c1")))
        self.assertEqual(request["session_id"], "s1")
        self.assertEqual(request["model"], "tier-complex")
        self.assertEqual(request["status"], "scheduled")
        self.assertEqual(self.store.load(_chat_key("c1")).resumes, 1)
        events = [e["event"] for e in self.store.read_ledger(_chat_key("c1"))]
        self.assertIn("resume_scheduled", events)

    def test_a_held_state_after_a_restart_gets_no_follow_up_run(self):
        import goal_loop
        self.started()
        state = self.store.load(_chat_key("c1"))
        state.boot_id = "an-other-process"
        self.store.save(state)
        out = make_transform_llm_output(self.store, Settings(self.ctx))(response_text="A", **IDS)
        self.assertEqual(self.jobs.calls, [])
        self.assertNotIn("follow-up run", out or "")
        self.assertNotEqual(goal_loop.BOOT_ID, "an-other-process")

    def test_a_pending_request_is_not_scheduled_again(self):
        self.started()
        transform = make_transform_llm_output(self.store, Settings(self.ctx))
        transform(response_text="A", **IDS)
        out = transform(response_text="B", **IDS)
        self.assertEqual(len(self.jobs.calls), 1)
        self.assertNotIn("follow-up run", out or "")

    def test_no_follow_up_before_the_go_or_when_off(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        make_transform_llm_output(self.store, Settings(self.ctx))(response_text="A", **IDS)
        self.assertEqual(self.jobs.calls, [])
        owner(self.store)
        self.tools.goal_go({}, **IDS)
        make_transform_llm_output(self.store, Settings(overrides={"resume": False}))(
            response_text="B", **IDS)
        self.assertEqual(self.jobs.calls, [])

    def test_the_limit_pauses_with_a_decision_and_a_new_go_resets_it(self):
        self.started(("g1",))
        settings = Settings(overrides={"resume_max": 1})
        transform = make_transform_llm_output(self.store, settings)
        transform(response_text="A", **IDS)
        resume.mark(resume.request_path(self.store.base_dir, _chat_key("c1")), "done")
        out = transform(response_text="B", **IDS)
        state = self.store.load(_chat_key("c1"))
        self.assertEqual(state.goal("g1").status, "paused")
        self.assertIn("follow-up runs", out)
        self.assertEqual(len(self.jobs.calls), 1)
        owner(self.store, "go")
        self.assertTrue(self.tools.goal_go({}, **IDS).startswith("Go recorded"))
        self.assertEqual(self.store.load(_chat_key("c1")).resumes, 0)


class ChainTest(Base):
    def test_the_follow_up_turn_schedules_the_next_follow_up_run(self):
        # live finding 2026-11-19: the request was still "running" (the post waits for the turn),
        # so the end of the follow-up turn scheduled nothing and the chain stopped after 1 run.
        self.started(("g1",))
        transform = make_transform_llm_output(self.store, Settings(self.ctx))
        transform(response_text="A", **IDS)
        path = resume.request_path(self.store.base_dir, _chat_key("c1"))
        first = resume.load_request(path)
        resume.mark(path, "running")  # resume_run.main before the post
        owner(self.store, resume.RESUME_PREFIX + first["nonce"] + "] Follow-up run")
        self.assertEqual(resume.load_request(path)["status"], "taken")
        transform(response_text="B", **IDS)
        self.assertIn("follow-up run", self.store.load(_chat_key("c1")).loop_note)
        self.assertEqual(len(self.jobs.calls), 2)
        second = resume.load_request(path)
        self.assertNotEqual(second["nonce"], first["nonce"])
        self.assertEqual(second["status"], "scheduled")

    def test_a_foreign_nonce_does_not_take_the_request(self):
        self.started(("g1",))
        make_transform_llm_output(self.store, Settings(self.ctx))(response_text="A", **IDS)
        path = resume.request_path(self.store.base_dir, _chat_key("c1"))
        owner(self.store, resume.RESUME_PREFIX + "other] Follow-up run")
        self.assertEqual(resume.load_request(path)["status"], "scheduled")


class StopWordCancelsTest(Base):
    def test_an_owner_message_cancels_a_scheduled_follow_up_run(self):
        from goal_loop import make_pre_stop
        self.started(("g1",))
        make_transform_llm_output(self.store, Settings(self.ctx))(response_text="A", **IDS)
        owner(self.store, "stop")
        make_pre_stop(self.store, Settings(self.ctx))(attempt=0, **IDS)
        request = resume.load_request(resume.request_path(self.store.base_dir, _chat_key("c1")))
        self.assertEqual(request["status"], "cancelled")
        self.assertFalse(self.store.load(_chat_key("c1")).go)

    def test_an_owner_message_pauses_at_the_owner_message_without_pre_stop(self):
        # Live 18:35: pre_stop did not run any more (fork counter used up), so the pause must act
        # in pre_llm_call. 0.4.10 (R1): the paused turn gets a note for the model.
        self.started(("g1",))
        make_transform_llm_output(self.store, Settings(self.ctx))(response_text="A", **IDS)
        out = owner(self.store, "stop")
        self.assertIn("paused the goal loop", out)
        self.assertIn("Do not call goal_go", out)
        state = self.store.load(_chat_key("c1"))
        self.assertFalse(state.go)
        request = resume.load_request(resume.request_path(self.store.base_dir, _chat_key("c1")))
        self.assertEqual(request["status"], "cancelled")
        events = [e["event"] for e in self.store.read_ledger(_chat_key("c1"))]
        self.assertEqual(events.count("paused_by_owner_message"), 1)
        from goal_loop import make_pre_stop
        self.assertIsNone(make_pre_stop(self.store, Settings(self.ctx))(attempt=0, **IDS))
        events = [e["event"] for e in self.store.read_ledger(_chat_key("c1"))]
        self.assertEqual(events.count("paused_by_owner_message"), 1)


class FollowUpIsNoOwnerMessageTest(Base):
    def test_a_follow_up_turn_does_not_open_the_go(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        owner(self.store, resume.RESUME_PREFIX + "abc] Follow-up run of the goal loop")
        self.assertTrue(self.tools.goal_go({}, **IDS).startswith("error"))
        owner(self.store, "go")
        self.assertTrue(self.tools.goal_go({}, **IDS).startswith("Go recorded"))

    def test_a_follow_up_turn_gets_the_goal_context(self):
        self.started(("g1",))
        out = owner(self.store, resume.RESUME_PREFIX + "abc] Follow-up run")
        self.assertIn("g1", out)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="glrr-")
        self.request = os.path.join(self.dir, "resume.json")
        self.state = os.path.join(self.dir, "state.json")

    def write(self, request, state):
        json.dump(request, open(self.request, "w"))
        json.dump(state, open(self.state, "w"))

    def test_due_reasons(self):
        req = {"status": "scheduled", "created": 100.0, "session_id": "s", "nonce": "n"}
        goals = {"go": True, "goals": [{"id": "g1", "done": False, "status": "open"}]}
        self.assertEqual(resume_run.due(req, goals, 50.0), "")
        self.assertIn("started again", resume_run.due(req, goals, 200.0))
        self.assertEqual(resume_run.due(req, dict(goals, go=False), 50.0), "no go")
        done = {"go": True, "goals": [{"id": "g1", "done": True, "status": "done"}]}
        self.assertEqual(resume_run.due(req, done, 50.0), "no open goal")
        self.assertEqual(resume_run.due(dict(req, status="done"), goals, 50.0), "no scheduled request")

    def test_main_posts_one_turn_prints_the_answer_and_marks_done(self):
        self.write({"status": "scheduled", "created": 9e12, "session_id": "s9", "nonce": "n1", "model": "m"},
                   {"go": True, "goals": [{"id": "g1", "done": False, "status": "in_work"}]})
        seen = {}

        def fake_post(request, key, opener=None):
            seen.update(request)
            return "ANSWER OF THE FOLLOW-UP RUN"
        out = io.StringIO()
        with mock.patch.object(resume_run, "post", fake_post), mock.patch("sys.stdout", out):
            resume_run.main(["x", self.request, self.state])
        self.assertEqual(seen["session_id"], "s9")
        self.assertIn("ANSWER OF THE FOLLOW-UP RUN", out.getvalue())
        self.assertEqual(json.load(open(self.request))["status"], "done")

    def test_main_is_silent_and_skips_without_a_go(self):
        self.write({"status": "scheduled", "created": 9e12, "session_id": "s9", "nonce": "n1"},
                   {"go": False, "goals": [{"id": "g1", "done": False, "status": "open"}]})
        out = io.StringIO()
        with mock.patch.object(resume_run, "post", side_effect=AssertionError("posted")), \
                mock.patch("sys.stdout", out):
            resume_run.main(["x", self.request, self.state])
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(json.load(open(self.request))["status"], "skipped")

    def test_the_post_sends_the_session_header_and_the_prefix(self):
        captured = {}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode())
            return Resp(json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode())
        answer = resume_run.post({"session_id": "s7", "nonce": "n7", "model": "tier-complex"}, "k", opener)
        self.assertEqual(answer, "OK")
        self.assertEqual(captured["headers"]["X-hermes-session-id"], "s7")
        self.assertTrue(captured["body"]["messages"][0]["content"].startswith(resume.RESUME_PREFIX + "n7]"))
        self.assertEqual(captured["body"]["model"], "tier-complex")


if __name__ == "__main__":
    unittest.main()
