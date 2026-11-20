"""Tests of issue #1: the goal loop does not stop at the turn iteration limit while goals are open
after the go. A local (CLI, TUI) session continues with a follow-up turn through inject_message; a
turn cut at the limit is not counted as idle."""
import unittest

import goal_loop
from goal_loop import make_pre_llm_call, make_transform_llm_output, resume_run
from tests.test_v03 import owner, set_and_go, setup, stop

LOCAL = {"session_id": "s1", "platform": "cli"}


class Ctx:
    def __init__(self, accept=True):
        self.accept, self.sent = accept, []

    def inject_message(self, content, role="user", **kw):
        self.sent.append(content)
        return self.accept


class LocalFollowUpTest(unittest.TestCase):
    def setUp(self):
        self._saved = goal_loop._CTX
        self.addCleanup(setattr, goal_loop, "_CTX", self._saved)

    def run_turn_end(self, accept=True, **settings):
        tools, store, _, s = setup(**settings)
        set_and_go(tools, store)
        goal_loop._CTX = Ctx(accept)
        # a turn cut at the iteration limit: no pre_stop, the finalizer still runs the transform
        out = make_transform_llm_output(store, s)(response_text="Teilbericht.", **LOCAL)
        return tools, store, s, out

    def test_a_local_turn_end_with_open_goals_injects_one_follow_up_turn(self):
        tools, store, _, out = self.run_turn_end()
        self.assertEqual(len(goal_loop._CTX.sent), 1)
        self.assertTrue(goal_loop._CTX.sent[0].startswith(resume_run.PREFIX))
        self.assertIn("g1", out)
        state = store.load(tools._load(LOCAL).chat_key)
        self.assertEqual(state.resumes, 1)
        self.assertEqual(store.read_ledger(state.chat_key)[-1]["event"], "resume_injected")

    def test_the_injected_turn_is_no_owner_message_and_gets_the_goal_context(self):
        tools, store, s, _ = self.run_turn_end()
        key = tools._load(LOCAL).chat_key
        before = (store.gate_count(key), store.gate_count(key, "change"))
        context = make_pre_llm_call(store, s)(session_id="s1", platform="cli",
                                              user_message=goal_loop._CTX.sent[0],
                                              conversation_history=[{"role": "user",
                                                                     "content": goal_loop._CTX.sent[0]}])
        self.assertIn("g1", context or "")
        # no owner message: neither the go gate nor the change gate counts it
        self.assertEqual((store.gate_count(key), store.gate_count(key, "change")), before)

    def test_a_refused_inject_gives_the_paused_line(self):
        tools, store, _, out = self.run_turn_end(accept=False)
        self.assertTrue("pausiert" in out.lower() or "paused" in out.lower())
        state = store.load(tools._load(LOCAL).chat_key)
        self.assertEqual(state.resumes, 0)
        self.assertEqual(store.read_ledger(state.chat_key)[-1]["event"], "resume_inject_refused")

    def test_resume_max_stops_the_injects(self):
        tools, store, s, _ = self.run_turn_end(resume_max=1)
        make_transform_llm_output(store, s)(response_text="Zweiter Bericht.", **LOCAL)
        self.assertEqual(len(goal_loop._CTX.sent), 1)

    def test_a_stopped_loop_injects_nothing(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        goal_loop._CTX = Ctx()
        owner(store, "stopp", settings=s)
        make_transform_llm_output(store, s)(response_text="Bericht.", **LOCAL)
        self.assertEqual(goal_loop._CTX.sent, [])

    def test_api_server_keeps_the_cron_follow_up_run(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        goal_loop._CTX = Ctx()
        make_transform_llm_output(store, s)(response_text="Bericht.", session_id="s1", platform="api_server")
        self.assertEqual(goal_loop._CTX.sent, [])


class IdleAfterLimitTest(unittest.TestCase):
    def test_idle_turns_before_a_follow_up_turn_do_not_pause_the_loop(self):
        tools, store, _, s = setup(idle_pause_after=3)
        set_and_go(tools, store)
        stop(store, s, 0)
        stop(store, s, 1)          # 2 continuations without a tool call
        make_pre_llm_call(store, s)(session_id="s1", platform="api_server",
                                    user_message=resume_run.TEXT.format(nonce="n1"),
                                    conversation_history=[])
        stop(store, s, 0)
        state = store.load(tools._load({"session_id": "s1"}).chat_key)
        self.assertNotEqual(state.goal("g1").status, "paused")
        self.assertLess(state.idle_nudges, 3)

    def test_an_owner_message_starts_a_new_idle_count(self):
        tools, store, _, s = setup(idle_pause_after=3)
        set_and_go(tools, store)
        stop(store, s, 0)
        stop(store, s, 1)
        owner(store, "Bitte mach mit den offenen Zielen weiter", settings=s)
        stop(store, s, 0)
        state = store.load(tools._load({"session_id": "s1"}).chat_key)
        self.assertNotEqual(state.goal("g1").status, "paused")


if __name__ == "__main__":
    unittest.main()
