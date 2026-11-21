"""Tests of 0.4.9 (z11): the background watchdog does not continue while a turn of the session runs (a stale
turn mark does not count), waits a grace after the first tick that sees all awaited processes ended, and
decision answers with a separator ('8. A', '8: A', '8 - A') close the decision, also with a '?' in the message."""
import sys
import unittest

import goal_loop
from goal_loop import _chat_key, make_post_tool_call, make_pre_llm_call, make_pre_stop, make_transform_llm_output

try:
    from tests.test_043 import IDS, owner, set_and_go, setup, tool
    from tests.test_047 import bg_modules
except ImportError:
    from test_043 import IDS, owner, set_and_go, setup, tool
    from test_047 import bg_modules

KEY = _chat_key("s1")


def waiting(**settings):
    """A local chat with go and 1 open goal that waits for proc_9; no turn mark is left."""
    tools, store, _, s = setup(idle_pause_after=0, **settings)
    set_and_go(tools, store)
    tool(store)
    goal_loop._turn_marks().clear()
    patch, proc, registry = bg_modules(running=True)
    with patch:
        make_pre_stop(store, s)(session_id="s1", platform="local", attempt=0, final_response="warte auf proc_9")
    state = store.load(KEY)
    assert state.wait_reason == "waiting_background", state.wait_reason
    return store, s


def tick(store, s, now, sent):
    return goal_loop.background_watchdog_tick(
        store, s, now=now, running=lambda p: False,
        inject=lambda text, session_key: sent.append(text) or True)


class ActiveTurnTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def tearDown(self):
        goal_loop._turn_marks().clear()

    def test_an_ended_process_and_a_running_turn_give_no_continuation(self):
        store, s = waiting(background_end_grace=0)
        since = store.load(KEY).bg_wait["since"]
        goal_loop._mark_turn("s1", KEY, now=since + 5)  # the completion-notice turn starts
        sent = []
        self.assertEqual(tick(store, s, since + 20, sent), [])
        self.assertEqual(sent, [])
        self.assertEqual(store.load(KEY).wait_reason, "waiting_background")

    def test_a_stale_turn_mark_does_not_block_the_continuation(self):
        store, s = waiting(background_end_grace=0, turn_active_max=600)
        since = store.load(KEY).bg_wait["since"]
        goal_loop._mark_turn("s1", now=since)  # a hard-stopped turn: no transform, the mark stays
        sent = []
        self.assertEqual(tick(store, s, since + 599, sent), [])
        self.assertEqual(tick(store, s, since + 601, sent), [KEY])
        self.assertEqual(len(sent), 1)

    def test_the_hooks_set_and_clear_the_turn_mark(self):
        tools, store, _, s = setup(idle_pause_after=0)
        set_and_go(tools, store)
        goal_loop._turn_marks().clear()
        make_pre_llm_call(store, s)(session_id="s1", platform="local", user_message="weiter mit z7",
                                    conversation_history=[{"role": "user", "content": "weiter mit z7"}])
        self.assertIn("s1", goal_loop._turn_marks())
        make_post_tool_call(store)(tool_name="terminal", session_id="s1", conversation_id="")
        self.assertIn("s1", goal_loop._turn_marks())
        make_transform_llm_output(store, s)(response_text="Stand", session_id="s1", platform="local")
        self.assertNotIn("s1", goal_loop._turn_marks())
        self.assertNotIn(KEY, goal_loop._turn_marks())

    def test_the_turn_marks_survive_a_module_reload(self):
        goal_loop._mark_turn("s7", now=1.0)
        self.assertIs(sys.modules["goal_loop_turns"].MARKS, goal_loop._turn_marks())
        self.assertEqual(goal_loop._turn_marks()["s7"], 1.0)


class EndGraceTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_the_continuation_waits_the_grace_after_the_first_ended_tick(self):
        store, s = waiting(background_end_grace=120)
        since = store.load(KEY).bg_wait["since"]
        sent = []
        self.assertEqual(tick(store, s, since + 60, sent), [])  # first tick that sees all ended
        self.assertEqual(store.load(KEY).bg_wait["ended_seen"], since + 60)
        self.assertEqual(tick(store, s, since + 179, sent), [])
        self.assertEqual(sent, [])
        self.assertEqual(tick(store, s, since + 180, sent), [KEY])
        self.assertEqual(len(sent), 1)
        last = [e for e in store.read_ledger(KEY) if e["event"] == "background_watchdog"][-1]
        self.assertEqual((last["why"], last["continued"]), ("ended", True))

    def test_the_default_grace_is_120_s(self):
        self.assertEqual(goal_loop.DEFAULTS["background_end_grace"], 120)
        self.assertEqual(goal_loop.DEFAULTS["turn_active_max"], 600)


def nine_decisions():
    tools, store, _, s = setup(idle_pause_after=0)
    set_and_go(tools, store)
    tool(store)
    for i in range(1, 10):
        tools.goal_decision({"text": f"Frage {i}: A/B/C"}, **IDS)
    make_transform_llm_output(store, s)(response_text="x", session_id="s1", platform="api_server")
    return tools, store, s


def open_numbers(store):
    return [i for i, _ in store.load(KEY).open_decisions()]


class SeparatorAnswerTest(unittest.TestCase):
    def test_the_real_answer_of_2015_closes_8_and_9(self):
        tools, store, s = nine_decisions()
        owner(store, "8. A / 9. B ?", settings=s)
        self.assertEqual(open_numbers(store), [1, 2, 3, 4, 5, 6, 7])

    def test_separators_close_the_decision(self):
        for text in ("8. A", "8: A", "8 - A", "8) a", "8.A"):
            with self.subTest(text=text):
                tools, store, s = nine_decisions()
                owner(store, text, settings=s)
                self.assertEqual(open_numbers(store), [1, 2, 3, 4, 5, 6, 7, 9])

    def test_a_numbered_answer_with_a_question_mark_lifts_the_decision_wait(self):
        tools, store, s = nine_decisions()
        state = store.load(KEY)
        state.decision_wait = True
        store.save(state)
        owner(store, "8. A / 9. B ?", settings=s)
        self.assertFalse(store.load(KEY).decision_wait)

    def test_questions_and_versions_close_none(self):
        for text in ("was bedeutet 8?", "läuft 0.4.8 A schon?", "um 20:15 A", "zeige mir 8"):
            with self.subTest(text=text):
                tools, store, s = nine_decisions()
                owner(store, text, settings=s)
                self.assertEqual(len(open_numbers(store)), 9)


if __name__ == "__main__":
    unittest.main()
