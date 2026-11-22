"""Tests of 0.4.5 (r32), with the 0.4.10 rules: the stop-cause line blocker or done holds the loop until the
next owner message; the stop cause owner-question holds it only when no goal can be worked (R2); an owner
message without 'go' (a question, an answer) does not restart the loop, only 'go' does (R1)."""
import unittest

from goal_loop import _chat_key, make_pre_stop, make_transform_llm_output

try:  # discover -s tests -t . (package) or discover -s tests (top level)
    from tests.test_043 import EV, IDS, fake_modules, ledger_events, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import EV, IDS, fake_modules, ledger_events, owner, set_and_go, setup, tool

QUESTION_END = "Decision 2 is open (A/B/C).\n\nStop cause: owner-question"
BLOCKER_END = "The host cannot be reached.\n\nStop cause: blocker"
NO_STATEMENT_END = "Step 3 is complete.\n\nStop cause: no-statement"


def stop_with(store, settings, text, attempt=0):
    return make_pre_stop(store, settings)(session_id="s1", platform="api_server", attempt=attempt,
                                          final_response=text)


def ready():
    tools, store, _, s = setup(idle_pause_after=0)
    set_and_go(tools, store)
    tool(store)
    return tools, store, s


class StopCauseHoldTest(unittest.TestCase):
    def test_owner_question_stop_cause_with_an_actionable_goal_continues(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, QUESTION_END)["action"], "continue")
        self.assertNotIn("stop_cause_hold", ledger_events(store))
        self.assertIn("owner_question_not_held", ledger_events(store))

    def test_blocker_and_done_stop_causes_hold_the_loop(self):
        for text in (BLOCKER_END, "All complete.\nStop cause: done"):
            tools, store, s = ready()
            with fake_modules():
                self.assertIsNone(stop_with(store, s, text))
                tool(store)
                self.assertIsNone(stop_with(store, s, "more text", attempt=1))
            self.assertNotIn("continued", ledger_events(store))
            self.assertIn("stop_cause_hold", ledger_events(store))

    def test_no_statement_lets_the_loop_go_on(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, NO_STATEMENT_END)["action"], "continue")

    def test_the_next_go_message_lifts_the_hold(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertIsNone(stop_with(store, s, BLOCKER_END))
            owner(store, "go on with r33, go", settings=s)
            tool(store)
            self.assertEqual(stop_with(store, s, "Interim report")["action"], "continue")

    def test_the_stop_cause_in_the_final_answer_also_holds(self):
        """transform_llm_output reads the line when pre_stop did not run for this answer."""
        tools, store, s = ready()
        with fake_modules():
            make_transform_llm_output(store, s)(response_text=BLOCKER_END, **IDS)
            self.assertIsNone(stop_with(store, s, "more in the same turn"))
        self.assertEqual(store.load(_chat_key("s1")).stop_hold, "blocker")


class OwnerQuestionKeepsDecisionTest(unittest.TestCase):
    def _waiting(self):
        """g1 waits for decision 1 (linked): no goal can be worked, the loop holds."""
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, "Interim report")["action"], "continue")
            tools.goal_decision({"text": "A/B/C, recommendation A", "goal_id": "g1", "evidence": EV}, **IDS)
            tool(store)
            self.assertIsNone(stop_with(store, s, QUESTION_END))
        return tools, store, s

    def test_an_owner_question_does_not_lift_the_open_decision(self):
        for question in ("how can we prevent that?", "What does option B mean?", "what else is still open?"):
            tools, store, s = self._waiting()
            owner(store, question, settings=s)
            with fake_modules():
                tool(store)
                self.assertIsNone(stop_with(store, s, "Answer to the question without a stop cause"), question)
            events = ledger_events(store)
            self.assertNotIn("continued", events[events.index("decision"):], question)
            self.assertEqual([i for i, _ in store.load(_chat_key("s1")).open_decisions()], [1], question)

    def test_go_lifts_the_decision_wait(self):
        tools, store, s = self._waiting()
        owner(store, "go", settings=s)
        with fake_modules():
            tool(store)
            self.assertEqual(stop_with(store, s, "Interim report")["action"], "continue")

    def test_an_option_answer_with_go_lifts_the_decision_wait(self):
        for text in ("1a go", "r4b A, 1B go", "the recommendation, go", "r32 ok, 1A, go!"):
            tools, store, s = self._waiting()
            owner(store, text, settings=s)
            with fake_modules():
                tool(store)
                self.assertEqual(stop_with(store, s, "Interim report")["action"], "continue", text)

    def test_an_option_answer_without_go_closes_the_decision_and_pauses(self):
        tools, store, s = self._waiting()
        owner(store, "1A", settings=s)
        self.assertEqual(store.load(_chat_key("s1")).open_decisions(), [])
        with fake_modules():
            tool(store)
            self.assertIsNone(stop_with(store, s, "Interim report"))

    def test_a_question_after_a_bare_owner_wait_keeps_it(self):
        """The 0.4.3 owner wait (decision + block on every goal) also survives an owner question."""
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, "Interim report")["action"], "continue")
            tools.goal_decision({"text": "A/B/C", "evidence": EV}, **IDS)
            tools.goal_block({"goal_id": "g1", "reason": "owner decision missing"}, **IDS)
            tool(store, "goal_block")
            self.assertIsNone(stop_with(store, s, "waits"))
        owner(store, "what else is still open?", settings=s)
        with fake_modules():
            tool(store)
            self.assertIsNone(stop_with(store, s, "List without a stop cause"))
        self.assertFalse(store.load(_chat_key("s1")).go)


if __name__ == "__main__":
    unittest.main()
