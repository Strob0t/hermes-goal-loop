"""Tests of 0.4.5 (r32): the stop-cause line holds the loop until the next owner message, and an owner
question does not lift an open owner decision (only go, an option answer, or a statement does)."""
import unittest

from goal_loop import _chat_key, make_pre_stop, make_transform_llm_output

try:  # discover -s tests -t . (package) or discover -s tests (top level)
    from tests.test_043 import IDS, fake_modules, ledger_events, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import IDS, fake_modules, ledger_events, owner, set_and_go, setup, tool

QUESTION_END = "Offen ist Entscheidung 2 (A/B/C).\n\nStopp-Grund: owner-question"
BLOCKER_END = "Der Host ist nicht erreichbar.\n\nStop cause: blocker"
NO_STATEMENT_END = "Schritt 3 ist fertig.\n\nStopp-Grund: no-statement"


def stop_with(store, settings, text, attempt=0):
    return make_pre_stop(store, settings)(session_id="s1", platform="api_server", attempt=attempt,
                                          final_response=text)


def ready():
    tools, store, _, s = setup(idle_pause_after=0)
    set_and_go(tools, store)
    tool(store)
    return tools, store, s


class StopCauseHoldTest(unittest.TestCase):
    def test_owner_question_stop_cause_gives_no_continuation(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertIsNone(stop_with(store, s, QUESTION_END))
            tool(store)
            self.assertIsNone(stop_with(store, s, "noch etwas", attempt=1))
        self.assertNotIn("continued", ledger_events(store))
        self.assertIn("stop_cause_hold", ledger_events(store))

    def test_blocker_and_done_stop_causes_hold_the_loop(self):
        for text in (BLOCKER_END, "Alles erledigt.\nStopp-Grund: done"):
            tools, store, s = ready()
            with fake_modules():
                self.assertIsNone(stop_with(store, s, text))
            self.assertNotIn("continued", ledger_events(store))

    def test_no_statement_lets_the_loop_go_on(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, NO_STATEMENT_END)["action"], "continue")

    def test_the_next_owner_message_lifts_the_hold(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertIsNone(stop_with(store, s, QUESTION_END))
            owner(store, "mach weiter mit r33", settings=s)
            tool(store)
            self.assertEqual(stop_with(store, s, "Zwischenstand")["action"], "continue")

    def test_the_stop_cause_in_the_final_answer_also_holds(self):
        """transform_llm_output reads the line when pre_stop did not run for this answer."""
        tools, store, s = ready()
        with fake_modules():
            make_transform_llm_output(store, s)(response_text=QUESTION_END, **IDS)
            self.assertIsNone(stop_with(store, s, "weiter im selben Turn"))
        self.assertEqual(store.load(_chat_key("s1")).stop_hold, "owner-question")


class OwnerQuestionKeepsDecisionTest(unittest.TestCase):
    def _waiting(self):
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, "Zwischenstand")["action"], "continue")
            tools.goal_decision({"text": "A/B/C, Empfehlung A"}, **IDS)
            tool(store)
            self.assertIsNone(stop_with(store, s, QUESTION_END))
        return tools, store, s

    def test_an_owner_question_does_not_lift_the_open_decision(self):
        tools, store, s = self._waiting()
        owner(store, "wie können wir das verhindern?", settings=s)
        with fake_modules():
            tool(store)
            self.assertIsNone(stop_with(store, s, "Antwort auf die Frage ohne Stopp-Grund"))
        events = ledger_events(store)
        self.assertNotIn("continued", events[events.index("decision"):])
        self.assertIn("owner_question_kept_wait", events)

    def test_go_lifts_the_decision_wait(self):
        tools, store, s = self._waiting()
        owner(store, "go", settings=s)
        with fake_modules():
            tool(store)
            self.assertEqual(stop_with(store, s, "Zwischenstand")["action"], "continue")

    def test_an_option_answer_lifts_the_decision_wait(self):
        for text in ("4a", "r4b A, passt das?", "nimm die Empfehlung", "r32 ok, r4b A nimm sie als goals auf!"):
            tools, store, s = self._waiting()
            owner(store, text, settings=s)
            with fake_modules():
                tool(store)
                self.assertEqual(stop_with(store, s, "Zwischenstand")["action"], "continue", text)

    def test_a_question_after_a_bare_owner_wait_keeps_it(self):
        """The 0.4.3 owner wait (decision + block on every goal) also survives an owner question."""
        tools, store, s = ready()
        with fake_modules():
            self.assertEqual(stop_with(store, s, "Zwischenstand")["action"], "continue")
            tools.goal_decision({"text": "A/B/C"}, **IDS)
            tools.goal_block({"goal_id": "g1", "reason": "owner decision missing"}, **IDS)
            tool(store, "goal_block")
            self.assertIsNone(stop_with(store, s, "warte"))
        owner(store, "was ist sonst noch offen?", settings=s)
        with fake_modules():
            tool(store)
            self.assertIsNone(stop_with(store, s, "Liste ohne Stopp-Grund"))
        self.assertEqual(store.load(_chat_key("s1")).decision_wait, True)


if __name__ == "__main__":
    unittest.main()
