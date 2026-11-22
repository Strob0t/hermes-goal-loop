"""Tests of 0.4.11 (z14): the plugin appends no text to the answer; what it showed before goes to the
model as English context, and the model shows it to the owner in the owner's language."""
import unittest

import goal_loop
from goal_loop import LABELS, _chat_key, make_transform_llm_output

try:
    from tests.test_043 import CHECKS, IDS, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import CHECKS, IDS, owner, set_and_go, setup, tool

KEY = _chat_key("s1")
EV = "ls -l /srv/app at 09:12, exit 0"
OWNER_LANGUAGE = "in the owner's language"
TWO = (("g1", "first goal", "check one"), ("g2", "second goal", "check two"))


def answer(store, s, text="Interim report"):
    return make_transform_llm_output(store, s)(response_text=text, session_id="s1", platform="api_server")


def shown(store, s, text="Interim report", session_id="s1", platform="api_server", **ids):
    """End a turn (the answer must stay unchanged) and return the goal display that the next turn
    gets as context ("" when there is none); the display is consumed as in pre_llm_call."""
    out = make_transform_llm_output(store, s)(response_text=text, session_id=session_id,
                                              platform=platform, **ids)
    assert out is None, out
    state = store.load(_chat_key(ids.get("conversation_id") or session_id))
    return goal_loop._with_display(store, state, "") or ""


def label_strings():
    out = []
    for value in LABELS.values():
        values = value.values() if isinstance(value, dict) else [value]
        out.extend(v.split("{")[0].strip() for v in values if len(v.split("{")[0].strip()) > 6)
    return out


class NoAppendedTextTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_goal_set_without_go_puts_the_list_and_the_go_question_into_the_context(self):
        tools, store, _, s = setup()
        reply = tools.goal_set({"goals": [{"id": g, "text": t, "check": c} for g, t, c in TWO]}, **IDS)
        self.assertIn(OWNER_LANGUAGE, reply)
        self.assertIn("go", reply)
        self.assertIsNone(answer(store, s, "Here is the plan."))
        context = owner(store, "what does g2 do?", settings=s) or ""
        self.assertIn("g1", context)
        self.assertIn("first goal", context)
        self.assertIn("check two", context)
        self.assertIn("ask the owner", context)
        self.assertIn("'go'", context)
        self.assertIn(OWNER_LANGUAGE, context)
        self.assertIn("keep the goal ids and decision numbers", context)

    def test_an_open_decision_without_actionable_goal_goes_into_the_context(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        tool(store)
        answer(store, s)
        out = tools.goal_decision({"text": "Use port 8080 or 8081?", "goal_id": "g1", "evidence": EV}, **IDS)
        self.assertNotIn("deferred", out)
        self.assertIsNone(answer(store, s, "I need a decision."))
        context = owner(store, "which one do you propose?", settings=s) or ""
        self.assertIn("Use port 8080 or 8081?", context)
        self.assertIn("1.", context)
        self.assertIn(OWNER_LANGUAGE, context)

    def test_the_final_list_goes_into_the_context_one_time(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        owner(store, "go", settings=s)
        tool(store)
        self.assertIsNone(answer(store, s))
        owner(store, "go", settings=s)
        self.assertIn("confirmed", tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS))
        self.assertIsNone(answer(store, s, "All done."))
        first = owner(store, "thanks, and now?", settings=s) or ""
        self.assertIn("g1", first)
        self.assertIn(LABELS["status"]["done"], first)
        self.assertIn(OWNER_LANGUAGE, first)
        self.assertIsNone(answer(store, s, "Nothing more."))
        second = owner(store, "ok", settings=s) or ""
        self.assertNotIn("g1", second)

    def test_goal_status_returns_the_list_and_tells_the_owner_language(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": g, "text": t, "check": c} for g, t, c in TWO]}, **IDS)
        out = tools.goal_status({}, **IDS)
        self.assertIn("g1: first goal", out)
        self.assertIn("check two", out)
        self.assertIn("show the result to the owner in the owner's language", goal_loop.GOAL_STATUS_DESCRIPTION)

    def test_no_label_appears_in_any_transform_result_over_a_full_loop(self):
        tools, store, _, s = setup()
        results = []
        tools.goal_set({"goals": [{"id": g, "text": t, "check": c} for g, t, c in TWO]}, **IDS)
        results.append(answer(store, s, "Plan."))
        owner(store, "go", settings=s)
        tools.goal_go({}, **IDS)
        tool(store)
        results.append(answer(store, s, "Started."))
        owner(store, "go", settings=s)
        tools.goal_decision({"text": "Keep the old log?", "goal_id": "g2", "evidence": EV}, **IDS)
        tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS)
        results.append(answer(store, s, "Decision needed."))
        owner(store, "1A go", settings=s)
        tools.goal_claim({"goal_id": "g2", "claim": {"summary": "s", **CHECKS}}, **IDS)
        results.append(answer(store, s, "Done."))
        self.assertTrue(store.load(KEY).all_done())
        for result in results:
            text = result or ""
            for label in label_strings():
                self.assertNotIn(label, text)
        self.assertEqual(results, [None] * len(results))


if __name__ == "__main__":
    unittest.main()
