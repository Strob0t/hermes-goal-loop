"""Tests of the one-block rule (goals d1-d3), 0.4.11: the plugin appends no block, the next context has
one goal list and the go question one time, and the model answer stays unchanged."""
import unittest

from goal_loop import GO_ASK, make_transform_llm_output
from tests.test_v03 import IDS, owner, setup

MODEL_BLOCKS = (
    "I changed the plan.\n\n"
    "---\nGoal status (needs the owner):\n"
    "- [ ] s0: precheck (check: the log shows X) [open]\n"
    "- [ ] s1: the service runs (check: healthy) [open]\n\n"
    "Reply \"go\" to start.\n\n"
    "---\nGoal status:\n"
    "- [ ] s0: precheck (check: the log shows X) [open]\n\n"
    "Open decisions:\n1. Option A or B\n\n"
    "Do you give the go?"
)


class OneBlockTest(unittest.TestCase):
    def _goal_set(self, tools, store, s, goals):
        owner(store, "Please check this and then write me a report on how it runs", settings=s)
        return tools.goal_set({"goals": [{"id": g, "text": t, "check": c} for g, t, c in goals]}, **IDS)

    def test_two_goal_set_in_one_turn_give_one_list_in_the_context(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("t1", "goal one", "P1")])
        tools.goal_set({"goals": [{"id": "t1", "text": "goal one", "check": "P1"},
                                  {"id": "t2", "text": "goal two", "check": "P2"}]}, **IDS)
        self.assertIsNone(make_transform_llm_output(store, s)(response_text="Two calls.", **IDS))
        context = owner(store, "and now?", settings=s)
        self.assertEqual(context.count("Goal status:"), 1)
        self.assertEqual(context.count("t1:"), 1)
        self.assertEqual(context.count("t2:"), 1)
        self.assertEqual(context.count(GO_ASK), 1)

    def test_model_written_blocks_stay_and_the_context_has_the_plugin_list(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("s0", "the plugin goal", "the plugin check")])
        self.assertIsNone(make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS))
        context = owner(store, "ok?", settings=s)
        self.assertIn("s0: the plugin goal", context)
        self.assertNotIn("precheck", context)
        self.assertNotIn("Option A or B", context)

    def test_the_go_question_goes_into_the_context_once_per_turn(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("g1", "goal", "P")])
        body = "Do you give the go?\nThe plan in short.\nDo you give the go?"
        self.assertIsNone(make_transform_llm_output(store, s)(response_text=body, **IDS))
        self.assertEqual(owner(store, "why?", settings=s).count(GO_ASK), 1)
        self.assertEqual(owner(store, "and?", settings=s), GO_ASK)

    def test_no_plugin_block_keeps_the_body(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("g1", "goal", "P")])
        make_transform_llm_output(store, s)(response_text="first answer", **IDS)
        # the plugin returns None and leaves the model text unchanged
        self.assertIsNone(make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS))


if __name__ == "__main__":
    unittest.main()
