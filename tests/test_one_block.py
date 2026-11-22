"""Tests of the one-block rule (goals d1-d3): the answer has at most one goal block, the plugin
block, and the go question is the last line, one time."""
import unittest

from goal_loop import make_transform_llm_output, strip_model_blocks
from tests.test_v03 import IDS, owner, setup

GO_QUESTION = "Do you give the go?"
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

    def test_two_goal_set_in_one_turn_give_one_block(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("t1", "goal one", "P1")])
        tools.goal_set({"goals": [{"id": "t1", "text": "goal one", "check": "P1"},
                                  {"id": "t2", "text": "goal two", "check": "P2"}]}, **IDS)
        answer = make_transform_llm_output(store, s)(response_text="Two calls.", **IDS)
        self.assertEqual(answer.count("Goal status:"), 1)
        self.assertEqual(answer.count("t1:"), 1)
        self.assertEqual(answer.count("t2:"), 1)
        self.assertEqual(answer.count(GO_QUESTION), 1)
        self.assertTrue(answer.rstrip().endswith(GO_QUESTION))

    def test_model_written_blocks_are_removed(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("s0", "the plugin goal", "the plugin check")])
        answer = make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS)
        self.assertNotIn("needs the owner", answer)
        self.assertNotIn("Reply \"go\"", answer)
        self.assertNotIn("precheck", answer)
        self.assertNotIn("Option A or B", answer)
        self.assertEqual(answer.count("Goal status:"), 1)
        self.assertEqual(answer.count("s0:"), 1)
        self.assertIn("s0: the plugin goal", answer)
        self.assertTrue(answer.startswith("I changed the plan."))
        self.assertTrue(answer.rstrip().endswith(GO_QUESTION))
        self.assertEqual(answer.count(GO_QUESTION), 1)

    def test_answer_ends_with_the_go_question_once(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("g1", "goal", "P")])
        body = "Do you give the go?\nThe plan in short.\nDo you give the go?"
        answer = make_transform_llm_output(store, s)(response_text=body, **IDS)
        self.assertEqual(answer.count(GO_QUESTION), 1)
        self.assertTrue(answer.rstrip().endswith(GO_QUESTION))
        self.assertIn("The plan in short.", answer)

    def test_text_without_a_block_is_kept(self):
        text = "First paragraph.\n\n---\n\nSecond paragraph with - [ ] a list.\n1. Item"
        self.assertEqual(strip_model_blocks(text), text)
        self.assertEqual(strip_model_blocks(text, go_question=True), text)

    def test_no_plugin_block_keeps_the_body(self):
        tools, store, _, s = setup()
        self._goal_set(tools, store, s, [("g1", "goal", "P")])
        make_transform_llm_output(store, s)(response_text="first answer", **IDS)
        # nothing new to show: the plugin returns None and leaves the model text unchanged
        self.assertIsNone(make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS))


if __name__ == "__main__":
    unittest.main()
