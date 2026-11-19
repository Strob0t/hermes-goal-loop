"""Tests of the one-block rule (goals d1-d3): the answer has at most one goal block, the plugin
block, and the go question is the last line, one time."""
import unittest

from goal_loop import make_transform_llm_output, strip_model_blocks
from tests.test_v03 import IDS, owner, setup

MODEL_BLOCKS = (
    "Ich habe den Plan angepasst.\n\n"
    "---\nGoal status (needs the owner):\n"
    "- [ ] s0: Vorabprüfung (check: Log zeigt X) [open]\n"
    "- [ ] s1: Dienst läuft (check: healthy) [open]\n\n"
    "Reply \"go\" to start.\n\n"
    "---\nZielstatus:\n"
    "- [ ] s0: Vorabprüfung (Prüfung: Log zeigt X) [offen]\n\n"
    "Offene Entscheidungen:\n1. Option A oder B\n\n"
    "Gibst du das Go?"
)


class OneBlockTest(unittest.TestCase):
    def _de_goal_set(self, tools, store, s, goals):
        owner(store, "Bitte prüfe das und schreibe mir dann einen Bericht, wie es läuft", settings=s)
        return tools.goal_set({"goals": [{"id": g, "text": t, "check": c} for g, t, c in goals]}, **IDS)

    def test_two_goal_set_in_one_turn_give_one_block(self):
        tools, store, _, s = setup()
        self._de_goal_set(tools, store, s, [("t1", "Ziel eins", "P1")])
        tools.goal_set({"goals": [{"id": "t1", "text": "Ziel eins", "check": "P1"},
                                  {"id": "t2", "text": "Ziel zwei", "check": "P2"}]}, **IDS)
        answer = make_transform_llm_output(store, s)(response_text="Zwei Aufrufe.", **IDS)
        self.assertEqual(answer.count("Zielstatus:"), 1)
        self.assertEqual(answer.count("t1:"), 1)
        self.assertEqual(answer.count("t2:"), 1)
        self.assertEqual(answer.count("Gibst du das Go?"), 1)
        self.assertTrue(answer.rstrip().endswith("Gibst du das Go?"))

    def test_model_written_blocks_are_removed(self):
        tools, store, _, s = setup()
        self._de_goal_set(tools, store, s, [("s0", "Vorabprüfung", "Log zeigt X")])
        answer = make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS)
        self.assertNotIn("Goal status", answer)
        self.assertNotIn("Reply \"go\"", answer)
        self.assertNotIn("[open]", answer)
        self.assertNotIn("Option A oder B", answer)
        self.assertEqual(answer.count("Zielstatus:"), 1)
        self.assertEqual(answer.count("s0:"), 1)
        self.assertTrue(answer.startswith("Ich habe den Plan angepasst."))
        self.assertTrue(answer.rstrip().endswith("Gibst du das Go?"))
        self.assertEqual(answer.count("Gibst du das Go?"), 1)

    def test_answer_ends_with_the_go_question_once(self):
        tools, store, _, s = setup()
        self._de_goal_set(tools, store, s, [("g1", "Ziel", "P")])
        body = "Gibst du das Go?\nKurz der Plan.\nGibst du das Go?"
        answer = make_transform_llm_output(store, s)(response_text=body, **IDS)
        self.assertEqual(answer.count("Gibst du das Go?"), 1)
        self.assertTrue(answer.rstrip().endswith("Gibst du das Go?"))
        self.assertIn("Kurz der Plan.", answer)

    def test_text_without_a_block_is_kept(self):
        text = "Erster Absatz.\n\n---\n\nZweiter Absatz mit - [ ] einer Liste.\n1. Punkt"
        self.assertEqual(strip_model_blocks(text), text)
        self.assertEqual(strip_model_blocks(text, go_question=True), text)

    def test_no_plugin_block_keeps_the_body(self):
        tools, store, _, s = setup()
        self._de_goal_set(tools, store, s, [("g1", "Ziel", "P")])
        make_transform_llm_output(store, s)(response_text="erste Antwort", **IDS)
        # nothing new to show: the plugin returns None and leaves the model text unchanged
        self.assertIsNone(make_transform_llm_output(store, s)(response_text=MODEL_BLOCKS, **IDS))


if __name__ == "__main__":
    unittest.main()
