"""Tests of 0.4.8 (z10): decisions close only on a number with an option, a go word or explicit consent; the
goal block shows the closed numbers once."""
import unittest

from goal_loop import _chat_key, make_transform_llm_output

try:
    from tests.test_043 import EV, IDS, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import EV, IDS, owner, set_and_go, setup, tool

REAL_1800 = ("give me a clear list of all open points (you check the points against the actual state!), if "
             "points are still open give me at least 3 solution proposals for each with a best-practice "
             "recommendation!")


def ready():
    tools, store, _, s = setup(idle_pause_after=0)
    set_and_go(tools, store)
    tool(store)
    for i in range(1, 8):
        tools.goal_decision({"text": f"Question {i}: A/B/C", "evidence": EV}, **IDS)
    make_transform_llm_output(store, s)(response_text="x", session_id="s1", platform="api_server")
    return tools, store, s


def open_numbers(store):
    return [i for i, _ in store.load(_chat_key("s1")).open_decisions()]


def block(store, s):
    return make_transform_llm_output(store, s)(response_text="Status", session_id="s1", platform="api_server") or ""


class StrictCloseTest(unittest.TestCase):
    def test_6a_closes_only_6(self):
        tools, store, s = ready()
        owner(store, "6A", settings=s)
        self.assertEqual(open_numbers(store), [1, 2, 3, 4, 5, 7])

    def test_1c_2b_closes_1_and_2(self):
        tools, store, s = ready()
        owner(store, "1c, 2b", settings=s)
        self.assertEqual(open_numbers(store), [3, 4, 5, 6, 7])

    def test_go_closes_all(self):
        tools, store, s = ready()
        owner(store, "go", settings=s)
        self.assertEqual(open_numbers(store), [])

    def test_consent_closes_all(self):
        for text in ("recommendation", "all recommendations", "ok", "yes, all your recommendations!"):
            with self.subTest(text=text):
                tools, store, s = ready()
                owner(store, text, settings=s)
                self.assertEqual(open_numbers(store), [])

    def test_the_real_request_of_1800_closes_none(self):
        tools, store, s = ready()
        owner(store, REAL_1800, settings=s)
        self.assertEqual(open_numbers(store), [1, 2, 3, 4, 5, 6, 7])

    def test_requests_and_questions_close_none(self):
        for text in ("show me the status", "what does option B mean for 2?", "create the goals",
                     "you have my explicit approval!"):
            with self.subTest(text=text):
                tools, store, s = ready()
                owner(store, text, settings=s)
                self.assertEqual(len(open_numbers(store)), 7)

    def test_the_block_shows_the_closed_numbers_once(self):
        tools, store, s = ready()
        owner(store, "1A", settings=s)
        first = block(store, s)
        self.assertIn("Closed by your answer: 1", first)
        self.assertIn("2. Question 2", first)
        self.assertNotIn("1. Question 1", first)
        self.assertNotIn("Closed by your answer", block(store, s))

    def test_no_closed_line_without_a_close(self):
        tools, store, s = ready()
        owner(store, REAL_1800, settings=s)
        self.assertNotIn("Closed by your answer", block(store, s))


if __name__ == "__main__":
    unittest.main()
