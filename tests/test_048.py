"""Tests of 0.4.8 (z10): decisions close only on a number with an option, a go word or explicit consent; the
goal block shows the closed numbers once."""
import unittest

from goal_loop import _chat_key, make_transform_llm_output

try:
    from tests.test_043 import IDS, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import IDS, owner, set_and_go, setup, tool

REAL_1800 = ("gib mir eine übersichtliche liste mit allen offenen punkte(du prüfst die punkte gegen den ist "
             "zustand!), wenn noch punkte offen sind gibst du mir zu jedem mind. 3 Lösungsvorschläge mit "
             "Empfehlung der bestpractice!")


def ready():
    tools, store, _, s = setup(idle_pause_after=0)
    set_and_go(tools, store)
    tool(store)
    for i in range(1, 8):
        tools.goal_decision({"text": f"Frage {i}: A/B/C"}, **IDS)
    make_transform_llm_output(store, s)(response_text="x", session_id="s1", platform="api_server")
    return tools, store, s


def open_numbers(store):
    return [i for i, _ in store.load(_chat_key("s1")).open_decisions()]


def block(store, s):
    return make_transform_llm_output(store, s)(response_text="Stand", session_id="s1", platform="api_server") or ""


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
        for text in ("Empfehlung", "alle Empfehlungen", "ok", "ja, alle deine Empfehlungen!"):
            with self.subTest(text=text):
                tools, store, s = ready()
                owner(store, text, settings=s)
                self.assertEqual(open_numbers(store), [])

    def test_the_real_request_of_1800_closes_none(self):
        tools, store, s = ready()
        owner(store, REAL_1800, settings=s)
        self.assertEqual(open_numbers(store), [1, 2, 3, 4, 5, 6, 7])

    def test_requests_and_questions_close_none(self):
        for text in ("zeige mir den Stand", "was bedeutet Option B bei 2?", "erstelle die goals",
                     "du hast meine explizite freigabe!"):
            with self.subTest(text=text):
                tools, store, s = ready()
                owner(store, text, settings=s)
                self.assertEqual(len(open_numbers(store)), 7)

    def test_the_block_shows_the_closed_numbers_once(self):
        tools, store, s = ready()
        owner(store, "1A", settings=s)
        first = block(store, s)
        self.assertIn("Closed by your answer: 1", first)
        self.assertIn("2. Frage 2", first)
        self.assertNotIn("1. Frage 1", first)
        self.assertNotIn("Closed by your answer", block(store, s))

    def test_german_label(self):
        tools, store, s = ready()
        owner(store, "1A, das passt so für mich und ist gut", settings=s)  # German chat
        self.assertIn("Mit deiner Antwort geschlossen: 1", block(store, s))

    def test_no_closed_line_without_a_close(self):
        tools, store, s = ready()
        owner(store, REAL_1800, settings=s)
        text = block(store, s)
        self.assertNotIn("geschlossen:", text)
        self.assertNotIn("Closed by your answer", text)


if __name__ == "__main__":
    unittest.main()
