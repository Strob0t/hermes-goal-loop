"""Tests of 0.4.10 (z13): each owner message pauses the loop (only 'go' keeps or gives the go), the
deferred list (owner items wait while a goal can still be worked), and English only."""
import importlib.util
import json
import os
import types
import unittest
from unittest import mock

import goal_loop
from goal_loop import GO_REFUSED, _chat_key, make_pre_llm_call, make_pre_stop, make_transform_llm_output, resume

try:
    from tests.test_043 import IDS, ledger_events, owner, set_and_go, setup, stop, tool
    from tests.test_047 import bg_modules
    from tests.test_0411 import shown
except ImportError:
    from test_043 import IDS, ledger_events, owner, set_and_go, setup, stop, tool
    from test_047 import bg_modules
    from test_0411 import shown

KEY = _chat_key("s1")
EV = "ls -l /srv/app at 09:12, exit 0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWO = (("g1", "first goal", "c1"), ("g2", "second goal", "c2"))


def running(goals=(("g1", "a", "c1"),), **settings):
    settings.setdefault("idle_pause_after", 0)
    tools, store, _, s = setup(**settings)
    set_and_go(tools, store, goals)
    tool(store)
    answer(store, s)  # the full list after goal_set is shown here
    return tools, store, s


def answer(store, s, text="Interim report", platform="api_server"):
    """0.4.11: the goal display that the next turn gets as context (the answer stays unchanged)."""
    return shown(store, s, text, platform=platform)


def open_numbers(store):
    return [i for i, _ in store.load(KEY).open_decisions()]


class PauseTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_an_owner_message_without_go_pauses_the_loop(self):
        tools, store, s = running()
        owner(store, "how far are you?", settings=s)
        self.assertFalse(store.load(KEY).go)
        self.assertIn("paused_by_owner_message", ledger_events(store))

    def test_the_paused_turn_gets_a_context_note_and_no_continuation(self):
        tools, store, s = running()
        note = owner(store, "show me the log", settings=s) or ""
        self.assertIn("paused", note)
        self.assertIn("'go'", note)
        self.assertIn("Do not call goal_go", note)
        self.assertIn("at the end", note)
        self.assertIsNone(stop(store, s))

    def test_no_follow_up_run_is_scheduled_or_injected_while_paused(self):
        calls, sent = [], []
        tools, store, s = running()
        owner(store, "wait a moment", settings=s)
        with mock.patch.object(resume, "CREATE_JOB", lambda **kw: calls.append(kw) or {"id": "j1"}):
            answer(store, s, platform="api_server")
        saved = goal_loop._CTX
        goal_loop._CTX = types.SimpleNamespace(inject_message=lambda text: sent.append(text) or True)
        try:
            answer(store, s, platform="local")
        finally:
            goal_loop._CTX = saved
        self.assertEqual((calls, sent), ([], []))

    def test_the_watchdog_does_not_continue_a_paused_loop(self):
        tools, store, s = running(background_end_grace=0)
        patch, proc, registry = bg_modules(running=True)
        with patch:
            make_pre_stop(store, s)(session_id="s1", platform="local", attempt=0, final_response="waits for proc_9")
        self.assertEqual(store.load(KEY).wait_reason, "waiting_background")
        owner(store, "what does proc_9 do?", settings=s, platform="local")
        state = store.load(KEY)
        self.assertEqual((state.go, state.bg_wait, state.wait_reason), (False, {}, ""))
        sent = []
        out = goal_loop.background_watchdog_tick(store, s, now=10 ** 10, running=lambda p: False,
                                                 inject=lambda text, session_key: sent.append(text) or True)
        self.assertEqual((out, sent), ([], []))

    def test_a_completion_notice_does_not_resume_a_paused_loop(self):
        tools, store, s = running()
        owner(store, "pause please", settings=s)
        notice = "Background process proc_9 finished.\n\n[INTERNAL NOTIFICATION: not from the owner]"
        make_pre_llm_call(store, s)(session_id="s1", platform="api_server", user_message=notice,
                                    conversation_history=[{"role": "user", "content": notice}])
        self.assertFalse(store.load(KEY).go)
        self.assertIsNone(stop(store, s))
        self.assertEqual(goal_loop.background_watchdog_tick(store, s, now=10 ** 10, running=lambda p: False,
                                                            inject=lambda *a, **k: True), [])

    def test_go_with_other_text_keeps_the_go_and_closes_the_named_decision(self):
        tools, store, s = running()
        for i in range(1, 5):
            tools.goal_decision({"text": f"Question {i}: A/B/C", "evidence": EV}, **IDS)
        owner(store, "4B go", settings=s)
        state = store.load(KEY)
        self.assertTrue(state.go)
        self.assertEqual(open_numbers(store), [1, 2, 3])
        self.assertNotIn("paused_by_owner_message", ledger_events(store))
        owner(store, "list ok GO", settings=s)
        self.assertTrue(store.load(KEY).go)

    def test_go_after_a_pause_resumes_the_loop(self):
        tools, store, s = running()
        owner(store, "stop", settings=s)
        self.assertFalse(store.load(KEY).go)
        owner(store, "go", settings=s)
        self.assertTrue(store.load(KEY).go)
        self.assertEqual(stop(store, s)["action"], "continue")

    def test_goal_go_without_a_go_message_is_refused(self):
        tools, store, s = running()
        owner(store, "explain the plan", settings=s)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)
        owner(store, "and then?", settings=s)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)
        self.assertFalse(store.load(KEY).go)

    def test_only_go_is_a_default_go_word(self):
        self.assertEqual(goal_loop.GO_WORDS, ("go",))
        self.assertNotIn("stop_words", goal_loop.DEFAULTS)
        tools, store, s = running()
        owner(store, "retiew"[::-1], settings=s)
        self.assertFalse(store.load(KEY).go)

    def test_the_vocabulary_file_adds_a_go_word_and_a_consent_pattern(self):
        tools, store, s = running()
        owner(store, "onward", settings=s)
        self.assertFalse(store.load(KEY).go)  # no file: English defaults only
        with open(os.path.join(store.base_dir, "vocabulary.json"), "w", encoding="utf-8") as handle:
            json.dump({"go_words": ["onward"], "consent_patterns": [r"fine\s+by\s+me"]}, handle)
        owner(store, "onward", settings=s)
        self.assertTrue(store.load(KEY).go)
        tools.goal_decision({"text": "Question 1: A/B", "evidence": EV}, **IDS)
        owner(store, "fine by me, onward", settings=s)
        self.assertEqual(open_numbers(store), [])
        self.assertTrue(store.load(KEY).go)


class DeferredTest(unittest.TestCase):
    def test_a_decision_with_an_actionable_goal_goes_to_the_deferred_list(self):
        tools, store, s = running(TWO)
        out = tools.goal_decision({"text": "Which port for g1? A) 80 B) 8080", "goal_id": "g1", "evidence": EV}, **IDS)
        self.assertIn("recorded in the deferred list", out)
        self.assertIn("continue with goal(s) g2", out)
        state = store.load(KEY)
        self.assertEqual([e["kind"] for e in state.deferred], ["decision"])
        self.assertNotIn("Which port", answer(store, s))
        verdict = stop(store, s)
        self.assertEqual(verdict["action"], "continue")
        self.assertIn("g2", verdict["message"])
        self.assertNotIn("- g1 ", verdict["message"])

    def test_a_block_with_an_actionable_goal_goes_to_the_deferred_list(self):
        tools, store, s = running(TWO, block_min_rounds=1)
        out = tools.goal_block({"goal_id": "g1", "reason": "the host is down"}, **IDS)
        self.assertIn("recorded in the deferred list", out)
        self.assertIn("g2", out)
        self.assertEqual([e["kind"] for e in store.load(KEY).deferred], ["block"])
        self.assertNotIn("the host is down", answer(store, s))
        self.assertEqual(stop(store, s)["action"], "continue")

    def test_a_refused_list_change_goes_to_the_deferred_list(self):
        tools, store, s = running(TWO)
        out = tools.goal_set({"goals": [{"id": "g3", "text": "new goal", "check": "c3"}]}, **IDS)
        self.assertIn("recorded in the deferred list", out)
        state = store.load(KEY)
        self.assertEqual([e["kind"] for e in state.deferred], ["change"])
        self.assertEqual([g.id for g in state.goals], ["g1", "g2"])
        self.assertNotIn("new goal", answer(store, s))

    def test_the_block_shows_the_deferred_list_when_no_goal_is_actionable(self):
        tools, store, s = running(TWO, block_min_rounds=1)
        tools.goal_decision({"text": "Which port for g1? A) 80 B) 8080", "goal_id": "g1", "evidence": EV}, **IDS)
        tools.goal_set({"goals": [{"id": "g3", "text": "new goal", "check": "c3"}]}, **IDS)
        out = tools.goal_block({"goal_id": "g2", "reason": "the host is down"}, **IDS)
        self.assertNotIn("recorded in the deferred list", out)
        text = answer(store, s)
        self.assertIn("Which port for g1", text)
        self.assertIn("the host is down", text)
        self.assertIn("new goal", text)
        self.assertIsNone(stop(store, s))
        self.assertNotIn("Which port", answer(store, s, "Later answer"))

    def test_an_owner_question_stop_cause_does_not_stop_while_a_goal_is_actionable(self):
        tools, store, s = running(TWO)
        tools.goal_decision({"text": "Which port for g1? A) 80 B) 8080", "goal_id": "g1", "evidence": EV}, **IDS)
        verdict = make_pre_stop(store, s)(session_id="s1", platform="api_server", attempt=0,
                                          final_response="Which port?\nStop cause: owner-question")
        self.assertEqual(verdict["action"], "continue")
        self.assertIn("goal_decision", verdict["message"])
        self.assertIn("continue with g2", verdict["message"])
        self.assertEqual(store.load(KEY).stop_hold, "")


EVIDENCE_SENTENCE = ("Before you propose options, list open points or claim a goal, check each point against "
                     "the live state in this turn; do not rely on the chat history or memory.")


class EvidenceTest(unittest.TestCase):
    def test_a_decision_without_evidence_is_refused_and_records_nothing(self):
        tools, store, s = running()
        before = (len(store.load(KEY).decisions), len(store.read_ledger(KEY)))
        for evidence in (None, "", "   ", "short text"):
            with self.subTest(evidence=evidence):
                args = {"text": "Question: A/B"}
                if evidence is not None:
                    args["evidence"] = evidence
                out = tools.goal_decision(args, **IDS)
                self.assertTrue(out.startswith("error:"), out)
                self.assertIn("examine the live state first and pass the result as evidence", out)
        self.assertEqual((len(store.load(KEY).decisions), len(store.read_ledger(KEY))), before)

    def test_a_decision_with_evidence_is_recorded(self):
        tools, store, s = running()
        out = tools.goal_decision({"text": "Question: A/B", "evidence": EV}, **IDS)
        self.assertIn("Decision 1 recorded", out)
        self.assertEqual(store.load(KEY).decisions, ["Question: A/B"])
        self.assertEqual([e for e in store.read_ledger(KEY) if e["event"] == "decision"][-1]["evidence"], EV)

    def test_a_decision_with_evidence_is_deferred_while_a_goal_is_actionable(self):
        tools, store, s = running(TWO)
        out = tools.goal_decision({"text": "Question: A/B", "goal_id": "g1", "evidence": EV}, **IDS)
        self.assertIn("recorded in the deferred list", out)
        self.assertEqual([e["kind"] for e in store.load(KEY).deferred], ["decision"])

    def test_the_prompt_section_and_the_pause_note_have_the_evidence_rule(self):
        self.assertIn(EVIDENCE_SENTENCE, goal_loop.prompt_section({"platform": "api_server"}))
        tools, store, s = running()
        self.assertIn(EVIDENCE_SENTENCE, owner(store, "status?", settings=s) or "")

    def test_the_judge_prompt_refuses_evidence_from_the_chat_or_memory(self):
        from goal_loop.judge import JUDGE_SYSTEM
        self.assertIn("Refuse a claim whose evidence comes only from the chat, memory or a worker's report",
                      " ".join(JUDGE_SYSTEM.split()))
        self.assertIn("evidence", goal_loop._GOAL_DECISION_SCHEMA["required"])


def load_check():
    path = os.path.join(ROOT, "scripts", "check_english.py")
    if not os.path.exists(path):
        raise AssertionError("scripts/check_english.py is missing")
    spec = importlib.util.spec_from_file_location("check_english", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    return check


class EnglishTest(unittest.TestCase):
    def test_an_all_caps_licence_name_is_no_hit(self):
        check = load_check()
        self.assertIsNone(check.WORD.search("MIT licence"))
        self.assertIsNotNone(check.WORD.search("ein Test " + "tim"[::-1] + " Text"))
        self.assertIsNotNone(check.WORD.search("tiM"[::-1] + " dem"))

    def test_the_plugin_files_are_english(self):
        check = load_check()
        hits = []
        for folder in ("goal_loop", "tests", "integration", "scripts"):
            for name in sorted(os.listdir(os.path.join(ROOT, folder))):
                if not name.endswith((".py", ".yaml")):
                    continue
                with open(os.path.join(ROOT, folder, name), encoding="utf-8") as handle:
                    for number, line in enumerate(handle, start=1):
                        if check.UMLAUT.search(line) or check.WORD.search(line):
                            hits.append(f"{folder}/{name}:{number}")
        self.assertEqual(hits, [])

    def test_no_language_detection_and_no_second_language_texts(self):
        self.assertFalse(hasattr(goal_loop, "detect_language"))
        self.assertEqual(set(goal_loop.DECISION_TEXT), {"budget", "idle", "reject", "stale", "resume_max"})


if __name__ == "__main__":
    unittest.main()
