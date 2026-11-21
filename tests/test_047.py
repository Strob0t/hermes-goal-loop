"""Tests of 0.4.7: decisions closed by number (z3), the restart line (z4), the background watchdog (z5),
the resume log line (z6) and the resume text after a Hermes system note."""
import contextlib
import json
import logging
import os
import sys
import threading
import types
import unittest
from unittest import mock

import goal_loop
from goal_loop import _chat_key, make_pre_llm_call, make_pre_stop, make_transform_llm_output
from goal_loop import resume

try:  # discover -s tests -t . (package) or discover -s tests (top level)
    from tests.test_043 import IDS, ledger_events, owner, set_and_go, setup, tool
except ImportError:
    from test_043 import IDS, ledger_events, owner, set_and_go, setup, tool


def bg_modules(running=True, session_key="agent:main:local:dm:s1"):
    """Hermetic process registry with 1 notify_on_complete process of session s1."""
    proc = types.SimpleNamespace(id="proc_9", parent_session_id="s1", exited=not running, notify_on_complete=True,
                                 watch_patterns=[], session_key=session_key)
    registry = types.SimpleNamespace(
        _lock=threading.Lock(), _running={"proc_9": proc} if running else {},
        is_session_waiting=lambda sid: sid == "proc_9" and running)
    mods = {
        "tools": types.ModuleType("tools"),
        "tools.process_registry": types.SimpleNamespace(process_registry=registry),
        "tools.async_delegation": types.SimpleNamespace(_LIVE_STATES={"running"},
                                                        _session_records=lambda *a: []),
        "hermes_cli": types.ModuleType("hermes_cli"),
        "hermes_cli.goals": types.SimpleNamespace(load_goal=lambda sid: None),
    }
    return mock.patch.dict(sys.modules, mods), proc, registry


def ready(**settings):
    tools, store, _, s = setup(idle_pause_after=0, **settings)
    set_and_go(tools, store)
    tool(store)
    return tools, store, s


def block_text(store, s, platform="api_server"):
    return make_transform_llm_output(store, s)(response_text="Zwischenstand", session_id="s1", platform=platform) or ""


class DecisionsByNumberTest(unittest.TestCase):
    def _three(self):
        tools, store, s = ready()
        for text in ("Frage eins: A/B/C", "Frage zwei: A/B/C", "Frage drei: A/B/C"):
            tools.goal_decision({"text": text}, **IDS)
        block_text(store, s)  # shown once
        return tools, store, s

    def test_a_numbered_answer_closes_only_that_decision(self):
        tools, store, s = self._three()
        owner(store, "2A", settings=s)
        state = store.load(_chat_key("s1"))
        self.assertEqual([i for i, _ in state.open_decisions()], [1, 3])
        text = block_text(store, s)
        self.assertIn("1. Frage eins", text)
        self.assertIn("3. Frage drei", text)
        self.assertNotIn("Frage zwei", text)
        self.assertIn("decisions_closed", ledger_events(store))

    def test_several_numbers_close_several(self):
        tools, store, s = self._three()
        owner(store, "1A 3c", settings=s)
        self.assertEqual([i for i, _ in store.load(_chat_key("s1")).open_decisions()], [2])

    def test_go_closes_all(self):
        tools, store, s = self._three()
        owner(store, "go", settings=s)
        self.assertEqual(store.load(_chat_key("s1")).open_decisions(), [])
        self.assertNotIn("Offene Entscheidungen", block_text(store, s))

    def test_an_answer_without_a_number_closes_all(self):
        tools, store, s = self._three()
        owner(store, "nimm die Empfehlung", settings=s)
        self.assertEqual(store.load(_chat_key("s1")).open_decisions(), [])

    def test_a_question_closes_none(self):
        tools, store, s = self._three()
        owner(store, "Was bedeutet Option B bei 2?", settings=s)
        self.assertEqual(len(store.load(_chat_key("s1")).open_decisions()), 3)

    def test_goal_ids_are_no_decision_numbers(self):
        tools, store, s = self._three()
        owner(store, "r4b A und z4 C", settings=s)  # no decision number: closes all (answer without number)
        self.assertEqual(store.load(_chat_key("s1")).open_decisions(), [])

    def test_a_new_decision_after_the_answer_is_shown(self):
        tools, store, s = self._three()
        owner(store, "1A", settings=s)
        block_text(store, s)
        tools.goal_decision({"text": "Frage vier: A/B/C"}, **IDS)
        text = block_text(store, s)
        self.assertIn("4. Frage vier", text)
        self.assertNotIn("Frage eins", text)

    def test_the_full_list_shows_only_open_decisions(self):
        tools, store, s = self._three()
        owner(store, "1B", settings=s)
        from goal_loop.state import status_block
        full = status_block(store.load(_chat_key("s1")), "de")
        self.assertNotIn("Frage eins", full)
        self.assertIn("2. Frage zwei", full)


class RestartLineTest(unittest.TestCase):
    def test_the_first_answer_after_a_restart_says_the_loop_is_held(self):
        tools, store, s = ready()
        owner(store, "bitte mach weiter mit dem Ziel und dann ist es gut", settings=s)  # German chat
        with mock.patch.object(goal_loop, "BOOT_ID", "f" * 32):
            first = block_text(store, s)
            second = block_text(store, s)
        self.assertIn("Neustart: Schleife angehalten, schreib weiter", first)
        self.assertNotIn("Neustart:", second)
        self.assertIn("restart_hold_note", ledger_events(store))

    def test_no_restart_line_in_the_same_process(self):
        tools, store, s = ready()
        text = block_text(store, s)
        self.assertNotIn("Neustart:", text)
        self.assertNotIn("Restart:", text)

    def test_the_loop_stays_held_after_the_line(self):
        tools, store, s = ready()
        with mock.patch.object(goal_loop, "BOOT_ID", "f" * 32):
            block_text(store, s)
            self.assertIsNone(make_pre_stop(store, s)(session_id="s1", platform="api_server", attempt=0,
                                                      final_response="x"))
        self.assertIn("restart_hold", ledger_events(store))


class BackgroundWatchdogTest(unittest.TestCase):
    def _waiting(self, platform="local", **settings):
        tools, store, s = ready(**settings)
        patch, proc, registry = bg_modules(running=True)
        with patch:
            self.assertIsNone(make_pre_stop(store, s)(session_id="s1", platform=platform, attempt=0,
                                                      final_response="warte auf proc_9"))
        state = store.load(_chat_key("s1"))
        self.assertEqual(state.wait_reason, "waiting_background")
        self.assertEqual(state.bg_wait["procs"], ["proc_9"])
        return tools, store, s

    def test_an_ended_process_without_a_completion_turn_continues_the_loop(self):
        tools, store, s = self._waiting()
        sent = []
        done = goal_loop.background_watchdog_tick(store, s, running=lambda p: False,
                                                  inject=lambda text, session_key: sent.append((text, session_key)) or True)
        self.assertEqual(done, [_chat_key("s1")])
        self.assertEqual(len(sent), 1)
        self.assertTrue(resume.is_resume_text(sent[0][0]))
        self.assertEqual(sent[0][1], "agent:main:local:dm:s1")
        events = store.read_ledger(_chat_key("s1"))
        last = [e for e in events if e["event"] == "background_watchdog"][-1]
        self.assertEqual((last["why"], last["continued"]), ("ended", True))
        self.assertEqual(store.load(_chat_key("s1")).wait_reason, "")

    def test_a_running_process_keeps_the_wait(self):
        tools, store, s = self._waiting()
        sent = []
        done = goal_loop.background_watchdog_tick(store, s, running=lambda p: True,
                                                  inject=lambda *a, **k: sent.append(a) or True)
        self.assertEqual((done, sent), ([], []))
        self.assertEqual(store.load(_chat_key("s1")).wait_reason, "waiting_background")

    def test_the_wait_ends_after_the_limit(self):
        tools, store, s = self._waiting(background_wait_max=3600)
        since = store.load(_chat_key("s1")).bg_wait["since"]
        sent = []
        inject = lambda text, session_key: sent.append(text) or True  # noqa: E731
        self.assertEqual(goal_loop.background_watchdog_tick(store, s, now=since + 3599, running=lambda p: True,
                                                            inject=inject), [])
        self.assertEqual(goal_loop.background_watchdog_tick(store, s, now=since + 3600, running=lambda p: True,
                                                            inject=inject), [_chat_key("s1")])
        last = [e for e in store.read_ledger(_chat_key("s1")) if e["event"] == "background_watchdog"][-1]
        self.assertEqual(last["why"], "timeout")
        self.assertIn("still run after 60 min", sent[0])

    def test_an_api_session_gets_a_follow_up_run(self):
        tools, store, s = self._waiting(platform="api_server")
        calls = []
        done = goal_loop.background_watchdog_tick(
            store, s, running=lambda p: False, schedule=lambda st, state, se, sid: calls.append(sid) or True)
        self.assertEqual((done, calls), ([_chat_key("s1")], ["s1"]))

    def test_a_state_of_another_boot_stays_held(self):
        tools, store, s = self._waiting()
        sent = []
        with mock.patch.object(goal_loop, "BOOT_ID", "e" * 32):
            done = goal_loop.background_watchdog_tick(store, s, running=lambda p: False,
                                                      inject=lambda *a, **k: sent.append(a) or True)
        self.assertEqual((done, sent), ([], []))
        self.assertIn("background_watchdog_held", ledger_events(store))

    def test_an_owner_message_clears_the_wait(self):
        tools, store, s = self._waiting()
        owner(store, "wie ist der Stand?", settings=s)
        self.assertEqual(store.load(_chat_key("s1")).bg_wait, {})


class ResumeLogLineTest(unittest.TestCase):
    def test_a_restart_resume_turn_logs_nonce_and_job(self):
        tools, store, s = ready()
        path = resume.request_path(store.base_dir, "restart:s1")
        resume._write(path, {"nonce": "abcdef0123456789", "job_id": "0b2220a91091", "reason": "restart",
                             "status": "scheduled"})
        text = ("[System note: The previous turn was interrupted by a gateway shutdown; the gateway is now back "
                "online. Review the history.]\n\n[goal-loop resume abcdef0123456789] Hermes restarted while ...")
        with self.assertLogs("goal_loop", level=logging.INFO) as logs:
            make_pre_llm_call(store, s)(session_id="s1", platform="api_server", user_message=text,
                                        conversation_history=[{"role": "user", "content": text}])
        line = [m for m in logs.output if "goal-loop resume turn" in m]
        self.assertEqual(len(line), 1)
        self.assertIn("nonce=abcdef0123456789 job=0b2220a91091 reason=restart session=s1", line[0])

    def test_the_resume_text_after_a_system_note_is_no_owner_message(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        text = "[System note: x]\n\n[goal-loop resume 0011223344556677] Hermes restarted ..."
        self.assertTrue(resume.is_resume_text(text))
        self.assertEqual(resume.nonce_of(text), "0011223344556677")
        make_pre_llm_call(store, s)(session_id="s1", platform="api_server", user_message=text,
                                    conversation_history=[{"role": "user", "content": text}])
        self.assertFalse(tools.goal_go({}, **IDS).startswith("Go recorded"))


if __name__ == "__main__":
    unittest.main()
