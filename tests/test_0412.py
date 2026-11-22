"""Tests of 0.4.12 (z15): a tool call creates the turn mark when it is absent (a turn that started before a
plugin reload), and a wait for a new background process starts a new wait time."""
import sys
import threading
import time
import types
import unittest
from unittest import mock

import goal_loop
from goal_loop import _chat_key, make_post_tool_call

try:
    from tests.test_049 import tick, waiting
except ImportError:
    from test_049 import tick, waiting

KEY = _chat_key("s1")


def registry_of(*ids):
    """Hermetic process registry with awaited notify_on_complete processes of session s1."""
    procs = {i: types.SimpleNamespace(id=i, parent_session_id="s1", exited=False, notify_on_complete=True,
                                      watch_patterns=[], session_key="agent:main:local:dm:s1") for i in ids}
    registry = types.SimpleNamespace(_lock=threading.Lock(), _running=procs,
                                     is_session_waiting=lambda sid: sid in procs)
    return mock.patch.dict(sys.modules, {"tools": types.ModuleType("tools"),
                                         "tools.process_registry": types.SimpleNamespace(process_registry=registry)})


def note(state, *ids):
    with registry_of(*ids):
        goal_loop._note_background(state, "s1", "local")


class TurnMarkAfterReloadTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def tearDown(self):
        goal_loop._turn_marks().clear()

    def test_a_tool_call_without_a_mark_keeps_the_turn_as_running(self):
        store, s = waiting(background_end_grace=120)
        self.assertNotIn("s1", goal_loop._turn_marks())
        make_post_tool_call(store)(tool_name="terminal", session_id="s1", conversation_id="")
        self.assertIn("s1", goal_loop._turn_marks())
        now = time.time()
        sent = []
        self.assertEqual(tick(store, s, now + 1, sent), [])
        self.assertEqual(tick(store, s, now + 200, sent), [])
        self.assertEqual(sent, [])
        self.assertEqual(store.load(KEY).wait_reason, "waiting_background")

    def test_a_created_mark_older_than_turn_active_max_does_not_count(self):
        store, s = waiting(background_end_grace=120, turn_active_max=600)
        make_post_tool_call(store)(tool_name="terminal", session_id="s1", conversation_id="")
        mark = goal_loop._turn_marks()["s1"]
        sent = []
        self.assertEqual(tick(store, s, mark + 601, sent), [])  # first tick that sees all ended
        self.assertEqual(tick(store, s, mark + 722, sent), [KEY])
        self.assertEqual(len(sent), 1)
        last = [e for e in store.read_ledger(KEY) if e["event"] == "background_watchdog"][-1]
        self.assertEqual(last["why"], "ended")


class NewWaitTimeTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def _old_wait(self, minutes=50):
        store, s = waiting(background_end_grace=0, background_wait_max=3600)
        state = store.load(KEY)
        old = time.time() - minutes * 60
        state.bg_wait = {**state.bg_wait, "procs": ["proc_A"], "since": old}
        store.save(state)
        return store, s, store.load(KEY), old

    def test_a_new_process_starts_a_new_wait_time(self):
        store, s, state, old = self._old_wait()
        before = time.time()
        note(state, "proc_B")
        self.assertGreaterEqual(state.bg_wait["since"], before)
        self.assertEqual(state.bg_wait["procs"], ["proc_B"])
        store.save(state)
        sent = []
        running = lambda p: p == "proc_B"
        done = goal_loop.background_watchdog_tick(store, s, now=before + 20 * 60, running=running,
                                                  inject=lambda text, session_key: sent.append(text) or True)
        self.assertEqual(done, [])
        self.assertEqual(sent, [])
        self.assertFalse([e for e in store.read_ledger(KEY) if e["event"] == "background_watchdog"])

    def test_the_same_set_keeps_the_wait_time(self):
        store, s, state, old = self._old_wait()
        note(state, "proc_A")
        self.assertEqual(state.bg_wait["since"], old)
        self.assertEqual(state.bg_wait["procs"], ["proc_A"])

    def test_a_subset_keeps_the_wait_time(self):
        store, s, state, old = self._old_wait()
        state.bg_wait = {**state.bg_wait, "procs": ["proc_A", "proc_C"]}
        note(state, "proc_C")
        self.assertEqual(state.bg_wait["since"], old)
        self.assertEqual(state.bg_wait["procs"], ["proc_A", "proc_C"])

    def test_a_long_wait_for_a_running_process_still_times_out(self):
        store, s, state, old = self._old_wait(minutes=61)
        note(state, "proc_A")
        store.save(state)
        sent = []
        done = goal_loop.background_watchdog_tick(store, s, now=time.time(), running=lambda p: True,
                                                  inject=lambda text, session_key: sent.append(text) or True)
        self.assertEqual(done, [KEY])
        last = [e for e in store.read_ledger(KEY) if e["event"] == "background_watchdog"][-1]
        self.assertEqual((last["why"], last["continued"]), ("timeout", True))


if __name__ == "__main__":
    unittest.main()
