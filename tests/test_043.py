"""Tests of 0.4.3: boot id per process (F-reload-hold), judge retry, stale pause, wait barriers."""
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import goal_loop
from goal_loop import (
    GoalStore, Settings, _Tools, _chat_key, make_post_tool_call, make_pre_llm_call, make_pre_stop,
    make_transform_llm_output, pause_stale_states,
)
from goal_loop.judge import Verdict, run_judge

IDS = {"session_id": "s1", "platform": "api_server"}
CHECKS = {"checks": [{"command": "c", "result": "passed", "exitCode": 0}]}


class Llm:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        item = self.texts.pop(0) if self.texts else self.texts_last
        if isinstance(item, Exception):
            raise item
        r = types.SimpleNamespace()
        r.text = item
        return r

    texts_last = "VERDICT: done"


class Ctx:
    def __init__(self, texts=("VERDICT: done",), settings=None):
        self.llm = Llm(texts)
        self.subagent_lifecycle = None
        self._settings = settings or {}

    def get_config(self, key, default=None):
        return self._settings.get(key, default)


def setup(texts=("VERDICT: done",), **settings):
    store = GoalStore(tempfile.mkdtemp(prefix="gl43-"))
    ctx = Ctx(texts, settings)
    s = Settings(ctx)
    return _Tools(store, ctx, s), store, ctx, s


def owner(store, text="go", settings=None, session="s1", platform="api_server"):
    return make_pre_llm_call(store, settings)(session_id=session, platform=platform, user_message=text,
                                              conversation_history=[{"role": "user", "content": text}])


def set_and_go(tools, store, goals=(("g1", "a", "c1"),)):
    tools.goal_set({"goals": [{"id": i, "text": t, "check": c} for i, t, c in goals]}, **IDS)
    owner(store, "go", settings=tools.settings)
    assert tools.goal_go({}, **IDS).startswith("Go recorded")


def stop(store, settings, attempt=0):
    return make_pre_stop(store, settings)(session_id="s1", platform="api_server", attempt=attempt)


def tool(store, name="terminal"):
    make_post_tool_call(store)(tool_name=name, session_id="s1", conversation_id="")


def ledger_events(store):
    return [e["event"] for e in store.read_ledger(_chat_key("s1"))]


class BootIdTest(unittest.TestCase):
    """r13: a plugin reload keeps the boot id; a new process gets a new one."""

    def test_reload_of_the_module_keeps_the_boot_id(self):
        # a real importlib.reload in a child process (a reload here would rebind the globals that
        # other test modules imported)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = ("import importlib, sys; sys.path.insert(0, %r); import goal_loop; a = goal_loop.BOOT_ID; "
                "importlib.reload(goal_loop); b = goal_loop.BOOT_ID; importlib.reload(goal_loop); "
                "print(a, b, goal_loop.BOOT_ID)") % (root,)
        out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60)
        ids = out.stdout.split()
        self.assertEqual(len(ids), 3, out.stderr[-300:])
        self.assertEqual(len(set(ids)), 1)

    def test_a_reload_does_not_hold_a_chat_with_a_go(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        # what a reload does: the module computes the boot id again
        with mock.patch.object(goal_loop, "BOOT_ID", goal_loop._process_boot_id()):
            tool(store)
            self.assertEqual(stop(store, s)["action"], "continue")
        self.assertNotIn("restart_hold", ledger_events(store))

    def test_the_old_import_id_would_have_held_the_chat(self):
        # control: with a changed id (the 0.4.2 behaviour after a reload) the chat is held
        tools, store, _, s = setup()
        set_and_go(tools, store)
        tool(store)
        with mock.patch.object(goal_loop, "BOOT_ID", "0" * 32):
            tool(store)
            self.assertIsNone(stop(store, s))
        self.assertIn("restart_hold", ledger_events(store))

    def test_another_process_gets_another_boot_id(self):
        code = "import sys; sys.path.insert(0, %r); import goal_loop; print(goal_loop.BOOT_ID)" % (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),)
        out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60)
        other = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        self.assertTrue(other, out.stderr[-300:])
        self.assertNotEqual(other, goal_loop.BOOT_ID)
        self.assertEqual(len(goal_loop.BOOT_ID), 32)

    def test_without_proc_the_id_survives_a_reload(self):
        real_open = open

        def no_proc(path, *a, **kw):
            if str(path) == "/proc/self/stat":
                raise OSError("no proc")
            return real_open(path, *a, **kw)
        sys.modules.pop("goal_loop_boot", None)
        with mock.patch("builtins.open", no_proc):
            first = goal_loop._process_boot_id()
            second = goal_loop._process_boot_id()
        sys.modules.pop("goal_loop_boot", None)
        self.assertEqual(first, second)


class JudgeRetryTest(unittest.TestCase):
    """r18: retry with backoff and a time limit per attempt before the judge counts as unavailable."""

    def test_first_attempt_fails_second_answers_confirms_and_notes_a_retry(self):
        tools, store, ctx, _ = setup([RuntimeError("timeout"), "VERDICT: done"])
        set_and_go(tools, store)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS)
        self.assertIn("confirmed", out)
        events = ledger_events(store)
        self.assertIn("judge_retry", events)
        self.assertNotIn("judge_unavailable", events)
        self.assertEqual(ctx.llm.calls, 2)
        self.assertTrue(store.load(_chat_key("s1")).goal("g1").done)

    def test_all_attempts_fail_gives_judge_unavailable_exactly_once(self):
        tools, store, ctx, _ = setup([RuntimeError("t1"), RuntimeError("t2"), RuntimeError("t3")])
        set_and_go(tools, store)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS)
        self.assertIn("Judge unavailable", out)
        events = ledger_events(store)
        self.assertEqual(events.count("judge_unavailable"), 1)
        self.assertEqual(events.count("judge_retry"), 2)
        self.assertEqual(ctx.llm.calls, 3)
        goal = store.load(_chat_key("s1")).goal("g1")
        self.assertEqual((goal.done, goal.rejects), (False, 0))

    def test_a_clear_rejection_is_not_tried_again(self):
        tools, store, ctx, _ = setup(["VERDICT: not-done the file is missing"])
        set_and_go(tools, store)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS)
        self.assertIn("rejected", out)
        self.assertEqual(ctx.llm.calls, 1)
        self.assertNotIn("judge_retry", ledger_events(store))

    def test_attempts_one_keeps_the_old_behaviour(self):
        tools, store, ctx, _ = setup([RuntimeError("t1"), "VERDICT: done"], judge_attempts=1)
        set_and_go(tools, store)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **IDS)
        self.assertIn("Judge unavailable", out)
        self.assertEqual(ctx.llm.calls, 1)

    def test_a_retry_gets_a_shortened_claim(self):
        seen = []

        class SpyLlm(Llm):
            def complete(self, messages, **kwargs):
                seen.append(len(messages[-1]["content"]))
                return super().complete(messages, **kwargs)
        ctx = Ctx([RuntimeError("t"), "VERDICT: done"])
        ctx.llm = SpyLlm([RuntimeError("t"), "VERDICT: done"])
        v = run_judge(ctx, "g", "c", "x" * 20000, route={}, attempts=2, backoff=0)
        self.assertTrue(v.done)
        self.assertLess(seen[1], seen[0])

    def test_total_budget_stops_further_attempts(self):
        ctx = Ctx([RuntimeError("t")] * 5)
        v = run_judge(ctx, "g", "c", "e", route={}, attempts=5, timeout=1, backoff=0)
        self.assertTrue(v.unavailable)
        self.assertLessEqual(ctx.llm.calls, 1)  # less than the minimum attempt time is left

    def test_backoff_is_waited_between_attempts(self):
        waits = []
        ctx = Ctx([RuntimeError("t"), RuntimeError("t"), "VERDICT: done"])
        v = run_judge(ctx, "g", "c", "e", route={}, attempts=3, timeout=300, backoff=5, sleep=waits.append)
        self.assertTrue(v.done)
        self.assertEqual(waits, [5.0, 10.0])

    def test_settings_are_in_defaults_readme_and_yaml(self):
        for key in ("judge_attempts", "judge_attempt_timeout", "judge_backoff", "stale_days"):
            self.assertIn(key, goal_loop.DEFAULTS)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
        for key in ("judge_attempts", "judge_attempt_timeout", "judge_backoff", "stale_days"):
            self.assertIn(key, readme)
        yaml = open(os.path.join(root, "goal_loop", "plugin.yaml"), encoding="utf-8").read()
        self.assertRegex(yaml, r"version: 0\.4\.[3-9]")


class StalePauseTest(unittest.TestCase):
    """r21: an old state with a go and open goals is paused at plugin start, not deleted."""

    def _make(self, age_days):
        tools, store, _, s = setup(stale_days=3)
        set_and_go(tools, store)
        path = store._state_path(_chat_key("s1"))
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
        return tools, store, s

    def test_an_old_state_is_paused_with_a_ledger_event(self):
        tools, store, s = self._make(5)
        self.assertEqual(pause_stale_states(store, s), [_chat_key("s1")])
        state = store.load(_chat_key("s1"))
        self.assertFalse(state.go)
        self.assertEqual(state.goal("g1").status, "paused")
        self.assertEqual(ledger_events(store)[-1], "paused_stale")
        self.assertTrue(state.decisions)

    def test_an_old_state_with_only_blocked_goals_is_paused(self):
        tools, store, _, s = setup(stale_days=3)
        set_and_go(tools, store)
        state = store.load(_chat_key("s1"))
        state.goal("g1").status = "blocked"
        store.save(state)
        old = time.time() - 4 * 86400
        os.utime(store._state_path(_chat_key("s1")), (old, old))
        self.assertEqual(pause_stale_states(store, s), [_chat_key("s1")])
        state = store.load(_chat_key("s1"))
        self.assertFalse(state.go)
        self.assertEqual(state.goal("g1").status, "paused")

    def test_a_fresh_state_stays(self):
        tools, store, s = self._make(1)
        self.assertEqual(pause_stale_states(store, s), [])
        self.assertTrue(store.load(_chat_key("s1")).go)

    def test_a_state_without_go_or_with_all_done_stays(self):
        tools, store, _, s = setup(stale_days=1)
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)  # no go
        path = store._state_path(_chat_key("s1"))
        old = time.time() - 9 * 86400
        os.utime(path, (old, old))
        self.assertEqual(pause_stale_states(store, s), [])

    def test_stale_days_zero_is_off(self):
        tools, store, s = self._make(30)
        self.assertEqual(pause_stale_states(store, Settings(overrides={"stale_days": 0})), [])

    def test_a_new_go_continues_a_paused_stale_state(self):
        tools, store, s = self._make(5)
        pause_stale_states(store, s)
        self.assertEqual(tools.goal_go({}, **IDS), goal_loop.GO_REFUSED)  # needs a new owner message
        owner(store, "go", settings=s)
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))
        state = store.load(_chat_key("s1"))
        self.assertTrue(state.go)
        self.assertEqual(state.goal("g1").status, "in_work")

    def test_the_decision_text_is_in_the_owner_language(self):
        tools, store, s = self._make(5)
        state = store.load(_chat_key("s1"))
        state.language = "de"
        store.save(state)
        old = time.time() - 5 * 86400
        os.utime(store._state_path(_chat_key("s1")), (old, old))
        pause_stale_states(store, Settings(overrides={"stale_days": 3, "language": "de"}))
        self.assertIn("ohne Aktivität", store.load(_chat_key("s1")).decisions[-1])


def fake_modules(background=False, delegation=False, builtin=False):
    """Hermetic stand-ins for the Hermes modules that the wait barriers read."""
    proc = types.SimpleNamespace(id="proc_1", parent_session_id="s1", exited=False, notify_on_complete=True,
                                 watch_patterns=[])
    registry = types.SimpleNamespace(
        _lock=__import__("threading").Lock(),
        _running={"proc_1": proc} if background else {},
        is_session_waiting=lambda sid: sid == "proc_1" and background)
    mods = {
        "tools": types.ModuleType("tools"),
        "tools.process_registry": types.SimpleNamespace(process_registry=registry),
        "tools.async_delegation": types.SimpleNamespace(
            _LIVE_STATES={"running"},
            _session_records=lambda statuses, key, ui, parent: [{"status": "running"}] if (delegation and parent == "s1") else []),
        "hermes_cli": types.ModuleType("hermes_cli"),
        "hermes_cli.goals": types.SimpleNamespace(
            load_goal=lambda sid: types.SimpleNamespace(status="active") if (builtin and sid == "s1") else None),
    }
    return mock.patch.dict(sys.modules, mods)


class WaitBarrierTest(unittest.TestCase):
    """r25: the loop does not push while Hermes work or the owner is awaited."""

    def _ready(self):
        tools, store, _, s = setup(idle_pause_after=0)
        set_and_go(tools, store)
        tool(store)
        return tools, store, s

    def test_without_a_barrier_the_loop_continues(self):
        tools, store, s = self._ready()
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")

    def test_a_running_background_process_gives_no_continuation(self):
        tools, store, s = self._ready()
        with fake_modules(background=True):
            self.assertIsNone(stop(store, s))
            self.assertIsNone(stop(store, s))
        events = ledger_events(store)
        self.assertEqual(events.count("waiting_background"), 1)  # one line per change, not per turn
        self.assertNotIn("continued", events)
        self.assertEqual(store.load(_chat_key("s1")).nudge_seq, 0)

    def test_a_live_delegation_gives_no_continuation(self):
        tools, store, s = self._ready()
        with fake_modules(delegation=True):
            self.assertIsNone(stop(store, s))
        self.assertIn("waiting_delegation", ledger_events(store))

    def test_an_active_builtin_goal_makes_the_plugin_yield(self):
        tools, store, s = self._ready()
        with fake_modules(builtin=True):
            self.assertIsNone(stop(store, s))
        self.assertIn("yield_builtin_goal", ledger_events(store))

    def test_all_open_goals_waiting_for_the_owner_give_no_continuation(self):
        tools, store, s = self._ready()
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")  # round 1: nudge_seq 1
            tools.goal_decision({"text": "A/B/C, recommendation A"}, **IDS)
            tools.goal_block({"goal_id": "g1", "reason": "owner decision missing"}, **IDS)
            tool(store, "goal_block")
            self.assertIsNone(stop(store, s))
        self.assertIn("waiting_owner", ledger_events(store))

    def test_decision_and_block_in_different_rounds_still_wait_for_the_owner(self):
        tools, store, s = self._ready()
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")
            tools.goal_decision({"text": "A/B/C"}, **IDS)
            tool(store, "goal_decision")
            self.assertEqual(stop(store, s)["action"], "continue")  # goal not blocked yet
            tool(store, "terminal")
            self.assertEqual(stop(store, s)["action"], "continue")
            tools.goal_block({"goal_id": "g1", "reason": "owner decision missing"}, **IDS)
            tool(store, "goal_block")
            self.assertIsNone(stop(store, s))
        self.assertIn("waiting_owner", ledger_events(store))

    def test_an_old_decision_before_the_go_is_no_owner_wait(self):
        tools, store, _, s = setup(idle_pause_after=0)
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        tools.goal_decision({"text": "old"}, **IDS)
        owner(store, "go", settings=s)
        tools.goal_go({}, **IDS)
        tool(store)
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")
            tools.goal_block({"goal_id": "g1", "reason": "tried"}, **IDS)
            tool(store, "goal_block")
            self.assertEqual(stop(store, s)["action"], "continue")

    def test_a_goal_block_without_a_decision_is_no_owner_wait(self):
        tools, store, s = self._ready()
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")
            tools.goal_block({"goal_id": "g1", "reason": "tried a thing"}, **IDS)
            tool(store, "goal_block")
            self.assertEqual(stop(store, s)["action"], "continue")

    def test_one_open_goal_that_does_not_wait_keeps_the_loop_going(self):
        tools, store, _, s = setup(idle_pause_after=0)
        set_and_go(tools, store, (("g1", "a", "c"), ("g2", "b", "c")))
        tool(store)
        with fake_modules():
            self.assertEqual(stop(store, s)["action"], "continue")
            tools.goal_decision({"text": "d"}, **IDS)
            tools.goal_block({"goal_id": "g1", "reason": "owner decision missing"}, **IDS)
            tool(store, "goal_block")
            self.assertEqual(stop(store, s)["action"], "continue")  # g2 is not blocked

    def test_the_owner_message_lifts_the_wait(self):
        tools, store, s = self._ready()
        with fake_modules(background=True):
            self.assertIsNone(stop(store, s))
            owner(store, "weiter", settings=s)
            self.assertEqual(store.load(_chat_key("s1")).wait_reason, "")
        with fake_modules():
            tool(store)
            self.assertEqual(stop(store, s)["action"], "continue")

    def test_the_end_of_the_background_work_lets_the_loop_continue(self):
        tools, store, s = self._ready()
        with fake_modules(background=True):
            self.assertIsNone(stop(store, s))
        with fake_modules():
            tool(store)
            self.assertEqual(stop(store, s)["action"], "continue")
        events = ledger_events(store)
        self.assertEqual(events.count("continued"), 1)

    def test_a_missing_hermes_module_does_not_block_the_loop(self):
        tools, store, s = self._ready()
        with mock.patch.dict(sys.modules, {"tools.process_registry": None, "tools.async_delegation": None,
                                          "hermes_cli.goals": None}):
            self.assertEqual(stop(store, s)["action"], "continue")

    def test_the_display_shows_a_wait_line_and_no_follow_up(self):
        tools, store, s = self._ready()
        sent = []
        goal_loop._CTX = types.SimpleNamespace(inject_message=lambda text: sent.append(text) or True)
        try:
            with fake_modules(background=True):
                out = make_transform_llm_output(store, s)(response_text="working", session_id="s1",
                                                          platform="local")
        finally:
            goal_loop._CTX = None
        self.assertIsNotNone(out)
        self.assertIn("waits", out)
        self.assertEqual(sent, [])
        self.assertNotIn("resume_injected", ledger_events(store))

    def test_inject_follow_up_runs_when_nothing_is_awaited(self):
        tools, store, s = self._ready()
        sent = []
        goal_loop._CTX = types.SimpleNamespace(inject_message=lambda text: sent.append(text) or True)
        try:
            with fake_modules():
                out = make_transform_llm_output(store, s)(response_text="working", session_id="s1",
                                                          platform="local")
        finally:
            goal_loop._CTX = None
        self.assertEqual(len(sent), 1)
        self.assertIn("continues in this chat", out)


if __name__ == "__main__":
    unittest.main()
