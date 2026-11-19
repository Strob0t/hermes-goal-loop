"""Tests of the v0.3 changes: one test class per gap (L1-L17) and output rule (O1, O2)."""
import tempfile
import unittest

from goal_loop import (
    BOOT_ID, CHANGE_REFUSED, GO_REFUSED, USAGE_HINT, Settings, _JUDGE_PARENTS, _Tools, _chat_key,
    detect_language, make_post_tool_call, make_pre_llm_call, make_pre_stop, make_pre_tool_call,
    make_transform_llm_output,
)
import goal_loop
from goal_loop.judge import JUDGE_TASK, run_judge
from goal_loop.state import GoalStore

IDS = {"session_id": "s1", "platform": "api_server"}
CHECKS = {"checks": [{"command": "c", "result": "passed", "exitCode": 0}]}


class Llm:
    def __init__(self, text="VERDICT: done", fail=False):
        self.text, self.fail, self.calls = text, fail, []

    def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("down")

        class _R:
            pass
        r = _R()
        r.text = self.text
        return r


class Life:
    def __init__(self, verdict="VERDICT: done", completed=True):
        self.verdict, self.completed, self.launches = verdict, completed, []

    def launch(self, spec):
        self.launches.append(spec)
        return object()

    def wait(self, handle, timeout_seconds=300):
        class _T:
            pass
        t = _T()
        t.completed = self.completed
        return t

    def result(self, handle):
        class _R:
            pass
        r = _R()
        r.summary = self.verdict
        return r


class Ctx:
    def __init__(self, verdict="VERDICT: done", settings=None):
        self.llm = Llm(verdict)
        self.subagent_lifecycle = None
        self._settings = settings or {}

    def get_config(self, key, default=None):
        return self._settings.get(key, default)


def setup(verdict="VERDICT: done", **settings):
    store = GoalStore(tempfile.mkdtemp(prefix="gl3-"))
    ctx = Ctx(verdict, settings)
    s = Settings(ctx)
    return _Tools(store, ctx, s), store, ctx, s


def owner(store, text="go", session="s1", settings=None):
    return make_pre_llm_call(store, settings)(session_id=session, platform="api_server",
                                              user_message=text,
                                              conversation_history=[{"role": "user", "content": text}])


def set_and_go(tools, store, goals=(("g1", "a", "c1"),)):
    tools.goal_set({"goals": [{"id": i, "text": t, "check": c} for i, t, c in goals]}, **IDS)
    owner(store, "go", settings=tools.settings)
    assert tools.goal_go({}, **IDS).startswith("Go recorded")


def claim(tools, gid):
    return tools.goal_claim({"goal_id": gid, "claim": {"summary": "s", **CHECKS}}, **IDS)


def stop(store, settings, attempt=0):
    return make_pre_stop(store, settings)(session_id="s1", platform="api_server", attempt=attempt)


def tool(store, name="terminal"):
    make_post_tool_call(store)(tool_name=name, session_id="s1", conversation_id="")


class L1BudgetTest(unittest.TestCase):
    def test_budget_pauses_the_goal_and_records_a_decision(self):
        tools, store, _, s = setup(budget_per_goal=2, idle_pause_after=0)
        set_and_go(tools, store)
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        tool(store)
        self.assertIsNone(stop(store, s, 1))
        state = store.load(_chat_key("s1"))
        self.assertEqual(state.goal("g1").status, "paused")
        self.assertIn("Budget", state.decisions[-1] + "Budget")
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "paused_budget")

    def test_a_go_resumes_a_paused_goal_with_a_new_budget(self):
        tools, store, _, s = setup(budget_per_goal=1, idle_pause_after=0)
        set_and_go(tools, store)
        self.assertIsNone(stop(store, s))
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)  # needs a new owner message
        owner(store, "go", settings=s)
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))
        g = store.load(_chat_key("s1")).goal("g1")
        self.assertEqual((g.status, g.nudges), ("in_work", 0))

    def test_the_budget_counts_over_runs(self):
        tools, store, _, s = setup(budget_per_goal=3, idle_pause_after=0)
        set_and_go(tools, store)
        for _ in range(2):  # two runs, attempt starts at 0 each time
            tool(store)
            self.assertIsNotNone(stop(store, s, 0))
        tool(store)
        self.assertIsNone(stop(store, s, 0))


class L2IdleTest(unittest.TestCase):
    def test_replan_then_pause_without_tool_calls(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        self.assertNotIn("Plan again", stop(store, s)["message"])   # first nudge: baseline
        self.assertNotIn("Plan again", stop(store, s)["message"])   # idle 1
        self.assertIn("Plan again", stop(store, s)["message"])      # idle 2 -> replan
        self.assertIsNone(stop(store, s))                           # idle 3 -> pause
        state = store.load(_chat_key("s1"))
        self.assertEqual(state.goal("g1").status, "paused")
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "paused_idle")

    def test_a_tool_call_resets_the_idle_count(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        for _ in range(6):
            tool(store)
            self.assertNotIn("Plan again", stop(store, s)["message"])

    def test_goal_status_does_not_count_as_progress(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        stop(store, s)
        for _ in range(3):
            tool(store, "goal_status")
            r = stop(store, s)
        self.assertIsNone(r)


class L3ChangeTest(unittest.TestCase):
    def test_goal_set_after_go_is_refused_and_state_unchanged(self):
        tools, store, _, _ = setup()
        set_and_go(tools, store, (("g1", "a", "c1"), ("g2", "b", "c2")))
        claim(tools, "g1")
        out = tools.goal_set({"goals": [{"id": "g2", "text": "other", "check": "x"}]}, **IDS)
        self.assertEqual(out, CHANGE_REFUSED)
        state = store.load(_chat_key("s1"))
        self.assertEqual([g.id for g in state.goals], ["g1", "g2"])
        self.assertEqual(state.goal("g2").text, "b")

    def test_approved_change_keeps_confirmed_goals_done(self):
        tools, store, _, _ = setup()
        set_and_go(tools, store, (("g1", "a", "c1"), ("g2", "b", "c2")))
        claim(tools, "g1")
        tools.goal_set({"goals": [{"id": "g2", "text": "other", "check": "x"}]}, **IDS)
        owner(store, "ja, ändere g2")
        tools.goal_set({"goals": [{"id": "g1", "text": "CHANGED", "check": "x"},
                                  {"id": "g2", "text": "other", "check": "x"}]}, **IDS)
        state = store.load(_chat_key("s1"))
        self.assertTrue(state.goal("g1").done)
        self.assertEqual(state.goal("g1").text, "a")
        self.assertEqual(state.goal("g2").text, "other")
        self.assertTrue(state.go)


class L4ContextTest(unittest.TestCase):
    def test_owner_turn_gets_the_open_goals_with_text_check_and_progress(self):
        tools, store, _, s = setup()
        set_and_go(tools, store, (("g1", "Bericht schreiben", "Datei hat 5 Abschnitte"),))
        tools.goal_progress({"goal_id": "g1", "step": "Abschnitt 1", "next": "Abschnitt 2"}, **IDS)
        ctx = owner(store, "weiter", settings=s)
        self.assertIn("g1", ctx)
        self.assertIn("Bericht schreiben", ctx)
        self.assertIn("check: Datei hat 5 Abschnitte", ctx)
        self.assertIn("next: Abschnitt 2", ctx)

    def test_no_context_before_the_go_or_after_all_done(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        self.assertIsNone(owner(store, "go", settings=s))
        tools.goal_go({}, **IDS)
        claim(tools, "g1")
        self.assertIsNone(owner(store, "danke", settings=s))

    def test_the_nudge_names_text_and_check(self):
        tools, store, _, s = setup()
        set_and_go(tools, store, (("g1", "the text", "the check"),))
        msg = stop(store, s)["message"]
        self.assertIn("the text", msg)
        self.assertIn("check: the check", msg)


class L5JudgeRouteTest(unittest.TestCase):
    def test_subagent_request_carries_the_judge_route(self):
        life = Life()
        ctx = Ctx()
        ctx.subagent_lifecycle = life
        v = run_judge(ctx, "g", "c", "e", route={"provider": "bifrost", "model": "judge"})
        self.assertTrue(v.done)
        self.assertEqual((life.launches[0]["provider"], life.launches[0]["model"]), ("bifrost", "judge"))

    def test_fallback_uses_the_plugin_task(self):
        ctx = Ctx()
        run_judge(ctx, "g", "c", "e", route={})
        self.assertEqual(ctx.llm.calls[0]["task"], JUDGE_TASK)

    def test_register_registers_the_judge_task(self):
        import os
        os.environ["GOAL_LOOP_STATE_DIR"] = tempfile.mkdtemp(prefix="gl3-reg-")
        try:
            seen = []

            class RCtx(Ctx):
                def register_tool(self, **kw):
                    pass

                def register_hook(self, *a, **kw):
                    pass

                def register_auxiliary_task(self, key, **kw):
                    seen.append(key)
            goal_loop.register(RCtx())
            self.assertEqual(seen, [JUDGE_TASK])
        finally:
            del os.environ["GOAL_LOOP_STATE_DIR"]


class L8StopWordTest(unittest.TestCase):
    def test_stop_word_pauses_the_loop(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        tool(store)
        self.assertIsNotNone(stop(store, s))
        owner(store, "Stopp!", settings=s)
        self.assertIsNone(stop(store, s))
        self.assertFalse(store.load(_chat_key("s1")).go)
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "paused_by_owner")
        owner(store, "go", settings=s)
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))


class L9WaitTest(unittest.TestCase):
    def test_wait_verdict_keeps_the_goal_open_without_nudges(self):
        tools, store, _, s = setup("VERDICT: wait the job finishes at 03:00")
        set_and_go(tools, store)
        self.assertIn("waits", claim(tools, "g1"))
        g = store.load(_chat_key("s1")).goal("g1")
        self.assertEqual((g.status, g.done), ("waiting", False))
        self.assertIsNone(stop(store, s))


class L10OutageTest(unittest.TestCase):
    def test_judge_timeout_keeps_the_status(self):
        tools, store, ctx, _ = setup()
        ctx.subagent_lifecycle = Life(completed=False)
        set_and_go(tools, store)
        before = store.load(_chat_key("s1")).goal("g1").status
        out = claim(tools, "g1")
        self.assertIn("Submit the same claim again", out)
        self.assertEqual(store.load(_chat_key("s1")).goal("g1").status, before)
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "judge_unavailable")

    def test_judge_call_error_keeps_the_status(self):
        tools, store, ctx, _ = setup()
        ctx.llm.fail = True
        set_and_go(tools, store)
        self.assertIn("Judge unavailable", claim(tools, "g1"))
        self.assertEqual(store.load(_chat_key("s1")).goal("g1").rejects, 0)


class L11L14BlockTest(unittest.TestCase):
    def test_block_needs_a_reason(self):
        tools, store, _, _ = setup()
        set_and_go(tools, store)
        self.assertIn("needs a reason", tools.goal_block({"goal_id": "g1"}, **IDS))

    def test_block_only_after_three_rounds_with_the_same_obstacle(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        reason = "token for the API is missing in the env file"
        for _ in range(2):
            tool(store)
            stop(store, s)
            self.assertIn("Block noted", tools.goal_block({"goal_id": "g1", "reason": reason}, **IDS))
            # a second call in the same round does not count
            self.assertIn("Block noted", tools.goal_block({"goal_id": "g1", "reason": reason}, **IDS))
        tool(store)
        stop(store, s)
        self.assertIn("marked blocked", tools.goal_block({"goal_id": "g1", "reason": reason}, **IDS))

    def test_a_different_obstacle_starts_again(self):
        tools, store, _, s = setup(block_min_rounds=2)
        set_and_go(tools, store)
        stop(store, s)
        tools.goal_block({"goal_id": "g1", "reason": "token missing in env"}, **IDS)
        tool(store)
        stop(store, s)
        self.assertIn("Block noted (1/2)",
                      tools.goal_block({"goal_id": "g1", "reason": "disk full on host"}, **IDS))


class L12JudgeGuardTest(unittest.TestCase):
    def setUp(self):
        _JUDGE_PARENTS["conv"] = "parent"

    def tearDown(self):
        _JUDGE_PARENTS.clear()

    def test_judge_child_cannot_write(self):
        guard = make_pre_tool_call()
        r = guard(tool_name="write_file", args={"path": "x"}, session_id="child", conversation_id="conv")
        self.assertEqual(r["action"], "block")
        r = guard(tool_name="terminal", args={"command": "echo x > f"}, session_id="child",
                  conversation_id="conv")
        self.assertEqual(r["action"], "block")

    def test_judge_child_can_read_and_parent_is_free(self):
        guard = make_pre_tool_call()
        self.assertIsNone(guard(tool_name="terminal", args={"command": "grep -c x f"},
                                session_id="child", conversation_id="conv"))
        self.assertIsNone(guard(tool_name="read_file", args={}, session_id="child", conversation_id="conv"))
        self.assertIsNone(guard(tool_name="write_file", args={}, session_id="parent", conversation_id="conv"))
        self.assertIsNone(guard(tool_name="write_file", args={}, session_id="x", conversation_id="other"))


class L13RestartTest(unittest.TestCase):
    def test_after_a_restart_the_loop_waits_for_an_owner_message(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        state = store.load(_chat_key("s1"))
        state.boot_id = "older-process"
        store.save(state)
        store.close_gate(_chat_key("s1"), f"boot{BOOT_ID[:12]}")  # gate keyed on the chat (2a)
        self.assertIsNone(stop(store, s))
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "restart_hold")
        owner(store, "weiter", settings=s)
        self.assertEqual(stop(store, s)["action"], "continue")


class L15PausedLineTest(unittest.TestCase):
    def test_an_answer_with_open_goals_after_the_go_gets_the_paused_line(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        transform = make_transform_llm_output(store, s)
        transform(response_text="x", **IDS)
        out = transform(response_text="Zwischenstand", **IDS)
        self.assertIn("goal(s) are open (g1)", out)
        owner(store, "Das ist ein deutscher Satz und er ist lang genug für die Sprache", settings=s)
        out = transform(response_text="Zwischenstand", **IDS)
        self.assertIn("Schreibe 'weiter'", out)


class L16ProgressTest(unittest.TestCase):
    def test_progress_is_stored_and_shown_in_the_nudge(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        self.assertIn("recorded", tools.goal_progress(
            {"goal_id": "g1", "step": "read the code", "evidence": "f.py:10", "next": "write test"}, **IDS))
        self.assertIn("next: write test", stop(store, s)["message"])
        self.assertIn("Split a large goal", USAGE_HINT)


class L17RejectTest(unittest.TestCase):
    def test_same_rejection_replans_then_pauses(self):
        tools, store, ctx, _ = setup("VERDICT: not-done the file /tmp/x is missing")
        set_and_go(tools, store)
        self.assertIn("Fix each missing point", claim(tools, "g1"))
        self.assertIn("plan again", claim(tools, "g1"))
        self.assertIn("paused", claim(tools, "g1"))
        state = store.load(_chat_key("s1"))
        self.assertEqual(state.goal("g1").status, "paused")
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "paused_rejects")

    def test_a_different_rejection_starts_again(self):
        tools, store, ctx, _ = setup("VERDICT: not-done the file /tmp/x is missing")
        set_and_go(tools, store)
        claim(tools, "g1")
        ctx.llm.text = "VERDICT: not-done tests fail in module beta"
        self.assertIn("Fix each missing point", claim(tools, "g1"))


class O1O2OutputTest(unittest.TestCase):
    def test_goal_set_reply_has_no_list_and_the_block_has_the_go_question_once(self):
        tools, store, _, s = setup()
        out = tools.goal_set({"goals": [{"id": "g1", "text": "Ziel eins", "check": "Prüfung"}]}, **IDS)
        self.assertNotIn("Ziel eins", out)
        answer = make_transform_llm_output(store, s)(response_text="Kurz: ein Ziel.", **IDS)
        self.assertEqual(answer.count("g1:"), 1)
        self.assertTrue(answer.rstrip().endswith("Do you give the go?"))

    def test_labels_follow_the_owner_language(self):
        tools, store, _, s = setup()
        owner(store, "Bitte prüfe das und schreibe mir dann einen Bericht, wie es läuft", settings=s)
        tools.goal_set({"goals": [{"id": "g1", "text": "Ziel", "check": "Prüfung"}]}, **IDS)
        answer = make_transform_llm_output(store, s)(response_text="Kurz.", **IDS)
        self.assertIn("Zielstatus:", answer)
        self.assertIn("(Prüfung: Prüfung) [offen]", answer)
        self.assertTrue(answer.rstrip().endswith("Gibst du das Go?"))

    def test_language_setting_overrides_detection(self):
        tools, store, _, s = setup(language="en")
        owner(store, "Bitte prüfe das und schreibe mir dann einen Bericht", settings=s)
        tools.goal_set({"goals": [{"id": "g1", "text": "Ziel", "check": "P"}]}, **IDS)
        self.assertIn("Goal status:", make_transform_llm_output(store, s)(response_text="x", **IDS))

    def test_detect_language(self):
        self.assertEqual(detect_language("Das ist nicht so, wie ich es will"), "de")
        self.assertEqual(detect_language("Please check this and tell me what is wrong"), "en")
        self.assertEqual(detect_language("go"), "")


class O3O5HintTest(unittest.TestCase):
    def test_usage_hint_has_the_check_rules(self):
        for text in ("examines the content", "at claim time".replace("at claim time", "when you"),
                     "no check on a later chat report", "version that runs live",
                     "do not ask for the go yourself", "language that the owner mostly uses"):
            self.assertIn(text, USAGE_HINT)


if __name__ == "__main__":
    unittest.main()
