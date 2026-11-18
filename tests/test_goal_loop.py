"""Tests of the goal-loop plugin, one or more per owner goal.

These tests are self-contained: they use a fake ctx and a temp state dir, so they run with plain
``python -m unittest`` without a Hermes install. The companion fork patch (the generic ``pre_stop``
hook) is tested in the fork; here the hook callback is called directly.
"""
import os
import tempfile
import unittest

from goal_loop import (
    FINAL_REPORT_NUDGE, NO_REPORT_MARKER, STOP_CAUSES, _Tools, _chat_key, make_pre_stop,
    make_transform_llm_output, register,
)
from goal_loop.judge import JUDGE_TOOLSETS, run_judge
from goal_loop.claim import ClaimError, serialize_completion_claim
from goal_loop.state import Goal, GoalState, GoalStore, status_block


class FakeLlm:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))

        class _R:
            text = self._text
        return _R()


class FakeCtx:
    """A ctx with only ctx.llm, so the judge takes the tool-less fallback path."""

    def __init__(self, verdict_text="VERDICT: done"):
        self.llm = FakeLlm(verdict_text)
        self.subagent_lifecycle = None  # force the ctx.llm path
        self.tools = []
        self.hooks = {}

    def register_tool(self, name, toolset, schema, handler, emoji="", description="", **kw):
        self.tools.append((name, toolset, handler))

    def register_hook(self, hook_name, callback, **kw):
        self.hooks.setdefault(hook_name, []).append(callback)



class _FakeHandle:
    pass


class FakeSubagentLifecycle:
    """Records each launch spec so a test can assert the requested toolsets."""

    def __init__(self, verdict="VERDICT: done"):
        self.launches = []
        self._verdict = verdict

    def launch(self, spec):
        self.launches.append(spec)
        return _FakeHandle()

    def wait(self, handle, timeout_seconds=300):
        class _T:
            completed = True
        return _T()

    def result(self, handle):
        verdict = self._verdict
        class _R:
            summary = verdict
        return _R()


def make_tools(verdict_text="VERDICT: done"):
    tmp = tempfile.mkdtemp(prefix="goal-loop-test-")
    store = GoalStore(tmp)
    ctx = FakeCtx(verdict_text)
    return _Tools(store, ctx), store, ctx


IDS = {"session_id": "s1", "platform": "api_server"}


class Goal1BeforeWorkTest(unittest.TestCase):
    """Goal 1: detect goals, show the list with checks, ask for the go, no work before the go."""

    def test_goal_set_records_goals_with_checks_and_withholds_go(self):
        tools, store, _ = make_tools()
        out = tools.goal_set({"goals": [
            {"id": "g1", "text": "branch exists", "check": "git rev-parse g1"},
            {"id": "g2", "text": "tests pass", "check": "pytest -q"},
        ]}, **IDS)
        self.assertIn("g1", out)
        self.assertIn("check: git rev-parse g1", out)
        self.assertIn("No loop runs before goal_go", out)
        state = store.load(_chat_key("s1"))
        self.assertEqual([g.id for g in state.goals], ["g1", "g2"])
        self.assertFalse(state.go)

    def test_each_goal_needs_a_check(self):
        tools, _, _ = make_tools()
        out = tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": ""}]}, **IDS)
        self.assertIn("needs id, text and check", out)

    def test_pre_stop_does_not_fire_before_the_go(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        hook = make_pre_stop(store)
        self.assertIsNone(hook(session_id="s1", platform="api_server", attempt=0))


class F3ConversationKeyTest(unittest.TestCase):
    """F3: state keys on the compression-stable conversation_id and survives session rotation."""

    def test_tool_keys_on_conversation_id_and_hook_reads_it_after_rotation(self):
        tools, store, _ = make_tools()
        # a tool call as the dispatcher sends it, before a compaction (session A)
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="A", conversation_id="C")
        tools.goal_go({}, session_id="A", conversation_id="C")
        hook = make_pre_stop(store)
        # after a compaction the session id is B; the conversation root C is unchanged
        result = hook(session_id="B", platform="local", attempt=0, conversation_id="C")
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "continue")
        self.assertIn("g1", result["message"])

    def test_hook_without_conversation_id_falls_back_to_session_id(self):
        # an unpatched fork passes no conversation_id: the hook then reads the session-keyed state
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="A")  # no conversation_id -> keyed on A
        tools.goal_go({}, session_id="A")
        hook = make_pre_stop(store)
        result = hook(session_id="A", platform="local", attempt=0)  # no conversation_id
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "continue")
        # a different session id with no conversation_id finds nothing
        self.assertIsNone(hook(session_id="B", platform="local", attempt=0))

    def test_transform_reads_state_via_conversation_id(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="A", conversation_id="C")
        transform = make_transform_llm_output(store)
        out = transform(response_text="REPORT", session_id="B", conversation_id="C")
        self.assertIn("REPORT", out)
        self.assertIn("Goal status:", out)
        # without the conversation id the rotated session finds no state
        self.assertIsNone(transform(response_text="REPORT", session_id="B"))


class Goal2ToTheEndTest(unittest.TestCase):
    """Goal 2: work to the end; state survives compaction (keyed by chat, not session id)."""

    def test_pre_stop_continues_while_open_goals_remain(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        tools.goal_go({}, **IDS)
        hook = make_pre_stop(store)
        result = hook(session_id="s1", platform="api_server", attempt=0)
        self.assertEqual(result["action"], "continue")
        self.assertIn("g1", result["message"])
        for cause in STOP_CAUSES:
            self.assertIn(cause, result["message"])

    def test_pre_stop_stops_when_all_goals_done(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        tools.goal_go({}, **IDS)
        state = store.load(_chat_key("s1"))
        state.goals[0].done = True
        store.save(state)
        hook = make_pre_stop(store)
        self.assertIsNone(hook(session_id="s1", platform="api_server", attempt=0))

    def test_cron_sessions_are_excluded(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1", platform="cron")
        tools.goal_go({}, session_id="s1", platform="cron")
        hook = make_pre_stop(store)
        self.assertIsNone(hook(session_id="s1", platform="cron", attempt=0))

    def test_pre_stop_reads_the_state_a_tool_wrote_despite_platform(self):
        # F2: tools get no platform kwarg (key off session_id only); the hook gets agent.platform.
        # Both must land on one key, or the loop never sees the goals the tool stored.
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1")            # tool call: no platform, as the dispatcher sends it
        tools.goal_go({}, session_id="s1")
        hook = make_pre_stop(store)
        result = hook(session_id="s1", platform="local", attempt=0)  # hook: agent.platform='local'
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "continue")
        self.assertIn("g1", result["message"])

    def test_state_survives_a_new_session_id_for_the_same_chat(self):
        # The chat key is platform-free (F2): the dispatcher gives tools no platform, so a
        # tool and the pre_stop hook must agree on one key. We assert the state reloads by it.
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        tools.goal_go({}, **IDS)
        reloaded = store.load(_chat_key("s1"))
        self.assertTrue(reloaded.go)
        self.assertEqual(reloaded.goals[0].id, "g1")


class F5FinalReportTest(unittest.TestCase):
    """F5: confirming the last open goal asks the model for the owner report."""

    def test_last_goal_claim_requests_the_final_report(self):
        tools, _, _ = make_tools("VERDICT: done")
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "done",
            "checks": [{"command": "c", "result": "passed", "exitCode": 0}]}}, **IDS)
        self.assertIn("confirmed by the judge", out)
        self.assertIn(FINAL_REPORT_NUDGE.strip(), out)

    def test_earlier_goal_claim_does_not_request_the_report(self):
        tools, _, _ = make_tools("VERDICT: done")
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"},
                                  {"id": "g2", "text": "y", "check": "d"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "done",
            "checks": [{"command": "c", "result": "passed", "exitCode": 0}]}}, **IDS)
        self.assertIn("confirmed by the judge", out)
        self.assertNotIn(FINAL_REPORT_NUDGE.strip(), out)  # g2 still open

    def test_transform_marks_an_empty_final_turn(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        transform = make_transform_llm_output(store)
        # the live field is response_text; empty response_text -> the marker shows
        out = transform(response_text="   ", session_id="s1", platform="api_server")
        self.assertIn(NO_REPORT_MARKER, out)

    def test_transform_keeps_the_models_real_answer(self):
        # F6: the model's text arrives as response_text and must be preserved, not overwritten
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        transform = make_transform_llm_output(store)
        out = transform(response_text="REPORT: g1 done", session_id="s1", platform="api_server")
        self.assertTrue(out.startswith("REPORT: g1 done"))
        self.assertIn("Goal status:", out)
        self.assertNotIn(NO_REPORT_MARKER, out)


class F4JudgeToolsetTest(unittest.TestCase):
    """F4: the judge child requests terminal so it can run the shell check itself."""

    def test_judge_requests_terminal_toolset(self):
        life = FakeSubagentLifecycle("VERDICT: done")
        ctx = FakeCtx()
        ctx.subagent_lifecycle = life
        verdict = run_judge(ctx, "file has one line", "test $(wc -l < f) -le 1", "{}")
        self.assertTrue(verdict.done)
        self.assertEqual(verdict.method, "subagent")
        self.assertEqual(len(life.launches), 1)
        allowed = tuple(life.launches[0]["allowed_toolsets"])
        self.assertIn("terminal", allowed)
        self.assertIn("file", allowed)

    def test_known_parent_toolsets_without_terminal_drop_it(self):
        life = FakeSubagentLifecycle("VERDICT: done")
        ctx = FakeCtx()
        ctx.subagent_lifecycle = life
        # When the caller DOES know the parent rights (tests), terminal is dropped if absent.
        run_judge(ctx, "g", "c", "{}", parent_toolsets={"file"})
        allowed = tuple(life.launches[0]["allowed_toolsets"])
        self.assertEqual(allowed, ("file",))


class F7SubagentExclusionTest(unittest.TestCase):
    """F7: a judge child inherits the parent's conversation_id, so the loop must not drive it."""

    def test_pre_stop_skips_a_subagent_and_leaves_continuations(self):
        # goals set in the PARENT conversation conv-P, with go
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="parent", conversation_id="conv-P")
        tools.goal_go({}, session_id="parent", conversation_id="conv-P")
        before = store.load(_chat_key("conv-P")).continuations
        hook = make_pre_stop(store)
        # the judge child runs with platform=subagent and the INHERITED parent conversation id
        self.assertIsNone(hook(session_id="child", platform="subagent",
                               conversation_id="conv-P", attempt=0))
        # the child must not have written continuations on the parent's state
        self.assertEqual(store.load(_chat_key("conv-P")).continuations, before)
        # the same open goals in a real local turn still continue
        result = hook(session_id="parent", platform="local",
                      conversation_id="conv-P", attempt=0)
        self.assertEqual(result["action"], "continue")
        self.assertIn("g1", result["message"])

    def test_transform_skips_a_subagent_answer(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="parent", conversation_id="conv-P")
        transform = make_transform_llm_output(store)
        # the judge's own answer is a VERDICT line; it must pass through untouched (None = no change)
        self.assertIsNone(transform(response_text="VERDICT: done", platform="subagent",
                                    conversation_id="conv-P"))
        # a local answer in the same conversation still gets the status block
        out = transform(response_text="REPORT", platform="local", conversation_id="conv-P")
        self.assertIn("REPORT", out)
        self.assertIn("Goal status:", out)


class Goal3JudgeTest(unittest.TestCase):
    """Goal 3: a fresh-context judge checks the real state and gives a reason on rejection."""

    def test_claim_confirmed_by_judge_marks_done(self):
        tools, store, ctx = make_tools("VERDICT: done")
        tools.goal_set({"goals": [{"id": "g1", "text": "branch exists", "check": "git rev-parse g1"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {
            "summary": "branch pushed",
            "checks": [{"command": "git rev-parse g1", "result": "passed", "exitCode": 0}],
        }}, **IDS)
        self.assertIn("confirmed", out)
        self.assertTrue(store.load(_chat_key("s1")).goal("g1").done)
        # the judge saw the goal + evidence as data, with temperature 0 and the goal_judge task
        self.assertTrue(ctx.llm.calls)
        self.assertEqual(ctx.llm.calls[0][1].get("task"), "goal_judge")

    def test_judge_rejection_gives_a_reason_and_keeps_goal_open(self):
        tools, store, _ = make_tools("VERDICT: not-done the branch has no commits")
        tools.goal_set({"goals": [{"id": "g1", "text": "branch exists", "check": "c"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "done",
                               "checks": [{"explanation": "I think so", "result": "passed"}]}}, **IDS)
        self.assertIn("rejected", out)
        self.assertIn("the branch has no commits", out)
        goal = store.load(_chat_key("s1")).goal("g1")
        self.assertFalse(goal.done)
        self.assertEqual(goal.judge_reason, "the branch has no commits")

    def test_malformed_claim_is_refused_before_the_judge(self):
        tools, _, ctx = make_tools("VERDICT: done")
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": ""}}, **IDS)
        self.assertIn("claim rejected", out)
        self.assertEqual(ctx.llm.calls, [])  # judge never ran

    def test_unknown_goal_id_names_known_ids_and_placement(self):
        tools, _, ctx = make_tools("VERDICT: done")
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "gX", "claim": {"summary": "s"}}, **IDS)
        self.assertIn("unknown goal_id", out)
        self.assertIn("g1", out)            # names the known id
        self.assertIn("top level", out)     # tells the model where goal_id goes
        self.assertEqual(ctx.llm.calls, [])  # judge never ran

    def test_judge_no_verdict_line_fails_closed(self):
        tools, store, _ = make_tools("the work looks fine to me")
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        out = tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s",
                               "checks": [{"explanation": "e", "result": "passed"}]}}, **IDS)
        self.assertIn("rejected", out)


class Goal4DoneStaysDoneTest(unittest.TestCase):
    """Goal 4: a confirmed goal stays done; each report shows all goals."""

    def test_confirmed_goal_stays_done_and_pre_stop_skips_it(self):
        tools, store, _ = make_tools("VERDICT: done")
        tools.goal_set({"goals": [
            {"id": "g1", "text": "a", "check": "c1"}, {"id": "g2", "text": "b", "check": "c2"},
        ]}, **IDS)
        tools.goal_go({}, **IDS)
        tools.goal_claim({"goal_id": "g1", "claim": {"summary": "s",
                         "checks": [{"command": "c1", "result": "passed", "exitCode": 0}]}}, **IDS)
        hook = make_pre_stop(store)
        result = hook(session_id="s1", platform="api_server", attempt=1)
        self.assertIn("g2", result["message"])
        self.assertNotIn("g1", result["message"])  # done goal is not re-listed

    def test_status_block_lists_all_goals(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [
            {"id": "g1", "text": "a", "check": "c1"}, {"id": "g2", "text": "b", "check": "c2"},
        ]}, **IDS)
        transform = make_transform_llm_output(store)
        out = transform(text="here is my answer", session_id="s1", platform="api_server")
        self.assertIn("here is my answer", out)
        self.assertIn("g1: a", out)
        self.assertIn("g2: b", out)
        self.assertIn("Goal status:", out)


class Goal5DecisionsTest(unittest.TestCase):
    """Goal 5: decisions at the end as a numbered text list, no card and no modal."""

    def test_decisions_render_as_a_numbered_text_list(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        tools.goal_decision({"text": "Repo name: A public, B private. Recommend A."}, **IDS)
        tools.goal_decision({"text": "Base tag: A v1, B v2. Recommend A."}, **IDS)
        state = store.load(_chat_key("s1"))
        block = status_block(state)
        self.assertIn("Open decisions:", block)
        self.assertIn("1. Repo name:", block)
        self.assertIn("2. Base tag:", block)


class ClaimUnitTest(unittest.TestCase):
    def test_failed_check_rejects_the_claim(self):
        with self.assertRaises(ClaimError):
            serialize_completion_claim({"summary": "s", "checks": [{"command": "c", "result": "failed"}]})

    def test_passed_check_with_nonzero_exit_rejects(self):
        with self.assertRaises(ClaimError):
            serialize_completion_claim({"summary": "s",
                                        "checks": [{"command": "c", "result": "passed", "exitCode": 1}]})

    def test_valid_claim_renders_evidence_lines(self):
        evidence = serialize_completion_claim({
            "summary": "branch pushed",
            "criteria": [{"criterion": "branch exists", "evidence": ["git rev-parse ok"]}],
            "checks": [{"command": "git rev-parse g1", "result": "passed", "exitCode": 0}],
            "changedFiles": ["patch_stop_hook.py"],
        })
        self.assertIn("Summary: branch pushed", evidence)
        self.assertIn("Criterion: branch exists", evidence)
        self.assertIn("Check: git rev-parse g1 | passed | exit 0", evidence)
        self.assertIn("Changed files: patch_stop_hook.py", evidence)


class StateUnitTest(unittest.TestCase):
    def test_ledger_is_append_only_and_records_terminal_events(self):
        tmp = tempfile.mkdtemp(prefix="goal-loop-ledger-")
        store = GoalStore(tmp)
        store.append_ledger("c1", "go")
        store.append_ledger("c1", "confirmed", goal_id="g1")
        events = store.read_ledger("c1")
        self.assertEqual([e["event"] for e in events], ["go", "confirmed"])
        self.assertEqual(events[1]["goal_id"], "g1")

    def test_open_goals_excludes_done_and_blocked(self):
        state = GoalState(chat_key="c1", goals=[
            Goal("g1", "a", "c", done=True), Goal("g2", "b", "c", status="blocked"),
            Goal("g3", "c", "c"),
        ])
        self.assertEqual([g.id for g in state.open_goals()], ["g3"])


class RegisterTest(unittest.TestCase):
    def test_register_wires_six_tools_and_two_hooks(self):
        os.environ["GOAL_LOOP_STATE_DIR"] = tempfile.mkdtemp(prefix="goal-loop-reg-")
        try:
            ctx = FakeCtx()
            register(ctx)
            names = {name for name, _, _ in ctx.tools}
            self.assertEqual(names, {"goal_set", "goal_go", "goal_status", "goal_claim",
                                     "goal_decision", "goal_block"})
            self.assertIn("pre_stop", ctx.hooks)
            self.assertIn("transform_llm_output", ctx.hooks)
        finally:
            del os.environ["GOAL_LOOP_STATE_DIR"]


if __name__ == "__main__":
    unittest.main()
