"""Tests of the goal-loop plugin, one or more per owner goal.

These tests are self-contained: they use a fake ctx and a temp state dir, so they run with plain
``python -m unittest`` without a Hermes install. The companion fork patch (the generic ``pre_stop``
hook) is tested in the fork; here the hook callback is called directly.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from goal_loop import (
    EXCLUDED_PLATFORMS, FINAL_REPORT_NUDGE, GO_REFUSED, GOAL_STATUS_DESCRIPTION, NO_REPORT_MARKER,
    PROMPT_SECTION_ID, STOP_CAUSES, USAGE_HINT, _Tools, _chat_key, _state_dir, make_pre_llm_call,
    make_pre_stop, make_transform_llm_output, prompt_section, register,
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
        self.sections = {}

    def register_tool(self, name, toolset, schema, handler, emoji="", description="", **kw):
        self.tools.append((name, toolset, handler, schema))

    def register_hook(self, hook_name, callback, **kw):
        self.hooks.setdefault(hook_name, []).append(callback)

    def register_system_prompt_section(self, id, content, **kw):
        self.sections[id] = content


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
CHECKS = {"checks": [{"command": "c", "result": "passed", "exitCode": 0}]}


def owner_says(store, session_id, text="go", platform="api_server", history=None):
    """One owner message as the fork's pre_llm_call hook delivers it (agent/turn_context.py)."""
    if history is None:
        history = [{"role": "user", "content": text}]
    make_pre_llm_call(store)(session_id=session_id, platform=platform, user_message=text,
                             conversation_history=history, task_id="t", turn_id="turn")


def go(tools, store, **ids):
    """The owner answers the goal list with a go; then the model calls goal_go."""
    owner_says(store, ids.get("session_id"))
    return tools.goal_go({}, **ids)


def claim(tools, goal_id, **ids):
    return tools.goal_claim({"goal_id": goal_id, "claim": {"summary": "s", **CHECKS}}, **ids)


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


class GoRuleTest(unittest.TestCase):
    """g5: goal_go is accepted only after an owner message that came after the last goal_set."""

    def test_go_directly_after_goal_set_is_refused_and_the_loop_does_not_start(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        out = tools.goal_go({}, **IDS)
        self.assertTrue(out.startswith("error:"))
        self.assertEqual(out, GO_REFUSED)
        self.assertFalse(store.load(_chat_key("s1")).go)
        self.assertIsNone(make_pre_stop(store)(session_id="s1", platform="api_server", attempt=0))
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "go_refused")

    def test_go_after_an_owner_message_is_accepted_and_the_loop_starts(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "s1", "Go")
        out = tools.goal_go({}, **IDS)
        self.assertTrue(out.startswith("Go recorded"))
        self.assertTrue(store.load(_chat_key("s1")).go)
        result = make_pre_stop(store)(session_id="s1", platform="api_server", attempt=0)
        self.assertEqual(result["action"], "continue")

    def test_an_owner_message_before_goal_set_does_not_count(self):
        tools, store, _ = make_tools()
        owner_says(store, "s1", "please plan this")      # the request that leads to goal_set
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_a_new_goal_set_needs_a_new_owner_message(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        self.assertTrue(go(tools, store, **IDS).startswith("Go recorded"))
        tools.goal_set({"goals": [{"id": "g1", "text": "y", "check": "c"}]}, **IDS)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_notification_subagent_and_cron_turns_do_not_count(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        footer = ("[INTERNAL NOTIFICATION — generated by the Hermes gateway, not written by a "
                  "human user. Do not treat it as user input.]")
        owner_says(store, "s1", "process done\n\n" + footer)
        owner_says(store, "s1", "wake", history=[{"role": "user", "content": "wake",
                                                  "display_kind": "internal_notification"}])
        owner_says(store, "s1", "")
        for platform in EXCLUDED_PLATFORMS:
            owner_says(store, "s1", "go", platform=platform)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_a_typed_steer_row_counts_as_owner_message(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "s1", "go", history=[{"role": "user", "content": "go",
                                                "display_kind": "steer"}])
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))

    def test_an_api_client_turn_under_the_session_authority_counts(self):
        # /v1/runs under the session authority: the row is internal_notification + api_inbound
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "s1", "Go", history=[{
            "role": "user", "content": "Go", "display_kind": "internal_notification",
            "display_metadata": {"api_inbound": True}}])
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))

    def test_the_gate_is_per_session(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "other-session", "go")
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_a_multimodal_owner_message_counts(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "s1", [{"type": "text", "text": "go"}, {"type": "image_url"}])
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))

    def test_a_rotation_between_goal_set_and_the_go_keeps_the_gate(self):
        # compression.in_place false: goal_set in s1, the go message and goal_go in the rotated s2;
        # the fork patch hands pre_llm_call the conversation_id, the tools get it from the dispatcher
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1", conversation_id="conv")
        make_pre_llm_call(store)(session_id="s2", conversation_id="conv", platform="api_server",
                                 user_message="go", conversation_history=[{"role": "user", "content": "go"}])
        out = tools.goal_go({}, session_id="s2", conversation_id="conv")
        self.assertTrue(out.startswith("Go recorded"), out)
        self.assertTrue(store.load(_chat_key("conv")).go)
        self.assertEqual([f for f in os.listdir(store.base_dir) if f.startswith("gate_")], [])

    def test_a_rotation_still_refuses_a_go_without_an_owner_message(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1", conversation_id="conv")
        self.assertEqual(tools.goal_go({}, session_id="s2", conversation_id="conv"), GO_REFUSED)

    def test_without_the_fork_patch_the_session_id_gate_still_works(self):
        # an unpatched Hermes gives pre_llm_call no conversation_id; the tools still have it
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1", conversation_id="conv")
        owner_says(store, "s1", "go")
        self.assertTrue(tools.goal_go({}, session_id="s1", conversation_id="conv")
                        .startswith("Go recorded"))

    def test_a_wake_self_post_row_without_api_inbound_does_not_count(self):
        # gateway/wake.py self-post under the session authority after the fork patch: the row is
        # internal_notification without api_inbound (kanban wakes have no footer)
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        owner_says(store, "s1", "[kanban] Task t1 completed.", history=[{
            "role": "user", "content": "[kanban] Task t1 completed.",
            "display_kind": "internal_notification", "display_metadata": {"gateway_input_owner": "o"}}])
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)


class DisplayOnSetAndChangeTest(unittest.TestCase):
    """g1: the status block appears exactly in a turn where goal_set set or changed the list."""

    def setUp(self):
        self.tools, self.store, _ = make_tools("VERDICT: not-done the file is missing")
        self.transform = make_transform_llm_output(self.store)

    def answer(self, text="ANSWER"):
        return self.transform(response_text=text, **IDS)

    def test_first_set_shows_the_block(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        out = self.answer()
        self.assertTrue(out.startswith("ANSWER"))
        self.assertIn("Goal status:", out)
        self.assertIn("g1: a (check: c1)", out)
        self.assertIsNone(self.answer())  # the next turn without a change shows nothing

    def test_change_shows_the_block_again_also_after_the_go(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        self.answer()
        go(self.tools, self.store, **IDS)
        self.assertIsNone(self.answer())
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"},
                                       {"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
        out = self.answer()
        self.assertIn("Goal status:", out)
        self.assertIn("g2: b (check: c2)", out)

    def test_a_status_change_without_a_list_change_shows_nothing(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"},
                                       {"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
        self.answer()
        go(self.tools, self.store, **IDS)
        self.assertIsNone(self.answer())
        # in work and rejected by the judge
        self.assertIn("rejected", claim(self.tools, "g1", **IDS))
        self.assertEqual(self.store.load(_chat_key("s1")).goal("g1").status, "in_work")
        self.assertIsNone(self.answer())
        # done (one goal of two)
        self.tools.ctx.llm._text = "VERDICT: done"
        self.assertIn("confirmed", claim(self.tools, "g1", **IDS))
        self.assertIsNone(self.answer())


class AfterGoOwnerItemsTest(unittest.TestCase):
    """g2: after the go an answer shows only blocked goals and open owner decisions."""

    def setUp(self):
        self.tools, self.store, _ = make_tools("VERDICT: not-done no evidence")
        self.transform = make_transform_llm_output(self.store)
        self.tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"},
                                       {"id": "g2", "text": "b", "check": "c2"},
                                       {"id": "g3", "text": "c", "check": "c3"}]}, **IDS)
        self.transform(response_text="LIST", **IDS)
        go(self.tools, self.store, **IDS)

    def answer(self, text="ANSWER"):
        return self.transform(response_text=text, **IDS)

    def test_a_blocked_goal_and_a_decision_appear(self):
        self.tools.goal_block({"goal_id": "g2", "reason": "token missing"}, **IDS)
        self.tools.goal_decision({"text": "Name: A, B or C. Recommend A."}, **IDS)
        out = self.answer()
        self.assertIn("needs the owner", out)
        self.assertIn("g2: b", out)
        self.assertIn("[blocked] — token missing", out)
        self.assertIn("1. Name: A, B or C. Recommend A.", out)
        self.assertNotIn("g1: a", out)   # no progress lines
        self.assertNotIn("g3: c", out)
        self.assertIsNone(self.answer())  # each owner item is shown one time

    def test_a_second_decision_shows_only_the_new_one_with_its_number(self):
        self.tools.goal_decision({"text": "first"}, **IDS)
        self.answer()
        self.tools.goal_decision({"text": "second"}, **IDS)
        out = self.answer()
        self.assertIn("2. second", out)
        self.assertNotIn("1. first", out)

    def test_a_turn_with_only_progress_appends_nothing(self):
        self.tools.ctx.llm._text = "VERDICT: done"
        claim(self.tools, "g1", **IDS)
        make_pre_stop(self.store)(session_id="s1", platform="api_server", attempt=0)
        self.assertIsNone(self.answer())

    def test_a_turn_with_only_a_judge_rejection_appends_nothing(self):
        self.assertIn("rejected", claim(self.tools, "g1", **IDS))
        self.assertIsNone(self.answer())


class FinalListOnceTest(unittest.TestCase):
    """g3: the full list with checkmarks appears one time when the judge confirmed all goals."""

    def test_final_list_once_then_nothing_until_a_new_goal_set(self):
        tools, store, _ = make_tools("VERDICT: done")
        transform = make_transform_llm_output(store)
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"},
                                  {"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
        transform(response_text="LIST", **IDS)
        go(tools, store, **IDS)
        claim(tools, "g1", **IDS)
        self.assertIn(FINAL_REPORT_NUDGE.strip(), claim(tools, "g2", **IDS))
        out = transform(response_text="FINAL REPORT", **IDS)
        self.assertTrue(out.startswith("FINAL REPORT"))
        self.assertIn("- [x] g1: a", out)
        self.assertIn("- [x] g2: b", out)
        self.assertTrue(store.load(_chat_key("s1")).final_shown)
        self.assertIsNone(transform(response_text="later", **IDS))
        # a decision after the final list still reaches the owner, one time
        tools.goal_decision({"text": "late pick: A or B"}, **IDS)
        out = transform(response_text="AFTER", **IDS)
        self.assertIn("needs the owner", out)
        self.assertIn("1. late pick: A or B", out)
        self.assertNotIn("g1: a", out)
        self.assertIsNone(transform(response_text="again", **IDS))
        tools.goal_set({"goals": [{"id": "n1", "text": "new", "check": "c"}]}, **IDS)
        out = transform(response_text="NEW", **IDS)
        self.assertIn("- [ ] n1: new", out)
        self.assertNotIn("g1: a", out)

    def test_an_old_state_file_without_the_new_fields_loads(self):
        tmp = tempfile.mkdtemp(prefix="goal-loop-old-")
        store = GoalStore(tmp)
        old = {"chat_key": "chat:s1", "version": 1, "go": True, "continuations": 2,
               "decisions": ["old decision"],
               "goals": [{"id": "g1", "text": "a", "check": "c", "status": "done", "done": True,
                          "note": "", "judge_reason": ""},
                         {"id": "g2", "text": "b", "check": "c", "status": "blocked",
                          "done": False, "note": "why", "judge_reason": ""}]}
        with open(os.path.join(tmp, "chat_s1.json"), "w", encoding="utf-8") as handle:
            json.dump(old, handle)
        state = store.load("chat:s1")
        self.assertEqual([g.id for g in state.goals], ["g1", "g2"])
        self.assertTrue(state.go)
        self.assertFalse(state.list_changed)
        self.assertEqual(state.decisions_shown, 1)       # the old decision was shown already
        self.assertTrue(state.goal("g2").block_shown)    # the old blocked goal was shown already
        self.assertFalse(state.final_shown)              # not all done
        # the old state shows nothing new
        self.assertIsNone(make_transform_llm_output(store)(response_text="x", session_id="s1"))
        # an old, completely done state counts as shown
        old["goals"][1].update(status="done", done=True)
        with open(os.path.join(tmp, "chat_s1.json"), "w", encoding="utf-8") as handle:
            json.dump(old, handle)
        self.assertTrue(store.load("chat:s1").final_shown)
        store.save(store.load("chat:s1"))                # it saves in the new format
        with open(os.path.join(tmp, "chat_s1.json"), encoding="utf-8") as handle:
            self.assertIn("final_shown", handle.read())


class GoalStatusToolTest(unittest.TestCase):
    """g4 b): goal_status returns the full list as finished text, the description asks for it 1:1."""

    def test_goal_status_returns_the_finished_block(self):
        tools, store, _ = make_tools()
        self.assertEqual(tools.goal_status({}, **IDS), "No goals set.")
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"},
                                  {"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
        tools.goal_decision({"text": "pick A"}, **IDS)
        out = tools.goal_status({}, **IDS)
        self.assertEqual(out, status_block(store.load(_chat_key("s1"))).lstrip("\n-"))
        self.assertTrue(out.startswith("Goal status:\n- [ ] g1: a (check: c1) [open]"))
        self.assertIn("1. pick A", out)

    def test_goal_status_does_not_change_the_display_state(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        tools.goal_status({}, **IDS)
        self.assertTrue(store.load(_chat_key("s1")).list_changed)

    def test_description_asks_for_the_text_unchanged(self):
        self.assertIn("unchanged", GOAL_STATUS_DESCRIPTION)
        self.assertIn("word for word", GOAL_STATUS_DESCRIPTION)


class UsageHintTest(unittest.TestCase):
    """g6: the usage hint is in the prompt of a normal session, not of a judge session."""

    def test_hint_for_a_normal_session_and_none_for_excluded_platforms(self):
        text = prompt_section({"platform": "api_server", "session_id": "s"})
        self.assertEqual(text, USAGE_HINT)
        self.assertIn("goal_set", text)
        self.assertIn("go", text)
        for platform in EXCLUDED_PLATFORMS:
            self.assertEqual(prompt_section({"platform": platform}), "")


class F3ConversationKeyTest(unittest.TestCase):
    """F3: state keys on the compression-stable conversation_id and survives session rotation."""

    def test_tool_keys_on_conversation_id_and_hook_reads_it_after_rotation(self):
        tools, store, _ = make_tools()
        # a tool call as the dispatcher sends it, before a compaction (session A)
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="A", conversation_id="C")
        go(tools, store, session_id="A", conversation_id="C")
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
        go(tools, store, session_id="A")
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
        # without the conversation id the rotated session finds no state
        self.assertIsNone(transform(response_text="REPORT", session_id="B"))
        out = transform(response_text="REPORT", session_id="B", conversation_id="C")
        self.assertIn("REPORT", out)
        self.assertIn("Goal status:", out)


class Goal2ToTheEndTest(unittest.TestCase):
    """Goal 2: work to the end; state survives compaction (keyed by chat, not session id)."""

    def test_pre_stop_continues_while_open_goals_remain(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        go(tools, store, **IDS)
        hook = make_pre_stop(store)
        result = hook(session_id="s1", platform="api_server", attempt=0)
        self.assertEqual(result["action"], "continue")
        self.assertIn("g1", result["message"])
        for cause in STOP_CAUSES:
            self.assertIn(cause, result["message"])

    def test_pre_stop_stops_when_all_goals_done(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        go(tools, store, **IDS)
        state = store.load(_chat_key("s1"))
        state.goals[0].done = True
        store.save(state)
        hook = make_pre_stop(store)
        self.assertIsNone(hook(session_id="s1", platform="api_server", attempt=0))

    def test_cron_sessions_are_excluded(self):
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1", platform="cron")
        go(tools, store, session_id="s1", platform="cron")
        hook = make_pre_stop(store)
        self.assertIsNone(hook(session_id="s1", platform="cron", attempt=0))

    def test_pre_stop_reads_the_state_a_tool_wrote_despite_platform(self):
        # F2: tools get no platform kwarg (key off session_id only); the hook gets agent.platform.
        # Both must land on one key, or the loop never sees the goals the tool stored.
        tools, store, _ = make_tools()
        tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                       session_id="s1")            # tool call: no platform, as the dispatcher sends it
        go(tools, store, session_id="s1")
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
        go(tools, store, **IDS)
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
        go(tools, store, session_id="parent", conversation_id="conv-P")
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
    """Goal 4: a confirmed goal stays done; the full list shows all goals."""

    def test_confirmed_goal_stays_done_and_pre_stop_skips_it(self):
        tools, store, _ = make_tools("VERDICT: done")
        tools.goal_set({"goals": [
            {"id": "g1", "text": "a", "check": "c1"}, {"id": "g2", "text": "b", "check": "c2"},
        ]}, **IDS)
        go(tools, store, **IDS)
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


class F8StateDirTest(unittest.TestCase):
    """F8: the state folder follows the active Hermes home, not a fixed ~/.hermes path."""

    def test_hermes_home_is_used_when_no_override_is_set(self):
        # get_hermes_home() is importable here (the Hermes tree is not required for these unit
        # tests, so the fallback path applies): HERMES_HOME/<goal-loop> is the folder.
        with patch.dict(os.environ, {"HERMES_HOME": tempfile.mkdtemp(prefix="f8-home-")},
                        clear=False):
            with patch.dict(os.environ, {}, clear=False):
                environ = os.environ.copy(); environ.pop("GOAL_LOOP_STATE_DIR", None)
                with patch.dict(os.environ, environ, clear=True):
                    self.assertEqual(_state_dir(),
                                     os.path.join(os.environ["HERMES_HOME"], "goal-loop"))

    def test_override_wins_over_hermes_home(self):
        with patch.dict(os.environ, {"GOAL_LOOP_STATE_DIR": "/tmp/f8-override"}, clear=False):
            with patch.dict(os.environ, {"HERMES_HOME": "/tmp/f8-other-home"}, clear=False):
                self.assertEqual(_state_dir(), "/tmp/f8-override")

    def test_fallback_without_hermes_home_ends_in_dot_hermes_goal_loop(self):
        # F9: pin HOME too, so the expanded ~/.hermes/goal-loop never touches the real user home.
        fake_user = tempfile.mkdtemp(prefix="f8-u-")
        environ = {k: v for k, v in os.environ.items()
                   if k not in ("GOAL_LOOP_STATE_DIR", "HERMES_HOME")}
        environ["HOME"] = fake_user
        with patch.dict(os.environ, environ, clear=True):
            self.assertEqual(_state_dir(),
                             os.path.join(fake_user, ".hermes", "goal-loop"))


class RegisterTest(unittest.TestCase):
    def test_register_wires_six_tools_three_hooks_and_the_usage_hint(self):
        os.environ["GOAL_LOOP_STATE_DIR"] = tempfile.mkdtemp(prefix="goal-loop-reg-")
        try:
            ctx = FakeCtx()
            register(ctx)
            names = {name for name, _, _, _ in ctx.tools}
            self.assertEqual(names, {"goal_set", "goal_go", "goal_status", "goal_claim",
                                     "goal_decision", "goal_block"})
            self.assertEqual(set(ctx.hooks), {"pre_stop", "transform_llm_output", "pre_llm_call"})
            self.assertIn(PROMPT_SECTION_ID, ctx.sections)
            self.assertEqual(ctx.sections[PROMPT_SECTION_ID]({"platform": "api_server"}), USAGE_HINT)
            self.assertEqual(ctx.sections[PROMPT_SECTION_ID]({"platform": "subagent"}), "")
            status_schema = next(s for n, _, _, s in ctx.tools if n == "goal_status")
            self.assertEqual(status_schema["description"], GOAL_STATUS_DESCRIPTION)
        finally:
            del os.environ["GOAL_LOOP_STATE_DIR"]

    def test_register_works_without_the_prompt_section_api(self):
        os.environ["GOAL_LOOP_STATE_DIR"] = tempfile.mkdtemp(prefix="goal-loop-reg2-")
        try:
            ctx = FakeCtx()
            del ctx.sections
            ctx.register_system_prompt_section = None
            register(ctx)
            self.assertEqual(len(ctx.tools), 6)
        finally:
            del os.environ["GOAL_LOOP_STATE_DIR"]


if __name__ == "__main__":
    unittest.main()
