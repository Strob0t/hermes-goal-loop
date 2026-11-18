"""Integration test (Deliverable 3): the goal-loop plugin and the patched fork operate together.

This test runs against a REAL Hermes source tree with the stop-hook patch applied (the temp copy
``hermes-copy1``), not a mock of it. It:

1. Registers the REAL plugin through a REAL ``PluginManager``/``PluginContext`` (the same
   ``register(ctx)`` path the loader uses), so the plugin's ``pre_stop`` and ``transform_llm_output``
   hooks land in the manager's hook registry.
2. Drives the REAL patched ``agent.turn_stop_gates.apply_stop_gates`` with a minimal fake agent.
   The patch's ``_pre_stop_nudge`` gate calls ``has_hook("pre_stop")`` and
   ``get_pre_stop_continue_message(...)``, which dispatch to the registered plugin callback. A
   continue verdict proves the two pieces work together end to end.
3. Proves the no-op property: with NO plugin registered, the patched gate changes nothing (the
   turn finishes), so the fork patch alone does not change Hermes behaviour.

Run (cwd = the patched copy root so its modules win over any installed Hermes):
    cd <hermes-copy1>
    PYTHONPATH=<plugin-dir> GOAL_LOOP_STATE_DIR=<tmp> \\
        <venv>/bin/python -m unittest integration.test_goal_loop_integration -v
"""
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

# Real Hermes modules (the patched copy, because cwd is its root):
from types import SimpleNamespace

from agent.turn_finalizer import apply_llm_output_transform
from agent.turn_stop_gates import apply_stop_gates
from hermes_cli import plugins as plugins_facade
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_context import PluginContext
from hermes_cli.plugins_manifest import PluginManifest

import goal_loop  # the real plugin under test

IDS = {"session_id": "it-sess", "platform": "api_server"}


class FakeAgent:
    """Only what apply_stop_gates / _append_interim_answer / _continue touch."""

    def __init__(self, session_id, platform):
        self.session_id = session_id
        self.platform = platform
        self._session_messages = []
        self._interim = []

    def _emit_interim_assistant_message(self, msg):
        self._interim.append(msg)

    def _flush_messages_to_session_db(self, messages, history):
        pass

    def _interim_content_was_streamed(self, text):
        return False


def load_real_plugin(manager):
    """Run the plugin's real register(ctx) against a real PluginContext on *manager*."""
    manifest = PluginManifest(name="goal-loop", version="0.1.0", source="user",
                              provides_hooks=["pre_stop", "transform_llm_output"])
    ctx = PluginContext(manifest, manager)
    goal_loop.register(ctx)
    # The manager is now a real consumer of pre_stop; mark discovery done so lazy scans are skipped.
    manager._discovered = True
    return ctx


def run_stop_gate_with_response(agent, final_response):
    """Like run_stop_gate but with a chosen final answer (used for a judge VERDICT line)."""
    final_msg = {"role": "assistant", "content": final_response}
    messages = [{"role": "user", "content": "judge"}]
    return apply_stop_gates(
        agent, final_msg, final_response=final_response, messages=messages,
        conversation_history=None, pending_verification_response=None,
        pending_verification_response_previewed=False,
    ), messages


def run_stop_gate(agent):
    """Call the real patched stop gate as the turn finalizer does, with a text answer."""
    final_msg = {"role": "assistant", "content": "interim answer"}
    messages = [{"role": "user", "content": "do the work"}]
    return apply_stop_gates(
        agent, final_msg, final_response="interim answer", messages=messages,
        conversation_history=None, pending_verification_response=None,
        pending_verification_response_previewed=False,
    ), messages


class PatchedGateNoPluginTest(unittest.TestCase):
    """The fork patch alone is a no-op: with no pre_stop consumer the turn finishes."""

    def setUp(self):
        self._saved = plugins_facade._plugin_manager
        plugins_facade._plugin_manager = PluginManager()
        plugins_facade._plugin_manager._discovered = True  # empty registry, discovery done

    def tearDown(self):
        plugins_facade._plugin_manager = self._saved

    def test_no_consumer_lets_the_turn_finish(self):
        agent = FakeAgent(**IDS)
        verdict, _ = run_stop_gate(agent)
        self.assertFalse(verdict.continue_turn)
        self.assertEqual(verdict.final_response, "interim answer")
        self.assertEqual(getattr(agent, "_pre_stop_nudges", 0), 0)


class PluginAndPatchedForkTogetherTest(unittest.TestCase):
    """The real plugin + the real patched gate: the goal loop keeps the turn going to the end."""

    def setUp(self):
        os.environ["GOAL_LOOP_STATE_DIR"] = tempfile.mkdtemp(prefix="goal-loop-it-")
        self._saved = plugins_facade._plugin_manager
        self.manager = PluginManager()
        plugins_facade._plugin_manager = self.manager
        self.ctx = load_real_plugin(self.manager)
        self.tools = goal_loop._Tools(goal_loop.GoalStore(os.environ["GOAL_LOOP_STATE_DIR"]), self.ctx)

    def tearDown(self):
        plugins_facade._plugin_manager = self._saved
        del os.environ["GOAL_LOOP_STATE_DIR"]

    def test_hook_is_registered_in_the_real_manager(self):
        self.assertTrue(self.manager.has_hook("pre_stop"))
        self.assertTrue(self.manager.has_hook("transform_llm_output"))
        self.assertIn("goal_set", goal_loop._Tools.__dict__)

    def test_open_goal_with_go_continues_the_turn_through_the_patched_gate(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "branch exists", "check": "git rev-parse g1"}]}, **IDS)
        self.tools.goal_go({}, **IDS)
        agent = FakeAgent(**IDS)
        verdict, messages = run_stop_gate(agent)
        # The patched gate asked the plugin and got a continue directive:
        self.assertTrue(verdict.continue_turn)
        self.assertEqual(agent._pre_stop_nudges, 1)
        nudge = messages[-1]
        self.assertTrue(nudge.get("_pre_stop_synthetic"))
        self.assertIn("g1", nudge["content"])
        self.assertIn("[goal-loop]", nudge["content"])

    def test_registry_exposes_goal_tool_schemas_to_the_model(self):
        # F1: the model reads function["description"]/function["parameters"] from the registry.
        # A bare parameter schema would surface empty parameters and no description.
        from tools.registry import registry
        names = {"goal_set", "goal_go", "goal_status", "goal_claim", "goal_decision", "goal_block"}
        defs = {d["function"]["name"]: d["function"]
                for d in registry.get_definitions(names)}
        self.assertEqual(set(defs), names)  # all six are registered and retrievable
        for name, fn in defs.items():
            self.assertTrue(fn.get("description"), f"{name}: empty description")
            params = fn.get("parameters") or {}
            self.assertEqual(params.get("type"), "object", f"{name}: parameters not an object schema")
        claim = defs["goal_claim"]["parameters"]
        self.assertEqual(set(claim.get("properties", {})) & {"goal_id", "claim"},
                         {"goal_id", "claim"})  # the model sees the real claim shape
        self.assertIn("goal_id", claim.get("required", []))
        set_props = defs["goal_set"]["parameters"].get("properties", {})
        self.assertIn("goals", set_props)

    def test_tool_without_platform_and_hook_with_platform_share_state(self):
        # F2: the dispatcher passes tools session_id/conversation_id but NOT platform (#68318);
        # the pre_stop hook sees agent.platform. Both must key on the same file or the loop is blind.
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                            session_id="it-sess")          # no platform, as the dispatcher sends it
        self.tools.goal_go({}, session_id="it-sess")
        agent = FakeAgent(session_id="it-sess", platform="local")  # hook side: a real platform
        verdict, messages = run_stop_gate(agent)
        self.assertTrue(verdict.continue_turn)              # the hook found the tool's state
        self.assertIn("g1", messages[-1]["content"])

    def test_real_finalizer_keeps_model_text_and_appends_status(self):
        # F6: the real fork caller passes the model answer as response_text; the status block must
        # be appended, not substituted for the answer.
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        agent = SimpleNamespace(session_id="it-sess", model="m", platform="local")
        final, transformed, _ = apply_llm_output_transform(agent, "MODEL REPORT", turn_id="t1")
        self.assertTrue(transformed)
        self.assertIn("MODEL REPORT", final)
        self.assertIn("Goal status:", final)
        self.assertNotIn("(no final report from the model)", final)

    def test_state_survives_session_id_rotation_via_conversation_id(self):
        # F3: goals are set under session A with conversation root C; after a compaction the
        # session id is B. The patched gate must still continue the turn (the fork now hands
        # conversation_id to the pre_stop hook; the plugin keys on it).
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                            session_id="A", conversation_id="C")
        self.tools.goal_go({}, session_id="A", conversation_id="C")
        agent = FakeAgent(session_id="B", platform="local")   # rotated session id
        agent._inherited_conversation_id = "C"                # the stable conversation root
        verdict, messages = run_stop_gate(agent)
        self.assertTrue(verdict.continue_turn)
        self.assertIn("g1", messages[-1]["content"])

    def test_transform_survives_session_id_rotation_via_conversation_id(self):
        # F3 end to end on the transform side: the real finalizer hands conversation_id to the
        # plugin hook, so the status block is appended even after the session id rotated.
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                            session_id="A", conversation_id="C")
        agent = SimpleNamespace(session_id="B", model="m", platform="local",
                                 _inherited_conversation_id="C")
        final, transformed, _ = apply_llm_output_transform(agent, "MODEL REPORT R", turn_id="t9")
        self.assertTrue(transformed)
        self.assertIn("MODEL REPORT R", final)
        self.assertIn("Goal status:", final)

    def test_judge_subagent_is_not_driven_by_the_parent_loop(self):
        # F7 through the REAL gate: the parent sets goals under conversation conv-P; a judge child
        # runs with platform=subagent and the inherited conv-P. The gate must NOT continue the child.
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                            session_id="parent", conversation_id="conv-P")
        self.tools.goal_go({}, session_id="parent", conversation_id="conv-P")
        # the parent turn continues
        parent = FakeAgent(session_id="parent", platform="local")
        parent._inherited_conversation_id = "conv-P"
        v_parent, _ = run_stop_gate(parent)
        self.assertTrue(v_parent.continue_turn)
        # the judge child does NOT: its verdict survives as the final response
        judge = FakeAgent(session_id="child", platform="subagent")
        judge._inherited_conversation_id = "conv-P"
        v_judge, _ = run_stop_gate_with_response(judge, "VERDICT: done")
        self.assertFalse(v_judge.continue_turn)
        self.assertEqual(v_judge.final_response, "VERDICT: done")

    def test_transform_leaves_a_subagent_verdict_untouched(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]},
                            session_id="parent", conversation_id="conv-P")
        judge = SimpleNamespace(session_id="child", model="m", platform="subagent",
                                 _inherited_conversation_id="conv-P")
        final, transformed, _ = apply_llm_output_transform(judge, "VERDICT: done", turn_id="tj")
        self.assertFalse(transformed)
        self.assertEqual(final, "VERDICT: done")

    def test_register_puts_state_under_the_active_hermes_home(self):
        # F8: without GOAL_LOOP_STATE_DIR the plugin's store must live under the Hermes home that
        # loaded the plugin, so an isolated home can never read or overwrite another home's state.
        # F9: isolate HOME too (Path.home() + ~/.hermes) and use a unique session id, so a stale
        # file from an earlier run of this test cannot pass or fail it. patch.dict restores env.
        saved_manager = plugins_facade._plugin_manager
        fake_home = tempfile.mkdtemp(prefix="f8-it-home-")
        fake_user = tempfile.mkdtemp(prefix="f8-it-user-")
        real_user_dir = os.path.expanduser("~")  # captured BEFORE patching HOME
        sid = f"s-f8-{uuid.uuid4().hex[:8]}"
        chat_file = f"chat_{sid}.json"
        env = {k: v for k, v in os.environ.items() if k != "GOAL_LOOP_STATE_DIR"}
        env.update(HERMES_HOME=fake_home, HOME=fake_user)
        try:
            with patch.dict(os.environ, env, clear=True):
                from hermes_constants import get_hermes_home
                self.assertEqual(str(get_hermes_home()), fake_home)
                manager = PluginManager()
                plugins_facade._plugin_manager = manager
                ctx = load_real_plugin(manager)  # register() fixes the store NOW
                out = ctx.dispatch_tool("goal_set", {"goals": [{"id": "g1", "text": "x",
                                                                "check": "c"}]}, session_id=sid)
                self.assertIn("g1", out)
                self.assertTrue(os.path.isfile(os.path.join(fake_home, "goal-loop", chat_file)),
                                "state must be under $HERMES_HOME/goal-loop")
                # with HOME patched, ~/.hermes expands into fake_user
                self.assertFalse((Path(fake_user) / ".hermes" / "goal-loop" / chat_file).exists())
            # after the env is restored: this session id must not appear in the real ~/.hermes.
            # real_user_dir may not exist (fresh user); treat the missing dir as "absent", not fail.
            real_leak = Path(real_user_dir) / ".hermes" / "goal-loop" / chat_file
            self.assertFalse(real_leak.exists(), f"test leaked a file into {real_leak}")
        finally:
            plugins_facade._plugin_manager = saved_manager

    def test_no_go_lets_the_turn_finish_through_the_patched_gate(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        # no goal_go
        agent = FakeAgent(**IDS)
        verdict, _ = run_stop_gate(agent)
        self.assertFalse(verdict.continue_turn)

    def test_all_goals_done_lets_the_turn_finish(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        self.tools.goal_go({}, **IDS)
        state = self.tools.store.load(goal_loop._chat_key("it-sess"))
        state.goals[0].done = True
        self.tools.store.save(state)
        agent = FakeAgent(**IDS)
        verdict, _ = run_stop_gate(agent)
        self.assertFalse(verdict.continue_turn)

    def test_bound_stops_the_loop_at_max_stop_continuations(self):
        from agent.verify_hooks import max_stop_continuations
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        self.tools.goal_go({}, **IDS)
        agent = FakeAgent(**IDS)
        agent._pre_stop_nudges = max_stop_continuations()  # at the ceiling
        verdict, _ = run_stop_gate(agent)
        self.assertFalse(verdict.continue_turn)  # the bound stops the continuation


if __name__ == "__main__":
    unittest.main()
