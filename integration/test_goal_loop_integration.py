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

# Real Hermes modules (the patched copy, because cwd is its root):
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

    def test_no_go_lets_the_turn_finish_through_the_patched_gate(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        # no goal_go
        agent = FakeAgent(**IDS)
        verdict, _ = run_stop_gate(agent)
        self.assertFalse(verdict.continue_turn)

    def test_all_goals_done_lets_the_turn_finish(self):
        self.tools.goal_set({"goals": [{"id": "g1", "text": "x", "check": "c"}]}, **IDS)
        self.tools.goal_go({}, **IDS)
        state = self.tools.store.load("api_server:it-sess")
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
