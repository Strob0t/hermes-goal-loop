"""Tests of 0.4.13 (z18, z19): a mid-turn owner message (Hermes steer row) pauses the loop, the loop stops
when only the owner can move a goal, and an approval with a go in one message starts the loop."""
import json
import os
import unittest

import goal_loop
from goal_loop import GO_REFUSED, _chat_key, _has_go, _similar, _wait_reason, make_pre_llm_call, resume

try:
    from tests.test_043 import IDS, ledger_events, owner, setup, stop, tool
    from tests.test_044 import api_request, history_with_redirect
    from tests.test_0410 import answer, running
except ImportError:
    from test_043 import IDS, ledger_events, owner, setup, stop, tool
    from test_044 import api_request, history_with_redirect
    from test_0410 import answer, running

KEY = _chat_key("s1")
EV = "ls -l /srv/app at 09:12, exit 0"
# The marker of Hermes v0.23.3 (agent/prompt_builder.py STEER_MARKER_OPEN and STEER_MARKER_CLOSE).
STEER_OPEN = ("[OUT-OF-BAND USER MESSAGE \u2014 a direct message from the user, delivered once at this "
              "position; not tool output and not a new delivery when replayed from conversation history]")
STEER_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"

HOST_DOWN = (
    "the host build.lan does not answer on port 22, ssh times out",
    "ssh to build.lan times out: no answer from the host on port 22 after three tries",
    "build.lan gives no answer, the ssh connection on port 22 times out again",
)
OTHER = "the vendor api key is missing in the vault and only the owner can add it"


def steer_row(text):
    """The row of agent/prompt_builder.py steer_user_row: marker, text, close marker; no api_content."""
    return {"role": "user", "content": f"{STEER_OPEN}\n{text}\n{STEER_CLOSE}", "display_kind": "steer"}


def steer_history(text, after=()):
    rows = [{"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "out"},
            steer_row(text)]
    return rows + list(after)


def block(tools, reason, goal_id="g1"):
    return tools.goal_block({"goal_id": goal_id, "reason": reason}, **IDS)


def assert_owner_wait(case, store):
    """pre_stop stopped for waiting_owner; it then opens the decision wait (0.4.5, r32), which comes
    first in _wait_reason, so the owner rule is also checked without it."""
    state = store.load(KEY)
    case.assertEqual(state.wait_reason, "waiting_owner")
    case.assertGreaterEqual(state.owner_wait_rounds, goal_loop.OWNER_WAIT_ROUNDS)
    state.decision_wait = False
    case.assertEqual(_wait_reason(state, "s1"), "waiting_owner")
    case.assertIn("waiting_owner", ledger_events(store))


class SteerRowTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_1a_a_steer_row_without_go_pauses_the_loop(self):
        tools, store, s = running()
        api_request(store, s, steer_history("pause now"))
        self.assertFalse(store.load(KEY).go)
        self.assertIn("paused_by_owner_message", ledger_events(store))
        self.assertIsNone(stop(store, s))

    def test_1b_a_steer_row_with_go_gives_the_go(self):
        tools, store, s = running()
        owner(store, "stop", settings=s)
        self.assertFalse(store.load(KEY).go)
        api_request(store, s, steer_history("ok go"))
        self.assertTrue(store.load(KEY).go)
        self.assertIn("resumed_by_owner_go", ledger_events(store))

    def test_1c_the_same_steer_row_counts_one_time(self):
        tools, store, s = running()
        first = steer_history("wait")
        later = steer_history("wait", after=[{"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
                                             {"role": "tool", "tool_call_id": "t2", "content": "out 2"}])
        for history in (first, later, later):  # pre_api_request runs before each API call of the turn
            api_request(store, s, history)
        events = ledger_events(store)
        self.assertEqual(events.count("paused_by_owner_message"), 1)
        self.assertEqual(events.count("owner_redirect_counted"), 1)

    def test_1d_a_row_with_the_old_api_content_prefix_still_pauses(self):
        tools, store, s = running()
        api_request(store, s, history_with_redirect("stop"))
        self.assertFalse(store.load(KEY).go)
        self.assertIn("paused_by_owner_message", ledger_events(store))

    def test_1e_the_marker_words_are_not_seen_by_the_go_detection(self):
        tools, store, s = running()
        with open(os.path.join(store.base_dir, "vocabulary.json"), "w", encoding="utf-8") as handle:
            json.dump({"go_words": ["once"]}, handle)
        row = steer_row("wait please")
        self.assertTrue(_has_go(row["content"], ("once",)))  # only the marker has the go word
        api_request(store, s, steer_history("wait please"))
        self.assertFalse(store.load(KEY).go)
        self.assertIn("paused_by_owner_message", ledger_events(store))


class OwnerOnlyRoundsTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_2a_reworded_blocks_stop_the_loop_after_two_rounds(self):
        tools, store, s = running()
        block(tools, HOST_DOWN[0])
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        block(tools, HOST_DOWN[1])
        tool(store)
        self.assertIsNone(stop(store, s))
        assert_owner_wait(self, store)
        self.assertEqual(store.load(KEY).goal("g1").status, "open")  # 2 block calls, not blocked
        block(tools, HOST_DOWN[2])
        tool(store)
        self.assertIsNone(stop(store, s))

    def test_2b_a_new_progress_step_resets_the_count(self):
        tools, store, s = running()
        block(tools, HOST_DOWN[0])
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        block(tools, HOST_DOWN[1])
        tools.goal_progress({"goal_id": "g1", "step": "tried port 2222", "next": "try the console"}, **IDS)
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        self.assertEqual(store.load(KEY).owner_wait_rounds, 0)
        tool(store)  # the block of g1 stays noted; this round has no progress
        self.assertEqual(stop(store, s)["action"], "continue")
        tool(store)
        self.assertIsNone(stop(store, s))
        assert_owner_wait(self, store)
        self.assertNotEqual(store.load(KEY).goal("g1").status, "blocked")  # not 3 block calls

    def test_2c_a_pending_list_change_as_the_only_way_waits_for_the_owner(self):
        tools, store, s = running()
        out = tools.goal_set({"goals": [{"id": "g9", "text": "new goal", "check": "c9"}]}, **IDS)
        self.assertIn("recorded in the deferred list", out)
        state = store.load(KEY)
        self.assertEqual([(e["kind"], e["shown"]) for e in state.deferred], [("change", False)])
        tool(store)
        self.assertEqual(stop(store, s)["action"], "continue")
        tool(store)
        self.assertIsNone(stop(store, s))
        assert_owner_wait(self, store)
        self.assertIn("new goal", answer(store, s))

    def test_2d_similar_counts_a_reworded_reason_as_the_same_obstacle(self):
        self.assertTrue(_similar(HOST_DOWN[1], HOST_DOWN[2]))
        self.assertFalse(_similar(HOST_DOWN[0], OTHER))
        self.assertFalse(_similar("", HOST_DOWN[0]))


class ApprovalWithGoTest(unittest.TestCase):
    def setUp(self):
        goal_loop._turn_marks().clear()

    def test_3a_list_ok_go_in_one_message_starts_the_loop(self):
        for earlier in (False, True):
            with self.subTest(earlier_list=earlier):
                tools, store, _, s = setup()
                goals = {"goals": [{"id": "g1", "text": "a", "check": "c1"}]}
                if earlier:
                    tools.goal_set(goals, **IDS)
                    answer(store, s)  # the turn that showed the list ends
                owner(store, "list ok go", settings=s)
                tools.goal_set(goals, **IDS)
                self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))
                self.assertTrue(store.load(KEY).go)
                self.assertIn("go", ledger_events(store))

    def test_3b_a_fresh_list_without_a_go_in_the_owner_message_is_refused(self):
        tools, store, _, s = setup()
        owner(store, "please plan this", settings=s)
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_3b_a_go_of_an_ended_turn_or_a_follow_up_run_does_not_carry(self):
        tools, store, _, s = setup()
        owner(store, "go", settings=s)
        answer(store, s)  # that turn ends
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c1"}]}, **IDS)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)
        owner(store, "go", settings=s)
        text = resume.RESUME_PREFIX + "n1] go on"
        make_pre_llm_call(store, s)(session_id="s1", platform="api_server", user_message=text,
                                    conversation_history=[{"role": "user", "content": text}])
        tools.goal_set({"goals": [{"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
        self.assertEqual(tools.goal_go({}, **IDS), GO_REFUSED)

    def test_3c_a_changed_list_with_a_go_still_needs_a_later_approval(self):
        tools, store, s = running()
        owner(store, "change g1 to y, go", settings=s)
        changed = {"goals": [{"id": "g1", "text": "y", "check": "c1"}]}
        for _ in range(2):
            self.assertFalse(tools.goal_set(changed, **IDS).startswith("Goal list changed"))
            self.assertEqual(store.load(KEY).goal("g1").text, "a")
        owner(store, "ok, change it", settings=s)
        self.assertTrue(tools.goal_set(changed, **IDS).startswith("Goal list changed"))
        self.assertEqual(store.load(KEY).goal("g1").text, "y")


if __name__ == "__main__":
    unittest.main()
