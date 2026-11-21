"""Tests of 0.4.4: todo_list is no progress, the restored task list is no owner message, an owner message
that interrupts a running turn counts for the go and the change approval."""
import unittest

from goal_loop import (
    _NO_PROGRESS_TOOLS, _chat_key, make_post_tool_call, make_pre_api_request, make_pre_llm_call,
)

try:  # discover -s tests -t . (package) or discover -s tests (top level)
    from tests.test_043 import IDS, ledger_events, owner, set_and_go, setup, stop, tool
except ImportError:
    from test_043 import IDS, ledger_events, owner, set_and_go, setup, stop, tool

RESTORE = ("[Your active task list was preserved across context compression]\n"
           "- [>] c1. claim r11 (in_progress)\n- [ ] r20. move test states (pending)")
REDIRECT = "[Context from the interrupted assistant response]\n[This response was interrupted by a user correction.]"


def history_with_redirect(text, earlier=()):
    rows = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "working"}]
    rows += list(earlier)
    rows.append({"role": "user", "content": text, "api_content": f"{REDIRECT}\n\n{text}"})
    return rows


def refused_change(tools):
    """After the go, a changed list is refused and opens the change gate with 0 owner messages."""
    out = tools.goal_set({"goals": [{"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
    assert not out.startswith("Goal list changed"), out
    assert tools.store.gate_count(_chat_key("s1"), "change") == 0


def api_request(store, settings, history, session="s1"):
    return make_pre_api_request(store, settings)(session_id=session, platform="api_server",
                                                 conversation_history=history, model="auto")


class TodoListIsNoProgressTest(unittest.TestCase):
    def test_todo_list_is_in_the_no_progress_tools(self):
        self.assertIn("todo_list", _NO_PROGRESS_TOOLS)

    def test_a_run_with_only_todo_list_calls_pauses_after_idle_pause_after(self):
        tools, store, _, s = setup(idle_pause_after=3)
        set_and_go(tools, store)
        stop(store, s)                       # baseline nudge
        results = []
        for _ in range(3):
            tool(store, "todo_list")         # only task-list updates between the nudges
            results.append(stop(store, s))
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[-1])       # third idle nudge -> pause
        self.assertEqual(store.read_ledger(_chat_key("s1"))[-1]["event"], "paused_idle")
        self.assertEqual(store.load(_chat_key("s1")).goal("g1").status, "paused")

    def test_a_real_tool_call_still_resets_the_idle_count(self):
        tools, store, _, s = setup(idle_pause_after=3)
        set_and_go(tools, store)
        stop(store, s)
        for _ in range(5):
            tool(store, "terminal")
            self.assertIsNotNone(stop(store, s))
        self.assertNotIn("paused_idle", ledger_events(store))


class RestoredTaskListTest(unittest.TestCase):
    def test_the_restored_task_list_is_no_owner_message_and_gives_no_go(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        before = store.gate_count(_chat_key("s1"))
        owner(store, RESTORE, settings=s)
        self.assertEqual(store.gate_count(_chat_key("s1")), before)
        self.assertFalse(tools.goal_go({}, **IDS).startswith("Go recorded"))

    def test_the_restored_task_list_does_not_count_as_a_change_approval(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        owner(store, RESTORE, settings=s)
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 0)
        self.assertFalse(tools.goal_set({"goals": [{"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
                         .startswith("Goal list changed"))


class OwnerRedirectTest(unittest.TestCase):
    def test_an_interrupting_owner_message_counts_for_the_go(self):
        tools, store, _, s = setup()
        tools.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **IDS)
        self.assertFalse(tools.goal_go({}, **IDS).startswith("Go recorded"))
        api_request(store, s, history_with_redirect("go"))
        self.assertTrue(tools.goal_go({}, **IDS).startswith("Go recorded"))
        self.assertIn("owner_redirect_counted", ledger_events(store))

    def test_an_interrupting_owner_message_counts_for_the_change_approval(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        api_request(store, s, history_with_redirect("Freigabe"))
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 1)
        self.assertTrue(tools.goal_set({"goals": [{"id": "g2", "text": "b", "check": "c2"}]}, **IDS)
                        .startswith("Goal list changed"))

    def test_the_same_redirect_row_counts_one_time(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        history = history_with_redirect("Freigabe")
        for _ in range(4):                   # pre_api_request runs before each API call of the turn
            api_request(store, s, history)
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 1)
        self.assertEqual(ledger_events(store).count("owner_redirect_counted"), 1)

    def test_a_second_redirect_counts_again(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        first = history_with_redirect("A")
        api_request(store, s, first)
        second = first + [{"role": "assistant", "content": "ok"}]
        second = history_with_redirect("B", earlier=second[2:])
        api_request(store, s, second)
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 2)

    def test_a_normal_user_row_without_the_prefix_is_not_counted_here(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        api_request(store, s, [{"role": "user", "content": "Freigabe"}])
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 0)

    def test_a_follow_up_run_row_is_not_counted(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        api_request(store, s, [{"role": "user", "content": "x", "api_content": "[goal-loop resume abc] x"}])
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 0)

    def test_a_stop_word_as_redirect_pauses_the_goals(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        api_request(store, s, history_with_redirect("stopp"))
        self.assertIn("paused_by_owner", ledger_events(store))

    def test_the_subagent_platform_is_excluded(self):
        tools, store, _, s = setup()
        set_and_go(tools, store)
        refused_change(tools)
        make_pre_api_request(store, s)(session_id="s1", platform="subagent",
                                       conversation_history=history_with_redirect("Freigabe"))
        self.assertEqual(store.gate_count(_chat_key("s1"), "change"), 0)


if __name__ == "__main__":
    unittest.main()
