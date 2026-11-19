"""Tests of the plugin parts taken from the goal-loop-quiet order (owner decisions 2a and 4a)."""
import tempfile
import unittest

from goal_loop import Settings, _Tools, _chat_key, make_pre_llm_call, make_transform_llm_output
from goal_loop.state import GoalStore

CHECKS = {"checks": [{"command": "c", "result": "passed", "exitCode": 0}]}


class Ctx:
    def __init__(self):
        class Llm:
            def complete(self, messages, **kw):
                class R:
                    text = "VERDICT: done"
                return R()
        self.llm = Llm()
        self.subagent_lifecycle = None

    def get_config(self, key, default=None):
        return default


def tools():
    store = GoalStore(tempfile.mkdtemp(prefix="glq-"))
    ctx = Ctx()
    return _Tools(store, ctx, Settings(ctx)), store


def owner(store, session, text="go", **extra):
    make_pre_llm_call(store)(session_id=session, platform="api_server", user_message=text,
                             conversation_history=[{"role": "user", "content": text}], **extra)


class GateOnConversationIdTest(unittest.TestCase):
    """2a: the go gate follows the chat across a session rotation (compression.in_place false)."""

    def test_owner_message_in_a_rotated_session_opens_the_go(self):
        t, store = tools()
        t.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]},
                   session_id="s-old", conversation_id="conv-1")
        # the compaction rotated the session; a tool call of the new session records the chat
        t.goal_status({}, session_id="s-new", conversation_id="conv-1")
        owner(store, "s-new")
        self.assertTrue(t.goal_go({}, session_id="s-new", conversation_id="conv-1")
                        .startswith("Go recorded"))

    def test_a_conversation_id_in_the_hook_is_used_directly(self):
        t, store = tools()
        t.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]},
                   session_id="s1", conversation_id="conv-2")
        owner(store, "s-unknown", conversation_id="conv-2")
        self.assertTrue(t.goal_go({}, session_id="s9", conversation_id="conv-2")
                        .startswith("Go recorded"))

    def test_another_chat_does_not_open_the_go(self):
        t, store = tools()
        t.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]},
                   session_id="s1", conversation_id="conv-3")
        owner(store, "s2", conversation_id="conv-other")
        self.assertTrue(t.goal_go({}, session_id="s1", conversation_id="conv-3").startswith("error"))


class DecisionAfterFinalListTest(unittest.TestCase):
    """4a: a decision recorded after the final list was shown still appears one time."""

    def test_decision_after_final_list(self):
        t, store = tools()
        ids = {"session_id": "s1", "conversation_id": "c1", "platform": "api_server"}
        transform = make_transform_llm_output(store, Settings(overrides={"paused_hint": False}))
        t.goal_set({"goals": [{"id": "g1", "text": "a", "check": "c"}]}, **ids)
        transform(response_text="LIST", **ids)
        owner(store, "s1", conversation_id="c1")
        t.goal_go({}, **ids)
        t.goal_claim({"goal_id": "g1", "claim": {"summary": "s", **CHECKS}}, **ids)
        self.assertIn("- [x] g1", transform(response_text="FINAL", **ids))
        t.goal_decision({"text": "Follow-up: A, B or C. Recommend A."}, **ids)
        out = transform(response_text="LATER", **ids)
        self.assertIn("Follow-up: A, B or C", out)
        self.assertNotIn("- [x] g1", out)
        self.assertIsNone(transform(response_text="AGAIN", **ids))
        self.assertTrue(store.load(_chat_key("c1")).final_shown)


if __name__ == "__main__":
    unittest.main()
