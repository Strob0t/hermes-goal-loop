"""Tests of E1 A (goals follow parent_session_id) and C1 (a blocked goal is shown again only for a
new reason)."""
import tempfile
import unittest

from goal_loop import _Tools, make_transform_llm_output
from goal_loop.state import Goal, GoalState, GoalStore
from tests.test_v03 import IDS, Ctx, Settings, owner, set_and_go


def store_with(parents):
    return GoalStore(tempfile.mkdtemp(prefix="gl-adopt-"), parent_of=parents.get)


class AdoptParentTest(unittest.TestCase):
    def _old(self, store, key="chat:old", go=True):
        state = GoalState(chat_key=key, go=go,
                          goals=[Goal(id="v1", text="alt", check="c"),
                                 Goal(id="v2", text="complete", check="c", done=True, status="done")])
        store.save(state)
        return state

    def test_a_child_session_takes_over_the_goals_of_its_parent(self):
        store = store_with({"new": "old"})
        self._old(store)
        state = store.load("chat:new")
        self.assertEqual([g.id for g in state.goals], ["v1", "v2"])
        self.assertTrue(state.go)
        self.assertEqual(state.adopted_from, "chat:old")
        self.assertEqual(state.chat_key, "chat:new")
        self.assertTrue(store.exists("chat:new"))

    def test_the_parent_state_is_marked_as_taken_over_in_state_and_ledger(self):
        store = store_with({"new": "old"})
        self._old(store)
        store.load("chat:new")
        old = store.load("chat:old")
        self.assertEqual(old.adopted_by, "chat:new")
        self.assertFalse(old.go)
        self.assertEqual([e["event"] for e in store.read_ledger("chat:old")], ["adopted_by"])
        new_events = store.read_ledger("chat:new")
        self.assertEqual(new_events[0]["event"], "adopted_from")
        self.assertEqual(new_events[0]["from_chat"], "chat:old")

    def test_a_taken_over_state_is_not_taken_over_a_second_time(self):
        store = store_with({"new": "old", "other": "old"})
        self._old(store)
        store.load("chat:new")
        self.assertEqual(store.load("chat:other").goals, [])
        self.assertFalse(store.exists("chat:other"))

    def test_the_walk_goes_up_to_the_nearest_ancestor_with_goals(self):
        store = store_with({"c": "b", "b": "a"})
        self._old(store, "chat:a")
        self.assertEqual(store.load("chat:c").adopted_from, "chat:a")

    def test_the_session_map_names_the_chat_of_a_parent_session(self):
        store = store_with({"new": "old-session"})
        self._old(store, "chat:conv-root")
        store.remember_session("old-session", "chat:conv-root")
        self.assertEqual(store.load("chat:new").adopted_from, "chat:conv-root")

    def test_without_a_parent_nothing_is_taken_over(self):
        store = store_with({})
        self._old(store)
        self.assertEqual(store.load("chat:new").goals, [])
        self.assertEqual(store.read_ledger("chat:new"), [])

    def test_an_existing_state_of_the_child_is_kept(self):
        store = store_with({"new": "old"})
        self._old(store)
        store.save(GoalState(chat_key="chat:new", goals=[Goal(id="n1", text="neu", check="c")]))
        self.assertEqual([g.id for g in store.load("chat:new").goals], ["n1"])
        self.assertEqual(store.load("chat:old").adopted_by, "")

    def test_a_parent_note_of_an_importer_is_followed(self):
        store = GoalStore(tempfile.mkdtemp(prefix="gl-adopt-"))
        self._old(store)
        store._write_side("parent", "new", {"parent_session_id": "old"})
        self.assertEqual(store.load("chat:new").adopted_from, "chat:old")

    def test_goal_status_of_the_child_shows_the_goals_of_the_parent(self):
        store = store_with({"s1": "old"})
        self._old(store)
        tools = _Tools(store, Ctx(), Settings(Ctx()))
        text = tools.goal_status({}, **IDS)
        self.assertIn("v1", text)
        self.assertIn("v2", text)


class BlockShownOnceTest(unittest.TestCase):
    def _blocked(self, reason):
        store = GoalStore(tempfile.mkdtemp(prefix="gl-c1-"))
        ctx = Ctx(settings={"block_min_rounds": 1})
        s = Settings(ctx)
        tools = _Tools(store, ctx, s)
        # 0.4.10 (R2): one goal only, so that its block is owner-bound and not deferred
        set_and_go(tools, store, goals=(("g1", "a", "c1"),))
        tools.goal_block({"goal_id": "g1", "reason": reason}, **IDS)
        out = make_transform_llm_output(store, s)(response_text="Report.", **IDS)
        return tools, store, s, out

    def test_the_first_block_is_shown(self):
        _, _, _, out = self._blocked("docker.sock permission denied on homeserver")
        self.assertIn("g1:", out)

    def test_the_same_reason_again_is_not_shown_again(self):
        tools, store, s, _ = self._blocked("docker.sock permission denied on homeserver")
        owner(store, "go", settings=s)
        tools.goal_block({"goal_id": "g1", "reason": "docker.sock permission denied on homeserver again"},
                         **IDS)
        state = store.load(tools._load(IDS).chat_key)
        self.assertTrue(state.goal("g1").block_shown)
        out = make_transform_llm_output(store, s)(response_text="Report two.", **IDS)
        self.assertNotIn("g1:", out or "")
        self.assertFalse(store.read_ledger(state.chat_key)[-1]["shown_again"])

    def test_a_new_reason_shows_the_goal_again(self):
        tools, store, s, _ = self._blocked("docker.sock permission denied on homeserver")
        owner(store, "go", settings=s)
        tools.goal_block({"goal_id": "g1", "reason": "GitHub token expired, push refused with 401"}, **IDS)
        state = store.load(tools._load(IDS).chat_key)
        self.assertFalse(state.goal("g1").block_shown)
        out = make_transform_llm_output(store, s)(response_text="Report two.", **IDS)
        self.assertIn("g1:", out)
        self.assertIn("GitHub token expired", out)


if __name__ == "__main__":
    unittest.main()
