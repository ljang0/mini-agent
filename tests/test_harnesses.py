"""The harness declarations, and the guards that keep them honest.

A malformed topology has to fail where it is written, not halfway through a
paid run, so every rejection below is a guard the registry depends on.
"""

from __future__ import annotations

import unittest

from mini_agent.harnesses import HARNESSES, harness_names, load_harness
from mini_agent.harnesses.base import Harness, Role


class RoleTests(unittest.TestCase):
    def test_an_unknown_action_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown agent action"):
            Role(actions=("teleport",))

    def test_a_repeated_action_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            Role(actions=("send", "send"))

    def test_a_role_that_can_do_nothing_is_refused(self) -> None:
        """No domain tools and no actions is an agent with no way to act."""

        with self.assertRaisesRegex(ValueError, "at least one action"):
            Role(domain_tools=False)

    def test_capabilities_are_sorted_for_the_spec(self) -> None:
        self.assertEqual(Role(actions=("send", "inbox")).capabilities,
                         ("inbox", "send"))


class HarnessTests(unittest.TestCase):
    def test_a_harness_cannot_name_a_role_it_lacks(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no role"):
            Harness(name="broken", roles={"solo": Role()}, lead="missing")

    def test_a_sized_harness_must_say_what_it_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must declare the role it seeds"):
            Harness(name="broken", roles={"solo": Role()}, lead="solo",
                    sizes=(3,))

    def test_team_size_is_validated_against_what_the_harness_accepts(self) -> None:
        team = load_harness("fixed-team")
        self.assertEqual(team.team_size(None), 3)     # first declared size
        self.assertEqual(team.team_size(10), 10)
        with self.assertRaisesRegex(ValueError, "accepts --team-size"):
            team.team_size(4)

    def test_team_size_does_not_apply_to_an_unsized_harness(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not apply"):
            load_harness("single").team_size(3)
        self.assertEqual(load_harness("single").team_size(None), 1)

    def test_an_unknown_harness_names_the_alternatives(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed-team"):
            load_harness("nope")

    def test_every_registered_harness_resolves_a_role_for_lead_and_child(
        self,
    ) -> None:
        for name in harness_names():
            harness = HARNESSES[name]
            lead = harness.role_of("/root", root_id="/root")
            child = harness.role_of("/root/1", root_id="/root")
            self.assertIn(lead, harness.roles.values(), name)
            self.assertIn(child, harness.roles.values(), name)

    def test_seeds_carry_the_task_verbatim(self) -> None:
        """A peer team shares the task; only delegated agents get instructions."""

        seeds = load_harness("fixed-team").seeds(size=3, task="the task")
        self.assertEqual([s[0] for s in seeds], ["peer-2", "peer-3"])
        self.assertTrue(all(s[1] == "the task" for s in seeds))
        self.assertEqual(load_harness("single").seeds(size=1, task="t"), ())


if __name__ == "__main__":
    unittest.main()
