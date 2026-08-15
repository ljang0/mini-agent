"""Coordination reporting: what each agent spent, and what it spent talking."""

from __future__ import annotations

import unittest

from mini_agent.coordination import SCHEMA, coordination_summary
from mini_agent.execution import BudgetLedger
from mini_agent.types import BudgetLimits, Usage


def event(name: str, agent_id: str, **data: object) -> dict[str, object]:
    elapsed = data.pop("elapsed", 0.0)
    return {
        "event": name,
        "agent_id": agent_id,
        "elapsed_seconds": elapsed,
        "data": data,
    }


class CoordinationSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def _ledger(self) -> BudgetLedger:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=50))
        await ledger.reserve_call("/eval/t/root")
        await ledger.record(Usage(input_tokens=5), "/eval/t/root")
        await ledger.reserve_call("/eval/t/root/1")
        await ledger.record(Usage(input_tokens=7), "/eval/t/root/1")
        return ledger

    async def test_spend_and_messages_are_reported_per_agent(self) -> None:
        ledger = await self._ledger()
        events = [
            event("message_sent", "/eval/t/root", recipient="/eval/t/root/1",
                  content_bytes=12),
            event("messages_read", "/eval/t/root/1", count=1),
            event("message_sent", "/eval/t/root/1", recipient="/eval/t/root",
                  content_bytes=30),
            event("agent_completed", "/eval/t/root/1", steps=2),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["schema"], SCHEMA)
        root = summary["agents"]["/eval/t/root"]
        child = summary["agents"]["/eval/t/root/1"]
        self.assertEqual(root["model_calls"], 1)
        self.assertEqual(root["usage"]["input_tokens"], 5)
        self.assertEqual(root["messages_sent"], 1)
        self.assertEqual(root["message_bytes_sent"], 12)
        self.assertEqual(child["messages_received"], 1)
        self.assertEqual(child["status"], "completed")
        self.assertEqual(summary["totals"]["messages"], 2)
        self.assertEqual(summary["totals"]["message_bytes"], 42)

    async def test_active_seconds_come_from_paired_model_calls(self) -> None:
        ledger = await self._ledger()
        events = [
            event("model_call_started", "/eval/t/root", elapsed=1.0),
            event("model_call_completed", "/eval/t/root", elapsed=2.5),
            event("model_call_started", "/eval/t/root", elapsed=3.0),
            # No completion: an unpaired call must not invent time.
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["agents"]["/eval/t/root"]["active_seconds"], 1.5)

    async def test_another_task_subtree_is_excluded(self) -> None:
        ledger = await self._ledger()
        events = [
            event("message_sent", "/eval/other/root", content_bytes=99),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertNotIn("/eval/other/root", summary["agents"])
        self.assertEqual(summary["totals"]["message_bytes"], 0)


class IdleTimeTests(unittest.IsolatedAsyncioTestCase):
    """Idle time is what an agent cost the team without spending anything."""

    async def _ledger(self) -> BudgetLedger:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=50))
        await ledger.reserve_call("/eval/t/root/1")
        return ledger

    async def test_idle_is_the_lifespan_not_spent_on_a_model_call(self) -> None:
        ledger = await self._ledger()
        events = [
            event("agent_spawned", "/eval/t/root/1", elapsed=1.0),
            event("model_call_started", "/eval/t/root/1", elapsed=2.0),
            event("model_call_completed", "/eval/t/root/1", elapsed=4.0),
            event("agent_completed", "/eval/t/root/1", elapsed=11.0),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        agent = summary["agents"]["/eval/t/root/1"]
        self.assertEqual(agent["lifespan_seconds"], 10.0)
        self.assertEqual(agent["active_seconds"], 2.0)
        self.assertEqual(agent["idle_seconds"], 8.0)
        self.assertEqual(summary["totals"]["idle_seconds"], 8.0)

    async def test_tool_execution_is_work_not_idleness(self) -> None:
        # Counting tool time as idle would report a sixty-second bash command
        # as sixty seconds of an agent doing nothing.
        ledger = await self._ledger()
        events = [
            event("agent_spawned", "/eval/t/root/1", elapsed=0.0),
            event("model_call_started", "/eval/t/root/1", elapsed=0.0),
            event("model_call_completed", "/eval/t/root/1", elapsed=1.0),
            event("tool_call_started", "/eval/t/root/1", elapsed=1.0, tool="bash"),
            event("tool_call_completed", "/eval/t/root/1", elapsed=7.0),
            event("agent_completed", "/eval/t/root/1", elapsed=10.0),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        agent = summary["agents"]["/eval/t/root/1"]
        self.assertEqual(agent["active_seconds"], 1.0)
        self.assertEqual(agent["tool_seconds"], 6.0)
        self.assertEqual(agent["idle_seconds"], 3.0)
        self.assertEqual(summary["totals"]["tool_seconds"], 6.0)

    async def test_an_agent_that_never_terminated_reports_no_lifespan(self) -> None:
        # Inventing a lifespan for an agent whose end is not in the slice would
        # make idle time depend on where the slice was cut.
        ledger = await self._ledger()
        events = [event("agent_spawned", "/eval/t/root/1", elapsed=1.0)]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        agent = summary["agents"]["/eval/t/root/1"]
        self.assertIsNone(agent["lifespan_seconds"])
        self.assertIsNone(agent["idle_seconds"])

    async def test_overlapping_calls_never_report_negative_idle(self) -> None:
        ledger = await self._ledger()
        events = [
            event("agent_spawned", "/eval/t/root/1", elapsed=0.0),
            event("model_call_started", "/eval/t/root/1", elapsed=0.0),
            event("model_call_completed", "/eval/t/root/1", elapsed=5.0),
            event("agent_completed", "/eval/t/root/1", elapsed=3.0),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["agents"]["/eval/t/root/1"]["idle_seconds"], 0.0)


class DuplicateWorkTests(unittest.IsolatedAsyncioTestCase):
    """Two agents issuing the same tool call is duplicate work, observably."""

    async def _ledger(self) -> BudgetLedger:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=50))
        for agent_id in ("/eval/t/root", "/eval/t/root/1"):
            await ledger.reserve_call(agent_id)
        return ledger

    async def test_the_same_call_from_two_agents_is_counted_once(self) -> None:
        ledger = await self._ledger()
        events = [
            event("tool_call_started", "/eval/t/root", tool="bash",
                  arguments_sha256="aa"),
            event("tool_call_started", "/eval/t/root/1", tool="bash",
                  arguments_sha256="aa"),
            event("tool_call_started", "/eval/t/root/1", tool="bash",
                  arguments_sha256="bb"),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["totals"]["duplicate_tool_calls"], 1)
        self.assertEqual(
            summary["agents"]["/eval/t/root"]["tool_calls_duplicated"], 1
        )
        self.assertEqual(
            summary["agents"]["/eval/t/root/1"]["tool_calls_duplicated"], 1
        )

    async def test_one_agent_repeating_itself_is_not_duplicate_work(self) -> None:
        # Retrying your own command is a single agent's business; the metric is
        # about two agents doing the same work unaware of each other.
        ledger = await self._ledger()
        events = [
            event("tool_call_started", "/eval/t/root", tool="bash",
                  arguments_sha256="aa"),
            event("tool_call_started", "/eval/t/root", tool="bash",
                  arguments_sha256="aa"),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["totals"]["duplicate_tool_calls"], 0)

    async def test_the_same_arguments_to_different_tools_are_distinct(self) -> None:
        ledger = await self._ledger()
        events = [
            event("tool_call_started", "/eval/t/root", tool="bash",
                  arguments_sha256="aa"),
            event("tool_call_started", "/eval/t/root/1", tool="agent",
                  arguments_sha256="aa"),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(summary["totals"]["duplicate_tool_calls"], 0)


class CancelledCallTests(unittest.IsolatedAsyncioTestCase):
    """A team run normally ends by cancelling whoever is still working."""

    async def _ledger(self) -> BudgetLedger:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=50))
        await ledger.reserve_call("/eval/t/root/1")
        return ledger

    async def test_a_cancelled_call_still_counts_its_time(self) -> None:
        ledger = await self._ledger()
        events = [
            event("model_call_started", "/eval/t/root/1", elapsed=1.0),
            event("model_call_cancelled", "/eval/t/root/1", elapsed=2.0),
            event("agent_cancelled", "/eval/t/root/1"),
        ]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        agent = summary["agents"]["/eval/t/root/1"]
        self.assertEqual(agent["active_seconds"], 1.0)
        self.assertEqual(agent["status"], "cancelled")

    async def test_an_agent_that_never_started_reports_that(self) -> None:
        ledger = await self._ledger()
        events = [event("agent_start_failed", "/eval/t/root/1", error="OSError")]

        summary = coordination_summary(events, ledger=ledger, prefix="/eval/t")

        self.assertEqual(
            summary["agents"]["/eval/t/root/1"]["status"], "start_failed"
        )


if __name__ == "__main__":
    unittest.main()
