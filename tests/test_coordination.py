"""Coordination reporting: what each agent spent, and what it spent talking."""

from __future__ import annotations

import unittest

from mini_agent.coordination import SCHEMA, coordination_summary
from mini_agent.runtime import BudgetLedger
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


if __name__ == "__main__":
    unittest.main()
