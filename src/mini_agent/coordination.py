"""What a team spent, per agent, and how much of it was talking.

The scheduler already records enough to answer this: the ledger tallies each
agent's spend, and the trace carries every message with its sender, recipient
and size. Nothing aggregated it, so a multi-agent run reported one number for
the whole team and coordination overhead was invisible.

Everything here is derived from recorded events, so the same summary can be
recomputed from ``trace.jsonl`` long after the run.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .runtime import BudgetLedger


SCHEMA = "mini-agent-coordination-v1"

_TERMINAL = ("agent_completed", "agent_failed", "agent_cancelled")


def _empty() -> dict[str, Any]:
    return {
        "messages_sent": 0,
        "message_bytes_sent": 0,
        "messages_received": 0,
        "messages_dropped": 0,
        "active_seconds": 0.0,
        "status": None,
    }


def coordination_summary(
    events: Iterable[Mapping[str, Any]],
    *,
    ledger: BudgetLedger,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Summarize one team's spend and communication.

    ``events`` are trace events as mappings, so this works equally on a live
    ``TraceRecorder.events`` slice and on parsed ``trace.jsonl`` lines.
    """

    agents: dict[str, dict[str, Any]] = {
        agent_id: {**_empty(), **dict(ledger.agent_snapshot(agent_id))}
        for agent_id in ledger.agent_ids(prefix=prefix)
    }

    def entry(agent_id: Any) -> dict[str, Any] | None:
        if not isinstance(agent_id, str) or not agent_id:
            return None
        if agent_id not in agents:
            if prefix is not None and not (
                agent_id == prefix or agent_id.startswith(prefix.rstrip("/") + "/")
            ):
                return None
            agents[agent_id] = _empty()
        return agents[agent_id]

    started: dict[str, float] = {}
    for event in events:
        name = event.get("event")
        agent_id = event.get("agent_id")
        data = event.get("data") or {}
        elapsed = event.get("elapsed_seconds")
        record = entry(agent_id)
        if record is None:
            continue
        if name == "message_sent":
            record["messages_sent"] += 1
            record["message_bytes_sent"] += int(data.get("content_bytes", 0))
        elif name == "messages_read":
            record["messages_received"] += int(data.get("count", 0))
        elif name == "message_dropped":
            record["messages_dropped"] += 1
        elif name == "model_call_started" and isinstance(elapsed, (int, float)):
            started[str(agent_id)] = float(elapsed)
        elif name in ("model_call_completed", "model_call_failed"):
            begin = started.pop(str(agent_id), None)
            if begin is not None and isinstance(elapsed, (int, float)):
                record["active_seconds"] += round(float(elapsed) - begin, 6)
        elif name in _TERMINAL:
            record["status"] = name[len("agent_") :]

    return {
        "schema": SCHEMA,
        "agents": agents,
        "totals": {
            "agents": len(agents),
            "messages": sum(item["messages_sent"] for item in agents.values()),
            "message_bytes": sum(
                item["message_bytes_sent"] for item in agents.values()
            ),
            "messages_dropped": sum(
                item["messages_dropped"] for item in agents.values()
            ),
        },
    }
