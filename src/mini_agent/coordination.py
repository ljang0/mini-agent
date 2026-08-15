"""What a team spent, per agent, and what the coordination cost.

The scheduler already records enough to answer this: the ledger tallies each
agent's spend, and the trace carries every message with its sender, recipient
and size, every agent's spawn and terminal event, and a hash of every tool
call's arguments. Nothing aggregated it, so a multi-agent run reported one
number for the whole team and coordination overhead was invisible.

Three costs a spend total cannot show:

- **Idle time.** An agent blocked on its inbox bills nothing and still occupies
  a slot. Without it, a topology that parks subagents looks free. Idle is the
  lifespan left over once both model calls and tool execution are subtracted —
  tool time is work, and counting it as idle would libel the busiest agents.
- **Duplicate work.** Two agents issuing the identical tool call did the same
  work twice; the trace hashes arguments already, so this needs no new
  instrumentation.
- **Dropped messages.** Coordination that did not arrive.

Everything here is derived from recorded events, so the same summary can be
recomputed from ``trace.jsonl`` long after the run.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .execution import BudgetLedger


SCHEMA = "mini-agent-coordination-v2"

# Failing to start is terminal too, and is the one an agent never recovers
# from; leaving it out reported such an agent as having no status at all.
_TERMINAL = (
    "agent_completed",
    "agent_failed",
    "agent_cancelled",
    "agent_start_failed",
)

# A cancelled call still consumed time. Peers are cancelled whenever a lead
# finishes first, so on a team run this is the common ending, not a rare one.
_CALL_ENDED = ("model_call_completed", "model_call_failed", "model_call_cancelled")

# Tool execution is work, not waiting. Counting it as idle would report a
# sixty-second `bash` command as sixty seconds of an agent doing nothing, and
# would overstate idle time worst on exactly the topologies that use tools most.
_TOOL_ENDED = ("tool_call_completed", "tool_call_failed", "tool_call_cancelled")


def _empty() -> dict[str, Any]:
    return {
        "messages_sent": 0,
        "message_bytes_sent": 0,
        "messages_received": 0,
        "messages_dropped": 0,
        "active_seconds": 0.0,
        "tool_seconds": 0.0,
        # An agent's own lifetime, and how much of it was spent neither
        # modelling nor running a tool. Idle time is the cost a topology pays
        # for coordination that spend alone cannot show: an agent blocked on
        # its inbox bills nothing and still occupies a slot.
        "lifespan_seconds": None,
        "idle_seconds": None,
        "tool_calls_duplicated": 0,
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
    tool_started: dict[str, float] = {}
    spawned: dict[str, float] = {}
    # Which agents issued each distinct tool call. Identical arguments from two
    # agents is the observable form of duplicate work; the trace already hashes
    # them, so this needs no new instrumentation.
    calls: dict[tuple[str, str], set[str]] = {}
    for event in events:
        name = event.get("event")
        agent_id = event.get("agent_id")
        data = event.get("data") or {}
        elapsed = event.get("elapsed_seconds")
        record = entry(agent_id)
        if record is None:
            continue
        # A board post is coordination whether or not it names a recipient, so
        # it is counted alongside messages; a topology that talks only through
        # the board would otherwise report as having said nothing.
        if name in ("message_sent", "board_post"):
            record["messages_sent"] += 1
            record["message_bytes_sent"] += int(data.get("content_bytes", 0))
        elif name in ("messages_read", "board_read"):
            record["messages_received"] += int(data.get("count", 0))
        elif name == "message_dropped":
            record["messages_dropped"] += 1
        elif name == "agent_spawned" and isinstance(elapsed, (int, float)):
            spawned.setdefault(str(agent_id), float(elapsed))
        elif name == "model_call_started" and isinstance(elapsed, (int, float)):
            started[str(agent_id)] = float(elapsed)
        elif name in _CALL_ENDED:
            begin = started.pop(str(agent_id), None)
            if begin is not None and isinstance(elapsed, (int, float)):
                record["active_seconds"] += round(float(elapsed) - begin, 6)
        elif name == "tool_call_started":
            if isinstance(elapsed, (int, float)):
                tool_started[str(agent_id)] = float(elapsed)
            digest = data.get("arguments_sha256")
            tool = data.get("tool")
            if isinstance(digest, str) and isinstance(tool, str):
                calls.setdefault((tool, digest), set()).add(str(agent_id))
        elif name in _TOOL_ENDED:
            begin = tool_started.pop(str(agent_id), None)
            if begin is not None and isinstance(elapsed, (int, float)):
                record["tool_seconds"] += round(float(elapsed) - begin, 6)
        elif name in _TERMINAL:
            record["status"] = name[len("agent_") :]
            begin = spawned.get(str(agent_id))
            if begin is not None and isinstance(elapsed, (int, float)):
                lifespan = round(float(elapsed) - begin, 6)
                record["lifespan_seconds"] = lifespan
                # Clamped because concurrent calls can overlap the lifespan;
                # a negative idle time would be an artifact, not a finding.
                busy = float(record["active_seconds"]) + float(record["tool_seconds"])
                record["idle_seconds"] = round(max(0.0, lifespan - busy), 6)

    shared = [issuers for issuers in calls.values() if len(issuers) > 1]
    for issuers in shared:
        for issuer in issuers:
            duplicated = entry(issuer)
            if duplicated is not None:
                duplicated["tool_calls_duplicated"] += 1

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
            "idle_seconds": _total(agents, "idle_seconds"),
            "active_seconds": _total(agents, "active_seconds"),
            "tool_seconds": _total(agents, "tool_seconds"),
            # Distinct tool calls more than one agent made. Counted once per
            # call rather than once per agent, so it reads as "this much work
            # was done twice" instead of double-counting the agents that did it.
            "duplicate_tool_calls": len(shared),
        },
    }


def _total(agents: Mapping[str, Mapping[str, Any]], field: str) -> float:
    return round(
        sum(
            float(item[field])
            for item in agents.values()
            if isinstance(item.get(field), (int, float))
        ),
        6,
    )
