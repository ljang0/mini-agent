from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class AnthropicManagedAgentsHarness(Harness):
    """One exact public invocation of an Anthropic Managed Agents session.

    Agent construction, coordinator topology, tools, thread scheduling, and the
    sandbox are configured on the referenced remote agent/environment.  This
    harness deliberately does not recreate any of those behaviors locally.
    """

    name = "anthropic_managed_agents"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/anthropic-managed/coordinator",
                role="anthropic_managed_session",
                prompt=prompt,
                metadata={
                    "anthropic_managed_agents": True,
                    "server_owned_tool_loop": True,
                },
            )
        )

        raw = response.raw if isinstance(response.raw, Mapping) else {}
        session = raw.get("session")
        session = dict(session) if isinstance(session, Mapping) else {}
        raw_events = raw.get("events")
        events = self._events(raw_events)
        event_counts = Counter(
            event["type"] for event in events if isinstance(event.get("type"), str)
        )
        thread_ids = {
            thread_id
            for event in events
            for thread_id in [self._thread_id(event)]
            if thread_id is not None
        }
        stop_reason = self._stop_reason(events)
        session_id = raw.get("session_id")
        if not isinstance(session_id, str):
            session_id = (
                session.get("id") if isinstance(session.get("id"), str) else None
            )
        underlying_model_calls = event_counts.get("span.model_request_end", 0)

        await context.trace.emit(
            "managed_session_observed",
            agent_id="/anthropic-managed/coordinator",
            role="anthropic_managed_session",
            data={
                "session_id": session_id,
                "status": session.get("status"),
                "stop_reason": stop_reason,
                "event_count": len(events),
                "thread_count_observed": len(thread_ids),
                "underlying_model_calls_observed": underlying_model_calls,
                "cleanup": raw.get("cleanup"),
                "resolved_agent_sha256": raw.get("resolved_agent_sha256"),
                "coordinator_roster_size": raw.get("coordinator_roster_size"),
            },
        )

        resolved_agent = session.get("agent")
        resolved_agent = (
            dict(resolved_agent) if isinstance(resolved_agent, Mapping) else {}
        )
        return response.text, {
            "hosted_runtime": True,
            "native_multiagent": True,
            "public_invocation_exact": True,
            "server_scheduler_open": False,
            "server_owned_tool_loop": True,
            "local_tool_loop_enabled": False,
            "usage_scope": "authoritative_full_tree_session_usage",
            "session_id": session_id,
            "session_status": session.get("status"),
            "session_stop_reason": stop_reason,
            "environment_id": session.get("environment_id"),
            "resolved_agent_id": resolved_agent.get("id"),
            "resolved_agent_version": resolved_agent.get("version"),
            "resolved_agent_sha256": raw.get("resolved_agent_sha256"),
            "coordinator_roster_size": raw.get("coordinator_roster_size"),
            "event_count": len(events),
            "event_type_counts": dict(sorted(event_counts.items())),
            "thread_count_observed": len(thread_ids),
            "underlying_model_calls": underlying_model_calls,
            "underlying_model_calls_observed": underlying_model_calls > 0,
            "cleanup": raw.get("cleanup"),
            "fidelity": "exact_public_managed_session_boundary",
        }

    @staticmethod
    def _events(raw: Any) -> list[Mapping[str, Any]]:
        if not isinstance(raw, Mapping):
            return []
        data = raw.get("data")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            return []
        return [dict(item) for item in data if isinstance(item, Mapping)]

    @staticmethod
    def _thread_id(event: Mapping[str, Any]) -> str | None:
        direct = event.get("session_thread_id")
        if isinstance(direct, str) and direct:
            return direct
        thread = event.get("thread")
        if isinstance(thread, Mapping):
            nested = thread.get("id")
            if isinstance(nested, str) and nested:
                return nested
        nested = event.get("thread_id")
        return nested if isinstance(nested, str) and nested else None

    @staticmethod
    def _stop_reason(events: Sequence[Mapping[str, Any]]) -> str | None:
        for event in reversed(events):
            if event.get("type") != "session.status_idle":
                continue
            reason = event.get("stop_reason")
            if isinstance(reason, str):
                return reason
            if isinstance(reason, Mapping) and isinstance(reason.get("type"), str):
                return str(reason["type"])
        return None
