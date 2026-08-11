from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class PrimeAgentHarness(Harness):
    """Adapter target for the released Prime Agent JSON runtime."""

    name = "prime_agent"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/prime-agent/root",
                role="prime_agent_session",
                prompt=prompt,
                metadata={"external_session_tree": True},
            )
        )
        events = response.raw if isinstance(response.raw, list) else []
        session: Mapping[str, Any] = next(
            (
                event
                for event in events
                if isinstance(event, dict) and event.get("type") == "session"
            ),
            {},
        )
        workspace: Mapping[str, Any] = next(
            (
                event
                for event in events
                if isinstance(event, dict)
                and event.get("type") == "scaffoldlab_workspace"
            ),
            {},
        )
        usage_message_roles: dict[str, int] = {}
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "message_end":
                continue
            message = event.get("message")
            if not isinstance(message, dict) or not isinstance(
                message.get("usage"), dict
            ):
                continue
            role = str(message.get("role", "unknown"))
            usage_message_roles[role] = usage_message_roles.get(role, 0) + 1
        return response.text, {
            "released_runtime_adapter": True,
            "external_session_tree": True,
            "json_events": len(events),
            "underlying_model_calls_observed": False,
            "full_tree_usage_verified": False,
            "prime_json_stream_version": session.get("version"),
            "prime_json_contract_validated": True,
            "session_scope": "single-backend-call",
            "cross_call_session_state_preserved": False,
            "usage_message_roles": usage_message_roles,
            "workspace": dict(workspace),
            "fidelity": "exact_public_protocol",
            "exactness_scope": "published_runtime_protocol_boundary",
        }
