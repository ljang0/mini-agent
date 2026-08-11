from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class KimiCodeUpstreamHarness(Harness):
    """Treat one pinned Kimi Code source run as one external session tree."""

    name = "kimi_code_upstream"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/kimi-code/root",
                role="kimi_code_upstream_session",
                prompt=prompt,
                metadata={
                    "external_session_tree": True,
                    "task_tools": False,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        events = raw.get("events")
        tool_calls = 0
        swarm_calls = 0
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                calls = event.get("tool_calls")
                if not isinstance(calls, list):
                    continue
                tool_calls += len(calls)
                for call in calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if isinstance(function, Mapping) and function.get("name") == (
                        "AgentSwarm"
                    ):
                        swarm_calls += 1
        return response.text, {
            "released_runtime_adapter": True,
            "external_session_tree": True,
            "shared_budget_ledger": True,
            "outer_model_calls": 1,
            "underlying_model_calls_observed": False,
            "whole_tree_usage_reported_by_upstream": False,
            "whole_tree_usage_independently_verified": False,
            "usage_scope": "not emitted by Kimi stream-json; unknown",
            "domain_tools_injected": False,
            "source_entrypoint_executed": True,
            "upstream_tool_calls_observed": tool_calls,
            "agent_swarm_calls_observed": swarm_calls,
            "workspace": raw.get("workspace"),
            "fidelity": "upstream_runtime_adapter",
            "exactness_scope": (
                "tracked_revision_tree_with_caller_installed_dependency_runtime"
            ),
            "caller_worktree_executed": True,
            "tracked_source_tree_content_verified": True,
            "ignored_or_generated_dependency_content_verified": False,
            "adversarial_full_runtime_content_attestation_verified": False,
            "source_or_protocol_pin_verified": True,
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }
