from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class RLMUpstreamHarness(Harness):
    """Adapter target for the pinned upstream ``rlms`` inference runtime.

    The external backend owns the complete RLM/REPL tree. Scaffold Lab observes one
    outer call so the upstream ``UsageSummary`` is charged once to the shared ledger.
    No Scaffold Lab SWE, browser, or computer tools are injected into the upstream
    REPL.
    """

    name = "rlm_upstream"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        external_context = task.context if task.context else task.prompt
        root_prompt = task.prompt if task.context else None
        response = await context.call(
            ModelRequest(
                agent_id="/rlm-upstream/root",
                role="rlm_upstream_session",
                prompt=external_context,
                metadata={
                    "external_session_tree": True,
                    "root_prompt": root_prompt,
                    "task_tools": False,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        raw_calls = raw.get("underlying_model_calls")
        calls_observed = (
            isinstance(raw_calls, int)
            and not isinstance(raw_calls, bool)
            and raw_calls >= 0
        )
        underlying_calls = raw_calls if calls_observed else 0
        return response.text, {
            "released_runtime_adapter": True,
            "external_session_tree": True,
            "shared_budget_ledger": True,
            "outer_model_calls": 1,
            "underlying_model_calls": underlying_calls,
            "underlying_model_calls_observed": calls_observed,
            "underlying_model_calls_are_lower_bound": True,
            "usage_scope": "root RLMChatCompletion.usage_summary lower bound",
            "whole_tree_usage_reported_by_upstream": False,
            "whole_tree_usage_independently_verified": False,
            "cost_known": False,
            "upstream_environment": raw.get("environment"),
            "default_environment_is_docker": raw.get("environment") == "docker",
            "domain_tools_injected": False,
            "swe_tool_parity_claimed": False,
            "fidelity": "upstream_runtime_adapter",
        }
