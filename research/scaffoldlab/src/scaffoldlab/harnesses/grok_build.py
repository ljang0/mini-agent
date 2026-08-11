from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class GrokBuildHarness(Harness):
    """Adapter target for xAI's released Grok Build headless runtime."""

    name = "grok_build"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/grok-build/root",
                role="grok_build_session",
                prompt=prompt,
                metadata={"external_session_tree": True},
            )
        )
        result = response.raw if isinstance(response.raw, dict) else {}
        workspace = result.get("_scaffoldlab_workspace")
        model_usage = result.get("modelUsage")
        underlying_calls = 0
        if isinstance(model_usage, dict):
            for value in model_usage.values():
                if not isinstance(value, dict):
                    continue
                calls = value.get("modelCalls")
                if (
                    isinstance(calls, int)
                    and not isinstance(calls, bool)
                    and calls >= 0
                ):
                    underlying_calls += calls
        return response.text, {
            "released_runtime_adapter": True,
            "external_session_tree": True,
            "native_subagents_enabled": True,
            "underlying_model_calls_observed": underlying_calls > 0,
            "underlying_model_calls": underlying_calls,
            "full_tree_usage_verified": False,
            "usage_scope": (
                "completed main/subagent calls reported by Grok; compaction, side-model, "
                "and unfinished nested calls can be excluded"
            ),
            "usage_is_incomplete": result.get("usage_is_incomplete") is True,
            "cost_is_partial": result.get("cost_is_partial") is True,
            "num_main_agent_turns": result.get("num_turns"),
            "session_id_present": isinstance(result.get("sessionId"), str),
            "workspace": workspace if isinstance(workspace, dict) else None,
            "fidelity": "exact_public_protocol",
            "exactness_scope": "published_runtime_protocol_boundary",
        }
