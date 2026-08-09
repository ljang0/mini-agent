from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class MACUUpstreamHarness(Harness):
    """Adapter target for the pinned released MACU runtime.

    One shared-ledger call represents the complete upstream manager/worker tree. The
    backend owns graph generation, scheduling, replanning, VM cloning, CUA processes,
    result persistence, and any upstream evaluator invocation.
    """

    name = "macu_upstream"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/macu/root",
                role="macu_upstream_session",
                prompt=prompt,
                metadata={
                    "external_session_tree": True,
                    "task_id": task.task_id,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, dict) else {}
        summary = raw.get("summary")
        final_results = raw.get("final_results")
        return response.text, {
            "released_runtime_adapter": True,
            "external_session_tree": True,
            "shared_budget_ledger": True,
            "full_tree_usage_verified": response.usage.complete,
            "upstream_task_id": raw.get("task_id"),
            "upstream_summary_present": isinstance(summary, dict),
            "upstream_final_results_present": isinstance(final_results, dict),
            "upstream_status": (
                final_results.get("status") if isinstance(final_results, dict) else None
            ),
            "upstream_result_directory": raw.get("result_directory"),
            "fidelity": "upstream_runtime_adapter",
        }
