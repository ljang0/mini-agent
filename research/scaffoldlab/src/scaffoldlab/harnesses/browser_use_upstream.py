from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class BrowserUseUpstreamHarness(Harness):
    """One pinned upstream Browser-Use Agent/Browser session.

    The backend owns Browser-Use's prompts, history, browser actions, model loop, and
    cleanup.  This is a source runtime adapter, not the local flat-parallel study.
    """

    name = "browser_use_upstream"

    def __init__(self, *, system: str = "") -> None:
        if not isinstance(system, str):
            raise ValueError("system must be a string")
        self.system = system

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/browser-use/root",
                role="browser_use_upstream_session",
                prompt=prompt,
                system=self.system,
                metadata={
                    "browser_use_upstream": True,
                    "external_agent_loop": True,
                    "task_id": task.task_id,
                    "task_tools": False,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        result = raw.get("result")
        result = dict(result) if isinstance(result, Mapping) else {}
        history = result.get("history")
        history = dict(history) if isinstance(history, Mapping) else {}
        raw_calls = raw.get("underlying_model_calls")
        calls_are_valid = (
            isinstance(raw_calls, int)
            and not isinstance(raw_calls, bool)
            and raw_calls >= 0
        )
        observed_flag = raw.get("underlying_model_calls_observed")
        calls_observed = (
            observed_flag if isinstance(observed_flag, bool) else calls_are_valid
        )
        return response.text, {
            "released_runtime_adapter": True,
            "external_agent_loop": True,
            "native_browser_use_agent": True,
            "native_browser_use_browser": True,
            "flat_parallel_reimplementation": False,
            "scaffoldlab_domain_tools_injected": False,
            "underlying_model_calls": raw_calls if calls_are_valid else 0,
            "underlying_model_calls_observed": calls_observed,
            "underlying_model_calls_are_lower_bound": True,
            "whole_session_usage_verified": False,
            "usage_scope": (
                "Browser-Use TokenCost entries whose model responses carried usage"
            ),
            "history_steps": history.get("steps"),
            "history_is_done": history.get("is_done"),
            "history_is_successful": history.get("is_successful"),
            "upstream_llm_class": result.get("llm_class"),
            "cost_tracking_enabled": result.get("cost_tracking_enabled") is True,
            "fidelity": "upstream_runtime_adapter",
            "exactness_scope": "private_revision_archive_with_recorded_runtime_identity",
            "caller_worktree_executed": False,
            "source_or_protocol_pin_verified": True,
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }
