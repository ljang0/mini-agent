from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness, require_int_at_least


class OpenAIHostedMultiAgentHarness(Harness):
    """Public Responses multi-agent boundary with optional developer tools."""

    name = "openai_hosted_multi_agent"

    def __init__(self, *, max_concurrent_subagents: int = 3) -> None:
        self.max_concurrent_subagents = require_int_at_least(
            max_concurrent_subagents, "max_concurrent_subagents"
        )

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/root",
                role="hosted_multi_agent",
                prompt=prompt,
                metadata={
                    "openai_multi_agent": True,
                    "max_concurrent_subagents": self.max_concurrent_subagents,
                },
            )
        )
        return response.text, {
            "hosted_orchestration": True,
            "server_scheduler_open_source": False,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "exact_scope": (
                "documented Responses request and developer-tool continuation; "
                "hosted scheduler remains closed"
            ),
            "fidelity": "exact_public_request_boundary",
        }
