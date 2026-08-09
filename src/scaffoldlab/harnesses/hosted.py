from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness, require_int_at_least


class OpenAIHostedMultiAgentHarness(Harness):
    """Public Responses multi-agent boundary with optional developer tools."""

    name = "openai_hosted_multi_agent"

    def __init__(
        self,
        *,
        max_concurrent_subagents: int = 3,
        hosted_tools: Sequence[str] = (),
    ) -> None:
        self.max_concurrent_subagents = require_int_at_least(
            max_concurrent_subagents, "max_concurrent_subagents"
        )
        if isinstance(hosted_tools, (str, bytes)) or not isinstance(
            hosted_tools, Sequence
        ):
            raise ValueError("hosted_tools must be a sequence of tool type strings")
        normalized_tools: list[str] = []
        for tool_type in hosted_tools:
            if not isinstance(tool_type, str) or tool_type != "web_search":
                raise ValueError(
                    "OpenAI hosted multi-agent hosted_tools only supports 'web_search'"
                )
            if tool_type in normalized_tools:
                raise ValueError(f"duplicate hosted tool {tool_type!r}")
            normalized_tools.append(tool_type)
        self.hosted_tools = tuple(normalized_tools)

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
                    "openai_hosted_tools": list(self.hosted_tools),
                },
            )
        )
        return response.text, {
            "hosted_orchestration": True,
            "server_scheduler_open_source": False,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "hosted_tools": list(self.hosted_tools),
            "built_in_tools_enabled": bool(self.hosted_tools),
            "exact_scope": (
                "documented Responses request and developer-tool continuation; "
                "hosted scheduler remains closed"
            ),
            "fidelity": "exact_public_request_boundary",
        }
