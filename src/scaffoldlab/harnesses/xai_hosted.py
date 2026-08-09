from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class XAIHostedMultiAgentHarness(Harness):
    """Exact public invocation surface for xAI's hosted multi-agent beta."""

    name = "xai_hosted_multi_agent"

    def __init__(self, agent_count: int = 4) -> None:
        if (
            not isinstance(agent_count, int)
            or isinstance(agent_count, bool)
            or agent_count not in {4, 16}
        ):
            raise ValueError("xAI hosted multi-agent supports agent_count 4 or 16")
        self.agent_count = agent_count
        # xAI documents low/medium as four agents and high/xhigh as sixteen.
        self.reasoning_effort = "low" if agent_count == 4 else "high"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/xai-hosted/leader",
                role="xai_hosted_multi_agent",
                prompt=prompt,
                metadata={
                    "xai_multi_agent": True,
                    "reasoning_effort": self.reasoning_effort,
                    "documented_agent_count": self.agent_count,
                },
            )
        )
        return response.text, {
            "hosted_runtime": True,
            "public_invocation_exact": True,
            "server_scheduler_open": False,
            "documented_agent_count": self.agent_count,
            "reasoning_effort": self.reasoning_effort,
            "leader_output_only": True,
            "built_in_tools_enabled": False,
            "intermediate_subagent_state_observed": False,
            "plaintext_subagent_state_available": False,
            "encrypted_continuation_implemented": False,
            "hosted_multi_agent_beta": True,
            "fidelity": "exact_public_request_boundary",
        }
