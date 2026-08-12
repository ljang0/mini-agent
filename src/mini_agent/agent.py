"""The complete mini-agent control loop.

Everything domain- or provider-specific is passed in. Keep this file boring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .environments.base import AgentEnvironment

from .models import Model
from .runtime import RunContext
from .types import (
    AgentResult,
    BudgetExceeded,
    Message,
    ProtocolError,
    ToolDefinition,
    ToolExecution,
    ToolResult,
)


class MiniAgent:
    def __init__(
        self,
        *,
        model: Model,
        environment: AgentEnvironment,
        system_prompt: str = "",
        max_steps: int = 64,
        context: Optional[RunContext] = None,
        agent_id: str = "/root",
        role: str = "solver",
    ) -> None:
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if not callable(getattr(model, "query", None)):
            raise ValueError("model must expose an async query method")
        if not callable(getattr(environment, "tools", None)):
            raise ValueError("environment must expose tools")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if context is not None and not isinstance(context, RunContext):
            raise ValueError("context must be RunContext or None")
        self.model = model
        self.environment = environment
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.context = context or RunContext()
        self.agent_id = agent_id
        self.role = role
        self.messages: list[Message] = []

    async def run(self, task: str) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        self.messages = []
        if self.system_prompt:
            self.messages.append(Message(role="system", content=self.system_prompt))
        self.messages.append(Message(role="user", content=task))

        initial = getattr(self.environment, "initial_observation", None)
        if initial is not None:
            if not callable(initial):
                raise ProtocolError("environment initial_observation must be callable")
            observation = await initial()
            if observation is not None:
                if not isinstance(observation, ToolExecution):
                    raise ProtocolError(
                        "environment initial_observation must return "
                        "ToolExecution or None"
                    )
                await self.context.record_initial_observation(
                    observation,
                    agent_id=self.agent_id,
                    role=self.role,
                )
                self.messages.append(
                    Message(
                        role="user",
                        content=observation.output,
                        image_data_url=observation.image_data_url,
                        metadata=dict(observation.metadata),
                    )
                )

        tools = tuple(self.environment.tools())
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise ProtocolError("environment tools must be ToolDefinition values")
        if len({tool.name for tool in tools}) != len(tools):
            raise ProtocolError("environment tool names must be unique")
        if tools and not callable(getattr(self.environment, "execute", None)):
            raise ProtocolError("an environment with tools must expose execute")
        for step in range(1, self.max_steps + 1):
            response = await self.context.query(
                self.model,
                self.messages,
                tools,
                agent_id=self.agent_id,
                role=self.role,
            )
            if len({call.call_id for call in response.tool_calls}) != len(
                response.tool_calls
            ):
                raise ProtocolError("model tool call ids must be unique per response")
            self.messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                if not response.text:
                    raise ProtocolError(
                        "model returned neither final text nor tool calls"
                    )
                finish = getattr(self.environment, "finish", None)
                if finish is not None:
                    if not callable(finish):
                        raise ProtocolError("environment finish must be callable")
                    await finish()
                return AgentResult(
                    answer=response.text,
                    messages=tuple(self.messages),
                    steps=step,
                    metadata={"agent_id": self.agent_id},
                )

            results: list[ToolResult] = []
            for action in response.tool_calls:
                results.append(
                    await self.context.execute(
                        self.environment,
                        action,
                        tools,
                        agent_id=self.agent_id,
                        role=self.role,
                    )
                )
            self.messages.append(
                Message(
                    role="tool",
                    content="\n".join(result.output for result in results),
                    tool_results=tuple(results),
                )
            )

        raise BudgetExceeded(f"agent exceeded max_steps ({self.max_steps})")
