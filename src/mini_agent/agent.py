"""The complete mini-agent control loop.

Everything domain- or provider-specific is passed in. Keep this file boring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .environments.base import AgentEnvironment

from .models import Model
from .execution import RunContext
from .types import (
    AgentResult,
    BudgetExceeded,
    Message,
    ProtocolError,
    ToolDefinition,
    ToolExecution,
    ToolResult,
    _require_positive_int,
    _require_str,
)


class MiniAgent:
    """One agent: a model, an environment, and a linear conversation.

    The loop is deliberately small -- query the model, execute whatever tools
    it called, append the results, repeat -- because every benchmark, every
    sandbox, and every multi-agent topology in this project runs through this
    one implementation. Anything specific to a domain belongs in the
    environment, and anything specific to a topology belongs in the
    orchestrator.

    Budgets, tracing, and accounting live on ``context`` rather than here, so
    an agent that is one of ten shares the run's limits with the other nine.

    ``run`` starts a fresh conversation; ``resume`` continues an existing one,
    which is how an agent can report a result and stay available for further
    instructions. ``max_steps`` is a lifetime cap across both.
    """

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
        finish_on_answer: bool = True,
    ) -> None:
        _require_positive_int(max_steps, "max_steps")
        _require_str(system_prompt, "system_prompt", non_empty=False)
        _require_str(agent_id, "agent_id")
        _require_str(role, "role")
        if not callable(getattr(model, "query", None)):
            raise ValueError("model must expose an async query method")
        if not callable(getattr(environment, "tools", None)):
            raise ValueError("environment must expose tools")
        if context is not None and not isinstance(context, RunContext):
            raise ValueError("context must be RunContext or None")
        self.model = model
        self.environment = environment
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.context = context or RunContext()
        self.agent_id = agent_id
        self.role = role
        self.finish_on_answer = finish_on_answer
        self.messages: list[Message] = []
        self.steps_used = 0

    async def run(self, task: str) -> AgentResult:
        """Start a fresh conversation for this task."""

        _require_str(task, "task")
        self.messages = []
        self.steps_used = 0
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

        return await self._loop()

    async def resume(self, instruction: str) -> AgentResult:
        """Continue this agent's existing conversation with new input.

        An agent that reports a result and stays available is still the same
        agent: it keeps its history, and ``max_steps`` stays a lifetime cap
        rather than resetting on every instruction.
        """

        _require_str(instruction, "instruction")
        if not self.messages:
            raise ProtocolError("resume requires a started conversation")
        self.messages.append(Message(role="user", content=instruction))
        return await self._loop()

    async def _loop(self) -> AgentResult:
        tools = tuple(self.environment.tools())
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise ProtocolError("environment tools must be ToolDefinition values")
        if len({tool.name for tool in tools}) != len(tools):
            raise ProtocolError("environment tool names must be unique")
        if tools and not callable(getattr(self.environment, "execute", None)):
            raise ProtocolError("an environment with tools must expose execute")
        while self.steps_used < self.max_steps:
            self.steps_used += 1
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
                if finish is not None and self.finish_on_answer:
                    if not callable(finish):
                        raise ProtocolError("environment finish must be callable")
                    await finish()
                return AgentResult(
                    answer=response.text,
                    messages=tuple(self.messages),
                    steps=self.steps_used,
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
