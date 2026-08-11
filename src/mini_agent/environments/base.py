from __future__ import annotations

from typing import Protocol, Sequence

from scaffoldlab.environments.base import ToolExecution

from ..types import ToolCall, ToolDefinition


class Environment(Protocol):
    """The entire domain boundary used by :class:`MiniAgent`."""

    def tools(self) -> Sequence[ToolDefinition]: ...

    async def execute(self, action: ToolCall) -> ToolExecution: ...


class BaseEnvironment:
    async def initial_observation(self) -> ToolExecution | None:
        return None

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        return None


class EnvironmentAdapter(BaseEnvironment):
    """Expose a legacy Scaffold Lab environment through the minimal contract."""

    def __init__(self, environment: object, *, provider_family: str = "generic") -> None:
        self.environment = environment
        self.provider_family = provider_family

    def tools(self) -> Sequence[ToolDefinition]:
        return self.environment.tools(self.provider_family)  # type: ignore[attr-defined,no-any-return]

    async def execute(self, action: ToolCall) -> ToolExecution:
        return await self.environment.execute(action)  # type: ignore[attr-defined,no-any-return]

    async def close(self) -> None:
        await self.environment.close()  # type: ignore[attr-defined]
