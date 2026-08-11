from __future__ import annotations

import uuid
from typing import Protocol, Sequence

from ..types import ToolCall, ToolDefinition, ToolExecution


class Environment(Protocol):
    """The complete domain boundary used by :class:`MiniAgent`."""

    def tools(self) -> Sequence[ToolDefinition]: ...

    async def execute(self, action: ToolCall) -> ToolExecution: ...

    def resource_identity(self) -> str: ...


class BaseEnvironment:
    """Optional lifecycle hooks shared by the small built-in environments."""

    async def initial_observation(self) -> ToolExecution | None:
        return None

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def resource_identity(self) -> str:
        """Return this wrapper's explicit resource identity for isolation checks."""

        identity = getattr(self, "_mini_agent_resource_identity", None)
        if identity is None:
            identity = f"{type(self).__module__}.{type(self).__qualname__}:{uuid.uuid4()}"
            setattr(self, "_mini_agent_resource_identity", identity)
        return identity


__all__ = ["BaseEnvironment", "Environment", "ToolExecution"]
