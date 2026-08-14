"""What an environment must expose, and the optional lifecycle hooks.

Two protocols and one mixin: tools plus execute is the whole required
surface, and everything else -- an initial observation, exported state a
sibling can adopt, cleanup -- is optional and defaulted here.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, Sequence

from .._lifecycle import (
    combine_lifecycle_errors,
    complete_in_thread,
    raise_lifecycle_errors,
)
from ..types import ToolCall, ToolDefinition, ToolExecution


class AgentEnvironment(Protocol):
    """The two required methods used by the minimal agent loop."""

    def tools(self) -> Sequence[ToolDefinition]: ...

    async def execute(self, action: ToolCall) -> ToolExecution: ...


class Environment(AgentEnvironment, Protocol):
    """The agent boundary plus lifecycle and identity required by orchestration."""

    def resource_identity(self) -> str: ...

    async def close(self) -> None: ...


class BaseEnvironment:
    """Optional lifecycle hooks shared by the small built-in environments."""

    async def initial_observation(self) -> ToolExecution | None:
        return None

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def export_state(self) -> Any:
        """Return an opaque, durable state reference for optional adoption."""

        return None

    async def adopt_state(self, state: Any) -> None:
        """Adopt a descendant state when the domain supports it."""

        del state
        raise NotImplementedError("environment state adoption is unsupported")

    def resource_identity(self) -> str:
        """Return this wrapper's explicit resource identity for isolation checks."""

        identity = getattr(self, "_mini_agent_resource_identity", None)
        if identity is None:
            identity = (
                f"{type(self).__module__}.{type(self).__qualname__}:{uuid.uuid4()}"
            )
            setattr(self, "_mini_agent_resource_identity", identity)
        return identity


__all__ = [
    "AgentEnvironment",
    "BaseEnvironment",
    "Environment",
    "ToolExecution",
    "combine_lifecycle_errors",
    "complete_in_thread",
    "raise_lifecycle_errors",
]
