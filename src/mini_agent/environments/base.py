from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Protocol, Sequence

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


async def complete_in_thread(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Let an uncancellable blocking operation finish before its owner is cleaned."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


def combine_lifecycle_errors(
    first: BaseException | None, second: BaseException
) -> BaseException:
    if first is None:
        return second
    return RuntimeError(
        f"{type(first).__name__}: {first}; {type(second).__name__}: {second}"
    )


def raise_lifecycle_errors(
    label: str,
    operation_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> None:
    """Retain the primary error while making a cleanup error observable.

    A cleanup-only failure is re-raised as-is; benchmark runners that need a
    labeled cleanup error use
    :func:`mini_agent.benchmarks.base.raise_after_cleanup` instead.
    """

    if operation_error is not None:
        if cleanup_error is not None:
            if isinstance(operation_error, asyncio.CancelledError):
                raise operation_error from cleanup_error
            raise RuntimeError(
                f"{label} failed ({type(operation_error).__name__}: "
                f"{operation_error}); cleanup also failed "
                f"({type(cleanup_error).__name__}: {cleanup_error})"
            ) from operation_error
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error


__all__ = [
    "AgentEnvironment",
    "BaseEnvironment",
    "Environment",
    "ToolExecution",
    "combine_lifecycle_errors",
    "complete_in_thread",
    "raise_lifecycle_errors",
]
