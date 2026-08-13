"""Shared async lifecycle helpers.

These keep "finish the blocking call before cleanup" and "report a cleanup
failure without hiding the primary error" in one place for environments,
runtimes, and benchmark adapters.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


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
    "combine_lifecycle_errors",
    "complete_in_thread",
    "raise_lifecycle_errors",
]
