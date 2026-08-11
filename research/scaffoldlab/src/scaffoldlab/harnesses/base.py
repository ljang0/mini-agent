from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

from ..environments.base import EnvironmentFactory
from ..runtime import ModelBackend, RunContext
from ..types import BudgetLimits, RunFailed, RunResult, Task


def require_int_at_least(value: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class Harness(ABC):
    name = "abstract"

    async def run(
        self,
        task: Task,
        backend: ModelBackend,
        limits: BudgetLimits | None = None,
        *,
        capture_content: bool = False,
        environment_factory: EnvironmentFactory | None = None,
    ) -> RunResult:
        context = RunContext(
            backend,
            limits or BudgetLimits(),
            capture_content=capture_content,
        )
        try:
            await context.trace.emit(
                "run_started",
                data={"task_id": task.task_id, "harness": self.name},
            )
            if environment_factory is not None:
                context.environment = await environment_factory.begin(task)
            answer, metadata = await asyncio.wait_for(
                self._execute(task, context),
                timeout=context.ledger.limits.wall_time_seconds,
            )
            await context.cancel_owned_tasks()
            environment_summary = await self._close_environment(context)
            if environment_summary is not None:
                metadata = {
                    **dict(metadata),
                    "environment": environment_summary,
                }
            await context.trace.emit(
                "run_completed",
                data={"task_id": task.task_id, "harness": self.name},
            )
        except asyncio.CancelledError:
            await context.cancel_owned_tasks()
            await self._close_environment(context, suppress_errors=True)
            await context.trace.emit(
                "run_cancelled",
                data={"task_id": task.task_id, "harness": self.name},
            )
            raise
        except Exception as exc:
            await context.cancel_owned_tasks()
            await self._close_environment(context, suppress_errors=True)
            await context.trace.emit(
                "run_failed",
                data={
                    "task_id": task.task_id,
                    "harness": self.name,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise RunFailed(
                str(exc),
                cause_type=type(exc).__name__,
                usage=context.ledger.usage,
                model_calls=context.ledger.calls,
                wall_time_seconds=context.trace.elapsed,
                backend_active_union_seconds=(
                    context.trace.backend_active_union_seconds
                ),
                trace=tuple(context.trace.events),
                tool_calls=context.ledger.tool_calls,
                tool_output_bytes=context.ledger.tool_output_bytes,
            ) from exc
        return RunResult(
            task_id=task.task_id,
            harness=self.name,
            answer=answer,
            usage=context.ledger.usage,
            model_calls=context.ledger.calls,
            wall_time_seconds=context.trace.elapsed,
            backend_active_union_seconds=context.trace.backend_active_union_seconds,
            trace=tuple(context.trace.events),
            metadata=metadata,
            tool_calls=context.ledger.tool_calls,
            tool_output_bytes=context.ledger.tool_output_bytes,
        )

    async def _close_environment(
        self, context: RunContext, *, suppress_errors: bool = False
    ) -> Mapping[str, Any] | None:
        environment = context.environment
        if environment is None:
            return None
        summary: Mapping[str, Any] = {}
        summary_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            summary = dict(await environment.summary())
        except BaseException as exc:
            summary_error = exc
        try:
            await environment.close()
        except BaseException as exc:
            close_error = exc
        finally:
            context.environment = None
        if summary_error is not None:
            await context.trace.emit(
                "environment_summary_failed",
                data={
                    "error": type(summary_error).__name__,
                    "message": str(summary_error),
                },
            )
        if close_error is not None:
            await context.trace.emit(
                "environment_close_failed",
                data={
                    "error": type(close_error).__name__,
                    "message": str(close_error),
                },
            )
        else:
            await context.trace.emit("environment_closed", data={"summary": summary})
        error = summary_error or close_error
        if error is not None and not suppress_errors:
            raise error
        return summary

    @abstractmethod
    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        raise NotImplementedError
