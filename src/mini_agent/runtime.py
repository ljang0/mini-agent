from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Optional

from scaffoldlab.environments.base import ToolExecution
from scaffoldlab.runtime import BudgetLedger, TraceRecorder

from .types import (
    BudgetLimits,
    Message,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)


class RunContext:
    """Shared accounting and trace state for one or many minimal agents."""

    def __init__(
        self,
        limits: Optional[BudgetLimits] = None,
        *,
        ledger: Optional[BudgetLedger] = None,
        trace: Optional[TraceRecorder] = None,
        capture_content: bool = False,
    ) -> None:
        if limits is None and ledger is None:
            limits = BudgetLimits()
        self.ledger = ledger or BudgetLedger(limits or BudgetLimits())
        self.trace = trace or TraceRecorder()
        self.capture_content = capture_content

    async def query(
        self,
        model: Any,
        messages: list[Message],
        tools: tuple[ToolDefinition, ...],
        *,
        agent_id: str,
        role: str,
    ) -> ModelResponse:
        data: dict[str, Any] = {
            "message_count": len(messages),
            "tool_count": len(tools),
            "history_sha256": hashlib.sha256(
                json.dumps(
                    [asdict(message) for message in messages],
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
        }
        if self.capture_content:
            data["messages"] = [asdict(message) for message in messages]
            data["tools"] = [asdict(tool) for tool in tools]
        await self.trace.emit("model_call_queued", agent_id=agent_id, role=role, data=data)
        started: Optional[float] = None
        finished: Optional[float] = None
        response: Optional[ModelResponse] = None
        try:
            async with self.ledger.semaphore:
                await self.ledger.reserve_call()
                await self.trace.emit("model_call_started", agent_id=agent_id, role=role)
                timeout = max(
                    0.001,
                    self.ledger.limits.wall_time_seconds - self.trace.elapsed,
                )
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        model.query(tuple(messages), tools), timeout=timeout
                    )
                finally:
                    finished = time.perf_counter()
            if not isinstance(response, ModelResponse):
                raise ProtocolError("model.query must return ModelResponse")
            await self.ledger.record(response.usage)
        except asyncio.CancelledError:
            if started is not None:
                await self.ledger.mark_incomplete()
            await self.trace.emit("model_call_cancelled", agent_id=agent_id, role=role)
            raise
        except Exception as exc:
            reported_usage = getattr(exc, "usage", None)
            if isinstance(reported_usage, Usage):
                await self.ledger.record(reported_usage)
            elif started is not None and response is None:
                await self.ledger.mark_incomplete()
            await self.trace.emit(
                "model_call_failed",
                agent_id=agent_id,
                role=role,
                data={"error": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            if started is not None and finished is not None:
                await self.trace.record_interval(started, finished, agent_id)
        assert response is not None
        completed: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": response.usage.cost_usd,
            "tool_calls": len(response.tool_calls),
            "response_chars": len(response.text),
        }
        if self.capture_content:
            completed["response"] = response.text
        await self.trace.emit(
            "model_call_completed", agent_id=agent_id, role=role, data=completed
        )
        return response

    async def execute(
        self,
        environment: Any,
        action: ToolCall,
        tools: tuple[ToolDefinition, ...],
        *,
        agent_id: str,
        role: str,
    ) -> ToolResult:
        await self.ledger.reserve_tool_call()
        arguments = json.dumps(dict(action.arguments), sort_keys=True, default=str)
        await self.trace.emit(
            "tool_call_started",
            agent_id=agent_id,
            role=role,
            data={
                "tool": action.name,
                "arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                **({"arguments": dict(action.arguments)} if self.capture_content else {}),
            },
        )
        allowed = {tool.name for tool in tools}
        if action.name not in allowed:
            return await self._invalid_action(
                action, f"unknown tool {action.name!r}", agent_id=agent_id, role=role
            )
        try:
            timeout = max(
                0.001,
                self.ledger.limits.wall_time_seconds - self.trace.elapsed,
            )
            execution: ToolExecution = await asyncio.wait_for(
                environment.execute(action), timeout=timeout
            )
        except ProtocolError as exc:
            return await self._invalid_action(
                action, str(exc), agent_id=agent_id, role=role
            )
        except asyncio.CancelledError:
            await self.trace.emit(
                "tool_call_cancelled",
                agent_id=agent_id,
                role=role,
                data={"tool": action.name},
            )
            raise
        except Exception as exc:
            await self.trace.emit(
                "tool_call_failed",
                agent_id=agent_id,
                role=role,
                data={
                    "tool": action.name,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        output_bytes = len(execution.output.encode("utf-8"))
        if execution.image_data_url:
            output_bytes += len(execution.image_data_url.encode("ascii"))
        await self.ledger.record_tool_output(output_bytes)
        await self.trace.emit(
            "tool_call_completed",
            agent_id=agent_id,
            role=role,
            data={
                "tool": action.name,
                "is_error": execution.is_error,
                "output_bytes": output_bytes,
                **dict(execution.metadata),
                **({"output": execution.output} if self.capture_content else {}),
            },
        )
        return ToolResult(
            call_id=action.call_id,
            name=action.name,
            output=execution.output,
            kind=action.kind,
            is_error=execution.is_error,
            image_data_url=execution.image_data_url,
            native_output=execution.native_output,
        )

    async def _invalid_action(
        self,
        action: ToolCall,
        message: str,
        *,
        agent_id: str,
        role: str,
    ) -> ToolResult:
        output = f"Invalid action: {message}"
        await self.ledger.record_tool_output(len(output.encode("utf-8")))
        await self.trace.emit(
            "tool_call_completed",
            agent_id=agent_id,
            role=role,
            data={"tool": action.name, "is_error": True, "invalid_action": True},
        )
        return ToolResult(
            call_id=action.call_id,
            name=action.name,
            output=output,
            kind=action.kind,
            is_error=True,
        )
