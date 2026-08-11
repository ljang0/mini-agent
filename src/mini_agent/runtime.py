"""Shared budgets and traces for one or many :class:`MiniAgent` instances."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Mapping, Optional

from .types import (
    BudgetExceeded,
    BudgetLimits,
    Message,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolResult,
    TraceEvent,
    Usage,
)


class BudgetLedger:
    """Concurrency-safe global accounting with optional per-agent limits."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.calls = 0
        self.tool_calls = 0
        self.tool_output_bytes = 0
        self.usage = Usage()
        self._exhausted_reason: str | None = None
        self._agent_limits: dict[str, BudgetLimits] = {}
        self._agent_calls: dict[str, int] = {}
        self._agent_tool_calls: dict[str, int] = {}
        self._agent_tool_output_bytes: dict[str, int] = {}
        self._agent_usage: dict[str, Usage] = {}
        self._agent_exhausted: dict[str, str] = {}
        self._agent_semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)

    def configure_agent(self, agent_id: str, limits: BudgetLimits) -> None:
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        existing = self._agent_limits.get(agent_id)
        if existing is not None and existing != limits:
            raise ValueError(f"agent budget already configured for {agent_id!r}")
        self._agent_limits[agent_id] = limits
        self._agent_semaphores.setdefault(
            agent_id, asyncio.Semaphore(limits.max_concurrency)
        )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    def agent_semaphore(self, agent_id: str) -> Any:
        limits = self._agent_limits.get(agent_id)
        if limits is None:
            return _UnlimitedSemaphore.instance()
        return self._agent_semaphores.setdefault(
            agent_id, asyncio.Semaphore(limits.max_concurrency)
        )

    def agent_snapshot(self, agent_id: str) -> Mapping[str, Any]:
        return {
            "model_calls": self._agent_calls.get(agent_id, 0),
            "tool_calls": self._agent_tool_calls.get(agent_id, 0),
            "tool_output_bytes": self._agent_tool_output_bytes.get(agent_id, 0),
            "usage": asdict(self._agent_usage.get(agent_id, Usage())),
            "exhausted_reason": self._agent_exhausted.get(agent_id),
        }

    @staticmethod
    def _resource_reason(limits: BudgetLimits, usage: Usage, *, crossed: bool) -> str | None:
        suffix = "exceeded" if crossed else "exhausted"
        comparisons = (
            (limits.max_input_tokens, usage.input_tokens, "input-token"),
            (limits.max_output_tokens, usage.output_tokens, "output-token"),
            (limits.max_cost_usd, usage.cost_usd, "cost"),
        )
        for limit, value, label in comparisons:
            if limit is not None and (value > limit if crossed else value >= limit):
                return f"{label} budget {suffix}"
        return None

    async def reserve_call(self, agent_id: str = "") -> None:
        async with self._lock:
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            reason = self._resource_reason(self.limits, self.usage, crossed=False)
            if reason is None and self.calls >= self.limits.max_model_calls:
                reason = f"model-call budget exhausted ({self.limits.max_model_calls})"
            if reason is not None:
                self._exhausted_reason = reason
                raise BudgetExceeded(reason)
            agent_limits = self._agent_limits.get(agent_id)
            if agent_limits is not None:
                agent_reason = self._agent_exhausted.get(agent_id)
                agent_usage = self._agent_usage.get(agent_id, Usage())
                agent_reason = agent_reason or self._resource_reason(
                    agent_limits, agent_usage, crossed=False
                )
                calls = self._agent_calls.get(agent_id, 0)
                if agent_reason is None and calls >= agent_limits.max_model_calls:
                    agent_reason = (
                        f"agent model-call budget exhausted "
                        f"({agent_limits.max_model_calls})"
                    )
                if agent_reason is not None:
                    self._agent_exhausted[agent_id] = agent_reason
                    raise BudgetExceeded(agent_reason)
            self.calls += 1
            if agent_limits is not None:
                self._agent_calls[agent_id] = self._agent_calls.get(agent_id, 0) + 1

    async def record(self, usage: Usage, agent_id: str = "") -> None:
        async with self._lock:
            self.usage = self.usage + usage
            agent_limits = self._agent_limits.get(agent_id)
            if agent_limits is not None:
                self._agent_usage[agent_id] = self._agent_usage.get(agent_id, Usage()) + usage
            reason: str | None = None
            if not usage.complete and any(
                value is not None
                for value in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            ):
                reason = "resource budget cannot be verified because usage is incomplete"
            elif self.limits.max_cost_usd is not None and not self.usage.cost_known:
                reason = "cost budget cannot be verified because usage cost is unknown"
            else:
                reason = self._resource_reason(self.limits, self.usage, crossed=True)
            if reason is not None:
                self._exhausted_reason = reason
            agent_reason: str | None = None
            if agent_limits is not None:
                agent_usage = self._agent_usage[agent_id]
                if not usage.complete and any(
                    value is not None
                    for value in (
                        agent_limits.max_input_tokens,
                        agent_limits.max_output_tokens,
                        agent_limits.max_cost_usd,
                    )
                ):
                    agent_reason = "agent resource usage is incomplete"
                elif agent_limits.max_cost_usd is not None and not agent_usage.cost_known:
                    agent_reason = "agent cost is unknown"
                else:
                    agent_reason = self._resource_reason(
                        agent_limits, agent_usage, crossed=True
                    )
                if agent_reason is not None:
                    self._agent_exhausted[agent_id] = agent_reason
            if reason is not None:
                raise BudgetExceeded(reason)
            if agent_reason is not None:
                raise BudgetExceeded(agent_reason)

    async def mark_incomplete(self, agent_id: str = "") -> None:
        async with self._lock:
            self.usage = _incomplete(self.usage)
            if any(
                value is not None
                for value in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            ):
                self._exhausted_reason = (
                    "resource budget cannot be verified after an incomplete call"
                )
            if agent_id in self._agent_limits:
                self._agent_usage[agent_id] = _incomplete(
                    self._agent_usage.get(agent_id, Usage())
                )
                self._agent_exhausted[agent_id] = "agent call usage is incomplete"

    async def reserve_tool_call(self, agent_id: str = "") -> None:
        async with self._lock:
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            if self.tool_calls >= self.limits.max_tool_calls:
                self._exhausted_reason = (
                    f"tool-call budget exhausted ({self.limits.max_tool_calls})"
                )
                raise BudgetExceeded(self._exhausted_reason)
            agent_limits = self._agent_limits.get(agent_id)
            if agent_limits is not None:
                reason = self._agent_exhausted.get(agent_id)
                calls = self._agent_tool_calls.get(agent_id, 0)
                if reason is None and calls >= agent_limits.max_tool_calls:
                    reason = (
                        f"agent tool-call budget exhausted "
                        f"({agent_limits.max_tool_calls})"
                    )
                if reason is not None:
                    self._agent_exhausted[agent_id] = reason
                    raise BudgetExceeded(reason)
            self.tool_calls += 1
            if agent_limits is not None:
                self._agent_tool_calls[agent_id] = (
                    self._agent_tool_calls.get(agent_id, 0) + 1
                )

    async def record_tool_output(self, byte_count: int, agent_id: str = "") -> None:
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ValueError("tool output byte count must be a non-negative integer")
        async with self._lock:
            self.tool_output_bytes += byte_count
            if self.tool_output_bytes > self.limits.max_tool_output_bytes:
                self._exhausted_reason = "tool-output byte budget exceeded"
            agent_reason: str | None = None
            agent_limits = self._agent_limits.get(agent_id)
            if agent_limits is not None:
                total = self._agent_tool_output_bytes.get(agent_id, 0) + byte_count
                self._agent_tool_output_bytes[agent_id] = total
                if total > agent_limits.max_tool_output_bytes:
                    agent_reason = "agent tool-output byte budget exceeded"
                    self._agent_exhausted[agent_id] = agent_reason
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            if agent_reason is not None:
                raise BudgetExceeded(agent_reason)


class _UnlimitedSemaphore:
    _instance: "_UnlimitedSemaphore | None" = None

    @classmethod
    def instance(cls) -> "_UnlimitedSemaphore":
        cls._instance = cls._instance or cls()
        return cls._instance

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: Any) -> None:
        return None


def _incomplete(usage: Usage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        cost_usd=usage.cost_usd,
        cost_known=False,
        complete=False,
    )


class TraceRecorder:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.events: list[TraceEvent] = []
        self._lock = asyncio.Lock()
        self._call_intervals: list[tuple[float, float, str]] = []

    async def emit(
        self,
        event: str,
        *,
        agent_id: str = "",
        role: str = "",
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        async with self._lock:
            self.events.append(
                TraceEvent(
                    event=event,
                    elapsed_seconds=time.perf_counter() - self.started,
                    agent_id=agent_id,
                    role=role,
                    data=dict(data or {}),
                )
            )

    async def record_interval(self, start: float, finish: float, agent_id: str) -> None:
        async with self._lock:
            self._call_intervals.append((start, finish, agent_id))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def backend_active_union_seconds(self) -> float:
        intervals = sorted((start, finish) for start, finish, _ in self._call_intervals)
        merged: list[tuple[float, float]] = []
        for start, finish in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, finish))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
        return sum(finish - start for start, finish in merged)


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

    def configure_agent(self, agent_id: str, limits: BudgetLimits) -> None:
        self.ledger.configure_agent(agent_id, limits)

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
        started: float | None = None
        finished: float | None = None
        response: ModelResponse | None = None
        try:
            async with self.ledger.semaphore:
                async with self.ledger.agent_semaphore(agent_id):
                    await self.ledger.reserve_call(agent_id)
                    await self.trace.emit(
                        "model_call_started", agent_id=agent_id, role=role
                    )
                    timeout = max(
                        0.001, self.ledger.limits.wall_time_seconds - self.trace.elapsed
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
            await self.ledger.record(response.usage, agent_id)
        except asyncio.CancelledError:
            if started is not None:
                await self.ledger.mark_incomplete(agent_id)
            await self.trace.emit("model_call_cancelled", agent_id=agent_id, role=role)
            raise
        except Exception as exc:
            reported_usage = getattr(exc, "usage", None)
            if isinstance(reported_usage, Usage):
                await self.ledger.record(reported_usage, agent_id)
            elif started is not None and response is None:
                await self.ledger.mark_incomplete(agent_id)
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
        await self.ledger.reserve_tool_call(agent_id)
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
        definitions = {tool.name: tool for tool in tools}
        definition = definitions.get(action.name)
        if definition is None:
            return await self._invalid_action(
                action, f"unknown tool {action.name!r}", agent_id=agent_id, role=role
            )
        schema_error = _validate_arguments(action.arguments, definition.input_schema)
        if schema_error is not None:
            return await self._invalid_action(
                action, schema_error, agent_id=agent_id, role=role
            )
        try:
            timeout = max(
                0.001, self.ledger.limits.wall_time_seconds - self.trace.elapsed
            )
            execution = await asyncio.wait_for(environment.execute(action), timeout=timeout)
            if not isinstance(execution, ToolExecution):
                try:
                    execution = ToolExecution(
                        output=execution.output,
                        is_error=execution.is_error,
                        image_data_url=execution.image_data_url,
                        metadata=getattr(execution, "metadata", {}),
                        native_output=getattr(execution, "native_output", None),
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ProtocolError(
                        "environment.execute must return a compatible ToolExecution"
                    ) from exc
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
        await self.ledger.record_tool_output(output_bytes, agent_id)
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
        await self.ledger.record_tool_output(len(output.encode("utf-8")), agent_id)
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


def _validate_arguments(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
    if schema.get("type", "object") != "object":
        return "tool schema root must be an object"
    required = schema.get("required", [])
    if not isinstance(required, list):
        return "tool schema required must be a list"
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"missing required arguments {missing}"
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return "tool schema properties must be an object"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(arguments).difference(properties))
        if extra:
            return f"unexpected arguments {extra}"
    expected_types: Mapping[str, tuple[type, ...]] = {
        "string": (str,),
        "array": (list,),
        "object": (dict,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
    }
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, Mapping) or "type" not in spec:
            continue
        expected = expected_types.get(str(spec["type"]))
        if expected is not None and (
            not isinstance(value, expected)
            or isinstance(value, bool) and spec["type"] in {"integer", "number"}
        ):
            return f"argument {name!r} must be {spec['type']}"
        if spec.get("type") == "array" and isinstance(value, list):
            minimum = spec.get("minItems")
            if isinstance(minimum, int) and len(value) < minimum:
                return f"argument {name!r} requires at least {minimum} items"
    return None


__all__ = ["BudgetLedger", "RunContext", "TraceRecorder"]
