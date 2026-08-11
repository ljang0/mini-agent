from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
)

from .environments.base import EnvironmentScope, ToolExecution
from .types import (
    BudgetExceeded,
    BudgetLimits,
    ModelRequest,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolResult,
    TraceEvent,
    Usage,
)


class ModelBackend(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one model completion for a harness-controlled request."""


class BudgetLedger:
    """Concurrency-safe accounting shared by every agent in one run."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.calls = 0
        self.tool_calls = 0
        self.tool_output_bytes = 0
        self.usage = Usage()
        self._exhausted_reason: Optional[str] = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)

    async def reserve_call(self) -> None:
        async with self._lock:
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            exhausted_resource: Optional[str] = None
            if (
                self.limits.max_input_tokens is not None
                and self.usage.input_tokens >= self.limits.max_input_tokens
            ):
                exhausted_resource = "input-token budget exhausted"
            elif (
                self.limits.max_output_tokens is not None
                and self.usage.output_tokens >= self.limits.max_output_tokens
            ):
                exhausted_resource = "output-token budget exhausted"
            elif (
                self.limits.max_cost_usd is not None
                and self.usage.cost_usd >= self.limits.max_cost_usd
            ):
                exhausted_resource = "cost budget exhausted"
            if exhausted_resource is not None:
                self._exhausted_reason = exhausted_resource
                raise BudgetExceeded(exhausted_resource)
            if self.calls >= self.limits.max_model_calls:
                self._exhausted_reason = (
                    f"model-call budget exhausted ({self.limits.max_model_calls})"
                )
                raise BudgetExceeded(self._exhausted_reason)
            self.calls += 1

    async def record(self, usage: Usage) -> None:
        async with self._lock:
            next_usage = self.usage + usage
            # Record billed/consumed usage even when this call crosses a hard limit.
            # The run stops, but its result must not make the over-budget call free.
            self.usage = next_usage
            if not usage.complete and any(
                limit is not None
                for limit in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            ):
                self._exhausted_reason = (
                    "resource budget cannot be verified because usage is incomplete"
                )
                raise BudgetExceeded(self._exhausted_reason)
            if (
                self.limits.max_input_tokens is not None
                and next_usage.input_tokens > self.limits.max_input_tokens
            ):
                self._exhausted_reason = "input-token budget exceeded"
                raise BudgetExceeded(self._exhausted_reason)
            if (
                self.limits.max_output_tokens is not None
                and next_usage.output_tokens > self.limits.max_output_tokens
            ):
                self._exhausted_reason = "output-token budget exceeded"
                raise BudgetExceeded(self._exhausted_reason)
            if self.limits.max_cost_usd is not None and not next_usage.cost_known:
                self._exhausted_reason = (
                    "cost budget cannot be verified because usage cost is unknown"
                )
                raise BudgetExceeded(self._exhausted_reason)
            if (
                self.limits.max_cost_usd is not None
                and next_usage.cost_usd > self.limits.max_cost_usd
            ):
                self._exhausted_reason = "cost budget exceeded"
                raise BudgetExceeded(self._exhausted_reason)

    async def mark_incomplete(self) -> None:
        async with self._lock:
            self.usage = Usage(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cache_read_input_tokens=self.usage.cache_read_input_tokens,
                cache_write_input_tokens=self.usage.cache_write_input_tokens,
                cost_usd=self.usage.cost_usd,
                cost_known=False,
                complete=False,
            )
            if any(
                limit is not None
                for limit in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            ):
                self._exhausted_reason = (
                    "resource budget cannot be verified after an incomplete call"
                )

    async def reserve_tool_call(self) -> None:
        async with self._lock:
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            if self.tool_calls >= self.limits.max_tool_calls:
                self._exhausted_reason = (
                    f"tool-call budget exhausted ({self.limits.max_tool_calls})"
                )
                raise BudgetExceeded(self._exhausted_reason)
            self.tool_calls += 1

    async def record_tool_output(self, byte_count: int) -> None:
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise ValueError("tool output byte count must be a non-negative integer")
        async with self._lock:
            self.tool_output_bytes += byte_count
            if self.tool_output_bytes > self.limits.max_tool_output_bytes:
                self._exhausted_reason = "tool-output byte budget exceeded"
                raise BudgetExceeded(self._exhausted_reason)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore


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
        """Union of overlapping backend-call intervals; not a causal critical path."""
        if not self._call_intervals:
            return 0.0
        intervals = sorted((start, finish) for start, finish, _ in self._call_intervals)
        merged: list[tuple[float, float]] = []
        for start, finish in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, finish))
            else:
                previous_start, previous_finish = merged[-1]
                merged[-1] = (previous_start, max(previous_finish, finish))
        return sum(finish - start for start, finish in merged)


class RunContext:
    def __init__(
        self,
        backend: ModelBackend,
        limits: BudgetLimits,
        *,
        trace: Optional[TraceRecorder] = None,
        capture_content: bool = False,
        environment: Optional[EnvironmentScope] = None,
    ) -> None:
        self.backend = backend
        self.ledger = BudgetLedger(limits)
        self.trace = trace or TraceRecorder()
        self.capture_content = capture_content
        self.environment = environment
        self.provider_family = str(getattr(backend, "tool_family", "generic"))
        self._owned_tasks: set[asyncio.Task[Any]] = set()

    def create_task(self, awaitable: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable)
        self._owned_tasks.add(task)
        return task

    async def gather(self, *awaitables: Any) -> list[Any]:
        """Gather owned branches and cancel every sibling if one branch fails."""
        tasks = [self.create_task(awaitable) for awaitable in awaitables]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def cancel_owned_tasks(self) -> None:
        owned = list(self._owned_tasks)
        for task in owned:
            if not task.done():
                task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)
        self._owned_tasks.clear()

    async def call(self, request: ModelRequest) -> ModelResponse:
        """Run one logical agent turn, including any client-tool round trips."""

        tools_disabled = (
            request.metadata.get("domain_tools") is False
            or request.metadata.get("tool_policy") == "none"
            or request.metadata.get("task_tools") is False
        )
        if self.environment is None or tools_disabled or request.tools:
            return await self._call_model(request)
        return await self._run_agent_tool_loop(request)

    async def _run_agent_tool_loop(self, request: ModelRequest) -> ModelResponse:
        assert self.environment is not None
        environment = await self.environment.get(request.agent_id)
        tools = tuple(environment.tools(self.provider_family))
        if not tools:
            return await self._call_model(request)
        continuation: Any = None
        pending_results: tuple[ToolResult, ...] = ()
        total_usage = Usage()
        for turn in range(self.ledger.limits.max_agent_turns):
            turn_metadata = {
                **dict(request.metadata),
                "agent_tool_turn": turn,
                "domain_tool_names": [tool.name for tool in tools],
            }
            response = await self._call_model(
                replace(
                    request,
                    prompt=request.prompt if turn == 0 else "",
                    tools=tools,
                    tool_results=pending_results,
                    continuation=continuation,
                    metadata=turn_metadata,
                )
            )
            total_usage = total_usage + response.usage
            if not response.tool_calls:
                if not response.text:
                    raise ProtocolError(
                        "provider returned neither final text nor client tool calls"
                    )
                return replace(response, usage=total_usage)
            pending: list[ToolResult] = []
            for call in response.tool_calls:
                pending.append(await self._execute_tool_call(request, call))
            pending_results = tuple(pending)
            continuation = response.continuation
        raise BudgetExceeded(
            "logical agent turn exceeded max_agent_turns "
            f"({self.ledger.limits.max_agent_turns})"
        )

    async def _execute_tool_call(
        self, request: ModelRequest, call: ToolCall
    ) -> ToolResult:
        assert self.environment is not None
        target_agent = call.agent_id or request.agent_id
        await self.ledger.reserve_tool_call()
        arguments_json = json.dumps(dict(call.arguments), sort_keys=True, default=str)
        queued_data: dict[str, Any] = {
            "tool": call.name,
            "kind": call.kind,
            "arguments_chars": len(arguments_json),
            "arguments_sha256": hashlib.sha256(
                arguments_json.encode("utf-8")
            ).hexdigest(),
        }
        if self.capture_content:
            queued_data["arguments"] = dict(call.arguments)
        await self.trace.emit(
            "tool_call_started",
            agent_id=target_agent,
            role=request.role,
            data=queued_data,
        )
        try:
            environment = await self.environment.get(target_agent)
            timeout = max(
                0.001,
                self.ledger.limits.wall_time_seconds - self.trace.elapsed,
            )
            execution: ToolExecution = await asyncio.wait_for(
                environment.execute(call), timeout=timeout
            )
            output_bytes = len(execution.output.encode("utf-8"))
            if execution.image_data_url is not None:
                output_bytes += len(execution.image_data_url.encode("ascii"))
            await self.ledger.record_tool_output(output_bytes)
        except asyncio.CancelledError:
            await self.trace.emit(
                "tool_call_cancelled",
                agent_id=target_agent,
                role=request.role,
                data={"tool": call.name, "kind": call.kind},
            )
            raise
        except Exception as exc:
            await self.trace.emit(
                "tool_call_failed",
                agent_id=target_agent,
                role=request.role,
                data={
                    "tool": call.name,
                    "kind": call.kind,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        completed_data: dict[str, Any] = {
            "tool": call.name,
            "kind": call.kind,
            "is_error": execution.is_error,
            "output_bytes": output_bytes,
            "output_chars": len(execution.output),
            "output_sha256": hashlib.sha256(
                execution.output.encode("utf-8")
            ).hexdigest(),
            "has_image": execution.image_data_url is not None,
            **dict(execution.metadata),
        }
        if self.capture_content:
            completed_data["output"] = execution.output
            completed_data["image_data_url"] = execution.image_data_url
        await self.trace.emit(
            "tool_call_completed",
            agent_id=target_agent,
            role=request.role,
            data=completed_data,
        )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=execution.output,
            kind=call.kind,
            is_error=execution.is_error,
            image_data_url=execution.image_data_url,
            native_output=execution.native_output,
        )

    async def _call_model(self, request: ModelRequest) -> ModelResponse:
        prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        queued_data: dict[str, Any] = {
            "prompt_chars": len(request.prompt),
            "prompt_sha256": prompt_hash,
            "tool_count": len(request.tools),
            "tool_result_count": len(request.tool_results),
            "continuation": request.continuation is not None,
        }
        if self.capture_content:
            queued_data.update(
                {
                    "prompt": request.prompt,
                    "system": request.system,
                    "request_metadata": dict(request.metadata),
                    "tools": [asdict(tool) for tool in request.tools],
                    "tool_results": [asdict(result) for result in request.tool_results],
                }
            )
        await self.trace.emit(
            "model_call_queued",
            agent_id=request.agent_id,
            role=request.role,
            data=queued_data,
        )
        started: Optional[float] = None
        finished: Optional[float] = None
        response: Optional[ModelResponse] = None
        usage_recorded = False
        partial_usage: Optional[Usage] = None

        def report_partial_usage(usage: Usage) -> None:
            nonlocal partial_usage
            partial_usage = usage

        backend_request = replace(request, usage_reporter=report_partial_usage)
        try:
            async with self.ledger.semaphore:
                await self.ledger.reserve_call()
                await self.trace.emit(
                    "model_call_started",
                    agent_id=request.agent_id,
                    role=request.role,
                )
                timeout = max(
                    0.001,
                    self.ledger.limits.wall_time_seconds - self.trace.elapsed,
                )
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        self.backend.complete(backend_request), timeout=timeout
                    )
                finally:
                    # Capture the endpoint around the backend await itself. Ledger
                    # locks and trace serialization are local overhead, not model time.
                    finished = time.perf_counter()
            assert response is not None
            try:
                await self.ledger.record(response.usage)
                usage_recorded = True
            except BudgetExceeded:
                # BudgetLedger commits observed usage before raising on a crossed cap.
                usage_recorded = True
                raise
        except asyncio.CancelledError as exc:
            cancelled_usage = getattr(exc, "usage", None) or partial_usage
            if response is not None and not usage_recorded:
                # The provider has returned billable usage. Finish committing it even
                # when a sibling cancels this task while it is waiting for the ledger.
                try:
                    await asyncio.shield(self.ledger.record(response.usage))
                    usage_recorded = True
                except BudgetExceeded:
                    usage_recorded = True
                except asyncio.CancelledError:
                    # A second cancellation can interrupt the shielded wait. The inner
                    # record continues, but conservatively fail closed meanwhile.
                    await asyncio.shield(self.ledger.mark_incomplete())
            elif isinstance(cancelled_usage, Usage) and not usage_recorded:
                try:
                    await asyncio.shield(self.ledger.record(cancelled_usage))
                    usage_recorded = True
                except BudgetExceeded:
                    usage_recorded = True
            elif started is not None and not usage_recorded:
                await self.ledger.mark_incomplete()
            await self.trace.emit(
                "model_call_cancelled",
                agent_id=request.agent_id,
                role=request.role,
            )
            raise
        except Exception as exc:
            reported_usage = getattr(exc, "usage", None) or partial_usage
            accounting_error: Optional[Exception] = None
            if isinstance(reported_usage, Usage):
                try:
                    await self.ledger.record(reported_usage)
                    usage_recorded = True
                except Exception as usage_exc:
                    if isinstance(usage_exc, BudgetExceeded):
                        usage_recorded = True
                    accounting_error = usage_exc
            if started is not None and not usage_recorded:
                await self.ledger.mark_incomplete()
            failure_data: dict[str, Any] = {
                "error": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(reported_usage, Usage):
                failure_data["reported_usage"] = asdict(reported_usage)
            if self.capture_content and getattr(exc, "raw", None) is not None:
                failure_data["provider_raw"] = getattr(exc, "raw")
            await self.trace.emit(
                "model_call_failed",
                agent_id=request.agent_id,
                role=request.role,
                data=failure_data,
            )
            if accounting_error is not None:
                raise accounting_error from exc
            raise
        finally:
            if started is not None and finished is not None:
                await self.trace.record_interval(started, finished, request.agent_id)
        assert response is not None
        completed_data: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens,
            "cache_write_input_tokens": response.usage.cache_write_input_tokens,
            "cost_usd": response.usage.cost_usd,
            "cost_known": response.usage.cost_known,
            "usage_complete": response.usage.complete,
            "provider_latency_seconds": response.provider_latency_seconds,
            "response_chars": len(response.text),
            "response_sha256": hashlib.sha256(
                response.text.encode("utf-8")
            ).hexdigest(),
            "tool_calls": len(response.tool_calls),
            "tool_call_names": [call.name for call in response.tool_calls],
        }
        if self.capture_content:
            completed_data.update(
                {"response_text": response.text, "provider_raw": response.raw}
            )
        await self.trace.emit(
            "model_call_completed",
            agent_id=request.agent_id,
            role=request.role,
            data=completed_data,
        )
        return response


class ScriptedBackend:
    """Deterministic backend for unit tests and offline harness smoke runs."""

    def __init__(
        self,
        scripts: Mapping[str, Sequence[str | ModelResponse]],
        *,
        delays: Optional[Mapping[str, float]] = None,
    ) -> None:
        serialized_scripts = {
            key: [
                asdict(value) if isinstance(value, ModelResponse) else value
                for value in values
            ]
            for key, values in scripts.items()
        }
        self._provenance = {
            "provider": "scripted",
            "script_sha256": hashlib.sha256(
                json.dumps(serialized_scripts, sort_keys=True, default=str).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "delays": dict(delays or {}),
        }
        self._scripts: MutableMapping[str, Deque[str | ModelResponse]] = {
            key: deque(values) for key, values in scripts.items()
        }
        self._delays = dict(delays or {})
        self.requests: list[ModelRequest] = []
        self._lock = asyncio.Lock()

    def provenance(self) -> Mapping[str, Any]:
        return dict(self._provenance)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        async with self._lock:
            self.requests.append(request)
            key = self._select_key(request)
            if key is None:
                raise AssertionError(
                    f"no scripted response for {request.agent_id!r}/{request.role!r}"
                )
            value = self._scripts[key].popleft()
            delay = self._delays.get(key, 0.0)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(value, ModelResponse):
            return value
        return ModelResponse(
            text=value,
            usage=Usage(
                input_tokens=max(1, len(request.prompt.split())),
                output_tokens=max(1, len(value.split())),
            ),
            provider_latency_seconds=delay,
        )

    def _select_key(self, request: ModelRequest) -> Optional[str]:
        candidates = (
            f"{request.agent_id}:{request.role}",
            request.agent_id,
            f"role:{request.role}",
            "*",
        )
        for key in candidates:
            if key in self._scripts and self._scripts[key]:
                return key
        return None


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ProtocolError(f"expected a JSON object: {exc}") from exc
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as nested:
            raise ProtocolError(f"invalid JSON object: {nested}") from nested
    if not isinstance(value, dict):
        raise ProtocolError("expected a JSON object")
    return value


def require_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ProtocolError(f"{key!r} must be a non-empty string")
    return item.strip()


def require_string_list(value: Mapping[str, Any], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in item
    ):
        raise ProtocolError(f"{key!r} must be a list of non-empty strings")
    return [entry.strip() for entry in item]


def write_trace_jsonl(path: Path, events: Iterable[TraceEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(asdict(event), sort_keys=True, default=str) + "\n")
