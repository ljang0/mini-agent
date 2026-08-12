"""Shared budgets and traces for one or many :class:`MiniAgent` instances."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import stat
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .environments.base import AgentEnvironment
    from .models import Model

from .types import (
    BudgetExceeded,
    BudgetLimits,
    InfrastructureError,
    InvalidAction,
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
        if not isinstance(limits, BudgetLimits):
            raise ValueError("budget limits must be BudgetLimits")
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
        self._agent_started: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)

    def configure_agent(self, agent_id: str, limits: BudgetLimits) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        if not isinstance(limits, BudgetLimits):
            raise ValueError("agent limits must be BudgetLimits")
        existing = self._agent_limits.get(agent_id)
        if existing is not None and existing != limits:
            raise ValueError(f"agent budget already configured for {agent_id!r}")
        self._agent_limits[agent_id] = limits
        self._agent_semaphores.setdefault(
            agent_id, asyncio.Semaphore(limits.max_concurrency)
        )
        self._agent_started.setdefault(agent_id, time.perf_counter())

    def agent_wall_time_remaining(self, agent_id: str) -> float | None:
        limits = self._agent_limits.get(agent_id)
        started = self._agent_started.get(agent_id)
        if limits is None or started is None:
            return None
        return limits.wall_time_seconds - (time.perf_counter() - started)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    def agent_semaphore(self, agent_id: str) -> Any:
        limits = self._agent_limits.get(agent_id)
        if limits is None:
            return nullcontext()
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

    def snapshot(self, *, prefix: str | None = None) -> Mapping[str, Any]:
        if prefix is None:
            return {
                "model_calls": self.calls,
                "tool_calls": self.tool_calls,
                "tool_output_bytes": self.tool_output_bytes,
                "usage": asdict(self.usage),
                "exhausted_reason": self._exhausted_reason,
            }
        agent_ids = {
            *self._agent_calls,
            *self._agent_tool_calls,
            *self._agent_tool_output_bytes,
            *self._agent_usage,
        }
        selected = [
            agent_id
            for agent_id in agent_ids
            if (
                agent_id.startswith("/")
                if prefix == "/"
                else agent_id == prefix or agent_id.startswith(prefix + "/")
            )
        ]
        usage = Usage()
        for agent_id in selected:
            usage = usage + self._agent_usage.get(agent_id, Usage())
        return {
            "model_calls": sum(
                self._agent_calls.get(agent_id, 0) for agent_id in selected
            ),
            "tool_calls": sum(
                self._agent_tool_calls.get(agent_id, 0) for agent_id in selected
            ),
            "tool_output_bytes": sum(
                self._agent_tool_output_bytes.get(agent_id, 0) for agent_id in selected
            ),
            "usage": asdict(usage),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore global counters before a resumed evaluation starts."""

        if not isinstance(snapshot, Mapping):
            raise ValueError("budget snapshot must be an object")
        if (
            self.calls
            or self.tool_calls
            or self.tool_output_bytes
            or self.usage != Usage()
        ):
            raise RuntimeError("budget ledger can be restored only before use")
        calls = _snapshot_count(snapshot, "model_calls")
        tool_calls = _snapshot_count(snapshot, "tool_calls")
        output_bytes = _snapshot_count(snapshot, "tool_output_bytes")
        raw_usage = snapshot.get("usage", {})
        if not isinstance(raw_usage, Mapping):
            raise ValueError("budget snapshot usage must be an object")
        usage = Usage(**dict(raw_usage))
        restored_reason = snapshot.get("exhausted_reason")
        if restored_reason is not None and (
            not isinstance(restored_reason, str) or not restored_reason.strip()
        ):
            raise ValueError(
                "budget snapshot exhausted_reason must be a string or null"
            )
        self.calls = calls
        self.tool_calls = tool_calls
        self.tool_output_bytes = output_bytes
        self.usage = usage
        self._exhausted_reason = restored_reason
        if (
            self._exhausted_reason is None
            and not usage.complete
            and any(
                value is not None
                for value in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            )
        ):
            self._exhausted_reason = (
                "restored resource budget cannot be verified because usage "
                "is incomplete"
            )
        elif (
            self._exhausted_reason is None
            and self.limits.max_cost_usd is not None
            and not usage.cost_known
        ):
            self._exhausted_reason = (
                "restored cost budget cannot be verified because usage cost is unknown"
            )
        elif self._exhausted_reason is None:
            self._exhausted_reason = self._resource_reason(
                self.limits, usage, crossed=True
            )
        if (
            self._exhausted_reason is None
            and output_bytes > self.limits.max_tool_output_bytes
        ):
            self._exhausted_reason = "restored tool-output byte budget was exceeded"

    @staticmethod
    def _resource_reason(
        limits: BudgetLimits, usage: Usage, *, crossed: bool
    ) -> str | None:
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
            if (
                reason is None
                and self.limits.max_cost_usd is not None
                and not self.usage.cost_known
            ):
                reason = "cost budget cannot be verified because usage cost is unknown"
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
            if agent_id:
                self._agent_calls[agent_id] = self._agent_calls.get(agent_id, 0) + 1

    async def record(self, usage: Usage, agent_id: str = "") -> None:
        if not isinstance(usage, Usage):
            raise ValueError("recorded usage must be Usage")
        async with self._lock:
            self.usage = self.usage + usage
            agent_limits = self._agent_limits.get(agent_id)
            if agent_id:
                self._agent_usage[agent_id] = (
                    self._agent_usage.get(agent_id, Usage()) + usage
                )
            reason: str | None = None
            if not usage.complete and any(
                value is not None
                for value in (
                    self.limits.max_input_tokens,
                    self.limits.max_output_tokens,
                    self.limits.max_cost_usd,
                )
            ):
                reason = (
                    "resource budget cannot be verified because usage is incomplete"
                )
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
                elif (
                    agent_limits.max_cost_usd is not None and not agent_usage.cost_known
                ):
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
            if agent_id:
                self._agent_usage[agent_id] = _incomplete(
                    self._agent_usage.get(agent_id, Usage())
                )
                if agent_id in self._agent_limits:
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
            if agent_id:
                self._agent_tool_calls[agent_id] = (
                    self._agent_tool_calls.get(agent_id, 0) + 1
                )

    async def record_tool_output(self, byte_count: int, agent_id: str = "") -> None:
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
            agent_reason: str | None = None
            agent_limits = self._agent_limits.get(agent_id)
            if agent_id:
                total = self._agent_tool_output_bytes.get(agent_id, 0) + byte_count
                self._agent_tool_output_bytes[agent_id] = total
                if (
                    agent_limits is not None
                    and total > agent_limits.max_tool_output_bytes
                ):
                    agent_reason = "agent tool-output byte budget exceeded"
                    self._agent_exhausted[agent_id] = agent_reason
            if self._exhausted_reason is not None:
                raise BudgetExceeded(self._exhausted_reason)
            if agent_reason is not None:
                raise BudgetExceeded(agent_reason)


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


def _snapshot_count(snapshot: Mapping[str, Any], name: str) -> int:
    value = snapshot.get(name, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"budget snapshot {name} must be a non-negative integer")
    return value


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "authorization",
        "cookie",
        "password",
        "proxy-authorization",
        "secret",
        "set-cookie",
        "token",
        "x-api-key",
    }
)
_URL_SECRET = re.compile(
    r"([?&](?:api[_-]?key|key|token|access_token)=)[^&#\s]+",
    re.IGNORECASE,
)


def _private_append_descriptor(path: Path) -> int:
    """Open a regular trace file privately before any content is written."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError("trace path must be a regular non-symlink file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("trace path must be a regular non-symlink file")
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _sync_directory(path: Path) -> None:
    """Persist a directory entry before it becomes crash-recovery evidence."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TraceRecorder:
    """Concurrency-safe trace sink with optional streamed JSONL output."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        secrets: tuple[str, ...] = (),
        elapsed_offset: float = 0.0,
        backend_active_offset: float = 0.0,
    ) -> None:
        for name, value in (
            ("elapsed_offset", elapsed_offset),
            ("backend_active_offset", backend_active_offset),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if path is not None and not isinstance(path, Path):
            raise ValueError("trace path must be a Path or None")
        self.started = time.perf_counter()
        self._elapsed_offset = float(elapsed_offset)
        self._backend_active_offset = float(backend_active_offset)
        self.events: list[TraceEvent] = []
        self._lock = asyncio.Lock()
        self._call_intervals: list[tuple[float, float, str]] = []
        if path is None:
            self.path = None
        else:
            expanded = path.expanduser()
            self.path = expanded.parent.resolve() / expanded.name
        self._secrets = _normalized_secrets(secrets, "trace secrets")
        if self.path is not None:
            missing: list[Path] = []
            ancestor = self.path.parent
            while not ancestor.exists():
                missing.append(ancestor)
                if ancestor == ancestor.parent:
                    break
                ancestor = ancestor.parent
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            for created in reversed(missing):
                created.chmod(0o700)
            descriptor = _private_append_descriptor(self.path)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _sync_directory(self.path.parent)
            for created in missing:
                _sync_directory(created.parent)

    async def emit(
        self,
        event: str,
        *,
        agent_id: str = "",
        role: str = "",
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if data is not None and not isinstance(data, Mapping):
            raise ValueError("trace data must be an object or None")
        async with self._lock:
            item = TraceEvent(
                event=event,
                elapsed_seconds=self.elapsed,
                agent_id=agent_id,
                role=role,
                data=_redact(dict(data or {}), self._secrets),
            )
            self.events.append(item)
            if self.path is not None:
                line = json.dumps(
                    asdict(item), sort_keys=True, allow_nan=False
                ) + "\n"
                descriptor = _private_append_descriptor(self.path)
                with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    if event in {"model_call_started", "tool_call_started"}:
                        # Resume treats operation starts as proof that a paid or
                        # side-effecting request may have begun, so persist the
                        # record before crossing that external boundary.
                        os.fsync(stream.fileno())

    async def sync(self) -> None:
        """Make every trace record emitted so far durable on disk."""

        async with self._lock:
            if self.path is None:
                return
            descriptor = _private_append_descriptor(self.path)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    async def record_interval(self, start: float, finish: float, agent_id: str) -> None:
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not math.isfinite(float(start))
            or not isinstance(finish, (int, float))
            or isinstance(finish, bool)
            or not math.isfinite(float(finish))
            or finish < start
            or not isinstance(agent_id, str)
        ):
            raise ValueError("model interval is invalid")
        async with self._lock:
            self._call_intervals.append((start, finish, agent_id))

    @property
    def elapsed(self) -> float:
        return self._elapsed_offset + time.perf_counter() - self.started

    @property
    def backend_active_union_seconds(self) -> float:
        intervals = sorted((start, finish) for start, finish, _ in self._call_intervals)
        merged: list[tuple[float, float]] = []
        for start, finish in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, finish))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
        return self._backend_active_offset + sum(
            finish - start for start, finish in merged
        )


def _redact(value: Any, secrets: tuple[str, ...], *, key: str = "") -> Any:
    if key.casefold() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                raise ValueError("trace data keys must be strings")
            redacted[item_key] = _redact(item, secrets, key=item_key)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/"):
            digest = hashlib.sha256(value.encode("ascii", errors="ignore")).hexdigest()
            return f"<image sha256={digest} chars={len(value)}>"
        result = _URL_SECRET.sub(r"\1<redacted>", value)
        for secret in secrets:
            result = result.replace(secret, "<redacted>")
        return result
    return value


def _normalized_secrets(secrets: tuple[str, ...], label: str) -> tuple[str, ...]:
    """Validate and order secrets longest-first for greedy redaction."""

    if not isinstance(secrets, tuple) or not all(
        isinstance(secret, str) for secret in secrets
    ):
        raise ValueError(f"{label} must be a tuple of strings")
    return tuple(
        sorted(
            {secret for secret in secrets if secret},
            key=lambda item: (-len(item), item),
        )
    )


def redact_artifact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Apply the trace redaction policy to another durable JSON artifact."""

    return _redact(value, _normalized_secrets(secrets, "artifact secrets"))


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
        if limits is not None and ledger is not None:
            raise ValueError("pass either limits or ledger, not both")
        if limits is not None and not isinstance(limits, BudgetLimits):
            raise ValueError("limits must be BudgetLimits or None")
        if ledger is not None and not isinstance(ledger, BudgetLedger):
            raise ValueError("ledger must be BudgetLedger or None")
        if trace is not None and not isinstance(trace, TraceRecorder):
            raise ValueError("trace must be TraceRecorder or None")
        if not isinstance(capture_content, bool):
            raise ValueError("capture_content must be a boolean")
        if limits is None and ledger is None:
            limits = BudgetLimits()
        self.ledger = ledger or BudgetLedger(limits or BudgetLimits())
        self.trace = trace or TraceRecorder()
        self.capture_content = capture_content

    def configure_agent(self, agent_id: str, limits: BudgetLimits) -> None:
        self.ledger.configure_agent(agent_id, limits)

    async def record_initial_observation(
        self,
        observation: ToolExecution,
        *,
        agent_id: str,
        role: str,
    ) -> None:
        """Charge and trace environment data delivered before the first model call."""

        if not isinstance(observation, ToolExecution):
            raise ValueError("initial observation must be ToolExecution")
        output_bytes = _tool_execution_bytes(observation)
        try:
            await self.ledger.record_tool_output(output_bytes, agent_id)
        except Exception as exc:
            await self.trace.emit(
                "initial_observation_failed",
                agent_id=agent_id,
                role=role,
                data={
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "output_bytes": output_bytes,
                },
            )
            raise
        await self.trace.emit(
            "initial_observation_completed",
            agent_id=agent_id,
            role=role,
            data={
                **dict(observation.metadata),
                "is_error": observation.is_error,
                "output_bytes": output_bytes,
                **(
                    {"output": observation.output}
                    if self.capture_content
                    else {}
                ),
            },
        )

    async def query(
        self,
        model: "Model",
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        *,
        agent_id: str,
        role: str,
    ) -> ModelResponse:
        tool_values = [_tool_definition_value(tool) for tool in tools]
        data: dict[str, Any] = {
            "message_count": len(messages),
            "tool_count": len(tools),
            "history_sha256": hashlib.sha256(
                json.dumps(
                    [_message_value(message) for message in messages],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "tools_sha256": hashlib.sha256(
                json.dumps(
                    tool_values,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        if self.capture_content:
            data["messages"] = [_message_value(message) for message in messages]
            data["tools"] = tool_values
        await self.trace.emit(
            "model_call_queued", agent_id=agent_id, role=role, data=data
        )
        started: float | None = None
        finished: float | None = None
        response: ModelResponse | None = None
        try:
            async with self.ledger.semaphore:
                async with self.ledger.agent_semaphore(agent_id):
                    self._remaining_time(agent_id)
                    await self.ledger.reserve_call(agent_id)
                    await self.trace.emit(
                        "model_call_started", agent_id=agent_id, role=role
                    )
                    timeout = self._remaining_time(agent_id)
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
            accounting_error: BudgetExceeded | None = None
            try:
                if isinstance(reported_usage, Usage):
                    await self.ledger.record(reported_usage, agent_id)
                elif started is not None and not isinstance(response, ModelResponse):
                    await self.ledger.mark_incomplete(agent_id)
            except BudgetExceeded as budget_error:
                accounting_error = budget_error
            await self.trace.emit(
                "model_call_failed",
                agent_id=agent_id,
                role=role,
                data={
                    "error": type(exc).__name__,
                    "message": str(exc),
                    **(
                        {"attempts": attempts}
                        if (attempts := getattr(exc, "attempts", None)) is not None
                        else {}
                    ),
                },
            )
            if accounting_error is not None:
                raise accounting_error from exc
            raise
        finally:
            if started is not None and finished is not None:
                await self.trace.record_interval(started, finished, agent_id)
        assert response is not None
        completed: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": response.usage.cost_usd,
            "cost_known": response.usage.cost_known,
            "usage_complete": response.usage.complete,
            "tool_calls": len(response.tool_calls),
            "response_chars": len(response.text),
            "retries": response.retries,
        }
        if response.resolved_model is not None:
            completed["resolved_model_sha256"] = hashlib.sha256(
                response.resolved_model.encode("utf-8")
            ).hexdigest()
        if self.capture_content:
            completed["response"] = response.text
        await self.trace.emit(
            "model_call_completed", agent_id=agent_id, role=role, data=completed
        )
        return response

    async def execute(
        self,
        environment: "AgentEnvironment",
        action: ToolCall,
        tools: Sequence[ToolDefinition],
        *,
        agent_id: str,
        role: str,
    ) -> ToolResult:
        self._remaining_time(agent_id)
        await self.ledger.reserve_tool_call(agent_id)
        arguments = json.dumps(
            dict(action.arguments),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        await self.trace.emit(
            "tool_call_started",
            agent_id=agent_id,
            role=role,
            data={
                "tool": action.name,
                "arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                **(
                    {"arguments": dict(action.arguments)}
                    if self.capture_content
                    else {}
                ),
            },
        )
        try:
            definitions = {tool.name: tool for tool in tools}
            definition = definitions.get(action.name)
            if definition is None:
                return await self._invalid_action(
                    action,
                    f"unknown tool {action.name!r}",
                    agent_id=agent_id,
                    role=role,
                )
            schema_error = _validate_arguments(
                action.arguments, definition.input_schema
            )
            if schema_error is not None:
                return await self._invalid_action(
                    action, schema_error, agent_id=agent_id, role=role
                )
            timeout = self._remaining_time(agent_id)
            execution = await asyncio.wait_for(
                environment.execute(action), timeout=timeout
            )
            if not isinstance(execution, ToolExecution):
                try:
                    execution = ToolExecution(
                        output=execution.output,
                        is_error=execution.is_error,
                        image_data_url=execution.image_data_url,
                        metadata=getattr(execution, "metadata", {}),
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise InfrastructureError(
                        "environment.execute must return a compatible ToolExecution"
                    ) from exc
        except InvalidAction as exc:
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
        output_bytes = _tool_execution_bytes(execution)
        await self._record_tool_output(
            action, output_bytes, agent_id=agent_id, role=role
        )
        await self.trace.emit(
            "tool_call_completed",
            agent_id=agent_id,
            role=role,
            data={
                **dict(execution.metadata),
                "tool": action.name,
                "is_error": execution.is_error,
                "output_bytes": output_bytes,
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
        await self._record_tool_output(
            action,
            len(output.encode("utf-8")),
            agent_id=agent_id,
            role=role,
        )
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

    async def _record_tool_output(
        self,
        action: ToolCall,
        byte_count: int,
        *,
        agent_id: str,
        role: str,
    ) -> None:
        try:
            await self.ledger.record_tool_output(byte_count, agent_id)
        except Exception as exc:
            await self.trace.emit(
                "tool_call_failed",
                agent_id=agent_id,
                role=role,
                data={
                    "tool": action.name,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "output_bytes": byte_count,
                },
            )
            raise

    def _remaining_time(self, agent_id: str) -> float:
        remaining = self.ledger.limits.wall_time_seconds - self.trace.elapsed
        agent_remaining = self.ledger.agent_wall_time_remaining(agent_id)
        if agent_remaining is not None:
            remaining = min(remaining, agent_remaining)
        if remaining <= 0:
            raise BudgetExceeded("global or per-agent wall-time budget exhausted")
        return remaining


def _tool_execution_bytes(execution: ToolExecution) -> int:
    total = len(execution.output.encode("utf-8"))
    if execution.image_data_url is not None:
        total += len(execution.image_data_url.encode("utf-8"))
    return total


def _validate_arguments(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
    if schema.get("type", "object") != "object":
        raise InfrastructureError("tool schema root must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        raise InfrastructureError("tool schema required must be a list of strings")
    if len(required) != len(set(required)):
        raise InfrastructureError("tool schema required entries must be unique")
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"missing required arguments {missing}"
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise InfrastructureError("tool schema properties must be an object")
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
        if spec is None:
            continue
        if not isinstance(spec, Mapping):
            raise InfrastructureError(
                f"tool schema property {name!r} must be an object"
            )
        if "type" not in spec:
            continue
        if not isinstance(spec["type"], str):
            raise InfrastructureError(
                f"tool schema property {name!r} type must be a string"
            )
        expected = expected_types.get(str(spec["type"]))
        if expected is not None and (
            not isinstance(value, expected)
            or isinstance(value, bool)
            and spec["type"] in {"integer", "number"}
        ):
            return f"argument {name!r} must be {spec['type']}"
        choices = spec.get("enum")
        if choices is not None:
            if not isinstance(choices, list) or not choices:
                raise InfrastructureError(
                    f"tool schema property {name!r} enum must be a non-empty list"
                )
            if value not in choices:
                return f"argument {name!r} must be one of {choices!r}"
        if spec.get("type") == "array" and isinstance(value, list):
            minimum = spec.get("minItems")
            if (
                isinstance(minimum, int)
                and not isinstance(minimum, bool)
                and len(value) < minimum
            ):
                return f"argument {name!r} requires at least {minimum} items"
    return None


def _tool_call_value(call: ToolCall) -> Mapping[str, Any]:
    return {
        "call_id": call.call_id,
        "name": call.name,
        "arguments": dict(call.arguments),
        "kind": call.kind,
    }


def _tool_result_value(result: ToolResult) -> Mapping[str, Any]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "output": result.output,
        "kind": result.kind,
        "is_error": result.is_error,
        "image_data_url": result.image_data_url,
    }


def _message_value(message: Message) -> Mapping[str, Any]:
    """Serialize only provider-neutral history, never raw/native SDK objects."""

    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [_tool_call_value(call) for call in message.tool_calls],
        "tool_results": [
            _tool_result_value(result) for result in message.tool_results
        ],
        "image_data_url": message.image_data_url,
        "metadata": dict(message.metadata),
    }


def _tool_definition_value(tool: ToolDefinition) -> Mapping[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.input_schema),
        "kind": tool.kind,
    }


__all__ = ["BudgetLedger", "RunContext", "TraceRecorder"]
