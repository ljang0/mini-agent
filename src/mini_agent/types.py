"""Small provider-neutral data contracts used by :mod:`mini_agent`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple


class MiniAgentError(RuntimeError):
    """Base error for the minimal runtime."""


class BudgetExceeded(MiniAgentError):
    """Raised before or immediately after a configured budget is crossed."""


class ProtocolError(MiniAgentError):
    """Raised when a model or environment violates a public contract."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
    )
    kind: str = "function"
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("tool description must be a string")
        if not isinstance(self.input_schema, Mapping):
            raise ValueError("tool input_schema must be an object")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("tool kind must be a non-empty string")
        if not isinstance(self.provider_options, Mapping):
            raise ValueError("tool provider_options must be an object")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    kind: str = "function"
    agent_id: str = ""
    raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("tool call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool arguments must be an object")


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    kind: str = "function"
    is_error: bool = False
    image_data_url: Optional[str] = None
    native_output: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("tool result call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool result name must be a non-empty string")
        if not isinstance(self.output, str):
            raise ValueError("tool result output must be a string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("tool result kind must be a non-empty string")
        if not isinstance(self.is_error, bool):
            raise ValueError("tool result is_error must be a boolean")
        if self.image_data_url is not None and not self.image_data_url.startswith(
            "data:image/"
        ):
            raise ValueError("image_data_url must be an image data URL")


@dataclass(frozen=True)
class ToolExecution:
    """An environment observation returned after one tool action."""

    output: str
    is_error: bool = False
    image_data_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    native_output: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise ValueError("tool execution output must be a string")
        if not isinstance(self.is_error, bool):
            raise ValueError("tool execution is_error must be a boolean")
        if self.image_data_url is not None and not self.image_data_url.startswith(
            "data:image/"
        ):
            raise ValueError("image_data_url must be an image data URL")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("tool execution metadata must be an object")


Observation = ToolExecution


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    complete: bool = True

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cache_read_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            raise ValueError("cache input-token classes cannot exceed input_tokens")
        if (
            not isinstance(self.cost_usd, (int, float))
            or isinstance(self.cost_usd, bool)
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and non-negative")
        if not isinstance(self.cost_known, bool) or not isinstance(self.complete, bool):
            raise ValueError("usage flags must be booleans")

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_write_input_tokens=self.cache_write_input_tokens
            + other.cache_write_input_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cost_known=self.cost_known and other.cost_known,
            complete=self.complete and other.complete,
        )


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int = 64
    max_concurrency: int = 4
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    wall_time_seconds: float = 900.0
    max_tool_calls: int = 256
    max_tool_output_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_model_calls", "max_concurrency", "max_tool_calls", "max_tool_output_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            not isinstance(self.wall_time_seconds, (int, float))
            or isinstance(self.wall_time_seconds, bool)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
        ):
            raise ValueError("wall_time_seconds must be positive")
        if self.max_cost_usd is not None and (
            not isinstance(self.max_cost_usd, (int, float))
            or isinstance(self.max_cost_usd, bool)
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be finite and non-negative or None")


@dataclass(frozen=True)
class Message:
    """One provider-neutral item in the agent's exact linear history."""

    role: str
    content: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_results: Tuple[ToolResult, ...] = ()
    image_data_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    provider_latency_seconds: float = 0.0
    raw: Any = field(default=None, repr=False, compare=False)
    tool_calls: Tuple[ToolCall, ...] = ()
    continuation: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("model response text must be a string")


@dataclass(frozen=True)
class ModelRequest:
    """Compatibility request for external reference backends."""

    agent_id: str
    role: str
    prompt: str
    system: str = ""
    max_output_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    input_images: Tuple[str, ...] = ()
    tools: Tuple[ToolDefinition, ...] = ()
    tool_results: Tuple[ToolResult, ...] = ()
    continuation: Any = field(default=None, repr=False, compare=False)
    usage_reporter: Optional[Callable[[Usage], None]] = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class TraceEvent:
    event: str
    elapsed_seconds: float
    agent_id: str = ""
    role: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    messages: Tuple[Message, ...]
    steps: int
    status: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentResult",
    "BudgetExceeded",
    "BudgetLimits",
    "Message",
    "MiniAgentError",
    "ModelRequest",
    "ModelResponse",
    "Observation",
    "ProtocolError",
    "ToolCall",
    "ToolDefinition",
    "ToolExecution",
    "ToolResult",
    "TraceEvent",
    "Usage",
]
