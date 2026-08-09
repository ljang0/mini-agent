from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple


class ScaffoldLabError(RuntimeError):
    """Base exception for the experiment runtime."""


class BudgetExceeded(ScaffoldLabError):
    """Raised when a run would exceed a configured resource limit."""


class ProtocolError(ScaffoldLabError):
    """Raised when a model emits an invalid harness action."""


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral description of a client-executed tool.

    ``kind`` identifies a provider-native schema when one is public (for example,
    OpenAI's ``computer`` tool or Anthropic's ``bash_20250124`` tool).  The default
    ``function`` kind is portable across providers that implement JSON tool calls.
    """

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
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
    """One structured client-tool request returned by a model provider."""

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
    """Provider-neutral output sent back for a client tool call."""

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
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    complete: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_tokens, int)
            or isinstance(self.input_tokens, bool)
            or self.input_tokens < 0
        ):
            raise ValueError("input_tokens must be a non-negative integer")
        if (
            not isinstance(self.output_tokens, int)
            or isinstance(self.output_tokens, bool)
            or self.output_tokens < 0
        ):
            raise ValueError("output_tokens must be a non-negative integer")
        for name in ("cache_read_input_tokens", "cache_write_input_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.cache_read_input_tokens + self.cache_write_input_tokens
            > self.input_tokens
        ):
            raise ValueError("cache input-token classes cannot exceed input_tokens")
        if (
            not isinstance(self.cost_usd, (int, float))
            or isinstance(self.cost_usd, bool)
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and non-negative")
        if not isinstance(self.cost_known, bool):
            raise ValueError("cost_known must be a boolean")
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be a boolean")

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
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
    max_depth: int = 3
    max_tool_calls: int = 256
    max_tool_output_bytes: int = 8 * 1024 * 1024
    max_agent_turns: int = 64

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_model_calls, int)
            or isinstance(self.max_model_calls, bool)
            or self.max_model_calls < 1
        ):
            raise ValueError("max_model_calls must be positive")
        if (
            not isinstance(self.max_concurrency, int)
            or isinstance(self.max_concurrency, bool)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be positive")
        if (
            not isinstance(self.wall_time_seconds, (int, float))
            or isinstance(self.wall_time_seconds, bool)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
        ):
            raise ValueError("wall_time_seconds must be positive")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth < 0
        ):
            raise ValueError("max_depth cannot be negative")
        for name in ("max_tool_calls", "max_tool_output_bytes", "max_agent_turns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.max_cost_usd is not None and (
            not isinstance(self.max_cost_usd, (int, float))
            or isinstance(self.max_cost_usd, bool)
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be finite and non-negative or None")


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    context: str = ""
    reference_answer: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    agent_id: str
    role: str
    prompt: str
    system: str = ""
    max_output_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: Tuple[ToolDefinition, ...] = ()
    tool_results: Tuple[ToolResult, ...] = ()
    continuation: Any = field(default=None, repr=False, compare=False)
    usage_reporter: Optional[Callable[[Usage], None]] = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    provider_latency_seconds: float = 0.0
    raw: Any = None
    tool_calls: Tuple[ToolCall, ...] = ()
    continuation: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TraceEvent:
    event: str
    elapsed_seconds: float
    agent_id: str = ""
    role: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    task_id: str
    harness: str
    answer: str
    usage: Usage
    model_calls: int
    wall_time_seconds: float
    backend_active_union_seconds: float
    trace: Tuple[TraceEvent, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    tool_output_bytes: int = 0


class RunFailed(ScaffoldLabError):
    """A harness failure carrying the resources and trace consumed before failure."""

    def __init__(
        self,
        message: str,
        *,
        cause_type: str,
        usage: Usage,
        model_calls: int,
        wall_time_seconds: float,
        backend_active_union_seconds: float,
        trace: Tuple[TraceEvent, ...],
        tool_calls: int = 0,
        tool_output_bytes: int = 0,
    ) -> None:
        super().__init__(message)
        self.cause_type = cause_type
        self.usage = usage
        self.model_calls = model_calls
        self.wall_time_seconds = wall_time_seconds
        self.backend_active_union_seconds = backend_active_union_seconds
        self.trace = trace
        self.tool_calls = tool_calls
        self.tool_output_bytes = tool_output_bytes
