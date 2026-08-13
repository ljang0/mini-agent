"""Small provider-neutral data contracts used by :mod:`mini_agent`."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Optional, Tuple


def _require_str(
    value: Any,
    label: str,
    *,
    non_empty: bool = True,
    stripped: bool = False,
    error: type[Exception] = ValueError,
) -> str:
    """Return ``value`` when it is a string of the requested shape."""

    if (
        not isinstance(value, str)
        or (non_empty and not value.strip())
        or (stripped and value != value.strip())
    ):
        article = "a non-empty string" if non_empty else "a string"
        raise error(f"{label} must be {article}")
    return value


def _require_bool(
    value: Any, label: str, *, error: type[Exception] = ValueError
) -> bool:
    """Return ``value`` when it is a real boolean."""

    if not isinstance(value, bool):
        raise error(f"{label} must be a boolean")
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    error: type[Exception] = ValueError,
) -> int:
    """Return ``value`` when it is a bounded integer (never a ``bool``)."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise error(f"{label} must be {_int_requirement(minimum, maximum)}")
    return value


def _int_requirement(minimum: int | None, maximum: int | None) -> str:
    if minimum is not None and maximum is not None:
        return f"an integer between {minimum} and {maximum}"
    if minimum == 1:
        return "a positive integer"
    if minimum == 0:
        return "a non-negative integer"
    if minimum is not None:
        return f"an integer of at least {minimum}"
    if maximum is not None:
        return f"an integer of at most {maximum}"
    return "an integer"


def _require_positive_int(
    value: Any, label: str, *, minimum: int = 1, error: type[Exception] = ValueError
) -> int:
    """Return ``value`` when it is an integer at or above ``minimum`` (default 1)."""

    return _require_int(value, label, minimum=minimum, error=error)


def _require_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    error: type[Exception] = ValueError,
) -> float:
    """Return ``value`` as a float when it is a finite, bounded real number."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (minimum is not None and value < minimum)
        or (exclusive_minimum is not None and value <= exclusive_minimum)
    ):
        shape = _number_requirement(minimum, exclusive_minimum)
        raise error(f"{label} must be {shape}")
    return float(value)


def _number_requirement(minimum: float | None, exclusive_minimum: float | None) -> str:
    if exclusive_minimum == 0:
        return "finite and positive"
    if minimum == 0:
        return "finite and non-negative"
    if minimum is not None:
        return f"finite and at least {minimum}"
    if exclusive_minimum is not None:
        return f"finite and above {exclusive_minimum}"
    return "a finite number"


def _require_mapping(
    value: Any, label: str, *, error: type[Exception] = ValueError
) -> Mapping[str, Any]:
    """Return ``value`` when it is a mapping (a decoded JSON object)."""

    if not isinstance(value, Mapping):
        raise error(f"{label} must be an object")
    return value


def _require_callable(
    value: Any, label: str, *, error: type[Exception] = ValueError
) -> Callable[..., Any]:
    """Return ``value`` when it is callable."""

    if not callable(value):
        raise error(f"{label} must be callable")
    return value


def _require_no_symlink(
    path: Path, label: str, *, error: type[Exception] = ValueError
) -> Path:
    """Return ``path`` when it is not a symlink; symlinks escape owned trees."""

    if path.is_symlink():
        raise error(f"{label} must not be a symlink")
    return path


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode deterministic JSON, rejecting duplicate keys and non-finite values."""

    def reject_constant(constant: str) -> NoReturn:
        raise ValueError(f"non-finite JSON number {constant!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = item
        return result

    return _json_value(
        json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        ),
        "JSON",
    )


def _require_utf8(value: str, label: str) -> str:
    """Reject strings that cannot be hashed or written as UTF-8 artifacts."""

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8 text") from exc
    return value


def _json_value(value: Any, label: str) -> Any:
    """Copy a public value while enforcing the JSON artifact contract."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _require_utf8(value, label)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite JSON numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            copied[_require_utf8(key, label)] = _json_value(item, label)
        return copied
    raise ValueError(f"{label} must contain only JSON values")


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    _require_mapping(value, label)
    copied = _json_value(value, label)
    assert isinstance(copied, dict)
    return copied


def _require_text(
    value: Any,
    label: str,
    *,
    non_empty: bool = True,
    stripped: bool = False,
    error: type[Exception] = ValueError,
) -> str:
    """Return ``value`` when it is a UTF-8 encodable string of the wanted shape."""

    return _require_utf8(
        _require_str(
            value, label, non_empty=non_empty, stripped=stripped, error=error
        ),
        label,
    )


def _require_image_url(value: Any, label: str) -> Optional[str]:
    """Return an optional inline image data URL, rejecting other strings."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError(f"{label} must be an image data URL")
    return _require_utf8(value, label)


def _require_tuple_of(
    value: Any,
    kind: type,
    label: str,
    *,
    brief: bool = False,
    error: type[Exception] = ValueError,
) -> tuple[Any, ...]:
    """Return ``value`` when it is a tuple whose items are all ``kind``."""

    if not isinstance(value, tuple) or not all(
        isinstance(item, kind) for item in value
    ):
        shape = "" if brief else "a tuple of "
        raise error(f"{label} must be {shape}{kind.__name__} values")
    return value


class MiniAgentError(RuntimeError):
    """Base error for the minimal runtime."""


class BudgetExceeded(MiniAgentError):
    """Raised before or immediately after a configured budget is crossed."""


class ProtocolError(MiniAgentError):
    """Raised when a model or environment violates a public contract."""


class InvalidAction(ProtocolError):
    """Raised when a model-selected tool action violates its contract."""


class InfrastructureError(MiniAgentError):
    """Raised when a provider or environment backend fails its contract."""


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

    def __post_init__(self) -> None:
        _require_text(self.name, "tool name")
        _require_text(self.description, "tool description", non_empty=False)
        _require_text(self.kind, "tool kind")
        object.__setattr__(
            self,
            "input_schema",
            _json_mapping(self.input_schema, "tool input_schema"),
        )


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    kind: str = "function"

    def __post_init__(self) -> None:
        _require_text(self.call_id, "tool call_id")
        _require_text(self.name, "tool name")
        _require_text(self.kind, "tool call kind")
        object.__setattr__(
            self,
            "arguments",
            _json_mapping(self.arguments, "tool arguments"),
        )


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    kind: str = "function"
    is_error: bool = False
    image_data_url: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.call_id, "tool result call_id")
        _require_text(self.name, "tool result name")
        _require_text(self.output, "tool result output", non_empty=False)
        _require_text(self.kind, "tool result kind")
        _require_bool(self.is_error, "tool result is_error")
        _require_image_url(self.image_data_url, "image_data_url")


@dataclass(frozen=True)
class ToolExecution:
    """An environment observation returned after one tool action."""

    output: str
    is_error: bool = False
    image_data_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.output, "tool execution output", non_empty=False)
        _require_bool(self.is_error, "tool execution is_error")
        _require_image_url(self.image_data_url, "image_data_url")
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "tool execution metadata"),
        )


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
            _require_int(getattr(self, name), name, minimum=0)
        if (
            self.cache_read_input_tokens + self.cache_write_input_tokens
            > self.input_tokens
        ):
            raise ValueError("cache input-token classes cannot exceed input_tokens")
        _require_finite_number(self.cost_usd, "cost_usd", minimum=0)
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
        for name in (
            "max_model_calls",
            "max_concurrency",
            "max_tool_calls",
            "max_tool_output_bytes",
        ):
            _require_positive_int(getattr(self, name), name)
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
        if not isinstance(self.role, str) or self.role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(f"unsupported message role {self.role!r}")
        _require_text(self.role, "message role")
        _require_text(self.content, "message content", non_empty=False)
        _require_tuple_of(self.tool_calls, ToolCall, "message tool_calls")
        _require_tuple_of(self.tool_results, ToolResult, "message tool_results")
        _require_image_url(self.image_data_url, "message image_data_url")
        object.__setattr__(
            self, "metadata", _json_mapping(self.metadata, "message metadata")
        )


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: Tuple[ToolCall, ...] = ()
    continuation: Any = field(default=None, repr=False, compare=False)
    resolved_model: Optional[str] = None
    retries: int = 0

    def __post_init__(self) -> None:
        _require_text(self.text, "model response text", non_empty=False)
        if not isinstance(self.usage, Usage):
            raise ValueError("model response usage must be Usage")
        _require_tuple_of(self.tool_calls, ToolCall, "model response tool_calls")
        if self.resolved_model is not None and (
            not isinstance(self.resolved_model, str)
            or not self.resolved_model.strip()
            or self.resolved_model != self.resolved_model.strip()
        ):
            raise ValueError(
                "model response resolved_model must be a non-empty string or None"
            )
        if self.resolved_model is not None:
            _require_utf8(self.resolved_model, "model response resolved_model")
        _require_int(self.retries, "model response retries", minimum=0)


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral request passed from :class:`BackendModel` to a codec."""

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

    def __post_init__(self) -> None:
        for name in ("agent_id", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model request {name} must be non-empty")
        if not isinstance(self.prompt, str) or not isinstance(self.system, str):
            raise ValueError("model request prompt and system must be strings")
        _require_utf8(self.agent_id, "model request agent_id")
        _require_utf8(self.role, "model request role")
        _require_utf8(self.prompt, "model request prompt")
        _require_utf8(self.system, "model request system")
        if self.max_output_tokens is not None and (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens < 1
        ):
            raise ValueError("model request max_output_tokens must be positive or None")
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "model request metadata"),
        )
        if not isinstance(self.input_images, tuple) or not all(
            isinstance(value, str) and value.startswith("data:image/")
            for value in self.input_images
        ):
            raise ValueError("model request input_images must be image data URLs")
        for image in self.input_images:
            _require_utf8(image, "model request input image")
        _require_tuple_of(self.tools, ToolDefinition, "model request tools", brief=True)
        _require_tuple_of(
            self.tool_results, ToolResult, "model request tool_results", brief=True
        )


@dataclass(frozen=True)
class TraceEvent:
    event: str
    elapsed_seconds: float
    agent_id: str = ""
    role: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.event, "trace event")
        _require_finite_number(self.elapsed_seconds, "trace elapsed_seconds", minimum=0)
        if not isinstance(self.agent_id, str) or not isinstance(self.role, str):
            raise ValueError("trace agent_id and role must be strings")
        _require_utf8(self.agent_id, "trace agent_id")
        _require_utf8(self.role, "trace role")
        object.__setattr__(
            self, "data", _json_mapping(self.data, "trace event data")
        )


@dataclass(frozen=True)
class AgentResult:
    answer: str
    messages: Tuple[Message, ...]
    steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.answer, "agent result answer", non_empty=False)
        _require_tuple_of(self.messages, Message, "agent result messages")
        _require_positive_int(self.steps, "agent result steps")
        object.__setattr__(
            self, "metadata", _json_mapping(self.metadata, "agent result metadata")
        )


__all__ = [
    "AgentResult",
    "BudgetExceeded",
    "BudgetLimits",
    "InfrastructureError",
    "InvalidAction",
    "Message",
    "MiniAgentError",
    "ModelRequest",
    "ModelResponse",
    "ProtocolError",
    "ToolCall",
    "ToolDefinition",
    "ToolExecution",
    "ToolResult",
    "TraceEvent",
    "Usage",
]
