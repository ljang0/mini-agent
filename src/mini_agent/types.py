"""Small provider-neutral data contracts used by :mod:`mini_agent`."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Optional, Tuple


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
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    copied = _json_value(value, label)
    assert isinstance(copied, dict)
    return copied


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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("tool description must be a string")
        _require_utf8(self.name, "tool name")
        _require_utf8(self.description, "tool description")
        object.__setattr__(
            self,
            "input_schema",
            _json_mapping(self.input_schema, "tool input_schema"),
        )
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("tool kind must be a non-empty string")
        _require_utf8(self.kind, "tool kind")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    kind: str = "function"

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("tool call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        _require_utf8(self.call_id, "tool call_id")
        _require_utf8(self.name, "tool name")
        object.__setattr__(
            self,
            "arguments",
            _json_mapping(self.arguments, "tool arguments"),
        )
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("tool call kind must be a non-empty string")
        _require_utf8(self.kind, "tool call kind")


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    kind: str = "function"
    is_error: bool = False
    image_data_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("tool result call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool result name must be a non-empty string")
        if not isinstance(self.output, str):
            raise ValueError("tool result output must be a string")
        _require_utf8(self.call_id, "tool result call_id")
        _require_utf8(self.name, "tool result name")
        _require_utf8(self.output, "tool result output")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("tool result kind must be a non-empty string")
        _require_utf8(self.kind, "tool result kind")
        if not isinstance(self.is_error, bool):
            raise ValueError("tool result is_error must be a boolean")
        if self.image_data_url is not None and (
            not isinstance(self.image_data_url, str)
            or not self.image_data_url.startswith("data:image/")
        ):
            raise ValueError("image_data_url must be an image data URL")
        if self.image_data_url is not None:
            _require_utf8(self.image_data_url, "image_data_url")


@dataclass(frozen=True)
class ToolExecution:
    """An environment observation returned after one tool action."""

    output: str
    is_error: bool = False
    image_data_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise ValueError("tool execution output must be a string")
        _require_utf8(self.output, "tool execution output")
        if not isinstance(self.is_error, bool):
            raise ValueError("tool execution is_error must be a boolean")
        if self.image_data_url is not None and (
            not isinstance(self.image_data_url, str)
            or not self.image_data_url.startswith("data:image/")
        ):
            raise ValueError("image_data_url must be an image data URL")
        if self.image_data_url is not None:
            _require_utf8(self.image_data_url, "image_data_url")
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
        if not isinstance(self.role, str) or self.role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(f"unsupported message role {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")
        _require_utf8(self.role, "message role")
        _require_utf8(self.content, "message content")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise ValueError("message tool_calls must be a tuple of ToolCall values")
        if not isinstance(self.tool_results, tuple) or not all(
            isinstance(result, ToolResult) for result in self.tool_results
        ):
            raise ValueError(
                "message tool_results must be a tuple of ToolResult values"
            )
        if self.image_data_url is not None and (
            not isinstance(self.image_data_url, str)
            or not self.image_data_url.startswith("data:image/")
        ):
            raise ValueError("message image_data_url must be an image data URL")
        if self.image_data_url is not None:
            _require_utf8(self.image_data_url, "message image_data_url")
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
        if not isinstance(self.text, str):
            raise ValueError("model response text must be a string")
        _require_utf8(self.text, "model response text")
        if not isinstance(self.usage, Usage):
            raise ValueError("model response usage must be Usage")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise ValueError(
                "model response tool_calls must be a tuple of ToolCall values"
            )
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
        if (
            not isinstance(self.retries, int)
            or isinstance(self.retries, bool)
            or self.retries < 0
        ):
            raise ValueError(
                "model response retries must be a non-negative integer"
            )


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
        if not isinstance(self.tools, tuple) or not all(
            isinstance(tool, ToolDefinition) for tool in self.tools
        ):
            raise ValueError("model request tools must be ToolDefinition values")
        if not isinstance(self.tool_results, tuple) or not all(
            isinstance(result, ToolResult) for result in self.tool_results
        ):
            raise ValueError("model request tool_results must be ToolResult values")


@dataclass(frozen=True)
class TraceEvent:
    event: str
    elapsed_seconds: float
    agent_id: str = ""
    role: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event, str) or not self.event.strip():
            raise ValueError("trace event must be a non-empty string")
        if (
            not isinstance(self.elapsed_seconds, (int, float))
            or isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("trace elapsed_seconds must be finite and non-negative")
        if not isinstance(self.agent_id, str) or not isinstance(self.role, str):
            raise ValueError("trace agent_id and role must be strings")
        _require_utf8(self.event, "trace event")
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
        if not isinstance(self.answer, str):
            raise ValueError("agent result answer must be a string")
        _require_utf8(self.answer, "agent result answer")
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, Message) for message in self.messages
        ):
            raise ValueError("agent result messages must be a tuple of Message values")
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or self.steps < 1
        ):
            raise ValueError("agent result steps must be a positive integer")
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
