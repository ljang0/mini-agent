from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from scaffoldlab.types import (
    BudgetExceeded,
    BudgetLimits,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)


@dataclass(frozen=True)
class Message:
    """One provider-neutral item in the agent's linear history."""

    role: str
    content: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_results: Tuple[ToolResult, ...] = ()
    image_data_url: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    "ModelResponse",
    "ProtocolError",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Usage",
]
