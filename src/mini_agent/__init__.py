"""A minimal agent loop for SWE, web, CUA, and communication experiments."""

from .agent import MiniAgent
from .models import BackendModel, Model, ScriptedModel
from .orchestrator import Orchestrator
from .runtime import RunContext
from .types import (
    AgentResult,
    BudgetExceeded,
    BudgetLimits,
    Message,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)

__all__ = [
    "AgentResult",
    "BudgetExceeded",
    "BudgetLimits",
    "BackendModel",
    "Message",
    "MiniAgent",
    "Model",
    "ModelResponse",
    "Orchestrator",
    "ProtocolError",
    "RunContext",
    "ScriptedModel",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Usage",
]

__version__ = "0.1.0"
