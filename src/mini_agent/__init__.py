"""A minimal agent loop for SWE, web, computer use, and communication."""

from .agent import MiniAgent
from .models import BackendModel, Model, ScriptedModel, build_model
from .orchestrator import Orchestrator
from .execution import RunContext
from .specs import AgentSpecV1, TranslationLoss, TranslationReport
from .types import (
    AgentResult,
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
    Usage,
)

__all__ = [
    "AgentResult",
    "AgentSpecV1",
    "BudgetExceeded",
    "BudgetLimits",
    "BackendModel",
    "InfrastructureError",
    "InvalidAction",
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
    "ToolExecution",
    "ToolResult",
    "TranslationLoss",
    "TranslationReport",
    "Usage",
    "build_model",
]

__version__ = "0.5.0"
