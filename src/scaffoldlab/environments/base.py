from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..types import Task, ToolCall, ToolDefinition


@dataclass(frozen=True)
class ToolExecution:
    """One environment action result before provider-specific serialization."""

    output: str
    is_error: bool = False
    image_data_url: Optional[str] = None
    native_output: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolEnvironment(ABC):
    """A stateful environment session owned by one trial or agent."""

    @abstractmethod
    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        """Return the tools exposed to a model using ``provider_family``."""

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolExecution:
        """Execute one validated client-tool call."""

    async def close(self) -> None:
        """Release processes, browsers, VMs, and temporary resources."""

    async def summary(self) -> Mapping[str, Any]:
        """Return non-sensitive, JSON-serializable trial provenance."""

        return {}


class EnvironmentScope(ABC):
    """Owns every environment session created during one harness trial."""

    @abstractmethod
    async def get(self, agent_id: str) -> ToolEnvironment:
        """Return the stateful environment assigned to ``agent_id``."""

    @abstractmethod
    async def close(self) -> None:
        """Close all sessions, even after cancellation or failure."""

    async def summary(self) -> Mapping[str, Any]:
        return {}


class EnvironmentFactory(ABC):
    """Creates a fresh environment scope for every matrix trial."""

    @abstractmethod
    async def begin(self, task: Task) -> EnvironmentScope:
        """Prepare one isolated trial scope."""

    @abstractmethod
    def provenance(self) -> Mapping[str, Any]:
        """Return immutable environment configuration for the run manifest."""

    def validate_artifact_path(self, output_dir: Path, tasks: Sequence[Task]) -> None:
        """Reject run-artifact locations that can contaminate environments."""

    def validate_trial_plan(self, trial_count: int) -> None:
        """Reject matrix plans that cannot preserve trial isolation."""
