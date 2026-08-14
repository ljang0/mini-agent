"""One way to run a team, whatever its shape.

Every benchmark adapter used to carry its own "if multi-agent build an
orchestrator, else build one agent" branch and then reach into the root
record for a domain artifact. The branch is the same everywhere; only the
artifact differs, so that is the only part an adapter still writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .harnesses import Harness, load_harness
from .orchestrator import AgentBuilder, AgentRecord, EnvironmentFactory, Orchestrator
from .runtime import RunContext
from .types import AgentResult, BudgetLimits


def selected_harness(harness: str, multi_agent: bool) -> str:
    """Resolve the harness a caller meant.

    ``multi_agent`` predates harness selection, so it keeps working: it names
    the free-form mesh it always ran, and an explicit ``harness`` wins.
    """

    if harness != "single":
        return harness
    return "recursive" if multi_agent else "single"


@dataclass(frozen=True)
class TeamRun:
    """A finished team, described without reference to any domain."""

    harness: Harness
    root_id: str
    records: Mapping[str, AgentRecord]
    size: int = 1
    result: AgentResult | None = None
    error: BaseException | None = None

    def require(self) -> AgentResult:
        """Return the lead's result, or re-raise what stopped it."""

        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result

    @property
    def root(self) -> AgentRecord:
        return self.records[self.root_id]

    @property
    def state(self) -> Any:
        """The lead's exported environment state, whatever the domain calls it."""

        return self.root.state

    def statuses(self) -> dict[str, str]:
        return {agent_id: record.status for agent_id, record in self.records.items()}

    def bases(self) -> dict[str, Any]:
        """Each agent's domain environment, behind its communication wrapper."""

        return {
            agent_id: record.environment.base
            for agent_id, record in self.records.items()
            if record.environment is not None
        }

    def metadata(self) -> dict[str, Any]:
        """The per-agent block every adapter records.

        Harnesses that predate this layer keep emitting exactly what they
        emitted before, so their artifacts stay comparable with older runs.
        """

        value: dict[str, Any] = {
            "mode": "single" if self.harness.name == "single" else "multi",
            "agents": self.statuses(),
        }
        if not self.harness.legacy:
            value["harness"] = self.harness.name
            value["team_size"] = self.size
        return value


async def run_team(
    task: str,
    *,
    harness: str | Harness,
    team_size: int | None = None,
    agent_builder: AgentBuilder,
    environment_factory: EnvironmentFactory,
    context: RunContext,
    root_id: str,
    max_active_agents: int,
    max_total_agents: int,
    per_agent_limits: BudgetLimits | None = None,
    tolerate_failure: bool = False,
) -> TeamRun:
    """Run one harness to the lead's answer and hand back the records.

    ``tolerate_failure`` captures the failure instead of raising, for domains
    that still have to score or clean up a live machine afterwards.
    """

    selected = load_harness(harness) if isinstance(harness, str) else harness
    size = selected.team_size(team_size)
    orchestrator = Orchestrator(
        agent_builder=agent_builder,
        environment_factory=environment_factory,
        context=context,
        max_active_agents=max(max_active_agents, selected.capacity(size)),
        max_total_agents=max(max_total_agents, selected.capacity(size)),
        per_agent_limits=per_agent_limits,
        root_id=root_id,
        harness=selected,
    )
    result: AgentResult | None = None
    error: BaseException | None = None
    try:
        result = await orchestrator.run(
            task, seeds=selected.seeds(size=size, task=task)
        )
    except BaseException as exc:
        if not tolerate_failure:
            raise
        error = exc
    return TeamRun(
        harness=selected,
        root_id=root_id,
        records=orchestrator.records,
        size=size,
        result=result,
        error=error,
    )
