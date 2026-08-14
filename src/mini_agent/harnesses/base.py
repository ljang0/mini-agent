"""What a multi-agent topology declares, and nothing more.

A harness is data: which kinds of agent exist, what each may do, and how the
team is created. All scheduling stays in :mod:`mini_agent.orchestrator`, so a
new topology never touches the scheduler, the adapters, or the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..types import _require_bool, _require_str


# The full model-facing vocabulary. `delegate` and `release` exist because
# blocking delegation and clean hand-back cannot be expressed by the other six:
# two roles with the same actions must mean the same thing, or the recorded
# agent-spec capabilities describe an agent that never existed.
ACTIONS = (
    "spawn",
    "send",
    "inbox",
    "wait",
    "stop",
    "adopt",
    "delegate",
    "release",
)

# The pre-harness action set. `recursive` reproduces it exactly so every
# manifest fingerprint recorded before harnesses existed stays reproducible.
LEGACY_ACTIONS = ("spawn", "send", "inbox", "wait", "stop", "adopt")


@dataclass(frozen=True)
class Role:
    """One kind of agent: what it can hold, do, and be told."""

    actions: tuple[str, ...] = ()
    domain_tools: bool = True
    prompt: str = ""
    idle: bool = False

    def __post_init__(self) -> None:
        for action in self.actions:
            if action not in ACTIONS:
                raise ValueError(f"unknown agent action {action!r}")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("agent actions must be unique")
        _require_bool(self.domain_tools, "role domain_tools")
        _require_bool(self.idle, "role idle")
        _require_str(self.prompt, "role prompt", non_empty=False)
        if not self.domain_tools and not self.actions:
            raise ValueError("a role with no domain tools needs at least one action")

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The sorted form recorded in an agent spec."""

        return tuple(sorted(self.actions))


@dataclass(frozen=True)
class Harness:
    """A named topology: its roles, its lead, and how its team is formed."""

    name: str
    roles: Mapping[str, Role]
    lead: str
    seed_role: str | None = None
    child_role: str | None = None
    sizes: tuple[int, ...] = ()
    legacy: bool = False

    def __post_init__(self) -> None:
        _require_str(self.name, "harness name")
        for role in (self.lead, self.seed_role, self.child_role):
            if role is not None and role not in self.roles:
                raise ValueError(f"harness {self.name!r} has no role {role!r}")
        if self.sizes and self.seed_role is None:
            raise ValueError("a sized harness must declare the role it seeds")

    def role_name_of(self, agent_id: str, *, root_id: str) -> str:
        """Resolve an agent's role from its id alone.

        Ancestry is already encoded in the id (`/root`, `/root/1`), so no
        separate bookkeeping is needed to know what an agent is.
        """

        if agent_id == root_id:
            return self.lead
        return self.child_role or self.seed_role or self.lead

    def role_of(self, agent_id: str, *, root_id: str) -> Role:
        return self.roles[self.role_name_of(agent_id, root_id=root_id)]

    def team_size(self, requested: int | None) -> int:
        """Validate a requested team size against what this harness accepts."""

        if not self.sizes:
            if requested is not None:
                raise ValueError(f"--team-size does not apply to --harness {self.name}")
            return 1
        if requested is None:
            return self.sizes[0]
        if requested not in self.sizes:
            raise ValueError(
                f"--harness {self.name} accepts --team-size "
                f"{', '.join(str(size) for size in self.sizes)}"
            )
        return requested

    def seeds(self, *, size: int, task: str) -> tuple[tuple[str, str], ...]:
        """Agents to start beside the lead, as (id suffix, task) pairs.

        Seeded peers see the task verbatim; spawned children see only what
        their parent tells them, which is what separates a peer team from
        delegation.
        """

        if self.seed_role is None or not self.sizes:
            return ()
        return tuple((f"peer-{number}", task) for number in range(2, size + 1))

    def capacity(self, size: int) -> int:
        """Live agents this harness needs before it stalls or fails."""

        return size if self.sizes else 1
