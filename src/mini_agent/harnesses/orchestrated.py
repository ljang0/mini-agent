"""One orchestrator with no task tools, delegating to blocking subagents.

`delegate` spawns one subagent and waits for it, returning its answer as the
tool result. The orchestrator's only capability is delegation, so the work can
only happen in a subagent.
"""

from __future__ import annotations

from .base import Harness, Role

_ORCHESTRATOR = Role(
    actions=("delegate",),
    domain_tools=False,
    prompt=(
        " You have no task tools. Use the agent tool to delegate a fully "
        "specified subtask; the call blocks and returns that subagent's answer."
    ),
)
_SUBAGENT = Role(
    prompt=" You were delegated one subtask. Complete it and report the result."
)

HARNESS = Harness(
    name="orchestrator",
    roles={"orchestrator": _ORCHESTRATOR, "subagent": _SUBAGENT},
    lead="orchestrator",
    child_role="subagent",
)
