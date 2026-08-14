"""The free-form mesh that predates named harnesses.

Kept so `--multi-agent` and every manifest fingerprint recorded before this
layer existed stay reproducible. It is not one of the design doc's four
topologies: every agent holds every action and may delegate to any depth.
"""

from __future__ import annotations

from .base import LEGACY_ACTIONS, Harness, Role

_SOLVER = Role(
    actions=LEGACY_ACTIONS,
    prompt=(
        " You may also use the agent tool to delegate bounded subtasks, exchange "
        "messages, block for inbox delivery, wait for or stop descendant work, or "
        "explicitly adopt supported child state."
    ),
)

HARNESS = Harness(
    name="recursive",
    roles={"solver": _SOLVER},
    lead="solver",
    child_role="solver",
    legacy=True,
)
