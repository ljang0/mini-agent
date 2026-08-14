"""N peer agents sharing one task, with a designated lead.

Every agent has the same tools and the same task text; the lead alone submits.
Peers idle rather than terminate after answering: a terminal agent cannot be
messaged, so a team whose members finished would silently shrink to one.
"""

from __future__ import annotations

from .base import Harness, Role

_TEAM = " Send findings to a teammate and block on your inbox for theirs."

_LEAD = Role(
    actions=("inbox", "send"),
    prompt=" You lead a team of peers working this same task; you alone submit."
    + _TEAM,
)
_PEER = Role(
    actions=("inbox", "send"),
    prompt=" You are one peer on a team working this same task; your lead submits."
    + _TEAM,
    idle=True,
)

HARNESS = Harness(
    name="fixed-team",
    roles={"lead": _LEAD, "peer": _PEER},
    lead="lead",
    seed_role="peer",
    sizes=(3, 5, 10),
)
