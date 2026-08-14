"""A lead with task tools plus long-lived asynchronous subagents.

Spawning returns immediately. A subagent sees only its lead's instructions,
reports its answer as a message, then idles until woken with new instructions.
`release` lets the lead retire an idle subagent cleanly so its workspace can be
adopted -- `stop` would cancel it, and a cancelled agent exports no state.
"""

from __future__ import annotations

from .base import Harness, Role

_LEAD = Role(
    actions=("spawn", "send", "inbox", "stop", "adopt", "release"),
    prompt=(
        " Use the agent tool to spawn long-lived subagents that return "
        "immediately, message them new instructions, and release one when you "
        "want its work back."
    ),
)
_SUBAGENT = Role(
    actions=("inbox", "send"),
    prompt=(
        " You work for a lead who sees your replies as messages. After "
        "answering you stay available for further instructions."
    ),
    idle=True,
)

HARNESS = Harness(
    name="async-subagents",
    roles={"lead": _LEAD, "subagent": _SUBAGENT},
    lead="lead",
    child_role="subagent",
)
