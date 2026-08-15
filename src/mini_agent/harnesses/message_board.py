"""N peers coordinating through one shared board instead of mailboxes.

The contrast with `fixed-team` is the whole point of having both. A mailbox
makes every agent choose a recipient, so coordination cost scales with who
talks to whom and a peer only learns what someone decided to tell it. A board
is addressed to nobody: one post is visible to the whole team, reading does not
consume it, and an agent that joins the conversation late still sees everything
said before it looked.

What that trades away is targeting. There is no way to tell one teammate
something without telling all of them, so the board's failure mode is noise
rather than partition.
"""

from __future__ import annotations

from .base import Harness, Role

_BOARD = (
    " Coordinate only through the shared board: post what you have found or "
    "claimed, and read the board before starting work so you do not repeat a "
    "teammate. Everyone sees every post."
)

_LEAD = Role(
    actions=("post", "board"),
    prompt=" You lead a team working this same task; you alone submit." + _BOARD,
)
_PEER = Role(
    actions=("post", "board"),
    prompt=" You are one peer on a team working this same task; your lead"
    " submits." + _BOARD,
    idle=True,
)

HARNESS = Harness(
    name="message-board",
    roles={"lead": _LEAD, "peer": _PEER},
    lead="lead",
    seed_role="peer",
    sizes=(3, 5, 10),
)
