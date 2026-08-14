"""One agent, no communication tool. The baseline every harness is measured against."""

from __future__ import annotations

from .base import Harness, Role

HARNESS = Harness(
    name="single",
    roles={"solo": Role()},
    lead="solo",
    legacy=True,
)
