"""The registry of selectable multi-agent topologies.

Adding a harness is one new module plus one name below -- no scheduler,
adapter, or CLI change. The list is explicit rather than discovered, because
the set of available harnesses is recorded in every manifest and must not
depend on what happens to be on disk.
"""

from __future__ import annotations

from importlib import import_module

from .base import ACTIONS, LEGACY_ACTIONS, Harness, Role

_MODULES = (
    "single",
    "recursive",
    "fixed_team",
    "orchestrated",
    "async_team",
    "message_board",
)
HARNESSES: dict[str, Harness] = {}

for _module in _MODULES:
    _harness = import_module(f"{__name__}.{_module}").HARNESS
    HARNESSES[_harness.name] = _harness


def load_harness(name: str) -> Harness:
    try:
        return HARNESSES[name]
    except KeyError:
        raise ValueError(
            f"unknown harness {name!r}; choose from {', '.join(harness_names())}"
        ) from None


def harness_names() -> tuple[str, ...]:
    return tuple(sorted(HARNESSES))


__all__ = [
    "ACTIONS",
    "HARNESSES",
    "LEGACY_ACTIONS",
    "Harness",
    "Role",
    "harness_names",
    "load_harness",
]
