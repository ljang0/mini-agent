"""Multi-agent front door: mini-agent's CLI with delegation defaults on."""

from __future__ import annotations

import sys
from typing import Sequence

from mini_agent import Orchestrator
from mini_agent.orchestrator import CommunicationEnvironment

_MULTI_AGENT_COMMANDS = frozenset({"profile", "run", "eval"})

__version__ = "0.5.0"


def main(argv: Sequence[str] | None = None) -> int:
    from mini_agent.cli import main as base_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments
        and arguments[0] in _MULTI_AGENT_COMMANDS
        and "--multi-agent" not in arguments
    ):
        arguments.insert(1, "--multi-agent")
    return base_main(arguments)


__all__ = ["CommunicationEnvironment", "Orchestrator", "main"]
