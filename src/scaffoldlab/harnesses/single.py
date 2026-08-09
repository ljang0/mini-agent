from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class SingleAgentHarness(Harness):
    name = "single"

    def __init__(self, *, system: str = "") -> None:
        if not isinstance(system, str):
            raise ValueError("system must be a string")
        self.system = system

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt = f"{prompt}\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/root",
                role="solver",
                prompt=prompt,
                system=self.system,
            )
        )
        return response.text, {
            "agents_created": 1,
            "fidelity": "single_agent_baseline",
        }
