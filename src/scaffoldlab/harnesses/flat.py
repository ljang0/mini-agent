from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


class FlatParallelHarness(Harness):
    """Browser-Use-style fan-out over distinct, pre-specified tasks.

    There is deliberately no manager, communication, aggregation, or selector. This is a
    throughput primitive and should not be reported as best-of-N.
    """

    name = "flat_parallel"

    def __init__(self, *, max_agents: int = 16) -> None:
        self.max_agents = require_int_at_least(max_agents, "max_agents")

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        raw_tasks = task.metadata.get("parallel_tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ProtocolError(
                "flat_parallel requires Task.metadata['parallel_tasks']"
            )
        if len(raw_tasks) > self.max_agents:
            raise ProtocolError("flat_parallel exceeded max_agents")
        subtasks: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_tasks):
            subtask_id: str
            prompt: str
            if isinstance(raw, str):
                subtask_id, prompt = str(index), raw
            elif isinstance(raw, dict):
                raw_id, raw_prompt = raw.get("id"), raw.get("prompt")
                if not isinstance(raw_id, str) or not isinstance(raw_prompt, str):
                    raise ProtocolError(
                        "parallel task objects require string id and prompt"
                    )
                subtask_id, prompt = raw_id, raw_prompt
            else:
                raise ProtocolError("parallel tasks must be strings or objects")
            if subtask_id in seen:
                raise ProtocolError(f"duplicate parallel task id {subtask_id!r}")
            seen.add(subtask_id)
            subtasks.append((subtask_id, prompt))

        async def run_one(subtask_id: str, prompt: str) -> dict[str, Any]:
            response = await context.call(
                ModelRequest(
                    agent_id=f"/flat/{subtask_id}",
                    role="independent_agent",
                    prompt=prompt,
                    metadata={
                        "separate_state": True,
                        "subtask_id": subtask_id,
                    },
                )
            )
            return {"id": subtask_id, "result": response.text}

        results = await context.gather(
            *(run_one(subtask_id, prompt) for subtask_id, prompt in subtasks)
        )
        return json.dumps(results, ensure_ascii=False, sort_keys=True), {
            "agents_created": len(subtasks),
            "parallel_task_count": len(subtasks),
            "selector": False,
            "manager": False,
            "semantic_aggregation": False,
            "browser_runtime_integrated": False,
            "fidelity": "browser_use_parallel_pattern_only",
        }
