from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from ..runtime import RunContext, parse_json_object, require_string
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


class BlockingOrchestratorHarness(Harness):
    """One-round Fable-inspired control simulation without executable task tools."""

    name = "blocking_orchestrator"

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = require_int_at_least(max_workers, "max_workers")

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        full_task = task.prompt
        if task.context:
            full_task += f"\n\n<context>\n{task.context}\n</context>"
        manager_prompt = (
            "Decompose the task into independent subtasks for fresh worker agents. "
            f"Use at most {self.max_workers}. The manager cannot perform task work. "
            'Return JSON exactly as {"subtasks": [{"id": "short-id", '
            '"instruction": "bounded worker instruction"}, ...]}.\n\n'
            f"TASK:\n{full_task}"
        )
        plan_response = await context.call(
            ModelRequest(
                agent_id="/orchestrator",
                role="plan",
                prompt=manager_prompt,
                system="You are a blocking orchestrator with no task tools.",
                metadata={"task_tools": False},
            )
        )
        plan = parse_json_object(plan_response.text)
        raw_subtasks = plan.get("subtasks")
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ProtocolError("orchestrator must emit at least one subtask")
        if len(raw_subtasks) > self.max_workers:
            raise ProtocolError("orchestrator exceeded max_workers")
        seen: set[str] = set()
        subtasks: list[tuple[str, str]] = []
        for raw in raw_subtasks:
            if not isinstance(raw, dict):
                raise ProtocolError("each subtask must be an object")
            subtask_id = require_string(raw, "id")
            instruction = require_string(raw, "instruction")
            if subtask_id in seen:
                raise ProtocolError(f"duplicate subtask id {subtask_id!r}")
            seen.add(subtask_id)
            subtasks.append((subtask_id, instruction))

        async def run_worker(subtask_id: str, instruction: str) -> dict[str, str]:
            prompt = instruction
            if task.context:
                prompt += f"\n\n<context>\n{task.context}\n</context>"
            result = await context.call(
                ModelRequest(
                    agent_id=f"/worker/{subtask_id}",
                    role="worker",
                    prompt=prompt,
                    metadata={"fresh_context": True, "subtask_id": subtask_id},
                )
            )
            return {"id": subtask_id, "result": result.text}

        reports = await context.gather(
            *(
                run_worker(subtask_id, instruction)
                for subtask_id, instruction in subtasks
            )
        )
        synthesis_prompt = (
            "Synthesize the fresh workers' reports into the final answer. Reconcile "
            "conflicts and do not invent facts absent from the reports.\n\n"
            f"TASK:\n{full_task}\n\nREPORTS:\n"
            f"{json.dumps(reports, ensure_ascii=False)}"
        )
        final_response = await context.call(
            ModelRequest(
                agent_id="/orchestrator",
                role="synthesize",
                prompt=synthesis_prompt,
                system="You are a blocking orchestrator with no task tools.",
                metadata={"task_tools": False},
            )
        )
        return final_response.text, {
            "agents_created": len(subtasks) + 1,
            "worker_count": len(subtasks),
            "barrier_rounds": 1,
            "anthropic_public_card_reproduction": False,
            "fidelity": "anthropic_card_topology_simulation",
        }
