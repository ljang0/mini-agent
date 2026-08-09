from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from ..runtime import RunContext, parse_json_object
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


class ParallelBestOfNHarness(Harness):
    """Independent candidates followed by a non-oracle model judge."""

    name = "parallel_best_of_n"

    def __init__(self, n: int = 3, *, candidate_system: str = "") -> None:
        self.n = require_int_at_least(n, "n", 2)
        if not isinstance(candidate_system, str):
            raise ValueError("candidate_system must be a string")
        self.candidate_system = candidate_system

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        candidate_prompt = task.prompt
        if task.context:
            candidate_prompt += f"\n\n<context>\n{task.context}\n</context>"

        async def generate(index: int) -> str:
            response = await context.call(
                ModelRequest(
                    agent_id=f"/candidate/{index}",
                    role="candidate",
                    prompt=candidate_prompt,
                    system=self.candidate_system,
                    metadata={"candidate_index": index},
                )
            )
            return response.text

        candidates = await context.gather(*(generate(index) for index in range(self.n)))
        judge_prompt = (
            "Select the strongest candidate answer to the task. Judge only the supplied "
            "answers; do not solve the task again and do not assume a hidden reference. "
            'Return JSON exactly as {"winner": <zero-based integer>, '
            '"reason": <short string>}.\n\n'
            f"TASK:\n{task.prompt}\n\n"
            f"TASK CONTEXT:\n{task.context}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, ensure_ascii=False)}"
        )
        judged = await context.call(
            ModelRequest(
                agent_id="/judge",
                role="judge",
                prompt=judge_prompt,
                system="You are a strict N-way answer selector.",
                metadata={"task_tools": False},
            )
        )
        decision = parse_json_object(judged.text)
        winner = decision.get("winner")
        if (
            not isinstance(winner, int)
            or isinstance(winner, bool)
            or not 0 <= winner < len(candidates)
        ):
            raise ProtocolError("judge winner must index one supplied candidate")
        return candidates[winner], {
            "agents_created": self.n + 1,
            "candidate_count": self.n,
            "winner": winner,
            "judge_reason": decision.get("reason", ""),
            "fidelity": "scaffoldlab_best_of_n_baseline",
        }
