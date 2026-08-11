from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from ..runtime import RunContext, parse_json_object, require_string, require_string_list
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


class RecursiveDelegationHarness(Harness):
    """Inference topology used by RAO: one policy recursively delegates to copies."""

    name = "recursive_delegation"

    def __init__(self, *, max_children: int = 4, max_turns_per_agent: int = 8) -> None:
        self.max_children = require_int_at_least(max_children, "max_children")
        self.max_turns_per_agent = require_int_at_least(
            max_turns_per_agent, "max_turns_per_agent"
        )

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        agents_created = 0
        peak_depth = 0

        async def solve(
            instruction: str,
            *,
            agent_id: str,
            depth: int,
            ancestry: tuple[str, ...] = (),
        ) -> str:
            nonlocal agents_created, peak_depth
            agents_created += 1
            peak_depth = max(peak_depth, depth)
            child_results: list[dict[str, str]] = []
            history: list[dict[str, Any]] = []
            for turn in range(self.max_turns_per_agent):
                prompt = (
                    "Solve the instruction or recursively delegate independent subproblems "
                    "to copies of this same policy. Return one JSON action only:\n"
                    '- {"type":"delegate","tasks":["bounded subproblem", ...]}\n'
                    '- {"type":"answer","content":"final answer"}\n'
                    f"You are at recursion depth {depth}; the maximum is "
                    f"{context.ledger.limits.max_depth}. Use at most {self.max_children} "
                    "children in one delegation.\n\n"
                    f"INSTRUCTION:\n{instruction}\n\n"
                    f"SHARED TASK CONTEXT:\n{task.context}\n\n"
                    "ANCESTRAL INSTRUCTIONS:\n"
                    f"{json.dumps(ancestry, ensure_ascii=False)}\n\n"
                    f"CHILD RESULTS:\n{json.dumps(child_results, ensure_ascii=False)}\n\n"
                    f"PRIOR ACTIONS:\n{json.dumps(history, ensure_ascii=False)}"
                )
                response = await context.call(
                    ModelRequest(
                        agent_id=agent_id,
                        role="recursive_agent",
                        prompt=prompt,
                        metadata={"depth": depth, "turn": turn},
                    )
                )
                action = parse_json_object(response.text)
                history.append(action)
                if action.get("type") == "answer":
                    return require_string(action, "content")
                if action.get("type") != "delegate":
                    raise ProtocolError("recursive agent must delegate or answer")
                if depth >= context.ledger.limits.max_depth:
                    raise ProtocolError("recursive agent delegated beyond max_depth")
                child_tasks = require_string_list(action, "tasks")
                if not child_tasks or len(child_tasks) > self.max_children:
                    raise ProtocolError("recursive delegation exceeded child bounds")

                async def run_child(index: int, child_task: str) -> dict[str, str]:
                    answer = await solve(
                        child_task,
                        agent_id=f"{agent_id}/turn-{turn}/child-{index}",
                        depth=depth + 1,
                        ancestry=(*ancestry, instruction),
                    )
                    return {"task": child_task, "answer": answer}

                child_results.extend(
                    await context.gather(
                        *(
                            run_child(index, child_task)
                            for index, child_task in enumerate(child_tasks)
                        )
                    )
                )
            raise ProtocolError(f"{agent_id} exhausted its turn limit")

        answer = await solve(
            task.prompt,
            agent_id="/recursive/root",
            depth=0,
        )
        return answer, {
            "agents_created": agents_created,
            "peak_depth": peak_depth,
            "same_policy_all_depths": True,
            "rao_inference_reproduced": False,
            "rao_training_reproduced": False,
            "rah_reproduced": False,
            "fidelity": "generic_recursive_control",
        }


class ExternalContextJSONSearchHarness(Harness):
    """Bounded JSON-action external-context search and subcall ablation.

    This is motivated by the RLM question—whether keeping a large context outside the
    controller helps—but it is not an RLM: there is no code REPL, persistent namespace,
    or symbolic recursive RLM function.
    """

    name = "external_context_json_search"

    def __init__(
        self,
        *,
        max_turns: int = 16,
        max_slice_chars: int = 12_000,
        max_subcalls: int = 8,
    ) -> None:
        self.max_turns = require_int_at_least(max_turns, "max_turns")
        self.max_slice_chars = require_int_at_least(max_slice_chars, "max_slice_chars")
        self.max_subcalls = require_int_at_least(max_subcalls, "max_subcalls")

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        corpus = task.context
        if not corpus:
            raise ValueError("ExternalContextJSONSearchHarness requires Task.context")
        observations: list[dict[str, Any]] = []
        subcalls = 0
        for turn in range(self.max_turns):
            controller_prompt = (
                "The large context is external and is not included in your prompt. Explore "
                "it through bounded JSON actions, query selected slices, "
                "then answer. Return one JSON action only:\n"
                '- {"type":"inspect","start":0,"end":1000}\n'
                '- {"type":"search","pattern":"literal text",'
                '"max_hits":10}\n'
                '- {"type":"subcall","query":"question",'
                '"slices":[[0,1000],[2000,3000]]}\n'
                '- {"type":"answer","content":"final answer"}\n\n'
                f"QUESTION:\n{task.prompt}\n\n"
                f'EXTERNAL CONTEXT STATS:\n{{"characters": {len(corpus)}}}\n\n'
                f"ENVIRONMENT OBSERVATIONS:\n{json.dumps(observations, ensure_ascii=False)}"
            )
            response = await context.call(
                ModelRequest(
                    agent_id="/context-search/controller",
                    role="external_context_controller",
                    prompt=controller_prompt,
                    metadata={"turn": turn, "external_context": True},
                )
            )
            action = parse_json_object(response.text)
            action_type = action.get("type")
            if action_type == "answer":
                return require_string(action, "content"), {
                    "agents_created": subcalls + 1,
                    "subcalls": subcalls,
                    "environment_turns": turn + 1,
                    "unrestricted_code_repl": False,
                    "rlm_reproduction": False,
                    "fidelity": "non_rlm_external_context_ablation",
                }
            if action_type == "inspect":
                start, end = action.get("start"), action.get("end")
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                ):
                    raise ProtocolError("inspect offsets must be integers")
                if start < 0 or end <= start or end > len(corpus):
                    raise ProtocolError(
                        "inspect offsets are outside the external context"
                    )
                if end - start > self.max_slice_chars:
                    raise ProtocolError("inspect exceeded max_slice_chars")
                observations.append(
                    {
                        "action": "inspect",
                        "start": start,
                        "end": end,
                        "text": corpus[start:end],
                    }
                )
            elif action_type == "search":
                pattern = require_string(action, "pattern")
                if len(pattern) > 512:
                    raise ProtocolError("literal search pattern exceeds 512 characters")
                max_hits = action.get("max_hits", 10)
                if (
                    not isinstance(max_hits, int)
                    or isinstance(max_hits, bool)
                    or not 1 <= max_hits <= 50
                ):
                    raise ProtocolError("max_hits must be an integer from 1 to 50")
                hits: list[dict[str, Any]] = []
                cursor = 0
                while len(hits) < max_hits:
                    match_start = corpus.find(pattern, cursor)
                    if match_start < 0:
                        break
                    match_end = match_start + len(pattern)
                    start = max(0, match_start - 240)
                    end = min(len(corpus), match_end + 240)
                    hits.append({"start": start, "end": end, "text": corpus[start:end]})
                    cursor = match_end
                observations.append(
                    {"action": "search", "pattern": pattern, "hits": hits}
                )
            elif action_type == "subcall":
                if subcalls >= self.max_subcalls:
                    raise ProtocolError("external-context subcall budget exhausted")
                query = require_string(action, "query")
                slices = action.get("slices")
                if not isinstance(slices, list) or not slices:
                    raise ProtocolError("subcall requires one or more slices")
                selected: list[str] = []
                total_chars = 0
                for offsets in slices:
                    if (
                        not isinstance(offsets, list)
                        or len(offsets) != 2
                        or not all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in offsets
                        )
                    ):
                        raise ProtocolError("each subcall slice must be [start, end]")
                    start, end = offsets
                    if start < 0 or end <= start or end > len(corpus):
                        raise ProtocolError(
                            "subcall slice is outside the external context"
                        )
                    total_chars += end - start
                    if total_chars > self.max_slice_chars:
                        raise ProtocolError("subcall slices exceeded max_slice_chars")
                    selected.append(corpus[start:end])
                subcalls += 1
                sub_response = await context.call(
                    ModelRequest(
                        agent_id=f"/context-search/subcall-{subcalls}",
                        role="external_context_subcall",
                        prompt=(
                            f"Answer the bounded query from only these selected context "
                            f"slices.\n\nQUERY:\n{query}\n\nSLICES:\n"
                            f"{json.dumps(selected, ensure_ascii=False)}"
                        ),
                        metadata={"bounded_subcall": True},
                    )
                )
                observations.append(
                    {
                        "action": "subcall",
                        "query": query,
                        "answer": sub_response.text,
                    }
                )
            else:
                raise ProtocolError(f"unknown external-context action {action_type!r}")
        raise ProtocolError("external-context controller exhausted its turn limit")
