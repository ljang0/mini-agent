from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable
from typing import Any, Callable, Mapping, Tuple

from ..runtime import RunContext, parse_json_object, require_string
from ..types import ModelRequest, ProtocolError, ScaffoldLabError, Task
from .base import Harness, require_int_at_least
from .repl import RestrictedPersistentPythonREPL, execute_repl_tool


class _TrackedSubagentCall:
    """Single-use awaitable used to reject forgotten launch_subagent calls."""

    def __init__(
        self,
        child_id: str,
        operation: Callable[[], Awaitable[str]],
    ) -> None:
        self.child_id = child_id
        self._operation = operation
        self.awaited = False

    def __await__(self):
        if self.awaited:
            raise ProtocolError(f"subagent call {self.child_id!r} awaited twice")
        self.awaited = True
        return self._operation().__await__()

    def __repr__(self) -> str:
        state = "running-or-complete" if self.awaited else "pending"
        return f"<subagent-call {self.child_id} {state}>"


class _RestrictedAsyncio:
    """Only the gather primitive needed by Platoon's parallel CodeAct pattern."""

    def __init__(self, context: RunContext) -> None:
        self._context = context

    async def gather(self, *awaitables: Any) -> list[Any]:
        async def await_one(value: Any) -> Any:
            if not isinstance(value, Awaitable):
                raise ProtocolError("asyncio.gather arguments must be awaitable")
            return await value

        return await self._context.gather(*(await_one(value) for value in awaitables))


class PlatoonRecursiveInferenceHarness(Harness):
    """Platoon/RAO-inspired recursive CodeAct inference without RAO training."""

    name = "platoon_recursive_inference"

    def __init__(
        self,
        *,
        max_steps_per_agent: int = 15,
        max_total_subagents: int = 32,
        max_children_per_action: int = 4,
        max_concurrent_subagents: int = 4,
        max_observations: int = 8,
        max_source_chars: int = 12_000,
        max_repl_output_chars: int = 20_000,
    ) -> None:
        self.max_steps_per_agent = require_int_at_least(
            max_steps_per_agent, "max_steps_per_agent"
        )
        self.max_total_subagents = require_int_at_least(
            max_total_subagents, "max_total_subagents"
        )
        self.max_children_per_action = require_int_at_least(
            max_children_per_action, "max_children_per_action"
        )
        self.max_concurrent_subagents = require_int_at_least(
            max_concurrent_subagents, "max_concurrent_subagents"
        )
        self.max_observations = require_int_at_least(
            max_observations, "max_observations"
        )
        self.max_source_chars = require_int_at_least(
            max_source_chars, "max_source_chars"
        )
        self.max_repl_output_chars = require_int_at_least(
            max_repl_output_chars, "max_repl_output_chars"
        )

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        agents_created = 0
        subagents_reserved = 0
        subagents_completed = 0
        peak_depth = 0
        model_semaphore = asyncio.Semaphore(
            min(
                self.max_concurrent_subagents,
                context.ledger.limits.max_concurrency,
            )
        )

        async def call_model(request: ModelRequest):
            async with model_semaphore:
                return await context.call(request)

        async def solve(
            instruction: str,
            *,
            agent_id: str,
            depth: int,
            max_steps: int,
            ancestry: tuple[str, ...],
            task_misc: Mapping[str, Any],
        ) -> str:
            nonlocal agents_created, peak_depth, subagents_reserved
            nonlocal subagents_completed
            agents_created += 1
            peak_depth = max(peak_depth, depth)
            local_child_serial = 0
            action_launches = 0
            action_calls: list[_TrackedSubagentCall] = []

            async def run_child(
                goal: str,
                *,
                child_id: str,
                child_steps: int,
                child_misc: Mapping[str, Any],
            ) -> str:
                nonlocal subagents_completed
                trace_data: dict[str, Any] = {
                    "parent_agent_id": agent_id,
                    "depth": depth + 1,
                    "max_steps": child_steps,
                    "goal_chars": len(goal),
                    "goal_sha256": hashlib.sha256(goal.encode("utf-8")).hexdigest(),
                }
                if context.capture_content:
                    trace_data["goal"] = goal
                    trace_data["task_misc"] = dict(child_misc)
                await context.trace.emit(
                    "platoon_subagent_started",
                    agent_id=child_id,
                    role="recursive_codeact_agent",
                    data=trace_data,
                )
                try:
                    result = await solve(
                        goal,
                        agent_id=child_id,
                        depth=depth + 1,
                        max_steps=child_steps,
                        ancestry=(*ancestry, instruction),
                        task_misc=child_misc,
                    )
                except BaseException as exc:
                    event = (
                        "platoon_subagent_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "platoon_subagent_failed"
                    )
                    await context.trace.emit(
                        event,
                        agent_id=child_id,
                        role="recursive_codeact_agent",
                        data={
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    if isinstance(exc, (asyncio.CancelledError, ScaffoldLabError)):
                        raise
                    raise RuntimeError(
                        f"Platoon subagent failed: {type(exc).__name__}: {exc}"
                    ) from exc
                subagents_completed += 1
                completed_data: dict[str, Any] = {
                    "answer_chars": len(result),
                    "answer_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
                }
                if context.capture_content:
                    completed_data["answer"] = result
                await context.trace.emit(
                    "platoon_subagent_completed",
                    agent_id=child_id,
                    role="recursive_codeact_agent",
                    data=completed_data,
                )
                return result

            def launch_subagent(
                goal: str,
                max_steps: int = self.max_steps_per_agent,
                task_misc: Mapping[str, Any] | None = None,
                verbose: bool = True,
            ) -> _TrackedSubagentCall:
                nonlocal local_child_serial, action_launches, subagents_reserved
                if not isinstance(goal, str) or not goal.strip():
                    raise ProtocolError(
                        "launch_subagent goal must be a non-empty string"
                    )
                if (
                    not isinstance(max_steps, int)
                    or isinstance(max_steps, bool)
                    or not 1 <= max_steps <= self.max_steps_per_agent
                ):
                    raise ProtocolError(
                        "launch_subagent max_steps must be between 1 and "
                        f"{self.max_steps_per_agent}"
                    )
                if task_misc is not None and not isinstance(task_misc, Mapping):
                    raise ProtocolError("launch_subagent task_misc must be an object")
                if not isinstance(verbose, bool):
                    raise ProtocolError("launch_subagent verbose must be a boolean")
                if depth >= context.ledger.limits.max_depth:
                    raise ProtocolError("launch_subagent exceeded max_depth")
                if action_launches >= self.max_children_per_action:
                    raise ProtocolError(
                        "launch_subagent exceeded the per-action child limit"
                    )
                if subagents_reserved >= self.max_total_subagents:
                    raise ProtocolError(
                        "launch_subagent exceeded the total subagent limit"
                    )
                action_launches += 1
                subagents_reserved += 1
                local_child_serial += 1
                child_id = f"{agent_id}/child-{local_child_serial}"
                child_misc = dict(task_misc or {})
                call = _TrackedSubagentCall(
                    child_id,
                    lambda: run_child(
                        goal,
                        child_id=child_id,
                        child_steps=max_steps,
                        child_misc=child_misc,
                    ),
                )
                action_calls.append(call)
                return call

            restricted_asyncio = _RestrictedAsyncio(context)
            repl = RestrictedPersistentPythonREPL(
                {
                    "context": task.context,
                    "task_misc": dict(task_misc),
                    "launch_subagent": launch_subagent,
                    "asyncio": restricted_asyncio,
                },
                max_source_chars=self.max_source_chars,
                max_output_chars=self.max_repl_output_chars,
            )
            observations: list[dict[str, Any]] = []
            for step in range(max_steps):
                action_launches = 0
                action_calls.clear()
                prompt = (
                    "You are a recursive CodeAct inference agent. This is an "
                    "inference-only Platoon-style topology; no RAO policy training or "
                    "trained checkpoint is being reproduced. Work through the persistent "
                    "restricted Python REPL. Task context is available as `context`. "
                    "Delegate with `await launch_subagent(goal, max_steps=...)`. Run "
                    "independent children in parallel with "
                    "`await asyncio.gather(launch_subagent(...), ...)`. Every launched "
                    "call must be awaited in the same action. Return exactly one JSON "
                    "action:\n"
                    '- {"type":"execute","code":"Python source"}\n'
                    '- {"type":"answer","content":"final answer"}\n'
                    "Imports, filesystem/network access, private attributes, dynamic "
                    "evaluation, function/class definitions, and unbounded while loops "
                    "are unavailable. This capability restriction is not an operating-"
                    "system sandbox.\n\n"
                    f"INSTRUCTION:\n{instruction}\n\n"
                    f"ANCESTRY:\n{json.dumps(ancestry, ensure_ascii=False)}\n\n"
                    f"DEPTH: {depth}/{context.ledger.limits.max_depth}\n"
                    f"STEP: {step + 1}/{max_steps}\n"
                    f"TASK MISC:\n{json.dumps(task_misc, ensure_ascii=False, default=str)}\n\n"
                    "RECENT REPL OBSERVATIONS:\n"
                    f"{json.dumps(observations[-self.max_observations :], ensure_ascii=False)}"
                )
                response = await call_model(
                    ModelRequest(
                        agent_id=agent_id,
                        role="recursive_codeact_agent",
                        prompt=prompt,
                        metadata={
                            "depth": depth,
                            "step": step,
                            "max_steps": max_steps,
                            "inference_only": True,
                            "rao_training_reproduced": False,
                            "domain_tools": False,
                        },
                    )
                )
                action = parse_json_object(response.text)
                action_type = action.get("type")
                if action_type == "answer":
                    return require_string(action, "content")
                if action_type != "execute":
                    raise ProtocolError(
                        "recursive CodeAct agent must execute or answer"
                    )
                output = await execute_repl_tool(
                    repl,
                    require_string(action, "code"),
                    context,
                    agent_id=agent_id,
                    role="recursive_codeact_agent",
                    tool_name="platoon_restricted_python",
                )
                forgotten = [call.child_id for call in action_calls if not call.awaited]
                if forgotten:
                    raise ProtocolError(
                        "launch_subagent calls must be awaited in the same action: "
                        + ", ".join(forgotten)
                    )
                observations.append(
                    {
                        "step": step + 1,
                        "output": output,
                        "variables": list(repl.user_variables),
                    }
                )
            raise ProtocolError(f"{agent_id} exhausted its CodeAct step limit")

        answer = await solve(
            task.prompt,
            agent_id="/platoon/root",
            depth=0,
            max_steps=self.max_steps_per_agent,
            ancestry=(),
            task_misc={},
        )
        return answer, {
            "agents_created": agents_created,
            "subagents_reserved": subagents_reserved,
            "subagents_completed": subagents_completed,
            "peak_depth": peak_depth,
            "same_policy_all_depths": True,
            "persistent_python_namespace": True,
            "recoverable_repl_errors": True,
            "shared_budget_ledger": True,
            "step_budget_semantics": "per_agent_steps_plus_shared_global_ledger",
            "inference_only": True,
            "platoon_inference_topology": True,
            "rao_inference_reproduced": False,
            "rao_training_reproduced": False,
            "fidelity": "clean_room_platoon_inspired_codeact_inference",
        }
