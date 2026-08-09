from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Awaitable, Callable, Mapping, Sequence, Tuple

from ..runtime import RunContext, parse_json_object, require_string
from ..types import ModelRequest, ProtocolError, ScaffoldLabError, Task
from .base import Harness, require_int_at_least
from .repl import RestrictedPersistentPythonREPL, execute_repl_tool


def _query_list(value: Any, *, maximum: int) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProtocolError("batched query requires a non-empty list or tuple")
    if len(value) > maximum:
        raise ProtocolError(f"batched query exceeds the {maximum}-query limit")
    queries: list[str] = []
    for query in value:
        if not isinstance(query, str) or not query.strip():
            raise ProtocolError("every batched query must be a non-empty string")
        queries.append(query)
    return queries


class RLMREPLHarness(Harness):
    """External-context recursive language model with a restricted Python REPL."""

    name = "rlm_repl"

    def __init__(
        self,
        *,
        max_iterations: int = 16,
        max_subcalls: int = 32,
        max_batch_size: int = 8,
        max_concurrent_subcalls: int = 4,
        max_observations: int = 8,
        max_source_chars: int = 12_000,
        max_repl_output_chars: int = 20_000,
    ) -> None:
        self.max_iterations = require_int_at_least(max_iterations, "max_iterations")
        self.max_subcalls = require_int_at_least(max_subcalls, "max_subcalls")
        self.max_batch_size = require_int_at_least(max_batch_size, "max_batch_size")
        self.max_concurrent_subcalls = require_int_at_least(
            max_concurrent_subcalls, "max_concurrent_subcalls"
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
        if not task.context:
            raise ValueError("RLMREPLHarness requires non-empty Task.context")

        subcall_lock = asyncio.Lock()
        model_semaphore = asyncio.Semaphore(
            min(
                self.max_concurrent_subcalls,
                context.ledger.limits.max_concurrency,
            )
        )
        subcalls = 0
        bare_subcalls = 0
        recursive_subcalls = 0
        depth_fallbacks = 0
        recursive_agents_created = 0
        peak_depth = 0

        async def reserve_subcalls(count: int) -> None:
            nonlocal subcalls
            async with subcall_lock:
                if subcalls + count > self.max_subcalls:
                    raise ProtocolError(
                        f"RLM subcall limit exceeded ({self.max_subcalls})"
                    )
                subcalls += count

        async def call_model(request: ModelRequest):
            async with model_semaphore:
                return await context.call(request)

        async def traced_helper(
            *,
            kind: str,
            helper_agent_id: str,
            parent_agent_id: str,
            depth: int,
            query: str,
            fallback: bool,
            operation: Callable[[], Awaitable[str]],
        ) -> str:
            data: dict[str, Any] = {
                "kind": kind,
                "parent_agent_id": parent_agent_id,
                "depth": depth,
                "fallback_to_llm": fallback,
                "query_chars": len(query),
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            }
            if context.capture_content:
                data["query"] = query
            await context.trace.emit(
                "rlm_subcall_started",
                agent_id=helper_agent_id,
                role=kind,
                data=data,
            )
            try:
                answer = await operation()
            except BaseException as exc:
                event = (
                    "rlm_subcall_cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "rlm_subcall_failed"
                )
                await context.trace.emit(
                    event,
                    agent_id=helper_agent_id,
                    role=kind,
                    data={
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "fallback_to_llm": fallback,
                    },
                )
                if isinstance(exc, (asyncio.CancelledError, ScaffoldLabError)):
                    raise
                raise RuntimeError(
                    f"RLM helper failed: {type(exc).__name__}: {exc}"
                ) from exc
            completed: dict[str, Any] = {
                "answer_chars": len(answer),
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "fallback_to_llm": fallback,
            }
            if context.capture_content:
                completed["answer"] = answer
            await context.trace.emit(
                "rlm_subcall_completed",
                agent_id=helper_agent_id,
                role=kind,
                data=completed,
            )
            return answer

        async def bare_query(
            query: str,
            *,
            helper_agent_id: str,
            parent_agent_id: str,
            depth: int,
            kind: str = "llm_query",
            fallback: bool = False,
        ) -> str:
            nonlocal bare_subcalls
            bare_subcalls += 1

            async def operation() -> str:
                response = await call_model(
                    ModelRequest(
                        agent_id=helper_agent_id,
                        role="rlm_bare_subcall",
                        prompt=query,
                        metadata={
                            "depth": depth,
                            "rlm_helper": kind,
                            "depth_fallback": fallback,
                            "domain_tools": False,
                        },
                    )
                )
                return response.text

            return await traced_helper(
                kind=kind,
                helper_agent_id=helper_agent_id,
                parent_agent_id=parent_agent_id,
                depth=depth,
                query=query,
                fallback=fallback,
                operation=operation,
            )

        async def solve(
            instruction: str,
            *,
            agent_id: str,
            depth: int,
            root_instruction: str,
        ) -> str:
            nonlocal recursive_agents_created, peak_depth
            nonlocal recursive_subcalls, depth_fallbacks
            recursive_agents_created += 1
            peak_depth = max(peak_depth, depth)
            helper_serial = 0

            def next_helper_id(label: str) -> str:
                nonlocal helper_serial
                helper_serial += 1
                return f"{agent_id}/{label}-{helper_serial}"

            async def llm_query(query: str) -> str:
                if not isinstance(query, str) or not query.strip():
                    raise ProtocolError("llm_query requires a non-empty string")
                await reserve_subcalls(1)
                return await bare_query(
                    query,
                    helper_agent_id=next_helper_id("llm"),
                    parent_agent_id=agent_id,
                    depth=depth,
                )

            async def llm_query_batched(queries: Sequence[str]) -> list[str]:
                validated = _query_list(queries, maximum=self.max_batch_size)
                await reserve_subcalls(len(validated))
                helper_ids = [next_helper_id("llm") for _ in validated]
                return await context.gather(
                    *(
                        bare_query(
                            query,
                            helper_agent_id=helper_id,
                            parent_agent_id=agent_id,
                            depth=depth,
                            kind="llm_query_batched",
                        )
                        for query, helper_id in zip(validated, helper_ids)
                    )
                )

            async def run_recursive_query(query: str, helper_id: str) -> str:
                nonlocal recursive_subcalls, depth_fallbacks
                recursive_subcalls += 1
                if depth >= context.ledger.limits.max_depth:
                    depth_fallbacks += 1
                    return await bare_query(
                        query,
                        helper_agent_id=helper_id,
                        parent_agent_id=agent_id,
                        depth=depth,
                        kind="rlm_query",
                        fallback=True,
                    )

                async def operation() -> str:
                    return await solve(
                        query,
                        agent_id=helper_id,
                        depth=depth + 1,
                        root_instruction=root_instruction,
                    )

                return await traced_helper(
                    kind="rlm_query",
                    helper_agent_id=helper_id,
                    parent_agent_id=agent_id,
                    depth=depth,
                    query=query,
                    fallback=False,
                    operation=operation,
                )

            async def rlm_query(query: str) -> str:
                if not isinstance(query, str) or not query.strip():
                    raise ProtocolError("rlm_query requires a non-empty string")
                await reserve_subcalls(1)
                return await run_recursive_query(query, next_helper_id("rlm"))

            async def rlm_query_batched(queries: Sequence[str]) -> list[str]:
                validated = _query_list(queries, maximum=self.max_batch_size)
                await reserve_subcalls(len(validated))
                helper_ids = [next_helper_id("rlm") for _ in validated]
                return await context.gather(
                    *(
                        run_recursive_query(query, helper_id)
                        for query, helper_id in zip(validated, helper_ids)
                    )
                )

            repl = RestrictedPersistentPythonREPL(
                {
                    "context": task.context,
                    "llm_query": llm_query,
                    "llm_query_batched": llm_query_batched,
                    "rlm_query": rlm_query,
                    "rlm_query_batched": rlm_query_batched,
                },
                max_source_chars=self.max_source_chars,
                max_output_chars=self.max_repl_output_chars,
            )
            observations: list[dict[str, Any]] = []
            for iteration in range(self.max_iterations):
                prompt = (
                    "You are an external-context Recursive Language Model controller. "
                    "The full context is deliberately absent from this prompt and is "
                    "available only as the persistent Python variable `context`. Use the "
                    "restricted REPL to inspect and transform it. The namespace persists "
                    "between iterations. Top-level await is supported. Available async "
                    "functions are llm_query(str), llm_query_batched(list[str]), "
                    "rlm_query(str), and rlm_query_batched(list[str]). Recursive calls "
                    "receive a fresh REPL over the same external context. Return exactly "
                    "one JSON action:\n"
                    '- {"type":"execute","code":"Python source"}\n'
                    '- {"type":"answer","content":"final answer"}\n'
                    "Imports, filesystem/network access, private attributes, dynamic "
                    "evaluation, function/class definitions, and unbounded while loops "
                    "are unavailable. This capability restriction is not an operating-"
                    "system sandbox.\n\n"
                    f"ROOT INSTRUCTION:\n{root_instruction}\n\n"
                    f"CURRENT INSTRUCTION:\n{instruction}\n\n"
                    f"DEPTH: {depth}/{context.ledger.limits.max_depth}\n"
                    f"EXTERNAL CONTEXT STATS: {json.dumps({'characters': len(task.context)})}\n\n"
                    "RECENT REPL OBSERVATIONS:\n"
                    f"{json.dumps(observations[-self.max_observations :], ensure_ascii=False)}"
                )
                response = await call_model(
                    ModelRequest(
                        agent_id=agent_id,
                        role="rlm_controller",
                        prompt=prompt,
                        metadata={
                            "depth": depth,
                            "iteration": iteration,
                            "external_context": True,
                            "domain_tools": False,
                        },
                    )
                )
                action = parse_json_object(response.text)
                action_type = action.get("type")
                if action_type == "answer":
                    return require_string(action, "content")
                if action_type != "execute":
                    raise ProtocolError("RLM controller must execute or answer")
                code = require_string(action, "code")
                output = await execute_repl_tool(
                    repl,
                    code,
                    context,
                    agent_id=agent_id,
                    role="rlm_controller",
                    tool_name="rlm_restricted_python",
                )
                observations.append(
                    {
                        "iteration": iteration,
                        "output": output,
                        "variables": list(repl.user_variables),
                    }
                )
            raise ProtocolError(f"{agent_id} exhausted its RLM iteration limit")

        answer = await solve(
            task.prompt,
            agent_id="/rlm/root",
            depth=0,
            root_instruction=task.prompt,
        )
        return answer, {
            "recursive_agents_created": recursive_agents_created,
            "peak_depth": peak_depth,
            "subcalls": subcalls,
            "bare_subcalls": bare_subcalls,
            "recursive_subcalls": recursive_subcalls,
            "depth_fallbacks": depth_fallbacks,
            "persistent_python_namespace": True,
            "external_context_variable": "context",
            "restricted_repl": True,
            "recoverable_repl_errors": True,
            "operating_system_sandbox": False,
            "shared_budget_ledger": True,
            "rlm_reproduction": False,
            "fidelity": "source_matched_restricted_rlm_subset",
        }
