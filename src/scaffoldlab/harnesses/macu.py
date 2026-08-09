from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from ..runtime import RunContext, parse_json_object, require_string
from ..types import BudgetExceeded, ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


TERMINAL_STATES = {"succeeded", "cancelled"}


@dataclass
class DAGNode:
    node_id: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    state: str = "pending"
    result: str = ""
    attempts: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "goal": self.goal,
            "depends_on": list(self.depends_on),
            "state": self.state,
            "result": self.result,
            "attempts": self.attempts,
        }


def _parse_node(raw: Any) -> DAGNode:
    if not isinstance(raw, dict):
        raise ProtocolError("DAG nodes must be objects")
    node_id = require_string(raw, "id")
    goal = require_string(raw, "goal")
    dependencies = raw.get("depends_on", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) and item for item in dependencies
    ):
        raise ProtocolError("depends_on must be a list of node ids")
    return DAGNode(node_id=node_id, goal=goal, depends_on=list(dependencies))


def _validate_dag(nodes: Mapping[str, DAGNode]) -> None:
    for node in nodes.values():
        unknown = set(node.depends_on) - set(nodes)
        if unknown:
            raise ProtocolError(
                f"node {node.node_id!r} has unknown dependencies {sorted(unknown)}"
            )
        if node.node_id in node.depends_on:
            raise ProtocolError(f"node {node.node_id!r} depends on itself")
    indegree = {node_id: 0 for node_id in nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in nodes.values():
        for dependency in node.depends_on:
            indegree[node.node_id] += 1
            children[dependency].append(node.node_id)
    frontier = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while frontier:
        current = frontier.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    if visited != len(nodes):
        raise ProtocolError("manager emitted a cyclic task graph")


class MACUHarness(Harness):
    """MACU-inspired manager-owned text DAG; not the upstream CUA runtime."""

    name = "macu_dynamic_dag"

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_nodes: int = 32,
        max_replans: int = 32,
    ) -> None:
        self.max_workers = require_int_at_least(max_workers, "max_workers")
        self.max_nodes = require_int_at_least(max_nodes, "max_nodes")
        self.max_replans = require_int_at_least(max_replans, "max_replans")

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        full_task = task.prompt
        if task.context:
            full_task += f"\n\n<context>\n{task.context}\n</context>"
        initial_prompt = (
            "Create a directed acyclic graph of bounded subgoals for parallel worker "
            "agents. Dependencies must encode every information handoff. Return JSON "
            'exactly as {"nodes":[{"id":"short-id","goal":"...",'
            '"depends_on":["id"]}, ...]}. The manager plans but does not perform '
            f"task work. Use at most {self.max_nodes} nodes.\n\nTASK:\n{full_task}"
        )
        initial = await context.call(
            ModelRequest(
                agent_id="/macu/manager",
                role="initial_plan",
                prompt=initial_prompt,
                system="You are the MACU manager. Maintain a valid mutable task DAG.",
                metadata={"task_tools": False},
            )
        )
        plan = parse_json_object(initial.text)
        raw_nodes = plan.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ProtocolError("MACU manager must emit a non-empty nodes list")
        if len(raw_nodes) > self.max_nodes:
            raise ProtocolError("initial MACU graph exceeds max_nodes")
        nodes: dict[str, DAGNode] = {}
        for raw_node in raw_nodes:
            node = _parse_node(raw_node)
            if node.node_id in nodes:
                raise ProtocolError(f"duplicate DAG node id {node.node_id!r}")
            nodes[node.node_id] = node
        _validate_dag(nodes)

        running: dict[asyncio.Task[str], str] = {}
        invalidated_running: set[str] = set()
        replans = 0
        peak_parallel_workers = 0
        stop_requested = False
        stale_results_discarded = 0

        def descendants(node_id: str) -> set[str]:
            found: set[str] = set()
            frontier = [node_id]
            while frontier:
                parent = frontier.pop()
                for candidate in nodes.values():
                    if (
                        parent in candidate.depends_on
                        and candidate.node_id not in found
                    ):
                        found.add(candidate.node_id)
                        frontier.append(candidate.node_id)
            return found

        async def run_node(node: DAGNode) -> str:
            node.attempts += 1
            parent_context = [
                {"id": dependency, "result": nodes[dependency].result}
                for dependency in node.depends_on
            ]
            prompt = (
                f"Complete this bounded DAG node: {node.goal}\n\n"
                f"PARENT HANDOFFS:\n{json.dumps(parent_context, ensure_ascii=False)}"
            )
            if task.context:
                prompt += f"\n\nTASK CONTEXT:\n{task.context}"
            response = await context.call(
                ModelRequest(
                    agent_id=f"/macu/worker/{node.node_id}",
                    role="worker" if node.attempts == 1 else "worker_follow_up",
                    prompt=prompt,
                    metadata={
                        "node_id": node.node_id,
                        "attempt": node.attempts,
                        "isolated_logical_context": True,
                    },
                )
            )
            return response.text

        async def replan(latest: DAGNode) -> None:
            nonlocal replans, stop_requested
            if replans >= self.max_replans:
                raise ProtocolError("MACU manager exhausted max_replans")
            replans += 1
            revision_prompt = (
                "Revise the task DAG after the latest worker observation. You may add, "
                "rewrite, cancel, or request a follow-up for nodes. Preserve a valid DAG. "
                'Return JSON exactly as {"add_nodes":[node...],'
                '"rewrite_nodes":[node...],"cancel_nodes":["id"],'
                '"follow_up":[{"id":"id","goal":"continuation"}],'
                '"stop":false}. Empty lists mean no change.\n\n'
                f"ORIGINAL TASK:\n{full_task}\n\n"
                f"CURRENT DAG:\n{json.dumps([node.public() for node in nodes.values()], ensure_ascii=False)}\n\n"
                f"LATEST OBSERVATION:\n{json.dumps(latest.public(), ensure_ascii=False)}"
            )
            response = await context.call(
                ModelRequest(
                    agent_id="/macu/manager",
                    role="replan",
                    prompt=revision_prompt,
                    system="You are the MACU manager. Maintain a valid mutable task DAG.",
                    metadata={"task_tools": False, "replan_index": replans},
                )
            )
            revision = parse_json_object(response.text)
            cancel_nodes = revision.get("cancel_nodes", [])
            if not isinstance(cancel_nodes, list):
                raise ProtocolError("cancel_nodes must be a list")
            for node_id in cancel_nodes:
                if not isinstance(node_id, str) or node_id not in nodes:
                    raise ProtocolError(f"cannot cancel unknown node {node_id!r}")
                if nodes[node_id].state != "pending":
                    raise ProtocolError("only pending nodes may be cancelled")
                nodes[node_id].state = "cancelled"

            rewrite_nodes = revision.get("rewrite_nodes", [])
            if not isinstance(rewrite_nodes, list):
                raise ProtocolError("rewrite_nodes must be a list")
            for raw_node in rewrite_nodes:
                replacement = _parse_node(raw_node)
                if replacement.node_id not in nodes:
                    raise ProtocolError(
                        f"cannot rewrite unknown node {replacement.node_id!r}"
                    )
                if nodes[replacement.node_id].state != "pending":
                    raise ProtocolError("only pending nodes may be rewritten")
                nodes[replacement.node_id].goal = replacement.goal
                nodes[replacement.node_id].depends_on = replacement.depends_on

            add_nodes = revision.get("add_nodes", [])
            if not isinstance(add_nodes, list):
                raise ProtocolError("add_nodes must be a list")
            for raw_node in add_nodes:
                node = _parse_node(raw_node)
                if node.node_id in nodes:
                    raise ProtocolError(f"cannot add duplicate node {node.node_id!r}")
                if len(nodes) >= self.max_nodes:
                    raise ProtocolError("MACU graph exceeded max_nodes")
                nodes[node.node_id] = node

            follow_ups = revision.get("follow_up", [])
            if not isinstance(follow_ups, list):
                raise ProtocolError("follow_up must be a list")
            for follow_up in follow_ups:
                if not isinstance(follow_up, dict):
                    raise ProtocolError("follow_up entries must be objects")
                node_id = require_string(follow_up, "id")
                if node_id not in nodes or nodes[node_id].state != "succeeded":
                    raise ProtocolError("follow-up must target a succeeded node")
                continuation = require_string(follow_up, "goal")
                previous = nodes[node_id].result
                nodes[
                    node_id
                ].goal = f"{continuation}\n\nPrevious attempt result:\n{previous}"
                nodes[node_id].state = "pending"
                nodes[node_id].result = ""
                for descendant_id in descendants(node_id):
                    descendant = nodes[descendant_id]
                    if descendant.state == "running":
                        invalidated_running.add(descendant_id)
                    elif descendant.state == "succeeded":
                        descendant.state = "pending"
                        descendant.result = ""
                    elif descendant.state == "pending":
                        descendant.result = ""
            _validate_dag(nodes)
            stop = revision.get("stop", False)
            if not isinstance(stop, bool):
                raise ProtocolError("stop must be a boolean")
            if stop:
                stop_requested = True
                for node in nodes.values():
                    if node.state == "pending":
                        node.state = "cancelled"

            # Cancelling or rewriting an ancestor makes every still-pending
            # descendant unreachable, so cancel those descendants transitively.
            changed = True
            while changed:
                changed = False
                for node in nodes.values():
                    if node.state == "pending" and any(
                        nodes[parent].state == "cancelled" for parent in node.depends_on
                    ):
                        node.state = "cancelled"
                        changed = True

        while True:
            running_node_ids = set(running.values())
            ready = [
                node
                for node in nodes.values()
                if not stop_requested
                and node.state == "pending"
                and node.node_id not in running_node_ids
                and all(
                    nodes[parent].state == "succeeded" for parent in node.depends_on
                )
            ]
            while ready and len(running) < self.max_workers:
                node = ready.pop(0)
                node.state = "running"
                running[context.create_task(run_node(node))] = node.node_id
                peak_parallel_workers = max(peak_parallel_workers, len(running))
                await context.trace.emit(
                    "dag_node_started",
                    agent_id=f"/macu/worker/{node.node_id}",
                    role="worker",
                    data={"node_id": node.node_id},
                )

            if not running:
                if all(node.state in TERMINAL_STATES for node in nodes.values()):
                    break
                blocked = [
                    node.node_id for node in nodes.values() if node.state == "pending"
                ]
                raise ProtocolError(f"MACU DAG has no ready frontier: {blocked}")

            completed, _ = await asyncio.wait(
                set(running), return_when=asyncio.FIRST_COMPLETED
            )
            for worker_task in completed:
                node_id = running.pop(worker_task, None)
                if node_id is None:
                    continue
                node = nodes[node_id]
                was_invalidated = node_id in invalidated_running
                try:
                    worker_result = await worker_task
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    if not was_invalidated:
                        raise
                    invalidated_running.remove(node_id)
                    stale_results_discarded += 1
                    node.result = ""
                    node.state = "cancelled" if stop_requested else "pending"
                    await context.trace.emit(
                        "dag_node_result_discarded",
                        agent_id=f"/macu/worker/{node_id}",
                        role="worker",
                        data={
                            "node_id": node_id,
                            "reason": "ancestor_follow_up",
                            "stale_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                if was_invalidated:
                    invalidated_running.remove(node_id)
                    stale_results_discarded += 1
                    node.result = ""
                    node.state = "cancelled" if stop_requested else "pending"
                    await context.trace.emit(
                        "dag_node_result_discarded",
                        agent_id=f"/macu/worker/{node_id}",
                        role="worker",
                        data={"node_id": node_id, "reason": "ancestor_follow_up"},
                    )
                    continue
                node.result = worker_result
                node.state = "succeeded"
                await context.trace.emit(
                    "dag_node_completed",
                    agent_id=f"/macu/worker/{node_id}",
                    role="worker",
                    data={"node_id": node_id},
                )
                if not stop_requested:
                    await replan(node)

        synthesis_prompt = (
            "Aggregate the completed DAG leaves into a final answer for the original task. "
            "Use only retained worker evidence and explicitly reconcile conflicts.\n\n"
            f"TASK:\n{full_task}\n\nFINAL DAG:\n"
            f"{json.dumps([node.public() for node in nodes.values()], ensure_ascii=False)}"
        )
        final = await context.call(
            ModelRequest(
                agent_id="/macu/manager",
                role="synthesize",
                prompt=synthesis_prompt,
                system="You are the MACU manager. Synthesize retained DAG evidence.",
                metadata={"task_tools": False},
            )
        )
        return final.text, {
            "agents_created": 1 + len(nodes),
            "dag_nodes": len(nodes),
            "replans": replans,
            "peak_parallel_workers": peak_parallel_workers,
            "dynamic_dag": True,
            "stop_requested": stop_requested,
            "stale_results_discarded": stale_results_discarded,
            "macu_reproduction": False,
            "fidelity": "macu_text_dag_subset",
        }
