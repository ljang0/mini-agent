from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from ..runtime import RunContext, parse_json_object, require_string
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


@dataclass(frozen=True)
class TeamMessage:
    sender: str
    content: str


def _normalize_recipients(action: Mapping[str, Any]) -> list[str]:
    raw = action.get("to")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return raw
    raise ProtocolError("message action requires 'to' as a string or list of strings")


class FixedAgentTeamHarness(Harness):
    """Long-lived peers with direct messaging and one designated lead."""

    name = "fixed_agent_team"

    def __init__(self, team_size: int = 3, *, max_turns_per_agent: int = 12) -> None:
        self.team_size = require_int_at_least(team_size, "team_size", 2)
        self.max_turns_per_agent = require_int_at_least(
            max_turns_per_agent, "max_turns_per_agent"
        )

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        lead = "/team/lead"
        full_task = task.prompt
        if task.context:
            full_task += f"\n\n<context>\n{task.context}\n</context>"
        agent_ids = [lead] + [
            f"/team/peer-{index}" for index in range(1, self.team_size)
        ]
        mailboxes: dict[str, asyncio.Queue[TeamMessage]] = {
            agent_id: asyncio.Queue() for agent_id in agent_ids
        }
        final: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        histories: dict[str, list[dict[str, Any]]] = {
            agent_id: [] for agent_id in agent_ids
        }
        terminated: set[str] = set()
        waiting: set[str] = set()

        async def deliver(sender: str, recipients: list[str], content: str) -> None:
            for recipient in recipients:
                if recipient not in mailboxes or recipient in terminated:
                    raise ProtocolError(
                        f"unknown or terminated recipient {recipient!r}"
                    )
                if recipient == sender:
                    raise ProtocolError("agents cannot message themselves")
                waiting.discard(recipient)
                await mailboxes[recipient].put(TeamMessage(sender, content))
                await context.trace.emit(
                    "message_sent",
                    agent_id=sender,
                    role="team",
                    data={"to": recipient, "content_chars": len(content)},
                )

        async def run_agent(agent_id: str) -> None:
            is_lead = agent_id == lead
            pending_messages: list[TeamMessage] = []
            for turn in range(self.max_turns_per_agent):
                if final.done() or agent_id in terminated:
                    return
                while True:
                    try:
                        pending_messages.append(mailboxes[agent_id].get_nowait())
                    except asyncio.QueueEmpty:
                        break
                prompt = (
                    f"You are {agent_id} in a fixed team of {self.team_size} peer agents. "
                    f"The designated lead is {lead}. Every peer sees the complete task and "
                    "has identical task capabilities. Only the lead may submit the final "
                    "user-facing answer. Coordinate through messages.\n\n"
                    "Return one JSON action only:\n"
                    '- {"type":"message","to":"agent-id or list",'
                    '"content":"message"}\n'
                    '- {"type":"wait"}\n'
                    '- {"type":"submit","content":"answer or peer report"}\n\n'
                    f"TASK:\n{full_task}\n\n"
                    f"NEW MESSAGES:\n{json.dumps([message.__dict__ for message in pending_messages], ensure_ascii=False)}\n\n"
                    f"YOUR PRIOR ACTIONS:\n{json.dumps(histories[agent_id], ensure_ascii=False)}"
                )
                pending_messages = []
                response = await context.call(
                    ModelRequest(
                        agent_id=agent_id,
                        role="team_lead" if is_lead else "team_peer",
                        prompt=prompt,
                        metadata={
                            "long_lived": True,
                            "full_task_visible": True,
                            "task_tools": True,
                            "turn": turn,
                        },
                    )
                )
                action = parse_json_object(response.text)
                histories[agent_id].append(action)
                action_type = action.get("type")
                if action_type == "message":
                    await deliver(
                        agent_id,
                        _normalize_recipients(action),
                        require_string(action, "content"),
                    )
                elif action_type == "wait":
                    try:
                        message = mailboxes[agent_id].get_nowait()
                    except asyncio.QueueEmpty:
                        waiting.add(agent_id)
                        if all(
                            peer in terminated or peer in waiting for peer in agent_ids
                        ) and all(
                            mailboxes[peer].empty()
                            for peer in agent_ids
                            if peer not in terminated
                        ):
                            raise ProtocolError(
                                "fixed team deadlocked with every live agent waiting"
                            )
                        try:
                            message = await mailboxes[agent_id].get()
                        finally:
                            waiting.discard(agent_id)
                    pending_messages.append(message)
                elif action_type == "submit":
                    content = require_string(action, "content")
                    if is_lead:
                        if not final.done():
                            final.set_result(content)
                        return
                    terminated.add(agent_id)
                    waiting.discard(agent_id)
                    await deliver(agent_id, [lead], content)
                    return
                else:
                    raise ProtocolError(f"unknown fixed-team action {action_type!r}")
            raise ProtocolError(f"{agent_id} exhausted its turn limit")

        tasks = [context.create_task(run_agent(agent_id)) for agent_id in agent_ids]
        observed: set[asyncio.Future[Any]] = set()
        try:
            while not final.done():
                for completed in tasks:
                    if completed.done() and completed not in observed:
                        observed.add(completed)
                        if (
                            not completed.cancelled()
                            and completed.exception() is not None
                        ):
                            raise completed.exception()  # type: ignore[misc]
                active = [running for running in tasks if not running.done()]
                if not active:
                    raise ProtocolError("fixed team terminated without a lead answer")
                done, _ = await asyncio.wait(
                    [final, *active], return_when=asyncio.FIRST_COMPLETED
                )
                for completed in done:
                    if completed is final or completed.cancelled():
                        continue
                    observed.add(completed)
                    error = completed.exception()
                    if error is not None:
                        raise error
            answer = final.result()
        finally:
            for running in tasks:
                if not running.done():
                    running.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return answer, {
            "agents_created": self.team_size,
            "team_size": self.team_size,
            "long_lived_agents": True,
            "peer_messaging": True,
            "anthropic_public_card_reproduction": False,
            "fidelity": "anthropic_card_topology_simulation",
        }


class AsyncSubagentsHarness(Harness):
    """A tool-capable lead dynamically manages long-lived asynchronous subagents."""

    name = "async_subagents"

    def __init__(
        self,
        *,
        max_concurrent_subagents: int = 4,
        max_total_subagents: int = 20,
        max_turns_per_agent: int = 20,
    ) -> None:
        self.max_concurrent_subagents = require_int_at_least(
            max_concurrent_subagents, "max_concurrent_subagents"
        )
        self.max_total_subagents = require_int_at_least(
            max_total_subagents, "max_total_subagents"
        )
        self.max_turns_per_agent = require_int_at_least(
            max_turns_per_agent, "max_turns_per_agent"
        )
        if self.max_total_subagents < self.max_concurrent_subagents:
            raise ValueError(
                "max_total_subagents cannot be smaller than max_concurrent_subagents"
            )

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        lead = "/async/lead"
        full_task = task.prompt
        if task.context:
            full_task += f"\n\n<context>\n{task.context}\n</context>"
        mailboxes: dict[str, asyncio.Queue[TeamMessage]] = {lead: asyncio.Queue()}
        instructions = {lead: full_task}
        histories: dict[str, list[dict[str, Any]]] = {lead: []}
        states: dict[str, str] = {lead: "working"}
        agent_tasks: dict[str, asyncio.Task[None]] = {}
        final: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        lock = asyncio.Lock()
        roster_changed = asyncio.Event()
        total_spawned = 0

        def is_deadlocked() -> bool:
            live = [
                agent_id for agent_id, state in states.items() if state != "deleted"
            ]
            return (
                bool(live)
                and all(states[agent_id] == "idle" for agent_id in live)
                and all(mailboxes[agent_id].empty() for agent_id in live)
            )

        async def deliver(sender: str, recipient: str, content: str) -> None:
            if recipient not in mailboxes or states.get(recipient) == "deleted":
                raise ProtocolError(f"unknown or deleted recipient {recipient!r}")
            await mailboxes[recipient].put(TeamMessage(sender, content))
            if states.get(recipient) == "idle":
                states[recipient] = "working"
            await context.trace.emit(
                "message_sent",
                agent_id=sender,
                role="async_team",
                data={"to": recipient, "content_chars": len(content)},
            )

        async def spawn(instruction: str, requested_name: Optional[str]) -> str:
            nonlocal total_spawned
            if requested_name is not None and (
                not isinstance(requested_name, str) or not requested_name.strip()
            ):
                raise ProtocolError("async subagent name must be a non-empty string")
            async with lock:
                active = sum(
                    1
                    for agent_id, state in states.items()
                    if agent_id != lead and state != "deleted"
                )
                if active >= self.max_concurrent_subagents:
                    raise ProtocolError(
                        "no async subagent concurrency slot is available"
                    )
                if total_spawned >= self.max_total_subagents:
                    raise ProtocolError("total async subagent cap reached")
                total_spawned += 1
                suffix = requested_name or f"worker-{total_spawned}"
                safe_suffix = (
                    "".join(
                        character if character.isalnum() or character in "-_" else "-"
                        for character in suffix
                    ).strip("-")
                    or f"worker-{total_spawned}"
                )
                agent_id = f"/async/{safe_suffix}"
                if agent_id in mailboxes:
                    agent_id = f"{agent_id}-{total_spawned}"
                mailboxes[agent_id] = asyncio.Queue()
                instructions[agent_id] = instruction
                histories[agent_id] = []
                states[agent_id] = "working"
                child_task = context.create_task(run_agent(agent_id))
                agent_tasks[agent_id] = child_task
                child_task.add_done_callback(lambda _: roster_changed.set())
                roster_changed.set()
            await context.trace.emit(
                "agent_spawned",
                agent_id=lead,
                role="async_team",
                data={"child": agent_id},
            )
            return agent_id

        async def delete(agent_id: str) -> None:
            if agent_id == lead or agent_id not in states:
                raise ProtocolError(f"cannot delete {agent_id!r}")
            states[agent_id] = "deleted"
            running = agent_tasks.get(agent_id)
            if running and not running.done():
                running.cancel()
            await context.trace.emit(
                "agent_deleted",
                agent_id=lead,
                role="async_team",
                data={"child": agent_id},
            )

        async def run_agent(agent_id: str) -> None:
            is_lead = agent_id == lead
            pending_messages: list[TeamMessage] = []
            for turn in range(self.max_turns_per_agent):
                if final.done() or states.get(agent_id) == "deleted":
                    return
                while True:
                    try:
                        pending_messages.append(mailboxes[agent_id].get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if states.get(agent_id) == "idle" and not pending_messages:
                    pending_messages.append(await mailboxes[agent_id].get())
                    states[agent_id] = "working"

                visible_task = instructions[agent_id]
                lead_actions = (
                    '- {"type":"spawn","instruction":"bounded task",'
                    '"name":"optional-name"}\n'
                    '- {"type":"delete","agent_id":"/async/..."}\n'
                    '- {"type":"status"}\n'
                    if is_lead
                    else ""
                )
                prompt = (
                    f"You are {agent_id} in an asynchronous subagent harness. "
                    f"The lead is {lead}. The lead retains task tools and is the only agent "
                    "that may submit the final answer. Subagents see only their delegated "
                    "instruction, remain long-lived, and become idle after reporting.\n\n"
                    "Return one JSON action only:\n"
                    f"{lead_actions}"
                    '- {"type":"message","to":"agent-id",'
                    '"content":"message"}\n'
                    '- {"type":"wait"}\n'
                    '- {"type":"submit","content":"answer or report"}\n\n'
                    f"YOUR INSTRUCTION:\n{visible_task}\n\n"
                    f"NEW MESSAGES:\n{json.dumps([message.__dict__ for message in pending_messages], ensure_ascii=False)}\n\n"
                    f"YOUR PRIOR ACTIONS:\n{json.dumps(histories[agent_id], ensure_ascii=False)}"
                )
                pending_messages = []
                response = await context.call(
                    ModelRequest(
                        agent_id=agent_id,
                        role="async_lead" if is_lead else "async_subagent",
                        prompt=prompt,
                        metadata={
                            "long_lived": True,
                            "full_task_visible": is_lead,
                            "task_tools": True,
                            "turn": turn,
                        },
                    )
                )
                action = parse_json_object(response.text)
                histories[agent_id].append(action)
                action_type = action.get("type")
                if action_type == "spawn":
                    if not is_lead:
                        raise ProtocolError("only the lead may spawn async subagents")
                    child = await spawn(
                        require_string(action, "instruction"), action.get("name")
                    )
                    pending_messages.append(
                        TeamMessage("runtime", f"Spawned {child}; execution continues.")
                    )
                elif action_type == "delete":
                    if not is_lead:
                        raise ProtocolError("only the lead may delete async subagents")
                    await delete(require_string(action, "agent_id"))
                    pending_messages.append(TeamMessage("runtime", "Agent deleted."))
                elif action_type == "status":
                    if not is_lead:
                        raise ProtocolError("only the lead may inspect async status")
                    pending_messages.append(
                        TeamMessage("runtime", json.dumps(states, sort_keys=True))
                    )
                elif action_type == "message":
                    recipient = require_string(action, "to")
                    if recipient == agent_id:
                        raise ProtocolError("agents cannot message themselves")
                    await deliver(
                        agent_id, recipient, require_string(action, "content")
                    )
                elif action_type == "wait":
                    states[agent_id] = "idle"
                    if is_deadlocked():
                        raise ProtocolError(
                            "async team deadlocked with every live agent idle"
                        )
                    try:
                        pending_messages.append(await mailboxes[agent_id].get())
                    finally:
                        if states.get(agent_id) != "deleted":
                            states[agent_id] = "working"
                elif action_type == "submit":
                    content = require_string(action, "content")
                    if is_lead:
                        if not final.done():
                            final.set_result(content)
                        return
                    await deliver(agent_id, lead, content)
                    states[agent_id] = "idle"
                    if is_deadlocked():
                        raise ProtocolError(
                            "async team deadlocked with every live agent idle"
                        )
                    try:
                        pending_messages.append(await mailboxes[agent_id].get())
                    finally:
                        if states.get(agent_id) != "deleted":
                            states[agent_id] = "working"
                else:
                    raise ProtocolError(f"unknown async-team action {action_type!r}")
            raise ProtocolError(f"{agent_id} exhausted its turn limit")

        lead_task = context.create_task(run_agent(lead))
        agent_tasks[lead] = lead_task
        lead_task.add_done_callback(lambda _: roster_changed.set())
        observed: set[asyncio.Future[Any]] = set()
        try:
            while not final.done():
                roster_changed.clear()
                current_tasks = list(agent_tasks.values())
                for completed in current_tasks:
                    if completed.done() and completed not in observed:
                        observed.add(completed)
                        if (
                            not completed.cancelled()
                            and completed.exception() is not None
                        ):
                            raise completed.exception()  # type: ignore[misc]
                active = [running for running in current_tasks if not running.done()]
                if not active:
                    raise ProtocolError("async team terminated without a final answer")
                roster_waiter = asyncio.create_task(roster_changed.wait())
                try:
                    waitables: list[asyncio.Future[Any]] = [
                        final,
                        roster_waiter,
                        *active,
                    ]
                    done, _ = await asyncio.wait(
                        waitables,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if not roster_waiter.done():
                        roster_waiter.cancel()
                        await asyncio.gather(roster_waiter, return_exceptions=True)
                for completed in done:
                    if (
                        completed is final
                        or completed is roster_waiter
                        or completed.cancelled()
                    ):
                        continue
                    observed.add(completed)
                    error = completed.exception()
                    if error is not None:
                        raise error
            answer = final.result()
        finally:
            current_tasks = list(agent_tasks.values())
            for running in current_tasks:
                if not running.done():
                    running.cancel()
            await asyncio.gather(*current_tasks, return_exceptions=True)
        return answer, {
            "agents_created": total_spawned + 1,
            "subagents_created": total_spawned,
            "long_lived_agents": True,
            "dynamic_spawning": True,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "max_total_subagents": self.max_total_subagents,
            "anthropic_public_card_reproduction": False,
            "fidelity": "anthropic_card_topology_simulation",
        }
