from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from scaffoldlab.environments.base import ToolExecution

from .agent import MiniAgent
from .environments.base import BaseEnvironment
from .runtime import RunContext
from .types import AgentResult, ProtocolError, ToolCall, ToolDefinition


AgentBuilder = Callable[
    [str, Any, RunContext, str | None], MiniAgent | Awaitable[MiniAgent]
]
EnvironmentFactory = Callable[[str], Any | Awaitable[Any]]


@dataclass(frozen=True)
class Message:
    sequence: int
    sender: str
    recipient: str
    content: str
    kind: str = "message"


@dataclass
class AgentRecord:
    agent_id: str
    parent_id: str | None
    inbox: deque[Message] = field(default_factory=deque)
    task: asyncio.Task[None] | None = None
    environment: Any = None
    status: str = "starting"
    result: AgentResult | None = None
    error: BaseException | None = None


class CommunicationEnvironment(BaseEnvironment):
    def __init__(self, base: Any, orchestrator: "Orchestrator", owner: str) -> None:
        self.base = base
        self.orchestrator = orchestrator
        self.owner = owner
        self._closed = False
        base_names = {tool.name for tool in base.tools()}
        conflicts = base_names.intersection(
            {"spawn_agent", "send_message", "read_messages", "wait"}
        )
        if conflicts:
            raise ValueError(f"environment tools conflict with communication: {conflicts}")

    @staticmethod
    def _schema(
        properties: Mapping[str, Any], required: Sequence[str] = ()
    ) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        }

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            *self.base.tools(),
            ToolDefinition(
                name="spawn_agent",
                description="Start another mini-agent on a bounded task.",
                input_schema=self._schema(
                    {
                        "task": {"type": "string"},
                        "profile": {"type": "string"},
                    },
                    ("task",),
                ),
            ),
            ToolDefinition(
                name="send_message",
                description="Send a text message to a known agent.",
                input_schema=self._schema(
                    {
                        "agent_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    ("agent_id", "message"),
                ),
            ),
            ToolDefinition(
                name="read_messages",
                description="Read and remove all messages currently in your inbox.",
                input_schema=self._schema({}),
            ),
            ToolDefinition(
                name="wait",
                description="Wait until one selected agent finishes and return statuses.",
                input_schema=self._schema(
                    {
                        "agent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    }
                ),
            ),
        )

    async def initial_observation(self) -> ToolExecution | None:
        initial = getattr(self.base, "initial_observation", None)
        return None if initial is None else await initial()

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name not in {
            "spawn_agent",
            "send_message",
            "read_messages",
            "wait",
        }:
            return await self.base.execute(action)
        if action.name == "spawn_agent":
            task = action.arguments.get("task")
            profile = action.arguments.get("profile")
            if not isinstance(task, str) or not task.strip():
                raise ProtocolError("spawn_agent task must be a non-empty string")
            if profile is not None and (
                not isinstance(profile, str) or not profile.strip()
            ):
                raise ProtocolError("spawn_agent profile must be a non-empty string")
            agent_id = await self.orchestrator.spawn(
                self.owner, task, profile=profile
            )
            return ToolExecution(output=json.dumps({"agent_id": agent_id}))
        if action.name == "send_message":
            target = action.arguments.get("agent_id")
            message = action.arguments.get("message")
            if not isinstance(target, str) or not target:
                raise ProtocolError("send_message agent_id must be a non-empty string")
            if not isinstance(message, str) or not message:
                raise ProtocolError("send_message message must be a non-empty string")
            await self.orchestrator.send_message(self.owner, target, message)
            return ToolExecution(output=json.dumps({"sent": True, "agent_id": target}))
        if action.name == "read_messages":
            messages = await self.orchestrator.read_messages(self.owner)
            return ToolExecution(
                output=json.dumps([message.__dict__ for message in messages])
            )
        targets = action.arguments.get("agent_ids")
        if targets is not None and (
            not isinstance(targets, list)
            or not all(isinstance(target, str) and target for target in targets)
        ):
            raise ProtocolError("wait agent_ids must be a string list")
        statuses = await self.orchestrator.wait(self.owner, targets)
        return ToolExecution(output=json.dumps(statuses, sort_keys=True))

    async def finish(self) -> None:
        finish = getattr(self.base, "finish", None)
        if finish is not None:
            await finish()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self.base.close()


class Orchestrator:
    """Bounded async tasks and mailboxes around otherwise ordinary MiniAgents."""

    def __init__(
        self,
        *,
        agent_builder: AgentBuilder,
        environment_factory: EnvironmentFactory,
        context: RunContext,
        max_agents: int = 4,
        allow_shared_environment: bool = False,
    ) -> None:
        if not isinstance(max_agents, int) or isinstance(max_agents, bool) or max_agents < 1:
            raise ValueError("max_agents must be a positive integer")
        self.agent_builder = agent_builder
        self.environment_factory = environment_factory
        self.context = context
        self.max_agents = max_agents
        if not isinstance(allow_shared_environment, bool):
            raise ValueError("allow_shared_environment must be a boolean")
        self.allow_shared_environment = allow_shared_environment
        self._records: dict[str, AgentRecord] = {}
        self._environment_ids: set[int] = set()
        self._child_counts: dict[str, int] = {}
        self._message_sequence = 0
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def records(self) -> Mapping[str, AgentRecord]:
        return dict(self._records)

    async def run(self, task: str, *, profile: str | None = None) -> AgentResult:
        if self._running:
            raise RuntimeError("orchestrator instances can run only once")
        self._running = True
        root = await self._create_agent(
            agent_id="/root",
            parent_id=None,
            task=task,
            profile=profile,
        )
        assert root.task is not None
        try:
            await root.task
            if root.error is not None:
                raise root.error
            if root.result is None:
                raise RuntimeError("root agent stopped without a result")
            return root.result
        finally:
            await self._cancel_remaining(except_agent="/root")

    async def spawn(
        self, parent_id: str, task: str, *, profile: str | None = None
    ) -> str:
        async with self._lock:
            if parent_id not in self._records:
                raise ProtocolError(f"unknown parent agent {parent_id!r}")
            if len(self._records) >= self.max_agents:
                raise ProtocolError(f"maximum agent count reached ({self.max_agents})")
            child_number = self._child_counts.get(parent_id, 0) + 1
            self._child_counts[parent_id] = child_number
            agent_id = f"{parent_id}/{child_number}"
            await self._create_agent_locked(
                agent_id=agent_id,
                parent_id=parent_id,
                task=task,
                profile=profile,
            )
            return agent_id

    async def _create_agent(
        self,
        *,
        agent_id: str,
        parent_id: str | None,
        task: str,
        profile: str | None,
    ) -> AgentRecord:
        async with self._lock:
            return await self._create_agent_locked(
                agent_id=agent_id,
                parent_id=parent_id,
                task=task,
                profile=profile,
            )

    async def _create_agent_locked(
        self,
        *,
        agent_id: str,
        parent_id: str | None,
        task: str,
        profile: str | None,
    ) -> AgentRecord:
        raw_base = self.environment_factory(agent_id)
        base: Any = await raw_base if inspect.isawaitable(raw_base) else raw_base
        if id(base) in self._environment_ids and not self.allow_shared_environment:
            raise ValueError(
                "environment_factory reused an environment; pass "
                "allow_shared_environment=True only for an explicit shared experiment"
            )
        self._environment_ids.add(id(base))
        try:
            environment = CommunicationEnvironment(base, self, agent_id)
        except BaseException:
            await base.close()
            raise
        try:
            built = self.agent_builder(agent_id, environment, self.context, profile)
            agent = await built if inspect.isawaitable(built) else built
        except BaseException:
            await environment.close()
            raise
        if not isinstance(agent, MiniAgent):
            await environment.close()
            raise TypeError("agent_builder must return MiniAgent")
        if agent.agent_id != agent_id or agent.context is not self.context:
            await environment.close()
            raise ValueError(
                "agent_builder must use the supplied agent_id and shared context"
            )
        record = AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
            environment=environment,
            status="running",
        )
        self._records[agent_id] = record
        record.task = asyncio.create_task(self._run_agent(record, agent, task))
        await self.context.trace.emit(
            "agent_spawned",
            agent_id=agent_id,
            role=agent.role,
            data={"parent_id": parent_id, "profile": profile},
        )
        return record

    async def _run_agent(
        self, record: AgentRecord, agent: MiniAgent, task: str
    ) -> None:
        try:
            record.result = await agent.run(task)
            record.status = "completed"
            await self.context.trace.emit(
                "agent_completed",
                agent_id=record.agent_id,
                role=agent.role,
                data={"steps": record.result.steps},
            )
            if record.parent_id is not None:
                await self._deliver(
                    record.agent_id,
                    record.parent_id,
                    record.result.answer,
                    kind="result",
                )
        except asyncio.CancelledError as exc:
            record.status = "cancelled"
            record.error = exc
            await self.context.trace.emit(
                "agent_cancelled", agent_id=record.agent_id, role=agent.role
            )
        except BaseException as exc:
            record.status = "failed"
            record.error = exc
            await self.context.trace.emit(
                "agent_failed",
                agent_id=record.agent_id,
                role=agent.role,
                data={"error": type(exc).__name__, "message": str(exc)},
            )
            if record.parent_id is not None:
                await self._deliver(
                    record.agent_id,
                    record.parent_id,
                    f"{type(exc).__name__}: {exc}",
                    kind="error",
                )
        finally:
            try:
                await record.environment.close()
            except BaseException as exc:
                if record.error is None:
                    record.status = "failed"
                    record.error = exc

    async def _deliver(
        self, sender: str, recipient: str, content: str, *, kind: str
    ) -> Message:
        async with self._lock:
            target = self._records.get(recipient)
            if target is None:
                raise ProtocolError(f"unknown target agent {recipient!r}")
            self._message_sequence += 1
            message = Message(
                sequence=self._message_sequence,
                sender=sender,
                recipient=recipient,
                content=content,
                kind=kind,
            )
            target.inbox.append(message)
        await self.context.trace.emit(
            "message_sent",
            agent_id=sender,
            role="communication",
            data={"recipient": recipient, "sequence": message.sequence, "kind": kind},
        )
        return message

    async def send_message(self, sender: str, recipient: str, content: str) -> None:
        if sender not in self._records:
            raise ProtocolError(f"unknown sender agent {sender!r}")
        await self._deliver(sender, recipient, content, kind="message")

    async def read_messages(self, agent_id: str) -> list[Message]:
        async with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                raise ProtocolError(f"unknown agent {agent_id!r}")
            messages = list(record.inbox)
            record.inbox.clear()
        await self.context.trace.emit(
            "messages_read",
            agent_id=agent_id,
            role="communication",
            data={"count": len(messages)},
        )
        return messages

    def _status(self, agent_id: str) -> dict[str, Any]:
        record = self._records[agent_id]
        status: dict[str, Any] = {"agent_id": agent_id, "status": record.status}
        if record.result is not None:
            status["answer"] = record.result.answer
        if record.error is not None and record.status == "failed":
            status["error"] = f"{type(record.error).__name__}: {record.error}"
        return status

    async def wait(
        self, owner: str, agent_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        if owner not in self._records:
            raise ProtocolError(f"unknown agent {owner!r}")
        targets = list(agent_ids) if agent_ids is not None else [
            agent_id
            for agent_id, record in self._records.items()
            if agent_id != owner and record.status in {"starting", "running"}
        ]
        if owner in targets:
            raise ProtocolError("an agent cannot wait for itself")
        if len(targets) != len(set(targets)):
            raise ProtocolError("wait agent_ids must be unique")
        unknown = [target for target in targets if target not in self._records]
        if unknown:
            raise ProtocolError(f"wait references unknown agents {unknown}")
        pending: list[asyncio.Task[None]] = []
        for target in targets:
            record = self._records[target]
            if record.status in {"starting", "running"} and record.task is not None:
                pending.append(record.task)
        if pending and not any(
            self._records[target].status not in {"starting", "running"}
            for target in targets
        ):
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        return [self._status(target) for target in targets]

    async def _cancel_remaining(self, *, except_agent: str) -> None:
        pending_records = [
            record
            for agent_id, record in self._records.items()
            if agent_id != except_agent
            and record.task is not None
            and not record.task.done()
        ]
        for record in pending_records:
            assert record.task is not None
            record.task.cancel()
        if pending_records:
            await asyncio.gather(
                *(record.task for record in pending_records if record.task is not None),
                return_exceptions=True,
            )
        # A task cancelled before its coroutine starts cannot run its own finally.
        for record in pending_records:
            if record.status in {"starting", "running"}:
                record.status = "cancelled"
                record.error = asyncio.CancelledError()
                await record.environment.close()
                await self.context.trace.emit(
                    "agent_cancelled",
                    agent_id=record.agent_id,
                    role="solver",
                )


__all__ = [
    "AgentBuilder",
    "AgentRecord",
    "CommunicationEnvironment",
    "EnvironmentFactory",
    "Message",
    "Orchestrator",
]
