"""Minimal recursive scheduling around ordinary :class:`MiniAgent` workers."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar

from .agent import MiniAgent
from .harnesses import load_harness
from .harnesses.base import ACTIONS, LEGACY_ACTIONS, Harness, Role
from .environments.base import (
    AgentEnvironment,
    BaseEnvironment,
    Environment,
    raise_lifecycle_errors,
)
from .execution import RunContext
from .types import (
    AgentResult,
    BudgetLimits,
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    _require_bool,
    _require_callable,
    _require_int,
    _require_positive_int,
    _require_str,
    _require_tuple_of,
)


AgentBuilder = Callable[
    [str, AgentEnvironment, RunContext], MiniAgent | Awaitable[MiniAgent]
]
EnvironmentFactory = Callable[
    [str], Environment | Awaitable[Environment]
]
_T = TypeVar("_T")
_ACTIVE = frozenset({"starting", "running"})

# One clause per action, so a role's tool description says exactly what that
# role can do instead of describing capabilities it does not hold.
_ACTION_HELP = {
    "spawn": "Spawn a mini-agent.",
    "send": "Send a message to another agent.",
    "inbox": "Read your inbox (wait=true blocks for delivery).",
    "wait": "Wait for a descendant to finish.",
    "stop": "Stop a descendant subtree.",
    "adopt": "Adopt a completed descendant's environment state.",
    "delegate": "Delegate a subtask and block until the subagent answers.",
    "release": "Release an idle descendant so its state can be adopted.",
}

# What an agent is when no harness was selected: the pre-harness mesh.
_LEGACY_ROLE = load_harness("recursive").roles["solver"]


@dataclass(frozen=True)
class MailboxMessage:
    sequence: int
    sender: str
    recipient: str
    content: str
    kind: str = "message"

    def __post_init__(self) -> None:
        _require_int(self.sequence, "mailbox sequence")
        for name in ("sender", "recipient", "content", "kind"):
            _require_str(getattr(self, name), f"mailbox {name}", non_empty=False)
        if self.kind not in {"message", "result", "error"}:
            raise ValueError("mailbox kind must be 'message', 'result', or 'error'")


@dataclass
class AgentRecord:
    agent_id: str
    parent_id: str | None
    inbox: deque[MailboxMessage] = field(default_factory=deque)
    task: asyncio.Task[None] | None = None
    environment: CommunicationEnvironment | None = None
    status: str = "starting"
    result: AgentResult | None = None
    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    terminal_error: BaseException | None = None
    state: object | None = None
    inbox_bytes: int = 0
    inbox_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    adopted_from: str | None = None
    adoption_history: list[str] = field(default_factory=list)
    # Idleness is a boolean rather than a status, so the status vocabulary
    # recorded in every benchmark artifact keeps its existing value domain.
    idle: bool = False
    released: bool = False
    delegated: bool = False


class CommunicationEnvironment(BaseEnvironment):
    """Add one communication tool to a domain environment.

    ``actions`` is the subset of the vocabulary this agent's role holds, and
    ``expose_domain_tools`` is how a role that only coordinates ends up with
    nothing else to call. The defaults reproduce the pre-harness tool exactly.
    """

    def __init__(
        self,
        base: Any,
        orchestrator: "Orchestrator",
        owner: str,
        parent_id: str | None = None,
        *,
        actions: Sequence[str] = LEGACY_ACTIONS,
        expose_domain_tools: bool = True,
    ) -> None:
        if not callable(getattr(base, "tools", None)):
            raise ValueError("domain environment must expose tools")
        tools = _require_tuple_of(
            tuple(base.tools()), ToolDefinition, "domain tools", brief=True
        )
        names = {tool.name for tool in tools}
        if len(names) != len(tools):
            raise ValueError("domain tool names must be unique")
        if tools and not callable(getattr(base, "execute", None)):
            raise ValueError("a domain environment with tools must expose execute")
        if "agent" in names:
            raise ValueError(
                "domain environment already defines the reserved 'agent' tool"
            )
        for name in (
            "initial_observation", "finish", "export_state",
            "adopt_state", "close", "resource_identity",
        ):
            hook = getattr(base, name, None)
            if hook is not None:
                _require_callable(hook, f"domain environment {name}")
        self.base = base
        self.orchestrator = orchestrator
        self.owner = owner
        self.parent_id = parent_id
        self._base_tools = tools if expose_domain_tools else ()
        self._actions = tuple(actions)
        for action in self._actions:
            if action not in ACTIONS:
                raise ValueError(f"unknown agent action {action!r}")
        self._closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        if not self._actions:
            return self._base_tools
        return (*self._base_tools, self._agent_tool())

    def _agent_tool(self) -> ToolDefinition:
        whose = (
            "you have no parent. "
            if self.parent_id is None
            else f"your parent is {self.parent_id!r}. "
        )
        if self._actions == LEGACY_ACTIONS:
            # Pinned verbatim: this string is hashed into every recorded
            # tools_sha256, so composing it would change existing traces.
            summary = (
                "Spawn a mini-agent, exchange messages (use inbox with "
                "wait=true to block for delivery), wait for completion, stop "
                "a descendant subtree, or adopt a completed descendant's "
                "environment state."
            )
        else:
            summary = " ".join(_ACTION_HELP[action] for action in self._actions)
        return ToolDefinition(
            name="agent",
            description=f"You are agent {self.owner!r}; " + whose + summary,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(self._actions)},
                    "task": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "agent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "message": {"type": "string"},
                    "wait": {"type": "boolean"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

    async def initial_observation(self) -> ToolExecution | None:
        hook = getattr(self.base, "initial_observation", None)
        return None if not callable(hook) else await hook()

    def _argument(self, call: ToolCall, key: str, action: str) -> str:
        """Read one bounded, non-empty string argument of ``action``."""

        return _bounded_text(
            call.arguments.get(key),
            f"agent {action} {key}",
            self.orchestrator.max_message_bytes,
        )

    async def execute(self, call: ToolCall) -> ToolExecution:
        """Dispatch one `agent` action, or hand the call to the domain tool.
        
                An action outside this agent's role is refused here, so restricting a
                role is a real restriction rather than a description. Scheduler
                refusals come back as recoverable tool results: hitting the agent
                limit is something the model can react to, not a reason to end the
                run.
                """
        if call.name != "agent":
            if not any(tool.name == call.name for tool in self._base_tools):
                raise InvalidAction(f"unsupported tool {call.name!r}")
            return await self.base.execute(call)
        scheduler = self.orchestrator
        action = call.arguments.get("action")
        if action not in self._actions:
            raise InvalidAction(f"unsupported agent action {action!r}")
        if action == "spawn":
            _only(call, "action", "task")
            task = self._argument(call, "task", action)
            agent_id = await _await_action(scheduler.spawn(self.owner, task))
            return _json_execution({"agent_id": agent_id, "status": "running"})
        if action == "send":
            _only(call, "action", "agent_id", "message")
            target = self._argument(call, "agent_id", action)
            message = self._argument(call, "message", action)
            await _await_action(scheduler.send_message(self.owner, target, message))
            return _json_execution({"sent": True, "agent_id": target})
        if action == "inbox":
            _only(call, "action", "wait")
            requested = call.arguments.get("wait", False)
            blocking = _require_bool(
                requested, "agent inbox wait", error=InvalidAction
            )
            messages = await _await_action(
                scheduler.read_messages(self.owner, wait=blocking)
            )
            return _json_execution([asdict(message) for message in messages])
        if action == "delegate":
            _only(call, "action", "task")
            task = self._argument(call, "task", action)
            return _json_execution(
                await _await_action(scheduler.delegate(self.owner, task))
            )
        if action == "release":
            _only(call, "action", "agent_id")
            target = self._argument(call, "agent_id", action)
            return _json_execution(
                await _await_action(scheduler.release(self.owner, target))
            )
        if action == "wait":
            _only(call, "action", "agent_ids")
            targets = call.arguments.get("agent_ids")
            if targets is not None:
                if not isinstance(targets, list) or not all(
                    isinstance(target, str) and target for target in targets
                ):
                    raise InvalidAction("agent wait agent_ids must be a string list")
                if len(targets) > scheduler.max_total_agents or any(
                    len(target.encode("utf-8")) > scheduler.max_message_bytes
                    for target in targets
                ):
                    raise InvalidAction(
                        "agent wait target list exceeds scheduler limits"
                    )
            return _json_execution(
                await _await_action(scheduler.wait(self.owner, targets))
            )
        if action == "stop":
            _only(call, "action", "agent_id")
            target = self._argument(call, "agent_id", action)
            stopped = await _await_action(scheduler.stop(self.owner, target))
            return _json_execution({"target": target, "agents": stopped})
        if action == "adopt":
            _only(call, "action", "agent_id")
            target = self._argument(call, "agent_id", action)
            await scheduler.adopt(self.owner, target)
            return _json_execution({"adopted": target})
        raise InvalidAction(f"unsupported agent action {action!r}")

    async def finish(self) -> None:
        hook = getattr(self.base, "finish", None)
        if callable(hook):
            await hook()

    async def export_state(self) -> Any:
        hook = getattr(self.base, "export_state", None)
        return None if not callable(hook) else await hook()

    async def adopt_state(self, state: Any) -> None:
        hook = getattr(self.base, "adopt_state", None)
        if not callable(hook):
            raise ProtocolError("this environment cannot adopt child state")
        try:
            await hook(state)
        except NotImplementedError as exc:
            raise ProtocolError("this environment cannot adopt child state") from exc

    async def close(self) -> None:
        if not self._closed:
            close = getattr(self.base, "close", None)
            if not callable(close):
                raise RuntimeError("domain environment must expose close")
            await close()
            self._closed = True

    def resource_identity(self) -> str:
        identity = getattr(self.base, "resource_identity", None)
        if not callable(identity):
            raise RuntimeError("domain environment must expose resource_identity")
        return identity()


class Orchestrator:
    """A bounded scheduler that executes whatever topology a harness declares.

    The scheduler holds the mechanisms -- admission control, mailboxes,
    cancellation, state hand-off -- and the harness decides which of them each
    agent may reach. Without a harness every agent gets the full pre-harness
    vocabulary, which is what keeps existing runs reproducible.
    """

    def __init__(
        self,
        *,
        agent_builder: AgentBuilder,
        environment_factory: EnvironmentFactory,
        context: RunContext,
        max_active_agents: int = 4,
        max_total_agents: int = 64,
        per_agent_limits: BudgetLimits | None = None,
        max_message_bytes: int = 64 * 1024,
        max_inbox_bytes: int = 1024 * 1024,
        root_id: str = "/root",
        harness: Harness | None = None,
    ) -> None:
        if not callable(agent_builder) or not callable(environment_factory):
            raise ValueError("agent_builder and environment_factory must be callable")
        if not isinstance(context, RunContext):
            raise ValueError("context must be RunContext")
        for name, value in (
            ("max_active_agents", max_active_agents),
            ("max_total_agents", max_total_agents),
            ("max_message_bytes", max_message_bytes),
            ("max_inbox_bytes", max_inbox_bytes),
        ):
            _require_positive_int(value, name)
        if max_total_agents < max_active_agents:
            raise ValueError(
                "max_total_agents cannot be smaller than max_active_agents"
            )
        if not isinstance(per_agent_limits, (BudgetLimits, type(None))):
            raise ValueError("per_agent_limits must be BudgetLimits or None")
        if not _valid_agent_id(root_id):
            raise ValueError("root_id must be an absolute agent path")
        if len(root_id.encode("utf-8")) > max_message_bytes:
            raise ValueError("root_id exceeds max_message_bytes")
        self.agent_builder = agent_builder
        self.environment_factory = environment_factory
        self.context = context
        self.max_active_agents = max_active_agents
        self.max_total_agents = max_total_agents
        self.per_agent_limits = per_agent_limits
        self.max_message_bytes = max_message_bytes
        self.max_inbox_bytes = max_inbox_bytes
        self.root_id = root_id.rstrip("/")
        self.harness = harness
        self._records: dict[str, AgentRecord] = {}
        self._resource_ids: set[str] = set()
        self._child_counts: dict[str, int] = {}
        self._message_sequence = 0
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def records(self) -> Mapping[str, AgentRecord]:
        return dict(self._records)

    def _role(self, agent_id: str) -> Role:
        if self.harness is None:
            return _LEGACY_ROLE
        return self.harness.role_of(agent_id, root_id=self.root_id)

    async def run(
        self, task: str, *, seeds: Sequence[tuple[str, str]] = ()
    ) -> AgentResult:
        """Run the team to the lead's answer.

        ``seeds`` are peers started beside the lead, as (id suffix, task)
        pairs. They are provisioned and running before the lead takes its
        first turn, which is what makes a fixed team a team rather than
        something the lead has to assemble.
        """

        if self._running:
            raise RuntimeError("orchestrator instances can run only once")
        task = _bounded_text(task, "root task", self.max_message_bytes)
        self._running = True
        root = await self._reserve(None, self.root_id)
        try:
            for suffix, seed_task in seeds:
                record = await self._reserve(
                    self.root_id, f"{self.root_id}/{suffix}"
                )
                await self._start(
                    record,
                    _bounded_text(seed_task, "seed task", self.max_message_bytes),
                )
            await self._start(root, task)
        except BaseException:
            # Peers are provisioned before the lead, so a failure here leaves
            # already-started containers or desktops with nothing to close
            # them. Tear the partial team down before giving up.
            await self._cancel_remaining(except_agent=self.root_id)
            raise
        assert root.task is not None
        result: AgentResult | None = None
        operation_error: BaseException | None = None
        try:
            await root.task
            if root.error is not None:
                raise root.error
            if root.result is None:
                raise RuntimeError("root agent stopped without a result")
            result = root.result
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await self._cancel_remaining(except_agent=self.root_id)
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors("orchestrator run", operation_error, cleanup_error)
        assert result is not None  # raise_lifecycle_errors raised on failure
        return result

    def _is_bounded_id(self, value: Any) -> bool:
        """Report whether ``value`` is a non-empty id within scheduler limits."""

        return (
            isinstance(value, str)
            and bool(value)
            and len(value.encode("utf-8")) <= self.max_message_bytes
        )

    def _check_ids(
        self, label: str, *values: Any, error: type[Exception] = ProtocolError
    ) -> None:
        if not all(self._is_bounded_id(value) for value in values):
            raise error(f"agent {label} exceeds scheduler limits")

    def _check_endpoints(
        self, kind: str, *values: Any, error: type[Exception] = ProtocolError
    ) -> None:
        if not all(isinstance(value, str) for value in values):
            raise error(f"agent {kind} endpoints must be strings")
        self._check_ids(f"{kind} endpoint", *values, error=error)

    async def spawn(self, parent_id: str, task: str) -> str:
        _require_str(parent_id, "agent parent id", non_empty=False, error=ProtocolError)
        task = _bounded_text(task, "agent spawn task", self.max_message_bytes)
        record = await self._reserve(parent_id)
        await self._start(record, task)
        return record.agent_id

    async def _reserve(
        self, parent_id: str | None, requested_id: str | None = None
    ) -> AgentRecord:
        async with self._lock:
            if parent_id is not None and parent_id not in self._records:
                raise ProtocolError(f"unknown parent agent {parent_id!r}")
            if parent_id is not None and self._records[parent_id].status not in _ACTIVE:
                raise ProtocolError("only a running agent may spawn descendants")
            active = sum(
                record.status in _ACTIVE for record in self._records.values()
            )
            if active >= self.max_active_agents:
                raise ProtocolError(
                    f"active agent limit reached ({self.max_active_agents})"
                )
            if len(self._records) >= self.max_total_agents:
                raise ProtocolError(
                    f"total agent limit reached ({self.max_total_agents})"
                )
            agent_id = requested_id
            if agent_id is not None and agent_id in self._records:
                raise ProtocolError(f"duplicate agent id {agent_id!r}")
            if agent_id is None:
                assert parent_id is not None
                number = self._child_counts.get(parent_id, 0) + 1
                self._child_counts[parent_id] = number
                agent_id = f"{parent_id}/{number}"
            if len(agent_id.encode("utf-8")) > self.max_message_bytes:
                raise ProtocolError("agent id exceeds scheduler limits")
            record = AgentRecord(agent_id=agent_id, parent_id=parent_id)
            self._records[agent_id] = record
            return record

    async def _start(self, record: AgentRecord, task: str) -> None:
        base: Any = None
        environment: CommunicationEnvironment | None = None
        identity: str | None = None
        identity_added = False
        try:
            raw = self.environment_factory(record.agent_id)
            if inspect.isawaitable(raw):
                provisioning = asyncio.ensure_future(raw)
                try:
                    base = await asyncio.shield(provisioning)
                except asyncio.CancelledError as cancelled:
                    try:
                        base = await provisioning
                    except BaseException as provisioning_error:
                        raise cancelled from provisioning_error
                    raise
            else:
                base = raw
            identity_hook = getattr(base, "resource_identity", None)
            if not callable(identity_hook):
                raise ValueError("environment must expose resource_identity")
            identity = identity_hook()
            if not isinstance(identity, str) or not identity:
                raise ValueError("environment resource_identity must be non-empty")
            async with self._lock:
                if identity in self._resource_ids:
                    raise ValueError("environment_factory reused a resource identity")
                self._resource_ids.add(identity)
                identity_added = True
            role = self._role(record.agent_id)
            environment = CommunicationEnvironment(
                base,
                self,
                record.agent_id,
                record.parent_id,
                actions=role.actions,
                expose_domain_tools=role.domain_tools,
            )
            built = self.agent_builder(record.agent_id, environment, self.context)
            agent = await built if inspect.isawaitable(built) else built
            if not isinstance(agent, MiniAgent):
                raise TypeError("agent_builder must return MiniAgent")
            # An agent that stays available reports each answer and waits; the
            # environment is finished once, when it is finally released.
            agent.finish_on_answer = not role.idle
            if agent.agent_id != record.agent_id or agent.context is not self.context:
                raise ValueError("agent_builder must use the supplied id and context")
            if self.per_agent_limits is not None:
                self.context.configure_agent(record.agent_id, self.per_agent_limits)
            record.environment = environment
            await self.context.trace.emit(
                "agent_spawned",
                agent_id=record.agent_id,
                role=agent.role,
                data={"parent_id": record.parent_id},
            )
            record.status = "running"
            record.task = asyncio.create_task(self._run_agent(record, agent, task))
        except BaseException as start_error:
            cleanup_error: BaseException | None = None
            close_target = environment if environment is not None else base
            if close_target is not None:
                try:
                    close = getattr(close_target, "close", None)
                    if not callable(close):
                        raise RuntimeError("environment must expose close")
                    await close()
                except BaseException as exc:
                    cleanup_error = exc
            if identity is not None and identity_added and cleanup_error is None:
                async with self._lock:
                    self._resource_ids.discard(identity)
            failure: BaseException = start_error
            if cleanup_error is not None:
                record.cleanup_error = cleanup_error
                failure = _combined_error("agent start", start_error, cleanup_error)
            record.error = failure
            clean_cancel = cleanup_error is None and isinstance(
                start_error, asyncio.CancelledError
            )
            record.status = "cancelled" if clean_cancel else "failed"
            trace_error: BaseException | None = None
            try:
                await self.context.trace.emit(
                    "agent_start_failed",
                    agent_id=record.agent_id,
                    role="scheduler",
                    data={
                        "error": type(start_error).__name__,
                        "cleanup_error": (
                            type(cleanup_error).__name__
                            if cleanup_error is not None
                            else None
                        ),
                    },
                )
            except BaseException as exc:
                trace_error = exc
            if trace_error is not None:
                failure = RuntimeError(
                    f"agent start failed ({type(failure).__name__}: {failure}); "
                    "failure trace also failed "
                    f"({type(trace_error).__name__}: {trace_error})"
                )
                record.error = failure
                record.status = "failed"
            if failure is start_error:
                raise
            raise failure from start_error

    async def _run_agent(
        self, record: AgentRecord, agent: MiniAgent, task: str
    ) -> None:
        environment = record.environment
        if environment is None:
            raise RuntimeError("started agent has no communication environment")
        result: AgentResult | None = None
        state: Any = None
        error: BaseException | None = None
        try:
            result = await agent.run(task)
            while self._role(record.agent_id).idle and not record.released:
                await self._deliver_answer(record, result)
                record.idle = True
                try:
                    messages = await self.read_messages(record.agent_id, wait=True)
                finally:
                    record.idle = False
                if record.released:
                    break
                result = await agent.resume(
                    "\n".join(message.content for message in messages)
                )
            state = await environment.export_state()
        except BaseException as exc:
            error = exc
        try:
            await environment.close()
        except BaseException as exc:
            record.cleanup_error = exc
            error = (
                exc if error is None
                else _combined_error("agent execution", error, exc)
            )
            result = None
            state = None

        record.error = error
        record.result = result
        record.state = state
        try:
            if error is None:
                assert result is not None
                record.status = "completed"
                await self.context.trace.emit(
                    "agent_completed",
                    agent_id=record.agent_id,
                    role=agent.role,
                    data={"steps": result.steps, "has_state": state is not None},
                )
            elif isinstance(error, asyncio.CancelledError):
                record.status = "cancelled"
                await self.context.trace.emit(
                    "agent_cancelled", agent_id=record.agent_id, role=agent.role
                )
            else:
                record.status = "failed"
                await self.context.trace.emit(
                    "agent_failed",
                    agent_id=record.agent_id,
                    role=agent.role,
                    data={"error": type(error).__name__, "message": str(error)},
                )
            # A released agent already reported this answer while idling, and
            # a delegated one hands its answer back through the tool result.
            reported = record.released or record.delegated
            if (
                record.parent_id is not None
                and record.status != "cancelled"
                and not (reported and error is None)
            ):
                await self._deliver_result(record)
        except BaseException as terminal_error:
            record.terminal_error = terminal_error
            error = (
                terminal_error if error is None
                else _combined_error("agent terminal handling", error, terminal_error)
            )
            record.error = error
            record.result = None
            record.state = None
            record.status = (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )

    async def _deliver_answer(self, record: AgentRecord, result: AgentResult) -> None:
        """Report an idle agent's answer without ending it."""

        if record.parent_id is not None:
            await self._report(record, result.answer, "result")

    async def _deliver_result(self, record: AgentRecord) -> None:
        assert record.parent_id is not None
        if record.result is not None:
            content = record.result.answer
            kind = "result"
        else:
            assert record.error is not None
            content = f"{type(record.error).__name__}: {record.error}"
            kind = "error"
        await self._report(record, content, kind)

    async def _report(self, record: AgentRecord, content: str, kind: str) -> None:
        assert record.parent_id is not None
        content = _clip(content, self.max_message_bytes)
        try:
            await self._deliver(record.agent_id, record.parent_id, content, kind=kind)
        except ProtocolError as exc:
            encoded = content.encode("utf-8")
            await self.context.trace.emit(
                "message_dropped",
                agent_id=record.agent_id,
                role="communication",
                data={
                    "recipient": record.parent_id,
                    "kind": kind,
                    "content_bytes": len(encoded),
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                    "reason": str(exc),
                },
            )

    async def _deliver(
        self, sender: str, recipient: str, content: str, *, kind: str
    ) -> MailboxMessage:
        encoded = content.encode("utf-8")
        byte_count = len(encoded)
        if byte_count > self.max_message_bytes:
            raise ProtocolError(f"message exceeds {self.max_message_bytes} bytes")
        async with self._lock:
            target = self._records.get(recipient)
            if target is None:
                raise ProtocolError(f"unknown target agent {recipient!r}")
            if target.status not in {"starting", "running"}:
                raise ProtocolError(f"target agent {recipient!r} is already terminal")
            if target.inbox_bytes + byte_count > self.max_inbox_bytes:
                raise ProtocolError(f"inbox limit reached for {recipient!r}")
            sequence = self._message_sequence + 1
            message = MailboxMessage(sequence, sender, recipient, content, kind)
            await self.context.trace.emit(
                "message_sent",
                agent_id=sender,
                role="communication",
                data={
                    "recipient": recipient,
                    "sequence": sequence,
                    "kind": kind,
                    "content_bytes": byte_count,
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                },
            )
            self._message_sequence = sequence
            target.inbox.append(message)
            target.inbox_bytes += byte_count
            target.inbox_event.set()
        return message

    async def send_message(self, sender: str, recipient: str, content: str) -> None:
        self._check_endpoints("message", sender, recipient)
        content = _bounded_text(content, "agent message", self.max_message_bytes)
        if sender not in self._records:
            raise ProtocolError(f"unknown sender agent {sender!r}")
        if self._records[sender].status not in _ACTIVE:
            raise ProtocolError("only a running agent may send messages")
        await self._deliver(sender, recipient, content, kind="message")

    async def read_messages(
        self, agent_id: str, *, wait: bool = False
    ) -> list[MailboxMessage]:
        self._check_ids("inbox id", agent_id)
        _require_bool(wait, "agent inbox wait", error=ProtocolError)
        while True:
            async with self._lock:
                record = self._records.get(agent_id)
                if record is None:
                    raise ProtocolError(f"unknown agent {agent_id!r}")
                # Being released is also a reason to stop waiting: otherwise a
                # woken idle agent finds an empty inbox and parks again.
                if record.inbox or not wait or record.released:
                    messages = list(record.inbox)
                    await self.context.trace.emit(
                        "messages_read",
                        agent_id=agent_id,
                        role="communication",
                        data={
                            "count": len(messages),
                            "sequences": [message.sequence for message in messages],
                        },
                    )
                    record.inbox.clear()
                    record.inbox_bytes = 0
                    record.inbox_event.clear()
                    return messages
                if record.status not in _ACTIVE:
                    raise ProtocolError("a terminal agent cannot wait for messages")
                event = record.inbox_event
                event.clear()
            await event.wait()

    async def wait(
        self, owner: str, agent_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        self._check_ids("wait owner", owner)
        if owner not in self._records:
            raise ProtocolError(f"unknown agent {owner!r}")
        if agent_ids is not None and (
            not isinstance(agent_ids, (list, tuple))
            or not all(self._is_bounded_id(agent_id) for agent_id in agent_ids)
            or len(agent_ids) > self.max_total_agents
        ):
            raise ProtocolError("agent wait targets must be a bounded string sequence")
        targets = (
            list(agent_ids)
            if agent_ids is not None
            else [
                agent_id
                for agent_id, record in self._records.items()
                if _is_descendant(owner, agent_id) and record.status in _ACTIVE
            ]
        )
        if owner in targets:
            raise ProtocolError("an agent cannot wait for itself")
        if len(targets) != len(set(targets)):
            raise ProtocolError("agent wait targets must be unique")
        unknown = [target for target in targets if target not in self._records]
        if unknown:
            raise ProtocolError(f"agent wait references unknown agents {unknown}")
        invalid = [target for target in targets if not _is_descendant(owner, target)]
        if invalid:
            raise ProtocolError("an agent may wait only for descendants")
        if self._records[owner].status not in _ACTIVE:
            raise ProtocolError("only a running agent may wait for descendants")
        tasks: list[asyncio.Task[None]] = []
        for target in targets:
            record = self._records[target]
            if record.task is not None and record.status in _ACTIVE:
                tasks.append(record.task)
        if tasks:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return [self._status(target) for target in targets]

    async def stop(self, owner: str, target: str) -> list[dict[str, Any]]:
        """Cancel one descendant subtree and await every affected cleanup."""

        self._check_endpoints("stop", owner, target)
        async with self._lock:
            owner_record = self._records.get(owner)
            target_record = self._records.get(target)
            if owner_record is None or target_record is None:
                raise ProtocolError("agent stop references an unknown agent")
            if owner_record.status not in _ACTIVE:
                raise ProtocolError("only a running agent may stop descendants")
            if not _is_descendant(owner, target):
                raise ProtocolError("an agent may stop only a descendant subtree")
            affected = [
                record for record in self._subtree(target) if record.status in _ACTIVE
            ]
            if not affected:
                raise ProtocolError("agent stop subtree has no running agents")
            affected_ids = [record.agent_id for record in affected]
            await self.context.trace.emit(
                "agent_stop_requested",
                agent_id=owner,
                role="communication",
                data={"target": target, "agents": affected_ids},
            )
            tasks: list[asyncio.Task[None]] = []
            for record in reversed(affected):
                if record.task is not None and not record.task.done():
                    tasks.append(record.task)
                    record.task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            finalized = self._subtree(target)
        still_running = [
            record.agent_id for record in finalized if record.status in _ACTIVE
        ]
        lifecycle_failure = _record_lifecycle_failure(finalized)
        if still_running or lifecycle_failure:
            await self.context.trace.emit(
                "agent_stop_failed",
                agent_id=owner,
                role="communication",
                data={
                    "target": target,
                    "agents": affected_ids,
                    "still_running": still_running,
                },
            )
            if still_running:
                raise RuntimeError(
                    "agent subtree did not stop: " + ", ".join(still_running)
                )
            raise RuntimeError(lifecycle_failure)
        statuses = [self._status(record.agent_id) for record in finalized]
        await self.context.trace.emit(
            "agent_subtree_stopped",
            agent_id=owner,
            role="communication",
            data={"target": target, "agents": statuses},
        )
        return statuses

    async def delegate(self, parent_id: str, task: str) -> dict[str, Any]:
        """Spawn one subagent, wait for it, and return its outcome.

        The caller blocks inside its own tool call, so the wait is bounded by
        the same wall clock as any other tool use and cannot outlive the run.
        """

        agent_id = await self.spawn(parent_id, task)
        record = self._records[agent_id]
        record.delegated = True
        if record.task is not None:
            await asyncio.wait([record.task])
        if record.result is not None:
            return {
                "agent_id": agent_id,
                "status": record.status,
                "answer": _clip(record.result.answer, self.max_message_bytes),
            }
        return {
            "agent_id": agent_id,
            "status": record.status,
            "error": f"{type(record.error).__name__}: {record.error}",
        }

    async def release(self, owner: str, target: str) -> dict[str, Any]:
        """Retire an idle descendant so its environment state can be adopted.

        ``stop`` cancels, and a cancelled agent exports no state, so without
        this a lead could never take back the workspace of a subagent that is
        working exactly as designed.
        """

        self._check_endpoints("release", owner, target, error=InvalidAction)
        owner_record = self._records.get(owner)
        record = self._records.get(target)
        if owner_record is None or record is None:
            raise InvalidAction("release requires known agents")
        if not _is_descendant(owner, target):
            raise InvalidAction("an agent may release only a descendant")
        if not record.idle:
            raise InvalidAction(f"agent {target!r} is not idle")
        record.released = True
        record.inbox_event.set()
        if record.task is not None:
            await asyncio.wait([record.task])
        return self._status(target)

    async def adopt(self, owner: str, target: str) -> None:
        self._check_endpoints("adopt", owner, target, error=InvalidAction)
        owner_record = self._records.get(owner)
        target_record = self._records.get(target)
        if owner_record is None or target_record is None:
            raise InvalidAction("agent adopt references an unknown agent")
        if not _is_descendant(owner, target):
            raise InvalidAction("an agent may adopt only a descendant's state")
        if owner_record.status not in _ACTIVE:
            raise InvalidAction("only a running agent may adopt descendant state")
        if target_record.status != "completed":
            raise InvalidAction("agent state can be adopted only after completion")
        if target_record.state is None:
            raise InvalidAction("completed agent has no adoptable state")
        if owner_record.environment is None:
            raise InvalidAction("running agent has no environment for state adoption")
        await owner_record.environment.adopt_state(target_record.state)
        async with self._lock:
            owner_record.adopted_from = target
            owner_record.adoption_history.append(target)
            adoption_number = len(owner_record.adoption_history)
        await self.context.trace.emit(
            "agent_state_adopted",
            agent_id=owner,
            role="communication",
            data={"source": target, "adoption_number": adoption_number},
        )

    def _status(self, agent_id: str) -> dict[str, Any]:
        record = self._records[agent_id]
        return {"agent_id": agent_id, "status": record.status}

    def _subtree(self, target: str) -> list[AgentRecord]:
        """Return ``target`` and its descendants ordered by agent id."""

        return sorted(
            (
                record
                for agent_id, record in self._records.items()
                if agent_id == target or _is_descendant(target, agent_id)
            ),
            key=lambda record: record.agent_id,
        )

    async def _cancel_remaining(self, *, except_agent: str) -> None:
        remaining = [
            record
            for agent_id, record in self._records.items()
            if agent_id != except_agent
        ]
        tasks = [
            record.task
            for record in remaining
            if record.task is not None and not record.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        failure = _record_lifecycle_failure(remaining)
        if failure:
            raise RuntimeError(failure)


def _clip(content: str, max_bytes: int) -> str:
    """Truncate to a byte budget, marking that something was cut."""

    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content
    marker = b"\n[truncated]"
    if len(marker) >= max_bytes:
        marker = b""
    kept = encoded[: max_bytes - len(marker)]
    return kept.decode("utf-8", errors="ignore") + marker.decode()


def _bounded_text(value: Any, name: str, max_bytes: int) -> str:
    value = _require_str(value, name, error=InvalidAction)
    if len(value.encode("utf-8")) > max_bytes:
        raise InvalidAction(f"{name} exceeds {max_bytes} bytes")
    return value


def _valid_agent_id(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or value.endswith("/"):
        return False
    return all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part) is not None and part not in {".", ".."}
        for part in value.split("/")[1:]
    )


def _is_descendant(owner: str, target: str) -> bool:
    return target.startswith(owner.rstrip("/") + "/")


def _only(call: ToolCall, *names: str) -> None:
    extra = sorted(set(call.arguments).difference(names))
    if extra:
        raise InvalidAction(f"unexpected arguments for agent action: {extra}")


async def _await_action(value: Awaitable[_T]) -> _T:
    """Translate scheduler contract rejections into model-repairable feedback."""

    try:
        return await value
    except InvalidAction:
        raise
    except ProtocolError as exc:
        raise InvalidAction(str(exc)) from exc


def _json_execution(value: Any) -> ToolExecution:
    return ToolExecution(output=json.dumps(value, sort_keys=True, allow_nan=False))


def _combined_error(
    label: str, operation_error: BaseException, cleanup_error: BaseException
) -> RuntimeError:
    return RuntimeError(
        f"{label} failed ({type(operation_error).__name__}: {operation_error}); "
        "environment cleanup also failed "
        f"({type(cleanup_error).__name__}: {cleanup_error})"
    )


def _record_lifecycle_failure(records: Sequence[AgentRecord]) -> str:
    """Describe terminal and cleanup failures across ``records``, if any."""

    parts: list[str] = []
    for label, attribute in (
        ("agent terminal handling failed", "terminal_error"),
        ("agent environment cleanup failed", "cleanup_error"),
    ):
        failures = "; ".join(
            f"{record.agent_id}: {type(error).__name__}: {error}"
            for record in records
            if (error := getattr(record, attribute)) is not None
        )
        if failures:
            parts.append(f"{label}: {failures}")
    return "; ".join(parts)


__all__ = [
    "AgentBuilder",
    "AgentRecord",
    "CommunicationEnvironment",
    "EnvironmentFactory",
    "MailboxMessage",
    "Orchestrator",
]
