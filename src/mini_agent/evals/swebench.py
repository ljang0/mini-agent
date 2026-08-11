from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import random
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..agent import MiniAgent
from ..orchestrator import Orchestrator
from ..runtime import RunContext
from ..types import (
    BudgetExceeded,
    BudgetLimits,
    Message,
    ModelRequest,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
)
from ..environments.swe import LocalProcessRunner, ProcessResult, ProcessRunner


RESULT_SCHEMA = "mini-agent-swebench-result-v1"
MANIFEST_SCHEMA = "mini-agent-swebench-run-v1"
OFFICIAL_PREDICTION_FIELDS = (
    "instance_id",
    "model_name_or_path",
    "model_patch",
)
_REUSABLE_STATUSES = {
    "completed",
    "agent_error",
    "budget_exhausted",
    "environment_error",
    "patch_error",
}
_OUTCOME_STATUSES = _REUSABLE_STATUSES | {"cancelled"}
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MINI_SWE_SUBMISSION = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_MINI_SWE_TEXT_PATTERNS = {
    "mini_swe_text": re.compile(
        r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL
    ),
    "mini_swe_backticks": re.compile(
        r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL
    ),
    "mini_swe_xml": re.compile(
        r"<mswea_bash_command>(.*?)</mswea_bash_command>", re.DOTALL
    ),
}


@dataclass(frozen=True)
class SWEbenchInstance:
    instance_id: str
    problem_statement: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class SWEbenchInstanceOutcome:
    status: str
    patch: bytes = b""
    answer: str = ""
    steps: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = ()
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SWEbenchInstanceCancelled(asyncio.CancelledError):
    def __init__(self, outcome: SWEbenchInstanceOutcome) -> None:
        super().__init__("SWE-bench instance cancelled after cleanup")
        self.outcome = outcome


@dataclass(frozen=True)
class SWEbenchBatchSummary:
    selected: int
    attempted: int
    skipped: int
    completed: int
    failed: int
    predictions_path: Path


InstanceWorker = Callable[[SWEbenchInstance], Awaitable[SWEbenchInstanceOutcome]]
SWEbenchMultiAgentBuilder = Callable[
    [SWEbenchInstance, str, Any, RunContext, str | None],
    MiniAgent | Awaitable[MiniAgent],
]
SWEbenchMultiEnvironmentFactory = Callable[
    [SWEbenchInstance, str, str | None], Any | Awaitable[Any]
]


def parse_mini_swe_action(content: str, response_parser: str) -> str:
    """Parse exactly one pinned mini-swe text action without executing it."""

    if not isinstance(content, str):
        raise ProtocolError("mini-swe text response must be a string")
    pattern = _MINI_SWE_TEXT_PATTERNS.get(response_parser)
    if pattern is None:
        raise ValueError(f"unsupported mini-swe response parser: {response_parser!r}")
    actions = [match.strip() for match in pattern.findall(content)]
    if len(actions) != 1:
        raise ProtocolError(
            f"{response_parser} expected exactly one action, found {len(actions)}"
        )
    if not actions[0]:
        raise ProtocolError(f"{response_parser} produced an empty bash action")
    return actions[0]


def _submission_from_messages(messages: Sequence[Message]) -> str | None:
    if not messages or messages[-1].role != "tool":
        return None
    for result in messages[-1].tool_results:
        lines = result.output.splitlines()
        if lines and lines[0].strip() == _MINI_SWE_SUBMISSION:
            submission = "\n".join(lines[1:]).strip()
            return submission or "Task complete."
    return None


def _is_direct_submission(command: str) -> bool:
    normalized = command.strip()
    return normalized in {
        f"echo {_MINI_SWE_SUBMISSION}",
        f"printf '{_MINI_SWE_SUBMISSION}\\n'",
        f'printf "{_MINI_SWE_SUBMISSION}\\n"',
    }


def _text_transcript(messages: Sequence[Message]) -> str:
    sections: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            label = "task"
            content = message.content
        elif message.role == "assistant":
            label = "assistant"
            content = message.content
        else:
            label = "observation"
            content = "\n".join(result.output for result in message.tool_results)
            if not content:
                content = message.content
        sections.append(f"<{label}>\n{content}\n</{label}>")
    return "\n\n".join(sections)


class MiniSWETextActionModel:
    """Adapt mini-swe text actions to ordinary MiniAgent tool calls.

    Pass either a provider-neutral ``model`` for tests/custom integrations or a
    provider ``backend``. Backend mode sends a deterministic full text transcript
    with no provider tool schema, so fake text actions never become unmatched
    provider tool continuations.
    """

    def __init__(
        self,
        *,
        response_parser: str,
        model: Any | None = None,
        backend: Any | None = None,
        max_output_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_id: str = "/root",
        role: str = "solver",
    ) -> None:
        if response_parser not in _MINI_SWE_TEXT_PATTERNS:
            raise ValueError(
                f"unsupported mini-swe response parser: {response_parser!r}"
            )
        if (model is None) == (backend is None):
            raise ValueError("select exactly one of model or backend")
        self.response_parser = response_parser
        self.model = model
        self.backend = backend
        self.max_output_tokens = max_output_tokens
        self.metadata = dict(metadata or {})
        self.agent_id = agent_id
        self.role = role
        self._action_number = 0

    async def _query_text(
        self, messages: Sequence[Message]
    ) -> ModelResponse:
        if self.model is not None:
            response = await self.model.query(messages, ())
        else:
            assert self.backend is not None
            system = "\n\n".join(
                message.content for message in messages if message.role == "system"
            )
            response = await self.backend.complete(
                ModelRequest(
                    agent_id=self.agent_id,
                    role=self.role,
                    prompt=_text_transcript(messages),
                    system=system,
                    max_output_tokens=self.max_output_tokens,
                    metadata=self.metadata,
                    tools=(),
                    tool_results=(),
                    continuation=None,
                )
            )
        if not isinstance(response, ModelResponse):
            raise ProtocolError("mini-swe text model must return ModelResponse")
        if response.tool_calls:
            raise ProtocolError(
                "mini-swe text model returned provider tool calls instead of text"
            )
        return response

    async def query(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        available = {tool.name for tool in tools}
        if available != {"bash"}:
            raise ProtocolError("mini-swe text adapters require exactly the bash tool")
        submission = _submission_from_messages(messages)
        if submission is not None:
            return ModelResponse(text=submission)
        response = await self._query_text(messages)
        command = parse_mini_swe_action(response.text, self.response_parser)
        if _is_direct_submission(command):
            return ModelResponse(
                text="Task complete.",
                usage=response.usage,
                provider_latency_seconds=response.provider_latency_seconds,
                raw=response.raw,
                continuation=None,
            )
        self._action_number += 1
        return ModelResponse(
            text=response.text,
            usage=response.usage,
            provider_latency_seconds=response.provider_latency_seconds,
            raw=response.raw,
            tool_calls=(
                ToolCall(
                    call_id=f"{self.response_parser}-{self._action_number}",
                    name="bash",
                    arguments={"command": command},
                ),
            ),
            continuation=None,
        )

    def provenance(self) -> Mapping[str, Any]:
        target = self.backend if self.backend is not None else self.model
        provenance = getattr(target, "provenance", None)
        value = dict(provenance()) if provenance is not None else {}
        return {
            **value,
            "response_parser": self.response_parser,
            "adapter": "mini_swe_text_action",
        }


def _json_bytes(value: Any, *, jsonl: bool = False) -> bytes:
    if jsonl:
        return (
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
                + "\n"
                for item in value
            )
        ).encode("utf-8")
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SWE-bench artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"SWE-bench artifact must be an object: {path}")
    return value


def load_swebench_jsonl(path: Path) -> tuple[SWEbenchInstance, ...]:
    path = path.expanduser().resolve()
    instances: list[SWEbenchInstance] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read SWE-bench JSONL dataset {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid SWE-bench JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"SWE-bench line {line_number} must be an object")
        instance_id = raw.get("instance_id")
        statement = raw.get("problem_statement")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(
                f"SWE-bench line {line_number} needs a non-empty instance_id"
            )
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance_id: {instance_id}")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(
                f"SWE-bench line {line_number} needs a problem_statement"
            )
        for image_key in ("image_name", "docker_image"):
            image = raw.get(image_key)
            if image is not None and (
                not isinstance(image, str) or not image or image.startswith("-")
            ):
                raise ValueError(
                    f"SWE-bench line {line_number} has invalid {image_key}"
                )
        seen.add(instance_id)
        instances.append(
            SWEbenchInstance(instance_id, statement, dict(raw))
        )
    if not instances:
        raise ValueError("SWE-bench JSONL dataset is empty")
    return tuple(instances)


def select_swebench_instances(
    instances: Sequence[SWEbenchInstance],
    *,
    instance_ids: Sequence[str] = (),
    filter_pattern: str | None = None,
    start: int | None = None,
    end: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> tuple[SWEbenchInstance, ...]:
    if start is not None and start < 0:
        raise ValueError("selection start must be non-negative")
    if end is not None and end < 0:
        raise ValueError("selection end must be non-negative")
    if start is not None and end is not None and end < start:
        raise ValueError("selection end must not precede start")
    selected = sorted(instances, key=lambda item: item.instance_id)
    if instance_ids:
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("instance_ids contains duplicates")
        requested = set(instance_ids)
        available = {item.instance_id for item in selected}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown SWE-bench instance IDs: {missing}")
        selected = [item for item in selected if item.instance_id in requested]
    if filter_pattern is not None:
        try:
            pattern = re.compile(filter_pattern)
        except re.error as exc:
            raise ValueError(f"invalid SWE-bench filter regex: {exc}") from exc
        selected = [item for item in selected if pattern.search(item.instance_id)]
    if shuffle:
        random.Random(seed).shuffle(selected)
    return tuple(selected[slice(start, end)])


async def run_mini_agent_instance(
    instance: SWEbenchInstance,
    *,
    model_factory: Callable[[SWEbenchInstance], Any],
    environment_factory: Callable[[SWEbenchInstance], Awaitable[Any]],
    system_prompt: str,
    max_steps: int,
    limits: BudgetLimits,
    capture_content: bool = False,
) -> SWEbenchInstanceOutcome:
    """Run an ordinary MiniAgent and capture its best-effort patch before cleanup."""

    context = RunContext(limits, capture_content=capture_content)
    environment: Any = None
    answer = ""
    steps = 0
    status = "completed"
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = {}
    patch = b""
    was_cancelled = False
    try:
        model = model_factory(instance)
        if inspect.isawaitable(model):
            model = await model
        try:
            environment = await environment_factory(instance)
        except asyncio.CancelledError:
            status = "cancelled"
            was_cancelled = True
        except Exception as exc:
            status = "environment_error"
            error_type = type(exc).__name__
            error_message = str(exc)
        if environment is None and status == "completed":
            status = "environment_error"
            error_type = "TypeError"
            error_message = "environment_factory returned None"
        if environment is not None:
            provenance = getattr(environment, "provenance", None)
            if callable(provenance):
                metadata["environment"] = dict(provenance())
            try:
                result = await MiniAgent(
                    model=model,
                    environment=environment,
                    system_prompt=system_prompt,
                    max_steps=max_steps,
                    context=context,
                ).run(instance.problem_statement)
                answer = result.answer
                steps = result.steps
            except asyncio.CancelledError:
                status = "cancelled"
                was_cancelled = True
            except BudgetExceeded as exc:
                status = "budget_exhausted"
                error_type = type(exc).__name__
                error_message = str(exc)
            except Exception as exc:
                status = "agent_error"
                error_type = type(exc).__name__
                error_message = str(exc)
    except asyncio.CancelledError:
        status = "cancelled"
        was_cancelled = True
    except Exception as exc:
        status = "agent_error"
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        if environment is not None:
            original_status = status
            try:
                patch_task = asyncio.create_task(environment.export_patch())
                try:
                    patch = await asyncio.shield(patch_task)
                except asyncio.CancelledError:
                    was_cancelled = True
                    status = "cancelled"
                    patch = await patch_task
            except Exception as exc:
                status = "patch_error"
                error_type = type(exc).__name__
                error_message = str(exc)
                metadata["status_before_patch_error"] = original_status
            try:
                close_task = asyncio.create_task(environment.close())
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    was_cancelled = True
                    status = "cancelled"
                    await close_task
            except Exception as exc:
                if status != "patch_error":
                    status = "environment_error"
                    error_type = type(exc).__name__
                    error_message = str(exc)
                else:
                    metadata["cleanup_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
    trace = tuple(asdict(event) for event in context.trace.events)
    outcome = SWEbenchInstanceOutcome(
        status=status,
        patch=patch,
        answer=answer,
        steps=steps,
        usage=asdict(context.ledger.usage),
        trace=trace,
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
    )
    if was_cancelled:
        raise SWEbenchInstanceCancelled(outcome)
    return outcome


class _RootPatchCaptureEnvironment:
    """Capture only the root workspace patch when Orchestrator closes it."""

    def __init__(self, base: Any, *, capture_patch: bool) -> None:
        self.base = base
        self.capture_patch = capture_patch
        self.patch = b""
        self.patch_error: BaseException | None = None
        self.cleanup_error: BaseException | None = None
        self._closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        return self.base.tools()

    async def initial_observation(self) -> Any:
        initial = getattr(self.base, "initial_observation", None)
        return None if initial is None else await initial()

    async def execute(self, action: ToolCall) -> Any:
        return await self.base.execute(action)

    async def finish(self) -> None:
        finish = getattr(self.base, "finish", None)
        if finish is not None:
            await finish()

    def resource_identity(self) -> str:
        return self.base.resource_identity()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.capture_patch:
            try:
                export_task = asyncio.create_task(self.base.export_patch())
                try:
                    self.patch = await asyncio.shield(export_task)
                except asyncio.CancelledError as exc:
                    self.patch_error = exc
                    try:
                        self.patch = await export_task
                    except BaseException as export_exc:
                        self.patch_error = export_exc
            except BaseException as exc:
                self.patch_error = exc
        try:
            close_task = asyncio.create_task(self.base.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                self.cleanup_error = exc
                try:
                    await close_task
                except BaseException as close_exc:
                    self.cleanup_error = close_exc
        except BaseException as exc:
            self.cleanup_error = exc
        if self.patch_error is not None:
            raise self.patch_error
        if self.cleanup_error is not None:
            raise self.cleanup_error


async def run_multi_agent_instance(
    instance: SWEbenchInstance,
    *,
    agent_builder: SWEbenchMultiAgentBuilder,
    environment_factory: SWEbenchMultiEnvironmentFactory,
    limits: BudgetLimits,
    per_agent_limits: BudgetLimits | None = None,
    max_agents: int = 4,
    allowed_child_profiles: Sequence[str] = (),
    root_profile: str | None = None,
    capture_content: bool = False,
) -> SWEbenchInstanceOutcome:
    """Run one SWE-bench task through the ordinary communication wrapper.

    The caller's builder selects downstream models, prompts, and step limits for
    the root or an allowlisted child profile. Every agent receives a fresh base
    environment; ``Orchestrator`` rejects repeated resource identities.
    """

    context = RunContext(limits, capture_content=capture_content)
    wrappers: dict[str, _RootPatchCaptureEnvironment] = {}
    selected_profiles: dict[str, str | None] = {}
    orchestrator: Orchestrator | None = None
    answer = ""
    steps = 0
    status = "completed"
    error_type: str | None = None
    error_message: str | None = None
    was_cancelled = False

    async def build_environment(
        agent_id: str, profile: str | None
    ) -> _RootPatchCaptureEnvironment:
        value = environment_factory(instance, agent_id, profile)
        base = await value if inspect.isawaitable(value) else value
        if base is None:
            raise TypeError("multi-agent environment_factory returned None")
        wrapper = _RootPatchCaptureEnvironment(
            base, capture_patch=agent_id == "/root"
        )
        wrappers[agent_id] = wrapper
        return wrapper

    async def build_agent(
        agent_id: str,
        environment: Any,
        shared: RunContext,
        profile: str | None,
    ) -> MiniAgent:
        selected_profiles[agent_id] = profile
        value = agent_builder(instance, agent_id, environment, shared, profile)
        built = await value if inspect.isawaitable(value) else value
        if not isinstance(built, MiniAgent):
            raise TypeError("multi-agent builder must return MiniAgent")
        return built

    try:
        orchestrator = Orchestrator(
            agent_builder=build_agent,
            environment_factory=build_environment,
            context=context,
            max_agents=max_agents,
            allowed_child_profiles=allowed_child_profiles,
            per_agent_limits=per_agent_limits,
        )
        result = await orchestrator.run(
            instance.problem_statement, profile=root_profile
        )
        answer = result.answer
        steps = result.steps
    except asyncio.CancelledError as exc:
        status = "cancelled"
        error_type = type(exc).__name__
        error_message = str(exc)
        was_cancelled = True
    except BudgetExceeded as exc:
        status = "budget_exhausted"
        error_type = type(exc).__name__
        error_message = str(exc)
    except Exception as exc:
        status = "agent_error"
        error_type = type(exc).__name__
        error_message = str(exc)

    root_environment = wrappers.get("/root")
    patch = b"" if root_environment is None else root_environment.patch
    metadata: dict[str, Any] = {
        "mode": "multi",
        "max_agents": max_agents,
        "root_profile": root_profile,
        "allowed_child_profiles": sorted(set(allowed_child_profiles)),
        "agents": {},
    }
    if orchestrator is not None:
        metadata["agents"] = {
            agent_id: {
                "parent_id": record.parent_id,
                "profile": selected_profiles.get(agent_id),
                "status": record.status,
                "environment": (
                    dict(provenance())
                    if callable(
                        provenance := getattr(wrappers[agent_id].base, "provenance", None)
                    )
                    else {}
                ),
            }
            for agent_id, record in sorted(orchestrator.records.items())
        }
    if root_environment is None and status != "cancelled":
        status = "environment_error"
        if error_type is None:
            error_type = "RuntimeError"
            error_message = "root environment was not created"
    elif root_environment is not None and root_environment.patch_error is not None:
        status = "patch_error"
        error_type = type(root_environment.patch_error).__name__
        error_message = str(root_environment.patch_error)
    elif root_environment is not None and root_environment.cleanup_error is not None:
        status = "environment_error"
        error_type = type(root_environment.cleanup_error).__name__
        error_message = str(root_environment.cleanup_error)

    outcome = SWEbenchInstanceOutcome(
        status=status,
        patch=patch,
        answer=answer,
        steps=steps,
        usage=asdict(context.ledger.usage),
        trace=tuple(asdict(event) for event in context.trace.events),
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
    )
    if was_cancelled:
        raise SWEbenchInstanceCancelled(outcome)
    return outcome


class SWEbenchBatchRunner:
    """Bounded deterministic scheduler and durable artifact writer."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name_or_path: str,
        worker: InstanceWorker,
        max_workers: int = 1,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("model_name_or_path must be a non-empty string")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool):
            raise ValueError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.output_dir = output_dir.expanduser().resolve()
        self.model_name_or_path = model_name_or_path
        self.worker = worker
        self.max_workers = max_workers
        self.manifest = dict(manifest or {})

    @staticmethod
    def _artifact_key(instance_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", instance_id).strip("-.")[:72]
        digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:12]
        return f"{safe or 'instance'}-{digest}"

    def _instance_dir(self, instance_id: str) -> Path:
        return self.output_dir / "instances" / self._artifact_key(instance_id)

    def _run_manifest(
        self, instances: Sequence[SWEbenchInstance]
    ) -> dict[str, Any]:
        identity = {
            "schema": MANIFEST_SCHEMA,
            "model_name_or_path": self.model_name_or_path,
            "instance_ids": sorted(item.instance_id for item in instances),
            "max_workers": self.max_workers,
            "config": self.manifest,
        }
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return {
            **identity,
            "fingerprint": hashlib.sha256(canonical).hexdigest(),
        }

    def _prepare(
        self, instances: Sequence[SWEbenchInstance], *, resume: bool
    ) -> None:
        expected = self._run_manifest(instances)
        manifest_path = self.output_dir / "manifest.json"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not resume:
                raise ValueError(f"output directory is not empty: {self.output_dir}")
            if not manifest_path.is_file():
                raise ValueError("cannot resume SWE-bench run without manifest.json")
            actual = _read_json(manifest_path)
            if actual.get("fingerprint") != expected["fingerprint"]:
                raise ValueError("SWE-bench resume manifest does not match this run")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(manifest_path, _json_bytes(expected))

    def _existing_result(self, instance_id: str) -> Mapping[str, Any] | None:
        path = self._instance_dir(instance_id) / "result.json"
        if not path.exists():
            return None
        result = _read_json(path)
        if result.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"unsupported SWE-bench result schema in {path}")
        if result.get("instance_id") != instance_id:
            raise ValueError(f"SWE-bench result instance mismatch in {path}")
        return result

    async def _persist(
        self, instance: SWEbenchInstance, outcome: SWEbenchInstanceOutcome
    ) -> None:
        if outcome.status not in _OUTCOME_STATUSES:
            raise ValueError(f"invalid SWE-bench outcome status: {outcome.status!r}")
        directory = self._instance_dir(instance.instance_id)
        directory.mkdir(parents=True, exist_ok=True)
        patch = bytes(outcome.patch)
        patch_valid_utf8 = True
        try:
            patch.decode("utf-8")
        except UnicodeDecodeError:
            patch_valid_utf8 = False
            outcome = replace(
                outcome,
                status="patch_error",
                error_type="UnicodeDecodeError",
                error_message="SWE-bench patch is not valid UTF-8",
            )
        await asyncio.to_thread(_atomic_write, directory / "patch.diff", patch)
        await asyncio.to_thread(
            _atomic_write, directory / "trace.jsonl", _json_bytes(outcome.trace, jsonl=True)
        )
        result = {
            "schema": RESULT_SCHEMA,
            "instance_id": instance.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "status": outcome.status,
            "answer": outcome.answer,
            "steps": outcome.steps,
            "usage": dict(outcome.usage),
            "error": (
                None
                if outcome.error_type is None and outcome.error_message is None
                else {
                    "type": outcome.error_type,
                    "message": outcome.error_message,
                }
            ),
            "patch": {
                "path": "patch.diff",
                "bytes": len(patch),
                "sha256": hashlib.sha256(patch).hexdigest(),
                "valid_utf8": patch_valid_utf8,
            },
            "trace": {"path": "trace.jsonl", "events": len(outcome.trace)},
            "metadata": dict(outcome.metadata),
        }
        await asyncio.to_thread(
            _atomic_write, directory / "result.json", _json_bytes(result)
        )
        (directory / "result.previous.json").unlink(missing_ok=True)
        (directory / ".running").unlink(missing_ok=True)

    def _prediction_for(
        self, instance: SWEbenchInstance, result: Mapping[str, Any]
    ) -> dict[str, str]:
        patch_metadata = result.get("patch")
        if not isinstance(patch_metadata, Mapping):
            raise ValueError(f"invalid patch metadata for {instance.instance_id}")
        patch_path = self._instance_dir(instance.instance_id) / "patch.diff"
        patch = patch_path.read_bytes()
        expected_size = patch_metadata.get("bytes")
        expected_hash = patch_metadata.get("sha256")
        if expected_size != len(patch) or expected_hash != hashlib.sha256(patch).hexdigest():
            raise ValueError(f"corrupt patch artifact for {instance.instance_id}")
        if patch_metadata.get("valid_utf8") is not True:
            model_patch = ""
        else:
            model_patch = patch.decode("utf-8")
        return {
            "instance_id": instance.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": model_patch,
        }

    async def _rebuild_predictions(
        self, instances: Sequence[SWEbenchInstance]
    ) -> tuple[list[dict[str, str]], list[Mapping[str, Any]]]:
        predictions: list[dict[str, str]] = []
        results: list[Mapping[str, Any]] = []
        for instance in sorted(instances, key=lambda item: item.instance_id):
            result = self._existing_result(instance.instance_id)
            if result is None:
                continue
            predictions.append(self._prediction_for(instance, result))
            results.append(result)
        await asyncio.to_thread(
            _atomic_write,
            self.output_dir / "predictions.jsonl",
            _json_bytes(predictions, jsonl=True),
        )
        return predictions, results

    async def run(
        self,
        instances: Sequence[SWEbenchInstance],
        *,
        resume: bool = False,
        retry_errors: bool = False,
    ) -> SWEbenchBatchSummary:
        if not instances:
            raise ValueError("SWE-bench run selected no instances")
        ids = [item.instance_id for item in instances]
        if len(ids) != len(set(ids)):
            raise ValueError("SWE-bench run contains duplicate instance IDs")
        ordered = tuple(sorted(instances, key=lambda item: item.instance_id))
        self._prepare(ordered, resume=resume)
        pending: list[SWEbenchInstance] = []
        skipped = 0
        for instance in ordered:
            existing = self._existing_result(instance.instance_id)
            running = (self._instance_dir(instance.instance_id) / ".running").exists()
            if running and existing is None:
                pending.append(instance)
                continue
            if running:
                (self._instance_dir(instance.instance_id) / ".running").unlink()
            if existing is None:
                pending.append(instance)
                continue
            status = existing.get("status")
            if status in _REUSABLE_STATUSES and not (
                retry_errors and status != "completed"
            ):
                skipped += 1
            else:
                pending.append(instance)

        queue: asyncio.Queue[SWEbenchInstance] = asyncio.Queue()
        for instance in pending:
            queue.put_nowait(instance)

        async def consume() -> None:
            while True:
                try:
                    instance = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                directory = self._instance_dir(instance.instance_id)
                directory.mkdir(parents=True, exist_ok=True)
                committed = directory / "result.json"
                if committed.exists():
                    os.replace(committed, directory / "result.previous.json")
                await asyncio.to_thread(
                    _atomic_write,
                    directory / ".running",
                    _json_bytes({"instance_id": instance.instance_id}),
                )
                try:
                    outcome = await self.worker(instance)
                    if not isinstance(outcome, SWEbenchInstanceOutcome):
                        raise TypeError("SWE-bench worker returned an invalid outcome")
                    if outcome.status not in _OUTCOME_STATUSES:
                        raise ValueError(
                            f"invalid SWE-bench outcome status: {outcome.status!r}"
                        )
                except asyncio.CancelledError as exc:
                    outcome = getattr(
                        exc,
                        "outcome",
                        SWEbenchInstanceOutcome(status="cancelled"),
                    )
                    persist = asyncio.create_task(
                        self._persist(instance, outcome)
                    )
                    try:
                        await asyncio.shield(persist)
                    except asyncio.CancelledError:
                        await persist
                    raise
                except Exception as exc:
                    outcome = SWEbenchInstanceOutcome(
                        status="agent_error",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                await self._persist(instance, outcome)
                queue.task_done()

        tasks = [
            asyncio.create_task(consume())
            for _ in range(min(self.max_workers, len(pending)))
        ]
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self._rebuild_predictions(ordered)
                raise
        predictions, results = await self._rebuild_predictions(ordered)
        completed = sum(result.get("status") == "completed" for result in results)
        summary = SWEbenchBatchSummary(
            selected=len(ordered),
            attempted=len(pending),
            skipped=skipped,
            completed=completed,
            failed=len(results) - completed,
            predictions_path=self.output_dir / "predictions.jsonl",
        )
        await asyncio.to_thread(
            _atomic_write,
            self.output_dir / "summary.json",
            _json_bytes(
                {
                    **asdict(summary),
                    "predictions_path": str(summary.predictions_path),
                    "predictions": len(predictions),
                }
            ),
        )
        return summary


def official_grader_argv(
    *,
    dataset_name: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    split: str = "test",
    instance_ids: Sequence[str] = (),
    namespace: str | None = "swebench",
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    for label, value in (
        ("dataset_name", dataset_name),
        ("split", split),
        ("python_executable", python_executable),
    ):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError(f"{label} must be a non-empty string")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, numbers, '.', '_', and '-'")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError("max_workers must be an integer")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    path = predictions_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"predictions file does not exist: {path}")
    argv = [
        python_executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        if not all(isinstance(item, str) and item for item in instance_ids):
            raise ValueError("instance_ids must contain non-empty strings")
        argv.extend(("--instance_ids", *instance_ids))
    if namespace is None:
        argv.extend(("--namespace", "none"))
    else:
        if not isinstance(namespace, str) or not namespace or "\x00" in namespace:
            raise ValueError("namespace must be a non-empty string or None")
        argv.extend(("--namespace", namespace))
    return tuple(argv)


async def run_official_grader(
    *,
    dataset_name: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    split: str = "test",
    instance_ids: Sequence[str] = (),
    namespace: str | None = "swebench",
    python_executable: str = sys.executable,
    cwd: Path | None = None,
    timeout_seconds: float = 24 * 60 * 60,
    max_output_bytes: int = 8 * 1024 * 1024,
    runner: ProcessRunner | None = None,
) -> ProcessResult:
    argv = official_grader_argv(
        dataset_name=dataset_name,
        predictions_path=predictions_path,
        run_id=run_id,
        max_workers=max_workers,
        split=split,
        instance_ids=instance_ids,
        namespace=namespace,
        python_executable=python_executable,
    )
    process_runner = runner or LocalProcessRunner()
    return await process_runner.run(
        argv,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )


__all__ = [
    "MANIFEST_SCHEMA",
    "MiniSWETextActionModel",
    "OFFICIAL_PREDICTION_FIELDS",
    "RESULT_SCHEMA",
    "SWEbenchBatchRunner",
    "SWEbenchBatchSummary",
    "SWEbenchInstance",
    "SWEbenchInstanceCancelled",
    "SWEbenchInstanceOutcome",
    "SWEbenchMultiAgentBuilder",
    "SWEbenchMultiEnvironmentFactory",
    "load_swebench_jsonl",
    "official_grader_argv",
    "parse_mini_swe_action",
    "run_mini_agent_instance",
    "run_multi_agent_instance",
    "run_official_grader",
    "select_swebench_instances",
]
