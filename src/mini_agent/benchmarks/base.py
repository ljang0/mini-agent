"""Shared evaluation scheduling and artifact contracts."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

from .._hash import canonical_bytes, harness_identity
from ..environments.base import complete_in_thread
from ..coordination import coordination_summary
from ..runtime import BudgetLedger, RunContext, TraceRecorder, redact_artifact
from ..specs import AgentSpecV1
from ..storage import atomic_json, read_committed_result, read_json_object

if TYPE_CHECKING:
    from ..agent import MiniAgent
from ..types import (
    BudgetLimits,
    Usage,
    _json_mapping,
    _require_bool,
    _require_callable,
    _require_finite_number,
    _require_int,
    _require_mapping,
    _require_positive_int,
    _require_str,
    _require_text,
    _require_tuple_of,
    strict_json_loads,
)


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    data: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _require_text(self.task_id, "benchmark task id")
        _require_text(self.prompt, "benchmark task prompt")
        if "\x00" in self.task_id:
            raise ValueError("benchmark task id cannot contain NUL")
        object.__setattr__(
            self, "data", _json_mapping(self.data, "benchmark task data")
        )


@dataclass(frozen=True)
class EvaluationOutcome:
    task_id: str
    status: str
    answer: str = ""
    score: float | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.task_id, "evaluation task id")
        # "blocked" is reserved in the mini-agent-eval-v2 summary shape;
        # no shipped worker currently produces it.
        if self.status not in {"completed", "failed", "blocked"}:
            raise ValueError(f"invalid evaluation status {self.status!r}")
        _require_text(self.answer, "evaluation answer", non_empty=False)
        if self.score is not None:
            _require_finite_number(self.score, "evaluation score")
        if self.error is not None:
            _require_text(self.error, "evaluation error", non_empty=False)
        object.__setattr__(
            self, "metadata", _json_mapping(self.metadata, "evaluation metadata")
        )


TaskWorker = Callable[[BenchmarkTask, RunContext, Path], Awaitable[EvaluationOutcome]]


class EvaluationRunner:
    """Run benchmark tasks with one ledger, trace, manifest, and resume contract."""

    def __init__(
        self,
        *,
        benchmark: str,
        tasks: Sequence[BenchmarkTask],
        output: Path,
        config: Mapping[str, Any],
        limits: BudgetLimits,
        max_workers: int = 1,
        capture_content: bool = False,
        secrets: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(benchmark, str) or not benchmark.strip() or not tasks:
            raise ValueError("benchmark and tasks must be non-empty")
        _require_positive_int(max_workers, "max_workers")
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark task ids must be unique")
        if not isinstance(output, Path):
            raise ValueError("evaluation output must be a Path")
        if not isinstance(limits, BudgetLimits):
            raise ValueError("evaluation limits must be BudgetLimits")
        _require_bool(capture_content, "capture_content")
        _require_tuple_of(secrets, str, "evaluation secrets")
        self.benchmark = benchmark
        self.tasks = tuple(tasks)
        self.output = output.expanduser().resolve()
        self.config = _json_mapping(config, "evaluation config")
        self.limits = limits
        self.max_workers = max_workers
        self.capture_content = capture_content
        self.secrets = secrets

    async def run(
        self, worker: TaskWorker, *, resume: bool = False
    ) -> Mapping[str, Any]:
        manifest = self._manifest()
        self._prepare(manifest, resume=resume)
        if resume:
            self._audit_resume_safety()
        ledger = BudgetLedger(self.limits)
        if resume:
            ledger.restore(self._restored_accounting())
        elapsed_offset, active_offset = self._resume_timing() if resume else (0.0, 0.0)
        trace = TraceRecorder(
            self.output / "trace.jsonl",
            secrets=self.secrets,
            elapsed_offset=elapsed_offset,
            backend_active_offset=active_offset,
        )
        context = RunContext(
            ledger=ledger, trace=trace, capture_content=self.capture_content
        )
        queue: asyncio.Queue[BenchmarkTask] = asyncio.Queue()
        for task in self.tasks:
            if not (resume and self._valid_result(task.task_id)):
                queue.put_nowait(task)

        async def consume() -> None:
            while True:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                self._instances_root()
                directory = self._instance(task.task_id)
                if directory.is_symlink() or (
                    directory.exists() and not directory.is_dir()
                ):
                    raise ValueError(
                        f"benchmark instance path must be a directory: {directory}"
                    )
                if directory.exists() and not self._valid_result(task.task_id):
                    self._validate_instance_path(directory)
                    await complete_in_thread(shutil.rmtree, directory)
                    self._instances_root()
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                directory.chmod(0o700)
                await trace.emit(
                    "benchmark_task_started",
                    agent_id=task_agent_root(task.task_id),
                    role="evaluation",
                    data={"benchmark": self.benchmark, "task_id": task.task_id},
                )
                # Where this task's events begin, so the coordination summary
                # scans its own slice rather than the whole run per task.
                first_event = len(trace.events)
                try:
                    outcome = await worker(task, context, directory)
                    if outcome.task_id != task.task_id:
                        raise ValueError("benchmark worker returned the wrong task id")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    outcome = EvaluationOutcome(
                        task_id=task.task_id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                prefix = task_agent_prefix(task.task_id)
                accounting = ledger.snapshot(prefix=prefix)
                coordination = coordination_summary(
                    (asdict(event) for event in trace.events[first_event:]),
                    ledger=ledger,
                    prefix=prefix,
                )
                outcome = replace(
                    outcome,
                    metadata={
                        **outcome.metadata,
                        "accounting": accounting,
                        "coordination": coordination,
                    },
                )
                redacted_outcome = redact_artifact(asdict(outcome), self.secrets)
                atomic_json(directory / "result.json", redacted_outcome)
                await trace.emit(
                    "benchmark_task_finished",
                    agent_id=task_agent_root(task.task_id),
                    role="evaluation",
                    data={
                        "benchmark": self.benchmark,
                        "task_id": task.task_id,
                        "status": outcome.status,
                        "score": outcome.score,
                    },
                )
                # A completion marker is the crash-recovery commit boundary.  Make
                # the usage, provider identity, and terminal trace evidence durable
                # before that marker can become durable.
                await trace.sync()
                result_bytes = (directory / "result.json").read_bytes()
                atomic_json(
                    directory / "completed.json",
                    {
                        "task_id": task.task_id,
                        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                    },
                )

        workers = [asyncio.create_task(consume()) for _ in range(self.max_workers)]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for running in workers:
                running.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        outcomes = [self._read_result(task.task_id) for task in self.tasks]
        scores = [
            float(outcome["score"])
            for outcome in outcomes
            if isinstance(outcome.get("score"), (int, float))
            and not isinstance(outcome.get("score"), bool)
        ]
        await trace.emit(
            "benchmark_run_finished",
            agent_id="/eval",
            role="evaluation",
            data={"benchmark": self.benchmark, "tasks": len(outcomes)},
        )
        elapsed_seconds = trace.events[-1].elapsed_seconds
        summary = {
            "benchmark": self.benchmark,
            "tasks": len(outcomes),
            "completed": sum(item.get("status") == "completed" for item in outcomes),
            "failed": sum(item.get("status") == "failed" for item in outcomes),
            "blocked": sum(item.get("status") == "blocked" for item in outcomes),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "model_calls": ledger.calls,
            "tool_calls": ledger.tool_calls,
            "tool_output_bytes": ledger.tool_output_bytes,
            "usage": asdict(ledger.usage),
            "elapsed_seconds": elapsed_seconds,
            "backend_active_union_seconds": trace.backend_active_union_seconds,
        }
        atomic_json(self.output / "summary.json", summary)
        return summary

    def _manifest(self) -> Mapping[str, Any]:
        value = {
            "schema": "mini-agent-eval-v2",
            "harness": harness_identity(),
            "benchmark": self.benchmark,
            "config": self.config,
            "limits": asdict(self.limits),
            "max_workers": self.max_workers,
            "capture_content": self.capture_content,
            "tasks": [
                {
                    "id": task.task_id,
                    "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                    "data_sha256": hashlib.sha256(
                        canonical_bytes(task.data)
                    ).hexdigest(),
                }
                for task in self.tasks
            ],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return {**value, "fingerprint": hashlib.sha256(encoded).hexdigest()}

    def _prepare(self, manifest: Mapping[str, Any], *, resume: bool) -> None:
        path = self.output / "manifest.json"
        if self.output.exists() and any(self.output.iterdir()):
            if not resume:
                raise ValueError(f"evaluation output is not empty: {self.output}")
            if not path.is_file() or read_json_object(path) != manifest:
                raise ValueError("resume manifest does not match this evaluation")
            self._instances_root()
            return
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.output.chmod(0o700)
        (self.output / "instances").mkdir(mode=0o700)
        atomic_json(path, manifest)

    def _instance(self, task_id: str) -> Path:
        safe = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return self.output / "instances" / safe

    def _instances_root(self) -> Path:
        root = self.output / "instances"
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise ValueError(
                "evaluation instances root must be an owned non-symlink directory"
            )
        return root

    def _validate_instance_path(self, directory: Path) -> None:
        root = self._instances_root()
        if (
            directory.parent != root
            or directory.is_symlink()
            or not directory.is_dir()
            or directory.resolve().parent != root
        ):
            raise ValueError(
                "benchmark instance must be an owned non-symlink directory"
            )

    def _valid_result(self, task_id: str) -> bool:
        try:
            self._instances_root()
            read_committed_result(self._instance(task_id), task_id)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def _read_result(self, task_id: str) -> Mapping[str, Any]:
        self._instances_root()
        return read_committed_result(self._instance(task_id), task_id)

    def _restored_accounting(self) -> Mapping[str, Any]:
        counts = dict.fromkeys(("model_calls", "tool_calls", "tool_output_bytes"), 0)
        usage = Usage()
        for task in self.tasks:
            if not self._valid_result(task.task_id):
                continue
            metadata = self._read_result(task.task_id).get("metadata", {})
            accounting = (
                metadata.get("accounting", {}) if isinstance(metadata, Mapping) else {}
            )
            _require_mapping(accounting, "resumed result accounting")
            for key in counts:
                counts[key] += _require_int(
                    accounting.get(key, 0), f"resumed {key}", minimum=0
                )
            raw_usage = _require_mapping(
                accounting.get("usage", {}), "resumed result usage accounting"
            )
            usage = usage + Usage(**dict(raw_usage))
        return {**counts, "usage": asdict(usage)}

    def _audit_resume_safety(self) -> None:
        """Refuse to repeat an external operation from an uncommitted task."""

        pending_prefixes = {
            task_agent_prefix(task.task_id): task.task_id
            for task in self.tasks
            if not self._valid_result(task.task_id)
        }
        trace_path = self.output / "trace.jsonl"
        if not trace_path.exists():
            return
        if trace_path.is_symlink() or not trace_path.is_file():
            raise ValueError("resume trace must be a regular non-symlink file")
        try:
            lines = trace_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("resume trace cannot be read safely") from exc
        for number, line in enumerate(lines, 1):
            if not line:
                raise ValueError(f"resume trace line {number} is empty")
            try:
                event = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"resume trace line {number} is invalid") from exc
            if not isinstance(event, Mapping):
                raise ValueError(f"resume trace line {number} is not an object")
            if event.get("event") not in {"model_call_started", "tool_call_started"}:
                continue
            agent_id = event.get("agent_id")
            if not isinstance(agent_id, str):
                raise ValueError(f"resume trace line {number} has an invalid agent id")
            for prefix, task_id in pending_prefixes.items():
                if agent_id == prefix or agent_id.startswith(prefix + "/"):
                    raise ValueError(
                        "cannot safely resume uncommitted task "
                        f"{task_id!r}: a model or tool operation may already "
                        "have started"
                    )

    def _resume_timing(self) -> tuple[float, float]:
        summary_path = self.output / "summary.json"
        if summary_path.is_file():
            summary = read_json_object(summary_path)
            return (
                _nonnegative_number(summary.get("elapsed_seconds", 0.0)),
                _nonnegative_number(summary.get("backend_active_union_seconds", 0.0)),
            )
        trace_path = self.output / "trace.jsonl"
        elapsed = 0.0
        starts: dict[str, list[float]] = {}
        intervals: list[tuple[float, float]] = []
        if trace_path.is_file():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = strict_json_loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(value, Mapping):
                    candidate = value.get("elapsed_seconds")
                    if (
                        isinstance(candidate, (int, float))
                        and not isinstance(candidate, bool)
                        and math.isfinite(float(candidate))
                    ):
                        timestamp = max(0.0, float(candidate))
                        elapsed = max(elapsed, timestamp)
                        event = value.get("event")
                        agent_id = value.get("agent_id", "")
                        if isinstance(agent_id, str):
                            if event == "model_call_started":
                                starts.setdefault(agent_id, []).append(timestamp)
                            elif event in {
                                "model_call_completed",
                                "model_call_failed",
                                "model_call_cancelled",
                            } and starts.get(agent_id):
                                start = starts[agent_id].pop(0)
                                intervals.append((start, max(start, timestamp)))
        for values in starts.values():
            intervals.extend((start, max(start, elapsed)) for start in values)
        return elapsed, _interval_union(intervals)


def spec_bound_agent(
    spec: AgentSpecV1,
    *,
    model: Any,
    environment: Any,
    context: RunContext,
    agent_id: str,
    system_prompt: str,
    max_steps: int,
) -> "MiniAgent":
    """Bind one worker agent against the manifest-recorded spec, fail closed."""

    if not isinstance(spec, AgentSpecV1):
        raise ValueError("agent_spec must be AgentSpecV1")
    if spec.system_prompt != system_prompt:
        raise ValueError("agent spec system_prompt does not match the prompt in use")
    if spec.max_steps != max_steps:
        raise ValueError("agent spec max_steps does not match the value in use")
    return spec.bind(
        model=model,
        environment=environment,
        model_id=spec.model,
        environment_id=spec.environment,
        context=context,
        agent_id=agent_id,
    )


def task_agent_builder(
    *,
    model_factory: Callable[[str], Any],
    system_prompt: str,
    max_steps: int,
    agent_spec: AgentSpecV1 | Mapping[str, AgentSpecV1] | None = None,
    harness: str = "single",
    root_id: str | None = None,
) -> Callable[[str, Any, RunContext], Awaitable["MiniAgent"]]:
    """Create the per-agent constructor every benchmark adapter shares.

    With ``agent_spec`` set (the CLI's production path), every agent —
    including orchestrator children — is constructed through
    :meth:`AgentSpecV1.bind`, so the fingerprint recorded in the evaluation
    manifest is enforced against the agent actually run.
    """

    async def agent_for(
        agent_id: str, environment: Any, shared: RunContext
    ) -> "MiniAgent":
        model = model_factory(agent_id)
        resolved = await model if inspect.isawaitable(model) else model
        spec, prompt = agent_spec, system_prompt
        if isinstance(spec, Mapping):
            # Roles differ in what they may do, so each binds against its own
            # spec; one spec could only describe one of them truthfully.
            from ..harnesses import load_harness

            selected = load_harness(harness)
            name = selected.role_name_of(agent_id, root_id=root_id or agent_id)
            spec = spec[name]
            prompt = system_prompt + selected.roles[name].prompt
        if spec is not None:
            return spec_bound_agent(
                spec,
                model=resolved,
                environment=environment,
                context=shared,
                agent_id=agent_id,
                system_prompt=prompt,
                max_steps=max_steps,
            )
        from ..agent import MiniAgent

        return MiniAgent(
            model=resolved,
            environment=environment,
            system_prompt=prompt,
            max_steps=max_steps,
            context=shared,
            agent_id=agent_id,
        )

    return agent_for


def task_agent_prefix(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return f"/eval/{digest}"


def task_agent_root(task_id: str) -> str:
    return task_agent_prefix(task_id) + "/root"


def owned_instance_artifacts(
    output: Path, name: str, *, label: str
) -> tuple[Path, Path, list[Path]]:
    """Return an evaluation's own ``instances/*/<name>`` artifacts.

    Official collectors must never follow a link out of the evaluation they
    were pointed at, so the root, the ``instances`` directory, every instance
    directory, and every artifact are each required to be non-symlink and
    owned by this evaluation.
    """

    expanded_root = output.expanduser()
    root = expanded_root.resolve()
    instances = root / "instances"
    if (
        expanded_root.is_symlink()
        or expanded_root.absolute() != root
        or not root.is_dir()
        or instances.is_symlink()
        or not instances.is_dir()
        or instances.resolve() != instances
    ):
        raise ValueError(
            f"{label} evaluation and instances must be non-symlink directories"
        )
    artifacts: list[Path] = []
    for path in sorted(instances.glob("*/" + name)):
        parent = path.parent
        if (
            parent.parent != instances
            or parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve().parent != instances
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                f"{label} artifact must be an owned regular instance artifact"
            )
        artifacts.append(path)
    return root, instances, artifacts


def raise_after_cleanup(
    label: str,
    operation_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> None:
    """Preserve the primary failure while making cleanup failures observable.

    Unlike :func:`mini_agent.environments.base.raise_lifecycle_errors`, a
    cleanup-only failure is wrapped in a labeled RuntimeError (and a cleanup
    CancelledError propagates unwrapped) so benchmark runners always report
    which stage's cleanup broke.
    """

    if operation_error is not None:
        if cleanup_error is not None:
            if isinstance(operation_error, asyncio.CancelledError):
                raise operation_error from cleanup_error
            raise RuntimeError(
                f"{label} failed ({_error_text(operation_error)}); "
                f"cleanup also failed ({_error_text(cleanup_error)})"
            ) from operation_error
        raise operation_error
    if cleanup_error is not None:
        if isinstance(cleanup_error, asyncio.CancelledError):
            raise cleanup_error
        raise RuntimeError(
            f"{label} cleanup failed: {_error_text(cleanup_error)}"
        ) from cleanup_error


async def shielded_create(
    creation: "asyncio.Future[Any]",
    *,
    label: str,
    close: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Await a resource's creation, closing it if this task is cancelled first.

    ``asyncio.shield`` lets an in-flight creation finish even when the awaiting
    task is cancelled, which means the resource it eventually produces still
    has to be closed. Both the original failure and any cleanup failure stay
    observable; on failure this never returns.
    """

    try:
        return await asyncio.shield(creation)
    except BaseException as operation_error:
        resource: Any = None
        cleanup_error: BaseException | None = None
        try:
            resource = await creation
        except BaseException as exc:
            if exc is not operation_error:
                cleanup_error = exc
        if resource is not None:
            try:
                await close(resource)
            except BaseException as exc:
                cleanup_error = combine_errors(cleanup_error, exc)
        raise_after_cleanup(label, operation_error, cleanup_error)
        raise AssertionError("raise_after_cleanup must raise")


def combine_errors(first: BaseException | None, second: BaseException) -> BaseException:
    """Combine secondary lifecycle failures without discarding either one."""

    if first is None:
        return second
    return RuntimeError(f"{_error_text(first)}; {_error_text(second)}")


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _nonnegative_number(value: Any) -> float:
    return _require_finite_number(value, "timing values", minimum=0)


def _interval_union(intervals: Sequence[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, finish in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, finish))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
    return sum(finish - start for start, finish in merged)


__all__ = [
    "BenchmarkTask",
    "EvaluationOutcome",
    "EvaluationRunner",
    "TaskWorker",
    "combine_errors",
    "raise_after_cleanup",
    "shielded_create",
    "task_agent_prefix",
    "task_agent_root",
]


@dataclass(frozen=True)
class TeamOptions:
    """How to staff and run a task, independent of which benchmark it is.

    These ten values always travel together and are always forwarded whole,
    so passing them as one keeps each benchmark's signature about that
    benchmark.
    """

    model_factory: Callable[[str], Any]
    system_prompt: str
    max_steps: int
    agent_spec: Any = None
    harness: str = "single"
    team_size: int | None = None
    multi_agent: bool = False
    max_active_agents: int = 4
    max_total_agents: int = 16
    per_agent_limits: Any = None

    def __post_init__(self) -> None:
        _require_callable(self.model_factory, "model_factory")
        _require_str(self.system_prompt, "system_prompt", non_empty=False)
        _require_positive_int(self.max_steps, "max_steps")
        _require_bool(self.multi_agent, "multi_agent")


async def run_benchmark_team(
    task: BenchmarkTask,
    context: RunContext,
    *,
    environment_factory: Any,
    options: TeamOptions,
    tolerate_failure: bool = False,
) -> Any:
    """Run one benchmark task as a team, whatever its topology.

    Resolving the harness, naming the root, and building the per-role agent
    builder is the same work for every benchmark. Keeping it here lets each
    adapter read as what it actually is: run the team, take the artifact.
    """

    from ..team import run_team, selected_harness

    selected = selected_harness(options.harness, options.multi_agent)
    root_id = task_agent_root(task.task_id)
    return await run_team(
        task.prompt,
        harness=selected,
        team_size=options.team_size,
        agent_builder=task_agent_builder(
            model_factory=options.model_factory,
            system_prompt=options.system_prompt,
            max_steps=options.max_steps,
            agent_spec=options.agent_spec,
            harness=selected,
            root_id=root_id,
        ),
        environment_factory=environment_factory,
        context=context,
        root_id=root_id,
        max_active_agents=options.max_active_agents,
        max_total_agents=options.max_total_agents,
        per_agent_limits=options.per_agent_limits,
        tolerate_failure=tolerate_failure,
    )
