"""BrowseComp-Plus task and artifact boundary.

The agent loop remains benchmark-independent.  This module owns only loading
questions, formatting the published task prompt, producing evaluator-compatible
records, resumable atomic writes, and the pinned upstream evaluator command.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..agent import MiniAgent
from ..environments.web import (
    BROWSECOMP_PLUS_REVISION,
    WebAccounting,
)
from ..runtime import RunContext
from ..types import BudgetExceeded, BudgetLimits
from ..orchestrator import Orchestrator


_SAFE_QUERY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_TASK_TEMPLATE_PATH = (
    Path(__file__).parents[1] / "profiles" / "web" / "browsecomp-task.md"
)
_REQUIRED_RECORD_FIELDS = frozenset(
    {"query_id", "tool_call_counts", "status", "retrieved_docids", "result"}
)
BROWSECOMP_RESULT_SCHEMA = "mini-agent-browsecomp-plus-result-v1"
BROWSECOMP_MANIFEST_SCHEMA = "mini-agent-browsecomp-plus-run-v1"
_REUSABLE_STATUSES = {
    "completed",
    "agent_error",
    "budget_exhausted",
    "environment_error",
}
_OUTCOME_STATUSES = _REUSABLE_STATUSES | {"cancelled"}


@dataclass(frozen=True)
class BrowseCompTask:
    """Inference-safe task; ground-truth answers are intentionally absent."""

    query_id: str
    query: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("BrowseComp query_id must be a non-empty string")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("BrowseComp query must be a non-empty string")

    def prompt(self) -> str:
        return format_browsecomp_task(self.query)


@dataclass(frozen=True)
class BrowseCompTaskSet:
    source: Path
    source_sha256: str
    source_format: str
    tasks: tuple[BrowseCompTask, ...]

    def manifest(self) -> dict[str, object]:
        ordered_ids = [task.query_id for task in self.tasks]
        return {
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "format": self.source_format,
            "task_count": len(self.tasks),
            "ordered_query_ids_sha256": hashlib.sha256(
                json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }


def _task(query_id: Any, query: Any, *, location: str) -> BrowseCompTask:
    if not isinstance(query_id, (str, int)) or isinstance(query_id, bool):
        raise ValueError(f"{location} query_id must be a string or integer")
    if not isinstance(query, str):
        raise ValueError(f"{location} query must be a string")
    try:
        return BrowseCompTask(str(query_id).strip(), query.strip())
    except ValueError as exc:
        raise ValueError(f"invalid {location}: {exc}") from exc


def load_browsecomp_tasks(
    path: Path, *, source_format: str | None = None
) -> BrowseCompTaskSet:
    """Load decrypted JSONL or the published two-column TSV.

    JSONL fields other than ``query_id`` and ``query`` are never retained.  In
    particular, an official ``answer`` field cannot enter the agent task object.
    """

    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"BrowseComp task file does not exist: {source}")
    selected_format = source_format or (
        "tsv" if source.suffix.casefold() == ".tsv" else "jsonl"
    )
    if selected_format not in {"jsonl", "tsv"}:
        raise ValueError("BrowseComp task format must be jsonl or tsv")

    tasks: list[BrowseCompTask] = []
    if selected_format == "jsonl":
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid BrowseComp JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"BrowseComp JSONL line {line_number} must be an object"
                )
            tasks.append(
                _task(
                    item.get("query_id"),
                    item.get("query"),
                    location=f"JSONL line {line_number}",
                )
            )
    else:
        with source.open(newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
                if not row or all(not field.strip() for field in row):
                    continue
                if len(row) != 2:
                    raise ValueError(
                        f"BrowseComp TSV line {line_number} must have exactly two columns"
                    )
                tasks.append(
                    _task(
                        row[0], row[1], location=f"TSV line {line_number}"
                    )
                )

    if not tasks:
        raise ValueError("BrowseComp task file contains no tasks")
    seen: set[str] = set()
    for task in tasks:
        if task.query_id in seen:
            raise ValueError(f"duplicate BrowseComp query_id {task.query_id!r}")
        seen.add(task.query_id)
    return BrowseCompTaskSet(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_format=selected_format,
        tasks=tuple(tasks),
    )


def browsecomp_task_template() -> str:
    if not _TASK_TEMPLATE_PATH.is_file():
        raise RuntimeError(f"packaged BrowseComp task template is missing: {_TASK_TEMPLATE_PATH}")
    return _TASK_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def format_browsecomp_task(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("BrowseComp query must be a non-empty string")
    template = browsecomp_task_template()
    if template.count("{Question}") != 1:
        raise RuntimeError("BrowseComp task template must contain one {Question} marker")
    return template.replace("{Question}", query.strip())


@dataclass(frozen=True)
class BrowseCompRunRecord:
    query_id: str
    tool_call_counts: Mapping[str, int]
    status: str
    retrieved_docids: tuple[str, ...]
    result: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_official_record(self.as_dict())

    @classmethod
    def from_answer(
        cls,
        task: BrowseCompTask,
        answer: str,
        accounting: WebAccounting,
        *,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> "BrowseCompRunRecord":
        if not isinstance(answer, str):
            raise ValueError("BrowseComp answer must be a string")
        result: tuple[Mapping[str, Any], ...] = (
            ({"type": "output_text", "output": answer},)
            if answer
            else ()
        )
        return cls(
            query_id=task.query_id,
            tool_call_counts=dict(accounting.tool_call_counts),
            status=status,
            retrieved_docids=tuple(sorted(set(accounting.retrieved_docids))),
            result=result,
            metadata=dict(metadata or {}),
            usage=dict(usage or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "query_id": self.query_id,
            "tool_call_counts": dict(sorted(self.tool_call_counts.items())),
            "status": self.status,
            "retrieved_docids": sorted(set(self.retrieved_docids)),
            "result": [dict(item) for item in self.result],
        }
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        if self.usage:
            value["usage"] = dict(self.usage)
        return value


@dataclass(frozen=True)
class BrowseCompTaskOutcome:
    """One worker result before it is committed to the run directory."""

    status: str
    answer: str = ""
    steps: int = 0
    accounting: WebAccounting = field(
        default_factory=lambda: WebAccounting({}, ())
    )
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = ()
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrowseCompBatchSummary:
    selected: int
    attempted: int
    skipped: int
    completed: int
    failed: int
    official_dir: Path
    summary_path: Path


TaskWorker = Callable[[BrowseCompTask], Awaitable[BrowseCompTaskOutcome]]
MultiAgentModelFactory = Callable[[BrowseCompTask, str, str | None], Any]
MultiAgentEnvironmentFactory = Callable[[BrowseCompTask, str, str | None], Any]
MultiAgentSystemPrompt = str | Callable[[str | None], str]
MultiAgentLimits = BudgetLimits | Callable[[str | None], BudgetLimits]
MultiAgentSteps = int | Callable[[str | None], int]


async def run_mini_agent_task(
    task: BrowseCompTask,
    *,
    model_factory: Callable[[BrowseCompTask], Any],
    environment_factory: Callable[[BrowseCompTask], Any],
    system_prompt: str,
    max_steps: int,
    limits: BudgetLimits,
    capture_content: bool = False,
) -> BrowseCompTaskOutcome:
    """Run one task through the ordinary domain-independent ``MiniAgent``."""

    context = RunContext(limits, capture_content=capture_content)
    environment: Any = None
    answer = ""
    steps = 0
    status = "completed"
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = {}
    accounting = WebAccounting({}, ())
    was_cancelled = False
    try:
        model = model_factory(task)
        if inspect.isawaitable(model):
            model = await model
        environment = environment_factory(task)
        if inspect.isawaitable(environment):
            environment = await environment
        if environment is None:
            raise TypeError("environment_factory returned None")
        try:
            result = await MiniAgent(
                model=model,
                environment=environment,
                system_prompt=system_prompt,
                max_steps=max_steps,
                context=context,
                agent_id=f"/browsecomp/{task.query_id}",
            ).run(task.prompt())
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
        status = "environment_error" if environment is None else "agent_error"
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        if environment is not None:
            try:
                measured = environment.accounting()
                if not isinstance(measured, WebAccounting):
                    raise TypeError("web environment accounting() returned an invalid value")
                accounting = measured
            except Exception as exc:
                if status == "completed":
                    status = "environment_error"
                    error_type = type(exc).__name__
                    error_message = str(exc)
                else:
                    metadata["accounting_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            try:
                await environment.close()
            except Exception as exc:
                if status == "completed":
                    status = "environment_error"
                    error_type = type(exc).__name__
                    error_message = str(exc)
                else:
                    metadata["cleanup_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
    outcome = BrowseCompTaskOutcome(
        status=status,
        answer=answer,
        steps=steps,
        accounting=accounting,
        usage=asdict(context.ledger.usage),
        trace=tuple(asdict(event) for event in context.trace.events),
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
    )
    if was_cancelled:
        raise asyncio.CancelledError
    return outcome


async def run_multi_agent_task(
    task: BrowseCompTask,
    *,
    model_factory: MultiAgentModelFactory,
    environment_factory: MultiAgentEnvironmentFactory,
    system_prompt: MultiAgentSystemPrompt,
    max_steps: MultiAgentSteps,
    limits: BudgetLimits,
    max_agents: int,
    per_agent_limits: MultiAgentLimits | None = None,
    allowed_child_profiles: Sequence[str] = (),
    capture_content: bool = False,
) -> BrowseCompTaskOutcome:
    """Run one BrowseComp task through the thin communication orchestrator.

    Every participant is an ordinary ``MiniAgent`` with its own web environment
    wrapper. Callers may safely close over one immutable retrieval backend when
    constructing those wrappers. Only retrieval accounting is aggregated into
    the evaluator record; communication remains visible in the shared trace.
    """

    context = RunContext(limits, capture_content=capture_content)
    environments: dict[str, Any] = {}
    result: Any = None
    status = "completed"
    error_type: str | None = None
    error_message: str | None = None
    was_cancelled = False
    metadata: dict[str, Any] = {
        "mode": "multi",
        "max_agents": max_agents,
        "allowed_child_profiles": list(allowed_child_profiles),
    }

    async def build_environment(agent_id: str, profile: str | None) -> Any:
        value = environment_factory(task, agent_id, profile)
        environment = await value if inspect.isawaitable(value) else value
        if environment is None:
            raise TypeError("multi-agent environment_factory returned None")
        if any(environment is existing for existing in environments.values()):
            raise ValueError("multi-agent web environments must be separate wrappers")
        accounting = getattr(environment, "accounting", None)
        if not callable(accounting):
            await environment.close()
            raise TypeError("multi-agent web environment requires accounting()")
        environments[agent_id] = environment
        return environment

    async def build_agent(
        agent_id: str,
        environment: Any,
        shared_context: RunContext,
        profile: str | None,
    ) -> MiniAgent:
        value = model_factory(task, agent_id, profile)
        model = await value if inspect.isawaitable(value) else value
        prompt = system_prompt(profile) if callable(system_prompt) else system_prompt
        if not isinstance(prompt, str):
            raise TypeError("multi-agent system prompt resolver must return a string")
        selected_limits = (
            per_agent_limits(profile)
            if callable(per_agent_limits)
            else per_agent_limits
        )
        if selected_limits is not None:
            if not isinstance(selected_limits, BudgetLimits):
                raise TypeError(
                    "multi-agent per-agent limit resolver must return BudgetLimits"
                )
            shared_context.configure_agent(agent_id, selected_limits)
        selected_steps = max_steps(profile) if callable(max_steps) else max_steps
        if (
            not isinstance(selected_steps, int)
            or isinstance(selected_steps, bool)
            or selected_steps < 1
        ):
            raise TypeError("multi-agent step resolver must return a positive integer")
        return MiniAgent(
            model=model,
            environment=environment,
            system_prompt=prompt,
            max_steps=selected_steps,
            context=shared_context,
            agent_id=agent_id,
        )

    orchestrator = Orchestrator(
        agent_builder=build_agent,
        environment_factory=build_environment,
        context=context,
        max_agents=max_agents,
        allowed_child_profiles=allowed_child_profiles,
        per_agent_limits=None,
    )
    try:
        result = await orchestrator.run(task.prompt())
    except asyncio.CancelledError:
        status = "cancelled"
        was_cancelled = True
    except BudgetExceeded as exc:
        status = "budget_exhausted"
        error_type = type(exc).__name__
        error_message = str(exc)
    except Exception as exc:
        status = "environment_error" if not environments else "agent_error"
        error_type = type(exc).__name__
        error_message = str(exc)

    counts: Counter[str] = Counter()
    retrieved_docids: set[str] = set()
    accounting_error: Exception | None = None
    for agent_id in sorted(environments):
        try:
            measured = environments[agent_id].accounting()
            if not isinstance(measured, WebAccounting):
                raise TypeError("web environment accounting() returned an invalid value")
            counts.update(measured.tool_call_counts)
            retrieved_docids.update(measured.retrieved_docids)
        except Exception as exc:
            accounting_error = exc
            break
    if accounting_error is not None:
        if status == "completed":
            status = "environment_error"
            error_type = type(accounting_error).__name__
            error_message = str(accounting_error)
        else:
            metadata["accounting_error"] = {
                "type": type(accounting_error).__name__,
                "message": str(accounting_error),
            }

    metadata["agents"] = {
        agent_id: {
            "parent_id": record.parent_id,
            "status": record.status,
            "budget": dict(context.ledger.agent_snapshot(agent_id)),
        }
        for agent_id, record in sorted(orchestrator.records.items())
    }
    metadata["agent_count"] = len(orchestrator.records)
    outcome = BrowseCompTaskOutcome(
        status=status,
        answer="" if result is None else result.answer,
        steps=0 if result is None else result.steps,
        accounting=WebAccounting(
            dict(sorted(counts.items())), tuple(sorted(retrieved_docids))
        ),
        usage=asdict(context.ledger.usage),
        trace=tuple(asdict(event) for event in context.trace.events),
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
    )
    if was_cancelled:
        raise asyncio.CancelledError
    return outcome


def validate_official_record(value: Mapping[str, Any]) -> None:
    """Validate the fields consumed by pinned ``evaluate_run.py``."""

    if not isinstance(value, Mapping):
        raise ValueError("BrowseComp run record must be an object")
    missing = _REQUIRED_RECORD_FIELDS - set(value)
    if missing:
        raise ValueError(f"BrowseComp run record is missing {sorted(missing)}")
    query_id = value.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("BrowseComp run query_id must be a non-empty string")
    counts = value.get("tool_call_counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for name, count in counts.items()
    ):
        raise ValueError("BrowseComp tool_call_counts must map names to non-negative ints")
    status = value.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("BrowseComp status must be a non-empty string")
    docids = value.get("retrieved_docids")
    if not isinstance(docids, list) or any(
        not isinstance(docid, str) or not docid for docid in docids
    ):
        raise ValueError("BrowseComp retrieved_docids must be a string list")
    result = value.get("result")
    if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
        raise ValueError("BrowseComp result must be an object list")
    if status == "completed":
        if not result or result[-1].get("type") != "output_text":
            raise ValueError("completed BrowseComp result must end in output_text")
        if not isinstance(result[-1].get("output"), str):
            raise ValueError("BrowseComp output_text output must be a string")


class BrowseCompArtifactStore:
    """Atomic, resumable store whose ``official`` directory contains only runs."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.official_dir = self.root / "official"

    @staticmethod
    def _filename(query_id: str) -> str:
        if _SAFE_QUERY_ID.fullmatch(query_id):
            return f"{query_id}.json"
        digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
        return f"query-{digest}.json"

    def path_for(self, query_id: str) -> Path:
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("BrowseComp query_id must be a non-empty string")
        return self.official_dir / self._filename(query_id)

    def write(
        self, record: BrowseCompRunRecord, *, overwrite: bool = False
    ) -> Path:
        payload = record.as_dict()
        validate_official_record(payload)
        destination = self.path_for(record.query_id)
        self.official_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"BrowseComp record already exists: {destination}")
        serialized = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.official_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        return destination

    def read(self, query_id: str) -> Mapping[str, Any] | None:
        path = self.path_for(query_id)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_official_record(value)
        if value["query_id"] != query_id:
            raise ValueError(f"BrowseComp record identity mismatch in {path}")
        return value

    def pending(
        self,
        tasks: Sequence[BrowseCompTask],
        *,
        rerun_failed: bool = False,
    ) -> tuple[BrowseCompTask, ...]:
        pending: list[BrowseCompTask] = []
        for task in tasks:
            existing = self.read(task.query_id)
            if existing is None or (
                rerun_failed and existing.get("status") != "completed"
            ):
                pending.append(task)
        return tuple(pending)


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
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BrowseComp artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"BrowseComp artifact must be an object: {path}")
    return value


class BrowseCompBatchRunner:
    """Bounded deterministic scheduler with resumable evaluator-ready output."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name_or_path: str,
        worker: TaskWorker,
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
        self.artifacts = BrowseCompArtifactStore(self.output_dir)

    @staticmethod
    def _artifact_key(query_id: str) -> str:
        if _SAFE_QUERY_ID.fullmatch(query_id):
            return query_id
        return "query-" + hashlib.sha256(query_id.encode("utf-8")).hexdigest()

    def _instance_dir(self, query_id: str) -> Path:
        return self.output_dir / "instances" / self._artifact_key(query_id)

    def _run_manifest(self, tasks: Sequence[BrowseCompTask]) -> dict[str, Any]:
        task_identity = [
            {
                "query_id": task.query_id,
                "query_sha256": hashlib.sha256(task.query.encode("utf-8")).hexdigest(),
            }
            for task in sorted(tasks, key=lambda item: item.query_id)
        ]
        identity = {
            "schema": BROWSECOMP_MANIFEST_SCHEMA,
            "benchmark_revision": BROWSECOMP_PLUS_REVISION,
            "model_name_or_path": self.model_name_or_path,
            "tasks": task_identity,
            "max_workers": self.max_workers,
            "config": self.manifest,
        }
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return {**identity, "fingerprint": hashlib.sha256(canonical).hexdigest()}

    def _prepare(self, tasks: Sequence[BrowseCompTask], *, resume: bool) -> None:
        expected = self._run_manifest(tasks)
        manifest_path = self.output_dir / "manifest.json"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not resume:
                raise ValueError(f"output directory is not empty: {self.output_dir}")
            if not manifest_path.is_file():
                raise ValueError("cannot resume BrowseComp run without manifest.json")
            actual = _read_json(manifest_path)
            if actual.get("fingerprint") != expected["fingerprint"]:
                raise ValueError("BrowseComp resume manifest does not match this run")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(manifest_path, _json_bytes(expected))

    def _existing_result(self, task: BrowseCompTask) -> Mapping[str, Any] | None:
        path = self._instance_dir(task.query_id) / "result.json"
        if not path.exists():
            return None
        result = _read_json(path)
        if result.get("schema") != BROWSECOMP_RESULT_SCHEMA:
            raise ValueError(f"unsupported BrowseComp result schema in {path}")
        if result.get("query_id") != task.query_id:
            raise ValueError(f"BrowseComp result query mismatch in {path}")
        trace = result.get("trace")
        if not isinstance(trace, Mapping):
            raise ValueError(f"invalid BrowseComp trace metadata in {path}")
        trace_path = path.parent / "trace.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"missing BrowseComp trace artifact for {task.query_id}")
        trace_bytes = trace_path.read_bytes()
        if trace.get("bytes") != len(trace_bytes) or trace.get("sha256") != hashlib.sha256(
            trace_bytes
        ).hexdigest():
            raise ValueError(f"corrupt BrowseComp trace artifact for {task.query_id}")
        official = self.artifacts.read(task.query_id)
        if official is None or official.get("status") != result.get("status"):
            raise ValueError(f"missing or mismatched official record for {task.query_id}")
        official_metadata = result.get("official")
        if not isinstance(official_metadata, Mapping):
            raise ValueError(f"invalid official metadata for {task.query_id}")
        official_path = self.artifacts.path_for(task.query_id)
        official_bytes = official_path.read_bytes()
        if official_metadata.get("bytes") != len(
            official_bytes
        ) or official_metadata.get("sha256") != hashlib.sha256(
            official_bytes
        ).hexdigest():
            raise ValueError(f"corrupt official record for {task.query_id}")
        return result

    async def _persist(
        self, task: BrowseCompTask, outcome: BrowseCompTaskOutcome
    ) -> None:
        if outcome.status not in _OUTCOME_STATUSES:
            raise ValueError(f"invalid BrowseComp outcome status: {outcome.status!r}")
        if not isinstance(outcome.accounting, WebAccounting):
            raise ValueError("BrowseComp outcome accounting must be WebAccounting")
        directory = self._instance_dir(task.query_id)
        directory.mkdir(parents=True, exist_ok=True)
        trace_bytes = _json_bytes(outcome.trace, jsonl=True)
        await asyncio.to_thread(_atomic_write, directory / "trace.jsonl", trace_bytes)
        error = (
            None
            if outcome.error_type is None and outcome.error_message is None
            else {"type": outcome.error_type, "message": outcome.error_message}
        )
        official_metadata = {
            "model_name_or_path": self.model_name_or_path,
            "steps": outcome.steps,
            **dict(outcome.metadata),
        }
        if error is not None:
            official_metadata["error"] = error
        official_record = BrowseCompRunRecord.from_answer(
            task,
            outcome.answer,
            outcome.accounting,
            status=outcome.status,
            metadata=official_metadata,
            usage=outcome.usage,
        )
        official_path = await asyncio.to_thread(
            self.artifacts.write, official_record, overwrite=True
        )
        official_bytes = official_path.read_bytes()
        result = {
            "schema": BROWSECOMP_RESULT_SCHEMA,
            "query_id": task.query_id,
            "model_name_or_path": self.model_name_or_path,
            "status": outcome.status,
            "answer": outcome.answer,
            "steps": outcome.steps,
            "usage": dict(outcome.usage),
            "accounting": outcome.accounting.as_dict(),
            "error": error,
            "trace": {
                "path": "trace.jsonl",
                "events": len(outcome.trace),
                "bytes": len(trace_bytes),
                "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            },
            "official": {
                "path": str(official_path.relative_to(self.output_dir)),
                "bytes": len(official_bytes),
                "sha256": hashlib.sha256(official_bytes).hexdigest(),
            },
            "metadata": dict(outcome.metadata),
        }
        await asyncio.to_thread(
            _atomic_write, directory / "result.json", _json_bytes(result)
        )
        (directory / ".running").unlink(missing_ok=True)

    async def _write_summary(
        self,
        tasks: Sequence[BrowseCompTask],
        *,
        attempted: int,
        skipped: int,
    ) -> BrowseCompBatchSummary:
        results = [
            result
            for task in sorted(tasks, key=lambda item: item.query_id)
            if (result := self._existing_result(task)) is not None
        ]
        completed = sum(result.get("status") == "completed" for result in results)
        summary_path = self.output_dir / "summary.json"
        summary = BrowseCompBatchSummary(
            selected=len(tasks),
            attempted=attempted,
            skipped=skipped,
            completed=completed,
            failed=len(results) - completed,
            official_dir=self.artifacts.official_dir,
            summary_path=summary_path,
        )
        statuses: dict[str, int] = {}
        for result in results:
            status = str(result.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        await asyncio.to_thread(
            _atomic_write,
            summary_path,
            _json_bytes(
                {
                    **asdict(summary),
                    "official_dir": str(summary.official_dir),
                    "summary_path": str(summary.summary_path),
                    "persisted": len(results),
                    "statuses": dict(sorted(statuses.items())),
                }
            ),
        )
        return summary

    async def run(
        self,
        tasks: Sequence[BrowseCompTask],
        *,
        resume: bool = False,
        retry_errors: bool = False,
    ) -> BrowseCompBatchSummary:
        if not tasks:
            raise ValueError("BrowseComp run selected no tasks")
        query_ids = [task.query_id for task in tasks]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("BrowseComp run contains duplicate query IDs")
        ordered = tuple(sorted(tasks, key=lambda item: item.query_id))
        self._prepare(ordered, resume=resume)
        pending: list[BrowseCompTask] = []
        skipped = 0
        for task in ordered:
            existing = self._existing_result(task)
            if existing is None:
                pending.append(task)
                continue
            (self._instance_dir(task.query_id) / ".running").unlink(missing_ok=True)
            status = existing.get("status")
            if status in _REUSABLE_STATUSES and not (
                retry_errors and status != "completed"
            ):
                skipped += 1
            else:
                pending.append(task)

        queue: asyncio.Queue[BrowseCompTask] = asyncio.Queue()
        for task in pending:
            queue.put_nowait(task)

        async def consume() -> None:
            while True:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                directory = self._instance_dir(task.query_id)
                directory.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    _atomic_write,
                    directory / ".running",
                    _json_bytes({"query_id": task.query_id}),
                )
                try:
                    outcome = await self.worker(task)
                    if not isinstance(outcome, BrowseCompTaskOutcome):
                        raise TypeError("BrowseComp worker returned an invalid outcome")
                except asyncio.CancelledError as exc:
                    outcome = getattr(
                        exc,
                        "outcome",
                        BrowseCompTaskOutcome(status="cancelled"),
                    )
                    persist = asyncio.create_task(
                        self._persist(task, outcome)
                    )
                    try:
                        await asyncio.shield(persist)
                    except asyncio.CancelledError:
                        await persist
                    raise
                except Exception as exc:
                    outcome = BrowseCompTaskOutcome(
                        status="agent_error",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                await self._persist(task, outcome)
                queue.task_done()

        workers = [
            asyncio.create_task(consume())
            for _ in range(min(self.max_workers, len(pending)))
        ]
        if workers:
            try:
                await asyncio.gather(*workers)
            except BaseException:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                await self._write_summary(
                    ordered, attempted=len(pending), skipped=skipped
                )
                raise
        return await self._write_summary(
            ordered, attempted=len(pending), skipped=skipped
        )


def preflight_official_directory(input_dir: Path) -> tuple[Mapping[str, Any], ...]:
    directory = input_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"BrowseComp official run directory is missing: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("BrowseComp official run directory contains no JSON records")
    records: list[Mapping[str, Any]] = []
    query_ids: set[str] = set()
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_official_record(value)
        query_id = value["query_id"]
        if query_id in query_ids:
            raise ValueError(f"duplicate BrowseComp record query_id {query_id!r}")
        query_ids.add(query_id)
        records.append(value)
    return tuple(records)


def verify_browsecomp_checkout(
    checkout: Path, *, git_executable: str = "git"
) -> str:
    root = checkout.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"BrowseComp checkout is not a directory: {root}")
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            (git_executable, "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify BrowseComp checkout {root}: {exc}") from exc
    if revision != BROWSECOMP_PLUS_REVISION:
        raise ValueError(
            f"BrowseComp checkout is {revision!r}, expected {BROWSECOMP_PLUS_REVISION!r}"
        )
    if dirty:
        raise ValueError("BrowseComp evaluator checkout must be clean")
    evaluator = root / "scripts_evaluation" / "evaluate_run.py"
    if not evaluator.is_file():
        raise ValueError(f"BrowseComp evaluator script is missing: {evaluator}")
    return revision


def official_evaluator_argv(
    *,
    checkout: Path,
    input_dir: Path,
    ground_truth: Path,
    eval_dir: Path,
    model: Path,
    qrel_evidence: Path | None = None,
    python_executable: str = sys.executable,
    tensor_parallel_size: int = 1,
) -> tuple[str, ...]:
    """Return literal argv for the pinned upstream Qwen judge; never shell text."""

    root = checkout.expanduser().resolve()
    verify_browsecomp_checkout(root)
    evaluator = root / "scripts_evaluation" / "evaluate_run.py"
    official_runs = input_dir.expanduser().resolve()
    truth = ground_truth.expanduser().resolve()
    qrels = (
        qrel_evidence.expanduser().resolve()
        if qrel_evidence is not None
        else root / "topics-qrels" / "qrel_evidence.txt"
    )
    destination = eval_dir.expanduser().resolve()
    judge_model = model.expanduser().resolve()
    if not evaluator.is_file():
        raise ValueError(f"BrowseComp evaluator script is missing: {evaluator}")
    preflight_official_directory(official_runs)
    if not truth.is_file():
        raise ValueError(f"BrowseComp ground truth is missing: {truth}")
    if not qrels.is_file():
        raise ValueError(f"BrowseComp evidence qrels are missing: {qrels}")
    if not judge_model.is_dir():
        raise ValueError(
            f"BrowseComp judge model must be a resolved local snapshot: {judge_model}"
        )
    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size < 1
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    if (
        not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
    ):
        raise ValueError("python_executable must be a non-empty safe argv string")
    return (
        python_executable,
        str(evaluator),
        "--input_dir",
        str(official_runs),
        "--ground_truth",
        str(truth),
        "--eval_dir",
        str(destination),
        "--qrel_evidence",
        str(qrels),
        "--model",
        str(judge_model),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
    )


__all__ = [
    "BROWSECOMP_MANIFEST_SCHEMA",
    "BROWSECOMP_RESULT_SCHEMA",
    "BrowseCompArtifactStore",
    "BrowseCompBatchRunner",
    "BrowseCompBatchSummary",
    "BrowseCompRunRecord",
    "BrowseCompTask",
    "BrowseCompTaskOutcome",
    "BrowseCompTaskSet",
    "MultiAgentEnvironmentFactory",
    "MultiAgentModelFactory",
    "TaskWorker",
    "browsecomp_task_template",
    "format_browsecomp_task",
    "load_browsecomp_tasks",
    "official_evaluator_argv",
    "preflight_official_directory",
    "run_mini_agent_task",
    "run_multi_agent_task",
    "validate_official_record",
    "verify_browsecomp_checkout",
]
