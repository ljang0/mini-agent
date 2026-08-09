from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import regex as timeout_regex  # type: ignore[import]

from . import __version__
from .harnesses import Harness
from .environments.base import EnvironmentFactory
from .environments.swe import SWEPatchPayload
from .runtime import ModelBackend, write_trace_jsonl
from .types import BudgetLimits, RunFailed, RunResult, Task, TraceEvent


def _atomic_write_text(path: Path, content: str) -> None:
    """Durably replace a text artifact without exposing a truncated destination."""

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace a binary artifact without exposing a partial destination."""

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class EvalOutcome:
    score: Optional[float]
    passed: Optional[bool]
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    observed_at_utc: str
    task_id: str
    harness: str
    variant_id: str
    repeat: int
    status: str
    score: Optional[float]
    passed: Optional[bool]
    answer: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    cost_usd: float
    cost_known: bool
    usage_complete: bool
    model_calls: int
    wall_time_seconds: float
    backend_active_union_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tool_calls: int = 0
    tool_output_bytes: int = 0


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"task at {path}:{line_number} must be an object")
            task_id = raw.get("task_id") or raw.get("id")
            prompt = raw.get("prompt")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"task at {path}:{line_number} needs a string id")
            if task_id in seen:
                raise ValueError(f"duplicate task id {task_id!r}")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"task {task_id!r} needs a non-empty prompt")
            seen.add(task_id)
            raw_metadata = raw.get("metadata")
            if raw_metadata is not None and not isinstance(raw_metadata, dict):
                raise ValueError(f"task {task_id!r} metadata must be an object or null")
            metadata = dict(raw_metadata or {})
            for passthrough in ("evaluator", "parallel_tasks", "split"):
                if passthrough in raw:
                    metadata[passthrough] = raw[passthrough]
            task = Task(
                task_id=task_id,
                prompt=prompt,
                context=str(raw.get("context") or ""),
                reference_answer=raw.get("reference_answer"),
                metadata=metadata,
            )
            _validated_evaluator_spec(task)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"task file {path} is empty")
    return tasks


def _validated_evaluator_spec(task: Task) -> Optional[Mapping[str, Any]]:
    spec = task.metadata.get("evaluator")
    if spec is None and task.reference_answer is not None:
        spec = {"type": "exact", "value": task.reference_answer}
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"task {task.task_id!r} evaluator must be an object")
    evaluator_type = spec.get("type")
    if evaluator_type not in {"exact", "contains", "regex", "json_equal"}:
        raise ValueError(
            f"task {task.task_id!r} has unknown evaluator {evaluator_type!r}"
        )
    if "value" not in spec:
        raise ValueError(f"task {task.task_id!r} evaluator requires value")
    if evaluator_type in {"contains", "regex"} and not isinstance(
        spec.get("value"), str
    ):
        raise ValueError(
            f"task {task.task_id!r} {evaluator_type} value must be a string"
        )
    if evaluator_type == "contains" and not spec["value"].strip():
        raise ValueError(f"task {task.task_id!r} contains value must be non-empty")
    if evaluator_type == "regex" and not spec["value"]:
        raise ValueError(f"task {task.task_id!r} regex value must be non-empty")
    if evaluator_type == "regex":
        try:
            timeout_regex.compile(spec["value"])
        except timeout_regex.error as exc:
            raise ValueError(
                f"task {task.task_id!r} has invalid regex evaluator: {exc}"
            ) from exc
    return spec


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    return left == right


def evaluate_answer(task: Task, answer: str) -> EvalOutcome:
    spec = _validated_evaluator_spec(task)
    if spec is None:
        return EvalOutcome(score=None, passed=None, details={"type": "unscored"})
    evaluator_type = spec["type"]
    value = spec.get("value")
    normalized = answer.strip()
    if evaluator_type == "exact":
        passed = normalized == str(value).strip()
    elif evaluator_type == "contains":
        passed = str(value).casefold() in normalized.casefold()
    elif evaluator_type == "regex":
        try:
            passed = (
                timeout_regex.search(
                    str(value),
                    answer,
                    flags=timeout_regex.MULTILINE,
                    timeout=0.25,
                )
                is not None
            )
        except TimeoutError as exc:
            raise ValueError(
                f"task {task.task_id!r} regex evaluator exceeded 250ms"
            ) from exc
    elif evaluator_type == "json_equal":
        try:
            passed = _json_values_equal(json.loads(answer), value)
        except json.JSONDecodeError:
            passed = False
    return EvalOutcome(
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"type": evaluator_type},
    )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolved_package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_filename_component(value: str, *, limit: int = 48) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return (component or "task")[:limit]


def _variant_id(harness: Harness) -> str:
    config = dict(harness.__dict__)
    return harness.name if not config else f"{harness.name}@{_fingerprint(config)[:8]}"


def validate_harness_variants(harnesses: Sequence[Harness]) -> None:
    variant_ids = [_variant_id(harness) for harness in harnesses]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("duplicate harness variants are not allowed in one matrix")


def _backend_provenance(backend: ModelBackend) -> Mapping[str, Any]:
    method = getattr(backend, "provenance", None)
    if callable(method):
        value = method()
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "backend_class": (
            f"{backend.__class__.__module__}.{backend.__class__.__qualname__}"
        )
    }


def _task_is_scored(task: Task) -> bool:
    return (
        task.reference_answer is not None or task.metadata.get("evaluator") is not None
    )


def _wilson(successes: int, attempts: int) -> tuple[float, float]:
    if attempts == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / attempts
    denominator = 1 + z * z / attempts
    center = (proportion + z * z / (2 * attempts)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / attempts + z * z / (4 * attempts * attempts)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


class MatrixRunner:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        limits: BudgetLimits,
        output_dir: Path,
        repeats: int = 1,
        random_seed: int = 0,
        capture_content: bool = False,
        matrix_max_cost_usd: Optional[float] = None,
        overwrite: bool = False,
        run_metadata: Optional[Mapping[str, Any]] = None,
        environment_factory: Optional[EnvironmentFactory] = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.backend = backend
        self.limits = limits
        self.output_dir = output_dir
        self.repeats = repeats
        self.random_seed = random_seed
        self.capture_content = capture_content
        if matrix_max_cost_usd is not None and (
            not isinstance(matrix_max_cost_usd, (int, float))
            or isinstance(matrix_max_cost_usd, bool)
            or not math.isfinite(matrix_max_cost_usd)
            or matrix_max_cost_usd < 0
        ):
            raise ValueError("matrix_max_cost_usd must be finite and non-negative")
        self.matrix_max_cost_usd = matrix_max_cost_usd
        self.overwrite = overwrite
        self.run_metadata = dict(run_metadata or {})
        self.environment_factory = environment_factory

    async def run(
        self, tasks: Sequence[Task], harnesses: Sequence[Harness]
    ) -> tuple[list[TrialRecord], dict[str, Any]]:
        if not harnesses:
            raise ValueError("at least one harness is required")
        if not tasks:
            raise ValueError("at least one task is required")
        for task in tasks:
            _validated_evaluator_spec(task)
        harness_variants = [(harness, _variant_id(harness)) for harness in harnesses]
        validate_harness_variants(harnesses)
        if self.environment_factory is not None:
            self.environment_factory.validate_artifact_path(self.output_dir, tasks)
            self.environment_factory.validate_trial_plan(
                len(tasks) * len(harnesses) * self.repeats
            )
        specs = [
            (task, harness, variant_id, repeat)
            for task in tasks
            for harness, variant_id in harness_variants
            for repeat in range(self.repeats)
        ]
        random.Random(self.random_seed).shuffle(specs)
        manifest = {
            "schema_version": "scaffoldlab-manifest-v2",
            "scaffoldlab_version": __version__,
            "source_revision": os.getenv("SCAFFOLDLAB_SOURCE_REVISION"),
            "scaffoldlab_source_tree_sha256": _source_tree_sha256(),
            "host_environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    "httpx": _resolved_package_version("httpx"),
                    "regex": _resolved_package_version("regex"),
                },
            },
            "tasks": [asdict(task) for task in tasks],
            "harnesses": [
                {
                    "name": harness.name,
                    "variant_id": variant_id,
                    "config": dict(harness.__dict__),
                }
                for harness, variant_id in harness_variants
            ],
            "backend": dict(_backend_provenance(self.backend)),
            "task_environment": (
                dict(self.environment_factory.provenance())
                if self.environment_factory is not None
                else None
            ),
            "per_trial_limits": asdict(self.limits),
            "matrix_max_cost_usd": self.matrix_max_cost_usd,
            "repeats": self.repeats,
            "seed": self.random_seed,
            "capture_content": self.capture_content,
            "run_metadata": self.run_metadata,
        }
        run_fingerprint = _fingerprint(manifest)
        manifest = {**manifest, "run_fingerprint": run_fingerprint}
        artifact_paths = [
            self.output_dir / "manifest.json",
            self.output_dir / "results.jsonl",
            self.output_dir / "summary.json",
        ]
        traces_dir = self.output_dir / "traces"
        patches_dir = self.output_dir / "patches"
        if self.output_dir.is_symlink():
            raise FileExistsError("output directory cannot be a symbolic link")
        linked = [path for path in artifact_paths if path.is_symlink()]
        if linked:
            raise FileExistsError(f"refusing symbolic-link artifacts: {linked}")
        if traces_dir.is_symlink():
            raise FileExistsError("refusing symbolic-link traces directory")
        if patches_dir.is_symlink():
            raise FileExistsError("refusing symbolic-link patches directory")
        if not self.overwrite and any(path.exists() for path in artifact_paths):
            raise FileExistsError(
                f"output directory already contains a run: {self.output_dir}"
            )
        if not self.overwrite and traces_dir.exists():
            raise FileExistsError(
                f"output directory already contains traces: {self.output_dir}"
            )
        if not self.overwrite and patches_dir.exists():
            raise FileExistsError(
                f"output directory already contains patches: {self.output_dir}"
            )
        if self.overwrite:
            for directory, label in (
                (traces_dir, "traces"),
                (patches_dir, "patches"),
            ):
                if directory.exists():
                    if not directory.is_dir():
                        raise FileExistsError(
                            f"{label} path exists and is not a directory"
                        )
                    shutil.rmtree(directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.output_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        )
        records: list[TrialRecord] = []
        termination_reason: Optional[str] = (
            "matrix cost cap is zero" if self.matrix_max_cost_usd == 0 else None
        )
        for index, (task, harness, variant_id, repeat) in enumerate(specs):
            if termination_reason is not None:
                break
            trial_id = (
                f"{run_fingerprint}-{index:04d}-"
                f"{_safe_filename_component(task.task_id)}-"
                f"{_safe_filename_component(variant_id)}-r{repeat}"
            )
            trace_path = self.output_dir / "traces" / f"{trial_id}.jsonl"
            result: Optional[RunResult] = None
            trace: tuple[TraceEvent, ...] = ()
            try:
                finish_patch_export: Any = None
                prepare_patch_export = getattr(
                    self.environment_factory,
                    "prepare_trial_patch_export",
                    None,
                )
                if callable(prepare_patch_export):
                    prepare_patch_export()
                    finish_patch_export = getattr(
                        self.environment_factory,
                        "finish_trial_patch_export",
                        None,
                    )
                try:
                    result = await harness.run(
                        task,
                        self.backend,
                        self.limits,
                        capture_content=self.capture_content,
                        environment_factory=self.environment_factory,
                    )
                    result = self._externalize_patch_artifacts(result, trial_id)
                    outcome = evaluate_answer(task, result.answer)
                    trace = result.trace
                finally:
                    if callable(finish_patch_export):
                        await finish_patch_export()
                record = self._success_record(
                    trial_id,
                    task,
                    harness,
                    variant_id,
                    repeat,
                    result,
                    outcome,
                    trace_path,
                )
            except RunFailed as exc:
                trace = exc.trace
                is_scored = _task_is_scored(task)
                record = TrialRecord(
                    trial_id=trial_id,
                    observed_at_utc=datetime.now(timezone.utc).isoformat(),
                    task_id=task.task_id,
                    harness=harness.name,
                    variant_id=variant_id,
                    repeat=repeat,
                    status="error",
                    score=0.0 if is_scored else None,
                    passed=False if is_scored else None,
                    answer=None,
                    input_tokens=exc.usage.input_tokens,
                    output_tokens=exc.usage.output_tokens,
                    cache_read_input_tokens=exc.usage.cache_read_input_tokens,
                    cache_write_input_tokens=exc.usage.cache_write_input_tokens,
                    cost_usd=exc.usage.cost_usd,
                    cost_known=exc.usage.cost_known,
                    usage_complete=exc.usage.complete,
                    model_calls=exc.model_calls,
                    wall_time_seconds=exc.wall_time_seconds,
                    backend_active_union_seconds=exc.backend_active_union_seconds,
                    metadata={"trace_path": str(trace_path)},
                    error=f"{exc.cause_type}: {exc}",
                    tool_calls=exc.tool_calls,
                    tool_output_bytes=exc.tool_output_bytes,
                )
            except Exception as exc:
                if result is not None:
                    trace = result.trace
                    usage = result.usage
                    model_calls = result.model_calls
                    wall_time = result.wall_time_seconds
                    backend_active_union = result.backend_active_union_seconds
                    metadata = {"trace_path": str(trace_path)}
                else:
                    usage = None
                    model_calls = 0
                    wall_time = 0.0
                    backend_active_union = 0.0
                    metadata = {}
                is_scored = _task_is_scored(task)
                record = TrialRecord(
                    trial_id=trial_id,
                    observed_at_utc=datetime.now(timezone.utc).isoformat(),
                    task_id=task.task_id,
                    harness=harness.name,
                    variant_id=variant_id,
                    repeat=repeat,
                    status="error",
                    score=0.0 if is_scored else None,
                    passed=False if is_scored else None,
                    answer=None,
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    cache_read_input_tokens=(
                        usage.cache_read_input_tokens if usage else 0
                    ),
                    cache_write_input_tokens=(
                        usage.cache_write_input_tokens if usage else 0
                    ),
                    cost_usd=usage.cost_usd if usage else 0.0,
                    cost_known=usage.cost_known if usage else False,
                    usage_complete=usage.complete if usage else False,
                    model_calls=model_calls,
                    wall_time_seconds=wall_time,
                    backend_active_union_seconds=backend_active_union,
                    metadata=metadata,
                    error=f"{type(exc).__name__}: {exc}",
                    tool_calls=result.tool_calls if result is not None else 0,
                    tool_output_bytes=(
                        result.tool_output_bytes if result is not None else 0
                    ),
                )
            write_trace_jsonl(trace_path, trace)
            records.append(record)
            self._write_records(records)
            if self.matrix_max_cost_usd is not None:
                if not record.cost_known or not record.usage_complete:
                    if index + 1 < len(specs):
                        termination_reason = (
                            "matrix spend became unverifiable after "
                            "incomplete/unknown usage"
                        )
                        break
                spent = sum(item.cost_usd for item in records)
                if spent >= self.matrix_max_cost_usd:
                    if index + 1 < len(specs):
                        termination_reason = (
                            f"matrix cost cap reached ({self.matrix_max_cost_usd})"
                        )
                        break
        summary = summarize(
            records,
            run_fingerprint=run_fingerprint,
            planned_trials=len(specs),
            termination_reason=termination_reason,
        )
        _atomic_write_text(
            self.output_dir / "summary.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
        return records, summary

    def _externalize_patch_artifacts(
        self, result: RunResult, trial_id: str
    ) -> RunResult:
        payloads: list[SWEPatchPayload] = []
        seen: set[int] = set()

        def collect(value: Any) -> None:
            if isinstance(value, SWEPatchPayload):
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    payloads.append(value)
                return
            if isinstance(value, Mapping):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(result.metadata)
        if not payloads:
            return result

        patches_dir = self.output_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        replacements: dict[int, Mapping[str, Any]] = {}
        for index, payload in enumerate(payloads):
            suffix = "" if len(payloads) == 1 else f"-{index:03d}"
            path = patches_dir / f"{trial_id}{suffix}.patch"
            content = payload.content
            _atomic_write_bytes(path, content)
            replacements[id(payload)] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "format": "git_diff_binary",
            }

        def sanitize(value: Any) -> Any:
            if isinstance(value, SWEPatchPayload):
                return dict(replacements[id(value)])
            if isinstance(value, Mapping):
                return {key: sanitize(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(child) for child in value]
            return value

        return replace(result, metadata=sanitize(result.metadata))

    def _success_record(
        self,
        trial_id: str,
        task: Task,
        harness: Harness,
        variant_id: str,
        repeat: int,
        result: RunResult,
        outcome: EvalOutcome,
        trace_path: Path,
    ) -> TrialRecord:
        return TrialRecord(
            trial_id=trial_id,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            task_id=task.task_id,
            harness=harness.name,
            variant_id=variant_id,
            repeat=repeat,
            status="completed",
            score=outcome.score,
            passed=outcome.passed,
            answer=result.answer if self.capture_content else None,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_read_input_tokens=result.usage.cache_read_input_tokens,
            cache_write_input_tokens=result.usage.cache_write_input_tokens,
            cost_usd=result.usage.cost_usd,
            cost_known=result.usage.cost_known,
            usage_complete=result.usage.complete,
            model_calls=result.model_calls,
            wall_time_seconds=result.wall_time_seconds,
            backend_active_union_seconds=result.backend_active_union_seconds,
            metadata={
                **dict(result.metadata),
                "trace_path": str(trace_path),
                "evaluator": dict(outcome.details),
                "answer_chars": len(result.answer),
                "answer_sha256": hashlib.sha256(
                    result.answer.encode("utf-8")
                ).hexdigest(),
            },
            tool_calls=result.tool_calls,
            tool_output_bytes=result.tool_output_bytes,
        )

    def _write_records(self, records: Iterable[TrialRecord]) -> None:
        content = "".join(
            json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
        )
        _atomic_write_text(self.output_dir / "results.jsonl", content)


def summarize(
    records: Sequence[TrialRecord],
    *,
    run_fingerprint: str,
    planned_trials: Optional[int] = None,
    termination_reason: Optional[str] = None,
) -> dict[str, Any]:
    by_harness: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_harness.setdefault(record.variant_id, []).append(record)
    harness_summaries: dict[str, Any] = {}
    for variant_id, group in sorted(by_harness.items()):
        completed = [record for record in group if record.status == "completed"]
        scored = [record for record in group if record.passed is not None]
        completed_scored = [record for record in completed if record.passed is not None]
        successes = sum(record.passed is True for record in scored)
        low, high = _wilson(successes, len(scored))
        error_types: dict[str, int] = {}
        for record in group:
            if record.error:
                error_type = record.error.split(":", 1)[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        divisor = len(group)
        all_cost_known = all(record.cost_known for record in group)
        all_usage_complete = all(record.usage_complete for record in group)
        harness_summaries[variant_id] = {
            "harness": group[0].harness,
            "variant_id": variant_id,
            "attempts": len(group),
            "completed": len(completed),
            "errors": len(group) - len(completed),
            "error_types": dict(sorted(error_types.items())),
            "operational_completion_rate": len(completed) / divisor,
            "scored": len(scored),
            "successes": successes,
            "attempt_success_rate": successes / len(scored) if scored else None,
            "attempt_success_wilson_95": [low, high] if scored else None,
            "completed_success_rate": (
                sum(record.passed is True for record in completed_scored)
                / len(completed_scored)
                if completed_scored
                else None
            ),
            "mean_wall_time_seconds": sum(record.wall_time_seconds for record in group)
            / divisor,
            "mean_backend_active_union_seconds": sum(
                record.backend_active_union_seconds for record in group
            )
            / divisor,
            "tool_calls": sum(record.tool_calls for record in group),
            "tool_output_bytes": sum(record.tool_output_bytes for record in group),
            "usage_complete": all_usage_complete,
            "input_tokens_lower_bound": sum(record.input_tokens for record in group),
            "output_tokens_lower_bound": sum(record.output_tokens for record in group),
            "cache_read_input_tokens_lower_bound": sum(
                record.cache_read_input_tokens for record in group
            ),
            "cache_write_input_tokens_lower_bound": sum(
                record.cache_write_input_tokens for record in group
            ),
            "mean_input_tokens": (
                sum(record.input_tokens for record in group) / divisor
                if all_usage_complete
                else None
            ),
            "mean_output_tokens": (
                sum(record.output_tokens for record in group) / divisor
                if all_usage_complete
                else None
            ),
            "cost_known": all_cost_known,
            "cost_usd_known_lower_bound": sum(record.cost_usd for record in group),
            "mean_cost_usd": (
                sum(record.cost_usd for record in group) / divisor
                if all_cost_known and all_usage_complete
                else None
            ),
        }
    total_errors = sum(record.status != "completed" for record in records)
    planned = len(records) if planned_trials is None else planned_trials
    return {
        "schema_version": "scaffoldlab-summary-v2",
        "run_fingerprint": run_fingerprint,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials": len(records),
        "planned_trials": planned,
        "matrix_completed": len(records) == planned and termination_reason is None,
        "termination_reason": termination_reason,
        "total_errors": total_errors,
        "total_cost_usd_known_lower_bound": sum(record.cost_usd for record in records),
        "all_cost_known": all(record.cost_known for record in records),
        "all_usage_complete": all(record.usage_complete for record in records),
        "harnesses": harness_summaries,
        "release_decision": "HUMAN_REVIEW_REQUIRED",
    }
