"""SWE-bench adapter: image binding, task loading, generation, grader contracts.

Container provisioning lives in :mod:`mini_agent.runtimes` and the bash tool in
:mod:`mini_agent.environments.bash`; everything here is what makes a run
*SWE-bench*: the official image-name rule, the pinned upstream revision, the
per-task Git baseline, and the official grader's prediction/source contracts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..environments.bash import (
    DEFAULT_MAX_ARCHIVE_BYTES,
    DEFAULT_MAX_PATCH_BYTES,
    BashEnvironment,
    SWEPatchState,
    container_bash_environment,
)
from ..models import Model
from ..team import run_team, selected_harness
from ..runtime import RunContext
from ..runtimes.apptainer import ApptainerRuntime, apptainer_image_identity
from ..runtimes.base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ProcessRunner,
    positive_int,
    positive_number,
    require_argv,
    require_ref,
    require_workdir,
    resolve_runner,
)
from ..runtimes.docker import (
    DockerRuntime,
    RuntimeDoctorReport,
    docker_doctor,
    docker_image_id,
)
from ..specs import AgentSpecV1
from ..types import (
    BudgetLimits,
    _require_bool,
    _require_callable,
    _require_finite_number,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
    strict_json_loads,
)
from .._hash import canonical_bytes, immutable_file_identity
from ..storage import (
    atomic_bytes,
    atomic_json,
    read_committed_result,
)
from .base import (
    task_agent_builder,
    owned_instance_artifacts,
    BenchmarkTask,
    EvaluationOutcome,
    task_agent_root,
)


SWEBENCH_REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
SWEBENCH_TAG = "v4.1.0"
SWEBENCH_VERSION = "4.1.0"
SWEBENCH_WORKDIR = "/testbed"
SWEBENCH_BASH_ENV = "/root/.bashrc"
SWEBENCH_SOURCE_SHA256 = (
    "63d4d3d0543de66520fa44f12badddaa810f708a0d780954684c24c7ce075cc8"
)
ModelFactory = Callable[[str], Model | Awaitable[Model]]

_CONTAINER_NAME_PREFIX = "mini-agent-swe-"
_APPTAINER_ROOT_PREFIX = "mini-agent-swe-apptainer-"
_CONTAINER_ENV = ("HOME=/root", "PAGER=cat", "MANPAGER=cat", "TQDM_DISABLE=1")
_CONTAINER_LABELS = ("mini-agent.swebench=true",)
_DOCKER_EXEC_ENV = {"BASH_ENV": SWEBENCH_BASH_ENV}
_APPTAINER_EXEC_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "TQDM_DISABLE": "1",
    "BASH_ENV": SWEBENCH_BASH_ENV,
}
_APPTAINER_TOOL_DESCRIPTION = (
    "Run one bash command in the persistent SWE-bench /testbed workspace. "
    "Each call starts a fresh shell."
)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_GRADER_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
_INVALID_SDK = "SWE-bench grader Docker SDK identity is invalid"
_INVALID_SOURCE = "official SWE-bench package source tree is invalid"
_UNSAFE_ENTRY = "official SWE-bench package contains an unsafe {kind}"


def _require_grader_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_GRADER_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be one path-safe component")
    return value


def _plain_string(value: Any) -> bool:
    """Return whether ``value`` is a non-empty string free of NUL bytes."""

    return isinstance(value, str) and bool(value) and "\x00" not in value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _instance_id(instance: Mapping[str, Any]) -> str:
    value = instance.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("SWE-bench instance requires instance_id")
    return value


def _expected_base_commit(instance: Mapping[str, Any]) -> str | None:
    value = instance.get("base_commit")
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("SWE-bench base_commit must be a full Git commit")
    return value.casefold()


def _benchmark_identity(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    if value is None:
        return {
            "benchmark": "swe_bench",
            "benchmark_revision": SWEBENCH_REVISION,
            "benchmark_tag": SWEBENCH_TAG,
        }
    if not isinstance(value, Mapping) or not value:
        raise ValueError("container benchmark identity must be a non-empty object")
    message = "container benchmark identity must contain strings"
    return {
        require_ref(name, message): require_ref(item, message)
        for name, item in value.items()
    }


def swebench_image_name(instance: Mapping[str, Any]) -> str:
    explicit = instance.get("image_name") or instance.get("docker_image")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit or explicit.startswith("-"):
            raise ValueError("SWE-bench image name must be a non-empty Docker image")
        if "\x00" in explicit:
            raise ValueError("SWE-bench image name contains a NUL byte")
        return explicit
    docker_id = _instance_id(instance).replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest"


@dataclass(frozen=True)
class SWEbenchImageBinding:
    """Immutable image selected for one SWE-bench task.

    ``requested`` is retained for auditability. ``execution_ref`` is the exact
    Docker image ID or local SIF selected during preflight and is intentionally
    omitted from the location-independent manifest identity.
    """

    runtime: str
    requested: str
    identity: str
    execution_ref: str

    def __post_init__(self) -> None:
        if self.runtime not in {"docker", "apptainer"}:
            raise ValueError("SWE-bench image binding runtime is invalid")
        require_ref(
            self.requested, "SWE-bench requested image is invalid", no_dash=True
        )
        if not isinstance(self.execution_ref, str) or not self.execution_ref:
            raise ValueError("SWE-bench image execution reference is invalid")
        if not _IMAGE_ID.fullmatch(self.identity):
            raise ValueError("SWE-bench image identity must be a SHA-256 digest")
        if self.runtime == "docker" and self.execution_ref != self.identity:
            raise ValueError("Docker must execute the resolved image ID")
        if self.runtime == "apptainer" and not Path(self.execution_ref).is_absolute():
            raise ValueError("Apptainer must execute an absolute local image path")

    def manifest_identity(self) -> Mapping[str, str]:
        return {
            "runtime": self.runtime,
            "requested": self.requested,
            "identity": self.identity,
        }


def _apptainer_source(instance: Mapping[str, Any], image: str | None) -> str:
    if image is not None and not isinstance(image, str):
        raise ValueError("Apptainer image must be a string or None")
    requested = image or "docker://" + swebench_image_name(instance).removeprefix(
        "docker.io/"
    )
    return require_ref(requested, "Apptainer image reference is invalid", no_dash=True)


async def resolve_swebench_image_binding(
    instance: Mapping[str, Any],
    *,
    runtime: str,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    apptainer_image_cache: Path | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 60.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> SWEbenchImageBinding:
    """Resolve a mutable task image reference to bytes before inference."""

    from ..runtimes.apptainer import materialize_apptainer_image

    _require_mapping(instance, "SWE-bench instance")
    if runtime not in {"docker", "apptainer"}:
        raise ValueError("SWE-bench runtime must be docker or apptainer")
    resolved_timeout = positive_number(timeout_seconds, "timeout_seconds")
    resolved_output = positive_int(max_output_bytes, "max_output_bytes")
    process_runner = resolve_runner(runner)
    requested = swebench_image_name(instance)
    if runtime == "docker":
        image_id = await docker_image_id(
            requested,
            runtime=require_argv(container_runtime, "container runtime"),
            runner=process_runner,
            timeout_seconds=resolved_timeout,
            max_output_bytes=resolved_output,
            pull_if_missing=True,
        )
        return SWEbenchImageBinding(
            runtime="docker",
            requested=requested,
            identity=image_id,
            execution_ref=image_id,
        )

    require_ref(apptainer_executable, "Apptainer executable must be non-empty")
    source = "docker://" + requested.removeprefix("docker.io/")
    selected = await materialize_apptainer_image(
        source,
        executable=apptainer_executable,
        runner=process_runner,
        cache=apptainer_image_cache,
        timeout_seconds=max(resolved_timeout, 1800.0),
        max_output_bytes=resolved_output,
    )
    return SWEbenchImageBinding(
        runtime="apptainer",
        requested=source,
        identity=await apptainer_image_identity(selected),
        execution_ref=str(Path(selected).expanduser().resolve()),
    )


async def swebench_doctor(
    *,
    runtime: Sequence[str] = ("docker",),
    image: str | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 64 * 1024,
    require_rootless: bool = True,
) -> RuntimeDoctorReport:
    """Probe the container runtime an SWE-bench Docker run requires."""

    return await docker_doctor(
        runtime=runtime,
        image=image,
        runner=runner,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        require_rootless=require_rootless,
    )


async def docker_swe_environment(
    instance: Mapping[str, Any],
    *,
    image_binding: SWEbenchImageBinding | None = None,
    runtime: Sequence[str] = ("docker",),
    platform: str | None = None,
    timeout_seconds: float = 60.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    workdir: str = SWEBENCH_WORKDIR,
    network_disabled: bool = False,
    require_git_baseline: bool = True,
    benchmark_identity: Mapping[str, Any] | None = None,
    runner: ProcessRunner | None = None,
) -> BashEnvironment:
    """One bash tool in a persistent rootless SWE-bench instance container."""

    _require_mapping(instance, "SWE-bench instance")
    _require_bool(network_disabled, "network_disabled")
    _require_bool(require_git_baseline, "require_git_baseline")
    resolved_workdir = require_workdir(workdir)
    resolved_identity = _benchmark_identity(benchmark_identity)
    resolved_runtime = require_argv(runtime, "container runtime")
    resolved_timeout = positive_number(timeout_seconds, "timeout_seconds")
    resolved_output = positive_int(max_output_bytes, "max_output_bytes")
    image = swebench_image_name(instance)
    expected_base_commit = _expected_base_commit(instance)
    if expected_base_commit is not None and not require_git_baseline:
        raise ValueError(
            "a task base_commit cannot be verified without a Git baseline"
        )
    instance_id = _instance_id(instance)
    process_runner = resolve_runner(runner)
    if image_binding is None:
        image_id = await docker_image_id(
            image,
            runtime=resolved_runtime,
            runner=process_runner,
            timeout_seconds=resolved_timeout,
            max_output_bytes=resolved_output,
            pull_if_missing=True,
        )
    elif image_binding.runtime != "docker":
        raise ValueError("Docker received a non-Docker image binding")
    elif image_binding.requested != image:
        raise ValueError("Docker image binding does not match the task image")
    else:
        image_id = image_binding.identity
    sandbox = await DockerRuntime.start(
        image=image,
        image_id=image_id,
        runtime=resolved_runtime,
        runner=process_runner,
        name_label=instance_id,
        name_prefix=_CONTAINER_NAME_PREFIX,
        workdir=resolved_workdir,
        exec_env=_DOCKER_EXEC_ENV,
        container_env=_CONTAINER_ENV,
        labels=_CONTAINER_LABELS,
        platform=platform,
        network_disabled=network_disabled,
        timeout_seconds=resolved_timeout,
        max_output_bytes=resolved_output,
    )
    return await container_bash_environment(
        sandbox,
        require_git_baseline=require_git_baseline,
        expected_base_commit=expected_base_commit,
        base_identity_prefix=image_id,
        tool_description=(
            f"Run one bash command in the persistent {resolved_workdir} "
            "workspace. Each call starts a new shell."
        ),
        provenance_extra=resolved_identity,
        destroy_on_timeout=True,
        timeout_seconds=resolved_timeout,
        max_output_bytes=resolved_output,
        max_patch_bytes=max_patch_bytes,
        max_archive_bytes=max_archive_bytes,
    )


async def apptainer_swe_environment(
    instance: Mapping[str, Any],
    *,
    image: str | None = None,
    image_binding: SWEbenchImageBinding | None = None,
    executable: str = "apptainer",
    scratch_root: Path | None = None,
    image_cache: Path | None = None,
    overlay_size_mib: int = 16 * 1024,
    timeout_seconds: float = 60,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    workdir: str = SWEBENCH_WORKDIR,
    network_disabled: bool = False,
    require_git_baseline: bool = True,
    benchmark_identity: Mapping[str, Any] | None = None,
    shared_binds: Mapping[str, Path] | None = None,
    runner: ProcessRunner | None = None,
) -> BashEnvironment:
    """One bash tool in a private Apptainer fakeroot overlay over the task SIF."""

    _require_mapping(instance, "SWE-bench instance")
    require_ref(executable, "Apptainer executable must be non-empty")
    requested = _apptainer_source(instance, image)
    expected_base_commit = _expected_base_commit(instance)
    if image_binding is not None:
        if image_binding.runtime != "apptainer":
            raise ValueError("Apptainer received a non-Apptainer image binding")
        if image_binding.requested != requested:
            raise ValueError("Apptainer image binding does not match the task image")
    sandbox = await ApptainerRuntime.start(
        image=requested if image_binding is None else image_binding.execution_ref,
        image_source=requested,
        materialize=image_binding is None,
        expected_identity=None if image_binding is None else image_binding.identity,
        executable=executable,
        runner=runner,
        workdir=workdir,
        exec_env=_APPTAINER_EXEC_ENV,
        scratch_root=scratch_root,
        image_cache=image_cache,
        root_prefix=_APPTAINER_ROOT_PREFIX,
        overlay_size_mib=overlay_size_mib,
        network_disabled=network_disabled,
        shared_binds=shared_binds,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    return await container_bash_environment(
        sandbox,
        expected_base_commit=expected_base_commit,
        base_identity_prefix=sandbox.image_identity,
        tool_description=_APPTAINER_TOOL_DESCRIPTION,
        provenance_extra=_benchmark_identity(benchmark_identity),
        startup_label="Apptainer SWE setup",
        require_git_baseline=require_git_baseline,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        max_patch_bytes=max_patch_bytes,
        max_archive_bytes=max_archive_bytes,
    )


def load_swebench(path: Path, *, limit: int | None = None) -> tuple[BenchmarkTask, ...]:
    if limit is not None:
        _require_positive_int(limit, "SWE-bench limit")
    source = path.expanduser().resolve()
    rows: list[tuple[int, Any]] = []
    if source.suffix.casefold() == ".json":
        try:
            value = strict_json_loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("SWE-bench JSON dataset is invalid") from exc
        if not isinstance(value, list):
            raise ValueError("SWE-bench .json dataset must contain an array")
        rows = list(enumerate(value, 1))
    else:
        lines = source.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"SWE-bench line {number} is invalid JSON") from exc
            rows.append((number, item))

    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    for number, item in rows:
        _require_mapping(item, f"SWE-bench line {number}")
        problem = item.get("problem_statement")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"SWE-bench line {number} is missing required fields")
        instance_id = _require_grader_component(
            item.get("instance_id"), f"SWE-bench line {number} instance_id"
        )
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance_id {instance_id!r}")
        seen.add(instance_id)
        tasks.append(BenchmarkTask(instance_id, problem, dict(item)))
        if limit is not None and len(tasks) == limit:
            break
    if not tasks:
        raise ValueError("SWE-bench task file contains no instances")
    return tuple(tasks)


async def prepare_swebench_image_bindings(
    tasks: Sequence[BenchmarkTask],
    *,
    runtime: str,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    apptainer_image_cache: Path | None = None,
) -> Mapping[str, SWEbenchImageBinding]:
    """Resolve every selected task image before a run manifest is committed."""

    if isinstance(tasks, (str, bytes)) or not tasks:
        raise ValueError("SWE-bench image preflight requires tasks")
    bindings: dict[str, SWEbenchImageBinding] = {}
    by_requested: dict[str, SWEbenchImageBinding] = {}
    for task in tasks:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("SWE-bench image preflight requires benchmark tasks")
        if task.task_id in bindings:
            raise ValueError(f"duplicate SWE-bench task id {task.task_id!r}")
        requested = task.data.get("image_name") or task.data.get("docker_image")
        cache_key = (
            requested if isinstance(requested, str) else "instance:" + task.task_id
        )
        binding = by_requested.get(cache_key)
        if binding is None:
            binding = await resolve_swebench_image_binding(
                task.data,
                runtime=runtime,
                container_runtime=container_runtime,
                apptainer_executable=apptainer_executable,
                apptainer_image_cache=apptainer_image_cache,
            )
            by_requested[cache_key] = binding
        bindings[task.task_id] = binding
    return bindings


async def run_swebench_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    model_factory: ModelFactory,
    system_prompt: str,
    max_steps: int,
    runtime: str,
    model_name: str = "mini-agent",
    scratch_root: Path | None = None,
    apptainer_image_cache: Path | None = None,
    image_binding: SWEbenchImageBinding | None = None,
    overlay_size_mib: int = 16 * 1024,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    multi_agent: bool = False,
    harness: str = "single",
    team_size: int | None = None,
    max_active_agents: int = 4,
    max_total_agents: int = 16,
    per_agent_limits: BudgetLimits | None = None,
    agent_spec: AgentSpecV1 | None = None,
) -> EvaluationOutcome:
    _require_callable(model_factory, "model_factory")
    _require_str(system_prompt, "system_prompt", non_empty=False)
    _require_positive_int(max_steps, "max_steps")
    _require_str(model_name, "model_name")
    _require_grader_component(task.task_id, "SWE-bench task instance_id")
    _require_grader_component(
        model_name.replace("/", "__"), "SWE-bench model_name_or_path"
    )
    _require_bool(multi_agent, "multi_agent")
    if runtime not in {"docker", "apptainer"}:
        raise ValueError("SWE-bench runtime must be docker or apptainer")

    async def environment_for(agent_id: str) -> Any:
        del agent_id
        if runtime == "docker":
            return await docker_swe_environment(
                task.data, image_binding=image_binding, runtime=container_runtime
            )
        return await apptainer_swe_environment(
            task.data,
            image_binding=image_binding,
            executable=apptainer_executable,
            scratch_root=scratch_root,
            image_cache=apptainer_image_cache,
            overlay_size_mib=overlay_size_mib,
        )

    agent_for = task_agent_builder(
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=max_steps,
        agent_spec=agent_spec,
        harness=selected_harness(harness, multi_agent),
        root_id=task_agent_root(task.task_id),
    )

    root_id = task_agent_root(task.task_id)
    team = await run_team(
        task.prompt,
        harness=selected_harness(harness, multi_agent),
        team_size=team_size,
        agent_builder=agent_for,
        environment_factory=environment_for,
        context=context,
        root_id=root_id,
        max_active_agents=max_active_agents,
        max_total_agents=max_total_agents,
        per_agent_limits=per_agent_limits,
    )
    result = team.require()
    if not isinstance(team.state, SWEPatchState):
        raise RuntimeError("root SWE agent produced no patch state")
    patch = team.state.patch
    metadata_value: Mapping[str, Any] = {
        **team.metadata(),
        "runtime": runtime,
        "environments": {
            agent_id: dict(base.provenance())
            for agent_id, base in team.bases().items()
        },
    }
    assert result is not None
    atomic_bytes(directory / "patch.diff", patch)
    prediction = {
        "instance_id": task.task_id,
        "model_patch": patch.decode("utf-8", errors="strict"),
        "model_name_or_path": model_name,
    }
    atomic_json(directory / "prediction.json", prediction)
    artifact_metadata = {
        **metadata_value,
        "patch_bytes": len(patch),
        "patch_sha256": _sha256(patch),
        "prediction_sha256": _sha256((directory / "prediction.json").read_bytes()),
    }
    return EvaluationOutcome(
        task.task_id, "completed", answer=result.answer, metadata=artifact_metadata
    )


def swebench_grader_image_name(instance_id: Any) -> str:
    """Return the exact default v4.1.0 remote image tag for one task."""

    resolved = _require_grader_component(instance_id, "SWE-bench instance_id")
    key = f"sweb.eval.x86_64.{resolved.casefold()}:latest"
    return f"swebench/{key}".replace("__", "_1776_")


def _grader_image_expectations(
    generation_manifest: Mapping[str, Any],
) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Return the recorded runtime and every task's exact expected image ID."""

    _require_mapping(generation_manifest, "SWE-bench generation manifest")
    config = generation_manifest.get("config")
    adapter = config.get("adapter") if isinstance(config, Mapping) else None
    if not isinstance(adapter, Mapping) or adapter.get("runtime") != "docker":
        raise ValueError(
            "official SWE-bench grading requires Docker generation image bindings"
        )
    recorded_runtime = adapter.get("container_runtime")
    if (
        not isinstance(recorded_runtime, list)
        or not recorded_runtime
        or not all(_plain_string(item) for item in recorded_runtime)
    ):
        raise ValueError("SWE-bench generation container runtime is invalid")
    bindings = adapter.get("image_bindings")
    tasks = generation_manifest.get("tasks")
    if not isinstance(bindings, Mapping) or not isinstance(tasks, list) or not tasks:
        raise ValueError("SWE-bench generation manifest has no Docker image bindings")
    task_ids: list[str] = []
    for task in tasks:
        _require_mapping(task, "SWE-bench generation task")
        task_ids.append(
            _require_grader_component(task.get("id"), "SWE-bench generation task id")
        )
    if len(task_ids) != len(set(task_ids)) or set(bindings) != set(task_ids):
        raise ValueError(
            "SWE-bench Docker image bindings must exactly cover generation tasks"
        )
    expected: list[tuple[str, str, str, str]] = []
    for instance_id in sorted(task_ids):
        binding = bindings[instance_id]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"runtime", "requested", "identity"}
            or binding.get("runtime") != "docker"
            or not isinstance(binding.get("requested"), str)
            or not _IMAGE_ID.fullmatch(str(binding.get("identity")))
        ):
            raise ValueError(
                f"SWE-bench task {instance_id!r} has an invalid Docker image binding"
            )
        expected.append(
            (
                instance_id,
                swebench_grader_image_name(instance_id),
                str(binding["requested"]),
                str(binding["identity"]),
            )
        )
    return list(recorded_runtime), expected


def _grader_environment(grader_environment: Mapping[str, str]) -> dict[str, str]:
    """Require an explicit, single-engine Docker environment for grading."""

    if not isinstance(grader_environment, Mapping) or not all(
        _plain_string(name) and isinstance(value, str) and "\x00" not in value
        for name, value in grader_environment.items()
    ):
        raise ValueError("SWE-bench grader environment must contain strings")
    environment = dict(grader_environment)
    if not environment.get("DOCKER_HOST"):
        raise ValueError(
            "SWE-bench grading requires an explicit DOCKER_HOST so image "
            "verification and the upstream grader address the same engine"
        )
    if environment.get("DOCKER_TLS_VERIFY") is not None and not environment.get(
        "DOCKER_CERT_PATH"
    ):
        raise ValueError(
            "DOCKER_TLS_VERIFY requires an explicit DOCKER_CERT_PATH for exact "
            "SWE-bench grader-engine binding"
        )
    return environment


def _require_current_grader_python(python_executable: Any) -> None:
    """Refuse to verify images on behalf of a different interpreter."""

    if not _plain_string(python_executable):
        raise ValueError("SWE-bench grader Python executable is invalid")
    resolved = shutil.which(python_executable)
    try:
        current = resolved is not None and os.path.samefile(resolved, sys.executable)
    except OSError:
        current = False
    if not current:
        raise ValueError(
            "SWE-bench image verification requires the current grader Python"
        )


def _docker_sdk_identity() -> tuple[str, str, str]:
    """Return the installed Docker SDK version and its exact package paths."""

    try:
        version = metadata.version("docker")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(_INVALID_SDK) from exc
    spec = importlib.util.find_spec("docker")
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    if spec is None or not _plain_string(spec.origin) or len(locations) != 1:
        raise RuntimeError(_INVALID_SDK)
    origin = os.path.abspath(str(spec.origin))
    package_root = os.path.abspath(str(locations[0]))
    if (
        not version
        or os.path.basename(origin) != "__init__.py"
        or os.path.islink(origin)
        or os.path.islink(package_root)
        or os.path.realpath(origin) != origin
        or os.path.realpath(package_root) != package_root
        or os.path.dirname(origin) != package_root
        or not os.path.isfile(origin)
        or not os.path.isdir(package_root)
    ):
        raise RuntimeError(_INVALID_SDK)
    return version, origin, package_root


def verify_swebench_grader_images(
    generation_manifest: Mapping[str, Any],
    *,
    python_executable: str,
    grader_environment: Mapping[str, str],
    timeout_seconds: float = 60.0,
) -> Mapping[str, Any]:
    """Prove every mutable ``:latest`` grader tag still resolves to generation bytes.

    Called before and after the upstream grader subprocess, so a tag that is
    re-pointed mid-run is caught even though grading itself is opaque.
    """

    recorded_runtime, expected = _grader_image_expectations(generation_manifest)
    environment = _grader_environment(grader_environment)
    _require_current_grader_python(python_executable)
    _require_finite_number(
        timeout_seconds, "SWE-bench image verification timeout", exclusive_minimum=0
    )
    version, origin, package_root = _docker_sdk_identity()
    import docker  # type: ignore[import-untyped]

    if (
        os.path.realpath(str(getattr(docker, "__file__", ""))) != origin
        or getattr(docker, "__version__", None) != version
    ):
        raise RuntimeError(_INVALID_SDK)
    try:
        client = docker.from_env(
            environment=environment, timeout=float(timeout_seconds)
        )
    except BaseException as exc:
        raise RuntimeError(
            "could not connect to the official SWE-bench grader Docker engine"
        ) from exc
    verified: list[Mapping[str, str]] = []
    try:
        for instance_id, grader_image, requested, image_id in expected:
            try:
                observed = client.images.get(grader_image).id
            except BaseException as exc:
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} is unavailable"
                ) from exc
            if not isinstance(observed, str) or not _IMAGE_ID.fullmatch(
                observed.strip().casefold()
            ):
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} "
                    "returned an invalid image ID"
                )
            if observed.strip().casefold() != image_id:
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} "
                    "changed identity"
                )
            verified.append(
                {
                    "instance_id": instance_id,
                    "grader_image": grader_image,
                    "generation_requested": requested,
                    "image_id": image_id,
                }
            )
    finally:
        client.close()
    return {
        "engine_contract": "in-process:docker.from_env",
        "docker_sdk_version": version,
        "docker_sdk": {
            "version": version,
            "origin": origin,
            "package_root": package_root,
        },
        "environment_sha256": _sha256(canonical_bytes(environment)),
        "generation_container_runtime": recorded_runtime,
        "images": verified,
    }


def official_grader_argv(
    *,
    predictions: Path,
    dataset_name: str,
    run_id: str,
    max_workers: int = 1,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    if (
        not _plain_string(dataset_name)
        or dataset_name.startswith("-")
        or not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or max_workers < 1
        or not _plain_string(python_executable)
    ):
        raise ValueError("invalid SWE-bench grader configuration")
    _require_grader_component(run_id, "SWE-bench run_id")
    return (
        python_executable,
        "-I",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions.expanduser().resolve()),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    )


def swebench_grader_source_identity(source_root: Path) -> Mapping[str, Any]:
    """Hash the explicit package root resolved by the isolated grader Python."""

    if not isinstance(source_root, Path):
        raise TypeError("SWE-bench package source root must be a path")
    root = source_root.expanduser()
    entry = root / "__init__.py"
    try:
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(_INVALID_SOURCE) from exc
    if (
        not root.is_absolute()
        or canonical != root
        or root.is_symlink()
        or not root.is_dir()
        or entry.is_symlink()
        or not entry.is_file()
    ):
        raise RuntimeError(_INVALID_SOURCE)
    candidates: list[Path] = []
    for directory, directory_names, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            path = parent / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(_UNSAFE_ENTRY.format(kind="directory"))
            if name != "__pycache__":
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(names):
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(_UNSAFE_ENTRY.format(kind="source"))
            candidates.append(path)
    files: list[Mapping[str, Any]] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        identity = immutable_file_identity(path, label="SWE-bench package source")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    # The digest covers every path, size, and content hash, so a separate file
    # count and total-size assertion could only ever fail with it.
    source_sha256 = _sha256(canonical_bytes(files))
    if source_sha256 != SWEBENCH_SOURCE_SHA256:
        raise RuntimeError(
            "installed SWE-bench package does not match pinned v4.1.0 source"
        )
    return {
        "project": "SWE-bench",
        "version": SWEBENCH_VERSION,
        "revision": SWEBENCH_REVISION,
        "source_root": str(root),
        "source_file_count": len(files),
        "source_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "source_sha256": source_sha256,
    }


def collect_predictions(output: Path, destination: Path) -> int:
    """Build the exact SWE-bench JSONL prediction input from task artifacts."""

    root, _, artifacts = owned_instance_artifacts(
        output, "prediction.json", label="SWE-bench"
    )
    target = _prediction_collection_target(root, destination)
    records: list[dict[str, str]] = []
    instance_ids: set[str] = set()
    for path in artifacts:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid SWE-bench prediction: {path}")
        model_patch = value.get("model_patch")
        model_name = value.get("model_name_or_path")
        if not isinstance(model_patch, str) or not isinstance(model_name, str):
            raise ValueError(f"incomplete SWE-bench prediction: {path}")
        instance_id = _require_grader_component(
            value.get("instance_id"), "SWE-bench prediction instance_id"
        )
        _require_grader_component(
            model_name.replace("/", "__"), "SWE-bench prediction model_name_or_path"
        )
        if instance_id in instance_ids:
            raise ValueError(f"duplicate SWE-bench instance_id {instance_id!r}")
        try:
            result = read_committed_result(path.parent, instance_id)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"SWE-bench prediction has no committed result: {path}"
            ) from exc
        if result.get("status") != "completed":
            raise ValueError(f"SWE-bench prediction result is not completed: {path}")
        metadata_value = result.get("metadata")
        digest = _sha256(path.read_bytes())
        if (
            not isinstance(metadata_value, Mapping)
            or metadata_value.get("prediction_sha256") != digest
        ):
            raise ValueError(f"SWE-bench prediction hash does not match: {path}")
        instance_ids.add(instance_id)
        records.append(
            {
                "instance_id": instance_id,
                "model_patch": model_patch,
                "model_name_or_path": model_name,
            }
        )
    if not records:
        raise ValueError("evaluation contains no SWE-bench predictions")
    records.sort(key=lambda value: str(value["instance_id"]))
    content = "".join(
        json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n"
        for value in records
    ).encode("utf-8")
    _prediction_collection_target(root, destination)
    atomic_bytes(target, content)
    return len(records)


def _prediction_collection_target(root: Path, destination: Path) -> Path:
    expanded = _require_no_symlink(
        destination.expanduser(), "SWE-bench predictions destination"
    )
    if expanded.parent.resolve() != root:
        raise ValueError(
            "SWE-bench predictions must be a direct child of the evaluation"
        )
    target = root / expanded.name
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError("SWE-bench predictions destination must be a regular file")
    return target


def inspect_swebench_grade_inputs(
    *, predictions: Path, dataset: Path
) -> Mapping[str, Any]:
    """Validate local official-grader inputs and bind visible task prompts."""

    prediction_path = predictions.expanduser().resolve()
    dataset_path = dataset.expanduser().resolve()
    if dataset_path.suffix.casefold() not in {".json", ".jsonl"}:
        raise ValueError("SWE-bench grading requires a local .json or .jsonl dataset")
    tasks = load_swebench(dataset_path)
    task_prompts: dict[str, str] = {}
    task_data: dict[str, str] = {}
    for task in tasks:
        task_prompts[task.task_id] = _sha256(task.prompt.encode("utf-8"))
        task_data[task.task_id] = _sha256(canonical_bytes(task.data))
    prediction_ids: set[str] = set()
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = strict_json_loads(line)
        _require_mapping(value, f"SWE-bench prediction line {number}")
        instance_id = _require_grader_component(
            value.get("instance_id"),
            f"SWE-bench prediction line {number} instance_id",
        )
        if instance_id in prediction_ids:
            raise ValueError(f"duplicate SWE-bench prediction {instance_id!r}")
        prediction_ids.add(instance_id)
    if not prediction_ids:
        raise ValueError("SWE-bench grader input contains no predictions")
    missing = sorted(prediction_ids.difference(task_prompts))
    if missing:
        raise ValueError(
            "SWE-bench predictions are missing from the local dataset: "
            + ", ".join(missing)
        )
    selected = sorted(prediction_ids)
    return {
        "predictions": immutable_file_identity(
            prediction_path, label="SWE-bench predictions"
        ),
        "dataset": immutable_file_identity(dataset_path, label="SWE-bench dataset"),
        "prediction_count": len(prediction_ids),
        "dataset_count": len(task_prompts),
        "task_prompt_sha256": {key: task_prompts[key] for key in selected},
        "task_data_sha256": {key: task_data[key] for key in selected},
    }


__all__ = [
    "SWEBENCH_BASH_ENV",
    "SWEBENCH_REVISION",
    "SWEBENCH_SOURCE_SHA256",
    "SWEBENCH_TAG",
    "SWEBENCH_VERSION",
    "SWEBENCH_WORKDIR",
    "SWEbenchImageBinding",
    "apptainer_swe_environment",
    "collect_predictions",
    "docker_swe_environment",
    "inspect_swebench_grade_inputs",
    "load_swebench",
    "official_grader_argv",
    "prepare_swebench_image_bindings",
    "resolve_swebench_image_binding",
    "run_swebench_task",
    "swebench_doctor",
    "swebench_grader_image_name",
    "swebench_grader_source_identity",
    "swebench_image_name",
    "verify_swebench_grader_images",
]
