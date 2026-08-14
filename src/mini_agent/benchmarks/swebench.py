"""SWE-bench adapter: image binding, task loading, generation, grader contracts.

Container provisioning lives in :mod:`mini_agent.runtimes` and the bash tool in
:mod:`mini_agent.environments.bash`; everything here is what makes a run
*SWE-bench*: the official image-name rule, the pinned upstream revision, the
per-task Git baseline, and the official grader's prediction/source contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
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
from ..execution import RunContext
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
from ..types import (
    _require_bool,
    _require_mapping,
    _require_positive_int,
    _require_str,
    strict_json_loads,
)
from ..storage import (
    atomic_bytes,
    atomic_json,
)
from .base import (
    TeamOptions,
    run_benchmark_team,
    BenchmarkTask,
    EvaluationOutcome,
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


def _require_grader_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_GRADER_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be one path-safe component")
    return value
_INVALID_SDK = "SWE-bench grader Docker SDK identity is invalid"
_INVALID_SOURCE = "official SWE-bench package source tree is invalid"
_UNSAFE_ENTRY = "official SWE-bench package contains an unsafe {kind}"




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
    runtime: str,
    model_name: str = "mini-agent",
    scratch_root: Path | None = None,
    apptainer_image_cache: Path | None = None,
    image_binding: SWEbenchImageBinding | None = None,
    overlay_size_mib: int = 16 * 1024,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    options: TeamOptions,
) -> EvaluationOutcome:
    """Run one task in its official image and export the agent's patch.

    Nothing here scores anything: the patch and the prediction record are the
    whole output, and only the official grader turns them into a result.
    """

    _require_str(model_name, "model_name")
    _require_grader_component(task.task_id, "SWE-bench task instance_id")
    _require_grader_component(
        model_name.replace("/", "__"), "SWE-bench model_name_or_path"
    )
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

    team = await run_benchmark_team(
        task,
        context,
        environment_factory=environment_for,
        options=options,
    )
    if not isinstance(team.state, SWEPatchState):
        raise RuntimeError("root SWE agent produced no patch state")
    patch = team.state.patch

    atomic_bytes(directory / "patch.diff", patch)
    atomic_json(
        directory / "prediction.json",
        {
            "instance_id": task.task_id,
            "model_patch": patch.decode("utf-8", errors="strict"),
            "model_name_or_path": model_name,
        },
    )
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=team.require().answer,
        metadata={
            **team.metadata(),
            "runtime": runtime,
            "environments": {
                agent_id: dict(base.provenance())
                for agent_id, base in team.bases().items()
            },
            "patch_bytes": len(patch),
            "patch_sha256": _sha256(patch),
            "prediction_sha256": _sha256(
                (directory / "prediction.json").read_bytes()
            ),
        },
    )




__all__ = [
    "SWEBENCH_BASH_ENV",
    "SWEBENCH_REVISION",
    "SWEBENCH_SOURCE_SHA256",
    "SWEBENCH_TAG",
    "SWEBENCH_VERSION",
    "SWEBENCH_WORKDIR",
    "SWEbenchImageBinding",
    "apptainer_swe_environment",
    "docker_swe_environment",
    "load_swebench",
    "prepare_swebench_image_bindings",
    "resolve_swebench_image_binding",
    "run_swebench_task",
    "swebench_doctor",
    "swebench_image_name",
]
