from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._hash import stable_file_sha256
from ..types import (
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    strict_json_loads,
)
from .base import (
    BaseEnvironment,
    combine_lifecycle_errors,
    complete_in_thread,
    raise_lifecycle_errors,
)
from .swe import (
    DEFAULT_MAX_PATCH_BYTES,
    LocalProcessRunner,
    ProcessResult,
    ProcessRunner,
    SWEPatchState,
    _atomic_write,
)


SWEBENCH_REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
SWEBENCH_TAG = "v4.1.0"
SWEBENCH_WORKDIR = "/testbed"
SWEBENCH_BASH_ENV = "/root/.bashrc"
DEFAULT_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_ARCHIVE_CONTAINER_PATH = "/tmp/mini-agent-workspace.tar.gz"
_SAFE_CONTAINER_PART = re.compile(r"[^a-z0-9_.-]+")
_OWNED_DOCKER = object()
_OWNED_APPTAINER = object()


def _runtime_argv(value: Sequence[str], label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not value
        or not all(
            isinstance(item, str) and item and "\x00" not in item for item in value
        )
    ):
        raise ValueError(f"{label} argv must contain non-empty strings")
    return tuple(value)


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _docker_security_is_rootless(output: str) -> bool:
    try:
        value = strict_json_loads(output)
    except ValueError:
        return False
    return isinstance(value, list) and any(
        item == "name=rootless" for item in value if isinstance(item, str)
    )


@dataclass(frozen=True)
class SWEArchiveState:
    """Whole-workspace state for images without an inspectable Git baseline."""

    base_identity: str
    archive: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.base_identity, str) or not self.base_identity:
            raise ValueError("SWE archive base identity must be non-empty")
        if not isinstance(self.archive, bytes):
            raise ValueError("SWE archive must be bytes")


def _benchmark_identity(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    if value is None:
        return {
            "benchmark": "swe_bench",
            "benchmark_revision": SWEBENCH_REVISION,
            "benchmark_tag": SWEBENCH_TAG,
        }
    if not isinstance(value, Mapping) or not value:
        raise ValueError("container benchmark identity must be a non-empty object")
    resolved: dict[str, str] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, str)
            or not item
            or "\x00" in name
            or "\x00" in item
        ):
            raise ValueError("container benchmark identity must contain strings")
        resolved[name] = item
    return resolved


def _container_workdir(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or value.strip() != value
    ):
        raise ValueError("container workdir must be an absolute path")
    return value


@dataclass(frozen=True)
class SWEbenchDoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SWEbenchDoctorReport:
    ok: bool
    runtime: tuple[str, ...]
    checks: tuple[SWEbenchDoctorCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "runtime": list(self.runtime),
            "checks": [asdict(check) for check in self.checks],
        }


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
        if (
            not isinstance(self.requested, str)
            or not self.requested
            or self.requested.startswith("-")
            or "\x00" in self.requested
        ):
            raise ValueError("SWE-bench requested image is invalid")
        if not isinstance(self.execution_ref, str) or not self.execution_ref:
            raise ValueError("SWE-bench image execution reference is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.identity):
            raise ValueError("SWE-bench image identity must be a SHA-256 digest")
        if self.runtime == "docker" and self.execution_ref != self.identity:
            raise ValueError("Docker must execute the resolved image ID")
        if self.runtime == "apptainer":
            selected = Path(self.execution_ref)
            if not selected.is_absolute():
                raise ValueError("Apptainer must execute an absolute local image path")

    def manifest_identity(self) -> Mapping[str, str]:
        return {
            "runtime": self.runtime,
            "requested": self.requested,
            "identity": self.identity,
        }


def swebench_image_name(instance: Mapping[str, Any]) -> str:
    explicit = instance.get("image_name") or instance.get("docker_image")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit or explicit.startswith("-"):
            raise ValueError("SWE-bench image name must be a non-empty Docker image")
        if "\x00" in explicit:
            raise ValueError("SWE-bench image name contains a NUL byte")
        return explicit
    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("SWE-bench instance requires instance_id")
    docker_id = instance_id.replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest"


def _expected_base_commit(instance: Mapping[str, Any]) -> str | None:
    value = instance.get("base_commit")
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("SWE-bench base_commit must be a full Git commit")
    return value.casefold()


async def resolve_swebench_image_binding(
    instance: Mapping[str, Any],
    *,
    runtime: str,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    apptainer_image_cache: Path | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 60.0,
    max_output_bytes: int = 256 * 1024,
) -> SWEbenchImageBinding:
    """Resolve a mutable task image reference to bytes before inference."""

    if not isinstance(instance, Mapping):
        raise ValueError("SWE-bench instance must be an object")
    if runtime not in {"docker", "apptainer"}:
        raise ValueError("SWE-bench runtime must be docker or apptainer")
    resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
    resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
    if runner is not None and not callable(getattr(runner, "run", None)):
        raise ValueError("runner must expose run")
    process_runner = runner or LocalProcessRunner()
    requested = swebench_image_name(instance)
    if runtime == "docker":
        resolved_runtime = _runtime_argv(container_runtime, "container runtime")
        image_id = await _docker_image_id(
            requested,
            runtime=resolved_runtime,
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

    if (
        not isinstance(apptainer_executable, str)
        or not apptainer_executable
        or "\x00" in apptainer_executable
    ):
        raise ValueError("Apptainer executable must be non-empty")
    source = "docker://" + requested.removeprefix("docker.io/")
    selected = await _materialize_apptainer_image(
        source,
        executable=apptainer_executable,
        runner=process_runner,
        cache=apptainer_image_cache,
        timeout_seconds=max(resolved_timeout, 1800.0),
        max_output_bytes=resolved_output,
    )
    identity = await _apptainer_image_identity(
        selected,
    )
    return SWEbenchImageBinding(
        runtime="apptainer",
        requested=source,
        identity=identity,
        execution_ref=str(Path(selected).expanduser().resolve()),
    )


def _container_name(instance_id: str) -> str:
    safe = _SAFE_CONTAINER_PART.sub("-", instance_id.lower()).strip("-.")
    safe = safe[:48] or "instance"
    return f"mini-agent-swe-{safe}-{uuid.uuid4().hex[:12]}"


def _docker_exec_argv(
    runtime: Sequence[str],
    container_name: str,
    command: str,
    workdir: str = SWEBENCH_WORKDIR,
) -> tuple[str, ...]:
    return (
        *runtime,
        "exec",
        "--workdir",
        workdir,
        "--env",
        f"BASH_ENV={SWEBENCH_BASH_ENV}",
        container_name,
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        command,
    )


def _apptainer_exec_argv(
    *,
    executable: str,
    overlay: Path,
    image: str,
    command: str,
    binds: Sequence[tuple[Path, str]] = (),
) -> tuple[str, ...]:
    argv = [
        executable,
        "--silent",
        "exec",
        "--cleanenv",
        "--containall",
        "--fakeroot",
        "--overlay",
        str(overlay),
        "--pwd",
        SWEBENCH_WORKDIR,
        "--env",
        f"PAGER=cat,MANPAGER=cat,TQDM_DISABLE=1,BASH_ENV={SWEBENCH_BASH_ENV}",
    ]
    for source, target in binds:
        argv.extend(("--bind", f"{source}:{target}:ro"))
    argv.extend(
        (
            image,
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        )
    )
    return tuple(argv)


async def _docker_image_id(
    image: str,
    *,
    runtime: Sequence[str],
    runner: ProcessRunner,
    timeout_seconds: float,
    max_output_bytes: int,
    pull_if_missing: bool,
) -> str:
    async def inspect_image() -> ProcessResult:
        return await runner.run(
            (*runtime, "image", "inspect", "--format", "{{.Id}}", image),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    inspected = await inspect_image()
    if inspected.timed_out:
        raise RuntimeError("Docker image inspection timed out")
    if inspected.returncode != 0 and pull_if_missing:
        pulled = await runner.run(
            (*runtime, "pull", image),
            timeout_seconds=max(timeout_seconds, 1800.0),
            max_output_bytes=max_output_bytes,
        )
        if pulled.timed_out or pulled.returncode != 0:
            raise RuntimeError(
                "could not materialize SWE-bench Docker image: "
                + (pulled.text() or f"exit code {pulled.returncode}")
            )
        inspected = await inspect_image()
    if inspected.timed_out or inspected.returncode != 0:
        raise RuntimeError(
            "could not inspect SWE-bench Docker image: "
            + (inspected.text() or f"exit code {inspected.returncode}")
        )
    image_id = inspected.text().strip().casefold()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeError("container runtime returned an invalid Docker image ID")
    return image_id


async def swebench_doctor(
    *,
    runtime: Sequence[str] = ("docker",),
    image: str | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 64 * 1024,
    require_rootless: bool = True,
) -> SWEbenchDoctorReport:
    """Perform non-mutating container-runtime and optional image checks."""

    resolved_runtime = _runtime_argv(runtime, "container runtime")
    resolved_timeout = _positive_number(timeout_seconds, "doctor timeout_seconds")
    resolved_max_output = _positive_int(max_output_bytes, "doctor max_output_bytes")
    if not isinstance(require_rootless, bool):
        raise ValueError("require_rootless must be boolean")
    if image is not None and (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or "\x00" in image
    ):
        raise ValueError("doctor image must be a valid Docker image name")
    if runner is not None and not callable(getattr(runner, "run", None)):
        raise ValueError("runner must expose run")
    process_runner = runner or LocalProcessRunner()
    requests: list[tuple[str, tuple[str, ...]]] = [
        (
            "runtime_version",
            (*resolved_runtime, "version", "--format", "{{.Server.Version}}"),
        ),
        (
            "daemon_platform",
            (*resolved_runtime, "info", "--format", "{{.OSType}}/{{.Architecture}}"),
        ),
    ]
    if require_rootless:
        requests.append(
            (
                "rootless_security",
                (*resolved_runtime, "info", "--format", "{{json .SecurityOptions}}"),
            )
        )
    if image is not None:
        requests.append(
            (
                "image_available",
                (*resolved_runtime, "image", "inspect", "--format", "{{.Id}}", image),
            )
        )
    checks: list[SWEbenchDoctorCheck] = []
    for name, argv in requests:
        try:
            result = await process_runner.run(
                argv,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_max_output,
            )
            ok = not result.timed_out and result.returncode == 0
            detail = result.text().strip()
            if name == "rootless_security" and not _docker_security_is_rootless(
                detail
            ):
                ok = False
                detail = detail or "daemon did not report rootless mode"
            elif name == "daemon_platform" and not re.fullmatch(
                r"linux/[A-Za-z0-9_.-]+", detail
            ):
                ok = False
                detail = detail or "daemon is not a Linux platform"
            elif name == "image_available" and not re.fullmatch(
                r"sha256:[0-9a-fA-F]{64}", detail
            ):
                ok = False
                detail = detail or "runtime returned no image identity"
            elif name == "runtime_version" and not detail:
                ok = False
                detail = "runtime returned no version"
            if result.timed_out:
                detail = "timed out"
            elif not detail:
                detail = f"exit code {result.returncode}"
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        checks.append(SWEbenchDoctorCheck(name=name, ok=ok, detail=detail))
    return SWEbenchDoctorReport(
        ok=all(check.ok for check in checks),
        runtime=resolved_runtime,
        checks=tuple(checks),
    )


class DockerSWEEnvironment(BaseEnvironment):
    """One bash tool in a persistent SWE-bench instance container."""

    def __init__(
        self,
        *,
        image: str,
        container_name: str,
        image_id: str,
        runtime: Sequence[str],
        runner: ProcessRunner,
        base_commit: str | None,
        platform: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        workdir: str = SWEBENCH_WORKDIR,
        network_disabled: bool = False,
        benchmark_identity: Mapping[str, Any] | None = None,
        _ownership_token: object | None = None,
    ) -> None:
        if _ownership_token is not _OWNED_DOCKER:
            raise ValueError("Docker SWE environments are created only by create()")
        if not isinstance(image, str) or not image or "\x00" in image:
            raise ValueError("Docker image must be non-empty")
        if (
            not isinstance(container_name, str)
            or not container_name.startswith("mini-agent-swe-")
            or not re.fullmatch(r"[a-z0-9_.-]+", container_name)
        ):
            raise ValueError("Docker container name is not mini-agent owned")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("Docker image identity must be non-empty")
        if base_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", base_commit):
            raise ValueError("SWE-bench base commit must be a full Git commit")
        if not isinstance(network_disabled, bool):
            raise ValueError("network_disabled must be boolean")
        resolved_runtime = _runtime_argv(runtime, "container runtime")
        if not callable(getattr(runner, "run", None)):
            raise ValueError("runner must expose run")
        if platform is not None and (
            not isinstance(platform, str)
            or not platform
            or platform.startswith("-")
            or "\x00" in platform
        ):
            raise ValueError("container platform is invalid")
        resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
        resolved_patch = _positive_int(max_patch_bytes, "max_patch_bytes")
        self.image = image
        self.container_name = container_name
        self.image_id = image_id
        self.runtime = resolved_runtime
        self.runner = runner
        self.base_commit = base_commit
        self.platform = platform
        self.timeout_seconds = resolved_timeout
        self.max_output_bytes = resolved_output
        self.max_patch_bytes = resolved_patch
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.workdir = _container_workdir(workdir)
        self.network_disabled = network_disabled
        self.benchmark_identity = _benchmark_identity(benchmark_identity)
        self._closed = False

    @classmethod
    async def create(
        cls,
        instance: Mapping[str, Any],
        *,
        image_binding: SWEbenchImageBinding | None = None,
        runtime: Sequence[str] = ("docker",),
        platform: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        workdir: str = SWEBENCH_WORKDIR,
        network_disabled: bool = False,
        require_git_baseline: bool = True,
        benchmark_identity: Mapping[str, Any] | None = None,
        runner: ProcessRunner | None = None,
    ) -> "DockerSWEEnvironment":
        if not isinstance(instance, Mapping):
            raise ValueError("SWE-bench instance must be an object")
        if not isinstance(network_disabled, bool) or not isinstance(
            require_git_baseline, bool
        ):
            raise ValueError("network_disabled and require_git_baseline must be bool")
        resolved_workdir = _container_workdir(workdir)
        resolved_identity = _benchmark_identity(benchmark_identity)
        resolved_runtime = _runtime_argv(runtime, "container runtime")
        if platform is not None and (
            not isinstance(platform, str)
            or not platform
            or platform.startswith("-")
            or "\x00" in platform
        ):
            raise ValueError("container platform is invalid")
        resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
        resolved_patch = _positive_int(max_patch_bytes, "max_patch_bytes")
        resolved_archive = _positive_int(max_archive_bytes, "max_archive_bytes")
        image = swebench_image_name(instance)
        expected_base_commit = _expected_base_commit(instance)
        if expected_base_commit is not None and not require_git_baseline:
            raise ValueError(
                "a task base_commit cannot be verified without a Git baseline"
            )
        instance_id = instance.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("SWE-bench instance requires instance_id")
        if runner is not None and not callable(getattr(runner, "run", None)):
            raise ValueError("runner must expose run")
        process_runner = runner or LocalProcessRunner()
        if image_binding is None:
            image_id = await _docker_image_id(
                image,
                runtime=resolved_runtime,
                runner=process_runner,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
                pull_if_missing=True,
            )
        else:
            if image_binding.runtime != "docker":
                raise ValueError("Docker received a non-Docker image binding")
            if image_binding.requested != image:
                raise ValueError("Docker image binding does not match the task image")
            image_id = image_binding.identity
        security = await process_runner.run(
            (
                *resolved_runtime,
                "info",
                "--format",
                "{{json .SecurityOptions}}",
            ),
            timeout_seconds=resolved_timeout,
            max_output_bytes=resolved_output,
        )
        if (
            security.timed_out
            or security.returncode != 0
            or not _docker_security_is_rootless(security.text())
        ):
            raise RuntimeError("Docker SWE-bench execution requires a rootless daemon")
        name = _container_name(instance_id)
        argv = [
            *resolved_runtime,
            "run",
            "--detach",
            "--name",
            name,
            "--workdir",
            resolved_workdir,
            "--label",
            "mini-agent.swebench=true",
            "--env",
            "HOME=/root",
            "--env",
            "PAGER=cat",
            "--env",
            "MANPAGER=cat",
            "--env",
            "TQDM_DISABLE=1",
        ]
        if network_disabled:
            argv.extend(("--network", "none"))
        if platform is not None:
            argv.extend(("--platform", platform))
        argv.extend((image_id, "sleep", "infinity"))
        try:
            started = await process_runner.run(
                argv,
                timeout_seconds=max(resolved_timeout, 300.0),
                max_output_bytes=resolved_output,
            )
            if started.timed_out or started.returncode != 0:
                raise RuntimeError(
                    "could not start SWE-bench container: " + started.text()
                )
            inspected = await process_runner.run(
                (*resolved_runtime, "inspect", "--format", "{{.Image}}", name),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            if inspected.timed_out or inspected.returncode != 0:
                raise RuntimeError(
                    "could not resolve running SWE-bench image identity: "
                    + inspected.text()
                )
            running_image_id = inspected.text().strip().casefold()
            if running_image_id != image_id:
                raise RuntimeError(
                    "running SWE-bench container image does not match its binding"
                )
            base_commit: str | None = None
            if require_git_baseline:
                baseline = await process_runner.run(
                    _docker_exec_argv(
                        resolved_runtime,
                        name,
                        "git rev-parse HEAD && "
                        "test -z \"$(git status --porcelain=v1 "
                        "--untracked-files=all)\"",
                        resolved_workdir,
                    ),
                    timeout_seconds=resolved_timeout,
                    max_output_bytes=resolved_output,
                )
                base_commit = baseline.text().strip().casefold()
                if (
                    baseline.timed_out
                    or baseline.returncode != 0
                    or not re.fullmatch(r"[0-9a-f]{40}", base_commit)
                ):
                    raise RuntimeError(
                        "SWE-bench image has no usable Git baseline: "
                        + baseline.text()
                    )
                if expected_base_commit is not None:
                    ancestry = await process_runner.run(
                        _docker_exec_argv(
                            resolved_runtime,
                            name,
                            "git merge-base --is-ancestor "
                            f"{expected_base_commit} {base_commit}",
                            resolved_workdir,
                        ),
                        timeout_seconds=resolved_timeout,
                        max_output_bytes=resolved_output,
                    )
                    if ancestry.timed_out or ancestry.returncode != 0:
                        raise RuntimeError(
                            "SWE-bench image does not contain task base_commit"
                        )
            return cls(
                image=image,
                container_name=name,
                image_id=image_id,
                runtime=resolved_runtime,
                runner=process_runner,
                base_commit=base_commit,
                platform=platform,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
                max_patch_bytes=resolved_patch,
                max_archive_bytes=resolved_archive,
                workdir=resolved_workdir,
                network_disabled=network_disabled,
                benchmark_identity=resolved_identity,
                _ownership_token=_OWNED_DOCKER,
            )
        except BaseException as operation_error:
            cleanup_error: BaseException | None = None
            try:
                removed = await process_runner.run(
                    (*resolved_runtime, "rm", "--force", name),
                    timeout_seconds=resolved_timeout,
                    max_output_bytes=resolved_output,
                )
                if removed.timed_out or removed.returncode != 0:
                    raise RuntimeError(
                        "container removal failed: "
                        + (removed.text() or f"exit code {removed.returncode}")
                    )
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "SWE-bench container startup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="bash",
                description=(
                    f"Run one bash command in the persistent {self.workdir} "
                    "workspace. Each call starts a new shell."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
        )

    async def _exec(
        self, command: str, *, max_output_bytes: int | None = None
    ) -> ProcessResult:
        if self._closed:
            raise RuntimeError("SWE-bench environment is closed")
        return await self.runner.run(
            _docker_exec_argv(
                self.runtime, self.container_name, command, self.workdir
            ),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "bash":
            raise InvalidAction(f"unsupported SWE tool {action.name!r}")
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise InvalidAction("bash command must be a non-empty string")
        result = await self._exec(command)
        if result.timed_out:
            operation_error = RuntimeError(
                "SWE-bench command timed out; its container was destroyed"
            )
            cleanup_error: BaseException | None = None
            try:
                await self.close()
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors("SWE-bench command", operation_error, cleanup_error)
            raise AssertionError("unreachable")
        output = result.text()
        return ToolExecution(
            output=output,
            is_error=result.returncode != 0,
            metadata={
                "exit_code": result.returncode,
                "output_bytes": result.total_output_bytes,
                "output_truncated": result.truncated,
                "timed_out": result.timed_out,
            },
        )

    async def export_patch(self, destination: Path | None = None) -> bytes:
        if self.base_commit is None:
            raise RuntimeError(
                "this container has no Git baseline; export_archive() is the "
                "only workspace export"
            )
        # Do not force-add ignored build artifacts baked into official images.
        # Only tracked changes and ordinary untracked files belong to the agent.
        staged = await self._exec("git add --all -- .")
        if staged.timed_out or staged.returncode != 0:
            raise RuntimeError("could not stage workspace changes: " + staged.text())
        patch = await self._exec(
            "git diff --cached --binary --full-index --no-ext-diff "
            f"--no-textconv --no-renames {self.base_commit} -- .",
            max_output_bytes=self.max_patch_bytes,
        )
        if patch.timed_out:
            raise RuntimeError("git diff timed out")
        if patch.truncated:
            raise RuntimeError(
                f"SWE patch exceeded the configured {self.max_patch_bytes}-byte limit"
            )
        if patch.returncode != 0:
            raise RuntimeError("could not capture SWE patch: " + patch.text())
        if destination is not None:
            target = _patch_destination(destination)
            await complete_in_thread(_atomic_write, target, patch.output)
        return patch.output

    async def export_archive(self, destination: Path | None = None) -> bytes:
        """Export the whole workspace tree as one gzip tar archive."""

        created = await self._exec(
            f"rm -f {_ARCHIVE_CONTAINER_PATH} && "
            f"tar -czf {_ARCHIVE_CONTAINER_PATH} -C {self.workdir} . && "
            f"stat -c %s {_ARCHIVE_CONTAINER_PATH}"
        )
        if created.timed_out or created.returncode != 0:
            raise RuntimeError(
                "could not archive the container workspace: " + created.text()
            )
        reported = created.text().strip().rsplit("\n", 1)[-1].strip()
        if not reported.isdigit() or int(reported) > self.max_archive_bytes:
            raise RuntimeError(
                "container workspace archive exceeded the configured "
                f"{self.max_archive_bytes}-byte limit"
            )
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-archive-"))
        operation_error: BaseException | None = None
        content = b""
        try:
            local = root / "workspace.tar.gz"
            copied = await self.runner.run(
                (
                    *self.runtime,
                    "cp",
                    f"{self.container_name}:{_ARCHIVE_CONTAINER_PATH}",
                    str(local),
                ),
                timeout_seconds=max(self.timeout_seconds, 300.0),
                max_output_bytes=self.max_output_bytes,
            )
            if copied.timed_out or copied.returncode != 0:
                raise RuntimeError(
                    "could not copy the container workspace archive: "
                    + copied.text()
                )
            content = await complete_in_thread(local.read_bytes)
            if len(content) > self.max_archive_bytes:
                raise RuntimeError(
                    "container workspace archive exceeded the configured "
                    f"{self.max_archive_bytes}-byte limit"
                )
            if destination is not None:
                await complete_in_thread(
                    _atomic_write, _patch_destination(destination), content
                )
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await complete_in_thread(shutil.rmtree, root)
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors(
            "container workspace archive", operation_error, cleanup_error
        )
        return content

    async def export_state(self) -> SWEPatchState | SWEArchiveState:
        if self.base_commit is None:
            return SWEArchiveState(
                f"{self.image_id}@{self.workdir}", await self.export_archive()
            )
        return SWEPatchState(
            f"{self.image_id}@{self.base_commit}", await self.export_patch()
        )

    async def adopt_state(self, state: Any) -> None:
        if self.base_commit is None:
            await self._adopt_archive_state(state)
            return
        if (
            not isinstance(state, SWEPatchState)
            or state.base_identity != f"{self.image_id}@{self.base_commit}"
        ):
            raise ProtocolError("SWE state came from a different container image")
        if len(state.patch) > self.max_patch_bytes:
            raise ProtocolError("SWE state exceeds the patch limit")
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-adopt-"))
        target = root / "target.patch"
        prior = root / "prior.patch"
        operation_error: BaseException | None = None
        rollback_error: BaseException | None = None
        previous: bytes | None = None
        mutated = False
        try:
            await complete_in_thread(_atomic_write, target, state.patch)
            previous = await self.export_patch()
            await complete_in_thread(_atomic_write, prior, previous)
            mutated = True
            await self._reset()
            if state.patch:
                applied = await self._copy_and_apply(target, "target.patch")
                if applied.timed_out or applied.returncode != 0:
                    raise ProtocolError(
                        "SWE state could not be applied: " + applied.text()
                    )
        except BaseException as exc:
            operation_error = exc
        if operation_error is not None and mutated:
            try:
                await self._reset()
                if previous:
                    restored = await self._copy_and_apply(prior, "prior.patch")
                    if restored.timed_out or restored.returncode != 0:
                        raise RuntimeError(
                            "prior SWE state could not be restored: " + restored.text()
                        )
            except BaseException as exc:
                rollback_error = exc
        cleanup_error: BaseException | None = rollback_error
        try:
            await complete_in_thread(shutil.rmtree, root)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_error = combine_lifecycle_errors(cleanup_error, exc)
        raise_lifecycle_errors("SWE state adoption", operation_error, cleanup_error)

    async def _adopt_archive_state(self, state: Any) -> None:
        if (
            not isinstance(state, SWEArchiveState)
            or state.base_identity != f"{self.image_id}@{self.workdir}"
        ):
            raise ProtocolError("SWE state came from a different container image")
        if len(state.archive) > self.max_archive_bytes:
            raise ProtocolError("SWE state exceeds the workspace archive limit")
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-adopt-"))
        target = root / "target.tar.gz"
        prior = root / "prior.tar.gz"
        operation_error: BaseException | None = None
        rollback_error: BaseException | None = None
        mutated = False
        try:
            await complete_in_thread(_atomic_write, prior, await self.export_archive())
            await complete_in_thread(_atomic_write, target, state.archive)
            mutated = True
            await self._replace_workspace(target, "mini-agent-target.tar.gz")
        except BaseException as exc:
            operation_error = exc
        if operation_error is not None and mutated:
            try:
                await self._replace_workspace(prior, "mini-agent-prior.tar.gz")
            except BaseException as exc:
                rollback_error = exc
        cleanup_error: BaseException | None = rollback_error
        try:
            await complete_in_thread(shutil.rmtree, root)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_error = combine_lifecycle_errors(cleanup_error, exc)
        raise_lifecycle_errors("SWE state adoption", operation_error, cleanup_error)

    async def _replace_workspace(self, source: Path, name: str) -> None:
        copied = await self.runner.run(
            (*self.runtime, "cp", str(source), f"{self.container_name}:/tmp/{name}"),
            timeout_seconds=max(self.timeout_seconds, 300.0),
            max_output_bytes=self.max_output_bytes,
        )
        if copied.timed_out or copied.returncode != 0:
            raise ProtocolError("SWE state could not be copied: " + copied.text())
        applied = await self._exec(
            f"find {self.workdir} -mindepth 1 -delete && "
            f"tar -xzf /tmp/{name} -C {self.workdir} && rm -f /tmp/{name}"
        )
        if applied.timed_out or applied.returncode != 0:
            raise ProtocolError("SWE state could not be applied: " + applied.text())

    async def _reset(self) -> None:
        # Keep ignored build products supplied by the benchmark image. They are
        # part of its runtime, even though they are not part of the git tree.
        result = await self._exec(
            f"git reset --hard {self.base_commit} && git clean -ffd -q"
        )
        if result.timed_out or result.returncode != 0:
            raise RuntimeError("could not reset SWE container: " + result.text())

    async def _copy_and_apply(self, source: Path, name: str) -> ProcessResult:
        copied = await self.runner.run(
            (*self.runtime, "cp", str(source), f"{self.container_name}:/tmp/{name}"),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        if copied.timed_out or copied.returncode != 0:
            return copied
        return await self._exec(f"git apply --binary --index /tmp/{name}")

    async def close(self) -> None:
        if self._closed:
            return
        result = await self.runner.run(
            (*self.runtime, "rm", "--force", self.container_name),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        if result.timed_out or result.returncode != 0:
            raise RuntimeError("could not remove SWE-bench container: " + result.text())
        self._closed = True

    def provenance(self) -> dict[str, object]:
        return {
            "application": "swe",
            **self.benchmark_identity,
            "tools": ["bash"],
            "container_runtime": list(self.runtime),
            "container_image": self.image,
            "container_image_id": self.image_id,
            "container_platform": self.platform,
            "base_commit": self.base_commit,
            "rootless_daemon_required": True,
            "network_disabled": self.network_disabled,
            "workdir": self.workdir,
            "host_credentials_mounted": False,
            "patch_export": (
                "git_diff_binary"
                if self.base_commit is not None
                else "workspace_tar_gz"
            ),
        }

    def resource_identity(self) -> str:
        return f"swe-container:{self.container_name}"


class ApptainerSWEEnvironment(BaseEnvironment):
    """Persistent SWE-bench workspace backed by a private fakeroot overlay."""

    def __init__(
        self,
        *,
        image: str,
        image_source: str,
        image_identity: str,
        base_commit: str,
        overlay: Path,
        owned_root: Path,
        executable: str,
        runner: ProcessRunner,
        timeout_seconds: float,
        max_output_bytes: int,
        max_patch_bytes: int,
        _ownership_token: object | None = None,
    ) -> None:
        if _ownership_token is not _OWNED_APPTAINER:
            raise ValueError("Apptainer SWE environments are created only by create()")
        if not isinstance(image, str) or not image or "\x00" in image:
            raise ValueError("Apptainer image must be non-empty")
        if (
            not isinstance(image_source, str)
            or not image_source
            or "\x00" in image_source
        ):
            raise ValueError("Apptainer image source must be non-empty")
        if not isinstance(image_identity, str) or not image_identity:
            raise ValueError("Apptainer image identity must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
            raise ValueError("SWE-bench base commit must be a full Git commit")
        resolved_root = owned_root.resolve()
        resolved_overlay = overlay.resolve()
        if (
            owned_root.is_symlink()
            or not resolved_root.is_dir()
            or not resolved_root.name.startswith("mini-agent-swe-apptainer-")
            or resolved_overlay != resolved_root / "overlay.img"
        ):
            raise ValueError("owned Apptainer root has an invalid internal layout")
        if not isinstance(executable, str) or not executable or "\x00" in executable:
            raise ValueError("Apptainer executable must be non-empty")
        if not callable(getattr(runner, "run", None)):
            raise ValueError("runner must expose run")
        self.image = image
        self.image_source = image_source
        self.image_identity = image_identity
        self.base_commit = base_commit
        self.overlay = resolved_overlay
        self.owned_root = resolved_root
        self.executable = executable
        self.runner = runner
        self.timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
        self.max_output_bytes = _positive_int(max_output_bytes, "max_output_bytes")
        self.max_patch_bytes = _positive_int(max_patch_bytes, "max_patch_bytes")
        self._closed = False

    @classmethod
    async def create(
        cls,
        instance: Mapping[str, Any],
        *,
        image: str | None = None,
        image_binding: SWEbenchImageBinding | None = None,
        executable: str = "apptainer",
        scratch_root: Path | None = None,
        image_cache: Path | None = None,
        overlay_size_mib: int = 16 * 1024,
        timeout_seconds: float = 60,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        runner: ProcessRunner | None = None,
    ) -> "ApptainerSWEEnvironment":
        if not isinstance(instance, Mapping):
            raise ValueError("SWE-bench instance must be an object")
        if image is not None and not isinstance(image, str):
            raise ValueError("Apptainer image must be a string or None")
        if not isinstance(executable, str) or not executable or "\x00" in executable:
            raise ValueError("Apptainer executable must be non-empty")
        requested = image or "docker://" + swebench_image_name(instance).removeprefix(
            "docker.io/"
        )
        expected_base_commit = _expected_base_commit(instance)
        if not requested or requested.startswith("-") or "\x00" in requested:
            raise ValueError("Apptainer image reference is invalid")
        if (
            not isinstance(overlay_size_mib, int)
            or isinstance(overlay_size_mib, bool)
            or overlay_size_mib < 1024
        ):
            raise ValueError("overlay_size_mib must be at least 1024")
        resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
        resolved_patch = _positive_int(max_patch_bytes, "max_patch_bytes")
        if runner is not None and not callable(getattr(runner, "run", None)):
            raise ValueError("runner must expose run")
        root_parent = (
            None if scratch_root is None else scratch_root.expanduser().resolve()
        )
        if root_parent is not None:
            root_parent.mkdir(parents=True, exist_ok=True)
        owned = Path(
            tempfile.mkdtemp(prefix="mini-agent-swe-apptainer-", dir=root_parent)
        )
        overlay = owned / "overlay.img"
        process_runner = runner or LocalProcessRunner()
        try:
            if image_binding is None:
                selected = await _materialize_apptainer_image(
                    requested,
                    executable=executable,
                    runner=process_runner,
                    cache=image_cache,
                    timeout_seconds=max(resolved_timeout, 1800),
                    max_output_bytes=resolved_output,
                )
                identity = await _apptainer_image_identity(
                    selected,
                )
            else:
                if image_binding.runtime != "apptainer":
                    raise ValueError(
                        "Apptainer received a non-Apptainer image binding"
                    )
                if image_binding.requested != requested:
                    raise ValueError(
                        "Apptainer image binding does not match the task image"
                    )
                selected = image_binding.execution_ref
                identity = await _apptainer_image_identity(
                    selected,
                )
                if identity != image_binding.identity:
                    raise RuntimeError(
                        "Apptainer image bytes do not match the preflight binding"
                    )
            created = await process_runner.run(
                (
                    executable,
                    "overlay",
                    "create",
                    "--fakeroot",
                    "--size",
                    str(overlay_size_mib),
                    str(overlay),
                ),
                timeout_seconds=max(resolved_timeout, 300),
                max_output_bytes=resolved_output,
            )
            if created.timed_out or created.returncode != 0:
                raise RuntimeError(
                    "could not create Apptainer overlay: " + created.text()
                )
            check = await process_runner.run(
                _apptainer_exec_argv(
                    executable=executable,
                    overlay=overlay,
                    image=selected,
                    command=(
                        "git rev-parse HEAD && "
                        "test -z \"$(git status --porcelain=v1 "
                        "--untracked-files=all)\""
                    ),
                ),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            base_commit = check.text().strip().casefold()
            if (
                check.timed_out
                or check.returncode != 0
                or not re.fullmatch(r"[0-9a-f]{40}", base_commit)
            ):
                raise RuntimeError(
                    "SWE-bench image has no usable /testbed: " + check.text()
                )
            if expected_base_commit is not None:
                ancestry = await process_runner.run(
                    _apptainer_exec_argv(
                        executable=executable,
                        overlay=overlay,
                        image=selected,
                        command=(
                            "git merge-base --is-ancestor "
                            f"{expected_base_commit} {base_commit}"
                        ),
                    ),
                    timeout_seconds=resolved_timeout,
                    max_output_bytes=resolved_output,
                )
                if ancestry.timed_out or ancestry.returncode != 0:
                    raise RuntimeError(
                        "SWE-bench image does not contain task base_commit"
                    )
            return cls(
                image=selected,
                image_source=requested,
                image_identity=identity,
                base_commit=base_commit,
                overlay=overlay,
                owned_root=owned,
                executable=executable,
                runner=process_runner,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
                max_patch_bytes=resolved_patch,
                _ownership_token=_OWNED_APPTAINER,
            )
        except BaseException as operation_error:
            cleanup_error: BaseException | None = None
            try:
                if owned.exists():
                    await complete_in_thread(shutil.rmtree, owned)
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "Apptainer SWE setup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="bash",
                description=(
                    "Run one bash command in the persistent SWE-bench /testbed "
                    "workspace. Each call starts a fresh shell."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
        )

    async def _exec(
        self,
        command: str,
        *,
        max_output_bytes: int | None = None,
        binds: Sequence[tuple[Path, str]] = (),
    ) -> ProcessResult:
        if self._closed:
            raise RuntimeError("Apptainer SWE environment is closed")
        return await self.runner.run(
            _apptainer_exec_argv(
                executable=self.executable,
                overlay=self.overlay,
                image=self.image,
                command=command,
                binds=binds,
            ),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "bash":
            raise InvalidAction(f"unsupported SWE tool {action.name!r}")
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise InvalidAction("bash command must be a non-empty string")
        result = await self._exec(command)
        output = result.text() + ("\n[command timed out]" if result.timed_out else "")
        return ToolExecution(
            output=output,
            is_error=result.returncode != 0 or result.timed_out,
            metadata={
                "exit_code": result.returncode,
                "output_bytes": result.total_output_bytes,
                "output_truncated": result.truncated,
                "timed_out": result.timed_out,
            },
        )

    async def export_patch(self, destination: Path | None = None) -> bytes:
        # SWE-bench images can contain ignored build products in /testbed.
        # Force-adding them would create a patch even before the agent edits code.
        staged = await self._exec("git add --all -- .")
        if staged.timed_out or staged.returncode != 0:
            raise RuntimeError("could not stage SWE changes: " + staged.text())
        patch = await self._exec(
            "git diff --cached --binary --full-index --no-ext-diff "
            f"--no-textconv --no-renames {self.base_commit} -- .",
            max_output_bytes=self.max_patch_bytes,
        )
        if patch.timed_out or patch.returncode != 0 or patch.truncated:
            raise RuntimeError("could not capture bounded SWE patch: " + patch.text())
        if destination is not None:
            target = _patch_destination(destination)
            await complete_in_thread(_atomic_write, target, patch.output)
        return patch.output

    async def export_state(self) -> SWEPatchState:
        return SWEPatchState(
            f"{self.image_identity}@{self.base_commit}", await self.export_patch()
        )

    async def adopt_state(self, state: Any) -> None:
        if (
            not isinstance(state, SWEPatchState)
            or state.base_identity
            != f"{self.image_identity}@{self.base_commit}"
        ):
            raise ProtocolError("SWE state came from a different container image")
        if len(state.patch) > self.max_patch_bytes:
            raise ProtocolError("SWE state exceeds the patch limit")
        previous = await self.export_patch()
        target = self.owned_root / "target.patch"
        prior = self.owned_root / "prior.patch"
        await complete_in_thread(_atomic_write, target, state.patch)
        await complete_in_thread(_atomic_write, prior, previous)
        operation_error: BaseException | None = None
        try:
            await self._reset()
            if state.patch:
                applied = await self._exec(
                    "git apply --binary --index /tmp/mini-agent-target.patch",
                    binds=((target, "/tmp/mini-agent-target.patch"),),
                )
                if applied.timed_out or applied.returncode != 0:
                    raise ProtocolError(
                        "SWE state could not be applied: " + applied.text()
                    )
        except BaseException as exc:
            operation_error = exc
        if operation_error is not None:
            rollback_error: BaseException | None = None
            try:
                await self._reset()
                if previous:
                    restored = await self._exec(
                        "git apply --binary --index /tmp/mini-agent-prior.patch",
                        binds=((prior, "/tmp/mini-agent-prior.patch"),),
                    )
                    if restored.timed_out or restored.returncode != 0:
                        raise RuntimeError(
                            "prior SWE state could not be restored: " + restored.text()
                        )
            except BaseException as exc:
                rollback_error = exc
            finally:
                target.unlink(missing_ok=True)
                prior.unlink(missing_ok=True)
            raise_lifecycle_errors(
                "SWE state adoption", operation_error, rollback_error
            )
            raise AssertionError("unreachable")
        target.unlink(missing_ok=True)
        prior.unlink(missing_ok=True)

    async def _reset(self) -> None:
        # Do not white-out ignored lower-layer files baked into the SIF image.
        reset = await self._exec(
            f"git reset --hard {self.base_commit} && git clean -ffd -q"
        )
        if reset.timed_out or reset.returncode != 0:
            raise RuntimeError("could not reset SWE overlay: " + reset.text())

    async def close(self) -> None:
        if self._closed:
            return
        if self.owned_root.exists():
            await complete_in_thread(shutil.rmtree, self.owned_root)
        self._closed = True

    def resource_identity(self) -> str:
        return f"swe-apptainer-overlay:{self.overlay}"

    def provenance(self) -> Mapping[str, Any]:
        return {
            "application": "swe",
            "benchmark": "swe_bench",
            "benchmark_revision": SWEBENCH_REVISION,
            "benchmark_tag": SWEBENCH_TAG,
            "container_runtime": "apptainer",
            "container_image": self.image,
            "container_image_source": self.image_source,
            "container_image_identity": self.image_identity,
            "base_commit": self.base_commit,
            "overlay": "private_fakeroot_ext3",
            "workdir": SWEBENCH_WORKDIR,
        }


async def _apptainer_image_identity(image: str) -> str:
    path = Path(image)
    if path.is_symlink():
        raise ValueError("Apptainer image path cannot be a symlink")
    if not path.is_file():
        raise ValueError("Apptainer execution requires a materialized local image")
    digest = await complete_in_thread(_sha256_file, path)
    return f"sha256:{digest}"


def _patch_destination(destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise ValueError("patch destination must be a Path or None")
    target = destination.expanduser()
    if target.is_symlink():
        raise ValueError("patch destination must not be a symlink")
    return target


async def _materialize_apptainer_image(
    image: str,
    *,
    executable: str,
    runner: ProcessRunner,
    cache: Path | None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> str:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Apptainer is Linux-only
        raise RuntimeError("Apptainer image caching requires POSIX file locks") from exc
    if not isinstance(image, str) or not image or "\x00" in image:
        raise ValueError("Apptainer image reference must be non-empty")
    if not isinstance(executable, str) or not executable or "\x00" in executable:
        raise ValueError("Apptainer executable must be non-empty")
    if not callable(getattr(runner, "run", None)):
        raise ValueError("runner must expose run")
    resolved_timeout = _positive_number(timeout_seconds, "cache timeout_seconds")
    resolved_output = _positive_int(max_output_bytes, "cache max_output_bytes")
    local = Path(image).expanduser()
    if local.is_symlink():
        raise ValueError("Apptainer image path cannot be a symlink")
    if local.is_file():
        return str(local.resolve())
    if cache is not None and cache.expanduser().is_symlink():
        raise ValueError("Apptainer image cache root cannot be a symlink")
    cache_root = (
        cache.expanduser().resolve()
        if cache is not None
        else Path.home() / ".cache" / "mini-agent" / "apptainer-images"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("Apptainer image cache must be a directory")
    key = hashlib.sha256(image.encode("utf-8")).hexdigest()
    destination = cache_root / f"{key}.sif"
    lock_path = cache_root / f"{key}.lock"
    if destination.is_symlink() or lock_path.is_symlink():
        raise ValueError("Apptainer cache entries cannot be symlinks")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    lock_stream = os.fdopen(descriptor, "a+b")
    temporary = cache_root / f".{key}.{uuid.uuid4().hex}.sif"
    acquired = False
    lock_deadline = time.monotonic() + resolved_timeout
    try:
        while True:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "timed out waiting for Apptainer image cache lock"
                    )
                await asyncio.sleep(min(0.05, remaining))
        if destination.is_symlink():
            raise ValueError("Apptainer cache entries cannot be symlinks")
        if not destination.is_file() or destination.stat().st_size == 0:
            pulled = await runner.run(
                (executable, "pull", "--force", str(temporary), image),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            if pulled.timed_out or pulled.returncode != 0 or not temporary.is_file():
                raise RuntimeError(
                    "could not materialize Apptainer image: " + pulled.text()
                )
            temporary.replace(destination)
        return str(destination)
    finally:
        temporary.unlink(missing_ok=True)
        if acquired:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _sha256_file(path: Path) -> str:
    return stable_file_sha256(path, label="Apptainer image")


__all__ = [
    "ApptainerSWEEnvironment",
    "DEFAULT_MAX_ARCHIVE_BYTES",
    "DockerSWEEnvironment",
    "SWEArchiveState",
    "SWEBENCH_REVISION",
    "SWEBENCH_TAG",
    "SWEBENCH_WORKDIR",
    "SWEbenchDoctorCheck",
    "SWEbenchDoctorReport",
    "SWEbenchImageBinding",
    "resolve_swebench_image_binding",
    "swebench_doctor",
    "swebench_image_name",
]
