from __future__ import annotations

import asyncio
import hashlib
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
    _require_bool,
    _require_finite_number,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
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
    return _require_finite_number(value, label, exclusive_minimum=0)


def _positive_int(value: Any, label: str) -> int:
    return _require_positive_int(value, label)


def _require_ref(value: Any, message: str, *, no_dash: bool = False) -> str:
    """Return a non-empty, NUL-free string usable as a runtime argument."""

    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or (no_dash and value.startswith("-"))
    ):
        raise ValueError(message)
    return value


def _require_runner(runner: Any) -> ProcessRunner:
    """Return ``runner`` once it exposes the process-runner contract."""

    if not callable(getattr(runner, "run", None)):
        raise ValueError("runner must expose run")
    return runner


def _resolved_runner(runner: ProcessRunner | None) -> ProcessRunner:
    """Return the caller's runner, or the default local one."""

    return LocalProcessRunner() if runner is None else _require_runner(runner)


def _failed(result: ProcessResult) -> bool:
    return result.timed_out or result.returncode != 0


def _require_ok(
    result: ProcessResult,
    message: str,
    *,
    error: type[Exception] = RuntimeError,
    fallback: bool = False,
) -> ProcessResult:
    """Return ``result``, or raise ``message`` with the captured output."""

    if _failed(result):
        detail = result.text()
        if fallback and not detail:
            detail = f"exit code {result.returncode}"
        raise error(f"{message}: {detail}")
    return result


def _instance_id(instance: Mapping[str, Any]) -> str:
    value = instance.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("SWE-bench instance requires instance_id")
    return value


def _bash_tools(description: str) -> Sequence[ToolDefinition]:
    """Return the single-command bash tool both SWE runtimes expose."""

    return (
        ToolDefinition(
            name="bash",
            description=description,
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
    )


def _execution_metadata(result: ProcessResult) -> dict[str, Any]:
    return {
        "exit_code": result.returncode,
        "output_bytes": result.total_output_bytes,
        "output_truncated": result.truncated,
        "timed_out": result.timed_out,
    }


def _bash_command(action: ToolCall) -> str:
    """Return the one bash command an SWE environment action may carry."""

    if action.name != "bash":
        raise InvalidAction(f"unsupported SWE tool {action.name!r}")
    return _require_str(
        action.arguments.get("command"), "bash command", error=InvalidAction
    )


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
    message = "container benchmark identity must contain strings"
    return {
        _require_ref(name, message): _require_ref(item, message)
        for name, item in value.items()
    }


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
        _require_ref(
            self.requested, "SWE-bench requested image is invalid", no_dash=True
        )
        if not isinstance(self.execution_ref, str) or not self.execution_ref:
            raise ValueError("SWE-bench image execution reference is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.identity):
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

    _require_mapping(instance, "SWE-bench instance")
    if runtime not in {"docker", "apptainer"}:
        raise ValueError("SWE-bench runtime must be docker or apptainer")
    resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
    resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
    process_runner = _resolved_runner(runner)
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

    _require_ref(apptainer_executable, "Apptainer executable must be non-empty")
    source = "docker://" + requested.removeprefix("docker.io/")
    selected = await _materialize_apptainer_image(
        source,
        executable=apptainer_executable,
        runner=process_runner,
        cache=apptainer_image_cache,
        timeout_seconds=max(resolved_timeout, 1800.0),
        max_output_bytes=resolved_output,
    )
    identity = await _apptainer_image_identity(selected)
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
    argv.extend((image, "/bin/bash", "--noprofile", "--norc", "-c", command))
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
        _require_ok(
            pulled, "could not materialize SWE-bench Docker image", fallback=True
        )
        inspected = await inspect_image()
    _require_ok(inspected, "could not inspect SWE-bench Docker image", fallback=True)
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
    _require_bool(require_rootless, "require_rootless")
    if image is not None:
        _require_ref(
            image, "doctor image must be a valid Docker image name", no_dash=True
        )
    process_runner = _resolved_runner(runner)
    version = (*resolved_runtime, "version", "--format", "{{.Server.Version}}")
    platform = (*resolved_runtime, "info", "--format", "{{.OSType}}/{{.Architecture}}")
    requests: list[tuple[str, tuple[str, ...]]] = [
        ("runtime_version", version),
        ("daemon_platform", platform),
    ]
    if require_rootless:
        rootless = (*resolved_runtime, "info", "--format", "{{json .SecurityOptions}}")
        requests.append(("rootless_security", rootless))
    if image is not None:
        probe = (*resolved_runtime, "image", "inspect", "--format", "{{.Id}}", image)
        requests.append(("image_available", probe))
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
            if name == "rootless_security" and not _docker_security_is_rootless(detail):
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
        if not isinstance(container_name, str) or not re.fullmatch(
            r"mini-agent-swe-[a-z0-9_.-]*", container_name
        ):
            raise ValueError("Docker container name is not mini-agent owned")
        if base_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", base_commit):
            raise ValueError("SWE-bench base commit must be a full Git commit")
        if platform is not None:
            _require_ref(platform, "container platform is invalid", no_dash=True)
        self.image = _require_ref(image, "Docker image must be non-empty")
        self.container_name = container_name
        self.image_id = _require_str(image_id, "Docker image identity")
        self.runtime = _runtime_argv(runtime, "container runtime")
        self.runner = _require_runner(runner)
        self.base_commit = base_commit
        self.platform = platform
        self.timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
        self.max_output_bytes = _positive_int(max_output_bytes, "max_output_bytes")
        self.max_patch_bytes = _positive_int(max_patch_bytes, "max_patch_bytes")
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.workdir = _container_workdir(workdir)
        self.network_disabled = _require_bool(network_disabled, "network_disabled")
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
        _require_mapping(instance, "SWE-bench instance")
        _require_bool(network_disabled, "network_disabled")
        _require_bool(require_git_baseline, "require_git_baseline")
        resolved_workdir = _container_workdir(workdir)
        resolved_identity = _benchmark_identity(benchmark_identity)
        resolved_runtime = _runtime_argv(runtime, "container runtime")
        if platform is not None:
            _require_ref(platform, "container platform is invalid", no_dash=True)
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
        instance_id = _instance_id(instance)
        process_runner = _resolved_runner(runner)
        if image_binding is None:
            image_id = await _docker_image_id(
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
        security = await process_runner.run(
            (*resolved_runtime, "info", "--format", "{{json .SecurityOptions}}"),
            timeout_seconds=resolved_timeout,
            max_output_bytes=resolved_output,
        )
        if _failed(security) or not _docker_security_is_rootless(security.text()):
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
            _require_ok(started, "could not start SWE-bench container")
            inspected = await process_runner.run(
                (*resolved_runtime, "inspect", "--format", "{{.Image}}", name),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            _require_ok(inspected, "could not resolve running SWE-bench image identity")
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
                if _failed(baseline) or not re.fullmatch(r"[0-9a-f]{40}", base_commit):
                    raise RuntimeError(
                        "SWE-bench image has no usable Git baseline: " + baseline.text()
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
                    if _failed(ancestry):
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
                _require_ok(removed, "container removal failed", fallback=True)
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "SWE-bench container startup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    def tools(self) -> Sequence[ToolDefinition]:
        return _bash_tools(
            f"Run one bash command in the persistent {self.workdir} "
            "workspace. Each call starts a new shell."
        )

    async def _exec(
        self, command: str, *, max_output_bytes: int | None = None
    ) -> ProcessResult:
        if self._closed:
            raise RuntimeError("SWE-bench environment is closed")
        return await self.runner.run(
            _docker_exec_argv(self.runtime, self.container_name, command, self.workdir),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        result = await self._exec(_bash_command(action))
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
            metadata=_execution_metadata(result),
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
        _require_ok(staged, "could not stage workspace changes")
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
        _require_ok(created, "could not archive the container workspace")
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
            source = f"{self.container_name}:{_ARCHIVE_CONTAINER_PATH}"
            copied = await self.runner.run(
                (*self.runtime, "cp", source, str(local)),
                timeout_seconds=max(self.timeout_seconds, 300.0),
                max_output_bytes=self.max_output_bytes,
            )
            _require_ok(copied, "could not copy the container workspace archive")
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
        expected = f"{self.image_id}@{self.base_commit}"
        if not isinstance(state, SWEPatchState) or state.base_identity != expected:
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
                _require_ok(
                    applied, "SWE state could not be applied", error=ProtocolError
                )
        except BaseException as exc:
            operation_error = exc
        if operation_error is not None and mutated:
            try:
                await self._reset()
                if previous:
                    restored = await self._copy_and_apply(prior, "prior.patch")
                    _require_ok(restored, "prior SWE state could not be restored")
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
        expected = f"{self.image_id}@{self.workdir}"
        if not isinstance(state, SWEArchiveState) or state.base_identity != expected:
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
        _require_ok(copied, "SWE state could not be copied", error=ProtocolError)
        applied = await self._exec(
            f"find {self.workdir} -mindepth 1 -delete && "
            f"tar -xzf /tmp/{name} -C {self.workdir} && rm -f /tmp/{name}"
        )
        _require_ok(applied, "SWE state could not be applied", error=ProtocolError)

    async def _reset(self) -> None:
        # Keep ignored build products supplied by the benchmark image. They are
        # part of its runtime, even though they are not part of the git tree.
        result = await self._exec(
            f"git reset --hard {self.base_commit} && git clean -ffd -q"
        )
        _require_ok(result, "could not reset SWE container")

    async def _copy_and_apply(self, source: Path, name: str) -> ProcessResult:
        copied = await self.runner.run(
            (*self.runtime, "cp", str(source), f"{self.container_name}:/tmp/{name}"),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        if _failed(copied):
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
        _require_ok(result, "could not remove SWE-bench container")
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
        self.image = _require_ref(image, "Apptainer image must be non-empty")
        self.image_source = _require_ref(
            image_source, "Apptainer image source must be non-empty"
        )
        self.image_identity = _require_str(image_identity, "Apptainer image identity")
        self.base_commit = base_commit
        self.overlay = resolved_overlay
        self.owned_root = resolved_root
        self.executable = _require_ref(
            executable, "Apptainer executable must be non-empty"
        )
        self.runner = _require_runner(runner)
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
        _require_mapping(instance, "SWE-bench instance")
        if image is not None and not isinstance(image, str):
            raise ValueError("Apptainer image must be a string or None")
        _require_ref(executable, "Apptainer executable must be non-empty")
        requested = image or "docker://" + swebench_image_name(instance).removeprefix(
            "docker.io/"
        )
        expected_base_commit = _expected_base_commit(instance)
        _require_ref(requested, "Apptainer image reference is invalid", no_dash=True)
        _require_positive_int(overlay_size_mib, "overlay_size_mib", minimum=1024)
        resolved_timeout = _positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = _positive_int(max_output_bytes, "max_output_bytes")
        resolved_patch = _positive_int(max_patch_bytes, "max_patch_bytes")
        root_parent = (
            None if scratch_root is None else scratch_root.expanduser().resolve()
        )
        if root_parent is not None:
            root_parent.mkdir(parents=True, exist_ok=True)
        owned = Path(
            tempfile.mkdtemp(prefix="mini-agent-swe-apptainer-", dir=root_parent)
        )
        overlay = owned / "overlay.img"
        process_runner = _resolved_runner(runner)
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
                identity = await _apptainer_image_identity(selected)
            else:
                if image_binding.runtime != "apptainer":
                    raise ValueError("Apptainer received a non-Apptainer image binding")
                if image_binding.requested != requested:
                    raise ValueError(
                        "Apptainer image binding does not match the task image"
                    )
                selected = image_binding.execution_ref
                identity = await _apptainer_image_identity(selected)
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
            _require_ok(created, "could not create Apptainer overlay")
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
            if _failed(check) or not re.fullmatch(r"[0-9a-f]{40}", base_commit):
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
                if _failed(ancestry):
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
        return _bash_tools(
            "Run one bash command in the persistent SWE-bench /testbed "
            "workspace. Each call starts a fresh shell."
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
        result = await self._exec(_bash_command(action))
        output = result.text() + ("\n[command timed out]" if result.timed_out else "")
        return ToolExecution(
            output=output,
            is_error=result.returncode != 0 or result.timed_out,
            metadata=_execution_metadata(result),
        )

    async def export_patch(self, destination: Path | None = None) -> bytes:
        # SWE-bench images can contain ignored build products in /testbed.
        # Force-adding them would create a patch even before the agent edits code.
        staged = await self._exec("git add --all -- .")
        _require_ok(staged, "could not stage SWE changes")
        patch = await self._exec(
            "git diff --cached --binary --full-index --no-ext-diff "
            f"--no-textconv --no-renames {self.base_commit} -- .",
            max_output_bytes=self.max_patch_bytes,
        )
        if _failed(patch) or patch.truncated:
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
        expected = f"{self.image_identity}@{self.base_commit}"
        if not isinstance(state, SWEPatchState) or state.base_identity != expected:
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
                _require_ok(
                    applied, "SWE state could not be applied", error=ProtocolError
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
                    _require_ok(restored, "prior SWE state could not be restored")
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
        _require_ok(reset, "could not reset SWE overlay")

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
    path = _require_no_symlink(Path(image), "Apptainer image path")
    if not path.is_file():
        raise ValueError("Apptainer execution requires a materialized local image")
    digest = await complete_in_thread(stable_file_sha256, path, label="Apptainer image")
    return f"sha256:{digest}"


def _patch_destination(destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise ValueError("patch destination must be a Path or None")
    return _require_no_symlink(destination.expanduser(), "patch destination")


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
    _require_ref(image, "Apptainer image reference must be non-empty")
    _require_ref(executable, "Apptainer executable must be non-empty")
    _require_runner(runner)
    resolved_timeout = _positive_number(timeout_seconds, "cache timeout_seconds")
    resolved_output = _positive_int(max_output_bytes, "cache max_output_bytes")
    local = _require_no_symlink(Path(image).expanduser(), "Apptainer image path")
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
            if _failed(pulled) or not temporary.is_file():
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
