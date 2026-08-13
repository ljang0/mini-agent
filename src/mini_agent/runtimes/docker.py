"""Rootless Docker runtime: start from an image ID, exec, copy, destroy."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._lifecycle import complete_in_thread, raise_lifecycle_errors
from ..types import _require_bool, strict_json_loads
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ProcessResult,
    ProcessRunner,
    atomic_write,
    failed,
    positive_int,
    positive_number,
    require_argv,
    require_ok,
    require_ref,
    require_runner,
    require_staging_name,
    require_workdir,
    resolve_runner,
)


DEFAULT_CONTAINER_PREFIX = "mini-agent-"
_SAFE_CONTAINER_PART = re.compile(r"[^a-z0-9_.-]+")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_OWNED_DOCKER = object()


def container_name(label: str, *, prefix: str = DEFAULT_CONTAINER_PREFIX) -> str:
    """Return one unique, runtime-owned container name derived from ``label``."""

    safe = _SAFE_CONTAINER_PART.sub("-", label.lower()).strip("-.")
    safe = safe[:48] or "instance"
    return f"{prefix}{safe}-{uuid.uuid4().hex[:12]}"


def docker_security_is_rootless(output: str) -> bool:
    try:
        value = strict_json_loads(output)
    except ValueError:
        return False
    return isinstance(value, list) and any(
        item == "name=rootless" for item in value if isinstance(item, str)
    )


async def docker_image_id(
    image: str,
    *,
    runtime: Sequence[str],
    runner: ProcessRunner,
    timeout_seconds: float,
    max_output_bytes: int,
    pull_if_missing: bool,
) -> str:
    """Resolve a mutable image reference to its immutable image ID."""

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
        require_ok(pulled, "could not materialize Docker image", fallback=True)
        inspected = await inspect_image()
    require_ok(inspected, "could not inspect Docker image", fallback=True)
    image_id = inspected.text().strip().casefold()
    if not _IMAGE_ID.fullmatch(image_id):
        raise RuntimeError("container runtime returned an invalid Docker image ID")
    return image_id


@dataclass(frozen=True)
class RuntimeDoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RuntimeDoctorReport:
    ok: bool
    runtime: tuple[str, ...]
    checks: tuple[RuntimeDoctorCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "runtime": list(self.runtime),
            "checks": [asdict(check) for check in self.checks],
        }


async def docker_doctor(
    *,
    runtime: Sequence[str] = ("docker",),
    image: str | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 64 * 1024,
    require_rootless: bool = True,
) -> RuntimeDoctorReport:
    """Perform non-mutating container-runtime and optional image checks."""

    resolved_runtime = require_argv(runtime, "container runtime")
    resolved_timeout = positive_number(timeout_seconds, "doctor timeout_seconds")
    resolved_max_output = positive_int(max_output_bytes, "doctor max_output_bytes")
    _require_bool(require_rootless, "require_rootless")
    if image is not None:
        require_ref(
            image, "doctor image must be a valid Docker image name", no_dash=True
        )
    process_runner = resolve_runner(runner)
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
    checks: list[RuntimeDoctorCheck] = []
    for name, argv in requests:
        try:
            result = await process_runner.run(
                argv,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_max_output,
            )
            ok = not result.timed_out and result.returncode == 0
            detail = result.text().strip()
            if name == "rootless_security" and not docker_security_is_rootless(detail):
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
        checks.append(RuntimeDoctorCheck(name=name, ok=ok, detail=detail))
    return RuntimeDoctorReport(
        ok=all(check.ok for check in checks),
        runtime=resolved_runtime,
        checks=tuple(checks),
    )


class DockerRuntime:
    """One detached, mini-agent-owned container addressed by image ID."""

    def __init__(
        self,
        *,
        runtime: Sequence[str],
        runner: ProcessRunner,
        name: str,
        image: str,
        image_id: str,
        workdir: str,
        exec_env: Mapping[str, str],
        platform: str | None = None,
        network_disabled: bool = False,
        staging_dir: str = "/tmp",
        name_prefix: str = DEFAULT_CONTAINER_PREFIX,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        _ownership_token: object | None = None,
    ) -> None:
        if _ownership_token is not _OWNED_DOCKER:
            raise ValueError("Docker runtimes are created only by start()")
        owned = re.escape(name_prefix) + r"[a-z0-9_.-]*"
        if not isinstance(name, str) or not re.fullmatch(owned, name):
            raise ValueError("Docker container name is not mini-agent owned")
        if platform is not None:
            require_ref(platform, "container platform is invalid", no_dash=True)
        self.runtime = require_argv(runtime, "container runtime")
        self.runner = require_runner(runner)
        self.name = name
        self.image = require_ref(image, "Docker image must be non-empty")
        self.image_id = require_ref(image_id, "Docker image identity must be non-empty")
        self.workdir = require_workdir(workdir)
        self.exec_env = dict(exec_env)
        self.platform = platform
        self.network_disabled = _require_bool(network_disabled, "network_disabled")
        self.staging_dir = require_workdir(staging_dir)
        self.timeout_seconds = positive_number(timeout_seconds, "timeout_seconds")
        self.max_output_bytes = positive_int(max_output_bytes, "max_output_bytes")
        self._staging_root: Path | None = None

    @classmethod
    async def start(
        cls,
        *,
        image: str,
        image_id: str,
        runtime: Sequence[str] = ("docker",),
        runner: ProcessRunner | None = None,
        name_label: str,
        name_prefix: str = DEFAULT_CONTAINER_PREFIX,
        workdir: str,
        exec_env: Mapping[str, str] | None = None,
        container_env: Sequence[str] = (),
        labels: Sequence[str] = (),
        platform: str | None = None,
        network_disabled: bool = False,
        require_rootless: bool = True,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> "DockerRuntime":
        """Start one detached container from an exact image ID."""

        resolved_runtime = require_argv(runtime, "container runtime")
        resolved_workdir = require_workdir(workdir)
        resolved_timeout = positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = positive_int(max_output_bytes, "max_output_bytes")
        _require_bool(network_disabled, "network_disabled")
        process_runner = resolve_runner(runner)
        if require_rootless:
            security = await process_runner.run(
                (*resolved_runtime, "info", "--format", "{{json .SecurityOptions}}"),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            if failed(security) or not docker_security_is_rootless(security.text()):
                raise RuntimeError(
                    "Docker execution requires a rootless daemon"
                )
        name = container_name(name_label, prefix=name_prefix)
        argv = [
            *resolved_runtime,
            "run",
            "--detach",
            "--name",
            name,
            "--workdir",
            resolved_workdir,
        ]
        for label in labels:
            argv.extend(("--label", label))
        for item in container_env:
            argv.extend(("--env", item))
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
            require_ok(started, "could not start the container")
            inspected = await process_runner.run(
                (*resolved_runtime, "inspect", "--format", "{{.Image}}", name),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            require_ok(inspected, "could not resolve the running container image")
            if inspected.text().strip().casefold() != image_id:
                raise RuntimeError(
                    "running container image does not match its binding"
                )
            return cls(
                runtime=resolved_runtime,
                runner=process_runner,
                name=name,
                image=image,
                image_id=image_id,
                workdir=resolved_workdir,
                exec_env=exec_env or {},
                platform=platform,
                network_disabled=network_disabled,
                name_prefix=name_prefix,
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
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
                require_ok(removed, "container removal failed", fallback=True)
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "container startup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult:
        command = [
            *self.runtime,
            "exec",
            "--workdir",
            self.workdir if cwd is None else require_workdir(cwd),
        ]
        for key, value in (self.exec_env if env is None else env).items():
            command.extend(("--env", f"{key}={value}"))
        command.append(self.name)
        command.extend(require_argv(argv, "container command"))
        return await self.runner.run(
            tuple(command),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def _staging(self) -> Path:
        if self._staging_root is None:
            self._staging_root = Path(
                tempfile.mkdtemp(prefix="mini-agent-container-stage-")
            )
        return self._staging_root

    async def read_file(self, path: str) -> bytes:
        """Copy one container file out and return its bytes."""

        require_ref(path, "container file path must be non-empty", no_dash=True)
        root = Path(tempfile.mkdtemp(prefix="mini-agent-container-read-"))
        operation_error: BaseException | None = None
        content = b""
        try:
            local = root / "payload"
            copied = await self.runner.run(
                (*self.runtime, "cp", f"{self.name}:{path}", str(local)),
                timeout_seconds=max(self.timeout_seconds, 300.0),
                max_output_bytes=self.max_output_bytes,
            )
            require_ok(copied, "could not copy the container file")
            content = await complete_in_thread(local.read_bytes)
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await complete_in_thread(shutil.rmtree, root)
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors("container file read", operation_error, cleanup_error)
        return content

    async def write_file(self, name: str, data: bytes) -> str:
        """Copy ``data`` into the container and return its in-container path."""

        resolved = require_staging_name(name)
        local = self._staging() / resolved
        target = f"{self.staging_dir.rstrip('/')}/{resolved}"
        await complete_in_thread(atomic_write, local, data)
        copied = await self.runner.run(
            (*self.runtime, "cp", str(local), f"{self.name}:{target}"),
            timeout_seconds=max(self.timeout_seconds, 300.0),
            max_output_bytes=self.max_output_bytes,
        )
        require_ok(copied, "could not copy a file into the container")
        return target

    async def remove_file(self, path: str) -> None:
        """Drop the host copy; the container copy dies with the container."""

        if self._staging_root is not None:
            (self._staging_root / Path(path).name).unlink(missing_ok=True)

    async def close(self) -> None:
        staging = self._staging_root
        self._staging_root = None
        result = await self.runner.run(
            (*self.runtime, "rm", "--force", self.name),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        try:
            require_ok(result, "could not remove the container")
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def provenance(self) -> Mapping[str, Any]:
        return {
            "container_runtime": list(self.runtime),
            "container_image": self.image,
            "container_image_id": self.image_id,
            "container_platform": self.platform,
            "rootless_daemon_required": True,
            "network_disabled": self.network_disabled,
            "workdir": self.workdir,
            "host_credentials_mounted": False,
        }

    def resource_identity(self) -> str:
        return f"container:{self.name}"


__all__ = [
    "DEFAULT_CONTAINER_PREFIX",
    "DockerRuntime",
    "RuntimeDoctorCheck",
    "RuntimeDoctorReport",
    "container_name",
    "docker_doctor",
    "docker_image_id",
    "docker_security_is_rootless",
]
