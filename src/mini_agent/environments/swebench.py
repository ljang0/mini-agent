from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition, ToolExecution
from .base import BaseEnvironment
from .swe import (
    DEFAULT_MAX_PATCH_BYTES,
    LocalProcessRunner,
    ProcessResult,
    ProcessRunner,
    _atomic_write,
)


SWEBENCH_REVISION = "v4.1.0"
SWEBENCH_WORKDIR = "/testbed"
_SAFE_CONTAINER_PART = re.compile(r"[^a-z0-9_.-]+")


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


def _container_name(instance_id: str) -> str:
    safe = _SAFE_CONTAINER_PART.sub("-", instance_id.lower()).strip("-.")
    safe = safe[:48] or "instance"
    return f"mini-agent-swe-{safe}-{uuid.uuid4().hex[:12]}"


async def swebench_doctor(
    *,
    runtime: Sequence[str] = ("docker",),
    image: str | None = None,
    runner: ProcessRunner | None = None,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 64 * 1024,
) -> SWEbenchDoctorReport:
    """Perform non-mutating container-runtime and optional image checks."""

    if not runtime or not all(isinstance(item, str) and item for item in runtime):
        raise ValueError("container runtime argv must contain non-empty strings")
    if timeout_seconds <= 0 or max_output_bytes < 1:
        raise ValueError("doctor limits must be positive")
    if image is not None and (
        not isinstance(image, str) or not image or image.startswith("-") or "\x00" in image
    ):
        raise ValueError("doctor image must be a valid Docker image name")
    process_runner = runner or LocalProcessRunner()
    requests: list[tuple[str, tuple[str, ...]]] = [
        (
            "runtime_version",
            (*runtime, "version", "--format", "{{.Server.Version}}"),
        ),
        (
            "daemon_platform",
            (*runtime, "info", "--format", "{{.OSType}}/{{.Architecture}}"),
        ),
    ]
    if image is not None:
        requests.append(
            (
                "image_available",
                (*runtime, "image", "inspect", "--format", "{{.Id}}", image),
            )
        )
    checks: list[SWEbenchDoctorCheck] = []
    for name, argv in requests:
        try:
            result = await process_runner.run(
                argv,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
            ok = not result.timed_out and result.returncode == 0
            detail = result.text().strip()
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
        runtime=tuple(runtime),
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
        platform: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.image_id = image_id
        self.runtime = tuple(runtime)
        self.runner = runner
        self.platform = platform
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.max_patch_bytes = max_patch_bytes
        self._closed = False

    @classmethod
    async def create(
        cls,
        instance: Mapping[str, Any],
        *,
        runtime: Sequence[str] = ("docker",),
        platform: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        runner: ProcessRunner | None = None,
    ) -> "DockerSWEEnvironment":
        if not runtime or not all(isinstance(item, str) and item for item in runtime):
            raise ValueError("container runtime argv must contain non-empty strings")
        if platform is not None and (not platform or platform.startswith("-")):
            raise ValueError("container platform is invalid")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes < 1 or max_patch_bytes < 1:
            raise ValueError("output and patch limits must be positive")
        image = swebench_image_name(instance)
        instance_id = instance.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("SWE-bench instance requires instance_id")
        process_runner = runner or LocalProcessRunner()
        name = _container_name(instance_id)
        argv = [
            *runtime,
            "run",
            "--detach",
            "--name",
            name,
            "--workdir",
            SWEBENCH_WORKDIR,
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
        if platform is not None:
            argv.extend(("--platform", platform))
        argv.extend((image, "sleep", "infinity"))
        started = await process_runner.run(
            argv,
            timeout_seconds=max(timeout_seconds, 300.0),
            max_output_bytes=max_output_bytes,
        )
        if started.timed_out or started.returncode != 0:
            raise RuntimeError("could not start SWE-bench container: " + started.text())
        environment: DockerSWEEnvironment | None = None
        try:
            inspected = await process_runner.run(
                (*runtime, "image", "inspect", "--format", "{{.Id}}", image),
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
            if inspected.timed_out or inspected.returncode != 0:
                raise RuntimeError(
                    "could not resolve SWE-bench image identity: " + inspected.text()
                )
            image_id = inspected.text().strip()
            if not image_id:
                raise RuntimeError("container runtime returned an empty image identity")
            environment = cls(
                image=image,
                container_name=name,
                image_id=image_id,
                runtime=runtime,
                runner=process_runner,
                platform=platform,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                max_patch_bytes=max_patch_bytes,
            )
            return environment
        except BaseException:
            try:
                await process_runner.run(
                    (*runtime, "rm", "--force", name),
                    timeout_seconds=timeout_seconds,
                    max_output_bytes=max_output_bytes,
                )
            except Exception:
                pass
            raise

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="bash",
                description=(
                    "Run one bash command in the persistent SWE-bench /testbed "
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
            (
                *self.runtime,
                "exec",
                "--workdir",
                SWEBENCH_WORKDIR,
                self.container_name,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-lc",
                command,
            ),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "bash":
            raise ProtocolError(f"unsupported SWE tool {action.name!r}")
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ProtocolError("bash command must be a non-empty string")
        result = await self._exec(command)
        output = result.text()
        if result.timed_out:
            output += "\n[command timed out]"
        return ToolExecution(
            output=output,
            is_error=result.returncode != 0 or result.timed_out,
            metadata={
                "exit_code": result.returncode,
                "output_bytes": result.total_output_bytes,
                "output_truncated": result.truncated,
                "timed_out": result.timed_out,
            },
            native_output={
                "stdout": output,
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": result.returncode},
            },
        )

    async def export_patch(self, destination: Path | None = None) -> bytes:
        staged = await self._exec("git add --all --force -- .")
        if staged.timed_out or staged.returncode != 0:
            raise RuntimeError("could not stage workspace changes: " + staged.text())
        patch = await self._exec(
            "git diff --cached --binary --full-index --no-ext-diff "
            "--no-textconv --no-renames HEAD -- .",
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
            await asyncio.to_thread(_atomic_write, destination.resolve(), patch.output)
        return patch.output

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        result = await self.runner.run(
            (*self.runtime, "rm", "--force", self.container_name),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        if result.timed_out or result.returncode != 0:
            raise RuntimeError("could not remove SWE-bench container: " + result.text())

    def provenance(self) -> dict[str, object]:
        return {
            "application": "swe",
            "benchmark": "swe_bench",
            "benchmark_revision": SWEBENCH_REVISION,
            "tools": ["bash"],
            "container_runtime": list(self.runtime),
            "container_image": self.image,
            "container_image_id": self.image_id,
            "container_platform": self.platform,
            "workdir": SWEBENCH_WORKDIR,
            "host_credentials_mounted": False,
            "patch_export": "git_diff_binary",
        }

    def resource_identity(self) -> str:
        return f"swe-container:{self.container_name}"


__all__ = [
    "DockerSWEEnvironment",
    "SWEBENCH_REVISION",
    "SWEBENCH_WORKDIR",
    "SWEbenchDoctorCheck",
    "SWEbenchDoctorReport",
    "swebench_doctor",
    "swebench_image_name",
]
