from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition, ToolExecution
from .base import BaseEnvironment


MINI_SWE_AGENT_REVISION = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
SWE_AGENT_REVISION = "3ea751c087f32b16e039a2233dd6eefecef325d5"
DEFAULT_MAX_PATCH_BYTES = 8 * 1024 * 1024
TRUSTED_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


@dataclass(frozen=True)
class ProcessResult:
    output: bytes
    returncode: int
    total_output_bytes: int
    truncated: bool = False
    timed_out: bool = False

    def text(self) -> str:
        return self.output.decode("utf-8", errors="replace")


class ProcessRunner(Protocol):
    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
    ) -> ProcessResult: ...


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _read_head_tail(
    stream: asyncio.StreamReader, limit: int
) -> tuple[bytes, int, bool]:
    head_limit = max(1, limit // 2)
    tail_limit = max(0, limit - head_limit)
    head = bytearray()
    tail = bytearray()
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        missing = head_limit - len(head)
        if missing > 0:
            head.extend(chunk[:missing])
            chunk = chunk[missing:]
        if chunk and tail_limit:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[: len(tail) - tail_limit]
    truncated = total > limit
    if not truncated:
        return bytes(head + tail), total, False
    marker = f"\n[... {total - limit} bytes omitted ...]\n".encode("ascii")
    return bytes(head) + marker + bytes(tail), total, True


class LocalProcessRunner:
    """Run one argv without a shell and retain bounded head-plus-tail output."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
    ) -> ProcessResult:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        reader = asyncio.create_task(_read_head_tail(process.stdout, max_output_bytes))
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate_process_group(process)
        except BaseException:
            await _terminate_process_group(process)
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise
        output, total, truncated = await reader
        return ProcessResult(
            output=output,
            returncode=process.returncode if process.returncode is not None else -1,
            total_output_bytes=total,
            truncated=truncated,
            timed_out=timed_out,
        )


def _minimal_environment(home: Path) -> dict[str, str]:
    return {
        "PATH": TRUSTED_PATH,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def _validate_workspace_symlinks(root: Path) -> None:
    """Reject links whose copied form could resolve outside ``root``."""

    root = root.resolve()
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name != ".git"]
        for name in [*names, *files]:
            if name == ".git":
                continue
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            raw_target = os.readlink(path)
            if os.path.isabs(raw_target):
                raise ValueError(f"workspace contains absolute symlink: {path}")
            try:
                target = (path.parent / raw_target).resolve(strict=False)
                target.relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"workspace symlink escapes its root: {path}") from exc


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BashEnvironment(BaseEnvironment):
    """One stateless bash action over a persistent, optionally copied workspace.

    Local bash is an execution mechanism, not a security boundary. Benchmark code
    should use :class:`DockerSWEEnvironment` from ``swebench.py``.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        owned_root: Path | None = None,
        source_workspace: Path | None = None,
        home: Path | None = None,
        runner: ProcessRunner | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {workspace}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        if max_patch_bytes < 1:
            raise ValueError("max_patch_bytes must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.max_patch_bytes = max_patch_bytes
        self._owned_root = owned_root
        self.source_workspace = (source_workspace or self.workspace).resolve()
        self._runner = runner or LocalProcessRunner()
        self._closed = False
        self._home_owned_root: Path | None = None
        if home is None:
            self._home_owned_root = Path(tempfile.mkdtemp(prefix="mini-agent-home-"))
            self.home = self._home_owned_root
        else:
            self.home = home.resolve()
            self.home.mkdir(parents=True, exist_ok=True)

    @classmethod
    async def isolated(
        cls,
        source: Path,
        *,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        runner: ProcessRunner | None = None,
    ) -> "BashEnvironment":
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {source}")
        await asyncio.to_thread(_validate_workspace_symlinks, source)
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-"))
        workspace = root / "workspace"
        home = root / "home"

        def copy_workspace() -> None:
            def ignore_git(_directory: str, names: list[str]) -> list[str]:
                return [".git"] if ".git" in names else []

            shutil.copytree(
                source,
                workspace,
                symlinks=True,
                ignore_dangling_symlinks=False,
                ignore=ignore_git,
            )
            _validate_workspace_symlinks(workspace)

        try:
            await asyncio.to_thread(copy_workspace)
            home.mkdir()
            environment = cls(
                workspace,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                max_patch_bytes=max_patch_bytes,
                owned_root=root,
                source_workspace=source,
                home=home,
                runner=runner,
            )
            await environment._initialize_git_baseline()
            return environment
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, root, True)
            raise

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="bash",
                description=(
                    "Run one bash command in the repository workspace. Each call "
                    "uses a new shell; filesystem changes persist."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
        )

    async def _git(self, *argv: str, max_bytes: int | None = None) -> ProcessResult:
        return await self._runner.run(
            ("git", *argv),
            cwd=self.workspace,
            environment=_minimal_environment(self.home),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_bytes or self.max_output_bytes,
        )

    async def _initialize_git_baseline(self) -> None:
        identity = {
            **_minimal_environment(self.home),
            "GIT_AUTHOR_NAME": "mini-agent",
            "GIT_AUTHOR_EMAIL": "mini-agent@invalid",
            "GIT_COMMITTER_NAME": "mini-agent",
            "GIT_COMMITTER_EMAIL": "mini-agent@invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
        commands = (
            ("git", "init", "--quiet"),
            ("git", "add", "--all", "--force", "--", "."),
            (
                "git",
                "-c",
                "commit.gpgSign=false",
                "commit",
                "--quiet",
                "--allow-empty",
                "--no-verify",
                "-m",
                "mini-agent temporary baseline",
            ),
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        )
        for index, argv in enumerate(commands):
            result = await self._runner.run(
                argv,
                cwd=self.workspace,
                environment=identity,
                timeout_seconds=self.timeout_seconds,
                max_output_bytes=self.max_output_bytes,
            )
            if result.timed_out or result.returncode != 0:
                raise RuntimeError(
                    "could not create temporary Git baseline: " + result.text()
                )
            if index == len(commands) - 1 and result.output:
                raise RuntimeError("temporary Git baseline was not clean after commit")

    async def execute(self, action: ToolCall) -> ToolExecution:
        if self._closed:
            raise RuntimeError("SWE environment is closed")
        if action.name != "bash":
            raise ProtocolError(f"unsupported SWE tool {action.name!r}")
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ProtocolError("bash command must be a non-empty string")
        result = await self._runner.run(
            ("/bin/bash", "--noprofile", "--norc", "-lc", command),
            cwd=self.workspace,
            environment=_minimal_environment(self.home),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
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

    def resource_identity(self) -> str:
        return f"swe-workspace:{self.workspace}"

    async def export_patch(self, destination: Path | None = None) -> bytes:
        if self._closed:
            raise RuntimeError("SWE environment is closed")
        staged = await self._git("add", "--all", "--force", "--", ".")
        if staged.timed_out or staged.returncode != 0:
            raise RuntimeError("could not stage workspace changes: " + staged.text())
        patch = await self._git(
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "HEAD",
            "--",
            ".",
            max_bytes=self.max_patch_bytes,
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
        roots = [root for root in (self._owned_root, self._home_owned_root) if root]
        self._owned_root = None
        self._home_owned_root = None
        for root in dict.fromkeys(roots):
            await asyncio.to_thread(shutil.rmtree, root, True)

    def provenance(self) -> dict[str, object]:
        return {
            "application": "swe",
            "tools": ["bash"],
            "workspace_isolated": self._owned_root is not None,
            "source_workspace": str(self.source_workspace),
            "local_execution_is_security_boundary": False,
            "patch_export": "git_diff_binary",
            "mini_swe_agent_revision": MINI_SWE_AGENT_REVISION,
            "swe_agent_reference_revision": SWE_AGENT_REVISION,
        }
