"""Local process execution: no isolation boundary, one optional private copy."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._lifecycle import (
    combine_lifecycle_errors,
    complete_in_thread,
    raise_lifecycle_errors,
)
from ..types import _require_finite_number, _require_positive_int
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ProcessResult,
    ProcessRunner,
    atomic_write,
    require_ref,
    require_runner,
    require_staging_name,
    resolve_runner,
)


TRUSTED_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
_OWNED_WORKSPACE = object()


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    # The shell can exit while a background descendant keeps stdout open.  Signal
    # the process group even when the group leader has already been reaped.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
    else:
        await asyncio.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.returncode is None:
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
    if len(marker) >= limit:
        return marker[:limit], total, True
    retained = limit - len(marker)
    head_bytes = (retained + 1) // 2
    tail_bytes = retained - head_bytes
    bounded = bytes(head[:head_bytes]) + marker
    if tail_bytes:
        bounded += bytes(tail[-tail_bytes:])
    return bounded, total, True


class LocalProcessRunner:
    """Run one argv without a shell and retain bounded head-plus-tail output."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult:
        """Run one command with hard bounds on time and output.
        
                A process that outlives its timeout is killed with its whole group,
                because an agent's runaway command must not survive the call that
                started it.
                """
        if (
            isinstance(argv, (str, bytes))
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError("argv must contain non-empty strings")
        _require_finite_number(
            timeout_seconds, "timeout_seconds", exclusive_minimum=0
        )
        _require_positive_int(max_output_bytes, "max_output_bytes")
        if environment is not None and (
            not isinstance(environment, Mapping)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            )
        ):
            raise ValueError("environment must map strings to strings")
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
        waiter = asyncio.create_task(process.wait())
        completion = asyncio.gather(waiter, reader)
        timed_out = False
        try:
            _, read_result = await asyncio.wait_for(
                asyncio.shield(completion), timeout_seconds
            )
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate_process_group(process)
            try:
                _, read_result = await asyncio.wait_for(
                    asyncio.shield(completion), timeout=2.0
                )
            except asyncio.TimeoutError as exc:
                completion.cancel()
                await asyncio.gather(completion, return_exceptions=True)
                raise RuntimeError(
                    "subprocess output pipe remained open after group termination"
                ) from exc
        except BaseException:
            await _terminate_process_group(process)
            completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)
            raise
        output, total, truncated = read_result
        return ProcessResult(
            output=output,
            returncode=process.returncode if process.returncode is not None else -1,
            total_output_bytes=total,
            truncated=truncated,
            timed_out=timed_out,
        )


def minimal_environment(home: Path) -> dict[str, str]:
    """Return the trusted, host-config-free environment local commands get."""

    return {
        "PATH": TRUSTED_PATH,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def validate_workspace_symlinks(root: Path) -> None:
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


class LocalRuntime:
    """Run commands directly on the host inside one persistent directory.

    Local execution is a mechanism, not a security boundary: commands run as
    the current user with host filesystem and network access. ``isolated()``
    adds a private copy and a private ``HOME``, nothing more.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        home: Path | None = None,
        source_workspace: Path | None = None,
        runner: ProcessRunner | None = None,
        owned_root: Path | None = None,
        _ownership_token: object | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {workspace}")
        if owned_root is not None:
            if _ownership_token is not _OWNED_WORKSPACE:
                raise ValueError("owned SWE roots are created only by isolated()")
            resolved_root = owned_root.resolve()
            resolved_home = home.resolve() if home is not None else None
            if (
                owned_root.is_symlink()
                or not resolved_root.is_dir()
                or not resolved_root.name.startswith("mini-agent-swe-")
                or self.workspace != resolved_root / "workspace"
                or resolved_home != resolved_root / "home"
            ):
                raise ValueError("owned SWE root has an invalid internal layout")
            owned_root = resolved_root
        self._owned_root = owned_root
        self.isolated_workspace = owned_root is not None
        self.source_workspace = (source_workspace or self.workspace).resolve()
        self.runner = resolve_runner(runner)
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
        source: Path | None = None,
        *,
        scratch_root: Path | None = None,
        runner: ProcessRunner | None = None,
    ) -> "LocalRuntime":
        """Copy ``source`` into a private root with a private ``HOME``.

        ``source=None`` provisions the same private root with an empty
        workspace. What isolates agents from each other is the fresh root, not
        the copy, so a caller that only needs scratch space should not have to
        invent a directory to copy.
        """

        if source is not None:
            source = source.expanduser().resolve()
            if not source.is_dir():
                raise ValueError(f"SWE workspace is not a directory: {source}")
            await asyncio.to_thread(validate_workspace_symlinks, source)
        if runner is not None:
            require_runner(runner)
        root_parent = (
            None if scratch_root is None else scratch_root.expanduser().resolve()
        )
        if root_parent is not None:
            root_parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-", dir=root_parent))
        workspace = root / "workspace"
        home = root / "home"

        def copy_workspace() -> None:
            if source is None:
                workspace.mkdir()
                return

            def ignore_git(_directory: str, names: list[str]) -> list[str]:
                return [".git"] if ".git" in names else []

            shutil.copytree(
                source,
                workspace,
                symlinks=True,
                ignore_dangling_symlinks=False,
                ignore=ignore_git,
            )
            validate_workspace_symlinks(workspace)

        try:
            await complete_in_thread(copy_workspace)
            home.mkdir()
            return cls(
                workspace,
                home=home,
                source_workspace=source,
                runner=runner,
                owned_root=root,
                _ownership_token=_OWNED_WORKSPACE,
            )
        except BaseException as operation_error:
            cleanup_error: BaseException | None = None
            try:
                await complete_in_thread(shutil.rmtree, root)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "isolated workspace setup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    @property
    def workdir(self) -> str:
        return str(self.workspace)

    @property
    def owned_root(self) -> Path | None:
        return self._owned_root

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult:
        return await self.runner.run(
            argv,
            cwd=self.workspace if cwd is None else Path(cwd),
            environment=minimal_environment(self.home) if env is None else env,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    async def read_file(self, path: str) -> bytes:
        require_ref(path, "local file path must be non-empty")
        content = await complete_in_thread(Path(path).read_bytes)
        assert isinstance(content, bytes)
        return content

    async def write_file(self, name: str, data: bytes) -> str:
        """Write ``data`` into the private HOME and return its host path."""

        target = self.home / require_staging_name(name)
        await complete_in_thread(atomic_write, target, data)
        return str(target)

    async def remove_file(self, path: str) -> None:
        Path(path).unlink(missing_ok=True)

    async def close(self) -> None:
        roots = [root for root in (self._owned_root, self._home_owned_root) if root]
        cleanup_error: BaseException | None = None
        for root in dict.fromkeys(roots):
            try:
                if root.exists():
                    await complete_in_thread(shutil.rmtree, root)
            except BaseException as exc:
                cleanup_error = combine_lifecycle_errors(cleanup_error, exc)
        if cleanup_error is not None:
            raise cleanup_error
        self._owned_root = None
        self._home_owned_root = None

    def provenance(self) -> Mapping[str, Any]:
        return {
            "workspace_isolated": self.isolated_workspace,
            "source_workspace": str(self.source_workspace),
            "local_execution_is_security_boundary": False,
        }

    def resource_identity(self) -> str:
        return f"workspace:{self.workspace}"


__all__ = [
    "LocalProcessRunner",
    "LocalRuntime",
    "TRUSTED_PATH",
    "minimal_environment",
    "validate_workspace_symlinks",
]
