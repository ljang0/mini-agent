from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .providers import ProviderError
from .types import ModelRequest, ModelResponse, Usage


_SEMVER = re.compile(
    r"(?<![0-9A-Za-z])v?(\d+\.\d+\.\d+"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)(?![0-9A-Za-z.-])"
)

_DEFAULT_WORKSPACE_HASH_MAX_ENTRIES = 250_000
_DEFAULT_WORKSPACE_HASH_MAX_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS = 30.0


class _WorkspaceHashError(ValueError):
    pass


class _WorkspaceHashLimitExceeded(_WorkspaceHashError):
    pass


class _WorkspaceHashTimedOut(_WorkspaceHashError):
    pass


class _WorkspaceHashCancelled(_WorkspaceHashError):
    pass


def _is_canonical_semver(value: Any) -> bool:
    match = _SEMVER.fullmatch(value) if isinstance(value, str) else None
    return match is not None and match.group(1) == value


def _single_cli_semver(output: str, label: str) -> str:
    versions = set(_SEMVER.findall(output))
    if len(versions) != 1:
        raise ProviderError(
            f"{label} version probe did not identify exactly one semantic version",
            raw={"version_output": output},
        )
    return next(iter(versions))


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {
            "text",
            "output_text",
        }:
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _usage_from_message(message: Any) -> Usage:
    if not isinstance(message, dict):
        return Usage(cost_known=False, complete=False)
    raw = message.get("usage")
    if not isinstance(raw, dict):
        return Usage(cost_known=False, complete=False)
    raw_usage: Mapping[str, Any] = raw

    def count(*names: str) -> tuple[int, bool]:
        for name in names:
            if name not in raw_usage or raw_usage[name] is None:
                continue
            value = raw_usage[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProviderError(f"Prime Agent emitted invalid {name!r} usage")
            return value, True
        return 0, False

    direct_input, direct_input_present = count("input_tokens", "inputTokens")
    cache_read, _ = count("cacheRead")
    cache_write, _ = count("cacheWrite")
    if not direct_input_present:
        uncached_input, input_present = count("input")
        input_tokens = uncached_input + cache_read + cache_write
    else:
        input_tokens = int(direct_input)
        input_present = True
    output_tokens, output_present = count(
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "output",
    )
    nested_cost = raw.get("cost")
    cost_value: Any = None
    if isinstance(nested_cost, dict):
        cost_value = nested_cost.get("total")
    if cost_value is None:
        cost_value = raw.get("cost_usd")
    if cost_value is None:
        cost_value = raw.get("costUsd")
    if (
        isinstance(cost_value, (int, float))
        and not isinstance(cost_value, bool)
        and math.isfinite(cost_value)
        and cost_value >= 0
    ):
        cost_usd = float(cost_value)
        cost_known = True
    else:
        cost_usd = 0.0
        cost_known = False
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cost_usd=cost_usd,
        cost_known=cost_known,
        complete=input_present and output_present,
    )


def _process_environment(pass_env: Sequence[str]) -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TERM",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        *pass_env,
    }
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _validate_workspace_hash_limits(
    *, max_entries: int, max_bytes: int, timeout_seconds: float
) -> None:
    if (
        not isinstance(max_entries, int)
        or isinstance(max_entries, bool)
        or max_entries < 1
    ):
        raise ValueError("workspace_hash_max_entries must be a positive integer")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("workspace_hash_max_bytes must be a positive integer")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("workspace_hash_timeout_seconds must be positive and finite")


def _workspace_tree_sha256(
    root: Path,
    *,
    max_entries: int = _DEFAULT_WORKSPACE_HASH_MAX_ENTRIES,
    max_bytes: int = _DEFAULT_WORKSPACE_HASH_MAX_BYTES,
    timeout_seconds: float = _DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Hash a non-Git working tree within explicit resource ceilings.

    ``max_entries`` counts both directories and non-directory entries so a tree made
    only of empty directories cannot evade the file-count guard. ``max_bytes`` counts
    regular-file content. The cancellation event is checked between directory entries
    and file chunks; the async wrapper supplies it so cancellation remains prompt even
    though portable regular-file I/O runs in a worker thread.
    """

    _validate_workspace_hash_limits(
        max_entries=max_entries,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )
    root = root.resolve()
    started = time.monotonic()
    digest = hashlib.sha256()
    entries_seen = 0
    content_bytes_seen = 0

    def check_interrupts() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _WorkspaceHashCancelled("workspace hashing was cancelled")
        if time.monotonic() - started >= timeout_seconds:
            raise _WorkspaceHashTimedOut(
                f"workspace hashing exceeded {timeout_seconds} seconds"
            )

    def count_entry() -> None:
        nonlocal entries_seen
        check_interrupts()
        entries_seen += 1
        if entries_seen > max_entries:
            raise _WorkspaceHashLimitExceeded(
                f"workspace hashing exceeded {max_entries} filesystem entries"
            )

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current_root, directory_names, file_names in os.walk(
        root, onerror=raise_walk_error
    ):
        check_interrupts()
        current = Path(current_root)
        kept_directories: list[str] = []
        for directory_name in sorted(
            name for name in directory_names if name != ".git"
        ):
            count_entry()
            directory = current / directory_name
            if directory.is_symlink():
                if not directory.resolve(strict=False).is_relative_to(root):
                    raise _WorkspaceHashError(
                        f"external workspace symlink escapes its root: {directory}"
                    )
                relative = directory.relative_to(root).as_posix()
                digest.update(f"L\0{relative}\0".encode("utf-8"))
                digest.update(os.fsencode(os.readlink(directory)))
                digest.update(b"\0")
            else:
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories
        for file_name in sorted(name for name in file_names if name != ".git"):
            count_entry()
            path = current / file_name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if not path.resolve(strict=False).is_relative_to(root):
                    raise _WorkspaceHashError(
                        f"external workspace symlink escapes its root: {path}"
                    )
                digest.update(f"L\0{relative}\0".encode("utf-8"))
                digest.update(os.fsencode(os.readlink(path)))
                digest.update(b"\0")
                continue
            if not path.is_file():
                file_stat = path.lstat()
                digest.update(b"S\0")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(file_stat.st_mode).encode("ascii"))
                digest.update(b"\0")
                continue
            file_stat = path.stat()
            if file_stat.st_size > max_bytes - content_bytes_seen:
                raise _WorkspaceHashLimitExceeded(
                    f"workspace hashing exceeded {max_bytes} content bytes"
                )
            digest.update(b"F\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.S_IMODE(file_stat.st_mode)).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while True:
                    check_interrupts()
                    remaining = max_bytes - content_bytes_seen
                    chunk = stream.read(min(1024 * 1024, remaining + 1))
                    if not chunk:
                        break
                    if len(chunk) > remaining:
                        raise _WorkspaceHashLimitExceeded(
                            f"workspace hashing exceeded {max_bytes} content bytes"
                        )
                    content_bytes_seen += len(chunk)
                    digest.update(chunk)
            digest.update(b"\0")
    check_interrupts()
    return digest.hexdigest()


async def _workspace_tree_sha256_async(
    root: Path,
    *,
    max_entries: int,
    max_bytes: int,
    timeout_seconds: float,
) -> str:
    """Hash without blocking the event loop and stop cooperatively on cancellation."""

    _validate_workspace_hash_limits(
        max_entries=max_entries,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )
    cancel_event = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _workspace_tree_sha256,
            root,
            max_entries=max_entries,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
    )
    try:
        return await asyncio.wait_for(asyncio.shield(worker), timeout_seconds)
    except asyncio.TimeoutError as exc:
        cancel_event.set()
        worker.cancel()
        raise _WorkspaceHashTimedOut(
            f"workspace hashing exceeded {timeout_seconds} seconds"
        ) from exc
    except asyncio.CancelledError:
        cancel_event.set()
        worker.cancel()
        raise


def _git_workspace_provenance(root: Path) -> Mapping[str, Any]:
    environment = _process_environment(())
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            env=environment,
            timeout=15,
        ).stdout.strip()
        status_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            env=environment,
            timeout=30,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False}
    return {
        "available": True,
        "head": head.decode("ascii", errors="replace"),
        "dirty": bool(status_bytes),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
    }


def _resolved_executable_identity(executable: str) -> Mapping[str, Any]:
    candidate = shutil.which(executable)
    if candidate is None:
        explicit = Path(executable)
        candidate = str(explicit) if explicit.is_file() else None
    if candidate is None:
        return {"available": False}
    invoked_path = Path(candidate).absolute()
    resolved_path = invoked_path.resolve()
    try:
        digest = hashlib.sha256()
        with resolved_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {
            "available": False,
            "invoked_path": str(invoked_path),
            "resolved_path": str(resolved_path),
        }
    return {
        "available": True,
        "invoked_path": str(invoked_path),
        "resolved_path": str(resolved_path),
        "sha256": digest.hexdigest(),
    }


def _prepare_executable(
    executable: str,
    expected_sha256: Optional[str],
    label: str,
) -> tuple[str, Mapping[str, Any]]:
    identity = _resolved_executable_identity(executable)
    if expected_sha256 is not None:
        observed = identity.get("sha256")
        if observed != expected_sha256:
            raise ProviderError(
                f"{label} executable SHA-256 mismatch",
                raw={"executable_identity": dict(identity)},
            )
    resolved = identity.get("resolved_path")
    return (
        resolved if isinstance(resolved, str) else executable,
        identity,
    )


class _CapturedOutput:
    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()


class _ProcessOutputLimitExceeded(RuntimeError):
    pass


async def _read_stream_limited(
    stream: asyncio.StreamReader,
    buffer: bytearray,
    *,
    limit: int,
    label: str,
) -> None:
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return
        remaining = limit - len(buffer)
        if len(chunk) > remaining:
            if remaining > 0:
                buffer.extend(chunk[:remaining])
            raise _ProcessOutputLimitExceeded(
                f"external process {label} exceeded {limit} bytes"
            )
        buffer.extend(chunk)


async def _write_process_input(
    stream: Optional[asyncio.StreamWriter], value: Optional[bytes]
) -> None:
    if stream is None:
        if value is not None:
            raise RuntimeError("external process has no stdin pipe")
        return
    try:
        if value is not None:
            stream.write(value)
            await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream.close()
        wait_closed = getattr(stream, "wait_closed", None)
        if callable(wait_closed):
            try:
                await wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass


async def _communicate_limited(
    process: asyncio.subprocess.Process,
    *,
    input_data: Optional[bytes],
    max_output_bytes: int,
    captured: _CapturedOutput,
) -> tuple[bytes, bytes]:
    stdout_stream = getattr(process, "stdout", None)
    stderr_stream = getattr(process, "stderr", None)
    if not isinstance(stdout_stream, asyncio.StreamReader) or not isinstance(
        stderr_stream, asyncio.StreamReader
    ):
        # Unit-test doubles use communicate directly; production PIPEs take the
        # bounded streaming path above.
        if input_data is None:
            stdout, stderr = await process.communicate()
        else:
            stdout, stderr = await process.communicate(input_data)
        stdout = stdout or b""
        stderr = stderr or b""
        captured.stdout.extend(stdout[:max_output_bytes])
        captured.stderr.extend(stderr[:max_output_bytes])
        if len(stdout) > max_output_bytes or len(stderr) > max_output_bytes:
            raise _ProcessOutputLimitExceeded(
                f"external process output exceeded {max_output_bytes} bytes"
            )
        return stdout, stderr

    tasks = [
        asyncio.create_task(
            _read_stream_limited(
                stdout_stream,
                captured.stdout,
                limit=max_output_bytes,
                label="stdout",
            )
        ),
        asyncio.create_task(
            _read_stream_limited(
                stderr_stream,
                captured.stderr,
                limit=max_output_bytes,
                label="stderr",
            )
        ),
        asyncio.create_task(_write_process_input(process.stdin, input_data)),
        asyncio.create_task(process.wait()),
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return bytes(captured.stdout), bytes(captured.stderr)


def _signal_process_group(
    process: asyncio.subprocess.Process, sig: signal.Signals
) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif process.returncode is None:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
    except ProcessLookupError:
        return


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        _signal_process_group(process, signal.SIGTERM)
    elif process.returncode is None:
        process.terminate()
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    # The direct process can exit while a descendant in its new session survives.
    if os.name == "posix":
        _signal_process_group(process, signal.SIGKILL)
    elif process.returncode is None:
        process.kill()
    if process.returncode is None:
        await process.wait()


def _grok_token_count(raw: Mapping[str, Any], name: str) -> int:
    if name not in raw:
        raise ProviderError(f"Grok Build omitted required {name!r} usage")
    value = raw[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderError(f"Grok Build emitted invalid {name!r} usage")
    return value


def _usage_from_grok_result(result: Any) -> Usage:
    """Project Grok's documented headless spend fields into Scaffold Lab usage.

    Grok reports uncached input separately. ``Usage.input_tokens`` intentionally
    remains the full prompt total, so cache reads and cache creation are added once.
    Reasoning tokens are already included in Grok's output/total policy and must not
    be added again.
    """

    if not isinstance(result, dict):
        return Usage(cost_known=False, complete=False)
    raw_usage = result.get("usage")
    if not isinstance(raw_usage, dict):
        return Usage(cost_known=False, complete=False)

    uncached_input = _grok_token_count(raw_usage, "input_tokens")
    cache_read = _grok_token_count(raw_usage, "cache_read_input_tokens")
    cache_write = _grok_token_count(raw_usage, "cache_creation_input_tokens")
    output_tokens = _grok_token_count(raw_usage, "output_tokens")
    token_lower_bound = Usage(
        input_tokens=uncached_input + cache_read + cache_write,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cost_known=False,
        complete=False,
    )

    raw_usage_incomplete = result.get("usage_is_incomplete", False)
    if not isinstance(raw_usage_incomplete, bool):
        raise ProviderError(
            "Grok Build emitted invalid 'usage_is_incomplete'",
            usage=token_lower_bound,
        )
    usage_incomplete = raw_usage_incomplete

    raw_cost_is_partial = result.get("cost_is_partial", False)
    if not isinstance(raw_cost_is_partial, bool):
        raise ProviderError(
            "Grok Build emitted invalid 'cost_is_partial'",
            usage=token_lower_bound,
        )
    cost_is_partial = raw_cost_is_partial
    cost_known = False
    cost_usd = 0.0
    ticks = result.get("total_cost_usd_ticks")
    dollars = result.get("total_cost_usd")
    if ticks is not None:
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
            raise ProviderError(
                "Grok Build emitted invalid 'total_cost_usd_ticks'",
                usage=token_lower_bound,
            )
        cost_usd = ticks / 10_000_000_000
        cost_known = not usage_incomplete and not cost_is_partial
    elif dollars is not None:
        if (
            not isinstance(dollars, (int, float))
            or isinstance(dollars, bool)
            or not math.isfinite(dollars)
            or dollars < 0
        ):
            raise ProviderError(
                "Grok Build emitted invalid 'total_cost_usd'",
                usage=token_lower_bound,
            )
        cost_usd = float(dollars)
        cost_known = not usage_incomplete and not cost_is_partial

    return Usage(
        input_tokens=uncached_input + cache_read + cache_write,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cost_usd=cost_usd,
        cost_known=cost_known,
        complete=not usage_incomplete,
    )


def _mark_whole_tree_unverified(usage: Usage) -> Usage:
    """Preserve observed lower bounds without claiming complete tree accounting."""

    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        cost_usd=usage.cost_usd,
        cost_known=False,
        complete=False,
    )


def _prime_usage_lower_bound(data: bytes) -> Usage:
    observed: Optional[Usage] = None
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        try:
            message_usage = _usage_from_message(message)
        except (ProviderError, ValueError):
            continue
        observed = message_usage if observed is None else observed + message_usage
    return _mark_whole_tree_unverified(
        observed or Usage(cost_known=False, complete=False)
    )


class GrokBuildJSONBackend:
    """Run xAI's released Grok Build harness in documented headless JSON mode.

    Each completion is a fresh external Grok session with native subagents enabled.
    A private prompt file and ephemeral ``GROK_HOME`` avoid prompt disclosure in the
    process list and cross-trial session/memory contamination. Authentication must be
    passed explicitly (normally ``XAI_API_KEY``); cached login state is not inherited.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        model: str,
        executable: str = "grok",
        sandbox: str = "strict",
        permission_mode: str = "dontAsk",
        max_turns: int = 64,
        timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        allow_rules: Sequence[str] = (),
        deny_rules: Sequence[str] = (),
        disallowed_tools: Sequence[str] = ("run_terminal_cmd",),
        expected_version: Optional[str] = None,
        expected_executable_sha256: Optional[str] = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        workspace_hash_max_entries: int = _DEFAULT_WORKSPACE_HASH_MAX_ENTRIES,
        workspace_hash_max_bytes: int = _DEFAULT_WORKSPACE_HASH_MAX_BYTES,
        workspace_hash_timeout_seconds: float = (
            _DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS
        ),
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Grok Build adapter currently requires POSIX process-group isolation"
            )
        self.cwd = cwd.resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"Grok Build cwd is not a directory: {self.cwd}")
        for name, value in (
            ("model", model),
            ("executable", executable),
            ("sandbox", sandbox),
            ("permission_mode", permission_mode),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        for label, values in (
            ("pass_env", pass_env),
            ("allow_rules", allow_rules),
            ("deny_rules", deny_rules),
            ("disallowed_tools", disallowed_tools),
        ):
            if any(
                not isinstance(value, str)
                or not value
                or (label == "pass_env" and "=" in value)
                for value in values
            ):
                raise ValueError(f"{label} entries must be non-empty strings")
        if (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes < 1024
        ):
            raise ValueError("max_output_bytes must be an integer of at least 1024")
        if expected_version is not None and not _is_canonical_semver(expected_version):
            raise ValueError("expected_version must be a canonical semantic version")
        if (
            expected_executable_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", expected_executable_sha256) is None
        ):
            raise ValueError(
                "expected_executable_sha256 must be 64 lowercase hex chars"
            )
        _validate_workspace_hash_limits(
            max_entries=workspace_hash_max_entries,
            max_bytes=workspace_hash_max_bytes,
            timeout_seconds=workspace_hash_timeout_seconds,
        )

        self.executable = executable
        self.model = model
        self.sandbox = sandbox
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.timeout_seconds = float(timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.allow_rules = tuple(allow_rules)
        self.deny_rules = tuple(deny_rules)
        self.disallowed_tools = tuple(disallowed_tools)
        self.expected_version = expected_version
        self.expected_executable_sha256 = expected_executable_sha256
        self.max_output_bytes = max_output_bytes
        self.workspace_hash_max_entries = workspace_hash_max_entries
        self.workspace_hash_max_bytes = workspace_hash_max_bytes
        self.workspace_hash_timeout_seconds = float(workspace_hash_timeout_seconds)
        self.workspace_tree_sha256 = _workspace_tree_sha256(
            self.cwd,
            max_entries=self.workspace_hash_max_entries,
            max_bytes=self.workspace_hash_max_bytes,
            timeout_seconds=self.workspace_hash_timeout_seconds,
        )
        self.git_workspace = _git_workspace_provenance(self.cwd)
        self._resolved_executable: Optional[str] = None
        self._executable_identity: Mapping[str, Any] = {"available": False}
        self._observed_version_output: Optional[str] = None
        self._observed_version: Optional[str] = None

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "grok-build-cli",
            "executable": self.executable,
            "cwd": str(self.cwd),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "workspace_isolation": "caller-provisioned-single-trial-workspace",
            "model": self.model,
            "sandbox": self.sandbox,
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
            "timeout_seconds": self.timeout_seconds,
            "allow_rules": list(self.allow_rules),
            "deny_rules": list(self.deny_rules),
            "disallowed_tools": list(self.disallowed_tools),
            "max_output_bytes_per_stream": self.max_output_bytes,
            "workspace_hash_limits": {
                "max_entries": self.workspace_hash_max_entries,
                "max_content_bytes": self.workspace_hash_max_bytes,
                "timeout_seconds": self.workspace_hash_timeout_seconds,
            },
            "passed_environment_names": sorted(self.pass_env),
            "expected_version": self.expected_version,
            "expected_executable_sha256": self.expected_executable_sha256,
            "runtime_executable": dict(self._executable_identity),
            "runtime_package_identity_matches_audited_release": False,
            "observed_version_output": self._observed_version_output,
            "observed_version": self._observed_version,
            "version_verified": self._observed_version_output is not None,
            "fresh_session": True,
            "memory_disabled": True,
            "subagents_enabled": True,
            "grok_home": "ephemeral-per-call",
            "headless_output_format": "json",
            "audited_release": {
                "package": "@xai-official/grok",
                "version": "1.0.0",
                "integrity": (
                    "sha512-71ZbT7qggTsgmqt3pBifamlv4HZ5BFliXUrtgNQDlj9NlQc"
                    "ZikYjxNYciXm5CzYI+ZDezM2tREWFuOdfZjVnXA=="
                ),
                "npm_git_head": "3cd0d0cbcebeb5b94a2830326ceb466d4341a5c4",
                "public_repository_commit": (
                    "8a14c91d88875a831a38b3a066b1683116bcb31c"
                ),
                "public_source_rev": "27b3c66635e2c0bf213429a36ab916f25d59df20",
                "npm_to_public_source_identity_verified": False,
            },
        }

    async def verify_version(self) -> str:
        """Probe the CLI version and enforce an optional declared version pin."""

        if self._observed_version_output is not None:
            return self._observed_version_output
        resolved_executable, identity = _prepare_executable(
            self.executable,
            self.expected_executable_sha256,
            "Grok Build",
        )
        self._resolved_executable = resolved_executable
        self._executable_identity = identity
        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-grok-version-"
        ) as temp_dir:
            environment = _process_environment(())
            environment["GROK_HOME"] = str(Path(temp_dir) / "grok-home")
            environment["GROK_DISABLE_AUTOUPDATER"] = "1"
            Path(environment["GROK_HOME"]).mkdir(mode=0o700)
            try:
                process = await asyncio.create_subprocess_exec(
                    resolved_executable,
                    "version",
                    cwd=str(self.cwd),
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"Grok Build executable not found: {self.executable!r}"
                ) from exc
            captured = _CapturedOutput()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data=None,
                        max_output_bytes=min(self.max_output_bytes, 1024 * 1024),
                        captured=captured,
                    ),
                    timeout=min(self.timeout_seconds, 30.0),
                )
            except asyncio.TimeoutError:
                await _terminate_process_tree(process)
                raise ProviderError("Grok Build version probe timed out")
            except asyncio.CancelledError:
                await _terminate_process_tree(process)
                raise
            except _ProcessOutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "Grok Build version probe exceeded its output limit",
                    raw={
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise
            await _terminate_process_tree(process)
            combined = "\n".join(
                part
                for part in (
                    stdout.decode("utf-8", errors="replace").strip(),
                    stderr.decode("utf-8", errors="replace").strip(),
                )
                if part
            )
            if process.returncode != 0:
                raise ProviderError(
                    "Grok Build version probe exited with status "
                    f"{process.returncode}: {combined}"
                )
            if not combined:
                raise ProviderError("Grok Build version probe returned no version")
            observed_version = _single_cli_semver(combined, "Grok Build")
            if (
                self.expected_version is not None
                and self.expected_version != observed_version
            ):
                raise ProviderError(
                    "Grok Build version mismatch: expected "
                    f"{self.expected_version!r}, observed {combined!r}"
                )
            self._observed_version_output = combined
            self._observed_version = observed_version
            return combined

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.expected_version is not None and self._observed_version_output is None:
            await self.verify_version()
        try:
            current_workspace_hash = await _workspace_tree_sha256_async(
                self.cwd,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.workspace_hash_max_bytes,
                timeout_seconds=self.workspace_hash_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError) as exc:
            raise ProviderError(
                "Grok Build could not hash its workspace before the session"
            ) from exc
        if current_workspace_hash != self.workspace_tree_sha256:
            raise ProviderError(
                "Grok Build workspace changed after backend initialization"
            )
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"

        with tempfile.TemporaryDirectory(prefix="scaffoldlab-grok-") as temp_dir:
            temp_root = Path(temp_dir)
            prompt_path = temp_root / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            prompt_path.chmod(0o600)
            grok_home = temp_root / "grok-home"
            grok_home.mkdir(mode=0o700)
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            home.mkdir(mode=0o700)
            xdg_config.mkdir(mode=0o700)

            command = [
                self._resolved_executable or self.executable,
                "--prompt-file",
                str(prompt_path),
                "--verbatim",
                "--cwd",
                str(self.cwd),
                "--output-format",
                "json",
                "--no-auto-update",
                "--no-memory",
                "--sandbox",
                self.sandbox,
                "--permission-mode",
                self.permission_mode,
                "--max-turns",
                str(self.max_turns),
                "--model",
                self.model,
            ]
            for rule in self.allow_rules:
                command.extend(["--allow", rule])
            for rule in self.deny_rules:
                command.extend(["--deny", rule])
            if self.disallowed_tools:
                command.extend(["--disallowed-tools", ",".join(self.disallowed_tools)])

            environment = _process_environment(self.pass_env)
            environment["GROK_HOME"] = str(grok_home)
            environment["GROK_DISABLE_AUTOUPDATER"] = "1"
            environment["HOME"] = str(home)
            environment["XDG_CONFIG_HOME"] = str(xdg_config)
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.cwd),
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"Grok Build executable not found: {self.executable!r}"
                ) from exc

            captured = _CapturedOutput()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data=None,
                        max_output_bytes=self.max_output_bytes,
                        captured=captured,
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "Grok Build session timed out",
                    usage=Usage(cost_known=False, complete=False),
                    raw={
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                )
            except asyncio.CancelledError:
                await _terminate_process_tree(process)
                raise
            except _ProcessOutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "Grok Build session exceeded its output limit",
                    usage=Usage(cost_known=False, complete=False),
                    raw={
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise

            # The CLI process may have spawned background tool/subagent children. Its
            # own exit is not proof that the whole process group is gone.
            await _terminate_process_tree(process)

            decoded = stdout.decode("utf-8", errors="replace").strip()
            try:
                result = json.loads(decoded) if decoded else None
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "Grok Build emitted invalid JSON",
                    usage=Usage(cost_known=False, complete=False),
                    raw={
                        "stdout": decoded,
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                ) from exc
            if not isinstance(result, dict):
                raise ProviderError(
                    "Grok Build emitted a non-object JSON result",
                    usage=Usage(cost_known=False, complete=False),
                    raw=result,
                )

            usage = _mark_whole_tree_unverified(_usage_from_grok_result(result))
            if request.usage_reporter is not None:
                request.usage_reporter(usage)
            try:
                post_workspace_hash = await _workspace_tree_sha256_async(
                    self.cwd,
                    max_entries=self.workspace_hash_max_entries,
                    max_bytes=self.workspace_hash_max_bytes,
                    timeout_seconds=self.workspace_hash_timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except Exception as exc:
                raise ProviderError(
                    "Grok Build could not hash its workspace after the session",
                    usage=usage,
                    raw={
                        "result": result,
                        "stderr": stderr.decode("utf-8", errors="replace"),
                        "workspace_hash_error": str(exc),
                    },
                ) from exc
            result = {
                **result,
                "_scaffoldlab_workspace": {
                    "cwd": str(self.cwd),
                    "pre_tree_sha256": self.workspace_tree_sha256,
                    "post_tree_sha256": post_workspace_hash,
                },
            }
            if process.returncode != 0:
                raise ProviderError(
                    f"Grok Build exited with status {process.returncode}",
                    usage=usage,
                    raw={
                        "result": result,
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )

            stop_reason = result.get("stopReason")
            answer = result.get("text")
            if stop_reason != "end_turn":
                raise ProviderError(
                    f"Grok Build ended with stopReason={stop_reason!r}",
                    usage=usage,
                    raw=result,
                )
            if not isinstance(answer, str) or not answer:
                raise ProviderError(
                    "Grok Build JSON result contained no final text",
                    usage=usage,
                    raw=result,
                )
            return ModelResponse(text=answer, usage=usage, raw=result)


class PrimeAgentJSONBackend:
    """Run the released Prime Agent runtime through its documented JSON mode.

    Prime Agent executes model-generated Python and project commands with the current
    user's permissions. Use a disposable clone/worktree or an external sandbox for
    untrusted tasks. One backend completion represents an entire Prime Agent session tree,
    not one underlying provider call.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        executable: str = "prime-agent",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        no_session: bool = True,
        timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        expected_version: Optional[str] = None,
        expected_executable_sha256: Optional[str] = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        allow_sensitive_environment: bool = False,
        workspace_hash_max_entries: int = _DEFAULT_WORKSPACE_HASH_MAX_ENTRIES,
        workspace_hash_max_bytes: int = _DEFAULT_WORKSPACE_HASH_MAX_BYTES,
        workspace_hash_timeout_seconds: float = (
            _DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS
        ),
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Prime Agent adapter currently requires POSIX process-group isolation"
            )
        self.cwd = cwd.resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"Prime Agent cwd is not a directory: {self.cwd}")
        if not isinstance(executable, str) or not executable:
            raise ValueError("executable must be a non-empty string")
        for name, value in (("provider", provider), ("model", model)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or null")
        if not isinstance(no_session, bool):
            raise ValueError("no_session must be a boolean")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if any(
            not isinstance(name, str) or not name or "=" in name for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        if not isinstance(allow_sensitive_environment, bool):
            raise ValueError("allow_sensitive_environment must be a boolean")
        if pass_env and not allow_sensitive_environment:
            raise ValueError(
                "Prime Agent can inspect every passed environment value; set "
                "allow_sensitive_environment only with scoped credentials and an "
                "appropriate outer sandbox"
            )
        if expected_version is not None and not _is_canonical_semver(expected_version):
            raise ValueError("expected_version must be a canonical semantic version")
        if (
            expected_executable_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", expected_executable_sha256) is None
        ):
            raise ValueError(
                "expected_executable_sha256 must be 64 lowercase hex chars"
            )
        if (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes < 1024
        ):
            raise ValueError("max_output_bytes must be an integer of at least 1024")
        _validate_workspace_hash_limits(
            max_entries=workspace_hash_max_entries,
            max_bytes=workspace_hash_max_bytes,
            timeout_seconds=workspace_hash_timeout_seconds,
        )
        self.executable = executable
        self.provider = provider
        self.model = model
        self.no_session = no_session
        self.timeout_seconds = float(timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.expected_version = expected_version
        self.expected_executable_sha256 = expected_executable_sha256
        self.max_output_bytes = max_output_bytes
        self.allow_sensitive_environment = allow_sensitive_environment
        self.workspace_hash_max_entries = workspace_hash_max_entries
        self.workspace_hash_max_bytes = workspace_hash_max_bytes
        self.workspace_hash_timeout_seconds = float(workspace_hash_timeout_seconds)
        self.workspace_tree_sha256 = _workspace_tree_sha256(
            self.cwd,
            max_entries=self.workspace_hash_max_entries,
            max_bytes=self.workspace_hash_max_bytes,
            timeout_seconds=self.workspace_hash_timeout_seconds,
        )
        self.git_workspace = _git_workspace_provenance(self.cwd)
        self._observed_version_output: Optional[str] = None
        self._observed_version: Optional[str] = None
        self._resolved_executable: Optional[str] = None
        self._executable_identity: Mapping[str, Any] = {"available": False}

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "prime-agent-cli",
            "executable": self.executable,
            "cwd": str(self.cwd),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "workspace_isolation": "caller-provisioned-single-trial-workspace",
            "prime_provider": self.provider,
            "model": self.model,
            "no_session": self.no_session,
            "timeout_seconds": self.timeout_seconds,
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "workspace_hash_limits": {
                "max_entries": self.workspace_hash_max_entries,
                "max_content_bytes": self.workspace_hash_max_bytes,
                "timeout_seconds": self.workspace_hash_timeout_seconds,
            },
            "expected_version": self.expected_version,
            "expected_executable_sha256": self.expected_executable_sha256,
            "runtime_executable": dict(self._executable_identity),
            "runtime_package_identity_matches_audited_release": False,
            "observed_version_output": self._observed_version_output,
            "observed_version": self._observed_version,
            "version_verified": self._observed_version_output is not None,
            "one_backend_call_is_external_session_tree": True,
            "home": "ephemeral-per-call",
            "xdg_config_home": "ephemeral-per-call",
            "audited_release": {
                "version": "0.7.1",
                "repository_commit": ("95afd319a78ae017a41241d50b013d656a0685ce"),
            },
        }

    async def verify_version(self) -> str:
        """Probe the released CLI and enforce an optional exact version pin."""

        if self._observed_version_output is not None:
            return self._observed_version_output
        resolved_executable, identity = _prepare_executable(
            self.executable,
            self.expected_executable_sha256,
            "Prime Agent",
        )
        self._resolved_executable = resolved_executable
        self._executable_identity = identity
        try:
            process = await asyncio.create_subprocess_exec(
                resolved_executable,
                "--version",
                cwd=str(self.cwd),
                env=_process_environment(()),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                f"Prime Agent executable not found: {self.executable!r}"
            ) from exc
        captured = _CapturedOutput()
        try:
            stdout, stderr = await asyncio.wait_for(
                _communicate_limited(
                    process,
                    input_data=None,
                    max_output_bytes=min(self.max_output_bytes, 1024 * 1024),
                    captured=captured,
                ),
                timeout=min(self.timeout_seconds, 30.0),
            )
        except asyncio.TimeoutError:
            await _terminate_process_tree(process)
            raise ProviderError("Prime Agent version probe timed out")
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        except _ProcessOutputLimitExceeded as exc:
            await _terminate_process_tree(process)
            raise ProviderError(
                "Prime Agent version probe exceeded its output limit",
                raw={
                    "stdout": bytes(captured.stdout).decode("utf-8", errors="replace"),
                    "stderr": bytes(captured.stderr).decode("utf-8", errors="replace"),
                },
            ) from exc
        except Exception:
            await _terminate_process_tree(process)
            raise
        await _terminate_process_tree(process)
        combined = "\n".join(
            part
            for part in (
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            )
            if part
        )
        if process.returncode != 0:
            raise ProviderError(
                "Prime Agent version probe exited with status "
                f"{process.returncode}: {combined}"
            )
        if not combined:
            raise ProviderError("Prime Agent version probe returned no version")
        observed_version = _single_cli_semver(combined, "Prime Agent")
        if (
            self.expected_version is not None
            and self.expected_version != observed_version
        ):
            raise ProviderError(
                "Prime Agent version mismatch: expected "
                f"{self.expected_version!r}, observed {combined!r}"
            )
        self._observed_version_output = combined
        self._observed_version = observed_version
        return combined

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.expected_version is not None and self._observed_version_output is None:
            await self.verify_version()
        try:
            current_workspace_hash = await _workspace_tree_sha256_async(
                self.cwd,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.workspace_hash_max_bytes,
                timeout_seconds=self.workspace_hash_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError) as exc:
            raise ProviderError(
                "Prime Agent could not hash its workspace before the session"
            ) from exc
        if current_workspace_hash != self.workspace_tree_sha256:
            raise ProviderError(
                "Prime Agent workspace changed after backend initialization"
            )
        with tempfile.TemporaryDirectory(prefix="scaffoldlab-prime-home-") as temp_dir:
            temp_root = Path(temp_dir)
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            home.mkdir(mode=0o700)
            xdg_config.mkdir(mode=0o700)
            environment = _process_environment(self.pass_env)
            environment["HOME"] = str(home)
            environment["XDG_CONFIG_HOME"] = str(xdg_config)
            return await self._complete_with_environment(request, environment)

    async def _complete_with_environment(
        self, request: ModelRequest, environment: Mapping[str, str]
    ) -> ModelResponse:
        command = [self._resolved_executable or self.executable, "--mode", "json"]
        if self.provider:
            command.extend(["--provider", self.provider])
        if self.model:
            command.extend(["--model", self.model])
        if self.no_session:
            command.append("--no-session")
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.cwd),
                env=dict(environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                f"Prime Agent executable not found: {self.executable!r}"
            ) from exc
        captured = _CapturedOutput()
        try:
            stdout, stderr = await asyncio.wait_for(
                _communicate_limited(
                    process,
                    input_data=prompt.encode("utf-8"),
                    max_output_bytes=self.max_output_bytes,
                    captured=captured,
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            await _terminate_process_tree(process)
            partial_usage = _prime_usage_lower_bound(bytes(captured.stdout))
            if request.usage_reporter is not None:
                request.usage_reporter(partial_usage)
            raise ProviderError(
                "Prime Agent session timed out",
                usage=partial_usage,
                raw={
                    "stdout": bytes(captured.stdout).decode("utf-8", errors="replace"),
                    "stderr": bytes(captured.stderr).decode("utf-8", errors="replace"),
                },
            )
        except asyncio.CancelledError as exc:
            await _terminate_process_tree(process)
            partial_usage = _prime_usage_lower_bound(bytes(captured.stdout))
            if request.usage_reporter is not None:
                request.usage_reporter(partial_usage)
            exc.usage = partial_usage  # type: ignore[attr-defined]
            raise
        except _ProcessOutputLimitExceeded as exc:
            await _terminate_process_tree(process)
            raise ProviderError(
                "Prime Agent session exceeded its output limit",
                usage=_prime_usage_lower_bound(bytes(captured.stdout)),
                raw={
                    "stdout": bytes(captured.stdout).decode("utf-8", errors="replace"),
                    "stderr": bytes(captured.stderr).decode("utf-8", errors="replace"),
                },
            ) from exc
        except Exception:
            await _terminate_process_tree(process)
            raise
        await _terminate_process_tree(process)
        usage_lower_bound = _prime_usage_lower_bound(stdout)
        if request.usage_reporter is not None:
            request.usage_reporter(usage_lower_bound)
        try:
            post_workspace_hash = await _workspace_tree_sha256_async(
                self.cwd,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.workspace_hash_max_bytes,
                timeout_seconds=self.workspace_hash_timeout_seconds,
            )
        except asyncio.CancelledError as exc:
            exc.usage = usage_lower_bound  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            raise ProviderError(
                "Prime Agent could not hash its workspace after the session",
                usage=usage_lower_bound,
                raw={
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "workspace_hash_error": str(exc),
                },
            ) from exc
        events: list[dict[str, Any]] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "Prime Agent emitted a non-JSON stdout line",
                    usage=_prime_usage_lower_bound(stdout),
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                ) from exc
            if isinstance(event, dict):
                events.append(event)
        events.append(
            {
                "type": "scaffoldlab_workspace",
                "cwd": str(self.cwd),
                "pre_tree_sha256": self.workspace_tree_sha256,
                "post_tree_sha256": post_workspace_hash,
            }
        )
        final_message: Any = None
        terminal_failure: Optional[str] = None
        usage: Optional[Usage] = None
        for event in events:
            if event.get("type") == "message_end":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if role == "assistant":
                    try:
                        message_usage = _usage_from_message(message)
                    except ProviderError as exc:
                        raise ProviderError(
                            "Prime Agent emitted malformed assistant usage",
                            usage=_prime_usage_lower_bound(stdout),
                            raw=events,
                        ) from exc
                    usage = message_usage if usage is None else usage + message_usage
                    reason = message.get("stopReason") or message.get("stop_reason")
                    if reason == "stop":
                        final_message = message
                        terminal_failure = None
                    elif reason in {"length", "error", "aborted"}:
                        terminal_failure = str(reason)
        observed_usage = _mark_whole_tree_unverified(
            usage or Usage(cost_known=False, complete=False)
        )
        if process.returncode != 0:
            raise ProviderError(
                f"Prime Agent exited with status {process.returncode}",
                usage=observed_usage,
                raw={
                    "events": events,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        if terminal_failure is not None:
            raise ProviderError(
                f"Prime Agent ended with stopReason={terminal_failure!r}",
                usage=observed_usage,
                raw=events,
            )
        answer = _message_text(final_message)
        if not answer:
            raise ProviderError(
                "Prime Agent JSON stream contained no final assistant text",
                usage=observed_usage,
                raw=events,
            )
        return ModelResponse(text=answer, usage=observed_usage, raw=events)


_MACU_RELEASE_REVISION = "5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105"


def _macu_usage_from_summary(summary: Any) -> Usage:
    """Aggregate the released MACU manager and CUA accounting artifacts.

    ``manager_calls`` contains the runtime's recorded manager-call list. Manager
    entries in ``subtask_costs`` are therefore skipped to avoid double counting; CUA
    entries are added because they are not present in ``manager_calls``. The v0
    release does not append initial graph-generation usage, so its totals remain a
    lower bound unless a future summary explicitly sets ``accounting_complete``.
    """

    if not isinstance(summary, Mapping):
        return Usage(cost_known=False, complete=False)
    manager_calls = summary.get("manager_calls")
    subtask_costs = summary.get("subtask_costs")
    if not isinstance(manager_calls, list) or not isinstance(subtask_costs, Mapping):
        return Usage(cost_known=False, complete=False)

    entries: list[Mapping[str, Any]] = []
    for entry in manager_calls:
        if not isinstance(entry, Mapping):
            raise ProviderError("MACU summary contains a malformed manager call")
        entries.append(entry)
    for entry in subtask_costs.values():
        if not isinstance(entry, Mapping):
            raise ProviderError("MACU summary contains malformed subtask usage")
        if entry.get("agent_type") == "cua":
            entries.append(entry)

    input_tokens = 0
    output_tokens = 0
    complete = bool(entries)
    for entry in entries:
        raw_input = entry.get("input_tokens")
        raw_output = entry.get("output_tokens")
        if (
            not isinstance(raw_input, int)
            or isinstance(raw_input, bool)
            or raw_input < 0
            or not isinstance(raw_output, int)
            or isinstance(raw_output, bool)
            or raw_output < 0
        ):
            complete = False
            continue
        input_tokens += raw_input
        output_tokens += raw_output

    raw_cost = summary.get("total_cost_usd")
    reported_cost_present = bool(
        isinstance(raw_cost, (int, float))
        and not isinstance(raw_cost, bool)
        and math.isfinite(raw_cost)
        and raw_cost >= 0
    )
    cost_usd = float(raw_cost) if isinstance(raw_cost, (int, float)) else 0.0
    accounting_complete = summary.get("accounting_complete") is True
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd if reported_cost_present else 0.0,
        # The audited release does not add initial dependency-graph generation
        # usage to manager_calls/total_cost_usd. Preserve the reported lower bound
        # without presenting it as complete unless a future artifact says so.
        cost_known=reported_cost_present and accounting_complete,
        complete=complete and accounting_complete,
    )


def _read_json_object_limited(
    path: Path, *, max_bytes: int, label: str
) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size > max_bytes:
        raise ProviderError(f"{label} exceeds the {max_bytes}-byte artifact limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderError(f"could not parse {label}") from exc
    if not isinstance(value, Mapping):
        raise ProviderError(f"{label} must contain a JSON object")
    return value


class MACUUpstreamBackend:
    """Run the pinned released MACU manager/CUA runtime as one external tree.

    The checkout supplies the published prompts, graph generation and validation,
    ready-frontier scheduler, continuous replanning, VM cloning/file handoff logic,
    and evaluator integration. Scaffold Lab supplies only the outer task boundary,
    process isolation/limits, shared ledger projection, trace, and artifact pointer.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        result_dir: Path,
        osworld_root: Path,
        manager_provider: str,
        manager_model: str,
        cua_provider: str,
        python_executable: str = "python3",
        max_parallelism: int = 4,
        max_replans: int = 10,
        max_task_timeout_seconds: int = 5400,
        timeout_seconds: float = 6000.0,
        cua_args: Sequence[str] = (),
        pass_env: Sequence[str] = (),
        expected_checkout_revision: str = _MACU_RELEASE_REVISION,
        expected_python_sha256: Optional[str] = None,
        allow_dirty_checkout: bool = False,
        allow_sensitive_environment: bool = False,
        max_output_bytes: int = 16 * 1024 * 1024,
        max_artifact_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if os.name != "posix":
            raise ValueError("MACU adapter currently requires POSIX process groups")
        self.checkout = checkout.resolve()
        self.result_dir = result_dir.resolve()
        self.osworld_root = osworld_root.resolve()
        if not self.checkout.is_dir() or not (self.checkout / "run_macu.py").is_file():
            raise ValueError("MACU checkout must contain run_macu.py")
        if not self.osworld_root.is_dir():
            raise ValueError(
                f"MACU OSWorld root is not a directory: {self.osworld_root}"
            )
        for left, right, label in (
            (self.result_dir, self.checkout, "checkout"),
            (self.result_dir, self.osworld_root, "OSWorld root"),
        ):
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise ValueError(f"MACU result directory must be disjoint from {label}")
        if manager_provider not in {"openai", "anthropic", "huggingface", "google"}:
            raise ValueError("unsupported MACU manager_provider")
        if cua_provider not in {"qwen", "openai"}:
            raise ValueError("unsupported MACU cua_provider")
        for string_name, string_value in (
            ("manager_model", manager_model),
            ("python_executable", python_executable),
        ):
            if not isinstance(string_value, str) or not string_value:
                raise ValueError(f"{string_name} must be a non-empty string")
        for integer_name, integer_value in (
            ("max_parallelism", max_parallelism),
            ("max_replans", max_replans),
            ("max_task_timeout_seconds", max_task_timeout_seconds),
            ("max_output_bytes", max_output_bytes),
            ("max_artifact_bytes", max_artifact_bytes),
        ):
            if (
                not isinstance(integer_value, int)
                or isinstance(integer_value, bool)
                or integer_value < 1
            ):
                raise ValueError(f"{integer_name} must be a positive integer")
        if max_output_bytes < 1024 or max_artifact_bytes < 1024:
            raise ValueError("MACU output and artifact limits must be at least 1024")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        for label, values in (("cua_args", cua_args), ("pass_env", pass_env)):
            if any(
                not isinstance(value, str)
                or not value
                or "\x00" in value
                or (label == "pass_env" and "=" in value)
                for value in values
            ):
                raise ValueError(f"{label} entries must be non-empty strings")
        if pass_env and not allow_sensitive_environment:
            raise ValueError(
                "MACU and its CUA children can inspect passed environment values; "
                "set allow_sensitive_environment only with scoped credentials and "
                "an outer VM/network sandbox"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_checkout_revision) is None:
            raise ValueError(
                "expected_checkout_revision must be 40 lowercase hex chars"
            )
        if (
            expected_python_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", expected_python_sha256) is None
        ):
            raise ValueError("expected_python_sha256 must be 64 lowercase hex chars")
        if not isinstance(allow_dirty_checkout, bool):
            raise ValueError("allow_dirty_checkout must be a boolean")
        if not isinstance(allow_sensitive_environment, bool):
            raise ValueError("allow_sensitive_environment must be a boolean")

        git_checkout = _git_workspace_provenance(self.checkout)
        if not git_checkout.get("available"):
            raise ValueError("MACU checkout must be a readable Git checkout")
        if git_checkout.get("head") != expected_checkout_revision:
            raise ValueError(
                "MACU checkout revision mismatch: expected "
                f"{expected_checkout_revision}, observed {git_checkout.get('head')}"
            )
        if git_checkout.get("dirty") and not allow_dirty_checkout:
            raise ValueError("MACU checkout is dirty; use an immutable pinned checkout")

        self.manager_provider = manager_provider
        self.manager_model = manager_model
        self.cua_provider = cua_provider
        self.python_executable = python_executable
        self.max_parallelism = max_parallelism
        self.max_replans = max_replans
        self.max_task_timeout_seconds = max_task_timeout_seconds
        self.timeout_seconds = float(timeout_seconds)
        self.cua_args = tuple(cua_args)
        self.pass_env = tuple(pass_env)
        self.expected_checkout_revision = expected_checkout_revision
        self.expected_python_sha256 = expected_python_sha256
        self.allow_dirty_checkout = allow_dirty_checkout
        self.allow_sensitive_environment = allow_sensitive_environment
        self.max_output_bytes = max_output_bytes
        self.max_artifact_bytes = max_artifact_bytes
        self.git_checkout = git_checkout
        self._resolved_python: Optional[str] = None
        self._python_identity: Mapping[str, Any] = {"available": False}

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "macu-upstream-runtime",
            "checkout": str(self.checkout),
            "result_dir": str(self.result_dir),
            "osworld_root": str(self.osworld_root),
            "git_checkout": dict(self.git_checkout),
            "expected_checkout_revision": self.expected_checkout_revision,
            "runtime_source_identity_verified": (
                self.git_checkout.get("head") == self.expected_checkout_revision
                and not self.git_checkout.get("dirty")
            ),
            "dirty_checkout_allowed": self.allow_dirty_checkout,
            "manager_provider": self.manager_provider,
            "manager_model": self.manager_model,
            "cua_provider": self.cua_provider,
            "max_parallelism": self.max_parallelism,
            "max_replans": self.max_replans,
            "max_task_timeout_seconds": self.max_task_timeout_seconds,
            "timeout_seconds": self.timeout_seconds,
            "cua_args_count": len(self.cua_args),
            "cua_args_sha256": hashlib.sha256(
                "\0".join(self.cua_args).encode("utf-8")
            ).hexdigest(),
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "python_executable": self.python_executable,
            "expected_python_sha256": self.expected_python_sha256,
            "runtime_python": dict(self._python_identity),
            "one_backend_call_is_external_session_tree": True,
            "home": "ephemeral-per-call",
            "xdg_config_home": "ephemeral-per-call",
            "artifact_limits": {
                "stdout_bytes_per_stream": self.max_output_bytes,
                "json_bytes_per_artifact": self.max_artifact_bytes,
            },
            "audited_release": {
                "repository": "kohjingyu/multi-agent-computer-use",
                "revision": _MACU_RELEASE_REVISION,
                "paper": "arXiv:2606.01533v1",
            },
        }

    def _verify_checkout(self) -> None:
        observed = _git_workspace_provenance(self.checkout)
        if observed.get("head") != self.expected_checkout_revision:
            raise ProviderError("MACU checkout changed after backend initialization")
        if observed.get("dirty") and not self.allow_dirty_checkout:
            raise ProviderError("MACU checkout became dirty before or during the trial")

    def _task_id(self, request: ModelRequest) -> str:
        raw = request.metadata.get("task_id")
        source = raw if isinstance(raw, str) and raw else request.prompt
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-._")[:48]
        slug = slug or "task"
        digest = hashlib.sha256(
            f"{source}\0{request.prompt}".encode("utf-8")
        ).hexdigest()[:12]
        return f"scaffoldlab-{slug}-{digest}"

    def _artifacts(
        self, task_directory: Path
    ) -> tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], Usage]:
        summary = _read_json_object_limited(
            task_directory / "summary.json",
            max_bytes=self.max_artifact_bytes,
            label="MACU summary.json",
        )
        final_results = _read_json_object_limited(
            task_directory / "final_results.json",
            max_bytes=self.max_artifact_bytes,
            label="MACU final_results.json",
        )
        return summary, final_results, _macu_usage_from_summary(summary)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._verify_checkout()
        resolved_python, identity = _prepare_executable(
            self.python_executable,
            self.expected_python_sha256,
            "MACU Python",
        )
        self._resolved_python = resolved_python
        self._python_identity = identity
        task_id = self._task_id(request)
        task_directory = self.result_dir / task_id
        if task_directory.exists():
            raise ProviderError(
                f"MACU task result already exists; refusing to overwrite: {task_directory}"
            )
        self.result_dir.mkdir(parents=True, exist_ok=True)

        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="scaffoldlab-macu-") as temp_dir:
            temp_root = Path(temp_dir)
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            home.mkdir(mode=0o700)
            xdg_config.mkdir(mode=0o700)
            task_file = temp_root / "task.json"
            task_file.write_text(
                json.dumps(
                    [
                        {
                            "task_id": task_id,
                            "confirmed_task": prompt,
                            "website": "",
                            "no_initial_setup": True,
                        }
                    ],
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            task_file.chmod(0o600)
            environment = _process_environment(self.pass_env)
            environment["HOME"] = str(home)
            environment["XDG_CONFIG_HOME"] = str(xdg_config)
            command = [
                resolved_python,
                str(self.checkout / "run_macu.py"),
                str(task_file),
                "--task-id",
                task_id,
                "--result-dir",
                str(self.result_dir),
                "--osworld-root",
                str(self.osworld_root),
                "--manager-provider",
                self.manager_provider,
                "--manager-model",
                self.manager_model,
                "--cua-provider",
                self.cua_provider,
                "--max-parallelism",
                str(self.max_parallelism),
                "--max-replans",
                str(self.max_replans),
                "--max-task-timeout",
                str(self.max_task_timeout_seconds),
            ]
            if self.cua_args:
                command.extend(["--", *self.cua_args])
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.checkout),
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"MACU Python executable not found: {self.python_executable!r}"
                ) from exc
            captured = _CapturedOutput()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data=None,
                        max_output_bytes=self.max_output_bytes,
                        captured=captured,
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process_tree(process)
                summary, final_results, usage = self._artifacts(task_directory)
                raise ProviderError(
                    "MACU upstream trial timed out",
                    usage=usage,
                    raw={
                        "summary": summary,
                        "final_results": final_results,
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except asyncio.CancelledError as exc:
                await _terminate_process_tree(process)
                _, _, usage = self._artifacts(task_directory)
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except _ProcessOutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                summary, final_results, usage = self._artifacts(task_directory)
                raise ProviderError(
                    "MACU upstream trial exceeded its process-output limit",
                    usage=usage,
                    raw={
                        "summary": summary,
                        "final_results": final_results,
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise
            await _terminate_process_tree(process)

        summary, final_results, usage = self._artifacts(task_directory)
        raw = {
            "task_id": task_id,
            "result_directory": str(task_directory),
            "summary": summary,
            "final_results": final_results,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        try:
            self._verify_checkout()
        except ProviderError as exc:
            raise ProviderError(str(exc), usage=usage, raw=raw) from exc
        if process.returncode != 0:
            raise ProviderError(
                f"MACU upstream trial exited with status {process.returncode}",
                usage=usage,
                raw=raw,
            )
        if final_results is None:
            raise ProviderError(
                "MACU upstream trial produced no final_results.json",
                usage=usage,
                raw=raw,
            )
        status = final_results.get("status")
        if (
            status in {"timed_out", "error", "failed"}
            or final_results.get("timed_out") is True
        ):
            raise ProviderError(
                f"MACU upstream trial ended with status {status!r}",
                usage=usage,
                raw=raw,
            )
        answer = final_results.get("final_response")
        if not isinstance(answer, str) or not answer:
            raise ProviderError(
                "MACU final_results.json contained no final_response",
                usage=usage,
                raw=raw,
            )
        return ModelResponse(text=answer, usage=usage, raw=raw)


_RLM_RELEASE_REVISION = "72d6940142ddfb84ee6be573dc999a37e633e671"
_RLM_RESULT_MARKER = "__SCAFFOLDLAB_RLM_RESULT_V1__="
_RLM_PROVIDERS = {
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "anthropic",
    "azure_openai",
    "gemini",
}
_RLM_ENVIRONMENTS = {
    "local",
    "ipython",
    "docker",
    "modal",
    "prime",
    "daytona",
    "e2b",
}


def _json_mapping_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a JSON object")
    return copied


def _rlm_usage_from_summary(summary: Any) -> tuple[Usage, int]:
    """Project the upstream root ``UsageSummary`` as an observed lower bound.

    v0.1.3 reports tokens and calls per model. Cost is optional per model and the
    aggregate property sums whichever costs are present. Recursive child RLMs own a
    separate LMHandler and are not merged into the root summary, so neither tokens nor
    cost can be marked complete even when every root model entry reports cost.
    """

    if not isinstance(summary, Mapping):
        raise ProviderError("RLM upstream result omitted UsageSummary")
    raw_models = summary.get("model_usage_summaries")
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise ProviderError("RLM upstream UsageSummary has no model entries")

    input_tokens = 0
    output_tokens = 0
    total_calls = 0
    cost_usd = 0.0

    def partial_usage() -> Usage:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cost_known=False,
            complete=False,
        )

    for model, raw_usage in raw_models.items():
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(raw_usage, Mapping)
        ):
            raise ProviderError(
                "RLM upstream UsageSummary contains a malformed model entry",
                usage=partial_usage(),
            )
        counts: dict[str, int] = {}
        for field in ("total_calls", "total_input_tokens", "total_output_tokens"):
            value = raw_usage.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProviderError(
                    f"RLM upstream UsageSummary emitted invalid {field!r}",
                    usage=partial_usage(),
                )
            counts[field] = value
        total_calls += counts["total_calls"]
        input_tokens += counts["total_input_tokens"]
        output_tokens += counts["total_output_tokens"]
        if "total_cost" in raw_usage:
            raw_cost = raw_usage.get("total_cost")
            if (
                not isinstance(raw_cost, (int, float))
                or isinstance(raw_cost, bool)
                or not math.isfinite(raw_cost)
                or raw_cost < 0
            ):
                raise ProviderError(
                    "RLM upstream UsageSummary emitted invalid total_cost",
                    usage=partial_usage(),
                )
            cost_usd += float(raw_cost)

    return (
        Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cost_known=False,
            complete=False,
        ),
        total_calls,
    )


def _rlm_marker_result(data: bytes) -> Mapping[str, Any]:
    marker_payloads = [
        line[len(_RLM_RESULT_MARKER) :]
        for line in data.decode("utf-8", errors="replace").splitlines()
        if line.startswith(_RLM_RESULT_MARKER)
    ]
    if len(marker_payloads) != 1:
        raise ProviderError(
            "RLM upstream runner did not emit exactly one marker result"
        )
    try:
        result = json.loads(marker_payloads[0])
    except json.JSONDecodeError as exc:
        raise ProviderError("RLM upstream runner emitted invalid marker JSON") from exc
    if not isinstance(result, Mapping) or result.get("schema_version") != 1:
        raise ProviderError("RLM upstream runner emitted an unsupported result schema")
    return result


class RLMUpstreamBackend:
    """Run the exact pinned ``rlms`` library in a bounded external process.

    The checkout must be a clean Git tree at the requested revision. The default
    upstream REPL is Docker; selecting a non-isolated upstream environment does not
    turn the process boundary into a security sandbox or add SWE tool parity.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        provider: str,
        model: str,
        python_executable: str = "python3",
        environment: str = "docker",
        backend_kwargs: Optional[Mapping[str, Any]] = None,
        environment_kwargs: Optional[Mapping[str, Any]] = None,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget_usd: Optional[float] = None,
        max_timeout_seconds: Optional[float] = 1500.0,
        max_tokens: Optional[int] = None,
        max_errors: Optional[int] = None,
        max_concurrent_subcalls: int = 4,
        process_timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        expected_checkout_revision: str = _RLM_RELEASE_REVISION,
        expected_python_sha256: Optional[str] = None,
        allow_sensitive_environment: bool = False,
        max_input_bytes: int = 64 * 1024 * 1024,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "RLM upstream adapter currently requires POSIX process groups"
            )
        self.checkout = checkout.resolve()
        package_init = self.checkout / "rlm" / "__init__.py"
        if not self.checkout.is_dir() or not package_init.is_file():
            raise ValueError("RLM checkout must contain rlm/__init__.py")
        if provider not in _RLM_PROVIDERS:
            raise ValueError(f"unsupported RLM provider {provider!r}")
        if environment not in _RLM_ENVIRONMENTS:
            raise ValueError(f"unsupported RLM environment {environment!r}")
        for label, value in (
            ("model", model),
            ("python_executable", python_executable),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{label} must be a non-empty string")
        if backend_kwargs is not None and "model_name" in backend_kwargs:
            raise ValueError(
                "set model explicitly instead of backend_kwargs.model_name"
            )
        self.backend_kwargs = _json_mapping_copy(backend_kwargs or {}, "backend_kwargs")
        self.environment_kwargs = _json_mapping_copy(
            environment_kwargs or {}, "environment_kwargs"
        )
        for integer_label, integer_value, minimum in (
            ("max_depth", max_depth, 1),
            ("max_iterations", max_iterations, 1),
            ("max_concurrent_subcalls", max_concurrent_subcalls, 1),
            ("max_input_bytes", max_input_bytes, 1024),
            ("max_output_bytes", max_output_bytes, 1024),
        ):
            if (
                not isinstance(integer_value, int)
                or isinstance(integer_value, bool)
                or integer_value < minimum
            ):
                raise ValueError(f"{integer_label} must be an integer >= {minimum}")
        for optional_integer_label, optional_integer_value in (
            ("max_tokens", max_tokens),
            ("max_errors", max_errors),
        ):
            if optional_integer_value is not None and (
                not isinstance(optional_integer_value, int)
                or isinstance(optional_integer_value, bool)
                or optional_integer_value < 1
            ):
                raise ValueError(
                    f"{optional_integer_label} must be a positive integer or null"
                )
        for number_label, number_value, allow_zero in (
            ("max_budget_usd", max_budget_usd, True),
            ("max_timeout_seconds", max_timeout_seconds, False),
            ("process_timeout_seconds", process_timeout_seconds, False),
        ):
            if number_value is not None and (
                not isinstance(number_value, (int, float))
                or isinstance(number_value, bool)
                or not math.isfinite(number_value)
                or number_value < 0
                or (not allow_zero and number_value == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(
                    f"{number_label} must be {qualifier} and finite or null"
                )
        if process_timeout_seconds is None:
            raise ValueError("process_timeout_seconds cannot be null")
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        if len(set(pass_env)) != len(pass_env):
            raise ValueError("pass_env entries must be unique")
        if not isinstance(allow_sensitive_environment, bool):
            raise ValueError("allow_sensitive_environment must be a boolean")
        if pass_env and not allow_sensitive_environment:
            raise ValueError(
                "the upstream RLM and local REPL modes can inspect passed environment "
                "values; acknowledge scoped credentials and use an outer sandbox"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_checkout_revision) is None:
            raise ValueError(
                "expected_checkout_revision must be 40 lowercase hex chars"
            )
        if (
            expected_python_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", expected_python_sha256) is None
        ):
            raise ValueError("expected_python_sha256 must be 64 lowercase hex chars")

        git_checkout = _git_workspace_provenance(self.checkout)
        if not git_checkout.get("available"):
            raise ValueError("RLM checkout must be a readable Git checkout")
        if git_checkout.get("head") != expected_checkout_revision:
            raise ValueError(
                "RLM checkout revision mismatch: expected "
                f"{expected_checkout_revision}, observed {git_checkout.get('head')}"
            )
        if git_checkout.get("dirty"):
            raise ValueError("RLM checkout must be clean and immutable")

        runner = Path(__file__).with_name("upstream_rlm_runner.py").resolve()
        if not runner.is_file():
            raise ValueError(f"RLM upstream runner is missing: {runner}")
        self.provider = provider
        self.model = model
        self.python_executable = python_executable
        self.environment = environment
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_budget_usd = (
            float(max_budget_usd) if max_budget_usd is not None else None
        )
        self.max_timeout_seconds = (
            float(max_timeout_seconds) if max_timeout_seconds is not None else None
        )
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.process_timeout_seconds = float(process_timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.expected_checkout_revision = expected_checkout_revision
        self.expected_python_sha256 = expected_python_sha256
        self.allow_sensitive_environment = allow_sensitive_environment
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.git_checkout = git_checkout
        self.runner = runner
        self.runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
        self._python_identity: Mapping[str, Any] = {"available": False}
        self._observed_python_version: Any = None

    def provenance(self) -> Mapping[str, Any]:
        backend_json = json.dumps(self.backend_kwargs, sort_keys=True).encode("utf-8")
        environment_json = json.dumps(self.environment_kwargs, sort_keys=True).encode(
            "utf-8"
        )
        return {
            "provider": "rlms-upstream-runtime",
            "checkout": str(self.checkout),
            "git_checkout": dict(self.git_checkout),
            "expected_checkout_revision": self.expected_checkout_revision,
            "runtime_source_identity_verified": True,
            "rlm_provider": self.provider,
            "model": self.model,
            "environment": self.environment,
            "default_environment_is_docker": self.environment == "docker",
            "backend_kwargs_keys": sorted(self.backend_kwargs),
            "backend_kwargs_sha256": hashlib.sha256(backend_json).hexdigest(),
            "environment_kwargs_keys": sorted(self.environment_kwargs),
            "environment_kwargs_sha256": hashlib.sha256(environment_json).hexdigest(),
            "limits": {
                "max_depth": self.max_depth,
                "max_iterations": self.max_iterations,
                "max_budget_usd": self.max_budget_usd,
                "max_timeout_seconds": self.max_timeout_seconds,
                "max_tokens": self.max_tokens,
                "max_errors": self.max_errors,
                "max_concurrent_subcalls": self.max_concurrent_subcalls,
                "process_timeout_seconds": self.process_timeout_seconds,
                "max_input_bytes": self.max_input_bytes,
                "max_output_bytes_per_stream": self.max_output_bytes,
            },
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "python_executable": self.python_executable,
            "expected_python_sha256": self.expected_python_sha256,
            "runtime_python": dict(self._python_identity),
            "observed_python_version": self._observed_python_version,
            "runner": str(self.runner),
            "runner_sha256": self.runner_sha256,
            "json_stdin_protocol": 1,
            "one_backend_call_is_external_session_tree": True,
            "usage_scope": "root RLMChatCompletion.usage_summary lower bound",
            "accounting_caveat": (
                "v0.1.3 does not merge recursive child RLM UsageSummary values into "
                "the root LMHandler summary"
            ),
            "cost_policy": "reported costs are retained as an incomplete lower bound",
            "scaffoldlab_domain_tools_injected": False,
            "swe_tool_parity_claimed": False,
            "audited_release": {
                "package": "rlms",
                "version": "0.1.3",
                "repository": "alexzhang13/rlm",
                "revision": _RLM_RELEASE_REVISION,
                "requires_python": ">=3.11",
            },
        }

    def _verify_checkout(self) -> None:
        observed = _git_workspace_provenance(self.checkout)
        if observed.get("head") != self.expected_checkout_revision:
            raise ProviderError("RLM checkout changed after backend initialization")
        if observed.get("dirty"):
            raise ProviderError("RLM checkout became dirty before or during the trial")

    def _payload(self, request: ModelRequest) -> bytes:
        if request.tools or request.tool_results or request.continuation is not None:
            raise ProviderError(
                "RLM upstream owns its REPL; client tool continuation is unsupported"
            )
        root_prompt = request.metadata.get("root_prompt")
        if root_prompt is not None and not isinstance(root_prompt, str):
            raise ProviderError(
                "RLM upstream root_prompt metadata must be string or null"
            )
        if request.system:
            root_prompt = (
                f"{request.system}\n\n{root_prompt}" if root_prompt else request.system
            )
        payload = {
            "schema_version": 1,
            "checkout": str(self.checkout),
            "provider": self.provider,
            "model": self.model,
            "environment": self.environment,
            "backend_kwargs": self.backend_kwargs,
            "environment_kwargs": self.environment_kwargs,
            "context": request.prompt,
            "root_prompt": root_prompt,
            "limits": {
                "max_depth": self.max_depth,
                "max_iterations": self.max_iterations,
                "max_budget_usd": self.max_budget_usd,
                "max_timeout_seconds": self.max_timeout_seconds,
                "max_tokens": self.max_tokens,
                "max_errors": self.max_errors,
                "max_concurrent_subcalls": self.max_concurrent_subcalls,
            },
        }
        try:
            encoded = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "RLM upstream request is not JSON serializable"
            ) from exc
        if len(encoded) > self.max_input_bytes:
            raise ProviderError(
                f"RLM upstream JSON input exceeds {self.max_input_bytes} bytes"
            )
        return encoded

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._verify_checkout()
        input_data = self._payload(request)
        resolved_python, identity = _prepare_executable(
            self.python_executable,
            self.expected_python_sha256,
            "RLM Python",
        )
        self._python_identity = identity
        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-rlm-upstream-"
        ) as temp_dir:
            temp_root = Path(temp_dir)
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            process_tmp = temp_root / "tmp"
            for directory in (home, xdg_config, process_tmp):
                directory.mkdir(mode=0o700)
            process_environment = _process_environment(self.pass_env)
            process_environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "TMPDIR": str(process_tmp),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            command = [resolved_python, "-I", "-B", str(self.runner)]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(temp_root),
                    env=process_environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"RLM Python executable not found: {self.python_executable!r}"
                ) from exc
            captured = _CapturedOutput()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data=input_data,
                        max_output_bytes=self.max_output_bytes,
                        captured=captured,
                    ),
                    timeout=self.process_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process_tree(process)
                usage = Usage(cost_known=False, complete=False)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                raise ProviderError(
                    "RLM upstream process timed out",
                    usage=usage,
                    raw={
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except asyncio.CancelledError as exc:
                await _terminate_process_tree(process)
                usage = Usage(cost_known=False, complete=False)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except _ProcessOutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "RLM upstream process exceeded its output limit",
                    usage=Usage(cost_known=False, complete=False),
                    raw={
                        "stdout": bytes(captured.stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured.stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise
            await _terminate_process_tree(process)

        raw: dict[str, Any] = {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "environment": self.environment,
        }
        try:
            result = _rlm_marker_result(stdout)
        except ProviderError as exc:
            raise ProviderError(
                str(exc),
                usage=Usage(cost_known=False, complete=False),
                raw=raw,
            ) from exc
        raw["result"] = dict(result)
        usage_summary = result.get("usage_summary")
        if usage_summary is None:
            usage = Usage(cost_known=False, complete=False)
            underlying_calls = 0
        else:
            usage, underlying_calls = _rlm_usage_from_summary(usage_summary)
        raw["underlying_model_calls"] = underlying_calls
        python_version = result.get("python_version")
        if isinstance(python_version, list):
            self._observed_python_version = list(python_version)

        try:
            self._verify_checkout()
        except ProviderError as exc:
            raise ProviderError(str(exc), usage=usage, raw=raw) from exc
        if process.returncode != 0:
            raise ProviderError(
                f"RLM upstream process exited with status {process.returncode}",
                usage=usage,
                raw=raw,
            )
        if result.get("ok") is not True:
            raise ProviderError(
                "RLM upstream runner reported failure",
                usage=usage,
                raw=raw,
            )
        answer = result.get("response")
        if not isinstance(answer, str) or not answer:
            raise ProviderError(
                "RLM upstream result contained no response text",
                usage=usage,
                raw=raw,
            )
        latency = result.get("execution_time_seconds")
        provider_latency = (
            float(latency)
            if isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(latency)
            and latency >= 0
            else 0.0
        )
        return ModelResponse(
            text=answer,
            usage=usage,
            provider_latency_seconds=provider_latency,
            raw=raw,
        )
