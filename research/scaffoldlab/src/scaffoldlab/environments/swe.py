from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import ToolEnvironment, ToolExecution


_TRUSTED_PROCESS_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
_DEFAULT_MAX_PATCH_BYTES = 8 * 1024 * 1024


class SWEPatchPayload:
    """Private in-memory patch carrier externalized by ``MatrixRunner``.

    Its string representation is intentionally redacted because environment summaries
    are also recorded in traces before the runner has an opportunity to write the
    patch artifact.
    """

    __slots__ = ("_content",)

    def __init__(self, content: bytes) -> None:
        self._content = bytes(content)

    @property
    def content(self) -> bytes:
        return self._content

    def __repr__(self) -> str:
        return f"<SWE patch payload redacted ({len(self._content)} bytes)>"

    __str__ = __repr__


def _object_schema(
    properties: Mapping[str, Any], required: Sequence[str] = ()
) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _minimal_process_environment() -> dict[str, str]:
    """Keep benchmark subprocesses from inheriting provider credentials."""

    return {
        "PATH": _TRUSTED_PROCESS_PATH,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }


def _minimal_git_environment() -> dict[str, str]:
    """Keep temporary repositories independent from host Git configuration."""

    return {
        **_minimal_process_environment(),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


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


class PersistentBash:
    """A bounded persistent Bash session for provider-native shell schemas.

    This is an execution mechanism, not a security boundary.  Callers must place the
    process inside a container/VM when model-generated commands are untrusted.
    """

    def __init__(
        self,
        cwd: Path,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.environment = dict(environment or {})
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _start(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        env = {**_minimal_process_environment(), **self.environment}
        self._process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return self._process

    async def restart(self) -> None:
        await self.close()
        await self._start()

    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> Mapping[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ProtocolError("shell command must be a non-empty string")
        timeout = timeout_seconds or self.timeout_seconds
        async with self._lock:
            process = await self._start()
            assert process.stdin is not None
            assert process.stdout is not None
            stdout_reader = process.stdout
            marker = f"__SCAFFOLDLAB_{uuid.uuid4().hex}__"
            trailer = (
                "\n__scaffoldlab_status=$?\n"
                f"printf '\\n{marker}:%s\\n' \"$__scaffoldlab_status\"\n"
            )
            process.stdin.write((command + trailer).encode("utf-8"))
            await process.stdin.drain()
            buffer = bytearray()
            marker_bytes = ("\n" + marker + ":").encode("utf-8")

            async def read_until_marker() -> tuple[bytes, int]:
                while True:
                    chunk = await stdout_reader.read(4096)
                    if not chunk:
                        raise RuntimeError(
                            "persistent bash exited before status marker"
                        )
                    buffer.extend(chunk)
                    marker_at = buffer.find(marker_bytes)
                    if marker_at >= 0:
                        if marker_at > self.max_output_bytes:
                            raise RuntimeError(
                                "shell output exceeded max_output_bytes before completion"
                            )
                        suffix = buffer[marker_at + len(marker_bytes) :]
                        newline_at = suffix.find(b"\n")
                        if newline_at < 0:
                            continue
                        try:
                            status = int(suffix[:newline_at].decode("ascii"))
                        except ValueError as exc:
                            raise RuntimeError(
                                "invalid persistent bash status"
                            ) from exc
                        return bytes(buffer[:marker_at]), status
                    marker_overhead = len(marker_bytes) + 32
                    if len(buffer) > self.max_output_bytes + marker_overhead:
                        raise RuntimeError(
                            "shell output exceeded max_output_bytes before completion"
                        )

            try:
                raw_output, status = await asyncio.wait_for(
                    read_until_marker(), timeout=timeout
                )
            except BaseException:
                await self.close()
                raise
            output = raw_output.decode("utf-8", errors="replace")
            return {
                "stdout": output,
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": status},
            }

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        await _terminate_process_group(process)


class SWEEnvironment(ToolEnvironment):
    """Filesystem and process tools scoped to one explicit repository workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        allow_write: bool = False,
        allow_shell: bool = False,
        allow_native_shell: bool = False,
        command_allowlist: Sequence[str] = (),
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        export_patch: bool = False,
        max_patch_bytes: int = _DEFAULT_MAX_PATCH_BYTES,
        git_baseline_owned: bool = False,
        protocol: str = "auto",
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {workspace}")
        if protocol not in {"auto", "generic"}:
            raise ValueError("SWE protocol must be 'auto' or 'generic'")
        self.allow_write = bool(allow_write)
        self.allow_shell = bool(allow_shell)
        self.allow_native_shell = bool(allow_native_shell)
        if self.allow_native_shell and not self.allow_shell:
            raise ValueError("allow_native_shell requires allow_shell")
        if self.allow_native_shell and protocol != "auto":
            raise ValueError("allow_native_shell requires protocol='auto'")
        self.command_allowlist = tuple(command_allowlist)
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        if not isinstance(export_patch, bool):
            raise ValueError("export_patch must be a boolean")
        if (
            not isinstance(max_patch_bytes, int)
            or isinstance(max_patch_bytes, bool)
            or max_patch_bytes < 1
        ):
            raise ValueError("max_patch_bytes must be a positive integer")
        self.export_patch = export_patch
        self.max_patch_bytes = max_patch_bytes
        self.git_baseline_owned = bool(git_baseline_owned)
        self.protocol = protocol
        self._bash = PersistentBash(
            self.workspace,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        self._calls = 0
        self._writes = 0

    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        if self.protocol == "auto" and provider_family == "anthropic":
            tools = [
                ToolDefinition(
                    name="str_replace_based_edit_tool",
                    kind="anthropic_text_editor_20250728",
                    provider_options={"max_characters": self.max_output_bytes},
                )
            ]
            if self.allow_shell:
                if self.allow_native_shell:
                    tools.append(
                        ToolDefinition(name="bash", kind="anthropic_bash_20250124")
                    )
                else:
                    tools.append(self._run_command_definition())
            return tools

        generic = [
            ToolDefinition(
                name="list_files",
                description="List workspace files matching an optional glob.",
                input_schema=_object_schema(
                    {
                        "glob": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1},
                    }
                ),
            ),
            ToolDefinition(
                name="read_file",
                description="Read a UTF-8 file inside the workspace.",
                input_schema=_object_schema(
                    {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    ("path",),
                ),
            ),
            ToolDefinition(
                name="search_files",
                description="Search UTF-8 workspace files for a literal string.",
                input_schema=_object_schema(
                    {
                        "query": {"type": "string"},
                        "glob": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1},
                    },
                    ("query",),
                ),
            ),
            ToolDefinition(
                name="git_diff",
                description="Return the current Git diff and status.",
                input_schema=_object_schema({}),
            ),
        ]
        if self.allow_write:
            generic.append(
                ToolDefinition(
                    name="apply_patch",
                    description="Apply one unified diff inside the workspace.",
                    input_schema=_object_schema(
                        {"patch": {"type": "string"}}, ("patch",)
                    ),
                )
            )
        if self.allow_shell:
            if (
                self.protocol == "auto"
                and provider_family == "openai"
                and self.allow_native_shell
            ):
                generic.append(ToolDefinition(name="shell", kind="openai_shell_local"))
            else:
                generic.append(self._run_command_definition())
        return generic

    @staticmethod
    def _run_command_definition() -> ToolDefinition:
        return ToolDefinition(
            name="run_command",
            description=(
                "Run an allowlisted executable with an argv array in the "
                "workspace; no shell expansion is performed."
            ),
            input_schema=_object_schema(
                {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 0.001,
                    },
                },
                ("argv",),
            ),
        )

    def _path(self, raw: Any, *, must_exist: bool = False) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ProtocolError("path must be a non-empty string")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != self.workspace and not resolved.is_relative_to(self.workspace):
            raise ProtocolError("path escapes the SWE workspace")
        if must_exist and not resolved.exists():
            raise ProtocolError(f"path does not exist: {raw}")
        return resolved

    def _bounded_text(self, value: str) -> str:
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return value
        prefix = encoded[: self.max_output_bytes].decode("utf-8", errors="ignore")
        return prefix + "\n[output truncated by Scaffold Lab]"

    async def execute(self, call: ToolCall) -> ToolExecution:
        self._calls += 1
        try:
            if call.name == "list_files":
                return await self._list_files(call.arguments)
            if call.name == "read_file":
                return await self._read_file(call.arguments)
            if call.name == "search_files":
                return await self._search_files(call.arguments)
            if call.name == "git_diff":
                return await self._git_diff()
            if call.name == "apply_patch":
                return await self._apply_patch(call.arguments)
            if call.name == "run_command":
                return await self._run_command(call.arguments)
            if call.name == "shell":
                return await self._openai_shell(call.arguments)
            if call.name == "bash":
                return await self._anthropic_bash(call.arguments)
            if call.name == "str_replace_based_edit_tool":
                return await self._text_editor(call.arguments)
            raise ProtocolError(f"unknown SWE tool {call.name!r}")
        except (ProtocolError, ValueError, OSError, RuntimeError) as exc:
            return ToolExecution(output=f"{type(exc).__name__}: {exc}", is_error=True)

    async def _list_files(self, args: Mapping[str, Any]) -> ToolExecution:
        pattern = args.get("glob", "**/*")
        maximum = args.get("max_results", 500)
        if not isinstance(pattern, str) or not pattern:
            raise ProtocolError("glob must be a non-empty string")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ProtocolError("max_results must be a positive integer")
        maximum = min(maximum, 5000)

        def collect() -> list[str]:
            values = []
            for path in sorted(self.workspace.glob(pattern)):
                resolved = path.resolve(strict=False)
                if resolved != self.workspace and not resolved.is_relative_to(
                    self.workspace
                ):
                    continue
                if path.is_file():
                    values.append(str(path.relative_to(self.workspace)))
                    if len(values) >= maximum:
                        break
            return values

        values = await asyncio.to_thread(collect)
        return ToolExecution(output=self._bounded_text("\n".join(values)))

    async def _read_file(self, args: Mapping[str, Any]) -> ToolExecution:
        path = self._path(args.get("path"), must_exist=True)
        if not path.is_file():
            raise ProtocolError("read_file path must be a regular file")
        start = args.get("start_line", 1)
        end = args.get("end_line")
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise ProtocolError("start_line must be a positive integer")
        if end is not None and (
            not isinstance(end, int) or isinstance(end, bool) or end < start
        ):
            raise ProtocolError("end_line must be an integer >= start_line")

        def read() -> str:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[start - 1 : end]
            return "\n".join(
                f"{line_number}: {line}"
                for line_number, line in enumerate(selected, start)
            )

        return ToolExecution(output=self._bounded_text(await asyncio.to_thread(read)))

    async def _search_files(self, args: Mapping[str, Any]) -> ToolExecution:
        query = args.get("query")
        pattern = args.get("glob", "**/*")
        maximum = args.get("max_results", 200)
        if not isinstance(query, str) or not query:
            raise ProtocolError("query must be a non-empty string")
        if not isinstance(pattern, str) or not pattern:
            raise ProtocolError("glob must be a non-empty string")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ProtocolError("max_results must be a positive integer")
        maximum = min(maximum, 2000)

        def search() -> list[str]:
            results: list[str] = []
            for path in sorted(self.workspace.glob(pattern)):
                resolved = path.resolve(strict=False)
                if resolved != self.workspace and not resolved.is_relative_to(
                    self.workspace
                ):
                    continue
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for line_number, line in enumerate(text.splitlines(), 1):
                    if query_text in line:
                        results.append(
                            f"{path.relative_to(self.workspace)}:{line_number}:{line}"
                        )
                        if len(results) >= maximum:
                            return results
            return results

        query_text = query
        results = await asyncio.to_thread(search)
        return ToolExecution(output=self._bounded_text("\n".join(results)))

    async def _run_process(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        stdin: bytes | None = None,
    ) -> tuple[int | None, bytes, bytes, bool]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workspace,
            env=_minimal_process_environment(),
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin), timeout=timeout_seconds
            )
            timed_out = False
        except asyncio.TimeoutError:
            await _terminate_process_group(process)
            stdout, stderr = await process.communicate()
            timed_out = True
        except BaseException:
            await _terminate_process_group(process)
            raise
        return process.returncode, stdout, stderr, timed_out

    async def _run_bounded_process(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> tuple[int, bytes, bytes]:
        """Run a process while bounding captured output before it reaches memory."""

        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workspace,
            env=_minimal_git_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        async def read_bounded(
            stream: asyncio.StreamReader, limit: int, label: str
        ) -> bytes:
            captured = bytearray()
            while True:
                chunk = await stream.read(64 * 1024)
                if not chunk:
                    return bytes(captured)
                if len(captured) + len(chunk) > limit:
                    raise RuntimeError(
                        f"{label} exceeded its configured {limit}-byte limit"
                    )
                captured.extend(chunk)

        stdout_task = asyncio.create_task(
            read_bounded(process.stdout, max_stdout_bytes, "SWE patch")
        )
        stderr_task = asyncio.create_task(
            read_bounded(process.stderr, max_stderr_bytes, "git diff stderr")
        )
        wait_task = asyncio.create_task(process.wait())
        tasks = (stdout_task, stderr_task, wait_task)
        try:
            stdout, stderr, returncode = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_group(process)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise RuntimeError("git diff timed out") from exc
        except BaseException:
            await _terminate_process_group(process)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return int(returncode), bytes(stdout), bytes(stderr)

    async def _capture_patch(self) -> bytes:
        intent = await self._run_process(
            (
                "git",
                "add",
                "--intent-to-add",
                "--force",
                "--all",
                "--",
                ".",
            ),
            timeout_seconds=self.timeout_seconds,
        )
        if intent[0] != 0:
            diagnostic = intent[2][: self.max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"could not prepare untracked files for diff: {diagnostic}"
            )
        returncode, patch, stderr = await self._run_bounded_process(
            (
                "git",
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "HEAD",
                "--",
                ".",
            ),
            timeout_seconds=self.timeout_seconds,
            max_stdout_bytes=self.max_patch_bytes,
            max_stderr_bytes=self.max_output_bytes,
        )
        if returncode != 0:
            diagnostic = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"could not capture SWE patch: {diagnostic}")
        return patch

    async def _run_command(self, args: Mapping[str, Any]) -> ToolExecution:
        if not self.allow_shell:
            raise ProtocolError("command execution is disabled")
        argv = args.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ProtocolError("argv must be a non-empty list of strings")
        if "/" in argv[0] or "\\" in argv[0]:
            raise ProtocolError(
                "argv[0] must be a bare executable name resolved through the "
                "trusted process PATH"
            )
        executable = argv[0]
        if not self.command_allowlist:
            raise ProtocolError("no command executables are allowlisted")
        if executable not in self.command_allowlist:
            raise ProtocolError(f"executable is not allowlisted: {executable}")
        timeout = args.get("timeout_seconds", self.timeout_seconds)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ProtocolError("timeout_seconds must be positive")
        timeout = min(float(timeout), self.timeout_seconds)
        code, stdout, stderr, timed_out = await self._run_process(
            argv, timeout_seconds=timeout
        )
        value = {
            "stdout": stdout[: self.max_output_bytes].decode("utf-8", errors="replace"),
            "stderr": stderr[: self.max_output_bytes].decode("utf-8", errors="replace"),
            "exit_code": code,
            "timed_out": timed_out,
        }
        return ToolExecution(output=json.dumps(value, sort_keys=True))

    async def _git_diff(self) -> ToolExecution:
        status = await self._run_process(
            ("git", "status", "--short"), timeout_seconds=self.timeout_seconds
        )
        diff = await self._run_process(
            ("git", "diff", "--no-ext-diff", "--"),
            timeout_seconds=self.timeout_seconds,
        )
        if status[0] != 0 or diff[0] != 0:
            return ToolExecution(
                output="workspace is not a readable Git worktree", is_error=True
            )
        value = (
            "STATUS\n"
            + status[1].decode("utf-8", errors="replace")
            + "\nDIFF\n"
            + diff[1].decode("utf-8", errors="replace")
        )
        return ToolExecution(output=self._bounded_text(value))

    def _validate_patch_paths(self, patch: str) -> None:
        paths = re.findall(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE)
        if not paths:
            raise ProtocolError("patch contains no unified-diff file headers")
        for raw in paths:
            if raw == "/dev/null":
                continue
            normalized = re.sub(r"^[ab]/", "", raw)
            self._path(normalized)

    async def _apply_patch(self, args: Mapping[str, Any]) -> ToolExecution:
        if not self.allow_write:
            raise ProtocolError("workspace writes are disabled")
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch:
            raise ProtocolError("patch must be a non-empty string")
        if len(patch.encode("utf-8")) > self.max_output_bytes * 4:
            raise ProtocolError("patch exceeds the configured size limit")
        self._validate_patch_paths(patch)
        check = await self._run_process(
            ("git", "apply", "--check", "--whitespace=nowarn", "-"),
            timeout_seconds=self.timeout_seconds,
            stdin=patch.encode("utf-8"),
        )
        if check[0] != 0:
            return ToolExecution(
                output=self._bounded_text(check[2].decode("utf-8", errors="replace")),
                is_error=True,
            )
        applied = await self._run_process(
            ("git", "apply", "--whitespace=nowarn", "-"),
            timeout_seconds=self.timeout_seconds,
            stdin=patch.encode("utf-8"),
        )
        if applied[0] != 0:
            return ToolExecution(
                output=self._bounded_text(applied[2].decode("utf-8", errors="replace")),
                is_error=True,
            )
        self._writes += 1
        return ToolExecution(output="patch applied")

    async def _openai_shell(self, args: Mapping[str, Any]) -> ToolExecution:
        if not self.allow_shell:
            raise ProtocolError("shell execution is disabled")
        if not self.allow_native_shell:
            raise ProtocolError("native shell execution is disabled")
        commands = args.get("commands")
        if (
            not isinstance(commands, list)
            or not commands
            or not all(isinstance(command, str) and command for command in commands)
        ):
            raise ProtocolError("shell commands must be a non-empty list of strings")
        timeout_ms = args.get("timeout_ms")
        timeout = self.timeout_seconds
        if timeout_ms is not None:
            if (
                not isinstance(timeout_ms, int)
                or isinstance(timeout_ms, bool)
                or timeout_ms < 1
            ):
                raise ProtocolError("timeout_ms must be a positive integer")
            timeout = min(timeout, timeout_ms / 1000)
        max_output_length = args.get("max_output_length")
        if max_output_length is not None and (
            not isinstance(max_output_length, int)
            or isinstance(max_output_length, bool)
            or max_output_length < 1
        ):
            raise ProtocolError("max_output_length must be a positive integer")
        outputs = []
        for command in commands:
            result = await self._bash.run(command, timeout_seconds=timeout)
            outputs.append(result)
            if result["outcome"].get("exit_code") != 0:
                break
        native: dict[str, Any] = {"output": outputs}
        if max_output_length is not None:
            native["max_output_length"] = max_output_length
        return ToolExecution(
            output=json.dumps(outputs, sort_keys=True), native_output=native
        )

    async def _anthropic_bash(self, args: Mapping[str, Any]) -> ToolExecution:
        if not self.allow_shell:
            raise ProtocolError("bash execution is disabled")
        if not self.allow_native_shell:
            raise ProtocolError("native shell execution is disabled")
        if args.get("restart") is True:
            await self._bash.restart()
            return ToolExecution(output="bash session restarted")
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ProtocolError("bash command must be a non-empty string")
        result = await self._bash.run(command)
        return ToolExecution(output=self._bounded_text(result["stdout"]))

    async def _text_editor(self, args: Mapping[str, Any]) -> ToolExecution:
        command = args.get("command")
        path = self._path(args.get("path"), must_exist=command not in {"create"})
        if command == "view":
            if path.is_dir():
                names = [entry.name for entry in sorted(path.iterdir())]
                return ToolExecution(output=self._bounded_text("\n".join(names)))
            text = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="replace"
            )
            view_range = args.get("view_range")
            if view_range is not None:
                if (
                    not isinstance(view_range, list)
                    or len(view_range) != 2
                    or not all(isinstance(item, int) for item in view_range)
                ):
                    raise ProtocolError("view_range must contain two integers")
                start, end = view_range
                lines = text.splitlines()
                text = "\n".join(lines[max(0, start - 1) : None if end == -1 else end])
            return ToolExecution(output=self._bounded_text(text))
        if not self.allow_write:
            raise ProtocolError("workspace writes are disabled")
        if command == "create":
            if path.exists():
                raise ProtocolError("create refuses to overwrite an existing path")
            file_text = args.get("file_text")
            if not isinstance(file_text, str):
                raise ProtocolError("file_text must be a string")
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, file_text, encoding="utf-8")
        elif command == "str_replace":
            old = args.get("old_str")
            new = args.get("new_str")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise ProtocolError("old_str must be non-empty and new_str a string")
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            if text.count(old) != 1:
                raise ProtocolError("old_str must occur exactly once")
            await asyncio.to_thread(
                path.write_text, text.replace(old, new), encoding="utf-8"
            )
        elif command == "insert":
            line_number = args.get("insert_line")
            new = args.get("new_str")
            if (
                not isinstance(line_number, int)
                or isinstance(line_number, bool)
                or line_number < 0
            ):
                raise ProtocolError("insert_line must be a non-negative integer")
            if not isinstance(new, str):
                raise ProtocolError("new_str must be a string")
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            lines = text.splitlines(keepends=True)
            if line_number > len(lines):
                raise ProtocolError("insert_line exceeds file length")
            lines.insert(
                line_number, new + ("\n" if new and not new.endswith("\n") else "")
            )
            await asyncio.to_thread(path.write_text, "".join(lines), encoding="utf-8")
        else:
            raise ProtocolError(f"unsupported text-editor command {command!r}")
        self._writes += 1
        return ToolExecution(
            output=f"{command} succeeded for {path.relative_to(self.workspace)}"
        )

    async def summary(self) -> Mapping[str, Any]:
        status = await self._run_process(
            ("git", "status", "--short"), timeout_seconds=self.timeout_seconds
        )
        status_text = (
            status[1].decode("utf-8", errors="replace") if status[0] == 0 else ""
        )
        summary: dict[str, Any] = {
            "type": "swe",
            "workspace_sha256": hashlib.sha256(
                str(self.workspace).encode("utf-8")
            ).hexdigest(),
            "workspace_path_redacted": True,
            "allow_write": self.allow_write,
            "allow_shell": self.allow_shell,
            "allow_native_shell": self.allow_native_shell,
            "protocol": self.protocol,
            "git_baseline_owned": self.git_baseline_owned,
            "patch_export_enabled": self.export_patch,
            "max_patch_bytes": self.max_patch_bytes,
            "tool_calls": self._calls,
            "write_calls": self._writes,
            "git_status_sha256": hashlib.sha256(
                status_text.encode("utf-8")
            ).hexdigest(),
            "git_status_chars": len(status_text),
        }
        if self.export_patch:
            summary["patch_artifact"] = SWEPatchPayload(await self._capture_patch())
        return summary

    async def close(self) -> None:
        await self._bash.close()
