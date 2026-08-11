from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Sequence

from scaffoldlab.environments.base import ToolExecution

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import BaseEnvironment


MINI_SWE_AGENT_REVISION = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
SWE_AGENT_REVISION = "3ea751c087f32b16e039a2233dd6eefecef325d5"


class BashEnvironment(BaseEnvironment):
    """One stateless bash action over a persistent, optionally copied workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
        owned_root: Path | None = None,
        source_workspace: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {workspace}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self._owned_root = owned_root
        self.source_workspace = (source_workspace or self.workspace).resolve()

    @classmethod
    async def isolated(
        cls,
        source: Path,
        *,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
    ) -> "BashEnvironment":
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"SWE workspace is not a directory: {source}")
        root = Path(tempfile.mkdtemp(prefix="mini-agent-swe-"))
        workspace = root / "workspace"
        try:
            await asyncio.to_thread(
                shutil.copytree,
                source,
                workspace,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git/worktrees"),
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, root, True)
            raise
        return cls(
            workspace,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            owned_root=root,
            source_workspace=source,
        )

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

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "bash":
            raise ProtocolError(f"unsupported SWE tool {action.name!r}")
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ProtocolError("bash command must be a non-empty string")
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "HOME": str(self.workspace),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-lc",
            command,
            cwd=self.workspace,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except BaseException:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            raise
        combined = stdout + stderr
        truncated = len(combined) > self.max_output_bytes
        combined = combined[: self.max_output_bytes]
        output = combined.decode("utf-8", errors="replace")
        if truncated:
            output += "\n[output truncated]"
        return ToolExecution(
            output=output,
            is_error=process.returncode != 0 or truncated,
            metadata={
                "exit_code": process.returncode,
                "output_truncated": truncated,
            },
            native_output={
                "stdout": stdout[: self.max_output_bytes].decode(
                    "utf-8", errors="replace"
                ),
                "stderr": stderr[: self.max_output_bytes].decode(
                    "utf-8", errors="replace"
                ),
                "outcome": {"type": "exit", "exit_code": process.returncode},
            },
        )

    async def close(self) -> None:
        if self._owned_root is not None:
            root, self._owned_root = self._owned_root, None
            await asyncio.to_thread(shutil.rmtree, root, True)

    def provenance(self) -> dict[str, object]:
        return {
            "application": "swe",
            "tools": ["bash"],
            "workspace_isolated": self._owned_root is not None,
            "source_workspace": str(self.source_workspace),
            "mini_swe_agent_revision": MINI_SWE_AGENT_REVISION,
            "swe_agent_reference_revision": SWE_AGENT_REVISION,
        }
