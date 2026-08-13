"""Benchmark-neutral sandbox-runtime contracts.

A runtime owns exactly three things an execution sandbox differs in:
provisioning, running one command inside it, and moving files in or out.
Nothing here may import a benchmark module or encode benchmark-specific image
names, task identifiers, or paths; every such detail is passed in as
configuration by the environment that sits on top of a runtime.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..types import _require_finite_number, _require_positive_int


DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024


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
    """Run one host argv without a shell and return bounded combined output."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult: ...


class SandboxRuntime(Protocol):
    """One provisioned sandbox: run a command, move bytes, destroy it.

    ``write_file`` stages caller-supplied bytes *into* the sandbox under a
    runtime-chosen location and returns the path the sandbox itself sees, so a
    caller never has to know whether that is a host path, a copied container
    path, or a bind mount. ``remove_file`` releases what ``write_file``
    allocated.
    """

    @property
    def workdir(self) -> str:
        """The absolute directory commands run in, as the sandbox sees it."""

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, name: str, data: bytes) -> str: ...

    async def remove_file(self, path: str) -> None: ...

    async def close(self) -> None: ...

    def provenance(self) -> Mapping[str, Any]: ...

    def resource_identity(self) -> str: ...


def require_argv(value: Sequence[str], label: str) -> tuple[str, ...]:
    """Return ``value`` once it is a usable list of command arguments."""

    if (
        isinstance(value, (str, bytes))
        or not value
        or not all(
            isinstance(item, str) and item and "\x00" not in item for item in value
        )
    ):
        raise ValueError(f"{label} argv must contain non-empty strings")
    return tuple(value)


def require_ref(value: Any, message: str, *, no_dash: bool = False) -> str:
    """Return a non-empty, NUL-free string usable as a runtime argument."""

    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or (no_dash and value.startswith("-"))
    ):
        raise ValueError(message)
    return value


def require_workdir(value: Any) -> str:
    """Return ``value`` once it is an absolute in-sandbox directory."""

    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or value.strip() != value
    ):
        raise ValueError("container workdir must be an absolute path")
    return value


def require_staging_name(value: Any) -> str:
    """Return a single path component safe to stage inside any sandbox."""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
        or value.startswith("-")
    ):
        raise ValueError("staged file name must be one plain path component")
    return value


def require_runner(runner: Any) -> ProcessRunner:
    """Return ``runner`` once it exposes the process-runner contract."""

    if not callable(getattr(runner, "run", None)):
        raise ValueError("runner must expose run")
    return runner


def resolve_runner(runner: ProcessRunner | None) -> ProcessRunner:
    """Return the caller's runner, or the default local one."""

    if runner is None:
        from .local import LocalProcessRunner

        return LocalProcessRunner()
    return require_runner(runner)


def positive_number(value: Any, label: str) -> float:
    return _require_finite_number(value, label, exclusive_minimum=0)


def positive_int(value: Any, label: str) -> int:
    return _require_positive_int(value, label)


def failed(result: ProcessResult) -> bool:
    return result.timed_out or result.returncode != 0


def require_ok(
    result: ProcessResult,
    message: str,
    *,
    error: type[Exception] = RuntimeError,
    fallback: bool = False,
) -> ProcessResult:
    """Return ``result``, or raise ``message`` with the captured output."""

    if failed(result):
        detail = result.text()
        if fallback and not detail:
            detail = f"exit code {result.returncode}"
        raise error(f"{message}: {detail}")
    return result


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "ProcessResult",
    "ProcessRunner",
    "SandboxRuntime",
    "atomic_write",
    "failed",
    "positive_int",
    "positive_number",
    "require_argv",
    "require_ok",
    "require_ref",
    "require_runner",
    "require_staging_name",
    "require_workdir",
    "resolve_runner",
]
