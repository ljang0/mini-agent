"""Shared Git-checkout inspection for pinned benchmark sources."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Callable

from .base import BenchmarkTask

EXECUTABLE_SOURCE_SUFFIXES = frozenset(
    {
        ".bash",
        ".bat",
        ".cmd",
        ".dylib",
        ".dll",
        ".fish",
        ".pth",
        ".py",
        ".pyc",
        ".pyd",
        ".pyi",
        ".pyo",
        ".pyw",
        ".ps1",
        ".sh",
        ".so",
        ".zsh",
    }
)
SCRIPT_DIRECTORIES = frozenset({"bin", "script", "scripts", "scripts_evaluation"})


def git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "git checkout inspection failed")
    return result.stdout.strip()


def reject_untracked_execution_files(
    checkout: Path,
    *,
    label: str,
    exempt: Callable[[Path, os.stat_result], bool] | None = None,
    run_git: Callable[..., str] = git,
) -> None:
    """Reject untracked files that can shadow or alter benchmark execution."""

    raw = run_git(checkout, "ls-files", "--others", "-z")
    entries = raw.split("\x00") if raw else []
    if entries and entries[-1] == "":
        entries.pop()
    dangerous: list[str] = []
    for entry in entries:
        relative = Path(entry)
        if not entry or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} checkout returned an invalid untracked path")
        candidate = checkout / relative
        try:
            status = candidate.lstat()
        except OSError as exc:
            raise ValueError(
                f"{label} untracked files changed during checkout inspection"
            ) from exc
        if exempt is not None and exempt(relative, status):
            continue
        executable = bool(status.st_mode & 0o111)
        script_directory = any(
            part.casefold() in SCRIPT_DIRECTORIES for part in relative.parts[:-1]
        )
        shebang = False
        if stat.S_ISREG(status.st_mode):
            try:
                with candidate.open("rb") as stream:
                    shebang = stream.read(2) == b"#!"
            except OSError as exc:
                raise ValueError(
                    f"{label} untracked files changed during checkout inspection"
                ) from exc
        if (
            executable
            or relative.suffix.casefold() in EXECUTABLE_SOURCE_SUFFIXES
            or script_directory
            or shebang
        ):
            dangerous.append(relative.as_posix())
    if run_git(checkout, "ls-files", "--others", "-z") != raw:
        raise ValueError(
            f"{label} untracked files changed during checkout inspection"
        )
    if dangerous:
        raise ValueError(
            f"{label} checkout contains an untracked executable or source file: "
            f"{dangerous[0]!r}"
        )


def task_string(task: BenchmarkTask, name: str, *, label: str) -> str:
    value = task.data.get(name)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} task {name} must be a non-empty string")
    return value


__all__ = [
    "EXECUTABLE_SOURCE_SUFFIXES",
    "SCRIPT_DIRECTORIES",
    "git",
    "reject_untracked_execution_files",
    "task_string",
]
