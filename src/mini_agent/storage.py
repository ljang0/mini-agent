"""Durable-evidence filesystem layout and the artifact commit primitives.

Artifacts become evidence only once they are durable, so every writer in
the harness goes through :func:`atomic_bytes`: write a private temporary,
fsync it, rename it into place, then fsync the directories that now name
it. :func:`read_committed_result` is the matching reader, admitting a
result only when its terminal commit marker matches its bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .types import strict_json_loads


@dataclass(frozen=True)
class StorageLayout:
    """Separate durable run data from latency-sensitive scratch data.

    ``root`` may live on durable network storage. ``scratch`` should live on a
    local filesystem when VM overlays or container layers are in use.
    """

    root: Path
    scratch: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not isinstance(self.scratch, Path):
            raise ValueError("storage root and scratch must be Paths")
        root = self.root.expanduser().resolve()
        scratch = self.scratch.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "scratch", scratch)
        protected = (root / "assets", root / "cache", root / "runs")
        if scratch == root or any(
            scratch == path or path in scratch.parents for path in protected
        ):
            raise ValueError(
                "scratch must not be the durable root or live under assets, cache, "
                "or runs"
            )

    @classmethod
    def resolve(
        cls,
        root: Path | None = None,
        scratch: Path | None = None,
    ) -> "StorageLayout":
        durable = root or _path_from_env("MINI_AGENT_HOME")
        if durable is None:
            durable = Path.home() / ".local" / "share" / "mini-agent"
        temporary = scratch or _path_from_env("MINI_AGENT_SCRATCH")
        if temporary is None:
            temporary = durable / "work"
        return cls(durable, temporary)

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def run(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _SAFE_COMPONENT.fullmatch(run_id):
            raise ValueError("run_id must be one safe path component")
        return self.runs / run_id

    def work(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _SAFE_COMPONENT.fullmatch(run_id):
            raise ValueError("run_id must be one safe path component")
        work = self.scratch / run_id
        durable_run = self.runs / run_id
        if _paths_overlap(work, durable_run):
            raise ValueError("scratch work and durable run paths must not overlap")
        return work

    def ensure(self) -> None:
        for path in (self.assets, self.root / "cache", self.runs, self.scratch):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def free_bytes(self, *, scratch: bool = False) -> int:
        path = self.scratch if scratch else self.root
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        return shutil.disk_usage(existing).free


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def atomic_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_bytes(path, content.encode())


def atomic_bytes(path: Path, content: bytes) -> None:
    missing: list[Path] = []
    ancestor = path.parent
    while not ancestor.exists():
        missing.append(ancestor)
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for created in reversed(missing):
        created.chmod(0o700)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        sync_directory(path.parent)
        for created in missing:
            sync_directory(created.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    """Persist a directory entry before it becomes crash-recovery evidence."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_committed_result(directory: Path, task_id: str) -> Mapping[str, Any]:
    """Read one result only when its terminal commit marker matches its bytes."""

    result_path = directory / "result.json"
    completion_path = directory / "completed.json"
    if result_path.is_symlink() or completion_path.is_symlink():
        raise ValueError(f"result artifacts must not be symlinks: {directory}")
    if not result_path.is_file() or not completion_path.is_file():
        raise ValueError(f"incomplete result artifact for {task_id!r}")
    result_bytes = result_path.read_bytes()
    value = json_object_load(result_bytes, result_path)
    completion = json_object_load(completion_path.read_bytes(), completion_path)
    if (
        value.get("task_id") != task_id
        or value.get("status") not in {"completed", "failed", "blocked"}
        or completion.get("task_id") != task_id
        or completion.get("result_sha256")
        != hashlib.sha256(result_bytes).hexdigest()
    ):
        raise ValueError(f"invalid committed result artifact for {task_id!r}")
    return value


def read_json_object(path: Path) -> Mapping[str, Any]:
    """Read one JSON artifact that must decode to an object."""

    return json_object_load(path.read_bytes(), path)


def json_object_load(raw: bytes, path: Path) -> Mapping[str, Any]:
    value = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


__all__ = [
    "StorageLayout",
    "atomic_bytes",
    "atomic_json",
    "json_object_load",
    "read_committed_result",
    "read_json_object",
    "sync_directory",
]
