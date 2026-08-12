"""Filesystem layout for durable evidence and disposable runtime state."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


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


__all__ = ["StorageLayout"]
