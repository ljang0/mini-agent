"""The single writer for upstream gym-anything's QEMU runtime configuration.

Upstream reads ``GYM_ANYTHING_QEMU_CACHE`` and ``GYM_ANYTHING_QEMU_WORK_DIR``
from the process environment itself, so the harness cannot hand it a config
object. What the harness *can* guarantee is that exactly one function writes
those variables, under one validation path: ``run``/``eval`` and ``doctor``
previously exported them with different checks, so a directory that ``doctor``
accepted could be rejected once a real run started.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..types import _require_no_symlink


def configure_qemu_runtime(
    qemu_cache: Path | None, work: Path | None = None
) -> None:
    """Validate and export the QEMU cache and work directories.

    ``qemu_cache`` is the operator's ``--qemu-cache``; ``work`` is the run's
    scratch root, omitted by callers (such as ``doctor``) that only inspect the
    cache and never launch a machine.
    """

    if qemu_cache is not None:
        expanded = _require_no_symlink(qemu_cache.expanduser(), "--qemu-cache")
        resolved_cache = expanded.resolve()
        if resolved_cache.exists() and not resolved_cache.is_dir():
            raise ValueError("--qemu-cache must be a directory")
        os.environ["GYM_ANYTHING_QEMU_CACHE"] = str(resolved_cache)
    if work is not None:
        qemu_work = _require_no_symlink(
            work / "cua-speed-run-qemu", "cua-speed-run QEMU work directory"
        )
        os.environ["GYM_ANYTHING_QEMU_WORK_DIR"] = str(qemu_work.resolve())


__all__ = ["configure_qemu_runtime"]
