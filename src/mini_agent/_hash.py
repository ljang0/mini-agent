"""Stable content hashing for files that must not change while being read."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_file_sha256(path: Path, *, label: str) -> str:
    """Hash a regular file, failing if it mutates while being read."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while hashing: {path}")
    return digest.hexdigest()


__all__ = ["stable_file_sha256"]
