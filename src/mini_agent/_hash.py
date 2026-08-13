"""Content identity for files, trees, and the harness itself.

Every artifact identity the harness records is computed here: a file or tree is
hashed only while its inode identity provably does not move, and the recorded
shape is canonical JSON so two runs of the same bytes produce the same digest.
Nothing in this module is benchmark-specific — ``cli``, ``grading``, the
environments, and the benchmark adapters are all callers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from functools import cache
from pathlib import Path
from typing import Any, Mapping

from .types import _require_mapping, strict_json_loads

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def canonical_bytes(value: Any) -> bytes:
    """Encode one artifact-identity value as deterministic UTF-8 JSON bytes."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    """Encode a value as canonical JSON *text*, keeping non-ASCII characters.

    Spec fingerprints hash this rendering, so its bytes are frozen: unlike
    :func:`canonical_bytes` it leaves non-ASCII characters unescaped.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_digest(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_bytes` for ``value``."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stat_key(info: os.stat_result, *, size: bool = True) -> tuple[int, ...]:
    """Return the inode identity fields that must not move while hashing."""

    key = (info.st_dev, info.st_ino, info.st_mtime_ns)
    return key + (info.st_size,) if size else key


ImageStatKey = tuple[int, int, int, int, int, int]


def image_stat_key(info: os.stat_result) -> ImageStatKey:
    """Detect a file being replaced, rewritten, or re-permissioned underneath us.

    Stricter than :func:`stat_key`: machine images are cached across a run, so
    a mode or ctime change must also invalidate the cached identity.
    """

    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash_unmoved_file(path: Path, label: str) -> tuple[str, os.stat_result]:
    """Hash one open regular file, proving its inode identity did not move."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    after = path.stat()
    if stat_key(before) != stat_key(after):
        raise RuntimeError(f"{label} changed while hashing: {path}")
    return digest.hexdigest(), after


def stable_file_sha256(path: Path, *, label: str) -> str:
    """Hash a regular file as given, failing if it mutates while being read."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return _hash_unmoved_file(path, label)[0]


def immutable_file_identity(path: Path, *, label: str = "asset") -> Mapping[str, Any]:
    """Hash one regular file while proving it did not change underneath us."""

    if not isinstance(path, Path):
        raise ValueError(f"{label} path must be a Path")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    digest, after = _hash_unmoved_file(resolved, label)
    return {
        "path": str(resolved),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def immutable_tree_identity(
    path: Path, *, label: str = "asset tree"
) -> Mapping[str, Any]:
    """Hash a directory tree without following or accepting symbolic links."""

    if not isinstance(path, Path):
        raise ValueError(f"{label} path must be a Path")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {expanded}")
    root = expanded.resolve()
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory: {root}")
    before = root.stat()
    entries = sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )
    observed_paths = tuple(item.relative_to(root).as_posix() for item in entries)
    files: list[Mapping[str, Any]] = []
    total = 0
    for candidate in entries:
        if candidate.is_symlink():
            raise ValueError(f"{label} contains a symlink: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"{label} contains a non-regular file: {candidate}")
        identity = immutable_file_identity(candidate, label=label)
        total += int(identity["size_bytes"])
        files.append(identity_entry(identity, candidate.relative_to(root).as_posix()))
    after_paths = tuple(
        sorted(item.relative_to(root).as_posix() for item in root.rglob("*"))
    )
    after = root.stat()
    if (
        stat_key(before, size=False) != stat_key(after, size=False)
        or observed_paths != after_paths
    ):
        raise RuntimeError(f"{label} changed while hashing: {root}")
    return {
        "path": str(root),
        "file_count": len(files),
        "size_bytes": total,
        "sha256": canonical_digest(files),
    }


def machine_image_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    """Hash a machine image and validate its optional adjacent provenance."""

    identity = dict(immutable_file_identity(path, label=label))
    sidecar = Path(identity["path"] + ".provenance.json")
    if not sidecar.exists() and not sidecar.is_symlink():
        return identity
    sidecar_identity = dict(
        immutable_file_identity(sidecar, label=f"{label} provenance")
    )
    content = sidecar.read_bytes()
    if hashlib.sha256(content).hexdigest() != sidecar_identity["sha256"]:
        raise RuntimeError(f"{label} provenance changed while reading: {sidecar}")
    try:
        provenance = strict_json_loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} provenance JSON: {sidecar}") from exc
    _require_mapping(provenance, f"{label} provenance")
    if provenance.get("final_image_sha256") != identity["sha256"]:
        raise ValueError(f"{label} provenance does not match the image: {sidecar}")
    identity["provenance"] = sidecar_identity
    identity["provenance_schema"] = provenance.get("schema")
    return identity


def identity_entry(identity: Mapping[str, Any], relative: str) -> Mapping[str, Any]:
    """Describe one hashed file by its path relative to the hashed root."""

    return {
        "path": relative,
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def harness_identity() -> Mapping[str, Any]:
    """Return a location-independent identity for the executing agent harness."""

    source_file_count, source_sha256 = _harness_source_identity()
    packages: dict[str, str | None] = {}
    for name in (
        "mini-agent",
        "httpx",
        "playwright",
        "huggingface-hub",
        "pyjnius",
        "tokenizers",
        "swebench",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "mini-agent-harness-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
        "source_file_count": source_file_count,
        "source_sha256": source_sha256,
    }


@cache
def _harness_source_identity() -> tuple[int, str]:
    """Hash every installed package source file, at most once per process.

    The three artifact writers that record the harness (``run``, evaluation
    manifests, and ``grade``) ask for the same immutable installed tree, so the
    walk-and-hash is memoized; each caller still receives its own mapping.
    """

    package = Path(__file__).resolve().parent
    files: list[Mapping[str, Any]] = []
    candidates = sorted(
        (
            path
            for path in package.rglob("*")
            if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
        ),
        key=lambda path: path.relative_to(package).as_posix(),
    )
    for path in candidates:
        identity = immutable_file_identity(path, label="harness source")
        files.append(identity_entry(identity, path.relative_to(package).as_posix()))
    return len(files), canonical_digest(files)


__all__ = [
    "ImageStatKey",
    "canonical_bytes",
    "canonical_digest",
    "canonical_text",
    "harness_identity",
    "identity_entry",
    "image_stat_key",
    "immutable_file_identity",
    "immutable_tree_identity",
    "machine_image_identity",
    "stable_file_sha256",
    "stat_key",
]
