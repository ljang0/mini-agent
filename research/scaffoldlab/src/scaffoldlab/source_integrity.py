"""Filesystem checks shared by source-executing adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def require_standalone_git_checkout(root: Path, *, label: str) -> None:
    """Require checkout-owned Git metadata without case aliases or indirection."""

    git_named_entries = [
        entry.name for entry in root.iterdir() if entry.name.casefold() == ".git"
    ]
    if git_named_entries != [".git"]:
        raise ValueError(
            f"{label} must contain exactly one case-sensitive .git metadata directory"
        )
    metadata = root / ".git"
    if metadata.is_symlink() or not metadata.is_dir():
        raise ValueError(
            f"{label} must be a standalone clone; linked worktree Git metadata is "
            "unsupported"
        )


def reject_linked_git_metadata(root: Path, *, label: str) -> None:
    """Allow no repository or a standalone repository, never Git indirection."""

    git_named_entries = [
        entry.name for entry in root.iterdir() if entry.name.casefold() == ".git"
    ]
    if not git_named_entries:
        return
    if git_named_entries != [".git"]:
        raise ValueError(f"{label} contains ambiguous case-variant Git metadata")
    metadata = root / ".git"
    if metadata.is_symlink() or not metadata.is_dir():
        raise ValueError(
            f"{label} cannot be a linked Git worktree; use a standalone clone or "
            "a directory without Git metadata"
        )


def reject_case_variant_git_metadata(
    root: Path, *, label: str, max_entries: int
) -> None:
    """Reject ambiguous aliases while pruning every exact ``.git`` entry."""

    entries = 0
    for _current_root, directory_names, file_names in os.walk(root, followlinks=False):
        for name in [*directory_names, *file_names]:
            entries += 1
            if entries > max_entries:
                raise ValueError(f"{label} Git-metadata scan exceeded its entry limit")
            if name.casefold() == ".git" and name != ".git":
                raise ValueError(f"{label} contains case-variant Git metadata")
        directory_names[:] = [name for name in directory_names if name != ".git"]


def copytree_ignore_git_metadata(_directory: str, names: Sequence[str]) -> set[str]:
    """Ignore administrative Git entries at every level of a copied tree."""

    return {name for name in names if name.casefold() == ".git"}
