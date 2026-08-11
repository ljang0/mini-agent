"""Optional launch boundary for exact external reference runtimes.

The default package never imports the archived Scaffold Lab harness. Source
checkouts may install or expose that audited runtime explicitly when replaying a
preserved reference. New references should use literal pinned upstream argv here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


class ReferenceRuntimeUnavailable(RuntimeError):
    """Raised when an explicitly selected optional reference is not installed."""


def preserved_scaffold_main(arguments: Sequence[str]) -> int:
    try:
        from scaffoldlab.cli import main  # type: ignore[import]
    except ImportError as exc:
        archive = Path(__file__).resolve().parents[3] / "research" / "scaffoldlab" / "src"
        if not (archive / "scaffoldlab" / "cli.py").is_file():
            raise ReferenceRuntimeUnavailable(
                "the archived Scaffold Lab reference runtime is not installed; "
                "use the preserved repository tag or source checkout"
            ) from exc
        sys.path.insert(0, str(archive))
        try:
            from scaffoldlab.cli import main  # type: ignore[import]
        finally:
            sys.path.remove(str(archive))
    return main(tuple(arguments))


__all__ = ["ReferenceRuntimeUnavailable", "preserved_scaffold_main"]
