from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..environments.cua import CUA_SPEED_RUN_REVISION
from ..environments.swe import LocalProcessRunner
from ..integrations.cua_speed_run import (
    build_agent_argv,
    build_cua_speed_run_argv,
)


@dataclass(frozen=True)
class ExternalProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


async def _run_external(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> ExternalProcessResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive")
    result = await LocalProcessRunner().run(
        tuple(argv),
        cwd=cwd,
        environment=environment,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    return ExternalProcessResult(
        argv=tuple(argv),
        returncode=result.returncode,
        stdout=result.text(),
        stderr="",
        timed_out=result.timed_out,
        output_truncated=result.truncated,
    )


async def verify_cua_speed_run_checkout(
    source_root: Path,
    *,
    git_executable: str = "git",
) -> str:
    """Fail closed unless a checkout is exactly the approved source revision."""

    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"cua-speed-run source root does not exist: {root}")
    result = await _run_external(
        (git_executable, "-C", str(root), "rev-parse", "HEAD"),
        cwd=None,
        environment=None,
        timeout_seconds=30,
        max_output_bytes=4096,
    )
    if result.returncode != 0 or result.timed_out:
        raise RuntimeError(f"could not identify cua-speed-run checkout: {result.stderr}")
    revision = result.stdout.strip()
    if revision != CUA_SPEED_RUN_REVISION:
        raise ValueError(
            f"cua-speed-run revision mismatch: expected {CUA_SPEED_RUN_REVISION}, got {revision}"
        )
    dirty = await _run_external(
        (git_executable, "-C", str(root), "status", "--porcelain=v1"),
        cwd=None,
        environment=None,
        timeout_seconds=30,
        max_output_bytes=64 * 1024,
    )
    if dirty.returncode != 0 or dirty.timed_out:
        raise RuntimeError("could not inspect cua-speed-run checkout state")
    if dirty.stdout.strip():
        raise ValueError("cua-speed-run checkout must be clean")
    return revision


def resolve_cua_speed_run_executable(
    source_root: Path, executable: str | Path
) -> Path:
    root = source_root.expanduser().resolve()
    raw = Path(executable).expanduser()
    if raw.is_absolute() or raw.parent != Path("."):
        resolved = raw.resolve()
    else:
        located = shutil.which(str(executable))
        if located is None:
            raise ValueError(f"cua-speed-run executable was not found: {executable}")
        resolved = Path(located).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"cua-speed-run executable is not executable: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "cua-speed-run executable must be inside the verified checkout"
        ) from exc
    return resolved


def _python_from_console_script(source_root: Path, console_script: Path) -> Path:
    """Resolve the checkout-local interpreter without trusting script contents."""

    root = source_root.expanduser().resolve()
    try:
        first_line = console_script.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError) as exc:
        raise ValueError("cua-speed-run console script has no readable shebang") from exc
    if not first_line.startswith("#!"):
        raise ValueError("cua-speed-run console script requires an absolute shebang")
    interpreter = Path(first_line[2:].strip()).expanduser()
    if not interpreter.is_absolute():
        raise ValueError("cua-speed-run console script requires an absolute shebang")
    # Preserve the checkout-local venv path instead of resolving its ordinary
    # ``python -> /usr/bin/python`` symlink out of the checkout. The source code
    # is pinned separately through the explicit PYTHONPATH below.
    interpreter = interpreter.parent.resolve() / interpreter.name
    try:
        interpreter.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "cua-speed-run console script must use a checkout-local interpreter"
        ) from exc
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"cua-speed-run Python interpreter is unavailable: {interpreter}")
    return interpreter


async def run_agent_plane(
    *,
    python_executable: str | Path,
    agent_script: Path,
    env_url: str,
    task_description: str,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 900,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> ExternalProcessResult:
    """Run the exact ``agent.py <env_url> <task>`` contract without a shell.

    The caller owns gateway arm/status/verifier operations. This function never
    accepts a control token and therefore cannot expose the hidden verifier.
    """

    argv = build_agent_argv(
        python_executable,
        agent_script,
        env_url,
        task_description,
    )
    return await _run_external(
        argv,
        cwd=agent_script.expanduser().resolve().parent,
        environment=environment,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )


async def run_cua_speed_run_reference(
    *,
    source_root: Path,
    executable: str | Path,
    submission: Path,
    benchmark: Path,
    output_root: Path,
    task_ids: Sequence[str] = (),
    remote: bool = False,
    agent_mode: str | None = None,
    agents_per_evaluation: int = 1,
    parallel_evaluations: int = 1,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 4 * 60 * 60,
) -> ExternalProcessResult:
    """Execute the exact pinned upstream evaluator as an external reference."""

    await verify_cua_speed_run_checkout(source_root)
    root = source_root.expanduser().resolve()
    pinned_executable = resolve_cua_speed_run_executable(root, executable)
    interpreter = _python_from_console_script(root, pinned_executable)
    source_cli = root / "src" / "cua_speedrun" / "cli.py"
    if not source_cli.is_file():
        raise ValueError(f"pinned cua-speed-run CLI source is missing: {source_cli}")
    upstream = build_cua_speed_run_argv(
        pinned_executable,
        submission=submission,
        benchmark=benchmark,
        output_root=output_root,
        task_ids=task_ids,
        remote=remote,
        agent_mode=agent_mode,
        agents_per_evaluation=agents_per_evaluation,
        parallel_evaluations=parallel_evaluations,
    )
    argv = (str(interpreter), "-m", "cua_speedrun.cli", *upstream[1:])
    selected_environment = dict(os.environ if environment is None else environment)
    source_paths = [
        str(root / "src"),
        str(root / "third_party" / "gym-anything" / "src"),
    ]
    if selected_environment.get("PYTHONPATH"):
        source_paths.append(selected_environment["PYTHONPATH"])
    selected_environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    return await _run_external(
        argv,
        cwd=root,
        environment=selected_environment,
        timeout_seconds=timeout_seconds,
        max_output_bytes=16 * 1024 * 1024,
    )


__all__ = [
    "ExternalProcessResult",
    "resolve_cua_speed_run_executable",
    "run_agent_plane",
    "run_cua_speed_run_reference",
    "verify_cua_speed_run_checkout",
]
