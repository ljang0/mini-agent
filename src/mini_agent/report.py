"""Join many runs into one table, so topologies can be compared.

Every artifact this project writes is per-run: one manifest, one summary, one
result per task. That is the right shape for evidence and the wrong shape for
the question the runs were made to answer, which is always comparative -- this
harness against that one, at this team size, on the same benchmark and model.

Nothing here re-grades anything. A row is a projection of what a run already
committed; the one figure it re-derives is the mean, from those same committed
results, so that a summary disagreeing with its own tasks is reported rather
than silently averaged in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmarks.base import mean_score, owned_instance_artifacts
from .storage import read_committed_result, read_json_object


SCHEMA = "mini-agent-report-v1"
BEST_OF_SCHEMA = "mini-agent-best-of-v1"

# What a comparison actually needs: what was run, how well it did, what it
# cost, and how much of the cost was coordination rather than work.
COLUMNS = (
    "benchmark",
    "harness",
    "team_size",
    "model",
    "tasks",
    "completed",
    "mean_score",
    "elapsed_s",
    "active_s",
    "output_tokens",
    "agents",
    "messages",
    "idle_s",
    "dup_calls",
    "run",
)


def load_run(path: Path) -> dict[str, Any]:
    """Project one evaluation output directory into a row.

    Raises if the directory is not an evaluation output at all; records a
    warning when it is one whose parts disagree, because a run that cannot be
    trusted should still appear in the table rather than vanish from it.
    """

    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"not an evaluation output directory: {root}")
    manifest = read_json_object(manifest_path)
    config = _mapping(manifest.get("config"))
    topology = _mapping(config.get("topology"))
    warnings: list[str] = []

    summary: Mapping[str, Any] = {}
    if summary_path.is_file():
        summary = read_json_object(summary_path)
    else:
        # An interrupted run has committed results but no summary; reporting
        # what finished is more useful than refusing the whole row.
        warnings.append("no summary.json: run did not finish")

    results, tampered = _committed_results(root)
    warnings.extend(tampered)
    completed = sum(result.get("status") == "completed" for result in results)
    declared = summary.get("mean_score")
    observed = mean_score(results)
    if (
        isinstance(declared, (int, float))
        and observed is not None
        and abs(float(declared) - observed) > 1e-9
    ):
        warnings.append("summary mean_score disagrees with committed results")

    coordination = _coordination_totals(results)
    usage = _mapping(summary.get("usage"))
    values = {
        "benchmark": manifest.get("benchmark"),
        "harness": topology.get("harness", "single"),
        "team_size": topology.get("team_size"),
        "model": config.get("model"),
        "tasks": len(results),
        "completed": completed,
        "mean_score": observed,
        "elapsed_s": _rounded(summary.get("elapsed_seconds")),
        "active_s": _rounded(summary.get("backend_active_union_seconds")),
        "output_tokens": usage.get("output_tokens"),
        "agents": coordination["agents"],
        "messages": coordination["messages"],
        "idle_s": coordination["idle_seconds"],
        "dup_calls": coordination["duplicate_tool_calls"],
        "run": root.name,
    }
    concurrency = _mapping(manifest.get("limits")).get("max_concurrency")
    size = topology.get("team_size")
    if (
        isinstance(concurrency, int)
        and isinstance(size, int)
        and concurrency < size
    ):
        # Documented in docs/harnesses.md: below this the team serializes on one
        # semaphore and every latency number in the row is meaningless.
        warnings.append(
            f"model concurrency {concurrency} is below team size {size}: "
            "latency columns are not comparable"
        )
    # `warnings` is deliberately not in COLUMNS: it is carried on the row for
    # the renderer and the JSON form, not shown as a column.
    return {**values, "warnings": warnings}


def report(paths: Sequence[Path]) -> Mapping[str, Any]:
    """Load every run and sort the rows into a stable comparison order."""

    if not paths:
        raise ValueError("report requires at least one run directory")
    return {
        "schema": SCHEMA,
        "columns": list(COLUMNS),
        "rows": sorted((load_run(path) for path in paths), key=_order),
    }


def format_table(value: Mapping[str, Any]) -> str:
    """Render the report as fixed-width columns, warnings beneath."""

    columns = [str(name) for name in value.get("columns", COLUMNS)]
    rows = [_mapping(row) for row in value.get("rows", [])]
    cells = [[_cell(row.get(name)) for name in columns] for row in rows]
    widths = [
        max([len(name), *(len(row[index]) for row in cells)])
        for index, name in enumerate(columns)
    ]
    lines = [
        "  ".join(name.ljust(width) for name, width in zip(columns, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in cells
    ]
    for row in rows:
        for warning in row.get("warnings", []):
            lines.append(f"! {row.get('run')}: {warning}")
    return "\n".join(lines)


def format_best_of(value: Mapping[str, Any]) -> str:
    """Render the best-of block, saying plainly what the gap means."""

    lines = [
        f"best@{value['runs']} over {value['tasks']} shared tasks",
        f"  per-run mean          {_cell(value['mean_score'])}",
        f"  observed best@k       {_cell(value['observed_best_of_k'])}",
        f"  independent expected  {_cell(value['independent_expectation'])}",
        f"  effective team size   {_cell(value['effective_team_size'])}"
        f" of {value['runs']}",
    ]
    size = value.get("effective_team_size")
    if isinstance(size, (int, float)):
        lines.append(
            f"  ({value['runs']} agents did the work of {size:.3g} independent"
            " ones; the shortfall is correlated behaviour, not budget)"
        )
    return "\n".join(lines)


def best_of(paths: Sequence[Path]) -> Mapping[str, Any]:
    """Compare what k runs achieved together against k independent runs.

    Treat the given runs as repeated samples of one configuration. Two numbers
    then say how much the extra agents actually bought:

    - **observed** is the fraction of tasks *some* run solved — best@k.
    - **expected** is what k statistically independent runs would have scored,
      ``1 - (1 - p)^k`` at the per-run mean ``p``.

    Agents that think alike solve the same tasks and fail the same tasks, so
    observed falls below expected. Inverting the same formula turns that gap
    into an **effective team size**: the number of genuinely independent runs
    that would have produced what these k actually produced. A team of ten
    worth 2.3 independent runs is the finding this exists to surface, and it is
    invisible in a mean score.

    This assumes per-task scores behave like successes, which is true for the
    graded benchmarks here. It is reported alongside the raw numbers rather
    than instead of them.
    """

    if len(paths) < 2:
        raise ValueError("best-of needs at least two runs to compare")
    per_run = [_task_scores(path) for path in paths]
    shared = set(per_run[0])
    for scores in per_run[1:]:
        shared &= set(scores)
    if not shared:
        raise ValueError("the runs share no task ids, so they are not comparable")
    means = [
        sum(scores[task] for task in shared) / len(shared) for scores in per_run
    ]
    mean = sum(means) / len(means)
    observed = sum(
        max(scores[task] for scores in per_run) for task in shared
    ) / len(shared)
    count = len(paths)
    expected = 1.0 - (1.0 - mean) ** count if 0.0 <= mean <= 1.0 else None
    return {
        "schema": BEST_OF_SCHEMA,
        "runs": count,
        "tasks": len(shared),
        "mean_score": round(mean, 6),
        "observed_best_of_k": round(observed, 6),
        "independent_expectation": (
            None if expected is None else round(expected, 6)
        ),
        "effective_team_size": _effective_team_size(mean, observed),
    }


def _effective_team_size(mean: float, observed: float) -> float | None:
    """Invert ``1 - (1 - p)^k`` at the observed score.

    ``None`` where the question has no answer rather than a misleading one: a
    per-run mean of 0 or 1 leaves nothing for extra agents to add, and a
    perfect union is consistent with any team size.
    """

    if not 0.0 < mean < 1.0 or observed >= 1.0:
        return None
    return round(math.log(1.0 - observed) / math.log(1.0 - mean), 3)


def _task_scores(path: Path) -> dict[str, float]:
    results, _ = _committed_results(path.expanduser().resolve())
    return {
        str(result["task_id"]): float(result["score"])
        for result in results
        if isinstance(result.get("task_id"), str)
        and isinstance(result.get("score"), (int, float))
        and not isinstance(result.get("score"), bool)
    }


def _committed_results(root: Path) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Read every result the run committed, and only those.

    An uncommitted instance is work that may have been interrupted mid-spend,
    and is skipped: including it would put a result in the table that resume
    itself refuses to trust. A committed one is read through the same checked
    reader every collector uses, so a `result.json` rewritten after its marker
    is reported rather than averaged in.
    """

    if not (root / "instances").is_dir():
        return [], []
    _, _, artifacts = owned_instance_artifacts(root, "result.json", label="report")
    results: list[Mapping[str, Any]] = []
    warnings: list[str] = []
    for artifact in artifacts:
        if not (artifact.parent / "completed.json").is_file():
            continue
        task_id = read_json_object(artifact).get("task_id")
        try:
            if not isinstance(task_id, str):
                raise ValueError("committed result has no task id")
            results.append(read_committed_result(artifact.parent, task_id))
        except ValueError as exc:
            warnings.append(f"{artifact.parent.name}: {exc}")
    return results, warnings


def _coordination_totals(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the per-task coordination blocks, tolerating older schemas.

    Runs recorded before idle time and duplicate work existed report neither;
    those columns stay ``None`` for the whole run rather than reading as zero,
    because "not measured" and "measured as none" are different findings.
    """

    blocks = [_totals_block(result) for result in results]
    idle = _summed(blocks, "idle_seconds", (int, float))
    return {
        "agents": sum(_int(block.get("agents")) for block in blocks),
        "messages": sum(_int(block.get("messages")) for block in blocks),
        "idle_seconds": None if idle is None else round(float(idle), 3),
        "duplicate_tool_calls": _summed(blocks, "duplicate_tool_calls", int),
    }


def _totals_block(result: Mapping[str, Any]) -> Mapping[str, Any]:
    coordination = _mapping(_mapping(result.get("metadata")).get("coordination"))
    return _mapping(coordination.get("totals"))


def _summed(
    blocks: Sequence[Mapping[str, Any]], key: str, kinds: Any
) -> float | int | None:
    """Sum one key across blocks, or ``None`` if any block never measured it."""

    if any(key not in block for block in blocks):
        return None
    return sum(block[key] for block in blocks if isinstance(block[key], kinds))


def _order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("benchmark") or ""),
        str(row.get("model") or ""),
        str(row.get("harness") or ""),
        _int(row.get("team_size")),
        str(row.get("run") or ""),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _rounded(value: Any) -> Any:
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _report(args: Any) -> int:
    """The ``mini-agent report`` command."""

    paths = [Path(path) for path in args.runs]
    value: dict[str, Any] = dict(report(paths))
    if getattr(args, "best_of", False):
        value["best_of"] = best_of(paths)
    if args.format == "json":
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(format_table(value))
        if "best_of" in value:
            print()
            print(format_best_of(value["best_of"]))
    return 0


__all__ = [
    "BEST_OF_SCHEMA",
    "COLUMNS",
    "SCHEMA",
    "best_of",
    "format_best_of",
    "format_table",
    "load_run",
    "report",
]
