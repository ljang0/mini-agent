"""Cross-run comparison: a projection of committed evidence, never a recompute."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mini_agent.cli import main
from mini_agent.report import (
    COLUMNS,
    SCHEMA,
    best_of,
    format_table,
    load_run,
    report,
)


def write_run(
    root: Path,
    name: str,
    *,
    harness: str | None = None,
    team_size: int | None = None,
    scores: list[float | None],
    summary: dict[str, Any] | None = None,
    coordination: dict[str, Any] | None = None,
    max_concurrency: int = 4,
    commit: bool = True,
) -> Path:
    run = root / name
    (run / "instances").mkdir(parents=True)
    topology: dict[str, Any] = {"mode": "single", "max_steps": 64}
    if harness is not None:
        topology["harness"] = harness
        topology["team_size"] = team_size
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "mini-agent-eval-v2",
                "benchmark": "aime",
                "limits": {"max_concurrency": max_concurrency},
                "config": {"model": "meta/m", "topology": topology},
            }
        )
    )
    numeric = [score for score in scores if score is not None]
    (run / "summary.json").write_text(
        json.dumps(
            summary
            if summary is not None
            else {
                "benchmark": "aime",
                "tasks": len(scores),
                "mean_score": (
                    sum(numeric) / len(numeric) if numeric else None
                ),
                "elapsed_seconds": 10.0,
                "backend_active_union_seconds": 9.0,
                "usage": {"output_tokens": 100},
            }
        )
    )
    for index, score in enumerate(scores):
        instance = run / "instances" / f"{index:064d}"
        instance.mkdir()
        metadata: dict[str, Any] = {}
        if coordination is not None:
            metadata["coordination"] = {"totals": coordination}
        commit_result(
            instance,
            {
                "task_id": f"t{index}",
                "status": "completed",
                "score": score,
                "metadata": metadata,
            },
            commit=commit,
        )
    return run


def commit_result(
    instance: Path, result: dict[str, Any], *, commit: bool = True
) -> None:
    """Write one instance the way the runner does: bytes first, then marker."""

    payload = json.dumps(result).encode()
    (instance / "result.json").write_bytes(payload)
    if commit:
        (instance / "completed.json").write_text(
            json.dumps(
                {
                    "task_id": result["task_id"],
                    "result_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )


class LoadRunTests(unittest.TestCase):
    def test_a_run_projects_onto_the_comparison_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = write_run(root, "a", harness="fixed-team", team_size=3,
                            scores=[1.0, 0.0])
            row = load_run(run)
            self.assertEqual(row["benchmark"], "aime")
            self.assertEqual(row["harness"], "fixed-team")
            self.assertEqual(row["team_size"], 3)
            self.assertEqual(row["tasks"], 2)
            self.assertEqual(row["mean_score"], 0.5)
            self.assertEqual(row["output_tokens"], 100)

    def test_a_legacy_manifest_without_a_harness_key_reads_as_single(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            self.assertEqual(load_run(run)["harness"], "single")

    def test_uncommitted_instances_are_not_counted(self) -> None:
        # Resume itself refuses to trust an uncommitted instance; a table that
        # averaged one in would report a score resume would not.
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0], commit=False)
            row = load_run(run)
            self.assertEqual(row["tasks"], 0)
            self.assertIsNone(row["mean_score"])

    def test_a_summary_disagreeing_with_its_results_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = write_run(
                root,
                "a",
                scores=[0.0],
                summary={"mean_score": 1.0, "elapsed_seconds": 1.0, "usage": {}},
            )
            row = load_run(run)
            self.assertEqual(row["mean_score"], 0.0)
            self.assertTrue(
                any("disagrees" in warning for warning in row["warnings"]),
                row["warnings"],
            )

    def test_concurrency_below_team_size_warns_about_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(
                Path(temporary), "a", harness="fixed-team", team_size=10,
                scores=[1.0], max_concurrency=4,
            )
            row = load_run(run)
            self.assertTrue(
                any("latency" in warning for warning in row["warnings"]),
                row["warnings"],
            )

    def test_matching_concurrency_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(
                Path(temporary), "a", harness="fixed-team", team_size=3,
                scores=[1.0], max_concurrency=3,
            )
            self.assertEqual(load_run(run)["warnings"], [])

    def test_coordination_totals_are_summed_across_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(
                Path(temporary), "a", scores=[1.0, 1.0],
                coordination={
                    "agents": 3,
                    "messages": 2,
                    "idle_seconds": 4.0,
                    "duplicate_tool_calls": 1,
                },
            )
            row = load_run(run)
            self.assertEqual(row["agents"], 6)
            self.assertEqual(row["messages"], 4)
            self.assertEqual(row["idle_s"], 8.0)
            self.assertEqual(row["dup_calls"], 2)

    def test_a_pre_v2_coordination_block_reports_unmeasured_not_zero(self) -> None:
        # "Not measured" and "measured as none" are different findings, and a
        # zero here would claim an old run proved there was no idle time.
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(
                Path(temporary), "a", scores=[1.0],
                coordination={"agents": 1, "messages": 0},
            )
            row = load_run(run)
            self.assertIsNone(row["idle_s"])
            self.assertIsNone(row["dup_calls"])

    def test_a_result_rewritten_after_its_marker_is_flagged_not_averaged(self) -> None:
        # The commit marker binds the result bytes. A row that averaged in a
        # result the collectors and resume both reject would report a score
        # nothing else in the system agrees with.
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            instance = next((run / "instances").iterdir())
            (instance / "result.json").write_text(
                json.dumps(
                    {"task_id": "t0", "status": "completed", "score": 0.0}
                )
            )
            row = load_run(run)
            self.assertEqual(row["tasks"], 0)
            self.assertTrue(row["warnings"], row)

    def test_an_unfinished_run_reports_what_it_committed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            (run / "summary.json").unlink()
            row = load_run(run)
            self.assertEqual(row["mean_score"], 1.0)
            self.assertIsNone(row["elapsed_s"])
            self.assertTrue(
                any("did not finish" in warning for warning in row["warnings"]),
                row["warnings"],
            )

    def test_a_non_numeric_score_is_not_averaged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0, None])
            row = load_run(run)
            self.assertEqual(row["tasks"], 2)
            self.assertEqual(row["mean_score"], 1.0)

    def test_a_directory_that_is_not_an_evaluation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                load_run(Path(temporary))


class ReportTests(unittest.TestCase):
    def test_rows_sort_into_a_stable_comparison_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                write_run(root, "z", harness="orchestrator", scores=[1.0]),
                write_run(root, "a", harness="fixed-team", team_size=10,
                          scores=[1.0], max_concurrency=10),
                write_run(root, "m", harness="fixed-team", team_size=3,
                          scores=[1.0], max_concurrency=3),
            ]
            value = report(runs)
            self.assertEqual(value["schema"], SCHEMA)
            self.assertEqual(
                [(row["harness"], row["team_size"]) for row in value["rows"]],
                [("fixed-team", 3), ("fixed-team", 10), ("orchestrator", None)],
            )

    def test_an_empty_request_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            report([])

    def test_the_table_renders_every_column_and_its_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(
                Path(temporary), "a", harness="fixed-team", team_size=10,
                scores=[1.0], max_concurrency=4,
            )
            text = format_table(report([run]))
            header, _separator, data, warning = text.splitlines()
            self.assertEqual(header.split(), list(COLUMNS))
            self.assertIn("fixed-team", data)
            self.assertTrue(warning.startswith("! a:"), warning)
            # A run with no coordination block reads as absent, not as zero.
            self.assertEqual(data.split()[COLUMNS.index("idle_s")], "-")

    def test_an_empty_report_still_renders_its_header(self) -> None:
        self.assertEqual(
            format_table({"schema": SCHEMA, "columns": list(COLUMNS), "rows": []})
            .splitlines()[0]
            .split(),
            list(COLUMNS),
        )


class BestOfTests(unittest.TestCase):
    """Best@k against the independent-agent expectation."""

    def _runs(self, root: Path, *score_sets: list[float | None]) -> list[Path]:
        return [
            write_run(root, f"r{index}", scores=scores)
            for index, scores in enumerate(score_sets)
        ]

    def test_independent_runs_reach_the_expected_union(self) -> None:
        # Disjoint failures: two runs at 0.5 each solve everything together,
        # which is exactly 1 - (1 - 0.5)^2.
        with tempfile.TemporaryDirectory() as temporary:
            runs = self._runs(
                Path(temporary), [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]
            )
            value = best_of(runs)
            self.assertEqual(value["mean_score"], 0.5)
            self.assertEqual(value["observed_best_of_k"], 1.0)
            self.assertEqual(value["independent_expectation"], 0.75)
            # A perfect union is consistent with any team size, so this
            # declines to name one rather than inventing infinity.
            self.assertIsNone(value["effective_team_size"])

    def test_identical_runs_are_worth_one_agent(self) -> None:
        # Perfectly correlated agents add nothing: the union equals one run.
        with tempfile.TemporaryDirectory() as temporary:
            runs = self._runs(
                Path(temporary), [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
            )
            value = best_of(runs)
            self.assertEqual(value["observed_best_of_k"], 0.25)
            self.assertEqual(value["independent_expectation"], 0.4375)
            self.assertEqual(value["effective_team_size"], 1.0)

    def test_partial_correlation_lands_between_one_and_k(self) -> None:
        # The runs overlap on t0 but not t1, so they are neither identical nor
        # independent, and the effective size has to land strictly between.
        with tempfile.TemporaryDirectory() as temporary:
            runs = self._runs(
                Path(temporary), [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
            )
            size = best_of(runs)["effective_team_size"]
            assert size is not None
            self.assertLess(1.0, size)
            self.assertLess(size, 2.0)

    def test_a_degenerate_mean_declines_to_name_a_team_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = self._runs(Path(temporary), [0.0, 0.0], [0.0, 0.0])
            value = best_of(runs)
            self.assertEqual(value["observed_best_of_k"], 0.0)
            self.assertIsNone(value["effective_team_size"])

    def test_only_tasks_every_run_scored_are_compared(self) -> None:
        # Otherwise a run that crashed early would look like a run that failed
        # the tasks it never attempted.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                write_run(root, "a", scores=[1.0, 0.0]),
                write_run(root, "b", scores=[0.0]),
            ]
            self.assertEqual(best_of(runs)["tasks"], 1)

    def test_one_run_cannot_be_compared_with_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            with self.assertRaises(ValueError):
                best_of([run])

    def test_runs_with_no_shared_tasks_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_run(root, "a", scores=[1.0])
            second = write_run(root, "b", scores=[1.0])
            instance = next((second / "instances").iterdir())
            commit_result(
                instance, {"task_id": "other", "status": "completed", "score": 1.0}
            )
            with self.assertRaises(ValueError):
                best_of([first, second])


class ReportCommandTests(unittest.TestCase):
    def _run(self, root: Path, *extra: str) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["report", "--runs", str(root), *extra]), 0)
        return stdout.getvalue()

    def test_the_command_prints_a_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            self.assertIn("mean_score", self._run(run))

    def test_the_command_reports_best_of_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_run(root, "a", scores=[1.0, 0.0])
            second = write_run(root, "b", scores=[0.0, 1.0])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        ["report", "--runs", str(first), str(second), "--best-of"]
                    ),
                    0,
                )
            self.assertIn("best@2", stdout.getvalue())

    def test_the_command_prints_json_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = write_run(Path(temporary), "a", scores=[1.0])
            value = json.loads(self._run(run, "--format", "json"))
            self.assertEqual(value["schema"], SCHEMA)
            self.assertEqual(value["rows"][0]["mean_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
