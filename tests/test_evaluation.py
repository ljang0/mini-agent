import json
import os
import tempfile
import unittest
from pathlib import Path

from scaffoldlab.environments.configured import ConfiguredEnvironmentFactory
from scaffoldlab.evaluation import MatrixRunner, evaluate_answer, load_tasks
from scaffoldlab.harnesses import SingleAgentHarness
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelResponse, Task, Usage


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_swe_source_and_artifacts_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            output = source / "runs" / "trial"
            task = Task(
                "overlap",
                "question",
                metadata={"environment": {"workspace": str(source)}},
            )
            factory = ConfiguredEnvironmentFactory(
                {"type": "swe", "workspace_mode": "copy"}
            )
            runner = MatrixRunner(
                backend=ScriptedBackend({"/root": ["answer"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=output,
                environment_factory=factory,
            )
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                await runner.run([task], [SingleAgentHarness()])
            self.assertFalse(output.exists())

    async def test_direct_swe_workspace_rejects_multi_trial_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            factory = ConfiguredEnvironmentFactory(
                {
                    "type": "swe",
                    "workspace": str(source),
                    "workspace_mode": "direct",
                    "isolation": "shared",
                }
            )
            runner = MatrixRunner(
                backend=ScriptedBackend({"/root": ["one", "two"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=root / "output",
                environment_factory=factory,
            )
            with self.assertRaisesRegex(ValueError, "exactly one planned trial"):
                await runner.run(
                    [Task("one", "q"), Task("two", "q")],
                    [SingleAgentHarness()],
                )
            self.assertFalse((root / "output").exists())

    def test_deterministic_evaluators(self) -> None:
        task = Task(
            "contains",
            "question",
            metadata={"evaluator": {"type": "contains", "value": "orbit"}},
        )
        self.assertTrue(evaluate_answer(task, "Project ORBIT shipped").passed)

        boolean_task = Task(
            "json-bool",
            "question",
            metadata={"evaluator": {"type": "json_equal", "value": True}},
        )
        self.assertFalse(evaluate_answer(boolean_task, "1").passed)
        self.assertTrue(evaluate_answer(boolean_task, "true").passed)

    def test_invalid_or_trivial_evaluators_fail_during_task_load(self) -> None:
        invalid_specs = [
            {"type": "unknown", "value": "x"},
            {"type": "regex", "value": "["},
            {"type": "regex", "value": ""},
            {"type": "contains", "value": "   "},
        ]
        for index, evaluator in enumerate(invalid_specs):
            with (
                self.subTest(evaluator=evaluator),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "tasks.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "task_id": f"task-{index}",
                            "prompt": "question",
                            "evaluator": evaluator,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_tasks(path)

    def test_task_metadata_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "bad-metadata",
                        "prompt": "question",
                        "metadata": ["not", "an", "object"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metadata must be an object"):
                load_tasks(path)

    async def test_matrix_writes_results_and_human_review_gate(self) -> None:
        backend = ScriptedBackend({"/root": ["ORBIT"]})
        task = Task(
            "t1",
            "question",
            metadata={"evaluator": {"type": "exact", "value": "ORBIT"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = MatrixRunner(
                backend=backend,
                limits=BudgetLimits(wall_time_seconds=5),
                output_dir=Path(directory),
                repeats=1,
                random_seed=7,
            )
            records, summary = await runner.run([task], [SingleAgentHarness()])
            self.assertTrue(records[0].passed)
            self.assertIsNone(records[0].answer)
            self.assertIn("answer_sha256", records[0].metadata)
            self.assertEqual(summary["release_decision"], "HUMAN_REVIEW_REQUIRED")
            self.assertTrue((Path(directory) / "results.jsonl").exists())
            self.assertTrue((Path(directory) / "summary.json").exists())

    async def test_failed_run_preserves_billed_usage_and_trace(self) -> None:
        backend = ScriptedBackend(
            {
                "/root": [
                    ModelResponse(
                        text="too expensive",
                        usage=Usage(input_tokens=9, output_tokens=5, cost_usd=1.25),
                    )
                ]
            }
        )
        task = Task("over-budget", "question")
        with tempfile.TemporaryDirectory() as directory:
            runner = MatrixRunner(
                backend=backend,
                limits=BudgetLimits(
                    max_output_tokens=1,
                    max_cost_usd=2.0,
                    wall_time_seconds=5,
                ),
                output_dir=Path(directory),
            )
            records, _ = await runner.run([task], [SingleAgentHarness()])
            self.assertEqual(records[0].status, "error")
            self.assertEqual(records[0].output_tokens, 5)
            self.assertEqual(records[0].cost_usd, 1.25)
            self.assertTrue(Path(records[0].metadata["trace_path"]).exists())

    async def test_final_trial_reaching_matrix_cap_is_still_complete(self) -> None:
        backend = ScriptedBackend(
            {"/root": [ModelResponse(text="answer", usage=Usage(cost_usd=0.25))]}
        )
        with tempfile.TemporaryDirectory() as directory:
            _, summary = await MatrixRunner(
                backend=backend,
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=Path(directory),
                matrix_max_cost_usd=0.25,
            ).run([Task("only", "question")], [SingleAgentHarness()])
        self.assertTrue(summary["matrix_completed"])
        self.assertIsNone(summary["termination_reason"])

    async def test_overwrite_removes_stale_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            await MatrixRunner(
                backend=ScriptedBackend({"/root": ["one", "two"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=output,
            ).run(
                [Task("one", "q"), Task("two", "q")],
                [SingleAgentHarness()],
            )
            self.assertEqual(len(list((output / "traces").glob("*.jsonl"))), 2)
            await MatrixRunner(
                backend=ScriptedBackend({"/root": ["fresh"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=output,
                overwrite=True,
            ).run([Task("fresh", "q")], [SingleAgentHarness()])
            self.assertEqual(len(list((output / "traces").glob("*.jsonl"))), 1)

    @unittest.skipUnless(os.name == "posix", "symlink semantics require POSIX")
    async def test_trace_symlink_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            escaped = root / "escaped"
            output.mkdir()
            escaped.mkdir()
            (output / "traces").symlink_to(escaped, target_is_directory=True)
            runner = MatrixRunner(
                backend=ScriptedBackend({"/root": ["answer"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=output,
            )
            with self.assertRaises(FileExistsError):
                await runner.run([Task("task", "q")], [SingleAgentHarness()])
            self.assertEqual(list(escaped.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
