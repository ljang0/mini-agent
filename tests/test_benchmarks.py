from __future__ import annotations

import base64
import asyncio
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, Mock, patch

from mini_agent.agent import MiniAgent
from mini_agent.benchmarks.base import (
    BenchmarkTask,
    EvaluationOutcome,
    EvaluationRunner,
    atomic_json,
    harness_identity,
    machine_image_identity,
    raise_after_cleanup,
    task_agent_root,
)
from mini_agent.benchmarks.cua_speedrun import (
    _AdapterPool,
    _activate,
    _cached_machine_image_identity,
    _checker,
    _generated_task_sha256,
    _gym_anything_task_identity,
    _on_task_clock,
    _prepared_runtime_assets,
    _TaskClockExpired,
    _task_source_sha256,
    _upstream_task,
    inspect_cua_speedrun_checkout,
    load_cua_speedrun,
    prepare_cua_speedrun_backend,
    preflight_cua_speedrun,
    run_cua_speedrun_task,
)
from mini_agent.benchmarks.osworld import (
    UpstreamDesktopFactory,
    _DesktopPool,
    _config_sha256,
    _reject_unsupported_v2_lifecycle,
    _task_config_for_benchmark,
    inspect_osworld_checkout,
    load_osworld,
    run_osworld_task,
)
from mini_agent.benchmarks.swebench import (
    SWEBENCH_SOURCE_SHA256,
    collect_predictions,
    inspect_swebench_grade_inputs,
    load_swebench,
    official_grader_argv,
    prepare_swebench_image_bindings,
    run_swebench_task,
    swebench_grader_source_identity,
    swebench_grader_image_name,
    verify_swebench_grader_images,
)
from mini_agent.benchmarks.web import (
    BROWSECOMP_PLUS_QUERY,
    collect_browsecomp_plus_runs,
    inspect_browsecomp_plus_grade_inputs,
    load_browsecomp,
    load_browsecomp_plus,
    official_browsecomp_plus_grader_argv,
    run_web_task,
)
from mini_agent.environments.base import BaseEnvironment
from mini_agent.environments.cua import CUAEnvironment, CUASpeedRunAdapterClient
from mini_agent.environments.swebench import SWEbenchImageBinding
from mini_agent.environments.web import BrowserEnvironment, JsonlSearchBackend
from mini_agent.models import ScriptedModel
from mini_agent.runtime import RunContext, TraceRecorder
from mini_agent.types import (
    BudgetLimits,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolExecution,
)

from support import EmptyEnvironment, WordTokenizer, png, working_directory


class EvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_artifacts_stay_private_under_permissive_umask(self) -> None:
        previous = os.umask(0)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "run"
                runner = EvaluationRunner(
                    benchmark="fixture",
                    tasks=(BenchmarkTask("id", "prompt"),),
                    output=root,
                    config={},
                    limits=BudgetLimits(),
                )
                runner._prepare(runner._manifest(), resume=False)
                private = root / "instances" / "private" / "grader.json"
                atomic_json(private, {"hidden": "answer"})
                self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
                self.assertEqual(
                    stat.S_IMODE((root / "instances").stat().st_mode), 0o700
                )
                self.assertEqual(stat.S_IMODE(private.parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
        finally:
            os.umask(previous)

    def test_atomic_artifact_does_not_chmod_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "shared"
            parent.mkdir()
            parent.chmod(0o1777)
            artifact = parent / "artifact.json"
            atomic_json(artifact, {"ok": True})
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o1777)
            self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o600)

    def test_atomic_artifact_syncs_bytes_and_directory_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            observed: list[str] = []
            real_fsync = os.fsync

            def record_fsync(descriptor: int) -> None:
                mode = os.fstat(descriptor).st_mode
                observed.append("directory" if stat.S_ISDIR(mode) else "file")
                real_fsync(descriptor)

            with patch(
                "mini_agent.benchmarks.base.os.fsync", side_effect=record_fsync
            ):
                atomic_json(parent / "artifact.json", {"ok": True})

            self.assertEqual(observed, ["file", "directory"])

    async def test_trace_is_durable_before_task_completion_commit(self) -> None:
        ordering: list[str] = []
        real_atomic_json = atomic_json

        class OrderingTrace(TraceRecorder):
            async def sync(self) -> None:
                ordering.append("trace")
                await super().sync()

        def recording_atomic_json(path: Path, value: Any) -> None:
            if path.name == "completed.json":
                ordering.append("commit")
            real_atomic_json(path, value)

        async def worker(
            task: BenchmarkTask, context: RunContext, directory: Path
        ) -> EvaluationOutcome:
            del context, directory
            return EvaluationOutcome(task_id=task.task_id, status="completed")

        with tempfile.TemporaryDirectory() as temporary:
            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(BenchmarkTask("id", "prompt"),),
                output=Path(temporary) / "run",
                config={},
                limits=BudgetLimits(),
            )
            with (
                patch("mini_agent.benchmarks.base.TraceRecorder", OrderingTrace),
                patch(
                    "mini_agent.benchmarks.base.atomic_json",
                    side_effect=recording_atomic_json,
                ),
            ):
                await runner.run(worker)

        self.assertEqual(ordering, ["trace", "commit"])

    def test_manifest_binds_location_independent_harness_identity(self) -> None:
        identity = harness_identity()
        self.assertEqual(identity["schema"], "mini-agent-harness-v1")
        self.assertEqual(len(identity["source_sha256"]), 64)
        self.assertGreater(identity["source_file_count"], 0)
        self.assertNotIn(str(Path.cwd()), json.dumps(identity))

        with tempfile.TemporaryDirectory() as temporary:
            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(BenchmarkTask("id", "prompt"),),
                output=Path(temporary) / "run",
                config={},
                limits=BudgetLimits(),
            )
            with patch(
                "mini_agent.benchmarks.base.harness_identity",
                return_value={"source_sha256": "first"},
            ):
                runner._prepare(runner._manifest(), resume=False)
            with patch(
                "mini_agent.benchmarks.base.harness_identity",
                return_value={"source_sha256": "second"},
            ):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    runner._prepare(runner._manifest(), resume=True)

    def test_machine_image_sidecar_must_bind_the_full_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "machine.qcow2"
            image.write_bytes(b"image")
            sidecar = Path(str(image) + ".provenance.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": "fixture-v1",
                        "final_image_sha256": "0" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                machine_image_identity(image, label="fixture image")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": "fixture-v1",
                        "final_image_sha256": hashlib.sha256(b"image").hexdigest(),
                    }
                )
            )
            identity = machine_image_identity(image, label="fixture image")
            self.assertEqual(identity["provenance_schema"], "fixture-v1")

            sidecar.unlink()
            sidecar.symlink_to(Path(temporary) / "missing.json")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                machine_image_identity(image, label="fixture image")

    def test_manifest_inputs_require_stable_json_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON"):
            BenchmarkTask("id", "prompt", {"bad": object()})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "JSON"):
                EvaluationRunner(
                    benchmark="fixture",
                    tasks=(BenchmarkTask("id", "prompt"),),
                    output=Path(temporary) / "run",
                    config={"bad": float("nan")},
                    limits=BudgetLimits(),
                )

    def test_instance_paths_use_full_task_identity_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(BenchmarkTask("task", "prompt"),),
                output=Path(temporary) / "run",
                config={},
                limits=BudgetLimits(),
            )
            self.assertEqual(
                runner._instance("task").name,
                hashlib.sha256(b"task").hexdigest(),
            )

    def test_primary_and_cleanup_failures_are_both_reported(self) -> None:
        primary = ValueError("primary")
        cleanup = RuntimeError("cleanup")
        with self.assertRaisesRegex(
            RuntimeError, "primary.*cleanup also failed.*cleanup"
        ) as caught:
            raise_after_cleanup("fixture", primary, cleanup)
        self.assertIs(caught.exception.__cause__, primary)

    async def test_shared_scheduler_trace_accounting_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            tasks = tuple(
                BenchmarkTask(f"task-{index}", f"prompt-{index}", {"hidden": index})
                for index in range(3)
            )
            active = 0
            peak = 0

            async def worker(
                task: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    result = await MiniAgent(
                        model=ScriptedModel([ModelResponse(f"answer-{task.task_id}")]),
                        environment=EmptyEnvironment(),
                        context=context,
                        agent_id=task_agent_root(task.task_id),
                    ).run(task.prompt)
                    return EvaluationOutcome(
                        task.task_id, "completed", answer=result.answer, score=1.0
                    )
                finally:
                    active -= 1

            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=tasks,
                output=output,
                config={"model": "scripted"},
                limits=BudgetLimits(max_model_calls=3),
                max_workers=2,
            )
            summary = await runner.run(worker)
            self.assertEqual(summary["completed"], 3)
            self.assertEqual(summary["model_calls"], 3)
            self.assertEqual(peak, 2)
            self.assertTrue((output / "trace.jsonl").is_file())
            first_elapsed = summary["elapsed_seconds"]
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("data_sha256", manifest["tasks"][0])
            self.assertNotIn("hidden", json.dumps(manifest))

            async def must_not_run(*args: Any) -> EvaluationOutcome:
                raise AssertionError(f"resume reran a completed task: {args}")

            resumed = EvaluationRunner(
                benchmark="fixture",
                tasks=tasks,
                output=output,
                config={"model": "scripted"},
                limits=BudgetLimits(max_model_calls=3),
                max_workers=2,
            )
            (output / "summary.json").unlink()
            resumed_elapsed, resumed_active = resumed._resume_timing()
            self.assertGreater(resumed_elapsed, 0)
            self.assertGreater(resumed_active, 0)
            resumed_summary = await resumed.run(must_not_run, resume=True)
            self.assertEqual(resumed_summary["model_calls"], 3)
            self.assertGreaterEqual(resumed_summary["elapsed_seconds"], first_elapsed)
            elapsed = [
                json.loads(line)["elapsed_seconds"]
                for line in (output / "trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(elapsed, sorted(elapsed))

    async def test_resume_rejects_hidden_data_or_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            task = BenchmarkTask("id", "same prompt", {"answer": "one"})

            async def worker(
                item: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                del context, directory
                return EvaluationOutcome(item.task_id, "completed")

            await EvaluationRunner(
                benchmark="fixture",
                tasks=(task,),
                output=output,
                config={"version": 1},
                limits=BudgetLimits(),
            ).run(worker)
            changed = BenchmarkTask("id", "same prompt", {"answer": "two"})
            with self.assertRaisesRegex(ValueError, "manifest"):
                await EvaluationRunner(
                    benchmark="fixture",
                    tasks=(changed,),
                    output=output,
                    config={"version": 1},
                    limits=BudgetLimits(),
                ).run(worker, resume=True)

    async def test_worker_errors_are_terminal_evidence_not_scheduler_crashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:

            async def worker(
                task: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                del task, context, directory
                raise RuntimeError("broken")

            summary = await EvaluationRunner(
                benchmark="fixture",
                tasks=(BenchmarkTask("id", "prompt"),),
                output=Path(temporary) / "run",
                config={},
                limits=BudgetLimits(),
            ).run(worker)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["blocked"], 0)

    async def test_worker_error_secrets_are_redacted_from_result_artifacts(
        self,
    ) -> None:
        secret = "AUDIT_SECRET_9f0f"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"

            async def worker(
                task: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                del task, context, directory
                raise RuntimeError("backend echoed " + secret)

            await EvaluationRunner(
                benchmark="fixture",
                tasks=(BenchmarkTask("id", "prompt"),),
                output=output,
                config={},
                limits=BudgetLimits(),
                secrets=(secret,),
            ).run(worker)
            artifacts = b"".join(
                path.read_bytes() for path in output.rglob("*") if path.is_file()
            )
            self.assertNotIn(secret.encode("utf-8"), artifacts)
            result = next(output.glob("instances/*/result.json")).read_text()
            self.assertIn("<redacted>", result)

    async def test_blocked_outcomes_are_committed_terminal_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            task = BenchmarkTask("id", "prompt")

            async def blocked(
                item: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                del context, directory
                return EvaluationOutcome(item.task_id, "blocked", error="prerequisite")

            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(task,),
                output=output,
                config={},
                limits=BudgetLimits(),
            )
            summary = await runner.run(blocked)
            self.assertEqual(summary["blocked"], 1)
            self.assertTrue(runner._valid_result(task.task_id))

            async def must_not_run(*args: Any) -> EvaluationOutcome:
                raise AssertionError(f"resume reran a blocked task: {args}")

            resumed = await runner.run(must_not_run, resume=True)
            self.assertEqual(resumed["blocked"], 1)

    async def test_resume_reruns_uncommitted_instance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            task = BenchmarkTask("id", "prompt")
            calls = 0

            async def worker(
                item: BenchmarkTask, context: RunContext, directory: Path
            ) -> EvaluationOutcome:
                nonlocal calls
                del context
                calls += 1
                self.assertFalse((directory / "stale.txt").exists())
                return EvaluationOutcome(item.task_id, "completed")

            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(task,),
                output=output,
                config={},
                limits=BudgetLimits(),
            )
            runner._prepare(runner._manifest(), resume=False)
            instance = runner._instance(task.task_id)
            instance.mkdir()
            (instance / "result.json").write_text(
                json.dumps({"task_id": "id", "status": "completed"})
            )
            (instance / "stale.txt").write_text("partial")
            await runner.run(worker, resume=True)
            self.assertEqual(calls, 1)
            self.assertTrue((instance / "completed.json").is_file())

    async def test_resume_rejects_a_symlinked_instances_root_without_deleting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            outside = root / "outside"
            task = BenchmarkTask("id", "prompt")
            runner = EvaluationRunner(
                benchmark="fixture",
                tasks=(task,),
                output=output,
                config={},
                limits=BudgetLimits(),
            )
            runner._prepare(runner._manifest(), resume=False)
            instances = output / "instances"
            instances.rmdir()
            instances.symlink_to(outside, target_is_directory=True)
            escaped = outside / hashlib.sha256(b"id").hexdigest()
            escaped.mkdir(parents=True)
            sentinel = escaped / "do-not-delete"
            sentinel.write_text("owned by another tree")

            async def must_not_run(*args: Any) -> EvaluationOutcome:
                raise AssertionError(f"unsafe resume ran worker: {args}")

            with self.assertRaisesRegex(ValueError, "instances root"):
                await runner.run(must_not_run, resume=True)
            self.assertEqual(sentinel.read_text(), "owned by another tree")

    async def test_resume_refuses_uncommitted_external_operation_evidence(self) -> None:
        for event in ("model_call_started", "tool_call_started"):
            with self.subTest(event=event), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "run"
                task = BenchmarkTask("id", "prompt")
                runner = EvaluationRunner(
                    benchmark="fixture",
                    tasks=(task,),
                    output=output,
                    config={},
                    limits=BudgetLimits(),
                )
                runner._prepare(runner._manifest(), resume=False)
                (output / "trace.jsonl").write_text(
                    json.dumps(
                        {
                            "event": event,
                            "elapsed_seconds": 1.0,
                            "agent_id": task_agent_root(task.task_id),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                async def must_not_run(*args: Any) -> EvaluationOutcome:
                    raise AssertionError(f"unsafe resume ran worker: {args}")

                with self.assertRaisesRegex(ValueError, "operation may already"):
                    await runner.run(must_not_run, resume=True)


def encrypt(value: str, password: str) -> str:
    raw = value.encode()
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(raw) // len(digest)) + digest[: len(raw) % len(digest)]
    return base64.b64encode(
        bytes(left ^ right for left, right in zip(raw, key))
    ).decode()


def commit_result(directory: Path, value: Mapping[str, Any]) -> None:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    (directory / "result.json").write_bytes(raw)
    (directory / "completed.json").write_text(
        json.dumps(
            {
                "task_id": value["task_id"],
                "result_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


class WebBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_browsecomp_plus_grader_inputs_bind_runs_to_hidden_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            (runs / "one.json").write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "status": "completed",
                        "tool_call_counts": {"search": 0, "get_document": 0},
                        "retrieved_docids": [],
                        "result": [{"type": "output_text", "output": "answer"}],
                    }
                )
            )
            truth = root / "truth.jsonl"
            truth.write_text(
                json.dumps({"query_id": "q1", "query": "question", "answer": "secret"})
                + "\n"
            )
            qrels = root / "qrels.txt"
            qrels.write_text("q1 0 document 1\n")
            identity = inspect_browsecomp_plus_grade_inputs(
                input_dir=runs,
                ground_truth=truth,
                qrel_evidence=qrels,
            )
            self.assertEqual(identity["run_count"], 1)
            self.assertEqual(
                identity["query_prompt_sha256"]["q1"],
                hashlib.sha256(
                    BROWSECOMP_PLUS_QUERY.format(Question="question").encode()
                ).hexdigest(),
            )
            self.assertNotIn("secret", json.dumps(identity))
            qrels.write_text("")
            with self.assertRaisesRegex(ValueError, "no qrel evidence"):
                inspect_browsecomp_plus_grade_inputs(
                    input_dir=runs,
                    ground_truth=truth,
                    qrel_evidence=qrels,
                )

    def test_browsecomp_and_plus_loaders_keep_grading_fields_off_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encrypted = root / "browsecomp.csv"
            canary = "canary"
            with encrypted.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["id", "canary", "problem", "answer"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "one",
                        "canary": canary,
                        "problem": encrypt("question", canary),
                        "answer": encrypt("secret answer", canary),
                    }
                )
            task = load_browsecomp(encrypted)[0]
            self.assertIn("question", task.prompt)
            self.assertNotIn("secret answer", task.prompt)
            self.assertEqual(task.data["answer"], "secret answer")

            stable = root / "browsecomp-stable.csv"
            with stable.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["canary", "problem", "answer"]
                )
                writer.writeheader()
                for index in range(3):
                    writer.writerow(
                        {
                            "canary": canary,
                            "problem": encrypt(f"question-{index}", canary),
                            "answer": encrypt(f"answer-{index}", canary),
                        }
                    )
            full_ids = {
                item.data["question"]: item.task_id for item in load_browsecomp(stable)
            }
            sampled = load_browsecomp(stable, limit=1, sample_seed=2)[0]
            self.assertEqual(sampled.task_id, full_ids[sampled.data["question"]])

            tsv = root / "queries.tsv"
            tsv.write_text(" q1 \t Where is alpha? \nq2\tWhere is beta?\n")
            plus = load_browsecomp_plus(tsv)
            self.assertEqual([item.task_id for item in plus], ["q1", "q2"])
            self.assertIn("Question: Where is alpha?", plus[0].prompt)
            self.assertIn("action=search", plus[0].prompt)
            self.assertNotIn("action=open", plus[0].prompt)
            self.assertIn("Exact Answer:", BROWSECOMP_PLUS_QUERY)
            self.assertTrue(
                BROWSECOMP_PLUS_QUERY.startswith(
                    "You are a deep research agent. You need to answer the given "
                    "question by interacting with a search engine, using the "
                    "browser tool with action=search provided."
                )
            )

            duplicate = root / "duplicate.tsv"
            duplicate.write_text("same\tone\nsame\ttwo\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_browsecomp_plus(duplicate)

            headed = root / "headed.tsv"
            headed.write_text("query_id\tquery\nq1\tone\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "headerless"):
                load_browsecomp_plus(headed)

    async def test_plus_run_artifact_matches_upstream_schema_and_collects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"docid":"1","text":"alpha evidence"}\n', encoding="utf-8"
            )
            task = BenchmarkTask(
                "q1",
                "find alpha",
                {"benchmark": "browsecomp-plus"},
            )
            model = ScriptedModel(
                [
                    ModelResponse(
                        "",
                        tool_calls=(
                            ToolCall(
                                "search",
                                "browser",
                                {"action": "search", "query": "alpha"},
                            ),
                        ),
                    ),
                    ModelResponse(
                        "Explanation: evidence [1]\n"
                        "Exact Answer: alpha\nConfidence: 90%"
                    ),
                ]
            )
            outcome = await run_web_task(
                task,
                RunContext(),
                root / "instances" / "task",
                browser_factory=lambda agent_id: BrowserEnvironment(
                    JsonlSearchBackend(corpus),
                    snippet_tokens=512,
                    tokenizer=WordTokenizer(),
                    max_observation_chars=None,
                    allow_open=False,
                ),
                model_factory=lambda agent_id: model,
                system_prompt="",
                max_steps=3,
                model_name="scripted/test",
            )
            self.assertEqual(outcome.status, "completed")
            artifact = json.loads(
                (root / "instances" / "task" / "browsecomp_plus_run.json").read_text()
            )
            self.assertEqual(
                set(artifact),
                {
                    "metadata",
                    "query_id",
                    "result",
                    "retrieved_docids",
                    "status",
                    "tool_call_counts",
                },
            )
            self.assertEqual(artifact["retrieved_docids"], ["1"])
            self.assertEqual(artifact["tool_call_counts"]["search"], 1)
            self.assertNotIn("get_document", artifact["tool_call_counts"])
            self.assertEqual(
                artifact["metadata"]["upstream_query_template"],
                "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            )
            self.assertEqual(
                artifact["result"][-1],
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": "Explanation: evidence [1]\n"
                    "Exact Answer: alpha\nConfidence: 90%",
                },
            )
            commit_result(
                root / "instances" / "task",
                {
                    "task_id": "q1",
                    "status": "completed",
                    "metadata": outcome.metadata,
                },
            )
            destination = root / "official"
            self.assertEqual(collect_browsecomp_plus_runs(root, destination), 1)
            self.assertEqual(len(list(destination.glob("*.json"))), 1)
            self.assertTrue((destination / "_index.tsv").is_file())

            run_path = root / "instances" / "task" / "browsecomp_plus_run.json"
            original_run = run_path.read_bytes()
            run_path.write_bytes(original_run + b"\n")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                collect_browsecomp_plus_runs(root, destination)
            run_path.write_bytes(original_run)

            stale_id = "stale"
            stale = destination / (
                hashlib.sha256(stale_id.encode()).hexdigest() + ".json"
            )
            stale.write_text(
                json.dumps(
                    {
                        "query_id": stale_id,
                        "tool_call_counts": {"search": 0, "get_document": 0},
                        "status": "completed",
                        "retrieved_docids": [],
                        "result": [{"type": "output_text", "output": "old"}],
                    }
                )
            )
            self.assertEqual(collect_browsecomp_plus_runs(root, destination), 1)
            self.assertFalse(stale.exists())

            unexpected = destination / "unexpected.json"
            unexpected.write_text(
                stale.read_text()
                if stale.exists()
                else json.dumps(
                    {
                        "query_id": stale_id,
                        "tool_call_counts": {"search": 0, "get_document": 0},
                        "status": "completed",
                        "retrieved_docids": [],
                        "result": [{"type": "output_text", "output": "old"}],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "unexpected JSON"):
                collect_browsecomp_plus_runs(root, destination)

    async def test_plus_generation_rejects_an_open_capability(self) -> None:
        class Search:
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return []

            def open(self, reference: str) -> None:
                del reference
                return None

        task = BenchmarkTask(
            "q1",
            BROWSECOMP_PLUS_QUERY.format(Question="question"),
            {"benchmark": "browsecomp-plus"},
        )
        with self.assertRaisesRegex(ValueError, "search-only"):
            await run_web_task(
                task,
                RunContext(),
                Path("unused"),
                browser_factory=lambda agent_id: BrowserEnvironment(Search()),
                model_factory=lambda agent_id: ScriptedModel(
                    [ModelResponse("answer")]
                ),
                system_prompt="",
                max_steps=1,
                model_name="scripted/test",
            )

    async def test_plus_generation_rejects_noncanonical_result_bounds(self) -> None:
        class Search:
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return []

        task = BenchmarkTask(
            "q1",
            BROWSECOMP_PLUS_QUERY.format(Question="question"),
            {"benchmark": "browsecomp-plus"},
        )
        with self.assertRaisesRegex(ValueError, "top-5, 512-token"):
            await run_web_task(
                task,
                RunContext(),
                Path("unused"),
                browser_factory=lambda agent_id: BrowserEnvironment(
                    Search(), allow_open=False
                ),
                model_factory=lambda agent_id: ScriptedModel(
                    [ModelResponse("answer")]
                ),
                system_prompt="",
                max_steps=1,
                model_name="scripted/test",
            )

    def test_plus_collection_rejects_symlink_destination_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            (root / "instances").mkdir(parents=True)
            outside = Path(temporary) / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.json"
            sentinel.write_text("do not delete")
            destination = root / "official_runs"
            destination.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                collect_browsecomp_plus_runs(root, destination)
            self.assertEqual(sentinel.read_text(), "do not delete")

    def test_official_plus_grader_requires_pinned_clean_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts_evaluation").mkdir()
            (root / "scripts_evaluation" / "evaluate_run.py").write_text("# grader\n")
            (root / "runs").mkdir()
            (root / "ground.jsonl").write_text("{}\n")
            (root / "qrels.txt").write_text("")
            with patch(
                "mini_agent.benchmarks.web._git",
                side_effect=[
                    "046949032b0328319cc9a02663a759ec601d9402",
                    "",
                    "",
                    "",
                ],
            ):
                argv = official_browsecomp_plus_grader_argv(
                    checkout=root,
                    input_dir=root / "runs",
                    ground_truth=root / "ground.jsonl",
                    eval_dir=root / "evals",
                    qrel_evidence=root / "qrels.txt",
                )
            self.assertEqual(argv[1], "-I")
            self.assertIn("scripts_evaluation/evaluate_run.py", argv[2])
            self.assertIn("--tensor_parallel_size", argv)

    def test_official_plus_grader_rejects_ignored_or_untracked_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts_evaluation"
            scripts.mkdir()
            (scripts / "evaluate_run.py").write_text("# grader\n")
            injected = root / "ignored_helper.py"
            injected.write_text("raise RuntimeError('injected')\n")
            (root / "runs").mkdir()
            (root / "ground.jsonl").write_text("{}\n")
            (root / "qrels.txt").write_text("")
            with patch(
                "mini_agent.benchmarks.web._git",
                side_effect=[
                    "046949032b0328319cc9a02663a759ec601d9402",
                    "",
                    "ignored_helper.py\x00",
                    "ignored_helper.py\x00",
                ],
            ) as git:
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    official_browsecomp_plus_grader_argv(
                        checkout=root,
                        input_dir=root / "runs",
                        ground_truth=root / "ground.jsonl",
                        eval_dir=root / "evals",
                        qrel_evidence=root / "qrels.txt",
                    )
            git.assert_any_call(root.resolve(), "ls-files", "--others", "-z")

    def test_official_plus_grader_rejects_untracked_bytecode_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts_evaluation"
            scripts.mkdir()
            (scripts / "evaluate_run.py").write_text("# grader\n")
            cache = root / "module" / "__pycache__"
            cache.mkdir(parents=True)
            cached = cache / "module.cpython-312.pyc"
            cached.write_bytes(b"generated")
            (root / "runs").mkdir()
            (root / "ground.jsonl").write_text("{}\n")
            (root / "qrels.txt").write_text("")
            with patch(
                "mini_agent.benchmarks.web._git",
                side_effect=[
                    "046949032b0328319cc9a02663a759ec601d9402",
                    "",
                    "module/__pycache__/module.cpython-312.pyc\x00",
                    "module/__pycache__/module.cpython-312.pyc\x00",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    official_browsecomp_plus_grader_argv(
                        checkout=root,
                        input_dir=root / "runs",
                        ground_truth=root / "ground.jsonl",
                        eval_dir=root / "evals",
                        qrel_evidence=root / "qrels.txt",
                    )

    def test_official_plus_grader_rejects_untracked_listing_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts_evaluation"
            scripts.mkdir()
            (scripts / "evaluate_run.py").write_text("# grader\n")
            (root / "runs").mkdir()
            (root / "ground.jsonl").write_text("{}\n")
            (root / "qrels.txt").write_text("")
            with patch(
                "mini_agent.benchmarks.web._git",
                side_effect=[
                    "046949032b0328319cc9a02663a759ec601d9402",
                    "",
                    "",
                    "appeared.py\x00",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    official_browsecomp_plus_grader_argv(
                        checkout=root,
                        input_dir=root / "runs",
                        ground_truth=root / "ground.jsonl",
                        eval_dir=root / "evals",
                        qrel_evidence=root / "qrels.txt",
                    )


class FakeSWEEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self.closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition("bash"),)

    async def execute(self, action: ToolCall) -> ToolExecution:
        return ToolExecution("ok")

    async def export_patch(self) -> bytes:
        return b"diff --git a/a b/a\n"

    def provenance(self) -> Mapping[str, Any]:
        return {"container_image_id": "sha256:image"}

    async def close(self) -> None:
        self.closed = True


class SWEBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_official_grader_images_are_exact_docker_generation_bindings(self) -> None:
        task_id = "Repo__Project-1"
        image_id = "sha256:" + "a" * 64
        manifest = {
            "config": {
                "adapter": {
                    "runtime": "docker",
                    "container_runtime": ["/artifact-controlled-runtime"],
                    "image_bindings": {
                        task_id: {
                            "runtime": "docker",
                            "requested": "fixture/image:latest",
                            "identity": image_id,
                        }
                    },
                }
            },
            "tasks": [{"id": task_id}],
        }
        self.assertEqual(
            swebench_grader_image_name(task_id),
            "swebench/sweb.eval.x86_64.repo_1776_project-1:latest",
        )
        environment = {
            "DOCKER_HOST": "unix:///fixture-docker.sock",
            "HOME": "/private-grade",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            docker_root = Path(temporary) / "docker"
            docker_root.mkdir()
            docker_origin = docker_root / "__init__.py"
            docker_origin.write_text("# isolated fixture\n")

            def run_probe(
                argv: Sequence[str], **kwargs: Any
            ) -> SimpleNamespace:
                request = json.loads(Path(argv[4]).read_text())
                observed = [
                    {
                        "grader_image": item["grader_image"],
                        "image_id": item["expected_image_id"],
                    }
                    for item in request["images"]
                ]
                output = Path(argv[5])
                output.write_text(
                    json.dumps(
                        {
                            "schema": "mini-agent-isolated-swebench-images-v1",
                            "ok": True,
                            "python": {
                                "executable": str(Path(sys.executable).absolute()),
                                "prefix": str(Path(sys.prefix).absolute()),
                                "base_prefix": str(Path(sys.base_prefix).absolute()),
                            },
                            "docker_sdk": {
                                "version": "7.fixture",
                                "origin": str(docker_origin.resolve()),
                                "package_root": str(docker_root.resolve()),
                            },
                            "images": observed,
                        }
                    )
                )
                output.chmod(0o600)
                return SimpleNamespace(returncode=0)

            parent_from_env = Mock(
                side_effect=AssertionError("parent Docker SDK must not be used")
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "docker": SimpleNamespace(
                            __version__="parent-shadow",
                            from_env=parent_from_env,
                        )
                    },
                ),
                patch(
                    "mini_agent.benchmarks.swebench.subprocess.run",
                    side_effect=run_probe,
                ) as invoked,
            ):
                identity = verify_swebench_grader_images(
                    manifest,
                    python_executable=sys.executable,
                    grader_environment=environment,
                )
            parent_from_env.assert_not_called()
        self.assertEqual(identity["images"][0]["image_id"], image_id)
        self.assertEqual(
            identity["engine_contract"], "isolated-python-I:docker.from_env"
        )
        self.assertEqual(identity["docker_sdk_version"], "7.fixture")
        self.assertEqual(
            identity["generation_container_runtime"],
            ["/artifact-controlled-runtime"],
        )
        child_argv = invoked.call_args.args[0]
        child_kwargs = invoked.call_args.kwargs
        self.assertEqual(tuple(child_argv[1:3]), ("-I", "-c"))
        self.assertEqual(child_kwargs["env"], environment)
        self.assertIs(child_kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(child_kwargs["stderr"], subprocess.DEVNULL)

        def changed_probe(argv: Sequence[str], **_kwargs: Any) -> SimpleNamespace:
            request = json.loads(Path(argv[4]).read_text())
            output = Path(argv[5])
            output.write_text(
                json.dumps(
                    {
                        "schema": "mini-agent-isolated-swebench-images-v1",
                        "ok": False,
                        "error": "changed",
                        "image": request["images"][0]["grader_image"],
                    }
                )
            )
            output.chmod(0o600)
            return SimpleNamespace(returncode=0)

        with patch(
            "mini_agent.benchmarks.swebench.subprocess.run",
            side_effect=changed_probe,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed identity"):
                verify_swebench_grader_images(
                    manifest,
                    python_executable=sys.executable,
                    grader_environment=environment,
                )

        manifest["config"]["adapter"]["runtime"] = "apptainer"
        with self.assertRaisesRegex(ValueError, "requires Docker generation"):
            identity = verify_swebench_grader_images(
                manifest,
                python_executable=sys.executable,
                grader_environment=environment,
            )

        manifest["config"]["adapter"]["runtime"] = "docker"
        with self.assertRaisesRegex(ValueError, "explicit DOCKER_HOST"):
            verify_swebench_grader_images(
                manifest,
                python_executable=sys.executable,
                grader_environment={"HOME": "/private-grade"},
            )

    def test_grader_source_identity_rejects_mutated_imported_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "swebench"
            harness = package / "harness"
            resources = package / "resources"
            harness.mkdir(parents=True)
            resources.mkdir()
            init = package / "__init__.py"
            source = harness / "run_evaluation.py"
            resource_init = resources / "__init__.py"
            environment = resources / "environment.yml"
            init.write_text('__version__ = "4.1.0"\n')
            source.write_text("def main():\n    return 0\n")
            resource_init.write_text("# package data\n")
            environment.write_text("name: testbed\n")
            files = []
            for path in (init, source, resource_init, environment):
                content = path.read_bytes()
                files.append(
                    {
                        "path": path.relative_to(package).as_posix(),
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            expected = hashlib.sha256(
                json.dumps(
                    files,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            with (
                patch(
                    "mini_agent.benchmarks.swebench.SWEBENCH_SOURCE_SHA256",
                    expected,
                ),
                patch(
                    "mini_agent.benchmarks.swebench.SWEBENCH_SOURCE_FILE_COUNT",
                    len(files),
                ),
                patch(
                    "mini_agent.benchmarks.swebench.SWEBENCH_SOURCE_SIZE_BYTES",
                    sum(item["size_bytes"] for item in files),
                ),
            ):
                identity = swebench_grader_source_identity(package.resolve())
                self.assertEqual(identity["source_sha256"], expected)
                self.assertEqual(identity["source_file_count"], 4)
                environment.write_text("name: mutated\n")
                with self.assertRaisesRegex(RuntimeError, "pinned v4.1.0"):
                    swebench_grader_source_identity(package.resolve())
                environment.unlink()
                with self.assertRaisesRegex(RuntimeError, "pinned v4.1.0"):
                    swebench_grader_source_identity(package.resolve())
        self.assertEqual(len(SWEBENCH_SOURCE_SHA256), 64)

    def test_grader_source_identity_rejects_a_symlink_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "source" / "swebench"
            (package / "harness").mkdir(parents=True)
            (package / "__init__.py").write_text('__version__ = "4.1.0"\n')
            linked = root / "linked-swebench"
            linked.symlink_to(package, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "source tree is invalid"):
                swebench_grader_source_identity(linked)

    def test_local_json_array_is_an_exact_swebench_grader_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "instance_id": "repo__issue-1",
                            "problem_statement": "fix it",
                        }
                    ]
                )
            )
            tasks = load_swebench(dataset)
            self.assertEqual(tasks[0].task_id, "repo__issue-1")
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__issue-1",
                        "model_patch": "",
                        "model_name_or_path": "fixture/model",
                    }
                )
                + "\n"
            )
            identity = inspect_swebench_grade_inputs(
                predictions=predictions, dataset=dataset
            )
            self.assertEqual(identity["prediction_count"], 1)
            self.assertEqual(identity["dataset_count"], 1)

            dataset.write_text(json.dumps({"instance_id": "repo__issue-1"}))
            with self.assertRaisesRegex(ValueError, "must contain an array"):
                inspect_swebench_grade_inputs(predictions=predictions, dataset=dataset)

            dataset.write_text(
                json.dumps(
                    [
                        {
                            "instance_id": "../escape",
                            "problem_statement": "fix it",
                        }
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "path-safe component"):
                load_swebench(dataset)

    def test_resume_rejects_a_changed_swe_image_binding(self) -> None:
        task = BenchmarkTask("repo__issue-1", "fix it")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = EvaluationRunner(
                benchmark="swebench",
                tasks=(task,),
                output=output,
                config={
                    "adapter": {
                        "image_bindings": {
                            task.task_id: {"identity": "sha256:" + "a" * 64}
                        }
                    }
                },
                limits=BudgetLimits(),
            )
            first._prepare(first._manifest(), resume=False)
            changed = EvaluationRunner(
                benchmark="swebench",
                tasks=(task,),
                output=output,
                config={
                    "adapter": {
                        "image_bindings": {
                            task.task_id: {"identity": "sha256:" + "b" * 64}
                        }
                    }
                },
                limits=BudgetLimits(),
            )
            with self.assertRaisesRegex(ValueError, "resume manifest"):
                changed._prepare(changed._manifest(), resume=True)

    async def test_image_preflight_deduplicates_and_binds_each_task(self) -> None:
        tasks = (
            BenchmarkTask(
                "repo__one",
                "one",
                {"instance_id": "repo__one", "image_name": "repo/image:latest"},
            ),
            BenchmarkTask(
                "repo__two",
                "two",
                {"instance_id": "repo__two", "image_name": "repo/image:latest"},
            ),
        )
        binding = SWEbenchImageBinding(
            runtime="docker",
            requested="repo/image:latest",
            identity="sha256:" + "a" * 64,
            execution_ref="sha256:" + "a" * 64,
        )
        resolver = AsyncMock(return_value=binding)
        with patch(
            "mini_agent.benchmarks.swebench.resolve_swebench_image_binding",
            resolver,
        ):
            bindings = await prepare_swebench_image_bindings(
                tasks,
                runtime="docker",
            )
        self.assertEqual(bindings, {"repo__one": binding, "repo__two": binding})
        resolver.assert_awaited_once()

    async def test_generation_is_separate_from_official_grading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            environment = FakeSWEEnvironment()
            task = BenchmarkTask(
                "repo__issue-1",
                "fix it",
                {"instance_id": "repo__issue-1", "problem_statement": "fix it"},
            )
            binding = SWEbenchImageBinding(
                runtime="docker",
                requested=(
                    "docker.io/swebench/sweb.eval.x86_64.repo_1776_issue-1:latest"
                ),
                identity="sha256:" + "a" * 64,
                execution_ref="sha256:" + "a" * 64,
            )
            with patch(
                "mini_agent.benchmarks.swebench.DockerSWEEnvironment.create",
                AsyncMock(return_value=environment),
            ) as create:
                outcome = await run_swebench_task(
                    task,
                    RunContext(),
                    directory / "instances" / "one",
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=2,
                    runtime="docker",
                    model_name="scripted/model",
                    image_binding=binding,
                )
            self.assertIs(create.await_args.kwargs["image_binding"], binding)
            self.assertEqual(outcome.status, "completed")
            instance_directory = directory / "instances" / "one"
            commit_result(
                instance_directory,
                {
                    "task_id": "repo__issue-1",
                    "status": "completed",
                    "metadata": outcome.metadata,
                },
            )
            predictions = directory / "predictions.jsonl"
            self.assertEqual(collect_predictions(directory, predictions), 1)
            value = json.loads(predictions.read_text())
            self.assertEqual(value["model_name_or_path"], "scripted/model")

            predictions.write_text('{"instance_id":"stale"}\n')
            self.assertEqual(collect_predictions(directory, predictions), 1)
            self.assertEqual(
                json.loads(predictions.read_text())["instance_id"],
                "repo__issue-1",
            )
            dataset_path = str(predictions.parent / "dataset.jsonl")
            argv = official_grader_argv(
                predictions=predictions,
                dataset_name=dataset_path,
                run_id="canary",
            )
            self.assertEqual(
                argv[1:4], ("-I", "-m", "swebench.harness.run_evaluation")
            )
            self.assertIn(dataset_path, argv)
            with self.assertRaisesRegex(ValueError, "path-safe component"):
                official_grader_argv(
                    predictions=predictions,
                    dataset_name=dataset_path,
                    run_id="../escape",
                )
            self.assertTrue(environment.closed)

            prediction_path = instance_directory / "prediction.json"
            prediction_path.write_bytes(prediction_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                collect_predictions(directory, predictions)

    def test_prediction_collection_rejects_symlink_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "evaluation"
            instance = root / "instances" / "one"
            instance.mkdir(parents=True)
            prediction = {
                "instance_id": "repo__issue-1",
                "model_patch": "",
                "model_name_or_path": "fixture/model",
            }
            prediction_path = instance / "prediction.json"
            prediction_path.write_text(json.dumps(prediction))
            commit_result(
                instance,
                {
                    "task_id": "repo__issue-1",
                    "status": "completed",
                    "metadata": {
                        "prediction_sha256": hashlib.sha256(
                            prediction_path.read_bytes()
                        ).hexdigest()
                    },
                },
            )
            sentinel = base / "do-not-overwrite"
            sentinel.write_text("owned by another path")
            destination = root / "predictions.jsonl"
            destination.symlink_to(sentinel)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                collect_predictions(root, destination)
            self.assertEqual(sentinel.read_text(), "owned by another path")

    async def test_generation_preserves_agent_and_cleanup_failures(self) -> None:
        class BrokenEnvironment(FakeSWEEnvironment):
            async def close(self) -> None:
                raise RuntimeError("close exploded")

        environment = BrokenEnvironment()
        task = BenchmarkTask(
            "repo__issue-1",
            "fix it",
            {"instance_id": "repo__issue-1", "problem_statement": "fix it"},
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "mini_agent.benchmarks.swebench.DockerSWEEnvironment.create",
                AsyncMock(return_value=environment),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "scripted model has no response.*close exploded"
            ):
                await run_swebench_task(
                    task,
                    RunContext(),
                    Path(temporary),
                    model_factory=lambda agent_id: ScriptedModel([]),
                    system_prompt="",
                    max_steps=1,
                    runtime="docker",
                    model_name="scripted/model",
                )


class FakeDesktop:
    def __init__(
        self,
        evaluator_result: Any = 0.5,
        close_error: BaseException | None = None,
    ) -> None:
        self.closed = False
        self.actions: list[str] = []
        self.evaluator_result = evaluator_result
        self.close_error = close_error

    def reset(self, *, task_config: Any) -> Mapping[str, Any]:
        self.task_config = task_config
        return {"screenshot": png()}

    def step(self, action: str, pause: float) -> tuple[Any, ...]:
        self.actions.append(action)
        return ({"screenshot": png()}, 0, False, {})

    def evaluate(self) -> Any:
        if isinstance(self.evaluator_result, BaseException):
            raise self.evaluator_result
        return self.evaluator_result

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class OSWorldBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_settles_and_refresh_are_injected_and_ordered(
        self,
    ) -> None:
        events: list[str] = []

        class Desktop:
            def reset(self, *, task_config: Any) -> Mapping[str, Any]:
                del task_config
                events.append("reset")
                return {"screenshot": png()}

            def _get_obs(self) -> Mapping[str, Any]:
                events.append("refresh")
                return {"screenshot": png(9, 7)}

            def step(self, action: str, pause: float) -> tuple[Any, ...]:
                del action, pause
                return ({"screenshot": png(9, 7)}, 0.0, False, {})

            def close(self) -> None:
                events.append("close")

        class Factory:
            initial_settle_seconds = 60.0
            evaluation_settle_seconds = 20.0
            refresh_initial_observation = True

            async def sleep(self, seconds: float) -> None:
                events.append(f"sleep:{seconds:g}")

            def __call__(self, agent_id: str, cache: Path) -> Desktop:
                del agent_id, cache
                return Desktop()

        with tempfile.TemporaryDirectory() as temporary:
            pool = _DesktopPool(
                task=BenchmarkTask("id", "task", {"version": "v1"}),
                directory=Path(temporary),
                desktop_factory=Factory(),
            )
            with patch(
                "mini_agent.benchmarks.osworld._task_config_for_benchmark",
                return_value={},
            ):
                environment = await pool.environment("/root")
            initial = await environment.initial_observation()
            self.assertIsNotNone(initial)
            self.assertEqual(json.loads(initial.output)["width"], 9)
            await pool.settle_before_evaluation()
            await pool.close()
        self.assertEqual(
            events,
            ["reset", "sleep:60", "refresh", "sleep:20", "close"],
        )

    def test_v2_special_lifecycles_fail_closed(self) -> None:
        task = BenchmarkTask("id", "task", {"version": "v2"})
        with self.assertRaisesRegex(NotImplementedError, "multi-phase"):
            _reject_unsupported_v2_lifecycle(
                task,
                SimpleNamespace(get_phases=lambda: [{"instruction": "phase"}]),
            )
        with self.assertRaisesRegex(NotImplementedError, "user-simulator"):
            _reject_unsupported_v2_lifecycle(
                task,
                {"user_simulator": {"type": "fixed", "response": "yes"}},
            )

    async def test_apptainer_factory_routes_only_its_constructor_thread(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "Ubuntu.qcow2"
            image.write_bytes(b"qcow2 fixture")
            sif = root / "osworld.sif"
            sif.write_bytes(b"sif fixture")
            original_client = object()
            routed_client = object()
            docker_module = SimpleNamespace(from_env=lambda: original_client)
            original_from_env = docker_module.from_env
            info = SimpleNamespace(
                path=root,
                version="v1",
                revision="pinned",
                as_dict=lambda: {
                    "path": str(root),
                    "version": "v1",
                    "revision": "pinned",
                    "dirty": False,
                },
            )

            def desktop(path_to_vm: str) -> tuple[Any, Mapping[str, Any]]:
                return docker_module.from_env(), {"path_to_vm": path_to_vm}

            with patch(
                "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                return_value=info,
            ):
                factory = UpstreamDesktopFactory(
                    root,
                    version="v1",
                    provider_name="docker",
                    path_to_vm=str(image),
                    apptainer_image=sif,
                    apptainer_executable="apptainer-test",
                )
            with (
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
                patch(
                    "mini_agent.benchmarks.osworld._desktop_env_class",
                    return_value=desktop,
                ),
                patch(
                    "mini_agent.environments.osworld_apptainer."
                    "OSWorldApptainerDockerClient",
                    return_value=routed_client,
                ),
                patch(
                    "mini_agent.benchmarks.osworld.importlib.import_module",
                    return_value=docker_module,
                ),
            ):
                selected, keywords = await factory("/root", root / "cache")

            self.assertIs(selected, routed_client)
            self.assertEqual(keywords["path_to_vm"], str(image.resolve()))
            self.assertIs(docker_module.from_env, original_from_env)
            provenance = factory.provenance()
            self.assertEqual(provenance["container_runtime"], "apptainer")
            self.assertEqual(
                provenance["apptainer_image"]["sha256"],
                hashlib.sha256(sif.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["runtime_adaptation"]["network"],
                "qemu-user-hostfwd",
            )

    def test_docker_factory_hashes_the_selected_vm_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "Ubuntu.qcow2"
            image.write_bytes(b"qcow2 fixture")
            info = SimpleNamespace(
                path=root,
                version="v1",
                revision="pinned",
                as_dict=lambda: {
                    "path": str(root),
                    "version": "v1",
                    "revision": "pinned",
                    "dirty": False,
                },
            )
            with patch(
                "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                return_value=info,
            ):
                factory = UpstreamDesktopFactory(
                    root,
                    version="v1",
                    provider_name="docker",
                    path_to_vm=str(image),
                )
            identity = factory.provenance()["vm_image"]
            self.assertEqual(identity["size_bytes"], len(b"qcow2 fixture"))
            self.assertEqual(
                identity["sha256"], hashlib.sha256(image.read_bytes()).hexdigest()
            )
            self.assertEqual(
                factory.provenance()["reference_timing"],
                {
                    "initial_settle_seconds": 60.0,
                    "fresh_observation_after_initial_settle": True,
                    "evaluation_settle_seconds": 20.0,
                    "step_pause_seconds": 0,
                },
            )

            with patch(
                "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                return_value=SimpleNamespace(
                    path=root,
                    version="v2",
                    revision="pinned",
                    as_dict=info.as_dict,
                ),
            ):
                v2_factory = UpstreamDesktopFactory(
                    root,
                    version="v2",
                    provider_name="docker",
                    path_to_vm=str(image),
                )
            self.assertEqual(
                v2_factory.provenance()["reference_timing"][
                    "evaluation_settle_seconds"
                ],
                0.0,
            )

    async def test_factory_rejects_runtime_image_changes_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "Ubuntu.qcow2"
            image.write_bytes(b"original vm")
            sif = root / "osworld.sif"
            sif.write_bytes(b"original sif")
            info = SimpleNamespace(
                path=root,
                version="v1",
                revision="pinned",
                as_dict=lambda: {
                    "path": str(root),
                    "version": "v1",
                    "revision": "pinned",
                    "dirty": False,
                },
            )

            with patch(
                "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                return_value=info,
            ):
                vm_factory = UpstreamDesktopFactory(
                    root,
                    version="v1",
                    provider_name="docker",
                    path_to_vm=str(image),
                )
                apptainer_factory = UpstreamDesktopFactory(
                    root,
                    version="v1",
                    provider_name="docker",
                    path_to_vm=str(image),
                    apptainer_image=sif,
                )

            image.write_bytes(b"mutated vm")
            with (
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
                self.assertRaisesRegex(RuntimeError, "VM image changed"),
            ):
                await vm_factory("/root", root / "vm-cache")

            image.write_bytes(b"original vm")
            sif.write_bytes(b"mutated sif")
            with (
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
                self.assertRaisesRegex(RuntimeError, "Apptainer image changed"),
            ):
                await apptainer_factory("/root", root / "sif-cache")

    async def test_osworld_revalidates_checkout_before_lease_and_after_cleanup(
        self,
    ) -> None:
        events: list[str] = []

        class Factory:
            initial_settle_seconds = 0.0
            evaluation_settle_seconds = 0.0
            refresh_initial_observation = False

            def __init__(self) -> None:
                self.checks = 0

            async def verify_checkout(self) -> None:
                self.checks += 1
                events.append(f"check:{self.checks}")
                if self.checks == 2:
                    raise RuntimeError("tracked provider changed")

            async def __call__(self, agent_id: str, cache: Path) -> FakeDesktop:
                del agent_id, cache
                await self.verify_checkout()
                return FakeDesktop(0.5)

        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = checkout / "evaluation_examples/examples/writer/task.json"
            config.parent.mkdir(parents=True)
            task_config = {
                "id": "task",
                "instruction": "inspect",
                "related_apps": ["writer"],
            }
            config.write_text(json.dumps(task_config))
            task = BenchmarkTask(
                "task",
                "inspect",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            factory = Factory()
            checkout_info = SimpleNamespace(
                path=checkout, version="v1", revision="pinned"
            )
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=checkout_info,
                ),
                self.assertRaisesRegex(RuntimeError, "tracked provider changed"),
            ):
                await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=factory,
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=1,
                )
            self.assertEqual(events, ["check:1", "check:2"])
            self.assertFalse((checkout / "run/score.json").exists())

    def test_v2_task_class_source_is_bound_into_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            base = checkout / "evaluation_examples"
            task_class = base / "task_class" / "writer" / "task_id.py"
            task_class.parent.mkdir(parents=True)
            task_class.write_text("ORIGINAL = True\n")
            task_list = base / "test_v2.json"
            task_list.write_text('{"writer": ["id"]}')
            config = {"instruction": "write"}
            loader = SimpleNamespace(
                find_task_class_path=lambda **kwargs: str(task_class),
                resolve_task_json_path=lambda **kwargs: str(base / "unused.json"),
                load_task_config=lambda *args, **kwargs: config,
            )
            info = SimpleNamespace(
                path=checkout,
                version="v2",
                revision="pinned",
            )
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
                patch("mini_agent.benchmarks.osworld._activate_checkout"),
                patch(
                    "mini_agent.benchmarks.osworld._import_from_checkout",
                    return_value=loader,
                ),
            ):
                task = load_osworld(
                    checkout,
                    version="v2",
                    task_list=task_list,
                )[0]
                self.assertEqual(
                    task.data["task_class_sha256"],
                    hashlib.sha256(task_class.read_bytes()).hexdigest(),
                )
                task_class.write_text("CHANGED = True\n")
                with self.assertRaisesRegex(RuntimeError, "class changed"):
                    _task_config_for_benchmark(task)

    def test_loader_rejects_path_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            task_list = checkout / "tasks.json"
            task_list.write_text('{"../outside": ["task"]}')
            info = SimpleNamespace(
                path=checkout,
                version="v1",
                revision="pinned",
            )
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "safe path component"):
                    load_osworld(checkout, version="v1", task_list=task_list)

    def test_v2_checkout_requires_the_release_tag_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=[
                    "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6",
                    "",
                    "v2026.06.24",
                    "",
                    "",
                ],
            ):
                info = inspect_osworld_checkout(checkout, version="v2")
            self.assertEqual(info.revision, "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6")

            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=["0" * 40, "", "v2026.06.24"],
            ):
                with self.assertRaisesRegex(ValueError, "at 2b9b7b4"):
                    inspect_osworld_checkout(checkout, version="v2")

    def test_v1_checkout_requires_the_exact_upstream_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=[
                    "091f5ef1d5544bc74953c77875d5feb5bed30108",
                    "",
                    "",
                    "",
                ],
            ):
                info = inspect_osworld_checkout(checkout, version="v1")
            self.assertEqual(info.revision, "091f5ef1d5544bc74953c77875d5feb5bed30108")

    def test_osworld_rejects_untracked_executable_or_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            injected = checkout / "sitecustomize.py"
            injected.write_text("raise RuntimeError('injected')\n")
            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=[
                    "091f5ef1d5544bc74953c77875d5feb5bed30108",
                    "",
                    "sitecustomize.py\x00",
                    "sitecustomize.py\x00",
                ],
            ) as git:
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    inspect_osworld_checkout(checkout, version="v1")
            git.assert_any_call(checkout.resolve(), "ls-files", "--others", "-z")

    def test_osworld_rejects_untracked_python_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            bytecode = checkout / "desktop_env" / "__pycache__" / "desktop_env.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"untrusted-bytecode")
            relative = "desktop_env/__pycache__/desktop_env.pyc\x00"
            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=[
                    "091f5ef1d5544bc74953c77875d5feb5bed30108",
                    "",
                    relative,
                    relative,
                ],
            ):
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    inspect_osworld_checkout(checkout, version="v1")

    def test_osworld_v2_allows_only_canonical_gated_task_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            gated = checkout / "evaluation_examples" / "task_class" / "task_001.py"
            gated.parent.mkdir(parents=True)
            gated.write_text("class Task: pass\n")
            success = [
                "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6",
                "",
                "v2026.06.24",
                "evaluation_examples/task_class/task_001.py\x00",
                "evaluation_examples/task_class/task_001.py\x00",
            ]
            with patch("mini_agent.benchmarks.osworld._git", side_effect=success):
                inspect_osworld_checkout(checkout, version="v2")

            noncanonical = checkout / "task_loader.py"
            noncanonical.write_text("raise RuntimeError('injected')\n")
            failure = [
                "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6",
                "",
                "v2026.06.24",
                "evaluation_examples/task_class/task_001.py\x00task_loader.py\x00",
                "evaluation_examples/task_class/task_001.py\x00task_loader.py\x00",
            ]
            with patch("mini_agent.benchmarks.osworld._git", side_effect=failure):
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    inspect_osworld_checkout(checkout, version="v2")

    def test_osworld_rejects_untracked_listing_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            desktop = checkout / "desktop_env" / "desktop_env.py"
            desktop.parent.mkdir()
            desktop.write_text("class DesktopEnv: pass\n")
            with patch(
                "mini_agent.benchmarks.osworld._git",
                side_effect=[
                    "091f5ef1d5544bc74953c77875d5feb5bed30108",
                    "",
                    "",
                    "appeared.py\x00",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    inspect_osworld_checkout(checkout, version="v1")

    async def test_hidden_evaluator_runs_after_agent_on_selected_desktop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {
                "id": "task",
                "instruction": "click",
                "related_apps": ["writer"],
            }
            config.write_text(json.dumps(task_config))
            desktop = FakeDesktop({"score": 0.5, "criteria": {"document_saved": True}})

            async def factory(agent_id: str, cache: Path) -> FakeDesktop:
                del agent_id, cache
                return desktop

            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            model = ScriptedModel(
                [
                    ModelResponse(
                        "",
                        tool_calls=(
                            ToolCall(
                                "click",
                                "computer",
                                {"actions": [{"type": "click", "x": 1, "y": 1}]},
                            ),
                        ),
                    ),
                    ModelResponse("done"),
                ]
            )
            checkout_info = SimpleNamespace(
                path=checkout,
                version="v1",
                revision="pinned",
            )
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=checkout_info,
                ),
            ):
                outcome = await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=factory,
                    model_factory=lambda agent_id: model,
                    system_prompt="",
                    max_steps=3,
                )
            self.assertEqual(outcome.score, 0.5)
            self.assertEqual(outcome.metadata["score_scale"], "0-1")
            self.assertEqual(
                outcome.metadata["evaluator_result"]["criteria"],
                {"document_saved": True},
            )
            self.assertFalse(outcome.metadata["verifier_exposed_to_agent"])
            self.assertEqual(
                outcome.metadata["score_sha256"],
                hashlib.sha256(
                    (checkout / "run" / "score.json").read_bytes()
                ).hexdigest(),
            )
            self.assertTrue(desktop.closed)
            trajectory = checkout / "run" / "branches"
            self.assertEqual(len(list(trajectory.glob("*/trajectory.jsonl"))), 1)

    async def test_multi_agent_records_actual_osworld_state_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {"id": "task", "instruction": "click"}
            config.write_text(json.dumps(task_config))
            desktops: list[FakeDesktop] = []

            def factory(agent_id: str, cache: Path) -> FakeDesktop:
                del agent_id, cache
                desktop = FakeDesktop(float(len(desktops) + 1) / 10.0)
                desktops.append(desktop)
                return desktop

            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            root_id: str | None = None

            def agent_call(
                call_id: str, arguments: Mapping[str, Any]
            ) -> ModelResponse:
                return ModelResponse(
                    "", tool_calls=(ToolCall(call_id, "agent", arguments),)
                )

            def model(agent_id: str) -> ScriptedModel:
                nonlocal root_id
                if agent_id.endswith("/root"):
                    root_id = agent_id
                    child = agent_id + "/1"
                    return ScriptedModel(
                        [
                            agent_call(
                                "spawn", {"action": "spawn", "task": "child"}
                            ),
                            agent_call(
                                "wait", {"action": "wait", "agent_ids": [child]}
                            ),
                            agent_call(
                                "adopt", {"action": "adopt", "agent_id": child}
                            ),
                            ModelResponse("selected"),
                        ]
                    )
                if root_id is not None and agent_id == root_id + "/1":
                    return ScriptedModel([ModelResponse("candidate")])
                raise AssertionError(f"unexpected agent id {agent_id!r}")

            info = SimpleNamespace(path=checkout, version="v1", revision="pinned")
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
            ):
                outcome = await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=factory,
                    model_factory=model,
                    system_prompt="",
                    max_steps=4,
                    multi_agent=True,
                    max_active_agents=2,
                    max_total_agents=2,
                )
        assert root_id is not None
        self.assertEqual(outcome.score, 0.2)
        self.assertEqual(
            outcome.metadata["state_selection"],
            "adopted_descendant_environment",
        )
        self.assertEqual(
            outcome.metadata["state_adoption_history"], [root_id + "/1"]
        )

    async def test_multi_agent_osworld_without_adoption_selects_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {"id": "task", "instruction": "click"}
            config.write_text(json.dumps(task_config))
            desktop = FakeDesktop(0.3)
            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            info = SimpleNamespace(path=checkout, version="v1", revision="pinned")
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
            ):
                outcome = await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=lambda agent_id, cache: desktop,
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=1,
                    multi_agent=True,
                    max_active_agents=1,
                    max_total_agents=1,
                )
        self.assertEqual(outcome.metadata["state_selection"], "root_environment")
        self.assertEqual(outcome.metadata["state_adoption_history"], [])

    async def test_agent_failure_does_not_run_osworld_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {"id": "task", "instruction": "click"}
            config.write_text(json.dumps(task_config))

            class Desktop(FakeDesktop):
                def evaluate(self) -> Any:
                    raise AssertionError("evaluator must not run after agent failure")

            desktop = Desktop()
            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            info = SimpleNamespace(path=checkout, version="v1", revision="pinned")
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
                self.assertRaisesRegex(AssertionError, "no response left"),
            ):
                await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=lambda agent_id, cache: desktop,
                    model_factory=lambda agent_id: ScriptedModel([]),
                    system_prompt="",
                    max_steps=1,
                )
            self.assertTrue(desktop.closed)

    async def test_step_budget_exhaustion_still_scores_osworld_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {"id": "task", "instruction": "click"}
            config.write_text(json.dumps(task_config))
            desktop = FakeDesktop(0.25)
            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            model = ScriptedModel(
                [
                    ModelResponse(
                        "",
                        tool_calls=(
                            ToolCall(
                                "click",
                                "computer",
                                {"actions": [{"type": "click", "x": 1, "y": 1}]},
                            ),
                        ),
                    )
                ]
            )
            info = SimpleNamespace(path=checkout, version="v1", revision="pinned")
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
            ):
                outcome = await run_osworld_task(
                    task,
                    RunContext(),
                    checkout / "run",
                    desktop_factory=lambda agent_id, cache: desktop,
                    model_factory=lambda agent_id: model,
                    system_prompt="",
                    max_steps=1,
                )
            self.assertEqual(outcome.score, 0.25)
            self.assertIn("BudgetExceeded", outcome.metadata["agent_error"])

    async def test_evaluation_and_cleanup_failures_are_both_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config.parent.mkdir(parents=True)
            task_config = {"id": "task", "instruction": "click"}
            config.write_text(json.dumps(task_config))
            desktop = FakeDesktop(
                RuntimeError("evaluator boom"), RuntimeError("close boom")
            )
            task = BenchmarkTask(
                "task",
                "click",
                {
                    "checkout": str(checkout),
                    "version": "v1",
                    "revision": "pinned",
                    "domain": "writer",
                    "task_config_sha256": _config_sha256(task_config),
                },
            )
            info = SimpleNamespace(path=checkout, version="v1", revision="pinned")
            with (
                working_directory(checkout),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=info,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "evaluator boom.*cleanup also failed.*close boom"
                ):
                    await run_osworld_task(
                        task,
                        RunContext(),
                        checkout / "run",
                        desktop_factory=lambda agent_id, cache: desktop,
                        model_factory=lambda agent_id: ScriptedModel(
                            [ModelResponse("done")]
                        ),
                        system_prompt="",
                        max_steps=2,
                    )
            self.assertTrue(desktop.closed)


class FakeAdapter:
    def __init__(self, number: int) -> None:
        self.number = number
        self.closed = False

    def observe(self) -> Any:
        return SimpleNamespace(png=png(), meta={})

    def step(self, actions: Any) -> Mapping[str, Any]:
        return {"done": False}

    def finalize(self) -> Any:
        return SimpleNamespace(passed=True, score=100.0, detail="")

    def close(self) -> None:
        self.closed = True


class CUASpeedRunPoolTests(unittest.IsolatedAsyncioTestCase):
    def test_loader_binds_seeded_expected_state_by_digest_only(self) -> None:
        generated = {"instruction": "do the task", "expected": "hidden-value"}
        generator = SimpleNamespace(generate=lambda seed: generated)
        upstream = SimpleNamespace(
            task_id="seeded",
            description="label",
            timeout_sec=120,
            grace_sec=0,
            load_generator=lambda: generator,
        )
        specification = SimpleNamespace(
            tasks=[upstream],
            benchmark_dir=Path("/benchmark"),
            name="fixture",
            version="1",
        )
        checkout = SimpleNamespace(
            path=Path("/checkout"),
            revision="parent",
            gym_anything_revision="gym",
        )
        with (
            patch(
                "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                return_value=checkout,
            ),
            patch("mini_agent.benchmarks.cua_speedrun._activate"),
            patch(
                "mini_agent.benchmarks.cua_speedrun._load_benchmark",
                return_value=specification,
            ),
            patch(
                "mini_agent.benchmarks.cua_speedrun._task_source_sha256",
                return_value="a" * 64,
            ),
            patch(
                "mini_agent.benchmarks.cua_speedrun._gym_anything_task_identity",
                return_value=None,
            ),
        ):
            task = load_cua_speedrun(Path("/checkout"), Path("/benchmark"))[0]
        self.assertEqual(task.prompt, "do the task")
        self.assertEqual(
            task.data["generated_task_sha256"],
            _generated_task_sha256(generated)[0],
        )
        self.assertNotIn("hidden-value", json.dumps(task.data))

    def test_gym_anything_prompt_and_sources_come_from_environment_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "tasks" / "exact"
            task_dir.mkdir(parents=True)
            (root / "env.json").write_text('{"id":"fixture"}')
            task_spec = task_dir / "task.json"
            task_spec.write_text(
                '{"id":"exact","natural_language":{"prompt":"exact prompt"}}'
            )
            scripts = root / "scripts"
            scripts.mkdir()
            setup = scripts / "setup.sh"
            setup.write_text("#!/bin/sh\necho first\n")
            upstream = SimpleNamespace(
                env={
                    "kind": "gym-anything",
                    "env_dir": str(root),
                    "task_id": "exact",
                }
            )
            loading = SimpleNamespace(
                _load_taskspec=lambda path: SimpleNamespace(
                    natural_language={"prompt": "exact prompt"},
                    description="placeholder",
                ),
                _load_envspec=lambda path: SimpleNamespace(
                    mounts=[
                        SimpleNamespace(
                            source=str(scripts), target="/workspace/scripts", mode="ro"
                        )
                    ]
                ),
                _resolve_mount_sources=lambda spec, env_dir: None,
            )
            with patch(
                "mini_agent.benchmarks.cua_speedrun.importlib.import_module",
                return_value=loading,
            ):
                identity = _gym_anything_task_identity(upstream)
            assert identity is not None
            self.assertEqual(identity["prompt"], "exact prompt")
            self.assertEqual(
                identity["prompt_source"],
                "gym-anything.task.natural_language.prompt",
            )
            cache = task_dir / "__pycache__"
            cache.mkdir()
            (cache / "verifier.cpython-312.pyc").write_bytes(b"generated")
            with patch(
                "mini_agent.benchmarks.cua_speedrun.importlib.import_module",
                return_value=loading,
            ):
                with self.assertRaisesRegex(ValueError, "Python bytecode"):
                    _gym_anything_task_identity(upstream)
            (cache / "verifier.cpython-312.pyc").unlink()
            cache.rmdir()
            task_spec.write_text(
                '{"id":"exact","natural_language":{"prompt":"changed"}}'
            )
            with patch(
                "mini_agent.benchmarks.cua_speedrun.importlib.import_module",
                return_value=loading,
            ):
                changed = _gym_anything_task_identity(upstream)
            assert changed is not None
            self.assertNotEqual(identity["sha256"], changed["sha256"])
            task_spec.write_text(
                '{"id":"exact","natural_language":{"prompt":"exact prompt"}}'
            )
            setup.write_text("#!/bin/sh\necho changed\n")
            with patch(
                "mini_agent.benchmarks.cua_speedrun.importlib.import_module",
                return_value=loading,
            ):
                mounted_change = _gym_anything_task_identity(upstream)
            assert mounted_change is not None
            self.assertNotEqual(identity["sha256"], mounted_change["sha256"])

    def test_evaluation_runs_upstream_backend_preflight(self) -> None:
        backend = SimpleNamespace(
            observed_runner_name="QemuNativeRunner",
            preflight=lambda specification: setattr(specification, "ready", True),
        )
        specification = SimpleNamespace(ready=False)
        source = {
            "status": "source_ready",
            "upstream_provisioning_preflight_run": False,
        }
        with (
            patch(
                "mini_agent.benchmarks.cua_speedrun.preflight_cua_speedrun",
                return_value=source,
            ),
            patch(
                "mini_agent.benchmarks.cua_speedrun._load_benchmark",
                return_value=specification,
            ),
            patch(
                "mini_agent.benchmarks.cua_speedrun._backend",
                return_value=backend,
            ),
            patch(
                "mini_agent.benchmarks.cua_speedrun._cua_machine_images",
                return_value=[{"sha256": "a" * 64}],
            ),
        ):
            report = prepare_cua_speedrun_backend(
                Path("/checkout"),
                Path("/benchmark"),
                backend_name="gym-anything-qemu-native",
            )
        self.assertTrue(specification.ready)
        self.assertEqual(report["status"], "backend_ready")
        self.assertTrue(report["upstream_provisioning_preflight_run"])
        self.assertEqual(report["observed_runner"], "QemuNativeRunner")

    def test_prepared_runtime_assets_are_effective_not_cache_name_guesses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "custom-base.qcow2"
            checkpoint = root / "selected-checkpoint.qcow2"
            base.write_bytes(b"base")
            checkpoint.write_bytes(b"checkpoint")

            class Runner:
                base_qcow2 = base
                _container_image = "docker://example/runtime:mutable"

                def _get_checkpoint_path(self) -> Path:
                    return checkpoint

            prepared = SimpleNamespace(
                adapter=SimpleNamespace(_env=SimpleNamespace(_runner=Runner()))
            )
            upstream = SimpleNamespace(env={"use_cache": True})
            identity = _prepared_runtime_assets(prepared, upstream)
            self.assertEqual(
                identity["identity_scope"],
                "effective_prepared_runner_inputs_before_model_call",
            )
            self.assertEqual(
                {item["role"] for item in identity["files"]},
                {"base_image", "selected_checkpoint"},
            )
            self.assertFalse(identity["container"]["content_addressed"])

    def test_runtime_image_cache_binds_adjacent_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "base.qcow2"
            sidecar = Path(str(image) + ".provenance.json")
            image.write_bytes(b"image")
            sidecar.write_bytes(b"first")
            with patch(
                "mini_agent.benchmarks.cua_speedrun.machine_image_identity",
                return_value={"sha256": "a" * 64},
            ) as identity:
                _cached_machine_image_identity(image)
                _cached_machine_image_identity(image)
                self.assertEqual(identity.call_count, 1)
                sidecar.write_bytes(b"changed provenance")
                _cached_machine_image_identity(image)
                self.assertEqual(identity.call_count, 2)

    async def test_verdict_artifact_is_bound_into_outcome(self) -> None:
        adapter = FakeAdapter(1)
        environment = CUAEnvironment(
            CUASpeedRunAdapterClient(adapter, owns_adapter=False),
            benchmark="cua-speed-run",
        )

        class Pool:
            evidence: list[Mapping[str, Any]] = []

            async def environment(self, agent_id: str) -> CUAEnvironment:
                del agent_id
                return environment

            async def evaluate(self, selected: CUAEnvironment) -> Any:
                self.selected = selected
                return SimpleNamespace(passed=True, score=100.0, detail="ok")

            async def close(self) -> None:
                return None

        pool = Pool()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": str(directory),
                    "revision": "pinned",
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "timeout_seconds": 1.0,
                    "grace_seconds": 0.0,
                },
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    return_value=SimpleNamespace(
                        path=directory, revision="pinned", dirty=False
                    ),
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._AdapterPool",
                    return_value=pool,
                ),
            ):
                outcome = await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    directory / "run",
                    backend_name="fixture",
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=1,
                )
            verdict = directory / "run" / "verdict.json"
            self.assertEqual(
                outcome.metadata["verdict_sha256"],
                hashlib.sha256(verdict.read_bytes()).hexdigest(),
            )

    async def test_step_budget_exhaustion_still_runs_the_hidden_checker(
        self,
    ) -> None:
        adapter = FakeAdapter(1)
        environment = CUAEnvironment(
            CUASpeedRunAdapterClient(adapter, owns_adapter=False),
            benchmark="cua-speed-run",
        )

        class Pool:
            evidence: list[Mapping[str, Any]] = []
            evaluated = False

            async def environment(self, agent_id: str) -> CUAEnvironment:
                del agent_id
                return environment

            async def evaluate(self, selected: CUAEnvironment) -> Any:
                self.evaluated = True
                return SimpleNamespace(passed=True, score=100.0, detail="state ok")

            async def close(self) -> None:
                return None

        pool = Pool()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": str(directory),
                    "revision": "pinned",
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "timeout_seconds": 5.0,
                    "grace_seconds": 0.0,
                },
            )
            step = ModelResponse(
                "",
                tool_calls=(
                    ToolCall("c1", "computer", {"actions": [{"type": "screenshot"}]}),
                ),
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    return_value=SimpleNamespace(
                        path=directory, revision="pinned", dirty=False
                    ),
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._AdapterPool",
                    return_value=pool,
                ),
            ):
                outcome = await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    directory / "run",
                    backend_name="fixture",
                    model_factory=lambda agent_id: ScriptedModel([step]),
                    system_prompt="",
                    max_steps=1,
                )
        self.assertTrue(pool.evaluated)
        self.assertEqual(outcome.score, 100.0)
        self.assertEqual(outcome.metadata["timing"]["finish_reason"], "step_budget")
        self.assertIn("BudgetExceeded", outcome.metadata["agent_error"])

    async def test_multi_agent_model_start_failure_is_zero_without_checker(
        self,
    ) -> None:
        adapter = FakeAdapter(1)
        environment = CUAEnvironment(
            CUASpeedRunAdapterClient(adapter, owns_adapter=False),
            benchmark="cua-speed-run",
        )

        class Pool:
            evidence: list[Mapping[str, Any]] = []
            by_adapter = {id(adapter): object()}
            evaluated = False

            async def prewarm(self, count: int, concurrency: int) -> None:
                del count, concurrency

            async def environment(self, agent_id: str) -> CUAEnvironment:
                del agent_id
                return environment

            async def evaluate(self, selected: CUAEnvironment) -> Any:
                del selected
                self.evaluated = True
                raise AssertionError("agent failure must not run checker")

            async def close(self) -> None:
                return None

        pool = Pool()

        def failed_model(agent_id: str) -> ScriptedModel:
            raise AssertionError(f"model process failed for {agent_id}")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": str(directory),
                    "revision": "pinned",
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "timeout_seconds": 1.0,
                    "grace_seconds": 0.0,
                },
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    return_value=SimpleNamespace(
                        path=directory, revision="pinned", dirty=False
                    ),
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._AdapterPool",
                    return_value=pool,
                ),
            ):
                outcome = await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    directory / "run",
                    backend_name="fixture",
                    model_factory=failed_model,
                    system_prompt="",
                    max_steps=1,
                    multi_agent=True,
                    max_active_agents=1,
                    max_total_agents=1,
                )
        self.assertEqual(outcome.score, 0.0)
        self.assertFalse(pool.evaluated)
        self.assertIn("model process failed", outcome.metadata["agent_error"])

    async def test_multi_agent_records_actual_state_adoption(self) -> None:
        class Pool:
            evidence: list[Mapping[str, Any]] = []
            by_adapter: dict[int, object] = {}
            next_number = 0
            selected_number: int | None = None

            async def prewarm(self, count: int, concurrency: int) -> None:
                del count, concurrency

            async def environment(self, agent_id: str) -> CUAEnvironment:
                del agent_id
                self.next_number += 1
                adapter = FakeAdapter(self.next_number)
                self.by_adapter[id(adapter)] = object()
                return CUAEnvironment(
                    CUASpeedRunAdapterClient(adapter, owns_adapter=False),
                    benchmark="cua-speed-run",
                )

            async def evaluate(self, selected: CUAEnvironment) -> Any:
                adapter = selected.client.adapter
                self.selected_number = adapter.number
                return SimpleNamespace(passed=True, score=100.0, detail="ok")

            async def close(self) -> None:
                return None

        pool = Pool()
        root_id: str | None = None

        def agent_call(call_id: str, arguments: Mapping[str, Any]) -> ModelResponse:
            return ModelResponse(
                "", tool_calls=(ToolCall(call_id, "agent", arguments),)
            )

        def model(agent_id: str) -> ScriptedModel:
            nonlocal root_id
            if agent_id.endswith("/root"):
                root_id = agent_id
                child = agent_id + "/1"
                return ScriptedModel(
                    [
                        agent_call("spawn", {"action": "spawn", "task": "child"}),
                        agent_call(
                            "wait", {"action": "wait", "agent_ids": [child]}
                        ),
                        agent_call(
                            "adopt", {"action": "adopt", "agent_id": child}
                        ),
                        ModelResponse("selected"),
                    ]
                )
            if root_id is not None and agent_id == root_id + "/1":
                return ScriptedModel([ModelResponse("candidate")])
            raise AssertionError(f"unexpected agent id {agent_id!r}")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": str(directory),
                    "revision": "pinned",
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "timeout_seconds": 1.0,
                    "grace_seconds": 0.0,
                },
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    return_value=SimpleNamespace(
                        path=directory, revision="pinned", dirty=False
                    ),
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._AdapterPool",
                    return_value=pool,
                ),
            ):
                outcome = await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    directory / "run",
                    backend_name="fixture",
                    model_factory=model,
                    system_prompt="",
                    max_steps=4,
                    multi_agent=True,
                    max_active_agents=2,
                    max_total_agents=2,
                )
        assert root_id is not None
        self.assertEqual(pool.selected_number, 2)
        self.assertEqual(
            outcome.metadata["state_selection"],
            "adopted_descendant_environment",
        )
        self.assertEqual(
            outcome.metadata["state_adoption_history"], [root_id + "/1"]
        )

    async def test_run_rejects_checkout_manifest_mismatch(self) -> None:
        task = BenchmarkTask(
            "id",
            "task",
            {"checkout": "/fixture", "revision": "old"},
        )
        with patch(
            "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
            return_value=SimpleNamespace(
                path=Path("/fixture"), revision="new", dirty=False
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "checkout changed"):
                await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    Path("/run"),
                    backend_name="fake",
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=1,
                )

    async def test_run_revalidates_checkout_before_verdict_commit(self) -> None:
        adapter = FakeAdapter(1)
        environment = CUAEnvironment(
            CUASpeedRunAdapterClient(adapter, owns_adapter=False),
            benchmark="cua-speed-run",
        )

        class Pool:
            evidence: list[Mapping[str, Any]] = []

            async def environment(self, agent_id: str) -> CUAEnvironment:
                del agent_id
                return environment

            async def evaluate(self, selected: CUAEnvironment) -> Any:
                del selected
                return SimpleNamespace(passed=True, score=100.0, detail="ok")

            async def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = SimpleNamespace(path=root, revision="pinned", dirty=False)
            changed = SimpleNamespace(path=root, revision="pinned", dirty=True)
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": str(root),
                    "revision": "pinned",
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "timeout_seconds": 1.0,
                    "grace_seconds": 0.0,
                },
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    side_effect=(original, changed),
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._AdapterPool",
                    return_value=Pool(),
                ),
                self.assertRaisesRegex(RuntimeError, "checkout changed"),
            ):
                await run_cua_speedrun_task(
                    task,
                    RunContext(),
                    root / "run",
                    backend_name="fixture",
                    model_factory=lambda agent_id: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                    system_prompt="",
                    max_steps=1,
                )
            self.assertFalse((root / "run/verdict.json").exists())

    async def test_task_clock_distinguishes_deadline_from_backend_timeout(
        self,
    ) -> None:
        with self.assertRaises(_TaskClockExpired):
            await _on_task_clock(asyncio.sleep(0.05), 0.001)

        async def backend_timeout() -> None:
            raise TimeoutError("backend request timed out")

        with self.assertRaisesRegex(TimeoutError, "backend request timed out"):
            await _on_task_clock(backend_timeout(), 1.0)

    def test_checkout_requires_the_exact_upstream_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            specs = checkout / "src" / "cua_speedrun" / "specs.py"
            specs.parent.mkdir(parents=True)
            specs.write_text("class Benchmark: pass\n")
            gym = checkout / "third_party" / "gym-anything"
            package = gym / "src" / "gym_anything" / "__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text("")

            def git(path: Path, *arguments: str) -> str:
                if path == checkout and arguments == ("rev-parse", "HEAD"):
                    return "7230223cbc57df68331cad32889adf01f3601651"
                if path == checkout and arguments[:2] == ("status", "--porcelain"):
                    return ""
                if path == checkout and arguments[:2] == ("ls-files", "--others"):
                    return ""
                if path == checkout and arguments[:2] == ("ls-tree", "HEAD"):
                    return (
                        "160000 commit "
                        "70d9e51d2517049d995cc820a319a355c3c6e979\t"
                        "third_party/gym-anything"
                    )
                if path == gym and arguments == ("rev-parse", "HEAD"):
                    return "70d9e51d2517049d995cc820a319a355c3c6e979"
                if path == gym and arguments[:2] == ("status", "--porcelain"):
                    return ""
                if path == gym and arguments[:2] == ("ls-files", "--others"):
                    return ""
                raise AssertionError((path, arguments))

            with patch(
                "mini_agent.benchmarks.cua_speedrun._git",
                side_effect=git,
            ):
                info = inspect_cua_speedrun_checkout(checkout)
            self.assertEqual(info.revision, "7230223cbc57df68331cad32889adf01f3601651")
            self.assertEqual(
                info.gym_anything_revision,
                "70d9e51d2517049d995cc820a319a355c3c6e979",
            )

            def wrong_submodule_head(path: Path, *arguments: str) -> str:
                if path == gym and arguments == ("rev-parse", "HEAD"):
                    return "f" * 40
                return git(path, *arguments)

            with patch(
                "mini_agent.benchmarks.cua_speedrun._git",
                side_effect=wrong_submodule_head,
            ):
                with self.assertRaisesRegex(ValueError, "submodule must be"):
                    inspect_cua_speedrun_checkout(checkout)

            def dirty_submodule(path: Path, *arguments: str) -> str:
                if path == gym and arguments[:2] == ("status", "--porcelain"):
                    return "?? generated-file"
                return git(path, *arguments)

            with patch(
                "mini_agent.benchmarks.cua_speedrun._git",
                side_effect=dirty_submodule,
            ):
                with self.assertRaisesRegex(ValueError, "submodule is dirty"):
                    inspect_cua_speedrun_checkout(checkout)

            bytecode = gym / "src" / "gym_anything" / "__pycache__" / "task.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"untrusted-bytecode")

            def bytecode_submodule(path: Path, *arguments: str) -> str:
                if path == gym and arguments[:2] == ("status", "--porcelain"):
                    return "?? src/gym_anything/__pycache__/task.pyc"
                if path == gym and arguments[:2] == ("ls-files", "--others"):
                    return "src/gym_anything/__pycache__/task.pyc\x00"
                return git(path, *arguments)

            with patch(
                "mini_agent.benchmarks.cua_speedrun._git",
                side_effect=bytecode_submodule,
            ):
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    inspect_cua_speedrun_checkout(checkout, allow_dirty=True)

    def test_cua_checkout_rejects_untracked_executable_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            specs = checkout / "src" / "cua_speedrun" / "specs.py"
            specs.parent.mkdir(parents=True)
            specs.write_text("class Benchmark: pass\n")
            injected = checkout / "src" / "sitecustomize.py"
            injected.write_text("raise RuntimeError('injected')\n")
            gym = checkout / "third_party" / "gym-anything"
            package = gym / "src" / "gym_anything" / "__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text("")

            def git(path: Path, *arguments: str) -> str:
                if path == checkout and arguments == ("rev-parse", "HEAD"):
                    return "7230223cbc57df68331cad32889adf01f3601651"
                if path == checkout and arguments[:2] == ("status", "--porcelain"):
                    return ""
                if path == checkout and arguments[:2] == ("ls-files", "--others"):
                    return "src/sitecustomize.py\x00"
                raise AssertionError((path, arguments))

            with patch("mini_agent.benchmarks.cua_speedrun._git", side_effect=git):
                with self.assertRaisesRegex(ValueError, "untracked executable"):
                    inspect_cua_speedrun_checkout(checkout)

    def test_activate_rejects_an_imported_gym_anything_from_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            (checkout / "src" / "cua_speedrun").mkdir(parents=True)
            (checkout / "third_party" / "gym-anything" / "src").mkdir(parents=True)
            external = Path(temporary) / "external" / "gym_anything" / "__init__.py"
            external.parent.mkdir(parents=True)
            external.write_text("")
            module = types.ModuleType("gym_anything")
            module.__file__ = str(external)
            with patch.dict("sys.modules", {"gym_anything": module}):
                with self.assertRaisesRegex(RuntimeError, "different gym-anything"):
                    _activate(checkout)

    def test_doctor_never_runs_upstream_provisioning_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = Path(temporary) / "benchmark"
            benchmark.mkdir()
            (benchmark / "manifest.yaml").write_text("fixture")

            class Backend:
                name = "fixture"

                def preflight(self, specification: Any) -> None:
                    raise AssertionError(f"provisioning preflight ran: {specification}")

            checkout = SimpleNamespace(
                path=Path(temporary),
                revision="pinned",
                dirty=False,
                gym_anything_path=Path(temporary) / "third_party" / "gym-anything",
                gym_anything_revision="gym-pinned",
                gym_anything_dirty=False,
            )
            specification = SimpleNamespace(
                name="fixture", version="1", tasks=[object()]
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun.inspect_cua_speedrun_checkout",
                    return_value=checkout,
                ),
                patch("mini_agent.benchmarks.cua_speedrun._activate"),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._load_benchmark",
                    return_value=specification,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=Backend(),
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._gym_anything_module_origin",
                    return_value="/fixture/gym_anything/__init__.py",
                ),
            ):
                report = preflight_cua_speedrun(
                    Path(temporary), benchmark, backend_name="fixture"
                )
            self.assertEqual(report["status"], "source_ready")
            self.assertFalse(report["upstream_provisioning_preflight_run"])
            self.assertFalse(report["machine_launch_canary_run"])

    async def test_multi_agent_pool_prepares_all_leases_off_clock(self) -> None:
        upstream = SimpleNamespace(
            env={"kind": "fake"},
            load_generator=lambda: None,
        )

        class Backend:
            def __init__(self) -> None:
                self.count = 0

            def prepare(self, env: Any, seed: int, directory: Path) -> Any:
                del env, seed, directory
                self.count += 1
                return SimpleNamespace(
                    adapter=FakeAdapter(self.count),
                    description="task",
                    prepare_time_sec=0.01,
                    info={"runner": "fake"},
                    checker=None,
                )

        backend = Backend()
        with tempfile.TemporaryDirectory() as temporary:
            pool = _AdapterPool(
                task=BenchmarkTask(
                    "id",
                    "task",
                    {"benchmark": temporary, "seed": 0},
                ),
                directory=Path(temporary) / "run",
                backend_name="fake",
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun._upstream_task",
                    return_value=upstream,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=backend,
                ) as backend_factory,
            ):
                await pool.prewarm(3, 2)
                self.assertEqual(backend_factory.call_count, 1)
                self.assertEqual(len(pool.available), 3)
                environments = [
                    await pool.environment(f"/root/{index}") for index in range(1, 4)
                ]
                self.assertEqual(len(pool.prepared), 3)
                self.assertEqual(
                    len(
                        {
                            environment.resource_identity()
                            for environment in environments
                        }
                    ),
                    3,
                )
                adapters = [prepared.adapter for prepared in pool.prepared]
                await pool.close()
                self.assertTrue(all(adapter.closed for adapter in adapters))

    async def test_runtime_assets_are_rechecked_at_lease_time(self) -> None:
        adapter = FakeAdapter(1)
        upstream = SimpleNamespace(
            env={"kind": "fake"},
            load_generator=lambda: None,
        )
        backend = SimpleNamespace(
            prepare=lambda env, seed, directory: SimpleNamespace(
                adapter=adapter,
                description="task",
                prepare_time_sec=0.1,
                info={},
                checker=None,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            pool = _AdapterPool(
                task=BenchmarkTask("id", "task", {"benchmark": temporary, "seed": 0}),
                directory=Path(temporary) / "run",
                backend_name="fake",
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun._upstream_task",
                    return_value=upstream,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=backend,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._prepared_runtime_assets",
                    side_effect=(
                        {"identity_scope": "prepared", "files": ["first"]},
                        {"identity_scope": "prepared", "files": ["changed"]},
                    ),
                ),
            ):
                await pool.prepare(1)
                with self.assertRaisesRegex(RuntimeError, "runtime assets changed"):
                    await pool.environment("/root")
                await pool.close()
        self.assertTrue(adapter.closed)

    async def test_post_prepare_failure_closes_unowned_adapter(self) -> None:
        adapter = FakeAdapter(1)
        upstream = SimpleNamespace(
            env={"kind": "fake"},
            load_generator=lambda: None,
        )
        backend = SimpleNamespace(
            prepare=lambda env, seed, directory: SimpleNamespace(
                adapter=adapter,
                description="different",
                prepare_time_sec=0.1,
                info={},
                checker=None,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            pool = _AdapterPool(
                task=BenchmarkTask("id", "task", {"benchmark": temporary, "seed": 0}),
                directory=Path(temporary) / "run",
                backend_name="fake",
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun._upstream_task",
                    return_value=upstream,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=backend,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "differs from the manifest"):
                    await pool.prepare(1)
            self.assertTrue(adapter.closed)
            self.assertEqual(pool.prepared, [])

    async def test_seeded_generator_runs_once_per_prepared_lease(self) -> None:
        class Generator:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, seed: int) -> Mapping[str, Any]:
                self.calls += 1
                return {"instruction": "task", "expected": seed}

            def check(self, adapter: Any, expected: Any) -> Mapping[str, Any]:
                return {"passed": expected == 7}

        generator = Generator()
        adapter = FakeAdapter(1)
        upstream = SimpleNamespace(
            env={"kind": "fake"},
            load_generator=lambda: generator,
        )
        backend = SimpleNamespace(
            prepare=lambda env, seed, directory: SimpleNamespace(
                adapter=adapter,
                description="ignored for seeded tasks",
                prepare_time_sec=0.1,
                info={},
                checker=None,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            pool = _AdapterPool(
                task=BenchmarkTask(
                    "id",
                    "task",
                    {
                        "benchmark": temporary,
                        "seed": 7,
                        "generated_task_sha256": _generated_task_sha256(
                            {"instruction": "task", "expected": 7}
                        )[0],
                    },
                ),
                directory=Path(temporary) / "run",
                backend_name="fake",
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun._upstream_task",
                    return_value=upstream,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=backend,
                ),
            ):
                await pool.prepare(1)
            self.assertEqual(generator.calls, 1)
            await pool.close()

    async def test_seeded_expected_state_must_match_the_manifest_digest(self) -> None:
        generator = SimpleNamespace(
            generate=lambda seed: {"instruction": "task", "expected": seed + 1},
            check=lambda adapter, expected: {"passed": True},
        )
        adapter = FakeAdapter(1)
        upstream = SimpleNamespace(
            env={"kind": "fake"},
            load_generator=lambda: generator,
        )
        backend = SimpleNamespace(
            prepare=lambda env, seed, directory: SimpleNamespace(
                adapter=adapter,
                description="ignored",
                prepare_time_sec=0.1,
                info={},
                checker=None,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            pool = _AdapterPool(
                task=BenchmarkTask(
                    "id",
                    "task",
                    {
                        "benchmark": temporary,
                        "seed": 7,
                        "generated_task_sha256": _generated_task_sha256(
                            {"instruction": "task", "expected": 7}
                        )[0],
                    },
                ),
                directory=Path(temporary) / "run",
                backend_name="fake",
            )
            with (
                patch(
                    "mini_agent.benchmarks.cua_speedrun._upstream_task",
                    return_value=upstream,
                ),
                patch(
                    "mini_agent.benchmarks.cua_speedrun._backend",
                    return_value=backend,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "generated cua-speed-run task differs"
                ):
                    await pool.prepare(1)
            self.assertTrue(adapter.closed)
            self.assertEqual(pool.prepared, [])

    def test_seeded_checker_rejects_truthy_non_boolean(self) -> None:
        generator = SimpleNamespace(check=lambda adapter, expected: {"passed": "false"})
        prepared = SimpleNamespace(adapter=object())
        with patch(
            "mini_agent.benchmarks.cua_speedrun.importlib.import_module",
            return_value=SimpleNamespace(Verdict=lambda **values: values),
        ):
            check = _checker(generator, prepared, None)
            with self.assertRaisesRegex(RuntimeError, "non-boolean"):
                check()

    def test_task_source_hash_is_rechecked_at_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "task"
            task_dir.mkdir()
            source = task_dir / "task.yaml"
            source.write_text("original")
            upstream = SimpleNamespace(
                task_id="id",
                task_dir=task_dir,
                timeout_sec=1.0,
                grace_sec=0.0,
            )
            specification = SimpleNamespace(
                name="fixture", version="1", tasks=[upstream]
            )
            task = BenchmarkTask(
                "id",
                "task",
                {
                    "checkout": temporary,
                    "benchmark": temporary,
                    "benchmark_name": "fixture",
                    "benchmark_version": "1",
                    "task_source_sha256": _task_source_sha256(upstream),
                    "timeout_seconds": 1.0,
                    "grace_seconds": 0.0,
                },
            )
            with patch(
                "mini_agent.benchmarks.cua_speedrun._load_benchmark",
                return_value=specification,
            ):
                self.assertIs(_upstream_task(task), upstream)
                source.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "sources changed"):
                    _upstream_task(task)

    def test_task_source_hash_rejects_python_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "task"
            task_dir.mkdir()
            (task_dir / "task.yaml").write_text("description: fixture\n")
            upstream = SimpleNamespace(task_dir=task_dir)
            _task_source_sha256(upstream)
            cache = task_dir / "__pycache__"
            cache.mkdir()
            (cache / "task.cpython-312.pyc").write_bytes(b"generated")
            with self.assertRaisesRegex(ValueError, "Python bytecode"):
                _task_source_sha256(upstream)


if __name__ == "__main__":
    unittest.main()
