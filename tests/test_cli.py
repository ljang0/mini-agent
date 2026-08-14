from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, patch

# The credential preflight fails closed before any run output exists; CLI tests
# use scripted models, so a fixture key satisfies it without any network use.
os.environ.setdefault("OPENAI_API_KEY", "fixture-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "fixture-key")

from mini_agent.environments.web import SnippetTokenizerAdapter
from mini_agent.cli import (
    _execution_config,
    _evaluation_config,
    _grader_model_factory,
    _load_tokenizer,
    _model_factory,
    _provider_headers,
    _run_single_agent,
    build_parser,
    main,
)
from mini_agent.grading import (
    _grade_eval_directory,
    _grader_runtime_identity,
    _required_path,
    _verify_browsecomp_plus_grader_assets,
    _verify_grade_prompt_binding,
)
from mini_agent.environments.base import BaseEnvironment
from mini_agent.models import ScriptedModel
from mini_agent.runtime import RunContext
from mini_agent.storage import StorageLayout
from mini_agent.types import ModelResponse, ToolCall, ToolDefinition


def _fixture_grader_runtime(root: Path, benchmark: str) -> dict[str, Any]:
    packages = {
        "swebench": {"swebench": "4.1.0"},
        "programbench": {"programbench": "1.2.4"},
    }.get(benchmark, {"numpy": "1.26.4", "tqdm": "4.67.1", "vllm": "0.9.0.1"})
    modules: dict[str, dict[str, str]] = {}
    for name in packages:
        package_root = root / "isolated" / name
        package_root.mkdir(parents=True, exist_ok=True)
        origin = package_root / "__init__.py"
        origin.touch()
        modules[name] = {
            "origin": str(origin.resolve()),
            "package_root": str(package_root.resolve()),
        }
    return {
        "isolation": "python-I",
        "python_executable": str(Path(sys.executable).absolute()),
        "python_version": "fixture",
        "python_implementation": "cpython",
        "python_prefix": str(Path(sys.prefix).absolute()),
        "python_base_prefix": str(Path(sys.base_prefix).absolute()),
        "packages": packages,
        "modules": modules,
    }


def _write_evaluation_manifest(
    root: Path,
    benchmark: str,
    tasks: Sequence[tuple[str, str] | tuple[str, str, dict[str, Any]]],
) -> None:
    identities: list[dict[str, str]] = []
    for item in tasks:
        task_id, prompt = item[:2]
        data = item[2] if len(item) == 3 else task_id
        data_bytes = (
            json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if isinstance(data, dict)
            else data.encode("utf-8")
        )
        identities.append(
            {
                "id": task_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            }
        )
    value: dict[str, Any] = {
        "schema": "mini-agent-eval-v2",
        "harness": {"source_sha256": "0" * 64},
        "benchmark": benchmark,
        "config": (
            {
                "adapter": {
                    "runtime": "docker",
                    "container_runtime": ["docker"],
                    "image_bindings": {
                        task["id"]: {
                            "runtime": "docker",
                            "requested": (
                                "docker.io/swebench/sweb.eval.x86_64."
                                + task["id"].casefold().replace("__", "_1776_")
                                + ":latest"
                            ),
                            "identity": "sha256:" + "b" * 64,
                        }
                        for task in identities
                    },
                }
            }
            if benchmark == "swebench"
            else {}
        ),
        "limits": {},
        "max_workers": 1,
        "capture_content": False,
        "tasks": identities,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(value))


def _commit_fixture_result(
    directory: Path, task_id: str, metadata: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": task_id,
        "status": "completed",
        "metadata": metadata,
    }
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    (directory / "result.json").write_bytes(raw)
    (directory / "completed.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "result_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )


class CLITests(unittest.TestCase):
    def test_swebench_grade_snapshots_inputs_and_records_private_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            task_id = "repo__issue-1"
            prompt = "fix the issue"
            dataset_row = {
                "instance_id": task_id,
                "problem_statement": prompt,
            }
            _write_evaluation_manifest(
                evaluation, "swebench", [(task_id, prompt, dataset_row)]
            )
            instance = evaluation / "instances" / "one"
            prediction = {
                "instance_id": task_id,
                "model_patch": "diff --git a/a b/a\n",
                "model_name_or_path": "fixture/model",
            }
            instance.mkdir(parents=True)
            prediction_path = instance / "prediction.json"
            prediction_path.write_text(json.dumps(prediction))
            _commit_fixture_result(
                instance,
                task_id,
                {
                    "prediction_sha256": hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest()
                },
            )
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([dataset_row]))
            invoked: dict[str, Any] = {}

            def run_grader(
                argv: Sequence[str],
                *,
                cwd: Path,
                check: bool,
                stdout: Any,
                stderr: Any,
                env: Mapping[str, str],
            ) -> SimpleNamespace:
                invoked.update(
                    {"argv": tuple(argv), "cwd": cwd, "check": check, "env": env}
                )
                stdout.write(b"official output\n")
                stderr.write(b"")
                report = cwd / "fixture.report.json"
                report.write_text('{"resolved": 1}\n')
                report.chmod(0o666)
                return SimpleNamespace(returncode=0)

            stdout = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": str(root / "shadow-modules"),
                        "PYTHONHOME": str(root / "shadow-home"),
                        "OPENAI_API_KEY": "solver-secret",
                        "SERPAPI_API_KEY": "browser-secret",
                        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
                        "MINI_AGENT_AUDIT_SECRET": "arbitrary-secret",
                    },
                ),
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "swebench"),
                ),
                patch(
                    "mini_agent.cli.harness_identity",
                    return_value={"schema": "fixture"},
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "swebench_grader_source_identity",
                    return_value={
                        "revision": "fixture",
                        "source_sha256": "a" * 64,
                    },
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "verify_swebench_grader_images",
                    return_value={
                        "container_runtime": ["docker"],
                        "images": [
                            {
                                "instance_id": task_id,
                                "image_id": "sha256:" + "b" * 64,
                            }
                        ],
                    },
                ),
                patch("mini_agent.grading.subprocess.run", side_effect=run_grader),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "swebench",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--dataset",
                        str(dataset),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(invoked["cwd"], grade.resolve())
            argv = invoked["argv"]
            self.assertEqual(argv[1], "-I")
            self.assertEqual(invoked["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertNotIn("PYTHONPATH", invoked["env"])
            self.assertNotIn("PYTHONHOME", invoked["env"])
            self.assertNotIn("OPENAI_API_KEY", invoked["env"])
            self.assertNotIn("SERPAPI_API_KEY", invoked["env"])
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", invoked["env"])
            self.assertNotIn("MINI_AGENT_AUDIT_SECRET", invoked["env"])
            self.assertEqual(invoked["env"]["HOME"], str(grade.resolve()))
            dataset_argument = Path(argv[argv.index("--dataset_name") + 1])
            predictions_argument = Path(argv[argv.index("--predictions_path") + 1])
            self.assertTrue(dataset_argument.is_relative_to(grade / "inputs"))
            self.assertTrue(predictions_argument.is_relative_to(grade / "inputs"))
            manifest = json.loads((grade / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], "mini-agent-grade-v1")
            self.assertRegex(manifest["fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                manifest["inputs"]["dataset"]["sha256"],
                manifest["inputs"]["sources"]["dataset"]["sha256"],
            )
            self.assertEqual(
                manifest["grader"]["images"]["images"][0]["image_id"],
                "sha256:" + "b" * 64,
            )
            result = json.loads((grade / "result.json").read_text())
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(
                result["grade_manifest"]["sha256"],
                hashlib.sha256((grade / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(result["artifacts"]["file_count"], 1)
            self.assertEqual(
                result["artifacts"]["files"][0]["path"],
                "fixture.report.json",
            )
            self.assertEqual(stat.S_IMODE(grade.stat().st_mode), 0o700)
            for path in (
                grade / "manifest.json",
                grade / "stdout.log",
                grade / "stderr.log",
                grade / "result.json",
                grade / "completed.json",
                grade / "fixture.report.json",
                dataset_argument,
                predictions_argument,
            ):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            completion = json.loads((grade / "completed.json").read_text())
            self.assertEqual(
                completion["result_sha256"],
                hashlib.sha256((grade / "result.json").read_bytes()).hexdigest(),
            )

    def test_programbench_grade_binds_the_checkout_and_official_eval(self) -> None:
        from test_benchmarks import (
            programbench_fixture_checkout,
            programbench_fixture_git,
        )

        from mini_agent.benchmarks.programbench import (
            PROGRAMBENCH_REVISION,
            load_programbench,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            checkout = programbench_fixture_checkout(root)
            run_git = programbench_fixture_git(checkout.resolve())
            with patch(
                "mini_agent.benchmarks.programbench._git", side_effect=run_git
            ):
                task = load_programbench(checkout)[0]
            _write_evaluation_manifest(
                evaluation,
                "programbench",
                [(task.task_id, task.prompt, dict(task.data))],
            )
            instance = evaluation / "instances" / "one"
            instance.mkdir(parents=True)
            archive = b"submission-archive"
            (instance / "submission.tar.gz").write_bytes(archive)
            _commit_fixture_result(
                instance,
                task.task_id,
                {"submission_sha256": hashlib.sha256(archive).hexdigest()},
            )
            invoked: dict[str, Any] = {}

            def run_grader(
                argv: Sequence[str],
                *,
                cwd: Path,
                check: bool,
                stdout: Any,
                stderr: Any,
                env: Mapping[str, str],
            ) -> SimpleNamespace:
                invoked.update({"argv": tuple(argv), "cwd": cwd, "env": env})
                stdout.write(b"official output\n")
                stderr.write(b"")
                target = Path(argv[argv.index("--output") + 1])
                report = target / "run" / task.task_id
                report.mkdir(parents=True)
                (report / f"{task.task_id}.eval.json").write_text('{"score": 1.0}\n')
                return SimpleNamespace(returncode=0)

            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "solver-secret"}),
                patch(
                    "mini_agent.benchmarks.programbench._git", side_effect=run_git
                ),
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "programbench"),
                ),
                patch(
                    "mini_agent.cli.harness_identity",
                    return_value={"schema": "fixture"},
                ),
                patch("mini_agent.grading.subprocess.run", side_effect=run_grader),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "programbench",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--checkout",
                        str(checkout),
                    ]
                )

            self.assertEqual(code, 0, stdout.getvalue())
            argv = invoked["argv"]
            self.assertEqual(argv[1:3], ("-I", "-c"))
            self.assertIn("programbench.cli.main", argv[3])
            self.assertEqual(argv[4], "eval")
            snapshot = Path(argv[5])
            self.assertTrue(snapshot.is_relative_to(grade / "inputs"))
            self.assertEqual(
                (snapshot / task.task_id / "submission.tar.gz").read_bytes(),
                archive,
            )
            eval_output = Path(argv[argv.index("--output") + 1])
            self.assertTrue(eval_output.is_relative_to(grade))
            self.assertFalse(eval_output.is_relative_to(grade / "inputs"))
            self.assertNotIn("OPENAI_API_KEY", invoked["env"])
            self.assertEqual(invoked["env"]["HOME"], str(grade.resolve()))
            manifest = json.loads((grade / "manifest.json").read_text())
            self.assertEqual(manifest["grader"]["revision"], PROGRAMBENCH_REVISION)
            self.assertEqual(manifest["grader"]["version"], "1.2.4")
            self.assertEqual(manifest["grader"]["image_tag"], "task_cleanroom_v6")
            self.assertEqual(
                manifest["inputs"]["submission_sha256"][task.task_id],
                hashlib.sha256(archive).hexdigest(),
            )
            result = json.loads((grade / "result.json").read_text())
            self.assertEqual(result["returncode"], 0)
            self.assertIn(
                f"official_evals/run/{task.task_id}/{task.task_id}.eval.json",
                [item["path"] for item in result["artifacts"]["files"]],
            )
            self.assertEqual(
                result["verified_grader_assets"]["checkout"]["revision"],
                PROGRAMBENCH_REVISION,
            )

    def test_swebench_grade_does_not_commit_mutated_grader_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            task_id = "repo__issue-1"
            prompt = "fix the issue"
            dataset_row = {
                "instance_id": task_id,
                "problem_statement": prompt,
            }
            _write_evaluation_manifest(
                evaluation, "swebench", [(task_id, prompt, dataset_row)]
            )
            instance = evaluation / "instances" / "one"
            instance.mkdir(parents=True)
            prediction_path = instance / "prediction.json"
            prediction_path.write_text(
                json.dumps(
                    {
                        "instance_id": task_id,
                        "model_patch": "diff --git a/a b/a\n",
                        "model_name_or_path": "fixture/model",
                    }
                )
            )
            _commit_fixture_result(
                instance,
                task_id,
                {
                    "prediction_sha256": hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest()
                },
            )
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([dataset_row]))
            identity = {"revision": "fixture", "source_sha256": "a" * 64}
            checks = 0

            def source_identity(_root: Path) -> dict[str, str]:
                nonlocal checks
                checks += 1
                if checks == 3:
                    raise RuntimeError("installed SWE-bench harness changed")
                return identity

            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": str(root / "shadow-modules"),
                        "PYTHONHOME": str(root / "shadow-home"),
                        "OPENAI_API_KEY": "solver-secret",
                        "SERPAPI_API_KEY": "browser-secret",
                        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
                        "MINI_AGENT_AUDIT_SECRET": "arbitrary-secret",
                    },
                ),
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "swebench"),
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "swebench_grader_source_identity",
                    side_effect=source_identity,
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "verify_swebench_grader_images",
                    return_value={"container_runtime": ["docker"], "images": []},
                ),
                patch(
                    "mini_agent.grading.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "swebench",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--dataset",
                        str(dataset),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(checks, 3)
            self.assertIn("harness changed", stderr.getvalue())
            self.assertFalse((grade / "completed.json").exists())

    def test_swebench_grade_does_not_commit_a_changed_grader_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            task_id = "repo__issue-1"
            prompt = "fix the issue"
            dataset_row = {
                "instance_id": task_id,
                "problem_statement": prompt,
            }
            _write_evaluation_manifest(
                evaluation, "swebench", [(task_id, prompt, dataset_row)]
            )
            instance = evaluation / "instances" / "one"
            instance.mkdir(parents=True)
            prediction_path = instance / "prediction.json"
            prediction_path.write_text(
                json.dumps(
                    {
                        "instance_id": task_id,
                        "model_patch": "diff --git a/a b/a\n",
                        "model_name_or_path": "fixture/model",
                    }
                )
            )
            _commit_fixture_result(
                instance,
                task_id,
                {
                    "prediction_sha256": hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest()
                },
            )
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([dataset_row]))
            image_identity = {
                "container_runtime": ["docker"],
                "images": [{"instance_id": task_id, "image_id": "sha256:" + "b" * 64}],
            }
            checks = 0

            def verify_images(
                _manifest: Mapping[str, Any], **_kwargs: Any
            ) -> Mapping[str, Any]:
                nonlocal checks
                checks += 1
                if checks == 3:
                    raise RuntimeError(
                        "official SWE-bench grader image changed identity"
                    )
                return image_identity

            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": str(root / "shadow-modules"),
                        "PYTHONHOME": str(root / "shadow-home"),
                        "OPENAI_API_KEY": "solver-secret",
                        "SERPAPI_API_KEY": "browser-secret",
                        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
                        "MINI_AGENT_AUDIT_SECRET": "arbitrary-secret",
                    },
                ),
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "swebench"),
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "swebench_grader_source_identity",
                    return_value={
                        "revision": "fixture",
                        "source_sha256": "a" * 64,
                    },
                ),
                patch(
                    "mini_agent.benchmarks.swebench."
                    "verify_swebench_grader_images",
                    side_effect=verify_images,
                ),
                patch(
                    "mini_agent.grading.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "swebench",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--dataset",
                        str(dataset),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(checks, 3)
            self.assertIn("grader image changed identity", stderr.getvalue())
            self.assertFalse((grade / "completed.json").exists())

    def test_browsecomp_plus_grade_binds_checkout_model_and_hidden_inputs(self) -> None:
        from mini_agent.benchmarks.web import (
            BROWSECOMP_PLUS_QUERY,
            BROWSECOMP_PLUS_REVISION,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            task_id = "7"
            question = "Which answer?"
            prompt = BROWSECOMP_PLUS_QUERY.format(Question=question)
            _write_evaluation_manifest(
                evaluation, "browsecomp-plus", [(task_id, prompt)]
            )
            instance = evaluation / "instances" / "one"
            run = {
                "query_id": task_id,
                "status": "completed",
                "tool_call_counts": {"search": 1, "get_document": 0},
                "retrieved_docids": [],
                "result": [
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": "Exact Answer: alpha\nConfidence: 90%",
                    }
                ],
            }
            instance.mkdir(parents=True)
            run_path = instance / "browsecomp_plus_run.json"
            run_path.write_text(json.dumps(run))
            _commit_fixture_result(
                instance,
                task_id,
                {
                    "browsecomp_plus_run_sha256": hashlib.sha256(
                        run_path.read_bytes()
                    ).hexdigest()
                },
            )
            truth = root / "ground_truth.jsonl"
            truth.write_text(
                json.dumps({"query_id": task_id, "query": question, "answer": "alpha"})
                + "\n"
            )
            qrels = root / "qrels.txt"
            qrels.write_text(f"{task_id} 0 123 1\n")
            judge_model = root / "judge-model"
            judge_model.mkdir()
            (judge_model / "weights.bin").write_bytes(b"weights")
            checkout = root / "checkout"
            (checkout / "scripts_evaluation").mkdir(parents=True)
            (checkout / "scripts_evaluation" / "evaluate_run.py").write_text(
                "# grader\n"
            )
            (checkout / "uv.lock").write_text("# exact lock\n")
            invoked: dict[str, Any] = {}

            def run_grader(
                argv: Sequence[str],
                *,
                cwd: Path,
                check: bool,
                stdout: Any,
                stderr: Any,
                env: Mapping[str, str],
            ) -> SimpleNamespace:
                invoked.update({"argv": tuple(argv), "env": dict(env)})
                stdout.write(b"hidden judge output\n")
                stderr.write(b"")
                eval_dir = Path(argv[argv.index("--eval_dir") + 1])
                eval_dir.mkdir(parents=True)
                artifact = eval_dir / "result.json"
                artifact.write_text('{"score": 1}\n')
                artifact.chmod(0o666)
                return SimpleNamespace(returncode=0)

            with (
                patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": str(root / "shadow-modules"),
                        "PYTHONHOME": str(root / "shadow-home"),
                        "OPENAI_API_KEY": "solver-secret",
                        "SERPAPI_API_KEY": "browser-secret",
                        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
                        "MINI_AGENT_AUDIT_SECRET": "arbitrary-secret",
                    },
                ),
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "browsecomp-plus"),
                ),
                patch(
                    "mini_agent.cli.harness_identity",
                    return_value={"schema": "fixture"},
                ),
                patch(
                    "mini_agent.benchmarks.web._git",
                    side_effect=[BROWSECOMP_PLUS_REVISION, "", "", ""] * 3,
                ),
                patch("mini_agent.grading.subprocess.run", side_effect=run_grader),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "browsecomp-plus",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--checkout",
                        str(checkout),
                        "--ground-truth",
                        str(truth),
                        "--qrel-evidence",
                        str(qrels),
                        "--judge-model",
                        str(judge_model),
                    ]
                )

            self.assertEqual(code, 0)
            argv = invoked["argv"]
            self.assertEqual(invoked["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertNotIn("PYTHONPATH", invoked["env"])
            self.assertNotIn("PYTHONHOME", invoked["env"])
            self.assertNotIn("OPENAI_API_KEY", invoked["env"])
            self.assertNotIn("SERPAPI_API_KEY", invoked["env"])
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", invoked["env"])
            self.assertNotIn("MINI_AGENT_AUDIT_SECRET", invoked["env"])
            self.assertEqual(invoked["env"]["HOME"], str(grade.resolve()))
            self.assertEqual(argv[1], "-I")
            for option in ("--input_dir", "--ground_truth", "--qrel_evidence"):
                value = Path(argv[argv.index(option) + 1])
                self.assertTrue(value.is_relative_to(grade / "inputs"))
            self.assertTrue(
                Path(argv[argv.index("--eval_dir") + 1]).is_relative_to(grade)
            )
            manifest = json.loads((grade / "manifest.json").read_text())
            self.assertEqual(
                manifest["runtime"]["environment"]["policy"], "allowlist-v1"
            )
            self.assertNotIn("solver-secret", json.dumps(manifest))
            self.assertEqual(manifest["grader"]["revision"], BROWSECOMP_PLUS_REVISION)
            self.assertEqual(
                manifest["grader"]["dependency_lock"]["sha256"],
                hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["grader"]["judge_model"]["file_count"], 1)
            result = json.loads((grade / "result.json").read_text())
            self.assertEqual(
                result["verified_grader_assets"]["judge_model"]["sha256"],
                manifest["grader"]["judge_model"]["sha256"],
            )
            self.assertEqual(
                result["artifacts"]["files"][0]["path"],
                "official_evals/result.json",
            )
            self.assertEqual(
                stat.S_IMODE((grade / "official_evals" / "result.json").stat().st_mode),
                0o600,
            )

    def test_browsecomp_plus_grader_assets_fail_closed_on_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "evaluate_run.py"
            lock = root / "uv.lock"
            model = root / "judge-model"
            model.mkdir()
            weights = model / "weights.bin"
            script.write_text("# grader\n")
            lock.write_text("# lock\n")
            weights.write_bytes(b"original weights")
            grader = {
                "grader_script": {
                    "path": str(script.resolve()),
                    "size_bytes": script.stat().st_size,
                    "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                },
                "dependency_lock": {
                    "path": str(lock.resolve()),
                    "size_bytes": lock.stat().st_size,
                    "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                },
                "judge_model": {
                    "path": str(model.resolve()),
                    "file_count": 1,
                    "size_bytes": weights.stat().st_size,
                    "sha256": hashlib.sha256(
                        json.dumps(
                            [
                                {
                                    "path": "weights.bin",
                                    "size_bytes": weights.stat().st_size,
                                    "sha256": hashlib.sha256(
                                        weights.read_bytes()
                                    ).hexdigest(),
                                }
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                },
            }
            _verify_browsecomp_plus_grader_assets(grader)

            script.write_text("# changed grader\n")
            with self.assertRaisesRegex(RuntimeError, "changed during grading"):
                _verify_browsecomp_plus_grader_assets(grader)

            script.write_text("# grader\n")
            weights.write_bytes(b"changed weights")
            with self.assertRaisesRegex(RuntimeError, "changed during grading"):
                _verify_browsecomp_plus_grader_assets(grader)

    def test_browsecomp_plus_grade_does_not_commit_mutated_grader(self) -> None:
        from mini_agent.benchmarks.web import (
            BROWSECOMP_PLUS_QUERY,
            BROWSECOMP_PLUS_REVISION,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            grade = root / "grade"
            task_id = "7"
            question = "Which answer?"
            _write_evaluation_manifest(
                evaluation,
                "browsecomp-plus",
                [(task_id, BROWSECOMP_PLUS_QUERY.format(Question=question))],
            )
            instance = evaluation / "instances" / "one"
            run = {
                "query_id": task_id,
                "status": "completed",
                "tool_call_counts": {"search": 0, "get_document": 0},
                "retrieved_docids": [],
                "result": [
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": "Exact Answer: alpha\nConfidence: 90%",
                    }
                ],
            }
            instance.mkdir(parents=True)
            run_path = instance / "browsecomp_plus_run.json"
            run_path.write_text(json.dumps(run))
            _commit_fixture_result(
                instance,
                task_id,
                {
                    "browsecomp_plus_run_sha256": hashlib.sha256(
                        run_path.read_bytes()
                    ).hexdigest()
                },
            )
            truth = root / "ground_truth.jsonl"
            truth.write_text(
                json.dumps(
                    {"query_id": task_id, "query": question, "answer": "alpha"}
                )
                + "\n"
            )
            qrels = root / "qrels.txt"
            qrels.write_text(f"{task_id} 0 123 1\n")
            judge_model = root / "judge-model"
            judge_model.mkdir()
            weights = judge_model / "weights.bin"
            weights.write_bytes(b"weights")
            checkout = root / "checkout"
            script = checkout / "scripts_evaluation" / "evaluate_run.py"
            script.parent.mkdir(parents=True)
            script.write_text("# grader\n")
            (checkout / "uv.lock").write_text("# exact lock\n")

            def mutate_grader(*args: Any, **kwargs: Any) -> SimpleNamespace:
                script.write_text("# changed grader\n")
                weights.write_bytes(b"changed weights")
                return SimpleNamespace(returncode=0)

            stderr = io.StringIO()
            with (
                patch(
                    "mini_agent.grading._grader_runtime_identity",
                    return_value=_fixture_grader_runtime(root, "browsecomp-plus"),
                ),
                patch(
                    "mini_agent.cli.harness_identity",
                    return_value={"schema": "fixture"},
                ),
                patch(
                    "mini_agent.benchmarks.web._git",
                    side_effect=[BROWSECOMP_PLUS_REVISION, "", "", ""] * 3,
                ),
                patch("mini_agent.grading.subprocess.run", side_effect=mutate_grader),
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "browsecomp-plus",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(grade),
                        "--checkout",
                        str(checkout),
                        "--ground-truth",
                        str(truth),
                        "--qrel-evidence",
                        str(qrels),
                        "--judge-model",
                        str(judge_model),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertIn("changed during grading", stderr.getvalue())
            self.assertFalse((grade / "completed.json").exists())

    def test_grade_rejects_tampered_manifest_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            output = root / "grade"
            _write_evaluation_manifest(
                evaluation, "swebench", [("task", "original prompt")]
            )
            manifest_path = evaluation / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["tasks"][0]["prompt_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "grade",
                        "--benchmark",
                        "swebench",
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(output),
                        "--dataset",
                        str(root / "missing.jsonl"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("fingerprint does not match", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_grade_rejects_inputs_with_a_different_visible_prompt(self) -> None:
        manifest = {
            "tasks": [
                {
                    "id": "task",
                    "prompt_sha256": hashlib.sha256(b"expected").hexdigest(),
                }
            ]
        }
        inputs = {
            "prediction_count": 1,
            "task_prompt_sha256": {"task": hashlib.sha256(b"different").hexdigest()},
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            _verify_grade_prompt_binding(manifest, inputs)

    def test_grade_rejects_a_strict_subset_of_evaluation_tasks(self) -> None:
        prompt_hash = hashlib.sha256(b"expected").hexdigest()
        manifest = {
            "tasks": [
                {"id": "task-one", "prompt_sha256": prompt_hash},
                {"id": "task-two", "prompt_sha256": prompt_hash},
            ]
        }
        inputs = {
            "prediction_count": 1,
            "task_prompt_sha256": {"task-one": prompt_hash},
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            _verify_grade_prompt_binding(manifest, inputs)

    def test_swe_grade_rejects_same_prompt_with_different_hidden_task_data(
        self,
    ) -> None:
        from mini_agent.benchmarks.swebench import (
            inspect_swebench_grade_inputs,
            load_swebench,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.json"
            changed = root / "changed.json"
            predictions = root / "predictions.jsonl"
            original_row = {
                "instance_id": "repo__issue-1",
                "problem_statement": "same visible prompt",
                "test_patch": "ORIGINAL HIDDEN TEST",
            }
            changed_row = {**original_row, "test_patch": "DIFFERENT HIDDEN TEST"}
            original.write_text(json.dumps([original_row]))
            changed.write_text(json.dumps([changed_row]))
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
            task = load_swebench(original)[0]
            manifest = {
                "tasks": [
                    {
                        "id": task.task_id,
                        "prompt_sha256": hashlib.sha256(
                            task.prompt.encode("utf-8")
                        ).hexdigest(),
                        "data_sha256": hashlib.sha256(
                            json.dumps(
                                task.data,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                ]
            }
            inputs = inspect_swebench_grade_inputs(
                predictions=predictions, dataset=changed
            )

            with self.assertRaisesRegex(ValueError, "does not match evaluation"):
                _verify_grade_prompt_binding(manifest, inputs)

    def test_grader_runtime_uses_isolated_python_and_allowlisted_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "HOME": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            isolated = _fixture_grader_runtime(root, "browsecomp-plus")
            observed: dict[str, Any] = {}

            def run_probe(argv: Sequence[str], **kwargs: Any) -> SimpleNamespace:
                observed.update({"argv": tuple(argv), **kwargs})
                output = Path(argv[4])
                output.write_text(
                    json.dumps(
                        {
                            "schema": "mini-agent-isolated-grader-runtime-v1",
                            "ok": True,
                            **{
                                key: isolated[key]
                                for key in (
                                    "python_executable",
                                    "python_version",
                                    "python_implementation",
                                    "python_prefix",
                                    "python_base_prefix",
                                    "packages",
                                    "modules",
                                )
                            },
                        }
                    )
                )
                output.chmod(0o600)
                return SimpleNamespace(returncode=0)

            with (
                patch("mini_agent.grading.subprocess.run", side_effect=run_probe),
                patch(
                    "importlib.metadata.version",
                    return_value="parent-shadow-version",
                ),
            ):
                identity = _grader_runtime_identity(
                    sys.executable,
                    "browsecomp-plus",
                    grader_environment=environment,
                )

            self.assertEqual(identity["packages"], isolated["packages"])
            self.assertEqual(observed["argv"][1:3], ("-I", "-c"))
            self.assertEqual(observed["env"], environment)
            self.assertIs(observed["stdout"], subprocess.DEVNULL)
            self.assertIs(observed["stderr"], subprocess.DEVNULL)
            self.assertEqual(observed["timeout"], 30.0)

    def test_grader_runtime_rejects_isolated_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "HOME": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            def run_probe(argv: Sequence[str], **_kwargs: Any) -> SimpleNamespace:
                output = Path(argv[4])
                output.write_text(
                    json.dumps(
                        {
                            "schema": "mini-agent-isolated-grader-runtime-v1",
                            "ok": False,
                            "error": "version",
                            "name": "numpy",
                            "expected": "1.26.4",
                            "observed": "2.0.0",
                        }
                    )
                )
                output.chmod(0o600)
                return SimpleNamespace(returncode=0)

            with (
                patch("mini_agent.grading.subprocess.run", side_effect=run_probe),
                patch(
                    "importlib.metadata.version",
                    return_value="1.26.4",
                ),
                self.assertRaisesRegex(ValueError, "numpy==1.26.4, found 2.0.0"),
            ):
                _grader_runtime_identity(
                    sys.executable,
                    "browsecomp-plus",
                    grader_environment=environment,
                )

    def test_browsecomp_plus_eval_dir_cannot_escape_private_grade_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grade = root / "grade"
            grade.mkdir()
            with self.assertRaisesRegex(ValueError, "private grade output"):
                _grade_eval_directory(grade, root / "public")

    def test_osworld_docker_doctor_fails_when_daemon_is_unavailable(self) -> None:
        factory = SimpleNamespace(
            checkout=SimpleNamespace(as_dict=lambda: {"revision": "fixture"}),
            provenance=lambda: {"factory": "fixture"},
        )
        doctor = SimpleNamespace(
            ok=False,
            as_dict=lambda: {
                "ok": False,
                "runtime": ["docker"],
                "checks": [
                    {
                        "name": "runtime_version",
                        "ok": False,
                        "detail": "docker is unavailable",
                    }
                ],
            },
        )
        stdout = io.StringIO()
        with (
            patch(
                "mini_agent.benchmarks.osworld.UpstreamDesktopFactory",
                return_value=factory,
            ),
            patch(
                "mini_agent.benchmarks.swebench.swebench_doctor",
                new=AsyncMock(return_value=doctor),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "doctor",
                    "--target",
                    "computer",
                    "--checkout",
                    "/fixture/osworld",
                    "--osworld-version",
                    "v1",
                    "--runtime",
                    "docker",
                ]
            )
        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertFalse(report["reports"]["computer"]["runtime"]["ok"])

    def test_storage_layout_separates_disposable_and_durable_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "durable"
            with self.assertRaisesRegex(ValueError, "scratch"):
                StorageLayout(root, root / "runs" / "scratch")
            layout = StorageLayout(root, root / "work")
            self.assertEqual(layout.scratch, (root / "work").resolve())

    def test_storage_layout_does_not_chmod_existing_external_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scratch = root / "shared-scratch"
            scratch.mkdir(mode=0o777)
            scratch.chmod(0o1777)
            StorageLayout(root / "durable", scratch).ensure()
            self.assertEqual(stat.S_IMODE(scratch.stat().st_mode), 0o1777)

    def test_required_assets_reject_symlink_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("asset")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _required_path(link, "--asset")

    def test_durable_output_and_scratch_cannot_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--task",
                        "task",
                        "--model",
                        "openai/test",
                        "--run-id",
                        "same",
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root),
                        "--output",
                        str(root / "same"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("must not overlap", stderr.getvalue())

    def test_profile_verification_contract(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "profile",
                    "--application",
                    "web",
                    "--profile",
                    "default",
                    "--model",
                    "openai/test-model",
                ]
            )
        self.assertEqual(code, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["environment"], "web")
        self.assertEqual(value["fidelity"], "minimal_baseline")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "profile",
                    "--application",
                    "web",
                    "--model",
                    "openai/test-model",
                    "--multi-agent",
                    "--format",
                    "translation-report",
                ]
            )
        translated = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(translated["exact"])
        self.assertEqual(
            [loss["field"] for loss in translated["losses"]],
            ["tool_kind", "tool_result_images", "tool_result_is_error"],
        )
        self.assertEqual(translated["claim_scope"], "declared_fields_only")
        self.assertIn(
            "stop",
            translated["target_spec"]["communication_capabilities"],
        )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "profile",
                    "--application",
                    "web",
                    "--profile",
                    "default",
                    "--model",
                    "openai/ padded",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("model must use", stderr.getvalue())

    def test_direct_web_run_uses_public_environment_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text('{"docid":"1","text":"evidence"}\n')
            output = root / "run"

            def model(*args: object, **kwargs: object) -> ScriptedModel:
                del args, kwargs
                return ScriptedModel([ModelResponse("answer")])

            stdout = io.StringIO()
            with (
                patch("mini_agent.cli.build_model", side_effect=model),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--web-backend",
                        "jsonl",
                        "--corpus",
                        str(corpus),
                        "--task",
                        "answer this",
                        "--model",
                        "openai/test",
                        "--output",
                        str(output),
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root / "scratch"),
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["answer"], "answer")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["environment"], "web")
            self.assertIn("task_sha256", manifest)
            self.assertEqual(manifest["schema"], "mini-agent-run-v2")
            self.assertEqual(len(manifest["agent_spec"]["fingerprint"]), 64)
            self.assertNotIn("system_prompt", manifest["agent_spec"])
            self.assertIn("source_sha256", manifest["harness"])
            self.assertTrue((output / "trace.jsonl").is_file())

    def test_direct_run_redacts_registered_secrets_from_result_and_stdout(self) -> None:
        secret = "fixture-direct-result-secret"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text('{"docid":"1","text":"evidence"}\n')
            output = root / "run"
            stdout = io.StringIO()

            with (
                patch.dict(os.environ, {"FIXTURE_API_KEY": secret}),
                patch(
                    "mini_agent.cli.build_model",
                    return_value=ScriptedModel([ModelResponse("echo " + secret)]),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--web-backend",
                        "jsonl",
                        "--corpus",
                        str(corpus),
                        "--task",
                        "answer this",
                        "--model",
                        "openai/test",
                        "--api-key-env",
                        "FIXTURE_API_KEY",
                        "--output",
                        str(output),
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root / "scratch"),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertNotIn(secret, (output / "result.json").read_text())
            self.assertNotIn(secret, stdout.getvalue())
            self.assertIn("<redacted>", stdout.getvalue())

    def test_cost_budget_requires_explicit_prices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--web-backend",
                        "jsonl",
                        "--corpus",
                        str(Path(temporary) / "missing"),
                        "--task",
                        "task",
                        "--model",
                        "openai/test",
                        "--max-cost-usd",
                        "1",
                        "--output",
                        str(Path(temporary) / "run"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("requires --input-price", stderr.getvalue())

    def test_provider_body_rejects_embedded_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--task",
                        "task",
                        "--model",
                        "openai/test",
                        "--provider-body",
                        '{"api_key":"leak"}',
                        "--output",
                        str(Path(temporary) / "run"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("secrets belong", stderr.getvalue())
            self.assertNotIn("leak", stderr.getvalue())

    def test_credentials_in_endpoint_or_env_name_fail_before_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for option, value in (
                ("--base-url", "https://user:do-not-persist@example.test/v1"),
                ("--api-key-env", "actual-secret-not-an-env-name"),
            ):
                output = root / option.removeprefix("--")
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            "run",
                            "--environment",
                            "web",
                            "--task",
                            "task",
                            "--model",
                            "openai/test",
                            option,
                            value,
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertFalse(output.exists())
                self.assertNotIn("do-not-persist", stderr.getvalue())
                self.assertNotIn("actual-secret", stderr.getvalue())

    def test_single_agent_rejects_an_ignored_per_agent_limit(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "run",
                    "--environment",
                    "web",
                    "--task",
                    "task",
                    "--model",
                    "openai/test",
                    "--per-agent-model-calls",
                    "1",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("requires --multi-agent", stderr.getvalue())

    def test_benchmark_content_capture_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "swebench",
                        "--dataset",
                        str(Path(temporary) / "dataset.jsonl"),
                        "--model",
                        "openai/test",
                        "--capture-content",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("content-redacted", stderr.getvalue())

    def test_storage_doctor_is_non_paid_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "doctor",
                        "--target",
                        "storage",
                        "--home",
                        str(Path(temporary) / "home"),
                        "--scratch",
                        str(Path(temporary) / "scratch"),
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(stdout.getvalue())
            self.assertTrue(report["ok"])
            self.assertEqual(report["reports"]["storage"]["status"], "ready")

    def test_live_web_doctor_checks_only_live_dependencies(self) -> None:
        stdout = io.StringIO()
        with (
            patch.dict("os.environ", {"SERPAPI_API_KEY": ""}, clear=False),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(["doctor", "--target", "web", "--web-mode", "live"])
        self.assertEqual(code, 1)
        report = json.loads(stdout.getvalue())
        self.assertEqual(set(report["reports"]), {"web"})
        self.assertEqual(report["reports"]["web"]["mode"], "browsecomp_live")
        self.assertFalse(report["reports"]["web"]["serpapi"]["ok"])
        self.assertNotIn("packages", report["reports"]["web"])

    def test_fixed_web_doctor_reports_unavailable_packages_without_crashing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index"
            index.mkdir()
            (index / "segments_1").write_bytes(b"fixture")
            stdout = io.StringIO()
            with (
                patch("importlib.util.find_spec", side_effect=ModuleNotFoundError),
                patch("mini_agent.doctor.shutil.which", return_value=None),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "doctor",
                        "--target",
                        "web",
                        "--web-mode",
                        "fixed",
                        "--index",
                        str(index),
                    ]
                )
            self.assertEqual(code, 1)
            report = json.loads(stdout.getvalue())["reports"]["web"]
            self.assertEqual(
                report["packages"],
                {
                    "huggingface-hub": False,
                    "pyjnius": False,
                    "tokenizers": False,
                },
            )
            self.assertEqual(
                report["package_versions"]["pyjnius"],
                {"ok": False, "observed": None, "required": "1.6.1"},
            )
            self.assertTrue(report["index"]["ok"])
            self.assertFalse(report["anserini_jar"]["ok"])

    def test_fixed_web_doctor_requires_java_21(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index"
            index.mkdir()
            (index / "segments_1").write_bytes(b"fixture")
            stdout = io.StringIO()
            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("mini_agent.doctor.shutil.which", return_value="/usr/bin/java"),
                patch(
                    "mini_agent.doctor.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="",
                        stderr='openjdk version "17.0.15" 2025-04-15\n',
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "doctor",
                        "--target",
                        "web",
                        "--web-mode",
                        "fixed",
                        "--index",
                        str(index),
                    ]
                )
            self.assertEqual(code, 1)
            java = json.loads(stdout.getvalue())["reports"]["web"]["java"]
            self.assertEqual(java["major"], 17)
            self.assertEqual(java["required_major"], 21)
            self.assertFalse(java["ok"])

    def test_eval_defaults_to_one_canary(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "--benchmark",
                "swebench",
                "--model",
                "openai/test",
            ]
        )
        self.assertEqual(args.limit, 1)
        self.assertFalse(args.all)

    def test_browsecomp_plus_eval_rejects_noncanonical_retrieval_limits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "queries.tsv"
            dataset.write_text("q1\tfixture question\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "browsecomp-plus",
                        "--model",
                        "openai/test",
                        "--dataset",
                        str(dataset),
                        "--index",
                        str(root / "index"),
                        "--anserini-jar",
                        str(root / "anserini.jar"),
                        "--top-k",
                        "6",
                        "--snippet-tokenizer-revision",
                        "a" * 40,
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root / "scratch"),
                        "--output",
                        str(root / "output"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn(
                "requires --top-k 5 and --snippet-tokens 512",
                stderr.getvalue(),
            )

    def test_browsecomp_plus_eval_rejects_noncanonical_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "queries.tsv"
            dataset.write_text("q1\tfixture question\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "browsecomp-plus",
                        "--model",
                        "openai/test",
                        "--dataset",
                        str(dataset),
                        "--index",
                        str(root / "index"),
                        "--anserini-jar",
                        str(root / "anserini.jar"),
                        "--snippet-tokenizer",
                        "other/tokenizer",
                        "--snippet-tokenizer-revision",
                        "a" * 40,
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root / "scratch"),
                        "--output",
                        str(root / "output"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn(
                "requires --snippet-tokenizer Qwen/Qwen3-0.6B",
                stderr.getvalue(),
            )

    def test_browsecomp_plus_eval_rejects_index_mutation_during_run(self) -> None:
        class MutatingRunner:
            def __init__(self, **kwargs: Any) -> None:
                del kwargs

            async def run(self, worker: Any, *, resume: bool) -> Mapping[str, int]:
                del worker, resume
                index_file.write_bytes(b"mutated")
                return {"completed": 1, "failed": 0, "blocked": 0}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "queries.tsv"
            dataset.write_text("q1\tfixture question\n", encoding="utf-8")
            index = root / "index"
            index.mkdir()
            index_file = index / "segments_1"
            index_file.write_bytes(b"original")
            jar = root / "anserini.jar"
            jar.write_bytes(b"fixture")
            stderr = io.StringIO()
            with (
                patch("mini_agent.cli.EvaluationRunner", MutatingRunner),
                patch(
                    "mini_agent.environments.web.validate_anserini_jar",
                    return_value=(jar.resolve(), "a" * 64),
                ),
                patch("mini_agent.cli._load_tokenizer", return_value=object()),
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "browsecomp-plus",
                        "--model",
                        "openai/test",
                        "--dataset",
                        str(dataset),
                        "--index",
                        str(index),
                        "--anserini-jar",
                        str(jar),
                        "--snippet-tokenizer-revision",
                        "a" * 40,
                        "--home",
                        str(root / "home"),
                        "--scratch",
                        str(root / "scratch"),
                        "--output",
                        str(root / "output"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("Lucene index changed", stderr.getvalue())
            self.assertFalse((root / "output" / "official_runs").exists())

    def test_evaluation_manifest_contains_only_its_adapter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = StorageLayout(root / "home", root / "scratch")
            swe = build_parser().parse_args(
                [
                    "eval",
                    "--benchmark",
                    "swebench",
                    "--model",
                    "openai/test",
                ]
            )
            swe_config = _evaluation_config(swe, layout)
            self.assertEqual(
                swe_config["adapter"],
                {"runtime": "docker", "container_runtime": ["docker"]},
            )
            self.assertNotIn("page_reader", swe_config)
            self.assertNotIn("provider_name", swe_config)

            swe._swebench_image_bindings = {
                "repo__issue-1": {
                    "runtime": "docker",
                    "requested": "repo/image:latest",
                    "identity": "sha256:" + "a" * 64,
                }
            }
            bound_swe_config = _evaluation_config(swe, layout)
            self.assertEqual(
                bound_swe_config["adapter"]["image_bindings"],
                swe._swebench_image_bindings,
            )

            web = build_parser().parse_args(
                [
                    "eval",
                    "--benchmark",
                    "browsecomp",
                    "--model",
                    "openai/test",
                    "--grader-model",
                    "openai/judge",
                    "--grader-input-price",
                    "1",
                    "--grader-output-price",
                    "2",
                ]
            )
            web_config = _evaluation_config(web, layout)
            grader = web_config["adapter"]["grader"]
            self.assertEqual(grader["pricing"]["input_per_million"], 1.0)
            self.assertEqual(grader["pricing"]["output_per_million"], 2.0)
            self.assertNotIn("runtime", web_config["adapter"])

            fixed = build_parser().parse_args(
                [
                    "eval",
                    "--benchmark",
                    "browsecomp-plus",
                    "--model",
                    "openai/test",
                    "--snippet-tokenizer-revision",
                    "a" * 40,
                ]
            )
            fixed._resolved_snippet_tokenizer_revision = "a" * 40
            fixed._snippet_tokenizer_json_sha256 = "b" * 64
            fixed_config = _evaluation_config(fixed, layout)
            self.assertEqual(
                fixed_config["adapter"]["snippet_tokenizer_json_sha256"],
                "b" * 64,
            )
            self.assertEqual(fixed_config["adapter"]["actions"], ["search"])
            self.assertEqual(
                fixed_config["adapter"]["index_repository_revision"],
                "b3f37f70c33829eb09d04784a54277a31871fd63",
            )
            self.assertIsNone(fixed_config["adapter"]["max_observation_chars"])
            self.assertEqual(
                fixed_config["adapter"]["upstream_query_template"],
                "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            )
            self.assertNotIn("grader", fixed_config["adapter"])

    def test_fixed_evaluation_requires_exact_tokenizer_commit(self) -> None:
        args = SimpleNamespace(
            snippet_tokenizer="Qwen/Qwen3-0.6B",
            snippet_tokenizer_revision=None,
        )
        with self.assertRaisesRegex(ValueError, "40-character commit"):
            _load_tokenizer(args, require_revision=True)

    def test_fixed_tokenizer_loads_only_pinned_json_artifact(self) -> None:
        revision = "a" * 40

        class Encoding:
            ids = [1, 2, 3]

        class Backend:
            def encode(self, text: str, *, add_special_tokens: bool) -> Encoding:
                self.encoded = (text, add_special_tokens)
                return Encoding()

            def decode(
                self, tokens: Sequence[Any], *, skip_special_tokens: bool
            ) -> str:
                self.decoded = (list(tokens), skip_special_tokens)
                return "decoded"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshots" / revision / "tokenizer.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{"version":"1.0"}')
            backend = Backend()
            hub = ModuleType("huggingface_hub")
            tokens = ModuleType("tokenizers")

            def download(**kwargs: Any) -> str:
                self.assertEqual(
                    kwargs,
                    {
                        "repo_id": "Qwen/Qwen3-0.6B",
                        "filename": "tokenizer.json",
                        "revision": revision,
                    },
                )
                return str(path)

            class Tokenizer:
                @staticmethod
                def from_str(value: str) -> Backend:
                    self.assertEqual(value, '{"version":"1.0"}')
                    return backend

            hub.hf_hub_download = download  # type: ignore[attr-defined]
            tokens.Tokenizer = Tokenizer  # type: ignore[attr-defined]
            args = SimpleNamespace(
                snippet_tokenizer="Qwen/Qwen3-0.6B",
                snippet_tokenizer_revision=revision,
            )
            with patch.dict(
                sys.modules,
                {"huggingface_hub": hub, "tokenizers": tokens},
            ), patch(
                "mini_agent.environments.web.importlib.metadata.version",
                side_effect=lambda name: {
                    "huggingface-hub": "0.33.4",
                    "tokenizers": "0.21.2",
                }[name],
            ):
                tokenizer = _load_tokenizer(args, require_revision=True)

        self.assertIsInstance(tokenizer, SnippetTokenizerAdapter)
        self.assertEqual(tokenizer.encode("text", add_special_tokens=False), [1, 2, 3])
        self.assertEqual(tokenizer.decode([1, 2], skip_special_tokens=True), "decoded")
        self.assertEqual(args._resolved_snippet_tokenizer_revision, revision)
        self.assertEqual(
            args._snippet_tokenizer_json_sha256,
            "c2823fb776dfaab48bfa06a33005d02a60492d87762cdb66c9c4155f97fbaa5d",
        )

    def test_fixed_tokenizer_rejects_dependency_drift(self) -> None:
        args = SimpleNamespace(
            snippet_tokenizer="Qwen/Qwen3-0.6B",
            snippet_tokenizer_revision="a" * 40,
        )
        with patch(
            "mini_agent.environments.web.importlib.metadata.version",
            return_value="unexpected",
        ):
            with self.assertRaisesRegex(RuntimeError, "huggingface-hub==0.33.4"):
                _load_tokenizer(args, require_revision=True)

    def test_browsecomp_grader_has_an_independent_provider_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "--benchmark",
                "browsecomp",
                "--model",
                "meta/solver",
                "--base-url",
                "https://solver.example/v1",
                "--expected-provider-model",
                "solver-snapshot",
                "--api-key-env",
                "SOLVER_KEY",
                "--grader-model",
                "anthropic/judge",
                "--grader-base-url",
                "https://judge.example/v1",
                "--grader-expected-provider-model",
                "judge-snapshot",
                "--grader-api-key-env",
                "JUDGE_KEY",
            ]
        )
        sentinel = ScriptedModel([ModelResponse("unused")])
        with patch("mini_agent.cli.build_model", return_value=sentinel) as build:
            returned = _grader_model_factory(args)("/eval/task/grader")
        self.assertIs(returned, sentinel)
        self.assertEqual(build.call_args.args[0], "anthropic/judge")
        self.assertEqual(build.call_args.kwargs["base_url"], "https://judge.example/v1")
        self.assertEqual(build.call_args.kwargs["api_key_env"], "JUDGE_KEY")
        self.assertEqual(
            build.call_args.kwargs["expected_resolved_model"], "judge-snapshot"
        )

    def test_protocol_and_provider_headers_reach_the_model_factory(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--environment",
                "swe",
                "--task",
                "demo task",
                "--model",
                "meta/test-model",
                "--base-url",
                "https://meta.example/v1",
                "--protocol",
                "chat-completions",
                "--expected-provider-model",
                "resolved-snapshot",
                "--provider-header",
                "x-eval-fixture-id=fixture-run-one",
            ]
        )
        sentinel = ScriptedModel([ModelResponse("unused")])
        with patch("mini_agent.cli.build_model", return_value=sentinel) as build:
            returned = _model_factory(args)("/root")
        self.assertIs(returned, sentinel)
        self.assertEqual(build.call_args.kwargs["protocol"], "chat-completions")
        self.assertEqual(
            build.call_args.kwargs["expected_resolved_model"],
            "resolved-snapshot",
        )
        self.assertEqual(
            build.call_args.kwargs["default_headers"],
            {"x-eval-fixture-id": "fixture-run-one"},
        )

    def test_expected_provider_model_is_validated_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            for extra, message in (
                (["--expected-provider-model", " "], "non-empty"),
                (
                    ["--grader-expected-provider-model", "judge-snapshot"],
                    "requires --grader-model",
                ),
            ):
                with self.subTest(extra=extra), contextlib.redirect_stderr(
                    io.StringIO()
                ) as stderr:
                    code = main(
                        [
                            "eval",
                            "--benchmark",
                            "browsecomp-plus",
                            "--model",
                            "openai/test",
                            "--output",
                            str(output),
                            *extra,
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertIn(message, stderr.getvalue())
                self.assertFalse(output.exists())

    def test_provider_headers_reject_credentials_and_malformed_entries(self) -> None:
        self.assertIsNone(_provider_headers([]))
        self.assertEqual(
            _provider_headers(["x-eval-fixture-id=fixture-run-two"]),
            {"x-eval-fixture-id": "fixture-run-two"},
        )
        with self.assertRaisesRegex(ValueError, "api-key-env"):
            _provider_headers(["Authorization=Bearer sneak"])
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            _provider_headers(["no-separator"])
        with self.assertRaisesRegex(ValueError, "repeats"):
            _provider_headers(["x-a=1", "x-a=2"])
        with self.assertRaisesRegex(ValueError, "repeats"):
            _provider_headers(["X-A=1", "x-a=2"])

    def test_provider_header_values_are_hashed_into_execution_identity(self) -> None:
        common = [
            "run",
            "--environment",
            "swe",
            "--task",
            "demo",
            "--model",
            "openai/test",
        ]
        first = build_parser().parse_args(
            [*common, "--provider-header", "X-Eval-Fixture-ID=first-private-value"]
        )
        second = build_parser().parse_args(
            [*common, "--provider-header", "x-eval-fixture-id=second-private-value"]
        )
        first_config = _execution_config(first)
        second_config = _execution_config(second)
        self.assertNotEqual(
            first_config["provider_headers"], second_config["provider_headers"]
        )
        encoded = json.dumps(first_config)
        self.assertNotIn("first-private-value", encoded)
        self.assertIn("x-eval-fixture-id", encoded)


class CLILifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_run_retains_agent_and_cleanup_failures(self) -> None:
        class BrokenEnvironment(BaseEnvironment):
            def tools(self) -> tuple[ToolDefinition, ...]:
                return ()

            async def close(self) -> None:
                raise RuntimeError("cleanup exploded")

        with self.assertRaisesRegex(
            RuntimeError, "no response.*cleanup also failed.*cleanup exploded"
        ):
            await _run_single_agent(
                BrokenEnvironment(),
                model_factory=lambda agent_id: ScriptedModel([]),
                system_prompt="",
                max_steps=1,
                context=RunContext(),
                task="task",
                label="fixture",
            )


class TransportHardeningCLITests(unittest.TestCase):
    def test_transport_flags_reach_build_model(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "--benchmark",
                "browsecomp",
                "--model",
                "openai/test",
                "--protocol",
                "chat-completions",
                "--provider-retries",
                "2",
                "--provider-timeout",
                "60",
                "--max-history-images",
                "3",
                "--grader-model",
                "openai/judge",
                "--grader-protocol",
                "chat-completions",
                "--grader-max-history-images",
                "unlimited",
            ]
        )
        sentinel = ScriptedModel([ModelResponse("unused")])
        with patch("mini_agent.cli.build_model", return_value=sentinel) as build:
            _model_factory(args)("/root")
        self.assertEqual(build.call_args.kwargs["max_retries"], 2)
        self.assertEqual(build.call_args.kwargs["timeout_seconds"], 60.0)
        self.assertEqual(build.call_args.kwargs["max_history_images"], 3)
        with patch("mini_agent.cli.build_model", return_value=sentinel) as build:
            _grader_model_factory(args)("/eval/x/grader")
        self.assertIsNone(build.call_args.kwargs["max_history_images"])
        config = _execution_config(args)
        self.assertEqual(config["provider_retries"], 2)
        self.assertEqual(config["provider_timeout"], 60.0)
        self.assertEqual(config["max_history_images"], 3)

    def test_swe_run_requires_an_explicit_workspace(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "run",
                    "--environment",
                    "swe",
                    "--task",
                    "demo",
                    "--model",
                    "openai/test",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("--workspace is required for swe runs", stderr.getvalue())

    def test_missing_model_credential_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
                contextlib.redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--environment",
                        "web",
                        "--task",
                        "demo",
                        "--model",
                        "openai/test",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("OPENAI_API_KEY is unset or empty", stderr.getvalue())
            self.assertFalse(output.exists())


class CLIEvalEndToEndTests(unittest.TestCase):
    """Full happy paths: argparse -> _evaluate -> worker -> agent -> artifacts.

    Model transports and machine planes are faked; the CLI worker wiring,
    spec binding, benchmark adapters, evaluation runner, and artifact
    contracts all run for real.
    """

    def _storage_args(self, root: Path) -> list[str]:
        return [
            "--home",
            str(root / "home"),
            "--scratch",
            str(root / "scratch"),
            "--output",
            str(root / "output"),
        ]

    @staticmethod
    def _instance(output: Path, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return output / "instances" / digest

    def _assert_completed(self, output: Path, task_id: str) -> Mapping[str, Any]:
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn("fingerprint", manifest["config"]["agent_spec"])
        instance = self._instance(output, task_id)
        self.assertTrue((instance / "completed.json").is_file())
        result = json.loads((instance / "result.json").read_text())
        self.assertEqual(result["status"], "completed")
        return result

    def test_swebench_eval_completes_end_to_end(self) -> None:
        from test_benchmarks import FakeSWEEnvironment

        from mini_agent.benchmarks.swebench import SWEbenchImageBinding

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "verified.jsonl"
            dataset.write_text("{}\n")
            from mini_agent.benchmarks.base import BenchmarkTask

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
            stdout = io.StringIO()
            with (
                patch(
                    "mini_agent.benchmarks.swebench.load_swebench",
                    return_value=(task,),
                ),
                patch(
                    "mini_agent.benchmarks.swebench.prepare_swebench_image_bindings",
                    AsyncMock(return_value={task.task_id: binding}),
                ),
                patch(
                    "mini_agent.benchmarks.swebench.docker_swe_environment",
                    AsyncMock(side_effect=lambda *a, **k: FakeSWEEnvironment()),
                ),
                patch(
                    "mini_agent.cli.build_model",
                    side_effect=lambda *a, **k: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "swebench",
                        "--model",
                        "openai/test",
                        "--dataset",
                        str(dataset),
                        *self._storage_args(root),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            self._assert_completed(output, task.task_id)
            predictions = (output / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(predictions), 1)
            self.assertEqual(
                json.loads(predictions[0])["instance_id"], "repo__issue-1"
            )

    def test_programbench_eval_completes_end_to_end(self) -> None:
        from test_benchmarks import (
            FakeProgramBenchEnvironment,
            programbench_fixture_checkout,
            programbench_fixture_git,
        )

        from mini_agent.benchmarks.programbench import PROGRAMBENCH_REVISION
        from mini_agent.benchmarks.swebench import SWEbenchImageBinding

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = programbench_fixture_checkout(root)
            task_id = "org__tool.abc1234"
            binding = SWEbenchImageBinding(
                runtime="docker",
                requested="programbench/org_1776_tool.abc1234:task_cleanroom_v6",
                identity="sha256:" + "a" * 64,
                execution_ref="sha256:" + "a" * 64,
            )
            stdout = io.StringIO()
            with (
                patch(
                    "mini_agent.benchmarks.programbench._git",
                    side_effect=programbench_fixture_git(checkout.resolve()),
                ),
                patch(
                    "mini_agent.benchmarks.swebench.prepare_swebench_image_bindings",
                    AsyncMock(return_value={task_id: binding}),
                ),
                patch(
                    "mini_agent.benchmarks.programbench.docker_swe_environment",
                    AsyncMock(
                        side_effect=lambda *a, **k: FakeProgramBenchEnvironment()
                    ),
                ),
                patch(
                    "mini_agent.cli.build_model",
                    side_effect=lambda *a, **k: ScriptedModel(
                        [ModelResponse("done")]
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "programbench",
                        "--model",
                        "openai/test",
                        "--checkout",
                        str(checkout),
                        *self._storage_args(root),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            result = self._assert_completed(output, task_id)
            self.assertIsNone(result["score"])
            self.assertEqual(
                result["metadata"]["scoring"], "official-programbench-eval-only"
            )
            submission = output / "official_run" / task_id / "submission.tar.gz"
            self.assertEqual(submission.read_bytes(), b"submission-archive")
            manifest = json.loads((output / "manifest.json").read_text())
            adapter = manifest["config"]["adapter"]
            self.assertEqual(adapter["agent_network"], "none")
            self.assertEqual(adapter["image_tag"], "task_cleanroom_v6")
            self.assertEqual(adapter["scoring"], "official-programbench-eval-only")
            self.assertEqual(adapter["checkout"]["revision"], PROGRAMBENCH_REVISION)
            # The manifest must report the runtime that actually ran, not a
            # constant that happened to be true while only one was reachable.
            self.assertEqual(adapter["runtime"], "docker")
            self.assertEqual(
                adapter["image_bindings"][task_id]["identity"],
                "sha256:" + "a" * 64,
            )
            self.assertRegex(manifest["fingerprint"], r"^[0-9a-f]{64}$")
            self.assertNotIn("test_never_visible", json.dumps(manifest))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["submissions"], 1)
            self.assertIsNone(summary["mean_score"])

    def test_browsecomp_eval_scores_end_to_end(self) -> None:
        from mini_agent.benchmarks.base import BenchmarkTask

        task = BenchmarkTask(
            "browsecomp-0001",
            "What is the answer?",
            {"question": "What is the answer?", "answer": "42"},
        )

        def scripted(spec: str, **kwargs: Any) -> ScriptedModel:
            del kwargs
            if spec == "openai/judge":
                return ScriptedModel(
                    [
                        ModelResponse(
                            "extracted_final_answer: 42\n"
                            "correct: yes\n"
                            "confidence: 100"
                        )
                    ]
                )
            return ScriptedModel([ModelResponse("Exact Answer: 42")])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "browsecomp.csv"
            dataset.write_text("problem,answer,canary\n")
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {"SERPAPI_API_KEY": "fixture-key"}),
                patch(
                    "mini_agent.benchmarks.web.load_browsecomp",
                    return_value=(task,),
                ),
                patch("mini_agent.cli.build_model", side_effect=scripted),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "browsecomp",
                        "--model",
                        "openai/test",
                        "--grader-model",
                        "openai/judge",
                        "--dataset",
                        str(dataset),
                        *self._storage_args(root),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            result = self._assert_completed(output, task.task_id)
            self.assertEqual(result["score"], 1.0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["mean_score"], 1.0)
            grader = json.loads(
                (self._instance(output, task.task_id) / "private" / "grader.json")
                .read_text()
            )
            self.assertTrue(grader["contains_hidden_benchmark_data"])

    def test_browsecomp_plus_eval_completes_end_to_end(self) -> None:
        from support import WordTokenizer

        from mini_agent.environments.web import (
            BrowserEnvironment,
            JsonlSearchBackend,
            directory_sha256,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "queries.tsv"
            dataset.write_text("q1\twhich doc mentions foxes\n", encoding="utf-8")
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"docid": "d1", "text": "quick brown foxes jump"}\n'
                '{"docid": "d2", "text": "lazy dogs sleep"}\n'
            )
            index = root / "index"
            index.mkdir()
            (index / "segments_1").write_bytes(b"fixture-index")
            index_sha = directory_sha256(index)
            jar = root / "anserini.jar"
            jar.write_bytes(b"fixture")

            def fixed_browser(args: Any, tokenizer: Any) -> BrowserEnvironment:
                return BrowserEnvironment(
                    JsonlSearchBackend(corpus),
                    top_k=5,
                    snippet_tokens=512,
                    tokenizer=tokenizer,
                    allow_open=False,
                    max_observation_chars=None,
                )

            def fake_tokenizer(args: Any, **kwargs: Any) -> WordTokenizer:
                args._resolved_snippet_tokenizer_revision = "a" * 40
                args._snippet_tokenizer_json_sha256 = "b" * 64
                return WordTokenizer()

            search = ModelResponse(
                "",
                tool_calls=(
                    ToolCall(
                        "s1",
                        "browser",
                        {"action": "search", "query": "foxes"},
                    ),
                ),
            )
            stdout = io.StringIO()
            with (
                patch(
                    "mini_agent.environments.web.validate_anserini_jar",
                    return_value=(jar.resolve(), "a" * 64),
                ),
                patch("mini_agent.cli._fixed_browser", side_effect=fixed_browser),
                patch("mini_agent.cli._load_tokenizer", side_effect=fake_tokenizer),
                patch(
                    "mini_agent.cli.build_model",
                    side_effect=lambda *a, **k: ScriptedModel(
                        [search, ModelResponse("Exact Answer: d1")]
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "browsecomp-plus",
                        "--model",
                        "openai/test",
                        "--dataset",
                        str(dataset),
                        "--index",
                        str(index),
                        "--index-sha256",
                        index_sha,
                        "--anserini-jar",
                        str(jar),
                        "--snippet-tokenizer-revision",
                        "a" * 40,
                        *self._storage_args(root),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            self._assert_completed(output, "q1")
            run_artifact = json.loads(
                (self._instance(output, "q1") / "browsecomp_plus_run.json")
                .read_text()
            )
            self.assertEqual(run_artifact["query_id"], "q1")

    def test_osworld_eval_completes_end_to_end(self) -> None:
        from test_benchmarks import FakeDesktop

        from mini_agent.benchmarks.base import BenchmarkTask
        from mini_agent.benchmarks.osworld import _config_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "OSWorld"
            config_path = (
                checkout / "evaluation_examples" / "examples" / "writer" / "task.json"
            )
            config_path.parent.mkdir(parents=True)
            task_config = {
                "id": "task",
                "instruction": "click",
                "related_apps": ["writer"],
            }
            config_path.write_text(json.dumps(task_config))
            (checkout / "docker_vm_data").mkdir()
            (checkout / "docker_vm_data" / "Ubuntu.qcow2").write_bytes(b"qcow2")
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
            desktop = FakeDesktop({"score": 1.0, "criteria": {}})

            class FakeFactory:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def provenance(self) -> Mapping[str, Any]:
                    return {"kind": "fixture"}

                async def __call__(self, agent_id: str, cache: Path) -> FakeDesktop:
                    del agent_id, cache
                    return desktop

            step = ModelResponse(
                "",
                tool_calls=(
                    ToolCall(
                        "c1",
                        "computer",
                        {"actions": [{"type": "click", "x": 1, "y": 1}]},
                    ),
                ),
            )
            stdout = io.StringIO()
            with (
                patch(
                    "mini_agent.benchmarks.osworld.load_osworld",
                    return_value=(task,),
                ),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=SimpleNamespace(
                        path=checkout, version="v1", revision="pinned"
                    ),
                ),
                patch(
                    "mini_agent.benchmarks.osworld.UpstreamDesktopFactory",
                    FakeFactory,
                ),
                patch(
                    "mini_agent.cli.build_model",
                    side_effect=lambda *a, **k: ScriptedModel(
                        [step, ModelResponse("done")]
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "osworld-v1",
                        "--model",
                        "openai/test",
                        "--checkout",
                        str(checkout),
                        *self._storage_args(root),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            result = self._assert_completed(output, "task")
            self.assertEqual(result["score"], 1.0)
            self.assertTrue(desktop.closed)
            self.assertTrue(
                (self._instance(output, "task") / "score.json").is_file()
            )

    def test_osworld_v2_eval_completes_end_to_end(self) -> None:
        """The v2 worker path runs with a fake upstream task loader.

        Real v2 task data is gated upstream (HTTP 401); the loader boundary is
        faked while the adapter's v2 class hashing, config binding, TOCTOU
        rechecks, and lifecycle gating all run for real.
        """

        from test_benchmarks import FakeDesktop

        from mini_agent.benchmarks import osworld as osworld_module
        from mini_agent.benchmarks.base import BenchmarkTask
        from mini_agent.benchmarks.osworld import _config_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "OSWorld-V2"
            base = checkout / "evaluation_examples"
            config_path = base / "examples_v2" / "writer" / "task.json"
            config_path.parent.mkdir(parents=True)
            task_config = {
                "id": "task",
                "instruction": "click",
                "related_apps": ["writer"],
            }
            config_path.write_text(json.dumps(task_config))
            class_path = base / "task_class" / "task_task.py"
            class_path.parent.mkdir(parents=True)
            class_path.write_text("class Task:\n    pass\n")
            (checkout / "docker_vm_data").mkdir(parents=True)
            (checkout / "docker_vm_data" / "Ubuntu.qcow2").write_bytes(b"qcow2")

            loader = SimpleNamespace(
                resolve_task_json_path=lambda **kwargs: str(config_path),
                find_task_class_path=lambda **kwargs: str(class_path),
                load_task_config=lambda path, **kwargs: json.loads(
                    Path(path).read_text()
                ),
            )
            desktop = FakeDesktop({"score": 1.0, "criteria": {}})

            class FakeFactory:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def provenance(self) -> Mapping[str, Any]:
                    return {"kind": "fixture"}

                async def __call__(self, agent_id: str, cache: Path) -> FakeDesktop:
                    del agent_id, cache
                    return desktop

            step = ModelResponse(
                "",
                tool_calls=(
                    ToolCall(
                        "c1",
                        "computer",
                        {"actions": [{"type": "click", "x": 1, "y": 1}]},
                    ),
                ),
            )
            stdout = io.StringIO()
            with (
                patch(
                    "mini_agent.benchmarks.osworld._activate_checkout",
                    lambda *a, **k: None,
                ),
                patch(
                    "mini_agent.benchmarks.osworld._import_from_checkout",
                    lambda *a, **k: loader,
                ),
                patch(
                    "mini_agent.benchmarks.osworld.inspect_osworld_checkout",
                    return_value=SimpleNamespace(
                        path=checkout, version="v2", revision="pinned"
                    ),
                ),
                patch(
                    "mini_agent.benchmarks.osworld.UpstreamDesktopFactory",
                    FakeFactory,
                ),
                patch(
                    "mini_agent.cli.build_model",
                    side_effect=lambda *a, **k: ScriptedModel(
                        [step, ModelResponse("done")]
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                checkout_info = SimpleNamespace(path=checkout, version="v2")
                class_sha = osworld_module._task_class_sha256(
                    checkout_info, "writer", "task"
                )
                self.assertIsInstance(class_sha, str)
                task = BenchmarkTask(
                    "task",
                    "click",
                    {
                        "checkout": str(checkout),
                        "version": "v2",
                        "revision": "pinned",
                        "domain": "writer",
                        "task_config_sha256": _config_sha256(task_config),
                        "task_class_sha256": class_sha,
                    },
                )
                with patch(
                    "mini_agent.benchmarks.osworld.load_osworld",
                    return_value=(task,),
                ):
                    code = main(
                        [
                            "eval",
                            "--benchmark",
                            "osworld-v2",
                            "--model",
                            "openai/test",
                            "--checkout",
                            str(checkout),
                            *self._storage_args(root),
                        ]
                    )
            self.assertEqual(code, 0, stdout.getvalue())
            output = root / "output"
            result = self._assert_completed(output, "task")
            self.assertEqual(result["score"], 1.0)
            self.assertTrue(desktop.closed)


if __name__ == "__main__":
    unittest.main()
