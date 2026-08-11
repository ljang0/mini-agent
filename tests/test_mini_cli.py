from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mini_agent import ModelResponse, ScriptedBackend, ToolCall
from mini_agent.cli import build_parser, main
from mini_agent.evals.cua import ExternalProcessResult


FIXTURES = Path(__file__).parent / "fixtures" / "browsecomp_plus"


def _mini_agent_wheel(root: Path) -> Path:
    wheel = root / "mini_agent-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mini_agent/__init__.py", "__version__ = '0.3.0'\n")
        archive.writestr(
            "mini_agent-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: mini-agent\nVersion: 0.3.0\n",
        )
        archive.writestr(
            "mini_agent-0.3.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: mini-agent-tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("mini_agent-0.3.0.dist-info/RECORD", "")
    return wheel


class MiniAgentPublicCLITests(unittest.TestCase):
    def test_public_command_surface_is_consolidated(self) -> None:
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices
        self.assertEqual(
            set(commands),
            {"run", "eval", "grade", "doctor", "export", "profile", "catalog", "reference"},
        )

    def test_offline_web_eval_writes_standard_resumable_layout(self) -> None:
        backend = ScriptedBackend(
            {
                agent_id: [
                    ModelResponse(
                        text="search",
                        tool_calls=(ToolCall(f"search-{query_id}", "search", {"query": term}),),
                        continuation={"provider": "test"},
                    ),
                    ModelResponse(text=f"answer for {query_id}"),
                ]
                for agent_id, query_id, term in (
                    ("/browsecomp/q1", "q1", "alpha"),
                    ("/browsecomp/q2", "q2", "beta"),
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            stdout = io.StringIO()
            with patch("mini_agent.cli._build_backend", return_value=backend):
                with contextlib.redirect_stdout(stdout):
                    status = main(
                        (
                            "eval",
                            "--application",
                            "web",
                            "--model",
                            "openai/test",
                            "--profile",
                            "default",
                            "--tasks",
                            str(FIXTURES / "tasks.jsonl"),
                            "--corpus",
                            str(FIXTURES / "corpus.jsonl"),
                            "--output",
                            str(output),
                            "--max-workers",
                            "2",
                        )
                    )
            self.assertEqual(status, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["config"]["retrieval_backend"]["backend"],
                "jsonl_bm25_test",
            )
            self.assertEqual(
                len(manifest["config"]["retrieval_backend"]["corpus_sha256"]),
                64,
            )
            self.assertEqual(
                [task["query_id"] for task in manifest["tasks"]], ["q1", "q2"]
            )
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(len(list((output / "official").glob("*.json"))), 2)
            self.assertEqual(len(list((output / "instances").glob("*/result.json"))), 2)

    def test_offline_multi_web_eval_uses_the_same_batch_layout(self) -> None:
        backend = ScriptedBackend(
            {
                "/root": [
                    ModelResponse(
                        text="spawn",
                        tool_calls=(
                            ToolCall("spawn", "spawn_agent", {"task": "search alpha"}),
                        ),
                        continuation={"provider": "test"},
                    ),
                    ModelResponse(
                        text="wait",
                        tool_calls=(
                            ToolCall("wait", "wait", {"agent_ids": ["/root/1"]}),
                        ),
                        continuation={"provider": "test"},
                    ),
                    ModelResponse(text="root answer"),
                ],
                "/root/1": [
                    ModelResponse(
                        text="search",
                        tool_calls=(
                            ToolCall("search", "search", {"query": "alpha"}),
                        ),
                        continuation={"provider": "test"},
                    ),
                    ModelResponse(text="child finding"),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("mini_agent.cli._build_backend", return_value=backend):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(
                        (
                            "eval",
                            "--application",
                            "web",
                            "--model",
                            "openai/test",
                            "--profile",
                            "default",
                            "--tasks",
                            str(FIXTURES / "tasks.jsonl"),
                            "--instance-id",
                            "q1",
                            "--corpus",
                            str(FIXTURES / "corpus.jsonl"),
                            "--output",
                            str(output),
                            "--mode",
                            "multi",
                            "--max-agents",
                            "2",
                        )
                    )
            self.assertEqual(status, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["mode"], "multi")
            official = json.loads((output / "official" / "q1.json").read_text())
            self.assertEqual(official["tool_call_counts"], {"search": 1})
            self.assertTrue(official["retrieved_docids"])
            trace = (output / "instances" / "q1" / "trace.jsonl").read_text()
            self.assertIn('"event":"agent_spawned"', trace)

    def test_multi_swe_run_captures_the_root_patch_before_cleanup(self) -> None:
        backend = ScriptedBackend(
            {
                "/root": [
                    ModelResponse(
                        text="edit",
                        tool_calls=(
                            ToolCall(
                                "edit",
                                "bash",
                                {"command": "printf changed > value.txt"},
                            ),
                        ),
                        continuation={"provider": "test"},
                    ),
                    ModelResponse(text="done"),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "value.txt").write_text("original", encoding="utf-8")
            output = root / "run"
            with patch("mini_agent.cli._build_backend", return_value=backend):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(
                        (
                            "run",
                            "--application",
                            "swe",
                            "--model",
                            "openai/test",
                            "--profile",
                            "default",
                            "--workspace",
                            str(workspace),
                            "--task",
                            "edit the file",
                            "--mode",
                            "multi",
                            "--max-agents",
                            "1",
                            "--output",
                            str(output),
                        )
                    )
            self.assertEqual(status, 0)
            patch_bytes = (output / "artifacts" / "patch.diff").read_bytes()
            self.assertIn(b"+changed", patch_bytes)
            payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["patch"]["bytes"], len(patch_bytes))
            self.assertEqual(
                (workspace / "value.txt").read_text(encoding="utf-8"), "original"
            )

    def test_doctor_export_catalog_and_reference_list_are_offline(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(
                (
                    "doctor",
                    "--application",
                    "web",
                    "--profile",
                    "default",
                    "--corpus",
                    str(FIXTURES / "corpus.jsonl"),
                )
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "submission"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        (
                            "export",
                            "--target",
                            "cua-speed-run",
                            "--output",
                            str(destination),
                            "--profile",
                            "gpt54",
                            "--model",
                            "gpt-5.4",
                            "--provider",
                            "openai-responses",
                            "--wheel",
                            str(_mini_agent_wheel(root)),
                        )
                    ),
                    0,
                )
            self.assertEqual(sorted(path.name for path in destination.iterdir()), ["agent.py", "init.py"])

        for argv in (
            ("catalog", "--frontiers", "--json"),
            ("reference", "list", "--application", "swe", "--json"),
        ):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(argv), 0)
                self.assertTrue(json.loads(output.getvalue()))

    def test_cua_doctor_blocks_without_a_vm_or_benchmark_target(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(("doctor", "--application", "cua", "--profile", "gpt54"))
        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(payload["status"], "blocked")
        execution = next(
            check for check in payload["checks"] if check["name"] == "execution_target"
        )
        self.assertFalse(execution["ok"])

    def test_doctor_infers_credentials_and_never_exposes_gateway_tokens(self) -> None:
        stdout = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with contextlib.redirect_stdout(stdout):
                status = main(
                    (
                        "doctor",
                        "--application",
                        "web",
                        "--profile",
                        "default",
                        "--model",
                        "openai/test",
                        "--corpus",
                        str(FIXTURES / "corpus.jsonl"),
                    )
                )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 1)
        credentials = next(
            check for check in payload["checks"] if check["name"] == "credentials"
        )
        self.assertFalse(credentials["ok"])
        self.assertEqual(payload["checks"][0]["detail"]["provider"], "openai-responses")

        secret_url = "https://gateway.invalid/super-secret-token?bad=1"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                main(
                    (
                        "doctor",
                        "--application",
                        "cua",
                        "--profile",
                        "default",
                        "--env-url",
                        secret_url,
                    )
                ),
                1,
            )
        self.assertNotIn("super-secret-token", stdout.getvalue())

    def test_cua_eval_propagates_the_external_runner_exit_status(self) -> None:
        failed = ExternalProcessResult(
            argv=("runner",),
            returncode=7,
            stdout="failed",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "mini_agent.evals.cua.run_cua_speed_run_reference",
                new=AsyncMock(return_value=failed),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(
                        (
                            "eval",
                            "--application",
                            "cua",
                            "--output",
                            str(root / "run"),
                            "--checkout",
                            str(root / "checkout"),
                            "--submission",
                            str(root / "submission"),
                            "--benchmark",
                            str(root / "benchmark"),
                        )
                    )
        self.assertEqual(status, 1)

    def test_cua_eval_rejects_generation_options_owned_by_submission(self) -> None:
        base = (
            "eval",
            "--application",
            "cua",
            "--output",
            "/tmp/run",
            "--checkout",
            "/tmp/checkout",
            "--submission",
            "/tmp/submission",
            "--benchmark",
            "/tmp/benchmark",
        )
        for extra in (
            ("--mode", "multi"),
            ("--child-profile", "gpt54"),
            ("--profile", "gpt54"),
            ("--model", "gpt-5.4"),
            ("--resume",),
        ):
            with self.subTest(extra=extra):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main((*base, *extra))
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("exported submission", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
