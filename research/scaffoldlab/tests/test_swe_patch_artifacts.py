import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Tuple
from unittest.mock import patch

from scaffoldlab.environments.configured import ConfiguredEnvironmentFactory
from scaffoldlab.environments.swe import SWEEnvironment
from scaffoldlab.evaluation import MatrixRunner, _atomic_write_bytes
from scaffoldlab.harnesses.base import Harness
from scaffoldlab.runtime import RunContext, ScriptedBackend
from scaffoldlab.types import BudgetLimits, Task


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    return subprocess.run(
        ("git", *args),
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _make_dirty_parent_repository(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "modified.txt").write_text(
        "repository commit baseline\n", encoding="utf-8"
    )
    (source / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    (source / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
    _git(source, "init", "--quiet")
    _git(source, "add", "--all", "--", ".")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@invalid",
        "-c",
        "commit.gpgSign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "parent baseline",
    )
    # These pre-existing dirty/untracked values are part of the copied snapshot and
    # therefore must be clean in Scaffold Lab's independent temporary baseline.
    (source / "modified.txt").write_text("source snapshot\n", encoding="utf-8")
    (source / "baseline-untracked.txt").write_text(
        "belongs to source snapshot\n", encoding="utf-8"
    )
    return source


class _MutatingHarness(Harness):
    name = "swe_patch_test"

    def __init__(self, *, oversized: bool = False) -> None:
        self.oversized = oversized
        self.workspace: Path | None = None
        self.initial_status = ""

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        assert context.environment is not None
        environment = await context.environment.get("/root")
        assert isinstance(environment, SWEEnvironment)
        self.workspace = environment.workspace
        self.initial_status = _git(
            self.workspace, "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout.decode("utf-8")
        if self.oversized:
            (self.workspace / "modified.txt").write_text(
                "OVERSIZED_PRIVATE_CONTENT\n" * 200, encoding="utf-8"
            )
        else:
            (self.workspace / "modified.txt").write_text(
                "TRIAL_SECRET_CONTENT\n", encoding="utf-8"
            )
            (self.workspace / "new.txt").write_text(
                "NEW_TRIAL_SECRET\n", encoding="utf-8"
            )
            (self.workspace / "deleted.txt").unlink()
            (self.workspace / "binary.bin").write_bytes(b"\x00\xff\xfe\xfd\xfc")
        return "done", {}


class SWEPatchArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_matrix_externalizes_complete_patch_before_copy_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_dirty_parent_repository(root)
            output = root / "output"
            factory = ConfiguredEnvironmentFactory(
                {
                    "type": "swe",
                    "workspace": str(source),
                    "workspace_mode": "copy",
                    "isolation": "shared",
                    "allow_write": True,
                    "protocol": "generic",
                    "export_patch": True,
                    "max_patch_bytes": 1024 * 1024,
                }
            )
            harness = _MutatingHarness()
            wrote_while_copy_existed = False

            def checked_write(path: Path, content: bytes) -> None:
                nonlocal wrote_while_copy_existed
                assert harness.workspace is not None
                wrote_while_copy_existed = harness.workspace.exists()
                _atomic_write_bytes(path, content)

            with patch(
                "scaffoldlab.evaluation._atomic_write_bytes",
                side_effect=checked_write,
            ):
                records, _ = await MatrixRunner(
                    backend=ScriptedBackend({"/root": ["unused"]}),
                    limits=BudgetLimits(max_model_calls=1, wall_time_seconds=10),
                    output_dir=output,
                    environment_factory=factory,
                ).run([Task("patch", "mutate")], [harness])

            self.assertEqual(records[0].status, "completed")
            self.assertEqual(harness.initial_status, "")
            self.assertTrue(wrote_while_copy_existed)
            assert harness.workspace is not None
            self.assertFalse(harness.workspace.exists())

            session = records[0].metadata["environment"]["sessions"][0]
            artifact = session["patch_artifact"]
            self.assertEqual(artifact["format"], "git_diff_binary")
            patch_path = Path(artifact["path"])
            patch_bytes = patch_path.read_bytes()
            self.assertEqual(artifact["bytes"], len(patch_bytes))
            self.assertEqual(
                artifact["sha256"], hashlib.sha256(patch_bytes).hexdigest()
            )
            self.assertEqual(patch_path.parent, (output / "patches").resolve())

            patch_text = patch_bytes.decode("utf-8")
            self.assertIn("diff --git a/modified.txt b/modified.txt", patch_text)
            self.assertIn("-source snapshot", patch_text)
            self.assertIn("+TRIAL_SECRET_CONTENT", patch_text)
            self.assertIn("new file mode", patch_text)
            self.assertIn("diff --git a/new.txt b/new.txt", patch_text)
            self.assertIn("deleted file mode", patch_text)
            self.assertIn("diff --git a/deleted.txt b/deleted.txt", patch_text)
            self.assertIn("GIT binary patch", patch_text)
            self.assertNotIn("repository commit baseline", patch_text)
            self.assertNotIn("baseline-untracked.txt", patch_text)

            result_text = (output / "results.jsonl").read_text(encoding="utf-8")
            trace_text = next((output / "traces").glob("*.jsonl")).read_text(
                encoding="utf-8"
            )
            for private_content in ("TRIAL_SECRET_CONTENT", "NEW_TRIAL_SECRET"):
                self.assertNotIn(private_content, result_text)
                self.assertNotIn(private_content, trace_text)
            parsed_record = json.loads(result_text)
            self.assertNotIn("_content", json.dumps(parsed_record))
            self.assertEqual(
                (source / "modified.txt").read_text(encoding="utf-8"),
                "source snapshot\n",
            )

    async def test_patch_size_limit_fails_closed_and_still_cleans_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_dirty_parent_repository(root)
            output = root / "output"
            factory = ConfiguredEnvironmentFactory(
                {
                    "type": "swe",
                    "workspace": str(source),
                    "workspace_mode": "copy",
                    "isolation": "shared",
                    "protocol": "generic",
                    "export_patch": True,
                    "max_patch_bytes": 128,
                }
            )
            harness = _MutatingHarness(oversized=True)
            records, _ = await MatrixRunner(
                backend=ScriptedBackend({"/root": ["unused"]}),
                limits=BudgetLimits(max_model_calls=1, wall_time_seconds=10),
                output_dir=output,
                environment_factory=factory,
            ).run([Task("oversized", "mutate")], [harness])

            self.assertEqual(records[0].status, "error")
            self.assertIn("SWE patch exceeded", records[0].error or "")
            assert harness.workspace is not None
            self.assertFalse(harness.workspace.exists())
            self.assertFalse((output / "patches").exists())
            self.assertNotIn(
                "OVERSIZED_PRIVATE_CONTENT",
                (output / "results.jsonl").read_text(encoding="utf-8"),
            )

    def test_patch_export_config_defaults_and_validation(self) -> None:
        factory = ConfiguredEnvironmentFactory(
            {"type": "swe", "workspace_mode": "copy"}
        )
        self.assertFalse(factory.export_patch)
        self.assertGreater(factory.max_patch_bytes, 0)
        with self.assertRaisesRegex(ValueError, "max_patch_bytes"):
            ConfiguredEnvironmentFactory(
                {"type": "swe", "workspace_mode": "copy", "max_patch_bytes": 0}
            )
        with self.assertRaisesRegex(ValueError, "requires an SWE environment"):
            ConfiguredEnvironmentFactory(
                {
                    "type": "browser",
                    "allowed_hosts": ["example.com"],
                    "export_patch": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "requires workspace_mode='copy'"):
            ConfiguredEnvironmentFactory(
                {
                    "type": "swe",
                    "workspace_mode": "direct",
                    "isolation": "shared",
                    "export_patch": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
