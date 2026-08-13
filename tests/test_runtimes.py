"""Sandbox-runtime tests: rootless Docker, Apptainer overlays, and doctors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from mini_agent.benchmarks.swebench import (
    SWEbenchImageBinding,
    apptainer_swe_environment,
    docker_swe_environment,
    resolve_swebench_image_binding,
    swebench_doctor,
)
from mini_agent.environments.bash import SWEArchiveState
from mini_agent.runtimes.apptainer import materialize_apptainer_image
from mini_agent.runtimes.base import ProcessResult
from mini_agent.types import ProtocolError, ToolCall


TEST_DOCKER_ID = "sha256:" + "a" * 64


class RecordingRunner:
    def __init__(self, *, rootless: bool = True) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.rootless = rootless
        self.pull_calls = 0

    async def run(
        self,
        argv: Sequence[str],
        **kwargs: Any,
    ) -> ProcessResult:
        del kwargs
        call = tuple(argv)
        self.calls.append(call)
        output = b""
        if any("SecurityOptions" in item for item in call):
            output = b'["name=rootless"]\n' if self.rootless else b"[]\n"
        elif call[1:2] == ("version",):
            output = b"26.1\n"
        elif call[1:2] == ("info",):
            output = b"linux/amd64\n"
        elif call[1:3] == ("image", "inspect"):
            output = (TEST_DOCKER_ID + "\n").encode()
        elif call[1:2] == ("inspect",):
            output = (TEST_DOCKER_ID + "\n").encode()
        elif call[1:3] == ("overlay", "create"):
            Path(call[-1]).write_bytes(b"overlay")
        elif call[1:2] == ("pull",) and "--force" in call:
            self.pull_calls += 1
            Path(call[-2]).write_bytes(b"sif-content")
        elif "exec" in call[1:3]:
            output = ("c" * 40 + "\n").encode()
        return ProcessResult(output, 0, len(output))


class ArchiveRecordingRunner(RecordingRunner):
    """Recording runner that also serves workspace archive export/adoption."""

    def __init__(self, archive: bytes = b"workspace-archive") -> None:
        super().__init__()
        self.archive = archive

    async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
        call = tuple(argv)
        if call[1:2] == ("cp",):
            self.calls.append(call)
            if ":" in call[2]:
                Path(call[3]).write_bytes(self.archive)
            return ProcessResult(b"", 0, 0)
        if "exec" in call[1:3] and "tar -czf" in call[-1]:
            self.calls.append(call)
            output = f"{len(self.archive)}\n".encode()
            return ProcessResult(output, 0, len(output))
        return await super().run(argv, **kwargs)


class SandboxRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_container_disables_network_and_exports_an_archive(
        self,
    ) -> None:
        runner = ArchiveRecordingRunner()
        environment = await docker_swe_environment(
            {
                "instance_id": "org__tool.abc1234",
                "image_name": "programbench/org_1776_tool.abc1234:task_cleanroom_v6",
            },
            runner=runner,
            workdir="/workspace",
            network_disabled=True,
            require_git_baseline=False,
            benchmark_identity={"benchmark": "programbench"},
        )
        try:
            start = next(call for call in runner.calls if call[1:2] == ("run",))
            self.assertEqual(start[start.index("--network") + 1], "none")
            self.assertEqual(start[start.index("--workdir") + 1], "/workspace")
            self.assertFalse(
                any("git rev-parse" in call[-1] for call in runner.calls)
            )
            self.assertIsNone(environment.base_commit)
            provenance = environment.provenance()
            self.assertTrue(provenance["network_disabled"])
            self.assertEqual(provenance["benchmark"], "programbench")
            self.assertEqual(provenance["workdir"], "/workspace")
            self.assertEqual(provenance["patch_export"], "workspace_tar_gz")
            with self.assertRaisesRegex(RuntimeError, "no Git baseline"):
                await environment.export_patch()
            self.assertEqual(await environment.export_archive(), b"workspace-archive")
            archived = next(
                call for call in reversed(runner.calls) if "tar -czf" in call[-1]
            )
            self.assertIn("-C /workspace .", archived[-1])
            state = await environment.export_state()
            self.assertIsInstance(state, SWEArchiveState)
            await environment.adopt_state(state)
            replaced = next(
                call for call in reversed(runner.calls) if "tar -xzf" in call[-1]
            )
            self.assertIn("find /workspace -mindepth 1 -delete", replaced[-1])
            with self.assertRaisesRegex(ProtocolError, "different baseline"):
                await environment.adopt_state(SWEArchiveState("elsewhere", b""))
            environment.max_archive_bytes = 4
            with self.assertRaisesRegex(RuntimeError, "byte limit"):
                await environment.export_archive()
        finally:
            await environment.close()
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_docker_image_preflight_materializes_and_binds_an_exact_id(
        self,
    ) -> None:
        class MissingImageRunner(RecordingRunner):
            def __init__(self) -> None:
                super().__init__()
                self.inspections = 0

            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:3] == ("image", "inspect"):
                    self.inspections += 1
                    if self.inspections == 1:
                        self.calls.append(call)
                        return ProcessResult(b"not found", 1, 9)
                return await super().run(argv, **kwargs)

        runner = MissingImageRunner()
        binding = await resolve_swebench_image_binding(
            {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:latest",
            },
            runtime="docker",
            runner=runner,
        )
        self.assertEqual(binding.identity, TEST_DOCKER_ID)
        self.assertEqual(binding.execution_ref, TEST_DOCKER_ID)
        self.assertEqual(
            binding.manifest_identity(),
            {
                "runtime": "docker",
                "requested": "example/image:latest",
                "identity": TEST_DOCKER_ID,
            },
        )
        self.assertEqual(runner.inspections, 2)
        self.assertTrue(any(call[1:2] == ("pull",) for call in runner.calls))

    async def test_docker_doctor_and_environment_use_rootless_unmounted_contract(
        self,
    ) -> None:
        runner = RecordingRunner()
        report = await swebench_doctor(
            runner=runner,
            image="example/image:tag",
        )
        self.assertTrue(report.ok)
        self.assertEqual(
            [check.name for check in report.checks],
            [
                "runtime_version",
                "daemon_platform",
                "rootless_security",
                "image_available",
            ],
        )

        environment = await docker_swe_environment(
            {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:tag",
            },
            runner=runner,
        )
        try:
            start = next(call for call in runner.calls if call[1:2] == ("run",))
            self.assertIn("--workdir", start)
            self.assertNotIn("--volume", start)
            self.assertNotIn("--mount", start)
            self.assertEqual(environment.runtime.image_id, TEST_DOCKER_ID)
            self.assertEqual(start[-3], TEST_DOCKER_ID)
            self.assertNotIn("example/image:tag", start)
            self.assertFalse(environment.provenance()["host_credentials_mounted"])
            self.assertEqual(
                environment.provenance()["benchmark_revision"],
                "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            )
            self.assertEqual(environment.provenance()["benchmark_tag"], "v4.1.0")
            await environment.execute(ToolCall("call", "bash", {"command": "pwd"}))
            execution = runner.calls[-1]
            self.assertIn("/bin/bash", execution)
            self.assertIn("BASH_ENV=/root/.bashrc", execution)
            self.assertEqual(execution[-1], "pwd")
            await environment.export_patch()
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "target.diff"
                target.write_text("keep")
                link = Path(temporary) / "link.diff"
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    await environment.export_patch(link)
                self.assertEqual(target.read_text(), "keep")
            stage_call = next(
                call
                for call in reversed(runner.calls)
                if call[-1].startswith("git add")
            )
            self.assertNotIn("--force", stage_call[-1])
            await environment.adopt_state(await environment.export_state())
            reset_call = next(
                call
                for call in reversed(runner.calls)
                if call[-1].startswith("git reset")
            )
            self.assertIn("git clean -ffd -q", reset_call[-1])
            self.assertNotIn("-ffdx", reset_call[-1])
            self.assertIn(environment.base_commit, reset_call[-1])
        finally:
            await environment.close()
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

        unprivileged = await swebench_doctor(runner=RecordingRunner(rootless=False))
        self.assertFalse(unprivileged.ok)
        with self.assertRaisesRegex(RuntimeError, "rootless daemon"):
            await docker_swe_environment(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=RecordingRunner(rootless=False),
            )

        class DeceptiveSecurityRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if any("SecurityOptions" in item for item in argv):
                    value = b'["name=notrootless"]\n'
                    self.calls.append(tuple(argv))
                    return ProcessResult(value, 0, len(value))
                return await super().run(argv, **kwargs)

        deceptive = await swebench_doctor(runner=DeceptiveSecurityRunner())
        self.assertFalse(deceptive.ok)

    async def test_docker_container_must_match_its_preflight_binding(self) -> None:
        other_id = "sha256:" + "b" * 64

        class DriftRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("inspect",):
                    self.calls.append(call)
                    output = (other_id + "\n").encode()
                    return ProcessResult(output, 0, len(output))
                return await super().run(argv, **kwargs)

        runner = DriftRunner()
        binding = SWEbenchImageBinding(
            runtime="docker",
            requested="example/image:tag",
            identity=TEST_DOCKER_ID,
            execution_ref=TEST_DOCKER_ID,
        )
        with self.assertRaisesRegex(RuntimeError, "does not match its binding"):
            await docker_swe_environment(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                image_binding=binding,
                runner=runner,
            )
        start = next(call for call in runner.calls if call[1:2] == ("run",))
        self.assertEqual(start[-3], TEST_DOCKER_ID)
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_task_base_commit_must_match_the_selected_image(self) -> None:
        class WrongCommitRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if "merge-base --is-ancestor" in tuple(argv)[-1]:
                    self.calls.append(tuple(argv))
                    return ProcessResult(b"", 1, 0)
                return await super().run(argv, **kwargs)

        runner = WrongCommitRunner()
        with self.assertRaisesRegex(RuntimeError, "task base_commit"):
            await docker_swe_environment(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                    "base_commit": "d" * 40,
                },
                runner=runner,
            )
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_docker_startup_surfaces_cleanup_failure(self) -> None:
        class FailingRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("inspect",):
                    self.calls.append(call)
                    return ProcessResult(b"inspect failed", 1, 14)
                if call[1:3] == ("rm", "--force"):
                    self.calls.append(call)
                    return ProcessResult(b"remove failed", 1, 13)
                return await super().run(argv, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "cleanup also failed"):
            await docker_swe_environment(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=FailingRunner(),
            )

    async def test_docker_run_failure_still_removes_named_container(self) -> None:
        class StartFailureRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("run",):
                    self.calls.append(call)
                    return ProcessResult(b"start failed", 1, 12)
                return await super().run(argv, **kwargs)

        runner = StartFailureRunner()
        with self.assertRaisesRegex(RuntimeError, "could not start"):
            await docker_swe_environment(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=runner,
            )
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_apptainer_overlay_exception_removes_owned_scratch(self) -> None:
        class OverlayFailureRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if tuple(argv)[1:3] == ("overlay", "create"):
                    raise RuntimeError("overlay failed")
                return await super().run(argv, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            with self.assertRaisesRegex(RuntimeError, "overlay failed"):
                await apptainer_swe_environment(
                    {
                        "instance_id": "repo__issue-1",
                        "image_name": "example/image:tag",
                    },
                    scratch_root=scratch,
                    overlay_size_mib=1024,
                    runner=OverlayFailureRunner(),
                )
            self.assertEqual(list(scratch.iterdir()), [])

    async def test_apptainer_materializes_once_and_uses_private_fakeroot_overlay(
        self,
    ) -> None:
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:tag",
            }
            first = await apptainer_swe_environment(
                arguments,
                image="docker://example/image:tag",
                scratch_root=root / "scratch",
                image_cache=root / "cache",
                overlay_size_mib=1024,
                runner=runner,
            )
            second = await apptainer_swe_environment(
                arguments,
                image="docker://example/image:tag",
                scratch_root=root / "scratch",
                image_cache=root / "cache",
                overlay_size_mib=1024,
                runner=runner,
            )
            try:
                self.assertEqual(runner.pull_calls, 1)
                self.assertNotEqual(first.runtime.overlay, second.runtime.overlay)
                self.assertTrue(first.runtime.image.startswith(str(root / "cache")))
                self.assertTrue(first.runtime.image_identity.startswith("sha256:"))
                exec_call = next(
                    call
                    for call in runner.calls
                    if "exec" in call[1:3] and "--overlay" in call
                )
                self.assertEqual(exec_call[1:3], ("--silent", "exec"))
                self.assertIn("--cleanenv", exec_call)
                self.assertIn("--containall", exec_call)
                self.assertIn("--fakeroot", exec_call)
                self.assertNotIn("HOME=/root", exec_call)
                self.assertTrue(
                    any(
                        "BASH_ENV=/root/.bashrc" in item for item in exec_call
                    )
                )
                await first.export_patch()
                stage_call = next(
                    call
                    for call in reversed(runner.calls)
                    if call[-1].startswith("git add")
                )
                self.assertNotIn("--force", stage_call[-1])
                await first.adopt_state(await first.export_state())
                reset_call = next(
                    call
                    for call in reversed(runner.calls)
                    if call[-1].startswith("git reset")
                )
                self.assertIn("git clean -ffd -q", reset_call[-1])
                self.assertNotIn("-ffdx", reset_call[-1])
                self.assertIn(first.base_commit, reset_call[-1])
            finally:
                await first.close()
                await second.close()

    async def test_apptainer_preflight_binding_rejects_changed_bytes(self) -> None:
        runner = RecordingRunner()
        instance = {
            "instance_id": "repo__issue-1",
            "image_name": "example/image:tag",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = await resolve_swebench_image_binding(
                instance,
                runtime="apptainer",
                apptainer_image_cache=root / "cache",
                runner=runner,
            )
            Path(binding.execution_ref).write_bytes(b"different-sif-content")
            with self.assertRaisesRegex(RuntimeError, "preflight binding"):
                await apptainer_swe_environment(
                    instance,
                    image_binding=binding,
                    scratch_root=root / "scratch",
                    image_cache=root / "cache",
                    overlay_size_mib=1024,
                    runner=runner,
                )
            self.assertEqual(list((root / "scratch").iterdir()), [])

    async def test_apptainer_cache_lock_wait_is_bounded(self) -> None:
        runner = RecordingRunner()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("fcntl.flock", side_effect=BlockingIOError),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache lock"):
                await materialize_apptainer_image(
                    "docker://example/image:tag",
                    executable="apptainer",
                    runner=runner,
                    cache=Path(temporary),
                    timeout_seconds=0.001,
                    max_output_bytes=1024,
                )
        self.assertEqual(runner.pull_calls, 0)
