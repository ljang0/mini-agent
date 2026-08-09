import asyncio
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, patch

from scaffoldlab.grok_source import (
    GROK_BUILD_CARGO_LOCK_SHA256,
    GROK_BUILD_PUBLIC_REVISION,
    GROK_BUILD_SOURCE_REV,
    GrokBuildSourceBackend,
    _extract_verified_git_archive,
)
from scaffoldlab.environments.swe import SWEPatchPayload
from scaffoldlab.evaluation import MatrixRunner
from scaffoldlab.harnesses.grok_source import GrokBuildSourceHarness
from scaffoldlab.providers import ProviderError
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelRequest, ModelResponse, Task, Usage


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        *,
        returncode: int = 0,
        on_communicate: Callable[[], None] | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = 424242
        self.on_communicate = on_communicate

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.on_communicate is not None:
            self.on_communicate()
        return self.stdout, self.stderr

    async def wait(self) -> int:
        return self.returncode


def _real_git_archive_process(
    args: tuple[str, ...], kwargs: dict[str, Any]
) -> _FakeProcess | None:
    """Let mocked process tests use the real local Git only for source export."""

    if "archive" not in args:
        return None
    return _real_process(args, kwargs)


def _real_process(args: tuple[str, ...], kwargs: dict[str, Any]) -> _FakeProcess:
    """Execute a short local command while retaining the async process-test seam."""

    completed = subprocess.run(
        list(args),
        cwd=kwargs["cwd"],
        env=kwargs["env"],
        check=False,
        capture_output=True,
    )
    return _FakeProcess(
        completed.stdout,
        completed.stderr,
        returncode=completed.returncode,
    )


def _success_result() -> dict[str, Any]:
    return {
        "text": "source-native answer",
        "stopReason": "end_turn",
        "sessionId": "session-1",
        "num_turns": 4,
        "usage": {
            "input_tokens": 7,
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 13,
            "output_tokens": 17,
        },
        "modelUsage": {"grok-build": {"modelCalls": 4}},
        "total_cost_usd_ticks": 250_000_000,
    }


class GrokSourceBackendTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, str, str, str]:
        checkout = root / "grok-build"
        manifest = checkout / "crates" / "codegen" / "xai-grok-pager-bin" / "Cargo.toml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            '[package]\nname = "xai-grok-pager-bin"\nversion = "1.0.0"\n',
            encoding="utf-8",
        )
        cargo_lock = checkout / "Cargo.lock"
        cargo_lock.write_text("synthetic locked graph\n", encoding="utf-8")
        source_rev = "2" * 40
        (checkout / "SOURCE_REV").write_text(source_rev + "\n", encoding="ascii")
        (checkout / ".gitignore").write_text("/target\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "add", "."],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Scaffold Lab",
                "-c",
                "user.email=scaffold@example.invalid",
                "commit",
                "-q",
                "-m",
                "synthetic source",
            ],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        lock_sha256 = hashlib.sha256(cargo_lock.read_bytes()).hexdigest()
        return checkout, revision, source_rev, lock_sha256

    def _backend(
        self,
        root: Path,
        *,
        pass_env: tuple[str, ...] = (),
        allow_sensitive_environment: bool = False,
        expected_executable_sha256: str | None = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        ignored_fixture: bool = False,
    ) -> tuple[GrokBuildSourceBackend, Path, Path]:
        checkout, revision, source_rev, lock_sha256 = self._checkout(root)
        workspace = root / "base-workspace"
        workspace.mkdir()
        (workspace / "task.txt").write_text("immutable seed\n", encoding="utf-8")
        if ignored_fixture:
            (workspace / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
            (workspace / "existing.ignored").write_text(
                "ignored baseline\n", encoding="utf-8"
            )
        backend = GrokBuildSourceBackend(
            checkout=checkout,
            workspace=workspace,
            model="grok-build-pinned",
            cargo_executable="cargo-test",
            sandbox="strict",
            permission_mode="dontAsk",
            max_turns=9,
            pass_env=pass_env,
            allow_sensitive_environment=allow_sensitive_environment,
            expected_checkout_revision=revision,
            expected_source_rev=source_rev,
            expected_cargo_lock_sha256=lock_sha256,
            expected_executable_sha256=expected_executable_sha256,
            max_output_bytes=max_output_bytes,
        )
        return backend, checkout, workspace

    def test_public_identity_constants_are_full_pins(self) -> None:
        self.assertRegex(GROK_BUILD_PUBLIC_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(GROK_BUILD_SOURCE_REV, r"^[0-9a-f]{40}$")
        self.assertRegex(GROK_BUILD_CARGO_LOCK_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            GROK_BUILD_PUBLIC_REVISION,
            "8a14c91d88875a831a38b3a066b1683116bcb31c",
        )
        self.assertEqual(
            GROK_BUILD_SOURCE_REV,
            "27b3c66635e2c0bf213429a36ab916f25d59df20",
        )
        self.assertEqual(
            GROK_BUILD_CARGO_LOCK_SHA256,
            "285e13b019551e76680a21fe300dda5934aba9bcf1f7e7ad24b2ddee7fd3eb92",
        )

    def test_rejects_revision_lock_source_and_dirty_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            common = {
                "checkout": checkout,
                "workspace": workspace,
                "model": "grok-build",
                "expected_checkout_revision": revision,
                "expected_source_rev": source_rev,
                "expected_cargo_lock_sha256": lock_sha256,
            }
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                GrokBuildSourceBackend(
                    **{**common, "expected_checkout_revision": "0" * 40}
                )
            with self.assertRaisesRegex(ValueError, "SOURCE_REV mismatch"):
                GrokBuildSourceBackend(**{**common, "expected_source_rev": "0" * 40})
            with self.assertRaisesRegex(ValueError, "Cargo.lock SHA-256 mismatch"):
                GrokBuildSourceBackend(
                    **{**common, "expected_cargo_lock_sha256": "0" * 64}
                )
            (checkout / "tracked-dirty.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remain clean"):
                GrokBuildSourceBackend(**common)

    def test_rejects_unacknowledged_environment_and_linked_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            common = {
                "checkout": checkout,
                "workspace": workspace,
                "model": "grok-build",
                "expected_checkout_revision": revision,
                "expected_source_rev": source_rev,
                "expected_cargo_lock_sha256": lock_sha256,
            }
            with self.assertRaisesRegex(ValueError, "can inspect passed"):
                GrokBuildSourceBackend(**common, pass_env=("XAI_API_KEY",))
            for runtime_override in ("GROK_HOME", "LD_PRELOAD", "NODE_OPTIONS"):
                with self.assertRaisesRegex(ValueError, "runtime-control"):
                    GrokBuildSourceBackend(
                        **common,
                        pass_env=(runtime_override,),
                        allow_sensitive_environment=True,
                    )
            (workspace / ".git").write_text("gitdir: /outside\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "linked Git worktree"):
                GrokBuildSourceBackend(**common)

    def test_rejects_linked_source_checkout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            external_metadata = root / "shared-git-metadata"
            (checkout / ".git").rename(external_metadata)
            (checkout / ".git").symlink_to(external_metadata, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "standalone clone"):
                GrokBuildSourceBackend(
                    checkout=checkout,
                    workspace=workspace,
                    model="grok-build",
                    expected_checkout_revision=revision,
                    expected_source_rev=source_rev,
                    expected_cargo_lock_sha256=lock_sha256,
                )

    async def test_builds_source_and_invokes_exact_isolated_protocol(self) -> None:
        captured: dict[str, Any] = {}
        calls: list[tuple[str, ...]] = []
        reported: list[Usage] = []

        with tempfile.TemporaryDirectory() as directory:
            backend, checkout, workspace = self._backend(
                Path(directory),
                pass_env=("XAI_API_KEY",),
                allow_sensitive_environment=True,
            )

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                calls.append(args)
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                if "build" in args:
                    self.assertEqual(
                        args[1:],
                        (
                            "build",
                            "--locked",
                            "--release",
                            "-p",
                            "xai-grok-pager-bin",
                        ),
                    )
                    self.assertEqual(kwargs["cwd"], str(backend.build_source_root))
                    self.assertEqual(
                        kwargs["env"]["CARGO_TARGET_DIR"], str(backend.build_target)
                    )
                    self.assertEqual(kwargs["env"]["HOME"], str(backend.build_home))
                    self.assertEqual(
                        kwargs["env"]["XDG_CONFIG_HOME"],
                        str(backend.build_xdg_config),
                    )
                    self.assertEqual(
                        kwargs["env"]["CARGO_HOME"], str(backend.build_cargo_home)
                    )
                    self.assertEqual(kwargs["env"]["TMPDIR"], str(backend.build_tmp))
                    self.assertEqual(kwargs["env"]["GIT_CONFIG_NOSYSTEM"], "1")

                    def create_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"synthetic-rust-binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_binary)

                if args[0] != str(backend.executable):

                    def create_fresh_metadata() -> None:
                        if "init" in args:
                            trial = Path(kwargs["cwd"])
                            (trial / ".git").mkdir()
                            (trial / ".git" / "fresh-baseline").write_text(
                                "local only\n", encoding="utf-8"
                            )

                    return _FakeProcess(
                        stdout=(b"1" * 40 + b"\n") if "rev-parse" in args else b"",
                        on_communicate=create_fresh_metadata,
                    )

                trial_workspace = Path(args[args.index("--cwd") + 1])
                prompt_path = Path(args[args.index("--prompt-file") + 1])

                def inspect_session() -> None:
                    captured["args"] = args
                    captured["kwargs"] = kwargs
                    captured["trial_workspace"] = trial_workspace
                    captured["prompt_path"] = prompt_path
                    captured["prompt"] = prompt_path.read_text(encoding="utf-8")
                    captured["prompt_mode"] = stat.S_IMODE(prompt_path.stat().st_mode)
                    captured["seed"] = (trial_workspace / "task.txt").read_text(
                        encoding="utf-8"
                    )
                    captured["fresh_git"] = (
                        trial_workspace / ".git" / "fresh-baseline"
                    ).is_file()
                    captured["grok_home"] = Path(kwargs["env"]["GROK_HOME"])
                    (trial_workspace / "agent-output.txt").write_text(
                        "trial only\n", encoding="utf-8"
                    )

                return _FakeProcess(
                    json.dumps(_success_result()).encode("utf-8"),
                    b"source diagnostic",
                    on_communicate=inspect_session,
                )

            terminate = AsyncMock()
            with (
                patch.dict(os.environ, {"XAI_API_KEY": "scoped-secret"}),
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=terminate,
                ),
            ):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/grok-source",
                        role="session",
                        system="system rule",
                        prompt="secret task",
                        usage_reporter=reported.append,
                    )
                )

            self.assertEqual(response.text, "source-native answer")
            self.assertEqual(response.usage.input_tokens, 31)
            self.assertEqual(response.usage.output_tokens, 17)
            self.assertEqual(response.usage.cost_usd, 0.025)
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            self.assertEqual(reported, [response.usage])
            self.assertEqual(len(calls), 9)
            session_args = captured["args"]
            expected = (
                str(backend.executable),
                "--cwd",
                str(captured["trial_workspace"]),
                "--no-auto-update",
                "--output-format",
                "json",
                "--prompt-file",
                str(captured["prompt_path"]),
                "--sandbox",
                "strict",
                "--permission-mode",
                "dontAsk",
                "--max-turns",
                "9",
                "--model",
                "grok-build-pinned",
            )
            self.assertEqual(session_args, expected)
            self.assertNotIn("secret task", session_args)
            self.assertEqual(captured["prompt"], "system rule\n\nsecret task")
            self.assertEqual(captured["prompt_mode"], 0o600)
            self.assertEqual(captured["seed"], "immutable seed\n")
            self.assertTrue(captured["fresh_git"])
            self.assertEqual(
                captured["kwargs"]["cwd"], str(captured["trial_workspace"])
            )
            self.assertEqual(captured["kwargs"]["stdin"], asyncio.subprocess.DEVNULL)
            self.assertTrue(captured["kwargs"]["start_new_session"])
            self.assertEqual(captured["kwargs"]["env"]["XAI_API_KEY"], "scoped-secret")
            self.assertEqual(captured["kwargs"]["env"]["GROK_DISABLE_AUTOUPDATER"], "1")
            self.assertFalse(captured["trial_workspace"].exists())
            self.assertFalse(captured["prompt_path"].exists())
            self.assertFalse(captured["grok_home"].exists())
            self.assertEqual(
                (workspace / "task.txt").read_text(encoding="utf-8"),
                "immutable seed\n",
            )
            self.assertFalse((workspace / "agent-output.txt").exists())
            self.assertEqual(terminate.await_count, 9)
            self.assertFalse(
                subprocess.run(
                    ["git", "-C", str(checkout), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            provenance = backend.provenance()
            self.assertEqual(
                provenance["workspace_isolation"], "fresh-disposable-copy-per-call"
            )
            self.assertTrue(provenance["native_subagents_enabled"])
            self.assertTrue(provenance["runtime_executable"]["available"])
            self.assertRegex(
                provenance["runtime_executable"]["sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertNotIn("scoped-secret", json.dumps(provenance))
            self.assertFalse(provenance["runtime_source_identity_verified"])
            self.assertFalse(provenance["checkout_cargo_target_used"])
            self.assertEqual(
                provenance["build_source_isolation"],
                "git-archive-of-pinned-commit",
            )
            self.assertRegex(provenance["build_source_tree_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                provenance["workspace_git_baseline"],
                "fresh standalone repository; inherited Git metadata and history are stripped",
            )

    async def test_executable_hash_pin_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = self._backend(
                Path(directory), expected_executable_sha256="0" * 64
            )

            def create_binary() -> None:
                backend.executable.parent.mkdir(parents=True)
                backend.executable.write_bytes(b"not-the-pinned-build")
                backend.executable.chmod(0o700)

            process = _FakeProcess(on_communicate=create_binary)

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                return archive_process or process

            with (
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaisesRegex(ProviderError, "executable SHA-256"):
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

    async def test_manifest_preflight_builds_only_the_pinned_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, checkout, _ = self._backend(Path(directory))
            ignored = checkout / "target" / "ignored-build-input"
            ignored.parent.mkdir()
            ignored.write_text("must not influence compilation\n", encoding="utf-8")
            build_cwds: list[Path] = []

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                self.assertIn("build", args)
                build_cwds.append(Path(kwargs["cwd"]))

                def create_binary() -> None:
                    self.assertFalse(
                        (backend.build_source_root / "target").exists(),
                        "ignored checkout content must not enter the source export",
                    )
                    backend.executable.parent.mkdir(parents=True)
                    backend.executable.write_bytes(b"private-built-binary")
                    backend.executable.chmod(0o700)

                return _FakeProcess(on_communicate=create_binary)

            try:
                with (
                    patch(
                        "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                        new=create_process,
                    ),
                    patch(
                        "scaffoldlab.grok_source._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                ):
                    await backend.prepare_for_manifest()
                self.assertEqual(build_cwds, [backend.build_source_root])
                self.assertTrue(backend.executable.is_file())
                provenance = backend.provenance()
                self.assertRegex(
                    provenance["build_source_tree_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertTrue(provenance["runtime_git"]["available"])
            finally:
                backend.close()

    async def test_rejects_external_ancestor_cargo_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = self._backend(Path(directory))
            external_config = backend.build_root / ".cargo" / "config.toml"
            external_config.parent.mkdir()
            external_config.write_text(
                '[build]\nrustc-wrapper = "/untrusted/wrapper"\n',
                encoding="utf-8",
            )
            try:
                with self.assertRaisesRegex(
                    ProviderError, "external Cargo configuration"
                ):
                    await backend.prepare_for_manifest()
                self.assertFalse(backend.executable.exists())
            finally:
                backend.close()

    async def test_build_fails_if_exported_source_is_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = self._backend(Path(directory))

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                self.assertIn("build", args)

                def mutate_export() -> None:
                    (backend.build_source_root / "SOURCE_REV").write_text(
                        "mutated during build\n", encoding="ascii"
                    )
                    backend.executable.parent.mkdir(parents=True)
                    backend.executable.write_bytes(b"private-built-binary")
                    backend.executable.chmod(0o700)

                return _FakeProcess(on_communicate=mutate_export)

            try:
                with (
                    patch(
                        "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                        new=create_process,
                    ),
                    patch(
                        "scaffoldlab.grok_source._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ProviderError, "mutated its pinned source export"
                    ):
                        await backend.prepare_for_manifest()
            finally:
                backend.close()

    def test_safe_archive_extraction_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar"
            payload = b"escape\n"
            with tarfile.open(archive, mode="w") as stream:
                member = tarfile.TarInfo("../escaped.txt")
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))
            destination = root / "source"
            with self.assertRaisesRegex(ProviderError, "unsafe path"):
                _extract_verified_git_archive(
                    archive,
                    destination,
                    max_entries=10,
                    max_bytes=1024,
                )
            self.assertFalse((root / "escaped.txt").exists())

    async def test_rejects_resolved_rustup_init_proxy_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            rustup = root / "rustup-init"
            rustup.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            rustup.chmod(0o700)
            cargo = root / "cargo"
            cargo.symlink_to(rustup)
            backend = GrokBuildSourceBackend(
                checkout=checkout,
                workspace=workspace,
                model="grok-build",
                cargo_executable=str(cargo),
                expected_checkout_revision=revision,
                expected_source_rev=source_rev,
                expected_cargo_lock_sha256=lock_sha256,
            )
            with self.assertRaisesRegex(ProviderError, "rustup cargo proxy"):
                await backend.complete(
                    ModelRequest(agent_id="/grok", role="session", prompt="p")
                )
            self.assertFalse(backend.executable.exists())

    async def test_exports_tracked_untracked_and_ignored_trial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, workspace = self._backend(root, ignored_fixture=True)

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                if "build" in args:

                    def create_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"private-built-binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_binary)
                if args[0] == str(backend.executable):
                    trial = Path(args[args.index("--cwd") + 1])

                    def edit_trial() -> None:
                        (trial / "task.txt").write_text(
                            "tracked trial edit\n", encoding="utf-8"
                        )
                        (trial / "new-binary.bin").write_bytes(
                            b"\x00\xff\x01\xfeuntracked"
                        )
                        (trial / "existing.ignored").write_text(
                            "edited ignored baseline\n", encoding="utf-8"
                        )
                        (trial / "new.ignored").write_text(
                            "new ignored output\n", encoding="utf-8"
                        )

                    return _FakeProcess(
                        json.dumps(_success_result()).encode("utf-8"),
                        on_communicate=edit_trial,
                    )
                return _real_process(args, kwargs)

            try:
                with (
                    patch(
                        "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                        new=create_process,
                    ),
                    patch(
                        "scaffoldlab.grok_source._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                ):
                    response = await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

                patch_payload = response.raw["_scaffoldlab_swe_patch"]
                self.assertIsInstance(patch_payload, SWEPatchPayload)
                self.assertTrue(
                    response.raw["_scaffoldlab_source"]["source_archive_verified"]
                )
                self.assertFalse(
                    response.raw["_scaffoldlab_source"]["official_public_pin_verified"]
                )
                self.assertFalse(
                    response.raw["_scaffoldlab_source"]["executable_hash_pin_verified"]
                )
                patch_bytes = patch_payload.content
                patch_text = patch_bytes.decode("utf-8")
                self.assertIn("diff --git a/task.txt b/task.txt", patch_text)
                self.assertIn("+tracked trial edit", patch_text)
                self.assertIn(
                    "diff --git a/new-binary.bin b/new-binary.bin", patch_text
                )
                self.assertIn("new file mode", patch_text)
                self.assertIn("GIT binary patch", patch_text)
                self.assertIn(
                    "diff --git a/existing.ignored b/existing.ignored", patch_text
                )
                self.assertIn("-ignored baseline", patch_text)
                self.assertIn("+edited ignored baseline", patch_text)
                self.assertIn("diff --git a/new.ignored b/new.ignored", patch_text)
                self.assertIn("+new ignored output", patch_text)
                self.assertEqual(
                    (workspace / "task.txt").read_text(encoding="utf-8"),
                    "immutable seed\n",
                )
                self.assertFalse((workspace / "new-binary.bin").exists())
                self.assertEqual(
                    (workspace / "existing.ignored").read_text(encoding="utf-8"),
                    "ignored baseline\n",
                )
                self.assertFalse((workspace / "new.ignored").exists())

                output = root / "matrix-output"
                records, _ = await MatrixRunner(
                    backend=ScriptedBackend({"/grok-build-source/root": [response]}),
                    limits=BudgetLimits(max_model_calls=1, wall_time_seconds=10),
                    output_dir=output,
                ).run(
                    [Task("grok-patch", "edit the task")],
                    [GrokBuildSourceHarness()],
                )
                artifact = records[0].metadata["patch_artifact"]
                artifact_path = Path(artifact["path"])
                self.assertEqual(artifact_path.read_bytes(), patch_bytes)
                self.assertEqual(artifact["bytes"], len(patch_bytes))
                self.assertEqual(artifact["format"], "git_diff_binary")
            finally:
                backend.close()

    async def test_rejects_patch_that_exceeds_the_output_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = self._backend(Path(directory), max_output_bytes=1024)
            reported: list[Usage] = []

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                if "build" in args:

                    def create_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"private-built-binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_binary)
                if args[0] == str(backend.executable):
                    return _FakeProcess(json.dumps(_success_result()).encode("utf-8"))
                if "rev-parse" in args:
                    return _FakeProcess(b"1" * 40 + b"\n")
                if "diff" in args:
                    return _FakeProcess(b"x" * 1025)
                return _FakeProcess()

            try:
                with (
                    patch(
                        "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                        new=create_process,
                    ),
                    patch(
                        "scaffoldlab.grok_source._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ProviderError, "SWE patch exceeded its output limit"
                    ) as raised:
                        await backend.complete(
                            ModelRequest(
                                agent_id="/grok",
                                role="session",
                                prompt="p",
                                usage_reporter=reported.append,
                            )
                        )
                self.assertEqual(reported, [raised.exception.usage])
                self.assertEqual(raised.exception.usage.input_tokens, 31)
                self.assertFalse(raised.exception.usage.complete)

                postprocessing_usage: list[Usage] = []
                with (
                    patch(
                        "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                        new=create_process,
                    ),
                    patch(
                        "scaffoldlab.grok_source._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                    patch.object(
                        backend,
                        "_capture_swe_patch",
                        new=AsyncMock(
                            side_effect=ProviderError(
                                "synthetic Grok postprocessing failure",
                                raw={"stage": "postprocessing"},
                            )
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ProviderError, "synthetic Grok postprocessing failure"
                    ) as postprocessing_raised:
                        await backend.complete(
                            ModelRequest(
                                agent_id="/grok",
                                role="session",
                                prompt="p",
                                usage_reporter=postprocessing_usage.append,
                            )
                        )
                self.assertEqual(
                    postprocessing_usage, [postprocessing_raised.exception.usage]
                )
                self.assertEqual(postprocessing_raised.exception.usage.input_tokens, 31)
                self.assertEqual(
                    postprocessing_raised.exception.raw,
                    {"stage": "postprocessing"},
                )
            finally:
                backend.close()

    async def test_source_mutation_during_session_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, checkout, _ = self._backend(Path(directory))
            calls = 0

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                nonlocal calls
                calls += 1
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                if "build" in args:

                    def create_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_binary)

                if args[0] != str(backend.executable):
                    return _FakeProcess(
                        stdout=(b"1" * 40 + b"\n") if "rev-parse" in args else b""
                    )

                return _FakeProcess(
                    json.dumps(_success_result()).encode("utf-8"),
                    on_communicate=lambda: (checkout / "SOURCE_REV").write_text(
                        "mutated\n", encoding="ascii"
                    ),
                )

            with (
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaisesRegex(ProviderError, "remain clean"):
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

    async def test_rejects_absolute_seed_symlink_before_agent_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            seed = workspace / "seed.txt"
            seed.write_text("immutable\n", encoding="utf-8")
            (workspace / "absolute-link.txt").symlink_to(seed.resolve())
            backend = GrokBuildSourceBackend(
                checkout=checkout,
                workspace=workspace,
                model="grok-build",
                cargo_executable="cargo-test",
                expected_checkout_revision=revision,
                expected_source_rev=source_rev,
                expected_cargo_lock_sha256=lock_sha256,
            )
            process_calls = 0

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                nonlocal process_calls
                process_calls += 1
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process

                def create_binary() -> None:
                    backend.executable.parent.mkdir(parents=True)
                    backend.executable.write_bytes(b"synthetic-rust-binary")
                    backend.executable.chmod(0o700)

                return _FakeProcess(on_communicate=create_binary)

            with (
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaisesRegex(ProviderError, "symlink escapes"):
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

            self.assertEqual(process_calls, 2, "agent binary must never execute")
            self.assertEqual(seed.read_text(encoding="utf-8"), "immutable\n")

    async def test_uses_private_target_not_preseeded_checkout_target_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, source_rev, lock_sha256 = self._checkout(root)
            external_target = root / "external-target"
            seeded_binary = external_target / "release" / "xai-grok-pager"
            seeded_binary.parent.mkdir(parents=True)
            seeded_binary.write_bytes(b"attacker-controlled-ignored-binary")
            seeded_binary.chmod(0o700)
            (checkout / "target").symlink_to(external_target, target_is_directory=True)
            self.assertEqual(
                subprocess.run(
                    ["git", "-C", str(checkout), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "task.txt").write_text("seed\n", encoding="utf-8")
            backend = GrokBuildSourceBackend(
                checkout=checkout,
                workspace=workspace,
                model="grok-build",
                cargo_executable="cargo-test",
                expected_checkout_revision=revision,
                expected_source_rev=source_rev,
                expected_cargo_lock_sha256=lock_sha256,
            )
            executed: list[str] = []

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                if "build" in args:
                    self.assertNotEqual(
                        kwargs["env"]["CARGO_TARGET_DIR"], str(checkout / "target")
                    )
                    self.assertEqual(kwargs["cwd"], str(backend.build_source_root))

                    def create_private_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"private-built-binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_private_binary)
                if args[0] != str(backend.executable):
                    return _FakeProcess(
                        stdout=(b"1" * 40 + b"\n") if "rev-parse" in args else b""
                    )
                executed.append(args[0])
                return _FakeProcess(json.dumps(_success_result()).encode("utf-8"))

            with (
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                response = await backend.complete(
                    ModelRequest(agent_id="/grok", role="session", prompt="p")
                )

            self.assertEqual(response.text, "source-native answer")
            self.assertEqual(executed, [str(backend.executable)])
            self.assertEqual(
                seeded_binary.read_bytes(), b"attacker-controlled-ignored-binary"
            )

    async def test_strips_inherited_git_metadata_before_agent_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, workspace = self._backend(root)
            external = root / "external-git-state"
            external.write_text("immutable\n", encoding="utf-8")
            metadata = workspace / ".git"
            metadata.mkdir()
            (metadata / "escape").symlink_to(external)

            # Recreate after adding metadata: worktree hashing intentionally excludes
            # .git, while the copy layer must strip it rather than trust it.
            backend.close()
            checkout = root / "grok-build"
            revision = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            backend = GrokBuildSourceBackend(
                checkout=checkout,
                workspace=workspace,
                model="grok-build",
                cargo_executable="cargo-test",
                expected_checkout_revision=revision,
                expected_source_rev="2" * 40,
                expected_cargo_lock_sha256=hashlib.sha256(
                    (checkout / "Cargo.lock").read_bytes()
                ).hexdigest(),
            )
            inherited_seen: list[bool] = []

            async def create_process(*args: str, **kwargs: Any) -> _FakeProcess:
                archive_process = _real_git_archive_process(args, kwargs)
                if archive_process is not None:
                    return archive_process
                if "build" in args:

                    def create_binary() -> None:
                        backend.executable.parent.mkdir(parents=True)
                        backend.executable.write_bytes(b"private-built-binary")
                        backend.executable.chmod(0o700)

                    return _FakeProcess(on_communicate=create_binary)
                if args[0] != str(backend.executable):
                    if "init" in args:
                        return _FakeProcess(
                            on_communicate=lambda: (
                                Path(kwargs["cwd"]) / ".git"
                            ).mkdir()
                        )
                    return _FakeProcess(
                        stdout=(b"1" * 40 + b"\n") if "rev-parse" in args else b""
                    )
                trial = Path(args[args.index("--cwd") + 1])
                inherited_seen.append((trial / ".git" / "escape").exists())
                return _FakeProcess(json.dumps(_success_result()).encode("utf-8"))

            with (
                patch(
                    "scaffoldlab.grok_source.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.grok_source._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                await backend.complete(
                    ModelRequest(agent_id="/grok", role="session", prompt="p")
                )

            self.assertEqual(inherited_seen, [False])
            self.assertEqual(external.read_text(encoding="utf-8"), "immutable\n")


class GrokSourceHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_uses_shared_ledger_and_marks_scope(self) -> None:
        raw = {
            **_success_result(),
            "_scaffoldlab_workspace": {"isolation": "fresh-disposable-copy"},
            "_scaffoldlab_source": {
                "revision": GROK_BUILD_PUBLIC_REVISION,
                "source_archive_verified": True,
                "official_public_pin_verified": True,
                "executable_sha256": "a" * 64,
                "executable_hash_pin_verified": False,
            },
            "_scaffoldlab_swe_patch": SWEPatchPayload(b"diff --git a/a.txt b/a.txt\n"),
        }
        response = ModelResponse(
            text="answer",
            usage=Usage(
                input_tokens=1,
                output_tokens=1,
                cost_known=False,
                complete=False,
            ),
            raw=raw,
        )
        result = await GrokBuildSourceHarness().run(
            Task(task_id="task", prompt="question", context="context"),
            ScriptedBackend({"/grok-build-source/root": [response]}),
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(result.harness, "grok_build_source")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.metadata["underlying_model_calls"], 4)
        self.assertTrue(result.metadata["native_subagents_enabled"])
        self.assertTrue(result.metadata["upstream_source_executed"])
        self.assertFalse(result.metadata["full_tree_usage_verified"])
        self.assertEqual(
            result.metadata["fidelity"],
            "pinned_public_source_with_recorded_local_build",
        )
        self.assertTrue(result.metadata["source_or_protocol_pin_verified"])
        self.assertTrue(result.metadata["swe_patch_exported"])
        self.assertFalse(result.metadata["swe_patch_applied_or_scored"])
        self.assertFalse(result.metadata["executable_hash_pin_verified"])
        self.assertFalse(result.metadata["bit_reproducible_runtime_verified"])
        self.assertFalse(result.metadata["hosted_multi_agent_parity_claimed"])
        self.assertFalse(result.metadata["flagship_system_card_parity_claimed"])
        self.assertIn("headless JSON protocol only", result.metadata["exactness_scope"])

    async def test_harness_does_not_infer_source_exactness_without_evidence(
        self,
    ) -> None:
        response = ModelResponse(
            text="answer",
            usage=Usage(cost_known=False, complete=False),
            raw=_success_result(),
        )
        result = await GrokBuildSourceHarness().run(
            Task(task_id="task", prompt="question"),
            ScriptedBackend({"/grok-build-source/root": [response]}),
            BudgetLimits(max_model_calls=1),
        )

        self.assertFalse(result.metadata["released_runtime_adapter"])
        self.assertFalse(result.metadata["upstream_source_executed"])
        self.assertFalse(result.metadata["source_or_protocol_pin_verified"])
        self.assertEqual(result.metadata["fidelity"], "custom_source_runtime_boundary")
        self.assertFalse(result.metadata["swe_patch_exported"])


if __name__ == "__main__":
    unittest.main()
