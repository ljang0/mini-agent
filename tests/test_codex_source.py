from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.codex_source import (
    CODEX_CARGO_LOCK_SHA256,
    CODEX_RUST_TOOLCHAIN,
    CODEX_RUST_TOOLCHAIN_SHA256,
    CODEX_SOURCE_REPOSITORY,
    CODEX_SOURCE_REVISION,
    CODEX_SOURCE_TAG,
    CODEX_SOURCE_VERSION,
    CodexSourceBackend,
)
from scaffoldlab.environments.swe import SWEPatchPayload
from scaffoldlab.harnesses.codex_source import CodexSourceHarness
from scaffoldlab.providers import ProviderError
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import (
    BudgetLimits,
    ModelRequest,
    ModelResponse,
    Task,
    ToolDefinition,
    Usage,
)


FAKE_CODEX = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if "--version" in sys.argv:
    print("codex-cli 0.147.0")
    raise SystemExit(0)

args = sys.argv[1:]
if not args or args[0] != "exec":
    raise SystemExit(2)
prompt = sys.stdin.read()
if prompt == "overflow":
    print("x" * 4096)
    raise SystemExit(0)

def option(name):
    return args[args.index(name) + 1]

configs = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "-c"]
answer = json.dumps({
    "cwd": os.getcwd(),
    "prompt": prompt,
    "credential_present": bool(os.environ.get("CODEX_API_KEY")),
    "codex_home": os.environ.get("CODEX_HOME"),
    "nested_git_metadata_present": os.path.lexists("nested/.git"),
    "configs": configs,
    "ephemeral": "--ephemeral" in args,
    "ignore_user_config": "--ignore-user-config" in args,
    "sandbox": option("--sandbox"),
}, sort_keys=True)
Path(option("--output-last-message")).write_text(answer, encoding="utf-8")
Path("agent-output.txt").write_text("disposable\n", encoding="utf-8")
if prompt == "edit files":
    Path("task.txt").write_text("edited by Codex\n", encoding="utf-8")
    Path("new-binary.bin").write_bytes(b"\x00\xff\xfe\xfd")
elif prompt == "edit ignored files":
    Path("existing.ignored").write_text(
        "edited ignored baseline\n", encoding="utf-8"
    )
    Path("new.ignored").write_text("new ignored output\n", encoding="utf-8")
elif prompt == "patch overflow":
    Path("large.txt").write_text("z" * 4096, encoding="utf-8")

events = [
    {"type": "thread.started", "thread_id": "thread-root"},
    {"type": "turn.started"},
]
if prompt == "failed spawn":
    events.append({
        "type": "item.completed",
        "item": {
            "id": "collab-failed",
            "type": "collab_tool_call",
            "tool": "spawn_agent",
            "sender_thread_id": "thread-root",
            "receiver_thread_ids": [],
            "prompt": "work",
            "agents_states": {},
            "status": "failed",
        },
    })
elif prompt != "no spawn":
    for index, tool in enumerate(
        ["spawn_agent", "send_input", "wait", "close_agent"], start=1
    ):
        events.append({
            "type": "item.completed",
            "item": {
                "id": f"collab-{index}",
                "type": "collab_tool_call",
                "tool": tool,
                "sender_thread_id": "thread-root",
                "receiver_thread_ids": ["thread-child"],
                "prompt": None,
                "agents_states": {},
                "status": "completed",
            },
        })
events.extend([
    {
        "type": "item.completed",
        "item": {"id": "message-1", "type": "agent_message", "text": answer},
    },
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 20,
            "cached_input_tokens": 3,
            "cache_write_input_tokens": 2,
            "output_tokens": 7,
            "reasoning_output_tokens": 5,
        },
    },
])
for event in events:
    print(json.dumps(event, sort_keys=True))
"""


FAKE_CARGO = r"""#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("cargo 1.95.0 (offline-test 2000-01-01)")
    raise SystemExit(0)

Path(__file__).with_suffix(".log").write_text(
    json.dumps(sys.argv[1:]), encoding="utf-8"
)
target = Path(os.environ["CARGO_TARGET_DIR"]) / "release"
target.mkdir(parents=True)
executable = target / "codex"
shutil.copy2(Path.cwd() / "cli" / "fake_codex.py", executable)
executable.chmod(0o755)
"""


FAKE_RUSTC = r"""#!/usr/bin/env python3
import sys

if sys.argv[1:] == ["--version"]:
    print("rustc 1.95.0 (offline-test 2000-01-01)")
    raise SystemExit(0)
raise SystemExit(2)
"""


def _git(*args: str, cwd: Path) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
    )
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


@unittest.skipUnless(os.name == "posix", "adapter requires POSIX process groups")
class CodexSourceBackendTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, Path, str, str, str, str, str]:
        checkout = root / "codex"
        codex_rs = checkout / "codex-rs"
        (codex_rs / "cli").mkdir(parents=True)
        cargo_lock = codex_rs / "Cargo.lock"
        cargo_lock.write_text("synthetic locked graph\n", encoding="utf-8")
        toolchain = codex_rs / "rust-toolchain.toml"
        toolchain.write_text(
            f'[toolchain]\nchannel = "{CODEX_RUST_TOOLCHAIN}"\n',
            encoding="utf-8",
        )
        (codex_rs / "Cargo.toml").write_text(
            '[workspace]\n[workspace.package]\nversion = "0.147.0"\n',
            encoding="utf-8",
        )
        (codex_rs / "cli" / "Cargo.toml").write_text(
            '[package]\nname = "codex-cli"\nversion.workspace = true\n'
            '[[bin]]\nname = "codex"\npath = "src/main.rs"\n',
            encoding="utf-8",
        )
        source_executable = codex_rs / "cli" / "fake_codex.py"
        source_executable.write_text(FAKE_CODEX, encoding="utf-8")
        cargo = root / "fake-cargo"
        cargo.write_text(FAKE_CARGO, encoding="utf-8")
        cargo.chmod(0o755)
        rustc = root / "rustc"
        rustc.write_text(FAKE_RUSTC, encoding="utf-8")
        rustc.chmod(0o755)

        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", ".", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned Codex", cwd=checkout)
        revision = _git("rev-parse", "HEAD", cwd=checkout)
        return (
            checkout,
            cargo,
            revision,
            hashlib.sha256(cargo_lock.read_bytes()).hexdigest(),
            hashlib.sha256(toolchain.read_bytes()).hexdigest(),
            hashlib.sha256(cargo.read_bytes()).hexdigest(),
            hashlib.sha256(source_executable.read_bytes()).hexdigest(),
        )

    def _backend(
        self,
        root: Path,
        *,
        multi_agent_version: str = "v1",
        max_subagents: int | None = 3,
        max_wait_seconds: float | None = 45,
        max_output_bytes: int = 16 * 1024 * 1024,
        max_patch_bytes: int = 8 * 1024 * 1024,
        ignored_fixture: bool = False,
    ) -> tuple[CodexSourceBackend, Path, Path, Path]:
        (
            checkout,
            cargo,
            revision,
            lock_sha,
            toolchain_sha,
            cargo_sha,
            executable_sha,
        ) = self._checkout(root)
        workspace = root / "seed-workspace"
        workspace.mkdir()
        (workspace / "task.txt").write_text("immutable seed\n", encoding="utf-8")
        if ignored_fixture:
            (workspace / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
            (workspace / "existing.ignored").write_text(
                "ignored baseline\n", encoding="utf-8"
            )
        backend = CodexSourceBackend(
            checkout=checkout,
            workspace=workspace,
            model="gpt-source-test",
            reasoning_effort="ultra",
            api_key_env="TEST_CODEX_API_KEY",
            cargo_executable=str(cargo),
            rustc_executable=str(cargo.with_name("rustc")),
            multi_agent_version=multi_agent_version,
            max_subagents=max_subagents,
            max_depth=2,
            max_wait_seconds=max_wait_seconds,
            max_output_bytes=max_output_bytes,
            max_patch_bytes=max_patch_bytes,
            allow_sensitive_environment=True,
            expected_revision=revision,
            expected_cargo_lock_sha256=lock_sha,
            expected_rust_toolchain_sha256=toolchain_sha,
            expected_cargo_sha256=cargo_sha,
            expected_rustc_sha256=hashlib.sha256(
                cargo.with_name("rustc").read_bytes()
            ).hexdigest(),
            expected_executable_sha256=executable_sha,
        )
        self.addCleanup(backend.close)
        return backend, checkout, workspace, cargo

    def test_public_identity_constants_are_full_pins(self) -> None:
        self.assertEqual(CODEX_SOURCE_VERSION, "0.147.0")
        self.assertEqual(
            CODEX_SOURCE_REVISION,
            "be6e8eac029b183056b7e4402879f15d2c85f61b",
        )
        self.assertRegex(CODEX_CARGO_LOCK_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(CODEX_RUST_TOOLCHAIN_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(CODEX_RUST_TOOLCHAIN, "1.95.0")
        self.assertEqual(CODEX_SOURCE_REPOSITORY, "https://github.com/openai/codex")
        self.assertEqual(CODEX_SOURCE_TAG, "rust-v0.147.0")

    def test_native_v1_and_v2_defaults_remain_upstream_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1, _, _, _ = self._backend(
                root / "v1", max_subagents=None, max_wait_seconds=None
            )
            v2, _, _, _ = self._backend(
                root / "v2",
                multi_agent_version="v2",
                max_subagents=None,
                max_wait_seconds=None,
            )
            v1_provenance = v1.provenance()
            self.assertEqual(v1_provenance["max_subagents"], 6)
            self.assertEqual(v1_provenance["requested_multi_agent_version"], "v1")
            self.assertIsNone(v1_provenance["effective_multi_agent_version"])
            self.assertFalse(v1_provenance["effective_multi_agent_version_verified"])
            self.assertEqual(v1_provenance["configured_v1_max_depth"], 2)
            self.assertIsNone(v1_provenance["max_depth"])
            self.assertIsNone(v1_provenance["effective_v2_max_wait_timeout_ms"])
            v2_provenance = v2.provenance()
            self.assertEqual(v2_provenance["max_subagents"], 3)
            self.assertEqual(v2_provenance["requested_multi_agent_version"], "v2")
            self.assertEqual(v2_provenance["effective_multi_agent_version"], "v2")
            self.assertTrue(v2_provenance["effective_multi_agent_version_verified"])
            self.assertIsNone(v2_provenance["max_depth"])
            self.assertEqual(v2_provenance["v2_total_concurrency_including_root"], 4)
            self.assertEqual(v2_provenance["effective_v2_min_wait_timeout_ms"], 10_000)
            self.assertEqual(
                v2_provenance["effective_v2_max_wait_timeout_ms"], 3_600_000
            )
            self.assertEqual(
                v2_provenance["effective_v2_default_wait_timeout_ms"], 30_000
            )

    def test_rejects_unacknowledged_credentials_and_sensitive_tool_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                checkout,
                cargo,
                revision,
                lock_sha,
                toolchain_sha,
                _,
                _,
            ) = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            common = {
                "checkout": checkout,
                "workspace": workspace,
                "model": "gpt-source-test",
                "api_key_env": "TEST_CODEX_API_KEY",
                "cargo_executable": str(cargo),
                "expected_revision": revision,
                "expected_cargo_lock_sha256": lock_sha,
                "expected_rust_toolchain_sha256": toolchain_sha,
            }
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                CodexSourceBackend(**common)
            with self.assertRaisesRegex(ValueError, "must be CODEX_API_KEY"):
                CodexSourceBackend(
                    **common,
                    auth_target_env="OPENAI_API_KEY",
                    allow_sensitive_environment=True,
                )
            with self.assertRaisesRegex(ValueError, "credential-like"):
                CodexSourceBackend(
                    **common,
                    pass_env=("MY_SECRET_TOKEN",),
                    allow_sensitive_environment=True,
                )
            with self.assertRaisesRegex(ValueError, "credential alias"):
                CodexSourceBackend(
                    **{**common, "api_key_env": "MODEL_AUTH"},
                    pass_env=("MODEL_AUTH",),
                    allow_sensitive_environment=True,
                )
            for injected_name in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "BASH_ENV"):
                with self.subTest(injected_name=injected_name):
                    with self.assertRaisesRegex(ValueError, "runtime-control"):
                        CodexSourceBackend(
                            **common,
                            pass_env=(injected_name,),
                            allow_sensitive_environment=True,
                        )

    def test_rejects_host_codex_system_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                checkout,
                cargo,
                revision,
                lock_sha,
                toolchain_sha,
                _,
                _,
            ) = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            with patch(
                "scaffoldlab.codex_source._present_codex_system_configuration",
                return_value=["/etc/codex/config.toml"],
            ):
                with self.assertRaisesRegex(ValueError, "system/managed"):
                    CodexSourceBackend(
                        checkout=checkout,
                        workspace=workspace,
                        model="gpt-source-test",
                        api_key_env="TEST_CODEX_API_KEY",
                        cargo_executable=str(cargo),
                        allow_sensitive_environment=True,
                        expected_revision=revision,
                        expected_cargo_lock_sha256=lock_sha,
                        expected_rust_toolchain_sha256=toolchain_sha,
                    )

    def test_rejects_revision_lock_and_dirty_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                checkout,
                cargo,
                revision,
                lock_sha,
                toolchain_sha,
                _,
                _,
            ) = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            common = {
                "checkout": checkout,
                "workspace": workspace,
                "model": "gpt-source-test",
                "api_key_env": "TEST_CODEX_API_KEY",
                "cargo_executable": str(cargo),
                "allow_sensitive_environment": True,
                "expected_revision": revision,
                "expected_cargo_lock_sha256": lock_sha,
                "expected_rust_toolchain_sha256": toolchain_sha,
            }
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                CodexSourceBackend(**{**common, "expected_revision": "0" * 40})
            with self.assertRaisesRegex(ValueError, "Cargo.lock"):
                CodexSourceBackend(**{**common, "expected_cargo_lock_sha256": "0" * 64})
            (checkout / "tracked-dirty.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remain clean"):
                CodexSourceBackend(**common)

    def test_rejects_linked_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                checkout,
                cargo,
                revision,
                lock_sha,
                toolchain_sha,
                _,
                _,
            ) = self._checkout(root)
            metadata = root / "external-git-metadata"
            (checkout / ".git").rename(metadata)
            (checkout / ".git").write_text(f"gitdir: {metadata}\n", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(ValueError, "standalone clone"):
                CodexSourceBackend(
                    checkout=checkout,
                    workspace=workspace,
                    model="gpt-source-test",
                    api_key_env="TEST_CODEX_API_KEY",
                    cargo_executable=str(cargo),
                    allow_sensitive_environment=True,
                    expected_revision=revision,
                    expected_cargo_lock_sha256=lock_sha,
                    expected_rust_toolchain_sha256=toolchain_sha,
                )

    async def test_rejects_rustup_init_proxy_and_wrong_cargo_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proxy_backend, _, _, _ = self._backend(root / "proxy")
            rustup_init = root / "rustup-init"
            rustup_init.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            rustup_init.chmod(0o700)
            rustup = root / "rustup"
            rustup.symlink_to(rustup_init)
            cargo_proxy = root / "cargo"
            cargo_proxy.symlink_to(rustup)
            proxy_backend.cargo_executable = str(cargo_proxy)
            proxy_backend.expected_cargo_sha256 = None
            with self.assertRaisesRegex(ProviderError, "rustup Cargo proxy"):
                await proxy_backend.prepare_for_manifest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version_backend, _, _, cargo = self._backend(root)
            cargo.write_text(
                FAKE_CARGO.replace("cargo 1.95.0", "cargo 1.96.0"),
                encoding="utf-8",
            )
            cargo.chmod(0o755)
            version_backend.expected_cargo_sha256 = None
            with self.assertRaisesRegex(ProviderError, "Cargo version"):
                await version_backend.prepare_for_manifest()

    async def test_executes_native_jsonl_with_v1_spawn_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, checkout, workspace, cargo = self._backend(Path(directory))
            reported: list[Usage] = []
            await backend.prepare_for_manifest()
            manifest_provenance = backend.provenance()
            self.assertTrue(manifest_provenance["cargo"]["available"])
            self.assertTrue(manifest_provenance["rustc"]["available"])
            self.assertTrue(manifest_provenance["git"]["available"])
            self.assertTrue(manifest_provenance["runtime_executable"]["available"])
            self.assertEqual(
                manifest_provenance["cargo_version_output"].split()[1], "1.95.0"
            )
            self.assertFalse(manifest_provenance["exact_runtime_boundary_verified"])
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="delegate this task",
                        usage_reporter=reported.append,
                    )
                )

            answer = json.loads(response.text)
            self.assertEqual(answer["prompt"], "delegate this task")
            self.assertTrue(answer["credential_present"])
            self.assertTrue(answer["ephemeral"])
            self.assertTrue(answer["ignore_user_config"])
            self.assertEqual(answer["sandbox"], "workspace-write")
            self.assertIn("features.multi_agent=true", answer["configs"])
            self.assertIn('model_reasoning_effort="ultra"', answer["configs"])
            self.assertIn("features.multi_agent_v2=false", answer["configs"])
            self.assertIn(
                'shell_environment_policy.exclude=["OPENAI_API_KEY","CODEX_API_KEY"]',
                answer["configs"],
            )
            self.assertIn(
                "sandbox_workspace_write.exclude_slash_tmp=true",
                answer["configs"],
            )
            self.assertEqual(response.usage.input_tokens, 20)
            self.assertEqual(response.usage.cache_read_input_tokens, 3)
            self.assertEqual(response.usage.cache_write_input_tokens, 2)
            self.assertEqual(response.usage.output_tokens, 7)
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            self.assertEqual(reported, [response.usage])
            codex = response.raw["_scaffoldlab_codex"]
            self.assertEqual(codex["requested_multi_agent_version"], "v1")
            self.assertIsNone(codex["effective_multi_agent_version"])
            self.assertFalse(codex["effective_multi_agent_version_verified"])
            self.assertEqual(codex["spawn_agent_calls_observed"], 1)
            self.assertEqual(
                codex["spawned_child_thread_ids_observed"], ["thread-child"]
            )
            self.assertEqual(
                codex["completed_collaboration_tools"],
                ["spawn_agent", "send_input", "wait", "close_agent"],
            )
            self.assertTrue(codex["multi_agent_execution_observed"])
            trial_cwd = Path(response.raw["_scaffoldlab_workspace"]["trial_cwd"])
            self.assertFalse(trial_cwd.exists())
            self.assertEqual(
                (workspace / "task.txt").read_text(encoding="utf-8"),
                "immutable seed\n",
            )
            self.assertFalse((workspace / "agent-output.txt").exists())
            self.assertEqual(
                json.loads(cargo.with_suffix(".log").read_text(encoding="utf-8")),
                ["build", "--locked", "--release", "--bin", "codex"],
            )
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")
            provenance = backend.provenance()
            self.assertEqual(
                provenance["workspace_isolation"],
                "fresh-disposable-copy-and-git-baseline-per-call",
            )
            self.assertFalse(provenance["runtime_source_identity_verified"])
            self.assertEqual(
                provenance["build_target_isolation"],
                "fresh-temporary-CARGO_TARGET_DIR",
            )
            self.assertFalse(backend.executable.is_relative_to(checkout))
            self.assertTrue(
                provenance["credential_removed_from_direct_shell_environment"]
            )
            self.assertFalse(provenance["credential_process_isolation_guaranteed"])
            self.assertNotIn("scoped-secret", json.dumps(provenance))
            source = response.raw["_scaffoldlab_source"]
            self.assertTrue(source["git"]["available"])
            self.assertTrue(source["cargo"]["available"])
            self.assertTrue(source["rustc"]["available"])
            self.assertTrue(source["executable"]["available"])
            self.assertIsNotNone(source["source_export_tree_sha256"])

    async def test_v2_limits_and_no_spawn_remain_single_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _, _ = self._backend(Path(directory), multi_agent_version="v2")
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="no spawn",
                    )
                )
            answer = json.loads(response.text)
            self.assertIn("features.multi_agent_v2.enabled=true", answer["configs"])
            self.assertIn(
                "features.multi_agent_v2.max_concurrent_threads_per_session=4",
                answer["configs"],
            )
            self.assertIn(
                "features.multi_agent_v2.min_wait_timeout_ms=10000",
                answer["configs"],
            )
            self.assertIn(
                "features.multi_agent_v2.default_wait_timeout_ms=30000",
                answer["configs"],
            )
            self.assertIn(
                "features.multi_agent_v2.max_wait_timeout_ms=45000",
                answer["configs"],
            )
            codex = response.raw["_scaffoldlab_codex"]
            self.assertEqual(codex["requested_multi_agent_version"], "v2")
            self.assertEqual(codex["effective_multi_agent_version"], "v2")
            self.assertTrue(codex["effective_multi_agent_version_verified"])
            self.assertEqual(codex["spawn_agent_calls_observed"], 0)
            self.assertFalse(codex["multi_agent_execution_observed"])
            self.assertEqual(codex["v2_total_concurrency_including_root"], 4)

    async def test_failed_spawn_is_not_claimed_as_multi_agent_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _, _ = self._backend(Path(directory))
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="failed spawn",
                    )
                )
            codex = response.raw["_scaffoldlab_codex"]
            self.assertEqual(codex["completed_collaboration_tools"], [])
            self.assertEqual(codex["spawn_agent_calls_observed"], 0)
            self.assertFalse(codex["multi_agent_execution_observed"])

    async def test_strips_inherited_git_metadata_before_fresh_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, workspace, _ = self._backend(root)
            nested = workspace / "nested"
            nested.mkdir()
            (nested / ".git").write_text(
                "gitdir: ../../outside/modules/nested\n", encoding="utf-8"
            )
            outside = root / "outside-git"
            outside.mkdir()
            (workspace / ".git").symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="no spawn",
                    )
                )
            answer = json.loads(response.text)
            self.assertFalse(answer["nested_git_metadata_present"])
            copied = response.raw["_scaffoldlab_workspace"]
            self.assertFalse(copied["inherited_git_metadata"])
            self.assertEqual(copied["git_baseline"], "fresh-standalone-commit")
            self.assertEqual(
                copied["pre_trial_tree_sha256"],
                copied["post_git_baseline_tree_sha256"],
            )
            self.assertEqual(list(outside.iterdir()), [])

    async def test_exports_bounded_binary_swe_patch_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, workspace, _ = self._backend(Path(directory))
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="edit files",
                    )
                )
            workspace_raw = response.raw["_scaffoldlab_workspace"]
            payload = workspace_raw["patch_artifact"]
            self.assertIsInstance(payload, SWEPatchPayload)
            patch_text = payload.content.decode("utf-8")
            self.assertIn("diff --git a/task.txt b/task.txt", patch_text)
            self.assertIn("+edited by Codex", patch_text)
            self.assertIn(
                "diff --git a/agent-output.txt b/agent-output.txt", patch_text
            )
            self.assertIn("diff --git a/new-binary.bin b/new-binary.bin", patch_text)
            self.assertIn("GIT binary patch", patch_text)
            self.assertEqual(workspace_raw["patch_bytes"], len(payload.content))
            self.assertEqual(
                (workspace / "task.txt").read_text(encoding="utf-8"),
                "immutable seed\n",
            )

    async def test_exports_preexisting_and_new_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, workspace, _ = self._backend(
                Path(directory), ignored_fixture=True
            )
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="edit ignored files",
                    )
                )

            payload = response.raw["_scaffoldlab_workspace"]["patch_artifact"]
            self.assertIsInstance(payload, SWEPatchPayload)
            patch_text = payload.content.decode("utf-8")
            self.assertIn(
                "diff --git a/existing.ignored b/existing.ignored", patch_text
            )
            self.assertIn("-ignored baseline", patch_text)
            self.assertIn("+edited ignored baseline", patch_text)
            self.assertIn("diff --git a/new.ignored b/new.ignored", patch_text)
            self.assertIn("+new ignored output", patch_text)
            self.assertEqual(
                (workspace / "existing.ignored").read_text(encoding="utf-8"),
                "ignored baseline\n",
            )
            self.assertFalse((workspace / "new.ignored").exists())

    async def test_rehashes_content_after_disposable_git_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, _, _ = self._backend(root)
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            wrapper = root / "mutating-git"
            wrapper.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, subprocess, sys\n"
                f"real_git = {real_git!r}\n"
                "if 'add' in sys.argv[1:]:\n"
                "    pathlib.Path('AGENTS.md').write_text('injected\\n')\n"
                "raise SystemExit(subprocess.run([real_git, *sys.argv[1:]]).returncode)\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o700)
            backend.git_executable = str(wrapper)
            backend.expected_git_sha256 = None
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "changed task content"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/codex-source/root",
                            role="session",
                            prompt="task",
                        )
                    )

    async def test_rejects_client_owned_actions_and_mutated_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _, workspace, _ = self._backend(Path(directory))
            with self.assertRaisesRegex(ProviderError, "owns system prompts"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="task",
                        system="replacement system",
                        tools=(ToolDefinition(name="shell"),),
                    )
                )
            with self.assertRaisesRegex(ProviderError, "continuation state"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/codex-source/root",
                        role="session",
                        prompt="task",
                        continuation={},
                    )
                )
            (workspace / "late.txt").write_text("changed\n", encoding="utf-8")
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "changed after"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/codex-source/root",
                            role="session",
                            prompt="task",
                        )
                    )

    async def test_output_and_process_time_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, workspace, _ = self._backend(
                root / "output", max_output_bytes=1024
            )
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "output limit"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/codex-source/root",
                            role="session",
                            prompt="overflow",
                        )
                    )
            patch_backend, _, _, _ = self._backend(root / "patch", max_patch_bytes=1024)
            reported: list[Usage] = []
            with patch.dict(os.environ, {"TEST_CODEX_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(
                    ProviderError, "patch export.*output limit"
                ) as raised:
                    await patch_backend.complete(
                        ModelRequest(
                            agent_id="/codex-source/root",
                            role="session",
                            prompt="patch overflow",
                            usage_reporter=reported.append,
                        )
                    )
            self.assertEqual(reported, [raised.exception.usage])
            self.assertEqual(raised.exception.usage.input_tokens, 20)
            self.assertEqual(raised.exception.usage.output_tokens, 7)
            self.assertFalse(raised.exception.usage.complete)
            with self.assertRaisesRegex(ProviderError, "timed out"):
                await backend._run_bounded_process(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    cwd=workspace,
                    environment={"PATH": os.environ.get("PATH", "")},
                    timeout_seconds=0.05,
                    label="Codex bounded sleeper",
                )

    async def test_cancellation_terminates_the_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, workspace, _ = self._backend(root)
            pid_file = root / "sleeper.pid"
            command = [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys, time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(60)"
                ),
                str(pid_file),
            ]
            running = asyncio.create_task(
                backend._run_bounded_process(
                    command,
                    cwd=workspace,
                    environment={"PATH": os.environ.get("PATH", "")},
                    timeout_seconds=30,
                    label="Codex cancellable sleeper",
                )
            )
            pid: int | None = None
            for _ in range(100):
                try:
                    pid = int(pid_file.read_text(encoding="utf-8"))
                except (FileNotFoundError, ValueError):
                    await asyncio.sleep(0.01)
                    continue
                break
            if pid is None:
                self.fail("cancellable child did not publish its PID")
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    async def test_harness_uses_shared_ledger_without_inventing_delegation(
        self,
    ) -> None:
        backend = ScriptedBackend(
            {
                "/codex-source/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(cost_known=False, complete=False),
                        raw={
                            "_scaffoldlab_codex": {
                                "requested_multi_agent_version": "v2",
                                "effective_multi_agent_version": "v2",
                                "effective_multi_agent_version_verified": True,
                                "native_multi_agent_tools_enabled": True,
                                "spawn_agent_calls_observed": 0,
                                # The harness must derive execution from concrete
                                # spawn evidence instead of trusting a label.
                                "multi_agent_execution_observed": True,
                                "max_subagents": 3,
                                "v2_total_concurrency_including_root": 4,
                            },
                            "usage_is_incomplete": True,
                            "cost_is_unknown": True,
                        },
                    )
                ]
            }
        )
        result = await CodexSourceHarness().run(
            Task("one", "Solve this"),
            backend,
            BudgetLimits(max_model_calls=1),
        )
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.metadata["shared_budget_ledger"])
        self.assertFalse(result.metadata["upstream_source_executed"])
        self.assertTrue(result.metadata["native_multi_agent_tools_enabled"])
        self.assertFalse(result.metadata["multi_agent_execution_observed"])
        self.assertTrue(result.metadata["single_agent_execution"])
        self.assertEqual(result.metadata["fidelity"], "custom_source_runtime_boundary")
        self.assertFalse(result.usage.complete)

    async def test_harness_downgrades_unattested_end_to_end_runtime(self) -> None:
        patch_payload = SWEPatchPayload(b"diff --git a/a b/a\n")
        backend = ScriptedBackend(
            {
                "/codex-source/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(cost_known=False, complete=False),
                        raw={
                            "_scaffoldlab_source": {
                                "repository": CODEX_SOURCE_REPOSITORY,
                                "revision": CODEX_SOURCE_REVISION,
                                "official_public_pin_verified": True,
                                "exact_runtime_build_verified": True,
                                "exact_runtime_boundary_verified": False,
                            },
                            "_scaffoldlab_codex": {
                                "native_multi_agent_tools_enabled": True,
                                "spawn_agent_calls_observed": 0,
                                "spawned_child_thread_ids_observed": [],
                            },
                            "_scaffoldlab_workspace": {"patch_artifact": patch_payload},
                        },
                    )
                ]
            }
        )
        result = await CodexSourceHarness().run(
            Task("pin", "Solve this"),
            backend,
            BudgetLimits(max_model_calls=1),
        )
        self.assertEqual(
            result.metadata["fidelity"],
            "pinned_public_source_with_recorded_local_build",
        )
        self.assertIs(result.metadata["workspace"]["patch_artifact"], patch_payload)


if __name__ == "__main__":
    unittest.main()
