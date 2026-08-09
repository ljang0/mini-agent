from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.harnesses.kimi_code import KimiCodeUpstreamHarness
from scaffoldlab.kimi_upstream import KimiCodeUpstreamBackend
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


FAKE_TSX = r"""#!/usr/bin/env python3
import json
import os
import sys

if "--version" in sys.argv:
    print("kimi 0.34.0")
    raise SystemExit(0)

mutation_target = os.environ.get("SCAFFOLDLAB_TEST_MUTATE_KIMI_SOURCE")
if mutation_target:
    with open(mutation_target, "a", encoding="utf-8") as stream:
        stream.write("// mutated by synthetic runtime\n")

print(json.dumps({"role": "meta", "type": "system.version", "version": "0.34.0"}))
print(json.dumps({
    "role": "assistant",
    "content": json.dumps({
        "cwd": os.getcwd(),
        "experimental": os.environ.get("KIMI_CODE_EXPERIMENTAL_FLAG"),
        "model": os.environ.get("KIMI_MODEL_NAME"),
        "provider": os.environ.get("KIMI_MODEL_PROVIDER_TYPE"),
        "credential_present": bool(os.environ.get("KIMI_MODEL_API_KEY")),
        "home": os.environ.get("HOME"),
        "xdg_config": os.environ.get("XDG_CONFIG_HOME"),
        "xdg_cache": os.environ.get("XDG_CACHE_HOME"),
        "tmpdir": os.environ.get("TMPDIR"),
        "node_runtime": os.environ.get("SCAFFOLDLAB_FAKE_NODE_EXECUTED"),
    }, sort_keys=True),
    "tool_calls": [{
        "type": "function",
        "id": "swarm-1",
        "function": {"name": "AgentSwarm", "arguments": "{}"},
    }],
}))
prompt_value = sys.argv[sys.argv.index("--prompt") + 1]
if prompt_value == "duplicate version":
    print(json.dumps({"role": "meta", "type": "system.version", "version": "0.34.0"}))
if prompt_value != "truncated":
    print(json.dumps({
        "role": "meta",
        "type": "session.resume_hint",
        "session_id": "synthetic",
        "command": "kimi -r synthetic",
        "content": "resume",
    }))
"""

FAKE_NODE = r"""#!/usr/bin/env python3
import os
import sys

os.environ["SCAFFOLDLAB_FAKE_NODE_EXECUTED"] = os.path.realpath(sys.argv[0])
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
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
class KimiCodeUpstreamTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, Path, str, str]:
        checkout = root / "kimi-code"
        entrypoint = checkout / "apps" / "kimi-code" / "src" / "main.ts"
        package_json = checkout / "apps" / "kimi-code" / "package.json"
        loader = checkout / "build" / "register-raw-text-loader.mjs"
        tsx = checkout / "tooling" / "tsx"
        node = checkout / "tooling" / "node"
        for parent in (entrypoint.parent, loader.parent, tsx.parent, node.parent):
            parent.mkdir(parents=True, exist_ok=True)
        entrypoint.write_text("// synthetic source entrypoint\n", encoding="utf-8")
        package_json.write_text(
            json.dumps({"name": "@moonshot-ai/kimi-code", "version": "0.34.0"}),
            encoding="utf-8",
        )
        loader.write_text("// synthetic loader\n", encoding="utf-8")
        (checkout / "pnpm-lock.yaml").write_text(
            "lockfileVersion: '9.0'\n", encoding="utf-8"
        )
        tsx.write_text(FAKE_TSX, encoding="utf-8")
        tsx.chmod(0o755)
        node.write_text(FAKE_NODE, encoding="utf-8")
        node.chmod(0o755)
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", ".", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned Kimi Code", cwd=checkout)
        revision = _git("rev-parse", "HEAD", cwd=checkout)
        digest = hashlib.sha256(tsx.read_bytes()).hexdigest()
        return checkout, tsx, revision, digest

    async def test_executes_clean_source_entrypoint_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("fixture\n", encoding="utf-8")
            backend = KimiCodeUpstreamBackend(
                checkout=checkout,
                cwd=workspace,
                model="kimi-k2.5",
                api_key_env="TEST_KIMI_API_KEY",
                expected_revision=revision,
                node_executable=str(checkout / "tooling" / "node"),
                tsx_executable=tsx,
                expected_node_sha256=hashlib.sha256(
                    (checkout / "tooling" / "node").read_bytes()
                ).hexdigest(),
                expected_tsx_sha256=digest,
                max_swarm_concurrency=3,
                max_steps_per_turn=9,
                allow_sensitive_environment=True,
            )
            with patch.dict(os.environ, {"TEST_KIMI_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/kimi-code/root",
                        role="kimi_code_upstream_session",
                        prompt="Solve the task",
                    )
                )

            answer = json.loads(response.text)
            self.assertEqual(answer["cwd"], str(workspace.resolve()))
            self.assertIsNone(answer["experimental"])
            self.assertEqual(answer["model"], "kimi-k2.5")
            self.assertEqual(answer["provider"], "kimi")
            self.assertTrue(answer["credential_present"])
            self.assertEqual(
                answer["node_runtime"], str((checkout / "tooling" / "node").resolve())
            )
            for field in ("home", "xdg_config", "xdg_cache", "tmpdir"):
                self.assertIn("scaffoldlab-kimi-home-", answer[field])
                self.assertFalse(Path(answer[field]).exists())
            self.assertFalse(response.usage.complete)
            self.assertFalse(response.usage.cost_known)
            self.assertEqual(response.raw["events"][0]["type"], "system.version")
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")
            provenance = backend.provenance()
            self.assertEqual(provenance["expected_revision"], revision)
            self.assertEqual(provenance["runtime_executable"]["sha256"], digest)
            self.assertEqual(
                provenance["node_runtime"]["sha256"],
                hashlib.sha256(
                    (checkout / "tooling" / "node").read_bytes()
                ).hexdigest(),
            )
            self.assertTrue(provenance["prompt_visible_to_local_process_inspection"])
            self.assertTrue(provenance["tracked_source_tree"]["verified"])
            self.assertTrue(provenance["runtime_source_identity_verified"])
            self.assertTrue(provenance["caller_worktree_executed"])
            self.assertFalse(provenance["private_source_export_executed"])
            self.assertFalse(
                provenance["adversarial_full_runtime_content_attestation_verified"]
            )
            self.assertFalse(
                provenance["ignored_or_generated_dependency_content_verified"]
            )
            self.assertTrue(provenance["git_runtime"]["available"])
            self.assertNotIn("scoped-secret", json.dumps(provenance))

    def test_rejects_assume_unchanged_tracked_source_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            entrypoint = checkout / "apps" / "kimi-code" / "src" / "main.ts"
            entrypoint.write_text("// hidden caller overlay\n", encoding="utf-8")
            _git(
                "update-index",
                "--assume-unchanged",
                "apps/kimi-code/src/main.ts",
                cwd=checkout,
            )
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")

            with self.assertRaisesRegex(ValueError, "differs from the pinned commit"):
                KimiCodeUpstreamBackend(
                    checkout=checkout,
                    cwd=workspace,
                    model="kimi-k2.5",
                    api_key_env="TEST_KIMI_API_KEY",
                    expected_revision=revision,
                    node_executable=str(checkout / "tooling" / "node"),
                    tsx_executable=tsx,
                    expected_tsx_sha256=digest,
                    allow_sensitive_environment=True,
                )

    def test_base_url_and_environment_name_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            common = {
                "checkout": checkout,
                "cwd": workspace,
                "model": "kimi-k2.5",
                "api_key_env": "TEST_KIMI_API_KEY",
                "expected_revision": revision,
                "tsx_executable": tsx,
                "expected_tsx_sha256": digest,
                "allow_sensitive_environment": True,
            }
            with self.assertRaisesRegex(ValueError, "use HTTPS"):
                KimiCodeUpstreamBackend(**common, base_url="http://models.example/v1")
            loopback = KimiCodeUpstreamBackend(
                **common, base_url="http://127.0.0.1:8080/v1"
            )
            self.assertFalse(loopback.provenance()["insecure_base_url_acknowledged"])
            acknowledged = KimiCodeUpstreamBackend(
                **common,
                base_url="http://models.example/v1",
                allow_insecure_base_url=True,
            )
            self.assertTrue(acknowledged.provenance()["insecure_base_url_acknowledged"])
            for keyword in (
                {"api_key_env": "BAD\x00NAME"},
                {"pass_env": ("BAD\x00NAME",)},
            ):
                with self.assertRaisesRegex(ValueError, "environment name"):
                    KimiCodeUpstreamBackend(**{**common, **keyword})

            for runtime_override in (
                "KIMI_CODE_LEGACY_FLAG",
                "KIMI_CODE_EXPERIMENTAL_FLAG",
                "NODE_OPTIONS",
                "DYLD_INSERT_LIBRARIES",
            ):
                with self.assertRaisesRegex(ValueError, "runtime-control"):
                    KimiCodeUpstreamBackend(**common, pass_env=(runtime_override,))

            (workspace / ".git").write_text(
                "gitdir: /external/shared/repository\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "linked Git worktree"):
                KimiCodeUpstreamBackend(**common)

    def test_rejects_linked_source_checkout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            external_metadata = root / "shared-git-metadata"
            (checkout / ".git").rename(external_metadata)
            (checkout / ".git").symlink_to(external_metadata, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "standalone clone"):
                KimiCodeUpstreamBackend(
                    checkout=checkout,
                    cwd=workspace,
                    model="kimi-k2.5",
                    api_key_env="TEST_KIMI_API_KEY",
                    expected_revision=revision,
                    tsx_executable=tsx,
                    expected_tsx_sha256=digest,
                    allow_sensitive_environment=True,
                )

    async def test_rejects_dirty_checkout_and_client_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            backend = KimiCodeUpstreamBackend(
                checkout=checkout,
                cwd=workspace,
                model="kimi-k2.5",
                api_key_env="TEST_KIMI_API_KEY",
                expected_revision=revision,
                node_executable=str(checkout / "tooling" / "node"),
                tsx_executable=tsx,
                expected_node_sha256=hashlib.sha256(
                    (checkout / "tooling" / "node").read_bytes()
                ).hexdigest(),
                expected_tsx_sha256=digest,
                allow_sensitive_environment=True,
            )
            with self.assertRaisesRegex(ProviderError, "owns system prompts"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/kimi-code/root",
                        role="kimi_code_upstream_session",
                        prompt="task",
                        tools=(ToolDefinition(name="shell"),),
                    )
                )
            (checkout / "untracked.txt").write_text("dirty", encoding="utf-8")
            with patch.dict(os.environ, {"TEST_KIMI_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ValueError, "must be clean"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/kimi-code/root",
                            role="kimi_code_upstream_session",
                            prompt="task",
                        )
                    )

    async def test_fails_closed_if_runtime_mutates_pinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            entrypoint = checkout / "apps" / "kimi-code" / "src" / "main.ts"
            backend = KimiCodeUpstreamBackend(
                checkout=checkout,
                cwd=workspace,
                model="kimi-k2.5",
                api_key_env="TEST_KIMI_API_KEY",
                expected_revision=revision,
                node_executable=str(checkout / "tooling" / "node"),
                tsx_executable=tsx,
                expected_node_sha256=hashlib.sha256(
                    (checkout / "tooling" / "node").read_bytes()
                ).hexdigest(),
                expected_tsx_sha256=digest,
                pass_env=("SCAFFOLDLAB_TEST_MUTATE_KIMI_SOURCE",),
                allow_sensitive_environment=True,
            )
            with patch.dict(
                os.environ,
                {
                    "TEST_KIMI_API_KEY": "scoped-secret",
                    "SCAFFOLDLAB_TEST_MUTATE_KIMI_SOURCE": str(entrypoint),
                },
            ):
                with self.assertRaisesRegex(ProviderError, "source identity violation"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/kimi-code/root",
                            role="kimi_code_upstream_session",
                            prompt="task",
                        )
                    )

    async def test_rejects_truncated_or_duplicate_terminal_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, tsx, revision, digest = self._checkout(root)
            workspace = root / "workspace"
            workspace.mkdir()
            node = checkout / "tooling" / "node"
            backend = KimiCodeUpstreamBackend(
                checkout=checkout,
                cwd=workspace,
                model="kimi-k2.5",
                api_key_env="TEST_KIMI_API_KEY",
                expected_revision=revision,
                node_executable=str(node),
                tsx_executable=tsx,
                expected_node_sha256=hashlib.sha256(node.read_bytes()).hexdigest(),
                expected_tsx_sha256=digest,
                allow_sensitive_environment=True,
            )
            with patch.dict(os.environ, {"TEST_KIMI_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "session.resume_hint"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/kimi-code/root",
                            role="kimi_code_upstream_session",
                            prompt="truncated",
                        )
                    )
                with self.assertRaisesRegex(ProviderError, "exactly one version"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/kimi-code/root",
                            role="kimi_code_upstream_session",
                            prompt="duplicate version",
                        )
                    )

    async def test_harness_records_external_tree_without_fake_usage(self) -> None:
        backend = ScriptedBackend(
            {
                "/kimi-code/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(cost_known=False, complete=False),
                        raw={
                            "events": [
                                {
                                    "role": "assistant",
                                    "content": "delegating",
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "AgentSwarm",
                                                "arguments": "{}",
                                            }
                                        }
                                    ],
                                }
                            ],
                            "workspace": {"post_tree_sha256": "abc"},
                        },
                    )
                ]
            }
        )
        result = await KimiCodeUpstreamHarness().run(
            Task("one", "Solve this"),
            backend,
            BudgetLimits(max_model_calls=1),
        )
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.metadata["source_entrypoint_executed"])
        self.assertEqual(result.metadata["agent_swarm_calls_observed"], 1)
        self.assertFalse(result.metadata["whole_tree_usage_reported_by_upstream"])
        self.assertFalse(result.usage.complete)


if __name__ == "__main__":
    unittest.main()
