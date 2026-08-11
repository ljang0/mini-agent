from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.harnesses.prime_source import PrimeAgentSourceHarness
from scaffoldlab.prime_source import (
    PRIME_AGENT_BUNDLE,
    PRIME_AGENT_PACKAGE_LOCK_SHA256,
    PRIME_AGENT_SOURCE_REVISION,
    PRIME_AGENT_SOURCE_VERSION,
    PrimeAgentSourceBackend,
)
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


FAKE_NODE = r"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

bundle = pathlib.Path(sys.argv[1])
args = sys.argv[2:]
if "--version" in args:
    print("probe-home=" + os.environ.get("HOME", ""))
    print("probe-xdg=" + os.environ.get("XDG_CONFIG_HOME", ""))
    print("prime-agent 0.7.1")
    raise SystemExit(0)

prompt = sys.stdin.read()
if prompt == "timeout":
    time.sleep(5)
if prompt == "mutate source":
    bundle.write_text(bundle.read_text(encoding="utf-8") + "\nmutated\n", encoding="utf-8")

print(json.dumps({"type": "session", "version": 3, "id": "synthetic"}))
print(json.dumps({
    "type": "message_end",
    "message": {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{
            "type": "text",
            "text": json.dumps({
                "args": args,
                "bundle": str(bundle),
                "home_is_ephemeral": "scaffoldlab-prime-home-" in os.environ.get("HOME", ""),
                "no_session": "--no-session" in args,
                "pi_skip_version_check": os.environ.get("PI_SKIP_VERSION_CHECK"),
                "prompt": prompt,
            }, sort_keys=True),
        }],
        "usage": {
            "input": 11,
            "cacheRead": 2,
            "cacheWrite": 1,
            "output": 4,
            "cost": {"total": 0.25},
        },
    },
}))
print(json.dumps({"type": "agent_end"}))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
class PrimeAgentSourceTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, str, str]:
        checkout = root / "prime-agent"
        package = checkout / "packages" / "coding-agent"
        source = package / "src"
        source.mkdir(parents=True)
        (source / "main.ts").write_text(
            "export const source = true;\n", encoding="utf-8"
        )
        (checkout / "package-lock.json").write_text(
            '{"lockfileVersion":3,"name":"prime-agent"}\n', encoding="utf-8"
        )
        (checkout / "package.json").write_text(
            json.dumps({"name": "prime-agent", "version": "0.7.1"}) + "\n",
            encoding="utf-8",
        )
        (package / "package.json").write_text(
            json.dumps({"name": "@earendil-works/pi-coding-agent", "version": "0.7.1"})
            + "\n",
            encoding="utf-8",
        )
        (checkout / ".gitignore").write_text(
            "packages/coding-agent/dist/\nnode_modules/\n", encoding="utf-8"
        )
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", ".", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned Prime Agent", cwd=checkout)

        bundle = checkout / PRIME_AGENT_BUNDLE
        bundle.parent.mkdir(parents=True)
        bundle.write_text("// synthetic caller-built bundle\n", encoding="utf-8")
        self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")
        return checkout, _git("rev-parse", "HEAD", cwd=checkout), _sha256(bundle)

    def _runtime_tools(self, root: Path) -> tuple[Path, Path]:
        node = root / "synthetic-node"
        npm = root / "synthetic-npm"
        node.write_text(FAKE_NODE, encoding="utf-8")
        npm.write_text("#!/bin/sh\nprintf '11.0.0\\n'\n", encoding="utf-8")
        node.chmod(0o700)
        npm.chmod(0o700)
        return node, npm

    def _backend(
        self,
        *,
        root: Path,
        checkout: Path,
        revision: str,
        bundle_sha256: str,
        timeout_seconds: float = 10.0,
    ) -> PrimeAgentSourceBackend:
        node, npm = self._runtime_tools(root)
        task_cwd = root / "task"
        task_cwd.mkdir()
        return PrimeAgentSourceBackend(
            checkout=checkout,
            cwd=task_cwd,
            node_executable=str(node),
            npm_executable=str(npm),
            provider="openai",
            model="synthetic-model",
            expected_revision=revision,
            expected_lock_sha256=_sha256(checkout / "package-lock.json"),
            expected_node_sha256=_sha256(node),
            expected_npm_sha256=_sha256(npm),
            expected_bundle_sha256=bundle_sha256,
            timeout_seconds=timeout_seconds,
        )

    async def test_executes_caller_built_bundle_through_pinned_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            backend = self._backend(
                root=root,
                checkout=checkout,
                revision=revision,
                bundle_sha256=bundle_sha256,
            )
            response = await backend.complete(
                ModelRequest(
                    agent_id="/prime-agent/root",
                    role="prime_agent_session",
                    prompt="implement the fix",
                )
            )

            answer = json.loads(response.text)
            self.assertEqual(answer["prompt"], "implement the fix")
            self.assertEqual(answer["pi_skip_version_check"], "1")
            self.assertTrue(answer["home_is_ephemeral"])
            self.assertTrue(answer["no_session"])
            self.assertEqual(Path(answer["bundle"]), backend.runtime_bundle)
            self.assertNotEqual(Path(answer["bundle"]), checkout / PRIME_AGENT_BUNDLE)
            self.assertEqual(
                answer["args"],
                [
                    "--mode",
                    "json",
                    "--provider",
                    "openai",
                    "--model",
                    "synthetic-model",
                    "--no-session",
                ],
            )
            self.assertEqual(
                (response.usage.input_tokens, response.usage.output_tokens), (14, 4)
            )
            self.assertEqual(response.usage.cost_usd, 0.25)
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")

            provenance = backend.provenance()
            self.assertTrue(
                provenance["caller_built_source_bundle_entrypoint_executed"]
            )
            self.assertTrue(provenance["package_lock"]["verified"])
            self.assertEqual(
                provenance["caller_built_source_bundle"]["sha256"], bundle_sha256
            )
            self.assertEqual(
                provenance["node_runtime"]["sha256"], _sha256(root / "synthetic-node")
            )
            self.assertEqual(
                provenance["npm_build_tool"]["sha256"], _sha256(root / "synthetic-npm")
            )
            self.assertTrue(provenance["caller_pinned_runtime_identity_complete"])
            self.assertFalse(provenance["reproducible_build_identity_complete"])
            self.assertFalse(
                provenance["runtime_package_identity_matches_audited_release"]
            )
            self.assertFalse(provenance["dependency_tree_content_verified"])
            self.assertFalse(provenance["source_identity_matches_audited_release"])
            self.assertFalse(provenance["source_or_protocol_pin_verified"])
            self.assertFalse(
                provenance["checkout_revision_and_lock_match_audited_release"]
            )
            self.assertFalse(
                provenance["adversarial_source_content_attestation_verified"]
            )
            self.assertTrue(provenance["git_runtime"]["available"])
            self.assertFalse(provenance["caller_worktree_bundle_executed_directly"])
            self.assertEqual(
                provenance["private_runtime_bundle"]["sha256"], bundle_sha256
            )

    async def test_version_probe_uses_disposable_home_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            backend = self._backend(
                root=root,
                checkout=checkout,
                revision=revision,
                bundle_sha256=bundle_sha256,
            )
            with patch.dict(
                os.environ,
                {
                    "HOME": "/host/home-must-not-be-observed",
                    "XDG_CONFIG_HOME": "/host/config-must-not-be-observed",
                },
            ):
                output = await backend.verify_version()
            self.assertIn("scaffoldlab-prime-version-home-", output)
            self.assertNotIn("/host/home-must-not-be-observed", output)
            self.assertNotIn("/host/config-must-not-be-observed", output)

    def test_rejects_linked_git_worktree_task_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            node, npm = self._runtime_tools(root)
            task = root / "linked-task"
            task.mkdir()
            (task / ".git").write_text(
                "gitdir: /external/shared/repository\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "linked Git worktree"):
                PrimeAgentSourceBackend(
                    checkout=checkout,
                    cwd=task,
                    node_executable=str(node),
                    npm_executable=str(npm),
                    expected_revision=revision,
                    expected_lock_sha256=_sha256(checkout / "package-lock.json"),
                    expected_bundle_sha256=bundle_sha256,
                )
            plain_task = root / "plain-task"
            plain_task.mkdir()
            for runtime_override in ("PI_SKIP_VERSION_CHECK", "NODE_OPTIONS"):
                with self.assertRaisesRegex(ValueError, "runtime-control"):
                    PrimeAgentSourceBackend(
                        checkout=checkout,
                        cwd=plain_task,
                        node_executable=str(node),
                        npm_executable=str(npm),
                        expected_revision=revision,
                        expected_lock_sha256=_sha256(checkout / "package-lock.json"),
                        expected_bundle_sha256=bundle_sha256,
                        pass_env=(runtime_override,),
                        allow_sensitive_environment=True,
                    )
            provider_credentials = PrimeAgentSourceBackend(
                checkout=checkout,
                cwd=plain_task,
                node_executable=str(node),
                npm_executable=str(npm),
                expected_revision=revision,
                expected_lock_sha256=_sha256(checkout / "package-lock.json"),
                expected_bundle_sha256=bundle_sha256,
                pass_env=("PRIME_API_KEY", "PRIME_TEAM_ID"),
                allow_sensitive_environment=True,
            )
            self.assertEqual(
                provider_credentials.pass_env,
                ("PRIME_API_KEY", "PRIME_TEAM_ID"),
            )

    def test_rejects_linked_source_checkout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            node, npm = self._runtime_tools(root)
            task = root / "task"
            task.mkdir()
            external_metadata = root / "shared-git-metadata"
            (checkout / ".git").rename(external_metadata)
            (checkout / ".git").symlink_to(external_metadata, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "standalone clone"):
                PrimeAgentSourceBackend(
                    checkout=checkout,
                    cwd=task,
                    node_executable=str(node),
                    npm_executable=str(npm),
                    expected_revision=revision,
                    expected_lock_sha256=_sha256(checkout / "package-lock.json"),
                    expected_bundle_sha256=bundle_sha256,
                )

    async def test_harness_preserves_one_outer_call_and_marks_source_scope(
        self,
    ) -> None:
        backend = ScriptedBackend(
            {
                "/prime-agent/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(
                            input_tokens=9,
                            output_tokens=3,
                            cost_known=False,
                            complete=False,
                        ),
                        raw=[
                            {"type": "session", "version": 3},
                            {"type": "agent_end"},
                        ],
                    )
                ]
            }
        )
        result = await PrimeAgentSourceHarness().run(
            Task("task", "question", context="context"),
            backend,
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.metadata["source_checkout_adapter"])
        self.assertTrue(result.metadata["caller_built_runtime_adapter"])
        self.assertFalse(result.metadata["released_runtime_adapter"])
        self.assertTrue(
            result.metadata["caller_built_source_bundle_entrypoint_executed"]
        )
        self.assertFalse(result.metadata["whole_tree_usage_verified"])
        self.assertEqual(result.metadata["fidelity"], "caller_built_runtime_study")
        self.assertFalse(result.metadata["source_or_protocol_pin_verified"])
        self.assertEqual(
            backend.requests[0].prompt,
            "question\n\n<context>\ncontext\n</context>",
        )

    def test_rejects_wrong_revision_lock_dirty_tree_and_missing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            node, npm = self._runtime_tools(root)
            task = root / "task"
            task.mkdir()
            common = {
                "checkout": checkout,
                "cwd": task,
                "node_executable": str(node),
                "npm_executable": str(npm),
                "expected_lock_sha256": _sha256(checkout / "package-lock.json"),
                "expected_node_sha256": _sha256(node),
                "expected_npm_sha256": _sha256(npm),
                "expected_bundle_sha256": bundle_sha256,
            }
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                PrimeAgentSourceBackend(
                    **common,
                    expected_revision="0" * 40,
                )
            with self.assertRaisesRegex(ValueError, "package-lock SHA-256 mismatch"):
                PrimeAgentSourceBackend(
                    **{**common, "expected_lock_sha256": "0" * 64},
                    expected_revision=revision,
                )

            (checkout / "dirty.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                PrimeAgentSourceBackend(**common, expected_revision=revision)
            (checkout / "dirty.txt").unlink()
            (checkout / PRIME_AGENT_BUNDLE).unlink()
            with self.assertRaisesRegex(
                ValueError, "official source bundle is missing"
            ):
                PrimeAgentSourceBackend(**common, expected_revision=revision)

    async def test_rejects_client_tools_and_runtime_bundle_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            backend = self._backend(
                root=root,
                checkout=checkout,
                revision=revision,
                bundle_sha256=bundle_sha256,
            )
            with self.assertRaisesRegex(ProviderError, "owns tools"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/prime-agent/root",
                        role="prime_agent_session",
                        prompt="task",
                        tools=(ToolDefinition(name="shell"),),
                    )
                )
            with self.assertRaisesRegex(
                ProviderError, "private runtime bundle changed"
            ) as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/prime-agent/root",
                        role="prime_agent_session",
                        prompt="mutate source",
                    )
                )
            self.assertIsNotNone(raised.exception.usage)
            assert raised.exception.usage is not None
            self.assertEqual(
                (
                    raised.exception.usage.input_tokens,
                    raised.exception.usage.output_tokens,
                ),
                (14, 4),
            )
            self.assertFalse(raised.exception.usage.complete)

    async def test_timeout_is_bounded_with_unknown_tree_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision, bundle_sha256 = self._checkout(root)
            backend = self._backend(
                root=root,
                checkout=checkout,
                revision=revision,
                bundle_sha256=bundle_sha256,
                timeout_seconds=0.1,
            )
            # Keep the source version probe outside the deliberately tiny session
            # deadline so this test measures the bounded agent tree, not interpreter
            # startup variance on a loaded CI host.
            backend.timeout_seconds = 10.0
            await backend.verify_version()
            backend.timeout_seconds = 0.1
            with self.assertRaisesRegex(ProviderError, "timed out") as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/prime-agent/root",
                        role="prime_agent_session",
                        prompt="timeout",
                    )
                )
            self.assertIsNotNone(raised.exception.usage)
            assert raised.exception.usage is not None
            self.assertFalse(raised.exception.usage.cost_known)
            self.assertFalse(raised.exception.usage.complete)

    def test_audited_constants_match_source_audit(self) -> None:
        self.assertEqual(PRIME_AGENT_SOURCE_VERSION, "0.7.1")
        self.assertEqual(
            PRIME_AGENT_SOURCE_REVISION,
            "95afd319a78ae017a41241d50b013d656a0685ce",
        )
        self.assertEqual(
            PRIME_AGENT_PACKAGE_LOCK_SHA256,
            "39ee303bca10c0933cf917275613c8f44099f50de1650a5c356e7cda02b701e8",
        )
        self.assertEqual(
            PRIME_AGENT_BUNDLE.as_posix(),
            "packages/coding-agent/dist/bundle/cli.js",
        )


if __name__ == "__main__":
    unittest.main()
