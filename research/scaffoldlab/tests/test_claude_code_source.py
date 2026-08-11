from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import scaffoldlab.claude_code_source as claude_code_source
from scaffoldlab.claude_code_source import (
    CLAUDE_CODE_DISTRIBUTION_VERSION,
    ClaudeCodeAgentTeamsDistributionBackend,
)
from scaffoldlab.harnesses.claude_code_source import (
    ClaudeCodeAgentTeamsDistributionHarness,
)
from scaffoldlab.environments.swe import SWEPatchPayload
from scaffoldlab.providers import ProviderError
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import (
    BudgetLimits,
    ModelRequest,
    ModelResponse,
    RunFailed,
    Task,
    ToolDefinition,
    Usage,
)


FAKE_CLAUDE = r"""#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if "--version" in sys.argv:
    if "--teammate-mode" not in sys.argv or "in-process" not in sys.argv:
        print("missing teammate flag", file=sys.stderr)
        raise SystemExit(2)
    if "scaffoldlab-claude-code-version-" not in os.environ.get("HOME", ""):
        print("version probe used caller HOME", file=sys.stderr)
        raise SystemExit(3)
    if os.environ.get("DISABLE_UPDATES") != "1":
        print("version probe did not disable updates", file=sys.stderr)
        raise SystemExit(4)
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        print("version probe used caller config", file=sys.stderr)
        raise SystemExit(5)
    print("2.1.226 (Claude Code)")
    raise SystemExit(0)

prompt = sys.stdin.read()
if prompt.startswith("FLOOD"):
    sys.stdout.write("x" * 65536)
    sys.stdout.flush()
    raise SystemExit(0)
if prompt.startswith("HANG"):
    marker = prompt.split("PIDFILE:", 1)[1].strip()
    child = subprocess.Popen(["sleep", "30"])
    Path(marker).write_text(str(child.pid), encoding="ascii")
    time.sleep(30)
    raise SystemExit(0)

Path("team-output.txt").write_text("edited disposable copy\n", encoding="utf-8")
if prompt.startswith("IGNORED_FILES"):
    Path("existing.ignored").write_text(
        "edited ignored baseline\n", encoding="utf-8"
    )
    Path("new.ignored").write_text("new ignored output\n", encoding="utf-8")
config = Path(os.environ["CLAUDE_CONFIG_DIR"])
team = config / "teams" / "session-00000000"
if not prompt.startswith("NO_TEAM"):
    tasks = config / "tasks" / "session-00000000"
    tasks.mkdir(parents=True)
    (tasks / "1.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    team.mkdir(parents=True)
    (team / "config.json").write_text(json.dumps({
        "members": [
            {"name": "team-lead", "agentId": "lead@session-00000000", "agentType": "team-lead"},
            {"name": "researcher", "agentId": "researcher@session-00000000", "agentType": "general-purpose"},
            {"name": "reviewer", "agentId": "reviewer@session-00000000"},
        ]
    }), encoding="utf-8")
    # Keep the live config around long enough for the parent-side observer to
    # snapshot it. Claude Code removes this directory when the session exits.
    time.sleep(0.05)

answer = json.dumps({
    "cwd": os.getcwd(),
    "home": os.environ.get("HOME"),
    "config": os.environ.get("CLAUDE_CONFIG_DIR"),
    "teams": os.environ.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"),
    "credential_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
    "bare": "--bare" in sys.argv,
    "setting_sources": sys.argv[sys.argv.index("--setting-sources") + 1],
    "teammate_mode": sys.argv[sys.argv.index("--teammate-mode") + 1],
    "max_budget": sys.argv[sys.argv.index("--max-budget-usd") + 1],
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"], text=True
    ).strip(),
    "nested_git_exists": Path("vendor/pkg/.git").exists(),
}, sort_keys=True)
print(json.dumps({
    "type": "system",
    "subtype": "init",
    "claude_code_version": "2.1.226",
}))
if not prompt.startswith("NO_TEAM"):
    print(json.dumps({
        "type": "assistant",
        "message": {"content": [
            {"type": "tool_use", "name": "Agent", "input": {
                "name": "researcher", "description": "Research",
                "prompt": "research", "run_in_background": True,
                "team_name": "legacy-team" if prompt.startswith("LEGACY_TEAM") else None,
            }},
            {"type": "tool_use", "name": "Agent", "input": {
                "name": "reviewer", "description": "Review",
                "prompt": "review", "run_in_background": True,
            }},
            {"type": "tool_use", "name": "SendMessage", "input": {
                "to": "reviewer", "message": "challenge this",
            }},
        ]},
    }))
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": answer,
    "session_id": "00000000-0000-4000-8000-000000000001",
    "total_cost_usd": 0.125,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 10,
    },
}))
if team.exists():
    shutil.rmtree(team.parent)
"""


@unittest.skipUnless(os.name == "posix", "adapter requires POSIX process groups")
class ClaudeCodeDistributionTests(unittest.IsolatedAsyncioTestCase):
    native_package = "@anthropic-ai/claude-code-test-posix"

    def _distribution(self, root: Path) -> tuple[Path, str]:
        distribution = root / "distribution"
        executable = distribution / "bin" / "claude.exe"
        native_root = (
            distribution / "node_modules" / "@anthropic-ai" / "claude-code-test-posix"
        )
        native_executable = native_root / "claude"
        executable.parent.mkdir(parents=True)
        native_root.mkdir(parents=True)
        executable.write_text(FAKE_CLAUDE, encoding="utf-8")
        executable.chmod(0o755)
        shutil.copy2(executable, native_executable)
        native_executable.chmod(0o755)
        (distribution / "package.json").write_text(
            json.dumps(
                {
                    "name": "@anthropic-ai/claude-code",
                    "version": CLAUDE_CODE_DISTRIBUTION_VERSION,
                    "bin": {"claude": "bin/claude.exe"},
                    "optionalDependencies": {
                        self.native_package: CLAUDE_CODE_DISTRIBUTION_VERSION
                    },
                }
            ),
            encoding="utf-8",
        )
        (native_root / "package.json").write_text(
            json.dumps(
                {
                    "name": self.native_package,
                    "version": CLAUDE_CODE_DISTRIBUTION_VERSION,
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        return distribution, digest

    def _backend(
        self,
        root: Path,
        *,
        timeout_seconds: float = 5.0,
        max_output_bytes: int = 1024 * 1024,
        require_team_evidence: bool = True,
        ignored_fixture: bool = False,
    ) -> ClaudeCodeAgentTeamsDistributionBackend:
        distribution, digest = self._distribution(root)
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "README.md").write_text("base fixture\n", encoding="utf-8")
        if ignored_fixture:
            (workspace / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
            (workspace / "existing.ignored").write_text(
                "ignored baseline\n", encoding="utf-8"
            )
        nested_git = workspace / "vendor" / "pkg" / ".git"
        nested_git.mkdir(parents=True)
        (nested_git / "commondir").write_text(
            "/external/shared/repository\n", encoding="utf-8"
        )
        with patch.dict(
            claude_code_source._KNOWN_EXECUTABLE_SHA256,
            {self.native_package: digest},
        ):
            return ClaudeCodeAgentTeamsDistributionBackend(
                distribution_root=distribution,
                workspace=workspace,
                model="claude-opus-5",
                max_budget_usd=2.5,
                api_key_env="TEST_ANTHROPIC_API_KEY",
                expected_executable_sha256=digest,
                native_package_name=self.native_package,
                max_turns=12,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                allow_sensitive_environment=True,
                require_team_evidence=require_team_evidence,
            )

    async def test_executes_pinned_distribution_with_native_team_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = self._backend(root)
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/claude-code-team/lead",
                        role="claude_code_agent_teams_session",
                        prompt="Use the team and solve this task",
                    )
                )

            answer = json.loads(response.text)
            self.assertNotEqual(answer["cwd"], str((root / "workspace").resolve()))
            self.assertEqual(answer["teams"], "1")
            self.assertTrue(answer["credential_present"])
            self.assertFalse(answer["bare"])
            self.assertEqual(answer["setting_sources"], "")
            self.assertEqual(answer["teammate_mode"], "in-process")
            self.assertEqual(answer["max_budget"], "2.5")
            self.assertRegex(answer["git_head"], r"^[0-9a-f]{40}$")
            self.assertFalse(answer["nested_git_exists"])
            self.assertFalse(Path(answer["cwd"]).exists())
            self.assertFalse(Path(answer["home"]).exists())
            self.assertFalse(Path(answer["config"]).exists())
            self.assertFalse((root / "workspace" / "team-output.txt").exists())
            self.assertEqual(response.usage.input_tokens, 140)
            self.assertEqual(response.usage.output_tokens, 20)
            self.assertEqual(response.usage.cost_usd, 0.125)
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            evidence = response.raw["team_evidence"]
            self.assertEqual(evidence["named_teammate_count"], 2)
            self.assertEqual(evidence["removed_team_create_calls"], 0)
            self.assertEqual(evidence["removed_team_delete_calls"], 0)
            self.assertEqual(evidence["deprecated_team_name_inputs"], 0)
            self.assertEqual(evidence["session_derived_team_name"], "session-00000000")
            self.assertEqual(evidence["live_team_config_snapshot_count"], 1)
            self.assertTrue(evidence["member_roster_matches_agent_calls"])
            self.assertTrue(evidence["session_team_config_removed_after_exit"])
            self.assertTrue(evidence["session_task_directory_present"])
            self.assertTrue(evidence["native_team_observed"])
            self.assertEqual(evidence["send_message_calls"], 1)
            self.assertEqual(evidence["persisted_task_file_count"], 1)
            patch_payload = response.raw["workspace"]["patch"]
            self.assertIsInstance(patch_payload, SWEPatchPayload)
            self.assertIn(b"team-output.txt", patch_payload.content)
            self.assertIn(b"edited disposable copy", patch_payload.content)
            provenance = backend.provenance()
            self.assertEqual(
                provenance["artifact_kind"],
                "official_binary_distribution_not_public_source",
            )
            self.assertFalse(provenance["public_repository_is_runtime_source"])
            self.assertTrue(provenance["official_distribution_verified"])
            self.assertFalse(provenance["server_managed_policy_observable"])
            self.assertFalse(provenance["bit_reproducible_runtime_verified"])
            self.assertNotIn("scoped-secret", json.dumps(provenance))

    async def test_patch_export_failure_preserves_terminal_usage(self) -> None:
        reported: list[Usage] = []
        export_error = ProviderError(
            "synthetic patch export failure", raw={"stage": "patch-export"}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = self._backend(root)
            with (
                patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}),
                patch.object(
                    backend,
                    "_export_disposable_patch",
                    new=AsyncMock(side_effect=export_error),
                ),
            ):
                with self.assertRaisesRegex(
                    ProviderError, "synthetic patch export failure"
                ) as caught:
                    await backend.complete(
                        ModelRequest(
                            agent_id="lead",
                            role="session",
                            prompt="Use the team and solve this task",
                            usage_reporter=reported.append,
                        )
                    )

        self.assertEqual(len(reported), 1)
        self.assertIs(caught.exception.usage, reported[0])
        assert caught.exception.usage is not None
        self.assertEqual(caught.exception.usage.input_tokens, 140)
        self.assertEqual(caught.exception.usage.output_tokens, 20)
        self.assertEqual(caught.exception.usage.cost_usd, 0.125)
        self.assertFalse(caught.exception.usage.cost_known)
        self.assertFalse(caught.exception.usage.complete)
        self.assertEqual(caught.exception.raw, {"stage": "patch-export"})

    async def test_rejects_removed_legacy_team_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = self._backend(root)
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(
                    ProviderError, "without observable native Agent Teams evidence"
                ) as caught:
                    await backend.complete(
                        ModelRequest(
                            agent_id="lead",
                            role="session",
                            prompt="LEGACY_TEAM",
                        )
                    )

        evidence = caught.exception.raw["team_evidence"]
        self.assertEqual(evidence["deprecated_team_name_inputs"], 1)
        self.assertFalse(evidence["native_team_observed"])

    async def test_exports_preexisting_and_new_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = self._backend(root, ignored_fixture=True)
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/claude-code-team/lead",
                        role="claude_code_agent_teams_session",
                        prompt="IGNORED_FILES",
                    )
                )

            payload = response.raw["workspace"]["patch"]
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
                (root / "workspace" / "existing.ignored").read_text(encoding="utf-8"),
                "ignored baseline\n",
            )
            self.assertFalse((root / "workspace" / "new.ignored").exists())

    async def test_rejects_client_tools_missing_team_and_mutated_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = self._backend(root)
            with self.assertRaisesRegex(ProviderError, "owns its system prompt"):
                await backend.complete(
                    ModelRequest(
                        agent_id="lead",
                        role="session",
                        prompt="task",
                        tools=(ToolDefinition(name="shell"),),
                    )
                )
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "without observable"):
                    await backend.complete(
                        ModelRequest(
                            agent_id="lead",
                            role="session",
                            prompt="NO_TEAM",
                        )
                    )
            native_manifest = backend.native_package_root / "package.json"
            native_manifest.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "native package name"):
                    await backend.complete(
                        ModelRequest(agent_id="lead", role="session", prompt="task")
                    )

    async def test_output_limit_timeout_and_cancellation_stop_process_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            flood_root = Path(directory) / "flood"
            flood_root.mkdir()
            flood = self._backend(flood_root, max_output_bytes=1024)
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "output limit"):
                    await flood.complete(
                        ModelRequest(agent_id="lead", role="session", prompt="FLOOD")
                    )

            timeout_root = Path(directory) / "timeout"
            timeout_root.mkdir()
            timed = self._backend(timeout_root, timeout_seconds=0.1)
            timeout_pid = Path(directory) / "timeout-child.pid"
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                with self.assertRaisesRegex(ProviderError, "timed out"):
                    await timed.complete(
                        ModelRequest(
                            agent_id="lead",
                            role="session",
                            prompt=f"HANG PIDFILE:{timeout_pid}",
                        )
                    )

            cancel_root = Path(directory) / "cancel"
            cancel_root.mkdir()
            cancellable = self._backend(cancel_root, timeout_seconds=10)
            cancel_pid = Path(directory) / "cancel-child.pid"
            with patch.dict(os.environ, {"TEST_ANTHROPIC_API_KEY": "scoped-secret"}):
                task = asyncio.create_task(
                    cancellable.complete(
                        ModelRequest(
                            agent_id="lead",
                            role="session",
                            prompt=f"HANG PIDFILE:{cancel_pid}",
                        )
                    )
                )
                child_pid: int | None = None
                try:
                    for _ in range(500):
                        try:
                            raw_pid = cancel_pid.read_text(encoding="ascii").strip()
                            if raw_pid:
                                child_pid = int(raw_pid)
                                break
                        except (FileNotFoundError, ValueError):
                            # File creation and the child write are not atomic.
                            pass
                        await asyncio.sleep(0.01)
                    self.assertIsNotNone(child_pid)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                assert child_pid is not None
                for _ in range(100):
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("cancelled Claude Code child survived process-group stop")

    def test_requires_acknowledgement_and_rejects_workspace_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution, digest = self._distribution(root)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "outside-link").symlink_to(root)
            common = {
                "distribution_root": distribution,
                "workspace": workspace,
                "model": "claude-opus-5",
                "max_budget_usd": 1.0,
                "expected_executable_sha256": digest,
                "native_package_name": self.native_package,
            }
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                ClaudeCodeAgentTeamsDistributionBackend(**common)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                ClaudeCodeAgentTeamsDistributionBackend(
                    **common, allow_sensitive_environment=True
                )

            clean_workspace = root / "clean-workspace"
            clean_workspace.mkdir()
            linked_workspace = root / "linked-workspace"
            linked_workspace.mkdir()
            (linked_workspace / ".git").write_text(
                "gitdir: /external/shared/repository\n", encoding="utf-8"
            )
            # Inherited Git administration is intentionally ignored and replaced
            # by a fresh standalone repository in the disposable trial.
            ClaudeCodeAgentTeamsDistributionBackend(
                **{
                    **common,
                    "workspace": linked_workspace,
                    "allow_sensitive_environment": True,
                }
            )
            ambiguous_workspace = root / "ambiguous-workspace"
            ambiguous_workspace.mkdir()
            (ambiguous_workspace / ".GIT").mkdir()
            with self.assertRaisesRegex(ValueError, "case-variant Git metadata"):
                ClaudeCodeAgentTeamsDistributionBackend(
                    **{
                        **common,
                        "workspace": ambiguous_workspace,
                        "allow_sensitive_environment": True,
                    }
                )
            for runtime_override in ("BASH_ENV", "GIT_DIR", "LD_PRELOAD"):
                with self.assertRaisesRegex(ValueError, "runtime-control"):
                    ClaudeCodeAgentTeamsDistributionBackend(
                        **{
                            **common,
                            "workspace": clean_workspace,
                            "allow_sensitive_environment": True,
                            "pass_env": (runtime_override,),
                        }
                    )

    def test_known_platform_digest_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution, digest = self._distribution(root)
            workspace = root / "workspace"
            workspace.mkdir()
            with patch.dict(
                claude_code_source._KNOWN_EXECUTABLE_SHA256,
                {self.native_package: digest},
            ):
                with self.assertRaisesRegex(ValueError, "cannot override"):
                    ClaudeCodeAgentTeamsDistributionBackend(
                        distribution_root=distribution,
                        workspace=workspace,
                        model="claude-opus-5",
                        max_budget_usd=1.0,
                        native_package_name=self.native_package,
                        expected_executable_sha256="0" * 64,
                        allow_sensitive_environment=True,
                    )

    def test_fails_closed_when_endpoint_managed_settings_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution, digest = self._distribution(root)
            workspace = root / "workspace"
            workspace.mkdir()
            managed = root / "managed-settings.json"
            managed.write_text("{}\n", encoding="utf-8")
            with patch.object(
                claude_code_source, "_CLAUDE_CODE_MANAGED_PATHS", (managed,)
            ):
                with self.assertRaisesRegex(ValueError, "cannot be disabled"):
                    ClaudeCodeAgentTeamsDistributionBackend(
                        distribution_root=distribution,
                        workspace=workspace,
                        model="claude-opus-5",
                        max_budget_usd=1.0,
                        native_package_name=self.native_package,
                        expected_executable_sha256=digest,
                        allow_sensitive_environment=True,
                    )


class ClaudeCodeDistributionHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_ledger_and_exact_team_size_validation(self) -> None:
        backend = ScriptedBackend(
            {
                "/claude-code-team/lead": [
                    ModelResponse(
                        text="synthesis",
                        usage=Usage(cost_known=False, complete=False),
                        raw={
                            "team_evidence": {
                                "native_team_observed": True,
                                "named_teammate_count": 2,
                                "agent_tool_calls": 2,
                                "send_message_calls": 1,
                                "persisted_task_file_count": 2,
                                "removed_team_create_calls": 0,
                                "removed_team_delete_calls": 0,
                                "deprecated_team_name_inputs": 0,
                                "session_derived_team_name": "session-00000000",
                                "live_team_config_snapshot_count": 1,
                                "session_team_config_removed_after_exit": True,
                            },
                            "distribution": {
                                "version": "2.1.226",
                                "official_distribution_verified": True,
                            },
                        },
                    )
                ]
            }
        )
        result = await ClaudeCodeAgentTeamsDistributionHarness(team_size=2).run(
            Task("one", "Solve this"),
            backend,
            BudgetLimits(max_model_calls=1),
        )
        self.assertEqual(result.answer, "synthesis")
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.metadata["shared_budget_ledger"])
        self.assertTrue(result.metadata["official_binary_distribution_executed"])
        self.assertFalse(result.metadata["upstream_source_executed"])
        self.assertFalse(result.metadata["flagship_system_card_parity_claimed"])

        mismatch = ScriptedBackend(
            {
                "/claude-code-team/lead": [
                    ModelResponse(
                        text="partial",
                        usage=Usage(cost_known=False, complete=False),
                        raw={
                            "team_evidence": {
                                "native_team_observed": True,
                                "named_teammate_count": 1,
                            },
                            "distribution": {"official_distribution_verified": True},
                        },
                    )
                ]
            }
        )
        with self.assertRaises(RunFailed) as caught:
            await ClaudeCodeAgentTeamsDistributionHarness(team_size=2).run(
                Task("two", "Solve this"),
                mismatch,
                BudgetLimits(max_model_calls=1),
            )
        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)


if __name__ == "__main__":
    unittest.main()
