from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.browser_use_external import (
    BROWSER_USE_RELEASE_REVISION,
    BROWSER_USE_RELEASE_VERSION,
    BrowserUseUpstreamBackend,
    _browser_use_usage,
)
from scaffoldlab.harnesses.browser_use_upstream import BrowserUseUpstreamHarness
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


FAKE_BROWSER_USE = r"""import asyncio
import json
import os
from pathlib import Path


class ChatOpenAI:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class Browser:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class UsageSummary:
    def model_dump(self, mode=None):
        return {
            "total_prompt_tokens": 17,
            "total_prompt_cost": 0.0,
            "total_prompt_cached_tokens": 3,
            "total_prompt_cached_cost": 0.0,
            "total_prompt_cache_creation_tokens": 2,
            "total_prompt_cache_creation_cost": 0.0,
            "total_completion_tokens": 5,
            "total_completion_cost": 0.0,
            "total_tokens": 22,
            "total_cost": 0.0,
            "entry_count": 2,
            "by_model": {
                "fake-model": {
                    "model": "fake-model",
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                    "total_tokens": 22,
                    "cost": 0.0,
                    "invocations": 2,
                    "average_tokens_per_invocation": 11.0,
                }
            },
        }


class TokenCostService:
    async def get_usage_summary(self):
        return UsageSummary()


class History:
    def __init__(self, answer):
        self.answer = answer
        self.usage = UsageSummary()

    def final_result(self):
        return self.answer

    def is_done(self):
        return True

    def is_successful(self):
        return True

    def number_of_steps(self):
        return 3


class Agent:
    instances = 0

    def __init__(self, task, llm, browser, **kwargs):
        type(self).instances += 1
        self.task = task
        self.llm = llm
        self.browser = browser
        self.kwargs = kwargs
        self.token_cost_service = TokenCostService()

    async def run(self, max_steps):
        if self.kwargs.get("dirty_checkout_before_delay"):
            Path(__file__).with_name("runtime-mutation.txt").write_text(
                "dirty", encoding="utf-8"
            )
        delay = self.kwargs.get("synthetic_delay_seconds", 0)
        if delay:
            await asyncio.sleep(delay)
        if self.kwargs.get("dirty_checkout"):
            Path(__file__).with_name("runtime-mutation.txt").write_text(
                "dirty", encoding="utf-8"
            )
        if self.kwargs.get("synthetic_failure"):
            raise RuntimeError("synthetic upstream failure")
        return History(json.dumps({
            "agent_instances": type(self).instances,
            "task": self.task,
            "model": self.llm.model,
            "llm_kwargs": self.llm.kwargs,
            "browser_kwargs": self.browser.kwargs,
            "agent_kwargs": self.kwargs,
            "max_steps": max_steps,
            "passed_secret": os.environ.get("TEST_BROWSER_USE_API_KEY"),
        }, sort_keys=True))
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
class BrowserUseUpstreamTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, str]:
        checkout = root / "browser-use"
        package = checkout / "browser_use"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(FAKE_BROWSER_USE, encoding="utf-8")
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", "browser_use/__init__.py", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned Browser-Use", cwd=checkout)
        return checkout, _git("rev-parse", "HEAD", cwd=checkout)

    def test_usage_is_projected_as_observed_lower_bound(self) -> None:
        unknown, unknown_calls = _browser_use_usage(None)
        self.assertEqual(unknown_calls, 0)
        self.assertFalse(unknown.cost_known)
        self.assertFalse(unknown.complete)

        usage, calls = _browser_use_usage(
            {
                "total_prompt_tokens": 17,
                "total_prompt_cached_tokens": 3,
                "total_prompt_cache_creation_tokens": 2,
                "total_completion_tokens": 5,
                "total_cost": 0.25,
                "entry_count": 2,
            }
        )
        self.assertEqual(calls, 2)
        self.assertEqual((usage.input_tokens, usage.output_tokens), (17, 5))
        self.assertEqual(usage.cache_read_input_tokens, 3)
        self.assertEqual(usage.cache_write_input_tokens, 2)
        self.assertEqual(usage.cost_usd, 0.25)
        self.assertFalse(usage.cost_known)
        self.assertFalse(usage.complete)

        cache_heavy, _ = _browser_use_usage(
            {
                "total_prompt_tokens": 4,
                "total_prompt_cached_tokens": 3,
                "total_prompt_cache_creation_tokens": 5,
                "total_completion_tokens": 1,
                "total_cost": 0.0,
                "entry_count": 1,
            }
        )
        self.assertEqual(cache_heavy.input_tokens, 8)
        self.assertEqual(cache_heavy.cache_read_input_tokens, 3)
        self.assertEqual(cache_heavy.cache_write_input_tokens, 5)

    async def test_runs_clean_checkout_via_native_agent_and_browser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="fake-model",
                python_executable=sys.executable,
                llm_kwargs={"temperature": 0.1},
                agent_kwargs={"use_vision": False},
                max_steps=7,
                pass_env=("TEST_BROWSER_USE_API_KEY",),
                allow_sensitive_environment=True,
                expected_checkout_revision=revision,
            )
            with patch.dict(os.environ, {"TEST_BROWSER_USE_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="Research the source",
                        system="Preserve citations.",
                        metadata={"task_id": "browser-task"},
                    )
                )

            answer = json.loads(response.text)
            self.assertEqual(answer["agent_instances"], 1)
            self.assertEqual(answer["task"], "Research the source")
            self.assertEqual(answer["model"], "fake-model")
            self.assertEqual(answer["llm_kwargs"], {"temperature": 0.1})
            self.assertEqual(
                answer["browser_kwargs"], {"headless": True, "keep_alive": False}
            )
            self.assertFalse(answer["agent_kwargs"]["calculate_cost"])
            self.assertFalse(answer["agent_kwargs"]["enable_signal_handler"])
            self.assertFalse(answer["agent_kwargs"]["use_vision"])
            self.assertEqual(
                answer["agent_kwargs"]["extend_system_message"],
                "Preserve citations.",
            )
            self.assertEqual(answer["agent_kwargs"]["task_id"], "browser-task")
            self.assertEqual(answer["max_steps"], 7)
            self.assertEqual(answer["passed_secret"], "scoped-secret")
            self.assertEqual(
                (response.usage.input_tokens, response.usage.output_tokens), (17, 5)
            )
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            self.assertEqual(response.raw["underlying_model_calls"], 2)
            self.assertTrue(response.raw["underlying_model_calls_observed"])
            self.assertTrue(response.raw["usage_is_lower_bound"])
            source_files = response.raw["result"]["source_files"]
            self.assertTrue(
                all(
                    Path(path).is_relative_to(backend.runtime_checkout)
                    for path in source_files.values()
                ),
                source_files,
            )
            self.assertTrue(
                all(
                    not Path(path).is_relative_to(checkout.resolve())
                    for path in source_files.values()
                )
            )
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")

            provenance = backend.provenance()
            self.assertEqual(provenance["expected_checkout_revision"], revision)
            self.assertTrue(provenance["runtime_source_identity_verified"])
            self.assertEqual(
                provenance["source_execution_scope"],
                "private_git_archive_of_expected_revision",
            )
            self.assertFalse(provenance["caller_worktree_executed"])
            self.assertIsNotNone(provenance["source_archive_sha256"])
            self.assertIsNotNone(provenance["source_export_tree_sha256"])
            self.assertTrue(provenance["runtime_git"]["available"])
            self.assertTrue(provenance["upstream_agent_and_browser_instantiated"])
            self.assertFalse(provenance["flat_parallel_reimplementation"])
            self.assertFalse(provenance["whole_session_usage_verified"])
            self.assertNotIn("scoped-secret", json.dumps(provenance))
            self.assertEqual(
                provenance["audited_release"],
                {
                    "package": "browser-use",
                    "version": BROWSER_USE_RELEASE_VERSION,
                    "repository": "browser-use/browser-use",
                    "revision": BROWSER_USE_RELEASE_REVISION,
                    "requires_python": ">=3.11,<4.0",
                },
            )

    async def test_harness_is_one_outer_call_without_local_browser_tools(self) -> None:
        backend = ScriptedBackend(
            {
                "/browser-use/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(
                            input_tokens=12,
                            output_tokens=3,
                            cost_known=False,
                            complete=False,
                        ),
                        raw={
                            "underlying_model_calls": 4,
                            "underlying_model_calls_observed": True,
                            "result": {
                                "history": {
                                    "steps": 3,
                                    "is_done": True,
                                    "is_successful": True,
                                },
                                "llm_class": "ChatOpenAI",
                                "cost_tracking_enabled": False,
                            },
                        },
                    )
                ]
            }
        )
        result = await BrowserUseUpstreamHarness(system="Keep sources.").run(
            Task("browser-task", "question", context="reference context"),
            backend,
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.metadata["native_browser_use_agent"])
        self.assertTrue(result.metadata["native_browser_use_browser"])
        self.assertFalse(result.metadata["flat_parallel_reimplementation"])
        self.assertFalse(result.metadata["scaffoldlab_domain_tools_injected"])
        self.assertEqual(result.metadata["underlying_model_calls"], 4)
        self.assertTrue(result.metadata["underlying_model_calls_are_lower_bound"])
        self.assertEqual(result.metadata["history_steps"], 3)
        request = backend.requests[0]
        self.assertEqual(
            request.prompt,
            "question\n\n<context>\nreference context\n</context>",
        )
        self.assertEqual(request.system, "Keep sources.")
        self.assertEqual(request.metadata["task_id"], "browser-task")
        self.assertFalse(request.metadata["task_tools"])
        self.assertEqual(request.tools, ())

    def test_rejects_wrong_revision_dirty_tree_and_unacknowledged_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                BrowserUseUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    expected_checkout_revision="0" * 40,
                )
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                BrowserUseUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    pass_env=("OPENAI_API_KEY",),
                    expected_checkout_revision=revision,
                )
            for runtime_override in ("PYTHONPATH", "LD_PRELOAD", "BROWSER_USE_HOME"):
                with self.assertRaisesRegex(ValueError, "runtime-control"):
                    BrowserUseUpstreamBackend(
                        checkout=checkout,
                        provider="openai",
                        model="model",
                        pass_env=(runtime_override,),
                        allow_sensitive_environment=True,
                        expected_checkout_revision=revision,
                    )
            (checkout / "untracked.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean"):
                BrowserUseUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    expected_checkout_revision=revision,
                )

    def test_rejects_linked_source_checkout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            external_metadata = root / "shared-git-metadata"
            (checkout / ".git").rename(external_metadata)
            (checkout / ".git").symlink_to(external_metadata, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "standalone clone"):
                BrowserUseUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    expected_checkout_revision=revision,
                )

    async def test_rejects_client_tools_and_bounds_json_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                max_input_bytes=1024,
                expected_checkout_revision=revision,
            )
            with self.assertRaisesRegex(ProviderError, "JSON input exceeds"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="x" * 2048,
                    )
                )
            with self.assertRaisesRegex(ProviderError, "client tool continuation"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="task",
                        tools=(ToolDefinition(name="browser"),),
                    )
                )

    async def test_reports_partial_usage_on_upstream_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                agent_kwargs={"synthetic_failure": True},
                expected_checkout_revision=revision,
            )
            with self.assertRaisesRegex(ProviderError, "reported failure") as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="fail after model usage",
                    )
                )
            usage = raised.exception.usage
            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertEqual((usage.input_tokens, usage.output_tokens), (17, 5))
            self.assertFalse(usage.complete)

    async def test_fails_if_runtime_dirties_verified_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                agent_kwargs={"dirty_checkout": True},
                expected_checkout_revision=revision,
            )
            with self.assertRaisesRegex(
                ProviderError, "private source export changed"
            ) as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="mutate checkout",
                    )
                )
            usage = raised.exception.usage
            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertEqual((usage.input_tokens, usage.output_tokens), (17, 5))

    async def test_process_timeout_is_bounded_and_reports_unknown_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                agent_kwargs={"synthetic_delay_seconds": 5},
                process_timeout_seconds=0.1,
                expected_checkout_revision=revision,
            )
            with self.assertRaisesRegex(ProviderError, "timed out") as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="timeout",
                    )
                )
            usage = raised.exception.usage
            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertFalse(usage.cost_known)
            self.assertFalse(usage.complete)

    async def test_timeout_still_detects_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                agent_kwargs={
                    "dirty_checkout_before_delay": True,
                    "synthetic_delay_seconds": 5,
                },
                process_timeout_seconds=0.2,
                expected_checkout_revision=revision,
            )
            with self.assertRaisesRegex(
                ProviderError, "private source export changed"
            ) as raised:
                await backend.complete(
                    ModelRequest(
                        agent_id="/browser-use/root",
                        role="browser_use_upstream_session",
                        prompt="mutate then timeout",
                    )
                )
            self.assertIsNotNone(raised.exception.usage)

    async def test_private_archive_ignores_assume_unchanged_worktree_overlay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            source = checkout / "browser_use" / "__init__.py"
            source.write_text(
                "raise RuntimeError('caller worktree overlay executed')\n",
                encoding="utf-8",
            )
            _git(
                "update-index",
                "--assume-unchanged",
                "browser_use/__init__.py",
                cwd=checkout,
            )
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")

            backend = BrowserUseUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model",
                python_executable=sys.executable,
                expected_checkout_revision=revision,
            )
            response = await backend.complete(
                ModelRequest(
                    agent_id="/browser-use/root",
                    role="browser_use_upstream_session",
                    prompt="use the committed tree",
                )
            )

            self.assertEqual(
                json.loads(response.text)["task"], "use the committed tree"
            )
            self.assertFalse(backend.provenance()["caller_worktree_executed"])


if __name__ == "__main__":
    unittest.main()
