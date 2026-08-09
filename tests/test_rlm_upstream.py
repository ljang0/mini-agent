from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.cli import _build_backend, _validate_compatibility, build_parser
from scaffoldlab.external import (
    RLMUpstreamBackend,
    _RLM_RELEASE_REVISION,
    _rlm_usage_from_summary,
)
from scaffoldlab.harnesses import RLMUpstreamHarness
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


FAKE_RLM = r"""import json
import os


class UsageSummary:
    def to_dict(self):
        return {
            "model_usage_summaries": {
                "root-model": {
                    "total_calls": 2,
                    "total_input_tokens": 17,
                    "total_output_tokens": 5,
                    "total_cost": 0.25,
                },
                "sub-model": {
                    "total_calls": 1,
                    "total_input_tokens": 3,
                    "total_output_tokens": 2,
                },
            },
            "total_cost": 0.25,
        }


class Completion:
    def __init__(self, response):
        self.response = response
        self.root_model = "root-model"
        self.execution_time = 1.25
        self.usage_summary = UsageSummary()
        self.error = None


class RLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def completion(self, context, root_prompt=None):
        return Completion(json.dumps({
            "context": context,
            "root_prompt": root_prompt,
            "provider": self.kwargs["backend"],
            "model": self.kwargs["backend_kwargs"]["model_name"],
            "base_url": self.kwargs["backend_kwargs"]["base_url"],
            "environment": self.kwargs["environment"],
            "image": self.kwargs["environment_kwargs"]["image"],
            "max_depth": self.kwargs["max_depth"],
            "passed_secret": os.environ.get("TEST_RLM_API_KEY"),
            "custom_tools": self.kwargs["custom_tools"],
        }, sort_keys=True))

    def close(self):
        pass
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


class RLMUpstreamTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, str]:
        checkout = root / "rlm"
        package = checkout / "rlm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(FAKE_RLM, encoding="utf-8")
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", "rlm/__init__.py", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned RLM", cwd=checkout)
        return checkout, _git("rev-parse", "HEAD", cwd=checkout)

    def test_usage_summary_is_parsed_as_recursive_lower_bound(self) -> None:
        usage, calls = _rlm_usage_from_summary(
            {
                "model_usage_summaries": {
                    "root": {
                        "total_calls": 2,
                        "total_input_tokens": 10,
                        "total_output_tokens": 4,
                        "total_cost": 0.2,
                    },
                    "child": {
                        "total_calls": 3,
                        "total_input_tokens": 7,
                        "total_output_tokens": 5,
                    },
                },
                "total_cost": 0.2,
            }
        )
        self.assertEqual(calls, 5)
        self.assertEqual((usage.input_tokens, usage.output_tokens), (17, 9))
        self.assertEqual(usage.cost_usd, 0.2)
        self.assertFalse(usage.cost_known)
        self.assertFalse(usage.complete)

    async def test_runs_exact_clean_checkout_through_json_stdin_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            backend = RLMUpstreamBackend(
                checkout=checkout,
                provider="openai",
                model="model-pinned",
                python_executable=sys.executable,
                backend_kwargs={"base_url": "https://example.invalid/v1"},
                environment_kwargs={"image": "python:3.11-slim@sha256:fake"},
                max_depth=2,
                max_iterations=7,
                max_tokens=1000,
                pass_env=("TEST_RLM_API_KEY",),
                allow_sensitive_environment=True,
                expected_checkout_revision=revision,
            )
            with patch.dict(os.environ, {"TEST_RLM_API_KEY": "scoped-secret"}):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/rlm-upstream/root",
                        role="rlm_upstream_session",
                        prompt="large external context",
                        metadata={"root_prompt": "answer this question"},
                    )
                )

            answer = json.loads(response.text)
            self.assertEqual(answer["context"], "large external context")
            self.assertEqual(answer["root_prompt"], "answer this question")
            self.assertEqual(answer["provider"], "openai")
            self.assertEqual(answer["model"], "model-pinned")
            self.assertEqual(answer["environment"], "docker")
            self.assertEqual(answer["max_depth"], 2)
            self.assertEqual(answer["passed_secret"], "scoped-secret")
            self.assertIsNone(answer["custom_tools"])
            self.assertEqual(response.usage.input_tokens, 20)
            self.assertEqual(response.usage.output_tokens, 7)
            self.assertEqual(response.usage.cost_usd, 0.25)
            self.assertFalse(response.usage.cost_known)
            self.assertFalse(response.usage.complete)
            self.assertEqual(response.raw["underlying_model_calls"], 3)
            self.assertEqual(_git("status", "--porcelain", cwd=checkout), "")

            provenance = backend.provenance()
            self.assertEqual(provenance["environment"], "docker")
            self.assertEqual(provenance["adapter_default_environment"], "docker")
            self.assertEqual(
                provenance["upstream_library_default_environment"], "local"
            )
            self.assertTrue(provenance["recursive_child_rlm_enabled"])
            self.assertEqual(provenance["expected_checkout_revision"], revision)
            self.assertIn("lower bound", provenance["usage_scope"])
            self.assertNotIn("scoped-secret", json.dumps(provenance))
            self.assertEqual(
                provenance["audited_release"]["revision"],
                _RLM_RELEASE_REVISION,
            )

    async def test_harness_uses_one_outer_call_and_withholds_domain_tools(self) -> None:
        backend = ScriptedBackend(
            {
                "/rlm-upstream/root": [
                    ModelResponse(
                        text="answer",
                        usage=Usage(
                            input_tokens=12,
                            output_tokens=3,
                            cost_usd=0.4,
                            cost_known=False,
                            complete=False,
                        ),
                        raw={
                            "underlying_model_calls": 4,
                            "environment": "docker",
                            "max_depth": 1,
                        },
                    )
                ]
            }
        )
        result = await RLMUpstreamHarness().run(
            Task("task", "question", context="external context"),
            backend,
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.metadata["underlying_model_calls"], 4)
        self.assertTrue(result.metadata["underlying_model_calls_are_lower_bound"])
        self.assertFalse(result.metadata["whole_tree_usage_reported_by_upstream"])
        self.assertEqual(result.metadata["adapter_default_environment"], "docker")
        self.assertEqual(
            result.metadata["upstream_library_default_environment"], "local"
        )
        self.assertFalse(result.metadata["recursive_child_rlm_enabled"])
        self.assertFalse(result.metadata["swe_tool_parity_claimed"])
        request = backend.requests[0]
        self.assertEqual(request.prompt, "external context")
        self.assertEqual(request.metadata["root_prompt"], "question")
        self.assertFalse(request.metadata["task_tools"])
        self.assertEqual(request.tools, ())

    def test_rejects_wrong_revision_dirty_tree_and_unacknowledged_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                RLMUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    expected_checkout_revision="0" * 40,
                )
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                RLMUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    pass_env=("OPENAI_API_KEY",),
                    expected_checkout_revision=revision,
                )
            (checkout / "untracked.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean"):
                RLMUpstreamBackend(
                    checkout=checkout,
                    provider="openai",
                    model="model",
                    expected_checkout_revision=revision,
                )

    async def test_rejects_client_tools_and_bounds_json_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout, revision = self._checkout(Path(directory))
            backend = RLMUpstreamBackend(
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
                        agent_id="/rlm-upstream/root",
                        role="rlm_upstream_session",
                        prompt="x" * 2048,
                    )
                )
            with self.assertRaisesRegex(ProviderError, "client tool continuation"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/rlm-upstream/root",
                        role="rlm_upstream_session",
                        prompt="context",
                        tools=(ToolDefinition(name="shell"),),
                    )
                )

    def test_cli_builds_explicit_pinned_backend_and_gates_harness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "run",
                    "--tasks",
                    str(root / "tasks.jsonl"),
                    "--config",
                    str(root / "config.json"),
                    "--provider",
                    "rlm-upstream",
                    "--output",
                    str(root / "results"),
                    "--rlm-checkout",
                    str(checkout),
                    "--rlm-provider",
                    "anthropic",
                    "--rlm-model",
                    "model-pinned",
                    "--rlm-python-executable",
                    sys.executable,
                    "--rlm-environment",
                    "docker",
                    "--rlm-backend-json",
                    '{"base_url":"https://example.invalid"}',
                    "--rlm-environment-json",
                    '{"image":"python:3.11-slim@sha256:fake"}',
                    "--rlm-expected-checkout-revision",
                    revision,
                ]
            )
            backend = _build_backend(args, {})
            self.assertIsInstance(backend, RLMUpstreamBackend)
            self.assertEqual(backend.provider, "anthropic")
            self.assertEqual(backend.model, "model-pinned")
            self.assertEqual(backend.environment, "docker")
            self.assertEqual(
                backend.backend_kwargs["base_url"],
                "https://example.invalid",
            )

            warnings = _validate_compatibility(
                [RLMUpstreamHarness()],
                "rlm-upstream",
                root / "tasks.jsonl",
                [Task("task", "question")],
            )
            self.assertTrue(any("lower bounds" in warning for warning in warnings))
            with self.assertRaisesRegex(ValueError, "requires --provider"):
                _validate_compatibility(
                    [RLMUpstreamHarness()],
                    "openai-responses",
                    root / "tasks.jsonl",
                    [Task("task", "question")],
                )


if __name__ == "__main__":
    unittest.main()
