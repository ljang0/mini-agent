import asyncio
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, patch

from scaffoldlab.cli import _build_backend, _validate_compatibility, build_parser
from scaffoldlab.external import (
    GrokBuildJSONBackend,
    _usage_from_grok_result,
    _workspace_tree_sha256_async,
)
from scaffoldlab.harnesses import (
    GrokBuildHarness,
    SingleAgentHarness,
    XAIHostedMultiAgentHarness,
)
from scaffoldlab.providers import ProviderError, TokenPricing, XAIResponsesBackend
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelRequest, ModelResponse, Task, Usage


class _FakeGrokProcess:
    def __init__(
        self,
        stdout: bytes,
        *,
        returncode: int = 0,
        on_communicate: Callable[[], None] | None = None,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.pid = 424242
        self.on_communicate = on_communicate

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.on_communicate is not None:
            self.on_communicate()
        return self.stdout, b"diagnostic stderr"

    async def wait(self) -> int:
        return self.returncode


def _success_result() -> dict[str, Any]:
    return {
        "text": "final answer",
        "stopReason": "end_turn",
        "sessionId": "session-1",
        "requestId": "request-1",
        "num_turns": 3,
        "usage": {
            "input_tokens": 7,
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 13,
            "output_tokens": 17,
            "reasoning_tokens": 5,
            "total_tokens": 48,
        },
        "modelUsage": {
            "grok-build": {
                "inputTokens": 7,
                "cacheReadInputTokens": 11,
                "outputTokens": 17,
                "modelCalls": 3,
            }
        },
        "total_cost_usd_ticks": 250_000_000,
    }


class GrokUsageTests(unittest.TestCase):
    def test_usage_includes_cache_once_and_does_not_add_reasoning(self) -> None:
        usage = _usage_from_grok_result(_success_result())

        self.assertEqual(usage.input_tokens, 31)
        self.assertEqual(usage.cache_read_input_tokens, 11)
        self.assertEqual(usage.cache_write_input_tokens, 13)
        self.assertEqual(usage.output_tokens, 17)
        self.assertEqual(usage.cost_usd, 0.025)
        self.assertTrue(usage.cost_known)
        self.assertTrue(usage.complete)

    def test_partial_and_incomplete_cost_is_a_lower_bound(self) -> None:
        partial = {**_success_result(), "cost_is_partial": True}
        partial_usage = _usage_from_grok_result(partial)
        self.assertEqual(partial_usage.cost_usd, 0.025)
        self.assertFalse(partial_usage.cost_known)
        self.assertTrue(partial_usage.complete)

        incomplete = {**_success_result(), "usage_is_incomplete": True}
        incomplete_usage = _usage_from_grok_result(incomplete)
        self.assertEqual(incomplete_usage.cost_usd, 0.025)
        self.assertFalse(incomplete_usage.cost_known)
        self.assertFalse(incomplete_usage.complete)

        missing = _usage_from_grok_result({"text": "answer"})
        self.assertFalse(missing.cost_known)
        self.assertFalse(missing.complete)

    def test_malformed_token_count_is_rejected(self) -> None:
        malformed = _success_result()
        malformed["usage"] = {"input_tokens": True}
        with self.assertRaises(ProviderError):
            _usage_from_grok_result(malformed)
        malformed_cost = _success_result()
        malformed_cost["total_cost_usd_ticks"] = True
        malformed_cost["total_cost_usd"] = 0.25
        with self.assertRaises(ProviderError):
            _usage_from_grok_result(malformed_cost)

    def test_missing_required_token_fields_are_rejected(self) -> None:
        with self.assertRaises(ProviderError):
            _usage_from_grok_result({"usage": {}, "total_cost_usd_ticks": 0})


class GrokBackendTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_defaults_to_audited_grok_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            args = build_parser().parse_args(
                [
                    "run",
                    "--tasks",
                    str(root / "tasks.jsonl"),
                    "--config",
                    str(root / "config.json"),
                    "--provider",
                    "grok-build",
                    "--model",
                    "grok-build-pinned",
                    "--output",
                    str(root / "results"),
                    "--grok-cwd",
                    str(workspace),
                ]
            )
            backend = _build_backend(args, {})
        self.assertEqual(backend.expected_version, "1.0.0")

    async def test_workspace_hash_limits_and_timeout_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").write_bytes(b"")
            (root / "two").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "filesystem entries"):
                await _workspace_tree_sha256_async(
                    root,
                    max_entries=1,
                    max_bytes=1024,
                    timeout_seconds=1,
                )

        worker_started = threading.Event()
        worker_stopped = threading.Event()

        def slow_hash(*args: object, **kwargs: object) -> str:
            del args
            cancel_event = kwargs["cancel_event"]
            assert isinstance(cancel_event, threading.Event)
            worker_started.set()
            cancel_event.wait(timeout=1)
            worker_stopped.set()
            return "late-result"

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("scaffoldlab.external._workspace_tree_sha256", new=slow_hash),
        ):
            with self.assertRaisesRegex(ValueError, "exceeded"):
                await _workspace_tree_sha256_async(
                    Path(directory),
                    max_entries=10,
                    max_bytes=1024,
                    timeout_seconds=0.01,
                )
        self.assertTrue(worker_started.is_set())
        self.assertTrue(await asyncio.to_thread(worker_stopped.wait, 1))

        worker_started.clear()
        worker_stopped.clear()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("scaffoldlab.external._workspace_tree_sha256", new=slow_hash),
        ):
            hashing = asyncio.create_task(
                _workspace_tree_sha256_async(
                    Path(directory),
                    max_entries=10,
                    max_bytes=1024,
                    timeout_seconds=5,
                )
            )
            self.assertTrue(await asyncio.to_thread(worker_started.wait, 1))
            hashing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await hashing
        self.assertTrue(await asyncio.to_thread(worker_stopped.wait, 1))

    async def test_post_workspace_hash_failure_preserves_grok_usage(self) -> None:
        reported: list[Usage] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate_workspace() -> None:
                (root / "oversized.bin").write_bytes(b"xx")

            process = _FakeGrokProcess(
                json.dumps(_success_result()).encode(),
                on_communicate=mutate_workspace,
            )
            backend = GrokBuildJSONBackend(
                cwd=root,
                model="grok-build",
                workspace_hash_max_bytes=1,
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(
                            agent_id="/grok",
                            role="session",
                            prompt="p",
                            usage_reporter=reported.append,
                        )
                    )

        self.assertIn("after the session", str(caught.exception))
        self.assertEqual(caught.exception.usage.input_tokens, 31)
        self.assertEqual(caught.exception.usage.output_tokens, 17)
        self.assertEqual(caught.exception.usage.cost_usd, 0.025)
        self.assertFalse(caught.exception.usage.cost_known)
        self.assertFalse(caught.exception.usage.complete)
        self.assertEqual(reported, [caught.exception.usage])

    async def test_exact_safe_argv_and_ephemeral_files(self) -> None:
        captured: dict[str, Any] = {}

        async def create_process(*args: str, **kwargs: Any) -> _FakeGrokProcess:
            if args == ("grok-test", "version"):
                return _FakeGrokProcess(b"grok 1.0.0\n")
            captured["args"] = args
            captured["kwargs"] = kwargs
            prompt_path = Path(args[args.index("--prompt-file") + 1])

            def inspect_files() -> None:
                captured["prompt_path"] = prompt_path
                captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
                captured["prompt_mode"] = stat.S_IMODE(prompt_path.stat().st_mode)
                environment = kwargs["env"]
                captured["grok_home"] = Path(environment["GROK_HOME"])
                captured["grok_home_exists"] = captured["grok_home"].is_dir()

            return _FakeGrokProcess(
                json.dumps(_success_result()).encode("utf-8"),
                on_communicate=inspect_files,
            )

        terminate = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            resolved_cwd = Path(directory).resolve()
            backend = GrokBuildJSONBackend(
                cwd=resolved_cwd,
                executable="grok-test",
                model="grok-build-pinned",
                sandbox="strict",
                permission_mode="dontAsk",
                max_turns=9,
                pass_env=("XAI_API_KEY",),
                allow_rules=("Read(**)",),
                deny_rules=("Read(**/.env)",),
                expected_version="1.0.0",
            )
            with (
                patch.dict(os.environ, {"XAI_API_KEY": "secret-key"}),
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch("scaffoldlab.external._terminate_process_tree", new=terminate),
            ):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/grok",
                        role="session",
                        system="system rule",
                        prompt="secret prompt",
                    )
                )

        self.assertEqual(response.text, "final answer")
        args = captured["args"]
        kwargs = captured["kwargs"]
        self.assertNotIn("secret prompt", args)
        self.assertEqual(captured["prompt_text"], "system rule\n\nsecret prompt")
        self.assertEqual(captured["prompt_mode"], 0o600)
        self.assertFalse(captured["prompt_path"].exists())
        self.assertTrue(captured["grok_home_exists"])
        self.assertFalse(captured["grok_home"].exists())
        self.assertEqual(kwargs["cwd"], str(resolved_cwd))
        self.assertEqual(kwargs["stdin"], asyncio.subprocess.DEVNULL)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["XAI_API_KEY"], "secret-key")
        self.assertEqual(kwargs["env"]["GROK_DISABLE_AUTOUPDATER"], "1")
        for expected in (
            "--verbatim",
            "--no-auto-update",
            "--no-memory",
            "--sandbox",
            "strict",
            "--permission-mode",
            "dontAsk",
            "--max-turns",
            "9",
            "--model",
            "grok-build-pinned",
            "--allow",
            "Read(**)",
            "--deny",
            "Read(**/.env)",
        ):
            self.assertIn(expected, args)
        self.assertEqual(terminate.await_count, 2)
        provenance = backend.provenance()
        self.assertNotIn("secret-key", json.dumps(provenance))
        self.assertEqual(provenance["expected_version"], "1.0.0")
        self.assertTrue(provenance["version_verified"])
        self.assertIn("grok 1.0.0", provenance["observed_version_output"])
        self.assertEqual(provenance["timeout_seconds"], 1800.0)
        self.assertGreater(provenance["workspace_hash_limits"]["max_entries"], 0)
        self.assertGreater(provenance["workspace_hash_limits"]["max_content_bytes"], 0)
        self.assertGreater(provenance["workspace_hash_limits"]["timeout_seconds"], 0)

    async def test_version_pin_mismatch_fails_before_prompt(self) -> None:
        process = _FakeGrokProcess(b"grok 1.1.0\n")
        with tempfile.TemporaryDirectory() as directory:
            backend = GrokBuildJSONBackend(
                cwd=Path(directory),
                executable="grok-test",
                model="grok-build",
                expected_version="1.0.0",
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ) as create_process,
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

        self.assertIn("version mismatch", str(caught.exception))
        create_process.assert_awaited_once()

    async def test_prerelease_does_not_satisfy_release_version_pin(self) -> None:
        process = _FakeGrokProcess(b"grok 1.0.0-beta\n")
        with tempfile.TemporaryDirectory() as directory:
            backend = GrokBuildJSONBackend(
                cwd=Path(directory),
                executable="grok-test",
                model="grok-build",
                expected_version="1.0.0",
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.verify_version()
        self.assertIn("version mismatch", str(caught.exception))

    async def test_output_limit_terminates_session_and_fails_closed(self) -> None:
        process = _FakeGrokProcess(b"x" * 2048)
        terminate = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            backend = GrokBuildJSONBackend(
                cwd=Path(directory),
                executable="grok-test",
                model="grok-build",
                max_output_bytes=1024,
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=terminate,
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )
        self.assertIn("output limit", str(caught.exception))
        self.assertFalse(caught.exception.usage.complete)
        self.assertEqual(len(caught.exception.raw["stdout"]), 1024)
        terminate.assert_awaited_once_with(process)

    async def test_nonzero_exit_preserves_usage(self) -> None:
        result = {
            **_success_result(),
            "type": "error",
            "message": "session failed",
        }
        process = _FakeGrokProcess(json.dumps(result).encode(), returncode=2)

        with tempfile.TemporaryDirectory() as directory:
            backend = GrokBuildJSONBackend(cwd=Path(directory), model="grok-build")
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )

        self.assertIn("status 2", str(caught.exception))
        self.assertEqual(caught.exception.usage.input_tokens, 31)
        self.assertEqual(caught.exception.raw["result"]["message"], "session failed")
        self.assertIn("_scaffoldlab_workspace", caught.exception.raw["result"])

    async def test_only_end_turn_with_nonempty_text_succeeds(self) -> None:
        for mutation in (
            {"stopReason": "MaxTurns"},
            {"stopReason": "EndTurn"},
            {"stopReason": None},
            {"text": ""},
        ):
            result = {**_success_result(), **mutation}
            process = _FakeGrokProcess(json.dumps(result).encode())
            with tempfile.TemporaryDirectory() as directory:
                backend = GrokBuildJSONBackend(cwd=Path(directory), model="grok-build")
                with (
                    patch(
                        "scaffoldlab.external.asyncio.create_subprocess_exec",
                        new=AsyncMock(return_value=process),
                    ),
                    patch(
                        "scaffoldlab.external._terminate_process_tree",
                        new=AsyncMock(),
                    ),
                ):
                    with self.assertRaises(ProviderError):
                        await backend.complete(
                            ModelRequest(agent_id="/grok", role="session", prompt="p")
                        )

    async def test_timeout_terminates_process_group(self) -> None:
        class HangingProcess(_FakeGrokProcess):
            def __init__(self) -> None:
                super().__init__(b"", returncode=0)

            async def communicate(self) -> tuple[bytes, bytes]:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        process = HangingProcess()
        terminate = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            backend = GrokBuildJSONBackend(
                cwd=Path(directory), model="grok-build", timeout_seconds=0.01
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch("scaffoldlab.external._terminate_process_tree", new=terminate),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(agent_id="/grok", role="session", prompt="p")
                    )
        self.assertIn("timed out", str(caught.exception))
        terminate.assert_awaited_once_with(process)


class GrokHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_surfaces_observed_calls_and_usage_scope(self) -> None:
        response = ModelResponse(
            text="answer",
            usage=Usage(input_tokens=1, output_tokens=1, cost_known=False),
            raw=_success_result(),
        )
        result = await GrokBuildHarness().run(
            Task(task_id="task", prompt="question"),
            ScriptedBackend({"/grok-build/root": [response]}),
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(result.metadata["underlying_model_calls"], 3)
        self.assertTrue(result.metadata["underlying_model_calls_observed"])
        self.assertFalse(result.metadata["full_tree_usage_verified"])

    def test_provider_and_harness_compatibility_is_bidirectional(self) -> None:
        task = Task(task_id="task", prompt="question")
        with self.assertRaises(ValueError):
            _validate_compatibility(
                [GrokBuildHarness()], "openai-responses", Path("tasks.jsonl"), [task]
            )
        with self.assertRaises(ValueError):
            _validate_compatibility(
                [SingleAgentHarness()], "grok-build", Path("tasks.jsonl"), [task]
            )
        with self.assertRaises(ValueError):
            _validate_compatibility(
                [GrokBuildHarness()],
                "grok-build",
                Path("tasks.jsonl"),
                [task, Task(task_id="second", prompt="question")],
            )


class XAIHostedTests(unittest.IsolatedAsyncioTestCase):
    async def test_documented_four_agent_request_shape(self) -> None:
        provider_result = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "leader answer"}],
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 8,
                "input_tokens_details": {"cached_tokens": 5},
            },
        }
        post = AsyncMock(return_value=provider_result)
        backend = XAIResponsesBackend(
            model="grok-4.20-multi-agent-0309",
            api_key="test-key",
            pricing=TokenPricing(1.0, 2.0),
        )
        request = ModelRequest(
            agent_id="/xai-hosted/leader",
            role="xai_hosted_multi_agent",
            prompt="question",
            metadata={"xai_multi_agent": True, "reasoning_effort": "low"},
        )
        with patch("scaffoldlab.providers._post_json", new=post):
            response = await backend.complete(request)

        self.assertEqual(response.text, "leader answer")
        self.assertEqual(response.usage.input_tokens, 20)
        self.assertEqual(response.usage.cache_read_input_tokens, 5)
        self.assertTrue(response.usage.cost_known)
        call = post.await_args
        self.assertEqual(call.args[0], "https://api.x.ai/v1/responses")
        payload = call.kwargs["payload"]
        self.assertEqual(payload["model"], "grok-4.20-multi-agent-0309")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["input"], [{"role": "user", "content": "question"}])
        self.assertNotIn("tools", payload)

    def test_hosted_agent_count_mapping_and_compatibility(self) -> None:
        four = XAIHostedMultiAgentHarness(agent_count=4)
        sixteen = XAIHostedMultiAgentHarness(agent_count=16)
        self.assertEqual(four.reasoning_effort, "low")
        self.assertEqual(sixteen.reasoning_effort, "high")
        with self.assertRaises(ValueError):
            XAIHostedMultiAgentHarness(agent_count=8)
        with self.assertRaisesRegex(ValueError, "pinned model"):
            XAIResponsesBackend(model="grok-4.20-multi-agent", api_key="test-key")

        task = Task(task_id="task", prompt="question")
        with self.assertRaises(ValueError):
            _validate_compatibility(
                [four], "openai-responses", Path("tasks.jsonl"), [task]
            )
        with self.assertRaises(ValueError):
            _validate_compatibility(
                [SingleAgentHarness()],
                "xai-responses",
                Path("tasks.jsonl"),
                [task],
            )


if __name__ == "__main__":
    unittest.main()
