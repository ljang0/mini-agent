from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, patch

from scaffoldlab.environments.base import ToolExecution
from scaffoldlab.providers import OpenAIResponsesBackend
from scaffoldlab.runtime import ScriptedBackend

from mini_agent import (
    BackendModel,
    BudgetExceeded,
    BudgetLimits,
    MiniAgent,
    ModelResponse,
    ProtocolError,
    RunContext,
    ScriptedModel,
    ToolCall,
    ToolDefinition,
    Usage,
)
from mini_agent.environments import (
    BashEnvironment,
    CUAEnvironment,
    ComputerObservation,
    JsonlSearchBackend,
    OSWorldClient,
    OSWorldEnvironment,
    WebEnvironment,
)
from mini_agent.environments.base import BaseEnvironment
from mini_agent.profiles import load_profile


class _EchoEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.finished = False
        self.closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition(name="echo"),)

    async def execute(self, action: ToolCall) -> ToolExecution:
        self.calls.append(action)
        if action.name != "echo":
            raise ProtocolError("not echo")
        return ToolExecution(output=str(action.arguments.get("value", "")))

    async def finish(self) -> None:
        self.finished = True

    async def close(self) -> None:
        self.closed = True


class _FailingModel:
    tool_family = "generic"

    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        raise RuntimeError("provider failed")


class _BlockingModel:
    tool_family = "generic"

    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class MiniAgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_linear_history_and_shared_accounting(self) -> None:
        environment = _EchoEnvironment()
        model = ScriptedModel(
            [
                ModelResponse(
                    text="thinking",
                    usage=Usage(input_tokens=3, output_tokens=2),
                    tool_calls=(ToolCall("call-1", "echo", {"value": "observed"}),),
                ),
                ModelResponse(
                    text="final answer", usage=Usage(input_tokens=4, output_tokens=2)
                ),
            ]
        )
        context = RunContext(BudgetLimits(max_model_calls=2, max_tool_calls=1))
        result = await MiniAgent(
            model=model,
            environment=environment,
            system_prompt="system",
            max_steps=2,
            context=context,
        ).run("task")

        self.assertEqual(result.answer, "final answer")
        self.assertEqual(
            [message.role for message in result.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(model.queries[1][0][-1].content, "observed")
        self.assertEqual(context.ledger.calls, 2)
        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual(context.ledger.usage.input_tokens, 7)
        self.assertTrue(environment.finished)

    async def test_invalid_tool_is_an_observation_the_model_can_repair(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    tool_calls=(ToolCall("bad", "missing", {}),),
                ),
                ModelResponse(text="recovered"),
            ]
        )
        result = await MiniAgent(
            model=model, environment=_EchoEnvironment(), max_steps=2
        ).run("task")
        tool_result = result.messages[2].tool_results[0]
        self.assertTrue(tool_result.is_error)
        self.assertIn("unknown tool", tool_result.output)

    async def test_empty_response_is_a_protocol_error(self) -> None:
        agent = MiniAgent(
            model=ScriptedModel([ModelResponse(text="")]),
            environment=_EchoEnvironment(),
        )
        with self.assertRaises(ProtocolError):
            await agent.run("task")

    async def test_step_and_shared_model_call_limits_stop(self) -> None:
        action = ModelResponse(
            text="", tool_calls=(ToolCall("c", "echo", {"value": "x"}),)
        )
        with self.assertRaisesRegex(BudgetExceeded, "max_steps"):
            await MiniAgent(
                model=ScriptedModel([action]),
                environment=_EchoEnvironment(),
                max_steps=1,
            ).run("task")

        context = RunContext(BudgetLimits(max_model_calls=1))
        with self.assertRaisesRegex(BudgetExceeded, "model-call budget"):
            await MiniAgent(
                model=ScriptedModel([action, ModelResponse(text="never")]),
                environment=_EchoEnvironment(),
                max_steps=2,
                context=context,
            ).run("task")

    async def test_provider_error_and_cancellation_are_traced(self) -> None:
        context = RunContext()
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await MiniAgent(
                model=_FailingModel(),
                environment=_EchoEnvironment(),
                context=context,
            ).run("task")
        self.assertIn("model_call_failed", [event.event for event in context.trace.events])

        cancelled_context = RunContext()
        running = asyncio.create_task(
            MiniAgent(
                model=_BlockingModel(),
                environment=_EchoEnvironment(),
                context=cancelled_context,
            ).run("task")
        )
        await asyncio.sleep(0)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertIn(
            "model_call_cancelled",
            [event.event for event in cancelled_context.trace.events],
        )

    async def test_backend_model_sends_initial_screenshot_on_first_turn(self) -> None:
        completed = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        post = AsyncMock(return_value=completed)
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        model = BackendModel(backend)
        from mini_agent.types import Message

        with patch("scaffoldlab.providers._post_json", post):
            await model.query(
                (
                    Message(role="system", content="system"),
                    Message(role="user", content="task"),
                    Message(
                        role="user",
                        content="initial",
                        image_data_url="data:image/png;base64,AAAA",
                    ),
                ),
                (),
            )
        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(payload["input"][0]["content"][1]["type"], "input_image")


class DomainEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_swe_has_one_stateless_bash_tool_and_an_isolated_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "value.txt").write_text("original", encoding="utf-8")
            environment = await BashEnvironment.isolated(source)
            copied = environment.workspace
            self.assertEqual([tool.name for tool in environment.tools()], ["bash"])
            first = await environment.execute(
                ToolCall("one", "bash", {"command": "export ONLY_THIS_CALL=yes; printf changed > value.txt"})
            )
            second = await environment.execute(
                ToolCall("two", "bash", {"command": "printf ${ONLY_THIS_CALL-unset}"})
            )
            self.assertFalse(first.is_error)
            self.assertEqual(second.output, "unset")
            self.assertEqual((source / "value.txt").read_text(), "original")
            self.assertEqual((copied / "value.txt").read_text(), "changed")
            with patch.dict(os.environ, {"MINI_AGENT_TEST_SECRET": "secret"}):
                secret = await environment.execute(
                    ToolCall(
                        "three",
                        "bash",
                        {"command": "printf ${MINI_AGENT_TEST_SECRET-unset}"},
                    )
                )
            self.assertEqual(secret.output, "unset")
            await environment.close()
            self.assertFalse(copied.exists())

    async def test_deterministic_offline_web_search_and_document_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"docid": "2", "text": "orchards and pears"}),
                        json.dumps({"docid": "1", "text": "apple apple orchard"}),
                        json.dumps({"docid": "3", "text": "unrelated ocean"}),
                    ]
                ),
                encoding="utf-8",
            )
            environment = WebEnvironment(
                JsonlSearchBackend(corpus), include_get_document=True
            )
            first = await environment.execute(
                ToolCall("search-1", "search", {"query": "apple orchard"})
            )
            second = await environment.execute(
                ToolCall("search-2", "search", {"query": "apple orchard"})
            )
            self.assertEqual(first.output, second.output)
            self.assertEqual(json.loads(first.output)[0]["docid"], "1")
            document = await environment.execute(
                ToolCall("doc", "get_document", {"docid": "1"})
            )
            self.assertEqual(json.loads(document.output)["text"], "apple apple orchard")

    async def test_profiles_resolve_prompt_tools_pins_and_fidelity(self) -> None:
        for application, tool in (("swe", "bash"), ("web", "search"), ("cua", "computer")):
            profile = load_profile(application)
            manifest = profile.manifest(selected_model="provider/model")
            self.assertEqual(profile.tools, (tool,))
            self.assertEqual(profile.fidelity, "baseline")
            self.assertTrue(profile.system_prompt)
            self.assertEqual(len(manifest["profile_sha256"]), 64)
            self.assertTrue(profile.source)

        openai_web = load_profile("web", "openai")
        self.assertEqual(openai_web.provider, "openai-responses")
        self.assertEqual(openai_web.response_parser, "provider_tool_calls")
        self.assertTrue(openai_web.fidelity_gaps)
        anthropic_cua = load_profile("cua", "anthropic")
        self.assertEqual(anthropic_cua.history["images_to_keep"], 7)


class _FakeComputerClient:
    def __init__(self) -> None:
        self.actions: list[list[dict[str, Any]]] = []
        self.observations = 0
        self.done_calls = 0
        self.closed = False

    async def observe(self) -> ComputerObservation:
        self.observations += 1
        return ComputerObservation(b"png" + bytes([self.observations]), {"width": 100})

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        self.actions.append(actions)
        return {"accepted": len(actions)}

    async def done(self) -> None:
        self.done_calls += 1

    async def close(self) -> None:
        self.closed = True


class CUAEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_step_done_preserves_pixel_actions_and_hides_verifier(self) -> None:
        client = _FakeComputerClient()
        environment = CUAEnvironment(client)
        initial = await environment.initial_observation()
        self.assertTrue(initial.image_data_url.startswith("data:image/png;base64,"))
        result = await environment.execute(
            ToolCall(
                "computer-1",
                "computer",
                {
                    "actions": [
                        {"mouse": {"left_click": [10, 20]}},
                        {"keyboard": {"text": "hello"}},
                    ]
                },
            )
        )
        self.assertEqual(client.actions[0][0]["mouse"]["left_click"], [10, 20])
        self.assertIn("accepted", result.output)
        self.assertFalse(hasattr(client, "finalize"))
        await environment.finish()
        await environment.finish()
        self.assertEqual(client.done_calls, 1)
        await environment.close()
        self.assertTrue(client.closed)

    async def test_forbidden_and_invalid_coordinate_actions_are_rejected(self) -> None:
        environment = CUAEnvironment(_FakeComputerClient())
        for action in (
            {"shell": "whoami"},
            {"script": "pyautogui.click(1, 2)"},
            {"action": "reset"},
            {"mouse": {"left_click": [-1, 2]}},
        ):
            with self.assertRaises(ProtocolError):
                await environment.execute(
                    ToolCall("bad", "computer", {"actions": [action]})
                )

    async def test_osworld_bridge_steps_without_running_evaluator(self) -> None:
        class FakeOSWorld:
            def __init__(self) -> None:
                self.actions: list[str] = []
                self.closed = False

            def step(self, action: str) -> tuple[Mapping[str, Any], int, bool, Mapping[str, Any]]:
                self.actions.append(action)
                return {"screenshot": b"next"}, 0, False, {"ok": True}

            def close(self) -> None:
                self.closed = True

        desktop = FakeOSWorld()
        client = OSWorldClient(desktop, {"screenshot": b"first"})
        environment = OSWorldEnvironment(client)
        await environment.initial_observation()
        await environment.execute(
            ToolCall(
                "osw",
                "computer",
                {"actions": [{"script": "pyautogui.click(1, 2)"}]},
            )
        )
        self.assertEqual(desktop.actions, ["pyautogui.click(1, 2)"])
        self.assertFalse(hasattr(desktop, "evaluate"))
        await environment.close()
        self.assertTrue(desktop.closed)


class MiniAgentCLITests(unittest.TestCase):
    def test_offline_web_run_writes_resolved_manifest_and_trace(self) -> None:
        from mini_agent.cli import main

        backend = ScriptedBackend(
            {
                "/root": [
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall("search", "search", {"query": "alpha"}),
                        ),
                        continuation={"test": True},
                    ),
                    ModelResponse(text="answer [1]"),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"docid": "1", "text": "alpha evidence"}) + "\n",
                encoding="utf-8",
            )
            output = root / "run"
            stdout = io.StringIO()
            with patch("mini_agent.cli._build_backend", return_value=backend):
                with contextlib.redirect_stdout(stdout):
                    status = main(
                        [
                            "run",
                            "--application",
                            "web",
                            "--model",
                            "openai/test",
                            "--profile",
                            "default",
                            "--corpus",
                            str(corpus),
                            "--task",
                            "find alpha",
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(status, 0)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["answer"], "answer [1]")
            self.assertEqual(run["manifest"]["tools"], ["search"])
            self.assertEqual(len(run["manifest"]["task"]["sha256"]), 64)
            self.assertEqual(
                run["manifest"]["environment"]["retrieval"]["corpus_sha256"],
                hashlib.sha256(corpus.read_bytes()).hexdigest(),
            )
            self.assertTrue((output / "trace.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
