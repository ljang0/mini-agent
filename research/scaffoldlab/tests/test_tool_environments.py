import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, patch

from scaffoldlab.environments.base import (
    EnvironmentScope,
    ToolEnvironment,
    ToolExecution,
)
from scaffoldlab.environments.browser import BrowserDriver, BrowserEnvironment
from scaffoldlab.environments.computer import ComputerDriver, ComputerEnvironment
from scaffoldlab.environments.configured import ConfiguredEnvironmentFactory
from scaffoldlab.environments.swe import SWEEnvironment
from scaffoldlab.providers import (
    AnthropicMessagesBackend,
    OpenAICompatibleChatBackend,
    OpenAIResponsesBackend,
)
from scaffoldlab.runtime import RunContext
from scaffoldlab.types import (
    BudgetExceeded,
    BudgetLimits,
    ModelRequest,
    ModelResponse,
    Task,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


PNG_BYTES = b"\x89PNG\r\n\x1a\nscaffoldlab-test"


class _FakeBrowserDriver(BrowserDriver):
    def __init__(self) -> None:
        self.actions: list[tuple[Any, ...]] = []
        self.url = "about:blank"
        self.closed = False

    async def navigate(self, url: str) -> str:
        self.actions.append(("navigate", url))
        self.url = url
        return "200"

    async def click(self, selector: str) -> None:
        self.actions.append(("click", selector))

    async def type_text(self, selector: str, text: str, clear: bool) -> None:
        self.actions.append(("type", selector, text, clear))

    async def press(self, key: str) -> None:
        self.actions.append(("press", key))

    async def extract(self, selector: str | None) -> str:
        self.actions.append(("extract", selector))
        return "visible text"

    async def screenshot(self) -> bytes:
        self.actions.append(("screenshot",))
        return PNG_BYTES

    async def scroll(self, delta_x: int, delta_y: int) -> None:
        self.actions.append(("scroll", delta_x, delta_y))

    async def current_url(self) -> str:
        self.actions.append(("current_url",))
        return self.url

    async def close(self) -> None:
        self.closed = True


class _FakeComputerDriver(ComputerDriver):
    def __init__(self) -> None:
        self.actions: list[Mapping[str, Any]] = []
        self.screenshot_calls = 0
        self.closed = False

    @property
    def width(self) -> int:
        return 1280

    @property
    def height(self) -> int:
        return 720

    async def execute_action(self, action: Mapping[str, Any]) -> None:
        self.actions.append(dict(action))

    async def screenshot(self) -> bytes:
        self.screenshot_calls += 1
        return PNG_BYTES

    async def close(self) -> None:
        self.closed = True


class _StaticEnvironmentScope(EnvironmentScope):
    def __init__(self, environment: ToolEnvironment) -> None:
        self.environment = environment
        self.requested_agents: list[str] = []
        self.closed = False

    async def get(self, agent_id: str) -> ToolEnvironment:
        self.requested_agents.append(agent_id)
        return self.environment

    async def close(self) -> None:
        self.closed = True
        await self.environment.close()


class _PerAgentEnvironmentScope(EnvironmentScope):
    def __init__(self) -> None:
        self.environments: dict[str, _RecordingEnvironment] = {}
        self.requested_agents: list[str] = []

    async def get(self, agent_id: str) -> ToolEnvironment:
        self.requested_agents.append(agent_id)
        environment = self.environments.get(agent_id)
        if environment is None:
            environment = _RecordingEnvironment(f"output from {agent_id}")
            self.environments[agent_id] = environment
        return environment

    async def close(self) -> None:
        for environment in self.environments.values():
            await environment.close()


class _RecordingEnvironment(ToolEnvironment):
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[ToolCall] = []

    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        del provider_family
        return (ToolDefinition(name="emit"),)

    async def execute(self, call: ToolCall) -> ToolExecution:
        self.calls.append(call)
        return ToolExecution(output=self.output)


class _ImageRecordingEnvironment(_RecordingEnvironment):
    async def execute(self, call: ToolCall) -> ToolExecution:
        self.calls.append(call)
        return ToolExecution(
            output=self.output,
            image_data_url=(
                "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
            ),
        )


class _ToolCallingBackend:
    tool_family = "generic"

    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.calls = tuple(calls)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                text="",
                tool_calls=self.calls,
                continuation={"opaque": "test-continuation"},
            )
        return ModelResponse(text="done")


class BrowserEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_driver_and_allowed_host_boundary(self) -> None:
        driver = _FakeBrowserDriver()
        environment = await BrowserEnvironment(
            driver,
            allowed_hosts=("example.com",),
        ).start()

        allowed = await environment.execute(
            ToolCall(
                call_id="nav-1",
                name="browser_navigate",
                arguments={"url": "https://Sub.Example.com/path"},
            )
        )
        self.assertFalse(allowed.is_error)
        self.assertEqual(
            driver.actions,
            [("navigate", "https://Sub.Example.com/path")],
        )

        rejected = await environment.execute(
            ToolCall(
                call_id="nav-2",
                name="browser_navigate",
                arguments={"url": "https://example.com.attacker.invalid/"},
            )
        )
        non_http = await environment.execute(
            ToolCall(
                call_id="nav-3",
                name="browser_navigate",
                arguments={"url": "file:///etc/passwd"},
            )
        )
        self.assertTrue(rejected.is_error)
        self.assertIn("not allowlisted", rejected.output)
        self.assertTrue(non_http.is_error)
        self.assertIn("HTTP(S)", non_http.output)
        self.assertEqual(len(driver.actions), 1)

        await environment.execute(
            ToolCall("click", "browser_click", {"selector": "#submit"})
        )
        await environment.execute(
            ToolCall(
                "type",
                "browser_type",
                {"selector": "#query", "text": "hello", "clear": False},
            )
        )
        await environment.execute(ToolCall("press", "browser_press", {"key": "Enter"}))
        extracted = await environment.execute(
            ToolCall("extract", "browser_extract", {})
        )
        await environment.execute(
            ToolCall("scroll", "browser_scroll", {"delta_x": 3, "delta_y": 7})
        )
        screenshot = await environment.execute(
            ToolCall("shot", "browser_screenshot", {})
        )
        current = await environment.execute(ToolCall("url", "browser_current_url", {}))

        self.assertEqual(extracted.output, "visible text")
        self.assertEqual(current.output, "https://Sub.Example.com/path")
        self.assertEqual(
            screenshot.image_data_url,
            "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii"),
        )
        self.assertIn(("click", "#submit"), driver.actions)
        self.assertIn(("type", "#query", "hello", False), driver.actions)
        self.assertIn(("press", "Enter"), driver.actions)
        self.assertIn(("scroll", 3, 7), driver.actions)

        await environment.close()
        self.assertTrue(driver.closed)


class ProviderToolLoopProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_function_strictness_is_schema_and_endpoint_aware(
        self,
    ) -> None:
        strict_schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                }
            },
            "required": ["config"],
            "additionalProperties": False,
        }
        optional_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        }
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
        chat_completed = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        post = AsyncMock(side_effect=[completed, completed, chat_completed])
        native = OpenAIResponsesBackend(model="gpt-test", api_key="key")
        compatible = OpenAIResponsesBackend(
            model="compatible-model",
            api_key="key",
            base_url="https://compatible.invalid/v1",
            tool_family="generic",
            provider_label="openai-compatible-responses",
        )
        compatible_chat = OpenAICompatibleChatBackend(
            model="compatible-model",
            api_key="key",
            base_url="https://compatible.invalid/v1",
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            await native.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="test",
                    tools=(
                        ToolDefinition(name="strict_tool", input_schema=strict_schema),
                        ToolDefinition(
                            name="optional_tool", input_schema=optional_schema
                        ),
                    ),
                )
            )
            await compatible.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="test",
                    tools=(
                        ToolDefinition(
                            name="default_tool", input_schema=optional_schema
                        ),
                        ToolDefinition(
                            name="explicit_tool",
                            input_schema=optional_schema,
                            provider_options={"strict": True},
                        ),
                    ),
                )
            )
            await compatible_chat.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="test",
                    tools=(
                        ToolDefinition(
                            name="explicit_chat_tool",
                            input_schema=optional_schema,
                            provider_options={"strict": True},
                        ),
                    ),
                )
            )

        native_tools = post.await_args_list[0].kwargs["payload"]["tools"]
        compatible_tools = post.await_args_list[1].kwargs["payload"]["tools"]
        compatible_chat_tools = post.await_args_list[2].kwargs["payload"]["tools"]
        self.assertTrue(native_tools[0]["strict"])
        self.assertFalse(native_tools[1]["strict"])
        self.assertNotIn("strict", compatible_tools[0])
        self.assertTrue(compatible_tools[1]["strict"])
        self.assertTrue(compatible_chat_tools[0]["function"]["strict"])

    async def test_hosted_openai_routes_subagent_tool_call_to_its_environment(
        self,
    ) -> None:
        raw_tool_call = {
            "type": "function_call",
            "call_id": "proposal-call",
            "name": "emit",
            "arguments": "{}",
            "agent": {"agent_name": "/root/researcher"},
        }
        first_response = {
            "status": "completed",
            "output": [raw_tool_call],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        final_response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "agent": {"agent_name": "/root"},
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        post = AsyncMock(side_effect=[first_response, final_response])
        backend = OpenAIResponsesBackend(model="gpt-5.6-sol", api_key="key")
        scope = _PerAgentEnvironmentScope()
        context = RunContext(
            backend,
            BudgetLimits(max_model_calls=2, wall_time_seconds=2),
            environment=scope,
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            response = await context.call(
                ModelRequest(
                    agent_id="/root",
                    role="hosted_multi_agent",
                    prompt="Compare the proposals",
                    metadata={
                        "openai_multi_agent": True,
                        "max_concurrent_subagents": 3,
                    },
                )
            )

        self.assertEqual(response.text, "done")
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(scope.requested_agents, ["/root", "/root/researcher"])
        self.assertEqual(scope.environments["/root"].calls, [])
        self.assertEqual(
            [call.call_id for call in scope.environments["/root/researcher"].calls],
            ["proposal-call"],
        )
        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual(
            [
                event.agent_id
                for event in context.trace.events
                if event.event == "tool_call_completed"
            ],
            ["/root/researcher"],
        )

        first_payload = post.await_args_list[0].kwargs["payload"]
        second_payload = post.await_args_list[1].kwargs["payload"]
        self.assertEqual(
            first_payload["multi_agent"],
            {"enabled": True, "max_concurrent_subagents": 3},
        )
        self.assertEqual(
            post.await_args_list[0].kwargs["headers"]["OpenAI-Beta"],
            "responses_multi_agent=v1",
        )
        self.assertEqual(
            second_payload["input"],
            [
                {"role": "user", "content": "Compare the proposals"},
                raw_tool_call,
                {
                    "type": "function_call_output",
                    "call_id": "proposal-call",
                    "output": "output from /root/researcher",
                },
            ],
        )

    async def test_openai_compatible_chat_function_continuation(self) -> None:
        assistant_tool_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "emit-call",
                    "type": "function",
                    "function": {"name": "emit", "arguments": "{}"},
                }
            ],
        }
        first_response = {
            "choices": [
                {
                    "message": assistant_tool_message,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
        final_response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "finished"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1},
        }
        post = AsyncMock(side_effect=[first_response, final_response])
        backend = OpenAICompatibleChatBackend(
            model="compatible-model",
            api_key="key",
            base_url="https://compatible.invalid/v1",
        )
        environment = _RecordingEnvironment("tool output")
        scope = _StaticEnvironmentScope(environment)
        context = RunContext(
            backend,
            BudgetLimits(max_model_calls=2, wall_time_seconds=2),
            environment=scope,
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            response = await context.call(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    system="Use the provided tool.",
                    prompt="Run it",
                )
            )

        self.assertEqual(response.text, "finished")
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 3)
        self.assertEqual(context.ledger.calls, 2)
        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual([call.call_id for call in environment.calls], ["emit-call"])
        self.assertEqual(scope.requested_agents, ["/root", "/root"])

        first_call, second_call = post.await_args_list
        self.assertEqual(
            [first_call.args[0], second_call.args[0]],
            [
                "https://compatible.invalid/v1/chat/completions",
                "https://compatible.invalid/v1/chat/completions",
            ],
        )
        self.assertEqual(
            first_call.kwargs["payload"]["messages"],
            [
                {"role": "system", "content": "Use the provided tool."},
                {"role": "user", "content": "Run it"},
            ],
        )
        self.assertNotIn(
            "strict",
            first_call.kwargs["payload"]["tools"][0]["function"],
        )
        self.assertEqual(
            second_call.kwargs["payload"]["messages"],
            [
                {"role": "system", "content": "Use the provided tool."},
                {"role": "user", "content": "Run it"},
                assistant_tool_message,
                {
                    "role": "tool",
                    "tool_call_id": "emit-call",
                    "content": "tool output",
                },
            ],
        )

    async def test_openai_compatible_chat_places_tool_images_in_user_message(
        self,
    ) -> None:
        assistant_tool_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "screenshot-call",
                    "type": "function",
                    "function": {"name": "emit", "arguments": "{}"},
                }
            ],
        }
        post = AsyncMock(
            side_effect=[
                {
                    "choices": [
                        {
                            "message": assistant_tool_message,
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "seen"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            ]
        )
        backend = OpenAICompatibleChatBackend(
            model="vision-compatible-model",
            api_key="key",
            base_url="https://compatible.invalid/v1",
        )
        environment = _ImageRecordingEnvironment("screenshot captured")
        context = RunContext(
            backend,
            BudgetLimits(max_model_calls=2, wall_time_seconds=2),
            environment=_StaticEnvironmentScope(environment),
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            response = await context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="Inspect it")
            )

        self.assertEqual(response.text, "seen")
        image_data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode(
            "ascii"
        )
        second_messages = post.await_args_list[1].kwargs["payload"]["messages"]
        self.assertEqual(
            second_messages,
            [
                {"role": "user", "content": "Inspect it"},
                assistant_tool_message,
                {
                    "role": "tool",
                    "tool_call_id": "screenshot-call",
                    "content": "screenshot captured",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Image returned by tool 'emit' for call "
                                "screenshot-call."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )
        self.assertIsInstance(second_messages[2]["content"], str)

    async def test_openai_multi_action_computer_and_screenshot_continuation(
        self,
    ) -> None:
        driver = _FakeComputerDriver()
        environment = ComputerEnvironment(driver)
        tool = environment.tools("openai")[0]
        actions = [
            {"type": "click", "x": 12, "y": 34, "button": "left"},
            {"type": "type", "text": "ordered"},
        ]
        raw_computer_call = {
            "type": "computer_call",
            "call_id": "computer-1",
            "actions": actions,
        }
        first_response = {
            "status": "completed",
            "output": [raw_computer_call],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        final_response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 1},
        }
        post = AsyncMock(side_effect=[first_response, final_response])
        backend = OpenAIResponsesBackend(model="test", api_key="key")

        with patch("scaffoldlab.providers._post_json", new=post):
            first = await backend.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="Operate the computer",
                    tools=(tool,),
                )
            )
            self.assertEqual(len(first.tool_calls), 1)
            call = first.tool_calls[0]
            self.assertEqual(call.kind, "openai_computer")
            self.assertEqual(call.arguments, {"actions": actions})

            execution = await environment.execute(call)
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=execution.output,
                kind=call.kind,
                is_error=execution.is_error,
                image_data_url=execution.image_data_url,
                native_output=execution.native_output,
            )
            final = await backend.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="",
                    tools=(tool,),
                    tool_results=(result,),
                    continuation=first.continuation,
                )
            )

        self.assertEqual(final.text, "done")
        self.assertEqual(driver.actions, actions)
        self.assertEqual(driver.screenshot_calls, 1)
        first_payload = post.await_args_list[0].kwargs["payload"]
        second_payload = post.await_args_list[1].kwargs["payload"]
        self.assertEqual(first_payload["tools"], [{"type": "computer"}])
        self.assertEqual(
            second_payload["input"],
            [
                {"role": "user", "content": "Operate the computer"},
                raw_computer_call,
                {
                    "type": "computer_call_output",
                    "call_id": "computer-1",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": execution.image_data_url,
                        "detail": "original",
                    },
                },
            ],
        )

    async def test_anthropic_assistant_then_tool_results_and_native_types(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            swe = SWEEnvironment(
                Path(directory),
                allow_write=True,
                allow_shell=True,
                allow_native_shell=True,
                protocol="auto",
            )
            computer_driver = _FakeComputerDriver()
            computer = ComputerEnvironment(computer_driver)
            tools = (*swe.tools("anthropic"), *computer.tools("anthropic"))
            raw_content = [
                {"type": "text", "text": "I will inspect and edit."},
                {
                    "type": "tool_use",
                    "id": "editor-1",
                    "name": "str_replace_based_edit_tool",
                    "input": {
                        "command": "create",
                        "path": "note.txt",
                        "file_text": "hello",
                    },
                },
                {
                    "type": "tool_use",
                    "id": "computer-1",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                },
                {
                    "type": "tool_use",
                    "id": "bash-1",
                    "name": "bash",
                    "input": {"command": "pwd"},
                },
            ]
            first_response = {
                "stop_reason": "tool_use",
                "content": raw_content,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            }
            final_response = {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "finished"}],
                "usage": {"input_tokens": 6, "output_tokens": 2},
            }
            post = AsyncMock(side_effect=[first_response, final_response])
            backend = AnthropicMessagesBackend(model="test", api_key="key")

            with patch("scaffoldlab.providers._post_json", new=post):
                first = await backend.complete(
                    ModelRequest(
                        agent_id="/root",
                        role="solver",
                        prompt="Fix the task",
                        tools=tools,
                    )
                )
                self.assertEqual(
                    [call.kind for call in first.tool_calls],
                    [
                        "anthropic_text_editor_20250728",
                        "anthropic_computer_20251124",
                        "anthropic_bash_20250124",
                    ],
                )
                computer_execution = await computer.execute(first.tool_calls[1])
                results = (
                    ToolResult(
                        call_id="editor-1",
                        name="str_replace_based_edit_tool",
                        output="create succeeded",
                        kind="anthropic_text_editor_20250728",
                    ),
                    ToolResult(
                        call_id="computer-1",
                        name="computer",
                        output=computer_execution.output,
                        kind="anthropic_computer_20251124",
                        image_data_url=computer_execution.image_data_url,
                        native_output=computer_execution.native_output,
                    ),
                    ToolResult(
                        call_id="bash-1",
                        name="bash",
                        output="/workspace\n",
                        kind="anthropic_bash_20250124",
                    ),
                )
                final = await backend.complete(
                    ModelRequest(
                        agent_id="/root",
                        role="solver",
                        prompt="",
                        tools=tools,
                        tool_results=results,
                        continuation=first.continuation,
                    )
                )

            self.assertEqual(final.text, "finished")
            first_payload = post.await_args_list[0].kwargs["payload"]
            first_headers = post.await_args_list[0].kwargs["headers"]
            self.assertEqual(
                [definition["type"] for definition in first_payload["tools"]],
                [
                    "text_editor_20250728",
                    "bash_20250124",
                    "computer_20251124",
                ],
            )
            self.assertEqual(
                [definition["name"] for definition in first_payload["tools"]],
                ["str_replace_based_edit_tool", "bash", "computer"],
            )
            self.assertEqual(
                first_payload["tools"][2],
                {
                    "type": "computer_20251124",
                    "name": "computer",
                    "display_width_px": 1280,
                    "display_height_px": 720,
                },
            )
            self.assertEqual(first_headers["anthropic-beta"], "computer-use-2025-11-24")

            second_payload = post.await_args_list[1].kwargs["payload"]
            messages = second_payload["messages"]
            self.assertEqual(
                [message["role"] for message in messages],
                ["user", "assistant", "user"],
            )
            self.assertEqual(messages[0]["content"], "Fix the task")
            self.assertEqual(messages[1]["content"], raw_content)
            tool_result_blocks = messages[2]["content"]
            self.assertEqual(
                [block["tool_use_id"] for block in tool_result_blocks],
                ["editor-1", "computer-1", "bash-1"],
            )
            self.assertIsInstance(tool_result_blocks[0]["content"], str)
            self.assertEqual(
                tool_result_blocks[1]["content"][1],
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(PNG_BYTES).decode("ascii"),
                    },
                },
            )

            await swe.close()
            await computer.close()
            self.assertTrue(computer_driver.closed)


class SWEEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_shell_requires_explicit_opt_in_and_preserves_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires protocol='auto'"):
                SWEEnvironment(
                    Path(directory),
                    allow_shell=True,
                    allow_native_shell=True,
                    protocol="generic",
                )
            safe = SWEEnvironment(
                Path(directory),
                allow_shell=True,
                command_allowlist=("git",),
                protocol="auto",
            )
            self.assertEqual(
                [tool.name for tool in safe.tools("openai")][-1], "run_command"
            )
            self.assertEqual(
                [tool.name for tool in safe.tools("anthropic")],
                ["str_replace_based_edit_tool", "run_command"],
            )
            denied = await safe.execute(
                ToolCall("shell", "shell", {"commands": ["pwd"]})
            )
            self.assertTrue(denied.is_error)
            self.assertIn("native shell execution is disabled", denied.output)

            native = SWEEnvironment(
                Path(directory),
                allow_shell=True,
                allow_native_shell=True,
                protocol="auto",
            )
            self.assertEqual(native.tools("openai")[-1].name, "shell")
            self.assertEqual(native.tools("anthropic")[-1].name, "bash")
            shell_output = {
                "stdout": "ok",
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": 0},
            }
            native._bash.run = AsyncMock(return_value=shell_output)  # type: ignore[method-assign]
            without_limit = await native.execute(
                ToolCall("shell-1", "shell", {"commands": ["pwd"]})
            )
            with_limit = await native.execute(
                ToolCall(
                    "shell-2",
                    "shell",
                    {"commands": ["pwd"], "max_output_length": 17},
                )
            )
            self.assertEqual(without_limit.native_output, {"output": [shell_output]})
            self.assertEqual(
                with_limit.native_output,
                {"output": [shell_output], "max_output_length": 17},
            )
            await safe.close()
            await native.close()

    async def test_subprocess_uses_trusted_path_and_drops_host_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            malicious = workspace / "env"
            malicious.write_text("#!/bin/sh\nprintf MALICIOUS\n", encoding="utf-8")
            malicious.chmod(0o755)
            environment = SWEEnvironment(
                workspace,
                allow_shell=True,
                command_allowlist=("env",),
                protocol="generic",
            )
            with patch.dict(
                os.environ,
                {
                    "PATH": f"{workspace}:/usr/bin:/bin",
                    "SCAFFOLDLAB_TEST_SECRET": "do-not-inherit",
                },
            ):
                execution = await environment.execute(
                    ToolCall("env", "run_command", {"argv": ["env"]})
                )
            self.assertFalse(execution.is_error, execution.output)
            stdout = json.loads(execution.output)["stdout"]
            self.assertIn("PATH=/usr/bin:/bin:/usr/sbin:/sbin", stdout)
            self.assertNotIn("SCAFFOLDLAB_TEST_SECRET", stdout)
            self.assertNotIn("MALICIOUS", stdout)
            await environment.close()

    async def test_containment_apply_patch_and_allowlisted_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "hello.txt"
            target.write_text("old\n", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("safe\n", encoding="utf-8")
            (workspace / "outside-link.txt").symlink_to(outside)
            environment = SWEEnvironment(
                workspace,
                allow_write=True,
                allow_shell=True,
                command_allowlist=("pwd",),
                protocol="generic",
            )

            parent_escape = await environment.execute(
                ToolCall("read-parent", "read_file", {"path": "../outside.txt"})
            )
            symlink_escape = await environment.execute(
                ToolCall("read-link", "read_file", {"path": "outside-link.txt"})
            )
            escape_patch = """--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-safe
+changed
"""
            rejected_patch = await environment.execute(
                ToolCall("patch-escape", "apply_patch", {"patch": escape_patch})
            )
            escaped_search = await environment.execute(
                ToolCall(
                    "search-escape",
                    "search_files",
                    {"query": "safe", "glob": "../outside.txt"},
                )
            )

            self.assertTrue(parent_escape.is_error)
            self.assertTrue(symlink_escape.is_error)
            self.assertTrue(rejected_patch.is_error)
            self.assertEqual(escaped_search.output, "")
            self.assertIn("escapes the SWE workspace", parent_escape.output)
            self.assertIn("escapes the SWE workspace", symlink_escape.output)
            self.assertIn("escapes the SWE workspace", rejected_patch.output)
            self.assertEqual(outside.read_text(encoding="utf-8"), "safe\n")

            valid_patch = """--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-old
+new
"""
            applied = await environment.execute(
                ToolCall("patch-good", "apply_patch", {"patch": valid_patch})
            )
            self.assertFalse(applied.is_error, applied.output)
            self.assertEqual(applied.output, "patch applied")
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")

            pwd = await environment.execute(
                ToolCall("pwd", "run_command", {"argv": ["pwd"]})
            )
            denied = await environment.execute(
                ToolCall(
                    "denied",
                    "run_command",
                    {"argv": ["sh", "-c", "printf escaped"]},
                )
            )
            disguised = await environment.execute(
                ToolCall(
                    "disguised",
                    "run_command",
                    {"argv": ["/tmp/evil/pwd"]},
                )
            )
            self.assertFalse(pwd.is_error, pwd.output)
            self.assertEqual(
                Path(json.loads(pwd.output)["stdout"].strip()).resolve(),
                workspace.resolve(),
            )
            self.assertTrue(denied.is_error)
            self.assertIn("not allowlisted: sh", denied.output)
            self.assertTrue(disguised.is_error)
            self.assertIn("bare executable name", disguised.output)

            await environment.close()


class ToolBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_budget_stops_before_next_environment_action(self) -> None:
        environment = _RecordingEnvironment("ok")
        backend = _ToolCallingBackend(
            (
                ToolCall("call-1", "emit", {"value": 1}),
                ToolCall("call-2", "emit", {"value": 2}),
            )
        )
        context = RunContext(
            backend,
            BudgetLimits(
                max_model_calls=4,
                max_tool_calls=1,
                wall_time_seconds=2,
            ),
            environment=_StaticEnvironmentScope(environment),
        )

        with self.assertRaisesRegex(BudgetExceeded, "tool-call budget exhausted"):
            await context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="go")
            )

        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual([call.call_id for call in environment.calls], ["call-1"])
        self.assertEqual(len(backend.requests), 1)

    async def test_tool_output_budget_records_crossing_output_then_stops(self) -> None:
        environment = _RecordingEnvironment("abcde")
        backend = _ToolCallingBackend((ToolCall("call-1", "emit", {}),))
        context = RunContext(
            backend,
            BudgetLimits(
                max_model_calls=4,
                max_tool_calls=2,
                max_tool_output_bytes=4,
                wall_time_seconds=2,
            ),
            environment=_StaticEnvironmentScope(environment),
        )

        with self.assertRaisesRegex(BudgetExceeded, "tool-output byte budget exceeded"):
            await context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="go")
            )

        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual(context.ledger.tool_output_bytes, 5)
        self.assertEqual(len(environment.calls), 1)
        self.assertEqual(len(backend.requests), 1)
        terminal_events = [
            event.event
            for event in context.trace.events
            if event.event.startswith("tool_call_")
        ]
        self.assertEqual(terminal_events, ["tool_call_started", "tool_call_failed"])


class ComputerEnvironmentRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_action_returns_recovery_screenshot(self) -> None:
        driver = _FakeComputerDriver()
        environment = ComputerEnvironment(driver)
        execution = await environment.execute(
            ToolCall("bad", "computer", {"actions": []}, kind="openai_computer")
        )
        self.assertTrue(execution.is_error)
        self.assertEqual(
            execution.image_data_url,
            "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii"),
        )
        self.assertEqual(
            execution.native_output,
            {"type": "computer_screenshot", "detail": "original"},
        )
        self.assertTrue(execution.metadata["recovery_screenshot"])


class ConfiguredEnvironmentIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_agent_workspace_copies_are_isolated_and_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "seed.txt").write_text("seed\n", encoding="utf-8")
            factory = ConfiguredEnvironmentFactory(
                {
                    "type": "swe",
                    "workspace": str(source),
                    "workspace_mode": "copy",
                    "isolation": "per_agent",
                    "allow_write": True,
                    "protocol": "generic",
                }
            )
            scope = await factory.begin(Task(task_id="copy", prompt="test"))
            first = await scope.get("/agent/one")
            second = await scope.get("/agent/two")
            self.assertIsInstance(first, SWEEnvironment)
            self.assertIsInstance(second, SWEEnvironment)
            assert isinstance(first, SWEEnvironment)
            assert isinstance(second, SWEEnvironment)
            first_workspace = first.workspace
            second_workspace = second.workspace

            self.assertNotEqual(first_workspace, second_workspace)
            self.assertNotEqual(first_workspace, source)
            self.assertNotEqual(second_workspace, source)
            self.assertEqual(
                (first_workspace / "seed.txt").read_text(encoding="utf-8"),
                "seed\n",
            )
            self.assertEqual(
                (second_workspace / "seed.txt").read_text(encoding="utf-8"),
                "seed\n",
            )

            patch_text = """--- a/seed.txt
+++ b/seed.txt
@@ -1 +1 @@
-seed
+first agent
"""
            changed = await first.execute(
                ToolCall("edit", "apply_patch", {"patch": patch_text})
            )
            self.assertFalse(changed.is_error, changed.output)
            self.assertEqual(
                (first_workspace / "seed.txt").read_text(encoding="utf-8"),
                "first agent\n",
            )
            self.assertEqual(
                (second_workspace / "seed.txt").read_text(encoding="utf-8"),
                "seed\n",
            )
            self.assertEqual(
                (source / "seed.txt").read_text(encoding="utf-8"), "seed\n"
            )

            await scope.close()
            self.assertFalse(first_workspace.exists())
            self.assertFalse(second_workspace.exists())
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
