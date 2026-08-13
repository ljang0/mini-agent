from __future__ import annotations

import json
import unittest
from typing import Any, Sequence
from unittest.mock import AsyncMock, patch

import httpx

from mini_agent.models import BackendModel, build_model
from mini_agent.providers import (
    AnthropicMessagesBackend,
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
    ProviderError,
    _credential,
    TokenPricing,
    _post_json,
)
from mini_agent.types import Message, ModelRequest, ToolDefinition, ToolResult


class FakeStreamResponse:
    def __init__(
        self, content: bytes, *, status_code: int = 200, chunk_size: int = 3
    ) -> None:
        self._content = content
        self.status_code = status_code
        self.is_error = status_code >= 400
        self.headers: dict[str, str] = {}
        self.chunk_size = chunk_size

    async def __aenter__(self) -> "FakeStreamResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def aiter_bytes(self, *, chunk_size: int) -> Any:
        size = min(chunk_size, self.chunk_size)
        for offset in range(0, len(self._content), size):
            yield self._content[offset : offset + size]


class FakeStreamClient:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> "FakeStreamClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def stream(self, *args: object, **kwargs: object) -> FakeStreamResponse:
        return self.response


class FakeSequenceClient:
    """One attempt per stored item: a FakeStreamResponse or an exception."""

    def __init__(self, attempts: Sequence[Any]) -> None:
        self.attempts = list(attempts)
        self.calls = 0

    async def __aenter__(self) -> "FakeSequenceClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def stream(self, *args: object, **kwargs: object) -> FakeStreamResponse:
        self.calls += 1
        item = self.attempts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


TOOL = ToolDefinition(
    "echo",
    "echo text",
    {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    },
)


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_continuation_and_screenshot_payload(self) -> None:
        first = {
            "id": "response-1",
            "model": "test-2026-08-01",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "echo",
                    "arguments": '{"value":"hello"}',
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "input_tokens_details": {
                    "cached_tokens": 3,
                    "cache_write_tokens": 2,
                },
            },
        }
        second = {
            "id": "response-2",
            "model": "test-2026-08-01",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        post = AsyncMock(side_effect=(first, second))
        model = BackendModel(
            OpenAIResponsesBackend(
                model="test",
                api_key="key",
                pricing=TokenPricing(1, 2, 0.5, 1.25),
            )
        )
        messages = [Message("user", "task")]
        with patch("mini_agent.providers._post_json", post):
            response = await model.query(messages, (TOOL,))
            self.assertEqual(response.tool_calls[0].arguments, {"value": "hello"})
            self.assertEqual(response.usage.cache_read_input_tokens, 3)
            self.assertEqual(response.usage.cache_write_input_tokens, 2)
            self.assertAlmostEqual(response.usage.cost_usd, 11 / 1_000_000)
            messages.extend(
                (
                    Message("assistant", tool_calls=response.tool_calls),
                    Message(
                        "tool",
                        tool_results=(
                            ToolResult(
                                "call-1",
                                "echo",
                                "observed",
                                image_data_url="data:image/png;base64,AAAA",
                            ),
                        ),
                    ),
                )
            )
            completed = await model.query(messages, (TOOL,))
        self.assertEqual(completed.text, "done")
        self.assertEqual(completed.resolved_model, "test-2026-08-01")
        payload = post.await_args_list[1].kwargs["payload"]
        self.assertEqual(payload["previous_response_id"], "response-1")
        self.assertEqual(payload["input"][0]["type"], "function_call_output")
        self.assertEqual(payload["input"][1]["content"][0]["type"], "input_image")
        self.assertTrue(completed.usage.cost_known)

    async def test_initial_multimodal_request_preserves_system_boundary(self) -> None:
        post = AsyncMock(
            return_value={
                "model": "test-2026-08-01",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        model = BackendModel(OpenAIResponsesBackend(model="test", api_key="key"))
        with patch("mini_agent.providers._post_json", post):
            await model.query(
                (
                    Message("system", "system"),
                    Message("user", "task"),
                    Message(
                        "user",
                        "screenshot",
                        image_data_url="data:image/png;base64,AAAA",
                    ),
                ),
                (),
            )
        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["instructions"], "system")
        self.assertEqual(
            [part["type"] for part in payload["input"][0]["content"]],
            ["input_text", "input_image"],
        )

    async def test_malformed_tool_call_fails_closed_with_usage(self) -> None:
        post = AsyncMock(
            return_value={
                "id": "r",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c",
                        "name": "echo",
                        "arguments": "not-json",
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", post):
            with self.assertRaises(ProviderError) as raised:
                await backend.complete(
                    __import__(
                        "mini_agent.types", fromlist=["ModelRequest"]
                    ).ModelRequest("task", tools=(TOOL,))
                )
        self.assertEqual(raised.exception.usage.input_tokens, 2)

    async def test_missing_usage_fields_are_reported_incomplete(self) -> None:
        post = AsyncMock(
            return_value={
                "model": "test-2026-08-01",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 2},
            }
        )
        backend = OpenAIResponsesBackend(
            model="test", api_key="key", pricing=TokenPricing(1, 1)
        )
        with patch("mini_agent.providers._post_json", post):
            response = await backend.complete(ModelRequest("task"))
        self.assertFalse(response.usage.complete)
        self.assertFalse(response.usage.cost_known)

    async def test_malformed_text_preserves_known_usage(self) -> None:
        post = AsyncMock(
            return_value={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": 7}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", post):
            with self.assertRaises(ProviderError) as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 2)

    async def test_refusal_is_terminal_and_preserves_usage(self) -> None:
        post = AsyncMock(
            return_value={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": "provider detail"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", post):
            with self.assertRaisesRegex(ProviderError, "refused") as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 2)
        self.assertNotIn("provider detail", str(raised.exception))

    async def test_missing_response_status_fails_closed(self) -> None:
        post = AsyncMock(
            return_value={
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", post):
            with self.assertRaisesRegex(ProviderError, "status") as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 1)

    async def test_output_message_role_is_validated(self) -> None:
        response = {
            "model": "test-2026-08-01",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "wrong"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        with patch(
            "mini_agent.providers._post_json", AsyncMock(return_value=response)
        ):
            with self.assertRaisesRegex(ProviderError, "role") as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 2)

    async def test_response_model_is_required_and_stable_within_a_run(self) -> None:
        backend = OpenAIResponsesBackend(model="alias", api_key="key")
        missing = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with patch(
            "mini_agent.providers._post_json", AsyncMock(return_value=missing)
        ):
            with self.assertRaisesRegex(ProviderError, "response model"):
                await backend.complete(ModelRequest("task"))

        first = {
            "id": "response-1",
            "model": "snapshot-one",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "echo",
                    "arguments": '{"value":"hello"}',
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        second = {
            "model": "snapshot-two",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        model = BackendModel(OpenAIResponsesBackend(model="alias", api_key="key"))
        messages = [Message("user", "task")]
        with patch(
            "mini_agent.providers._post_json",
            AsyncMock(side_effect=(first, second)),
        ):
            response = await model.query(messages, (TOOL,))
            messages.extend(
                (
                    Message("assistant", tool_calls=response.tool_calls),
                    Message(
                        "tool",
                        tool_results=(ToolResult("call-1", "echo", "ok"),),
                    ),
                )
            )
            with self.assertRaisesRegex(ProviderError, "changed resolved model"):
                await model.query(messages, (TOOL,))

        expected = BackendModel(
            OpenAIResponsesBackend(model="alias", api_key="key"),
            expected_resolved_model="expected-snapshot",
        )
        with patch(
            "mini_agent.providers._post_json",
            AsyncMock(
                return_value={
                    "model": "other-snapshot",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "done"}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                }
            ),
        ):
            with self.assertRaisesRegex(ProviderError, "expected snapshot") as raised:
                await expected.query((Message("user", "task"),), ())
        self.assertEqual(raised.exception.usage.input_tokens, 2)


class AnthropicProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_use_continuation_and_cache_usage(self) -> None:
        first = {
            "model": "test-2026-08-01",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {
                    "type": "tool_use",
                    "id": "use-1",
                    "name": "echo",
                    "input": {"value": "hello"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
                "output_tokens": 1,
            },
        }
        second = {
            "model": "test-2026-08-01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        post = AsyncMock(side_effect=(first, second))
        model = BackendModel(
            AnthropicMessagesBackend(
                model="test",
                api_key="key",
                pricing=TokenPricing(1, 2, 0.5, 1.5),
            )
        )
        messages = [Message("user", "task")]
        with patch("mini_agent.providers._post_json", post):
            response = await model.query(messages, (TOOL,))
            self.assertEqual(response.usage.input_tokens, 9)
            messages.extend(
                (
                    Message("assistant", tool_calls=response.tool_calls),
                    Message(
                        "tool",
                        tool_results=(
                            ToolResult(
                                "use-1",
                                "echo",
                                "result",
                                image_data_url="data:image/png;base64,AAAA",
                            ),
                        ),
                    ),
                )
            )
            completed = await model.query(messages, (TOOL,))
        self.assertEqual(completed.text, "done")
        payload = post.await_args_list[1].kwargs["payload"]
        tool_result = payload["messages"][-1]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(
            [item["type"] for item in tool_result["content"]], ["text", "image"]
        )

    async def test_max_tokens_without_tools_is_an_error(self) -> None:
        post = AsyncMock(
            return_value={
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "partial"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 1, "output_tokens": 10},
            }
        )
        backend = AnthropicMessagesBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", post):
            with self.assertRaisesRegex(ProviderError, "max_tokens"):
                await backend.complete(
                    __import__(
                        "mini_agent.types", fromlist=["ModelRequest"]
                    ).ModelRequest("task")
                )

    async def test_terminal_stop_reasons_fail_closed_with_usage(self) -> None:
        backend = AnthropicMessagesBackend(model="test", api_key="key")
        for stop_reason, message in (
            ("refusal", "refused"),
            ("model_context_window_exceeded", "context window"),
            ("pause_turn", "server-tool continuation"),
        ):
            response = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "partial"}],
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
            with (
                self.subTest(stop_reason=stop_reason),
                patch(
                    "mini_agent.providers._post_json",
                    AsyncMock(return_value=response),
                ),
            ):
                with self.assertRaisesRegex(ProviderError, message) as raised:
                    await backend.complete(ModelRequest("task"))
                self.assertEqual(raised.exception.usage.input_tokens, 2)

    async def test_tool_use_stop_reason_must_match_tool_blocks(self) -> None:
        backend = AnthropicMessagesBackend(model="test", api_key="key")
        response = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "not a tool"}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with patch(
            "mini_agent.providers._post_json", AsyncMock(return_value=response)
        ):
            with self.assertRaisesRegex(ProviderError, "without client tool calls"):
                await backend.complete(ModelRequest("task"))

    async def test_response_role_is_validated(self) -> None:
        response = {
            "type": "message",
            "role": "user",
            "content": [{"type": "text", "text": "wrong"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        backend = AnthropicMessagesBackend(model="test", api_key="key")
        with patch(
            "mini_agent.providers._post_json", AsyncMock(return_value=response)
        ):
            with self.assertRaisesRegex(ProviderError, "assistant") as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 2)


class ProviderConfigurationTests(unittest.TestCase):
    def test_backend_model_rejects_invalid_generation_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            BackendModel(object(), max_output_tokens=0)
        with self.assertRaisesRegex(ValueError, "expected_resolved_model"):
            BackendModel(object(), expected_resolved_model=" ")

    def test_expected_resolved_model_is_bound_by_the_model_adapter(self) -> None:
        model = build_model(
            "openai/alias", expected_resolved_model="snapshot-2026-08-01"
        )
        self.assertEqual(model.expected_resolved_model, "snapshot-2026-08-01")

    def test_model_configuration_does_not_silently_replace_explicit_empty_values(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "base_url"):
            build_model("openai/test", base_url="")
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            build_model("anthropic/test", api_key_env="")
        with self.assertRaisesRegex(ValueError, "A-Za-z"):
            build_model("openai/test", api_key_env="KEY-NAME")
        with self.assertRaisesRegex(ValueError, "A-Za-z"):
            AnthropicMessagesBackend(model="test", api_key_env="9KEY")
        ChatCompletionsBackend(model="test", api_key_env="_PRIVATE_KEY_2")
        with self.assertRaisesRegex(ValueError, "provider/model"):
            build_model(1)  # type: ignore[arg-type]

    def test_build_model_keeps_provider_logic_downstream(self) -> None:
        self.assertIsInstance(
            build_model("openai/test").backend, OpenAIResponsesBackend
        )
        self.assertIsInstance(
            build_model("anthropic/test").backend, AnthropicMessagesBackend
        )
        with self.assertRaisesRegex(ValueError, "explicit.*base-url"):
            build_model("meta/test-model")
        meta = build_model("meta/test-model", base_url="https://meta.example/v1").backend
        self.assertIsInstance(meta, OpenAIResponsesBackend)
        self.assertEqual(meta.provider, "meta")
        self.assertEqual(meta.api_key_env, "MODEL_API_KEY")
        self.assertEqual(meta.provenance()["protocol"], "responses")
        with self.assertRaisesRegex(ValueError, "unsupported provider"):
            build_model("unknown/model")

    def test_build_model_protocol_selects_chat_completions(self) -> None:
        chat = build_model(
            "meta/test-model",
            base_url="https://meta.example/v1",
            protocol="chat-completions",
        ).backend
        self.assertIsInstance(chat, ChatCompletionsBackend)
        self.assertEqual(chat.provider, "meta")
        self.assertEqual(chat.api_key_env, "MODEL_API_KEY")
        self.assertEqual(chat.provenance()["protocol"], "chat-completions")
        self.assertEqual(chat.provenance()["continuation"], "full-transcript")
        openai_chat = build_model(
            "openai/test", protocol="chat-completions"
        ).backend
        self.assertIsInstance(openai_chat, ChatCompletionsBackend)
        self.assertEqual(openai_chat.provider, "openai")
        meta_responses = build_model(
            "meta/test-model",
            base_url="https://meta.example/v1",
            protocol="responses",
        ).backend
        self.assertIsInstance(meta_responses, OpenAIResponsesBackend)
        self.assertEqual(meta_responses.provider, "meta")
        with self.assertRaisesRegex(ValueError, "Messages protocol"):
            build_model("anthropic/test", protocol="chat-completions")
        with self.assertRaisesRegex(ValueError, "protocol"):
            build_model("openai/test", protocol="grpc")
        with self.assertRaisesRegex(ValueError, "explicit --base-url"):
            build_model("meta/test-model", protocol="chat-completions")

    def test_endpoint_validation_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIResponsesBackend(
                model="test", base_url="https://user:pass@example.test/v1"
            )
        with self.assertRaises(ValueError):
            AnthropicMessagesBackend(model="test", base_url="file:///tmp/provider")
        with self.assertRaisesRegex(ValueError, "require HTTPS"):
            OpenAIResponsesBackend(
                model="test", base_url="http://provider.example/v1"
            )
        OpenAIResponsesBackend(model="test", base_url="http://127.0.0.1:8000/v1")
        OpenAIResponsesBackend(model="test", base_url="http://localhost:8000/v1")
        with self.assertRaisesRegex(ValueError, "invalid port"):
            OpenAIResponsesBackend(model="test", base_url="http://localhost:bad/v1")
        with self.assertRaisesRegex(ValueError, "store=false"):
            OpenAIResponsesBackend(model="test", default_body={"store": False})
        with self.assertRaisesRegex(ValueError, "store must be boolean"):
            OpenAIResponsesBackend(model="test", default_body={"store": 0})
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            OpenAIResponsesBackend(model="test", api_key="")
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            OpenAIResponsesBackend(
                model="test", default_body={"previous_response_id": "foreign"}
            )
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            AnthropicMessagesBackend(
                model="test", default_body={"tools": [{"name": "foreign"}]}
            )
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            OpenAIResponsesBackend(
                model="test", default_body={"conversation": "foreign"}
            )
        for backend in (OpenAIResponsesBackend, AnthropicMessagesBackend):
            with self.subTest(backend=backend.__name__):
                with self.assertRaisesRegex(ValueError, "streaming"):
                    backend(model="test", default_body={"stream": True})
                backend(model="test", default_body={"stream": False})

    def test_missing_credentials_fail_without_network(self) -> None:
        backend = OpenAIResponsesBackend(
            model="test", api_key_env="DEFINITELY_MISSING_MINI_AGENT_KEY"
        )
        with self.assertRaisesRegex(ProviderError, "missing credential"):
            _credential(backend.api_key, backend.api_key_env)


class ChatCompletionsProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_roundtrip_resends_full_transcript(self) -> None:
        first = {
            "model": "test-model-2026-08-01",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"value":"hello"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
        second = {
            "model": "test-model-2026-08-01",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        }
        post = AsyncMock(side_effect=(first, second))
        model = BackendModel(
            ChatCompletionsBackend(
                model="test-model",
                provider="meta",
                api_key="key",
                base_url="https://meta.example/v1",
                default_headers={"x-eval-fixture-id": "fixture-run-one"},
                pricing=TokenPricing(1, 2, 0.5),
            )
        )
        messages = [Message("system", "rules"), Message("user", "task")]
        with patch("mini_agent.providers._post_json", post):
            response = await model.query(messages, (TOOL,))
            self.assertEqual(response.tool_calls[0].arguments, {"value": "hello"})
            self.assertEqual(response.usage.cache_read_input_tokens, 3)
            messages.extend(
                (
                    Message("assistant", tool_calls=response.tool_calls),
                    Message(
                        "tool",
                        tool_results=(ToolResult("call-1", "echo", "observed"),),
                    ),
                )
            )
            completed = await model.query(messages, (TOOL,))
        self.assertEqual(completed.text, "done")
        first_call = post.await_args_list[0].kwargs
        second_call = post.await_args_list[1].kwargs
        self.assertTrue(post.await_args_list[0].args[0].endswith("/chat/completions"))
        self.assertEqual(
            first_call["headers"]["x-eval-fixture-id"], "fixture-run-one"
        )
        self.assertEqual(first_call["headers"]["Authorization"], "Bearer key")
        transcript = second_call["payload"]["messages"]
        self.assertEqual(
            [message["role"] for message in transcript],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(transcript[3]["tool_call_id"], "call-1")
        self.assertEqual(transcript[3]["content"], "observed")
        for payload in (first_call["payload"], second_call["payload"]):
            for forbidden in (
                "previous_response_id",
                "temperature",
                "top_p",
                "top_k",
                "reasoning_effort",
            ):
                self.assertNotIn(forbidden, payload)
        self.assertNotIn("max_completion_tokens", first_call["payload"])
        self.assertTrue(completed.usage.complete)

    async def test_usage_stays_complete_without_prompt_tokens_details(self) -> None:
        response = {
            "model": "test-2026-08-01",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        backend = ChatCompletionsBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", AsyncMock(return_value=response)):
            result = await backend.complete(ModelRequest("task"))
        self.assertTrue(result.usage.complete)
        self.assertEqual(result.usage.cache_read_input_tokens, 0)

        partial = dict(response)
        partial["usage"] = {"prompt_tokens": 5}
        with patch("mini_agent.providers._post_json", AsyncMock(return_value=partial)):
            result = await backend.complete(ModelRequest("task"))
        self.assertFalse(result.usage.complete)

    async def test_multiple_response_choices_fail_closed(self) -> None:
        choice = {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "done"},
        }
        response = {
            "choices": [choice, choice],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        backend = ChatCompletionsBackend(model="test", api_key="key")
        with patch(
            "mini_agent.providers._post_json", AsyncMock(return_value=response)
        ):
            with self.assertRaisesRegex(ProviderError, "exactly one") as raised:
                await backend.complete(ModelRequest("task"))
        self.assertEqual(raised.exception.usage.input_tokens, 1)

    async def test_length_finish_without_tool_calls_is_error(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "trunca"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        backend = ChatCompletionsBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", AsyncMock(return_value=response)):
            with self.assertRaisesRegex(ProviderError, "exhausted max tokens") as ctx:
                await backend.complete(ModelRequest("task"))
        self.assertIsNotNone(ctx.exception.usage)
        self.assertTrue(ctx.exception.usage.complete)

    async def test_malformed_tool_arguments_fail_closed_with_usage(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "not-json"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        backend = ChatCompletionsBackend(model="test", api_key="key")
        with patch("mini_agent.providers._post_json", AsyncMock(return_value=response)):
            with self.assertRaisesRegex(ProviderError, "invalid JSON") as ctx:
                await backend.complete(ModelRequest("task"))
        self.assertIsNotNone(ctx.exception.usage)

    async def test_refusal_and_inconsistent_tool_finish_fail_closed(self) -> None:
        backend = ChatCompletionsBackend(model="test", api_key="key")
        responses = (
            (
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "refusal": "provider detail",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                "refused",
            ),
            (
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                "without tool calls",
            ),
            (
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "user", "content": "wrong"},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                "role must be assistant",
            ),
        )
        for response, message in responses:
            with (
                self.subTest(message=message),
                patch(
                    "mini_agent.providers._post_json",
                    AsyncMock(return_value=response),
                ),
            ):
                with self.assertRaisesRegex(ProviderError, message) as raised:
                    await backend.complete(ModelRequest("task"))
                self.assertEqual(raised.exception.usage.input_tokens, 1)
                self.assertNotIn("provider detail", str(raised.exception))

    def test_constructor_rejects_reserved_fields_and_headers(self) -> None:
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            ChatCompletionsBackend(model="test", default_body={"messages": []})
        with self.assertRaisesRegex(ValueError, "streaming"):
            ChatCompletionsBackend(model="test", default_body={"stream": True})
        ChatCompletionsBackend(model="test", default_body={"stream": False})
        with self.assertRaisesRegex(ValueError, "n must be 1"):
            ChatCompletionsBackend(model="test", default_body={"n": 2})
        ChatCompletionsBackend(model="test", default_body={"n": 1})
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            ChatCompletionsBackend(
                model="test", default_headers={"Authorization": "Bearer sneak"}
            )
        with self.assertRaisesRegex(ValueError, "control characters"):
            ChatCompletionsBackend(
                model="test", default_headers={"x-fixture-id": "bad\r\nvalue"}
            )
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            OpenAIResponsesBackend(
                model="test", default_headers={"authorization": "Bearer sneak"}
            )
        with self.assertRaisesRegex(ValueError, "harness-owned"):
            AnthropicMessagesBackend(
                model="test", default_headers={"X-Api-Key": "sneak"}
            )


class ProviderTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_rejects_ambiguous_json_objects(self) -> None:
        client = FakeStreamClient(
            FakeStreamResponse(b'{"output": [], "output": []}')
        )
        with patch("mini_agent.providers.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(ProviderError, "invalid JSON"):
                await _post_json(
                    "https://provider.example/v1/responses",
                    headers={},
                    payload={},
                    timeout_seconds=1,
                )

    async def test_http_error_does_not_echo_untrusted_response_body(self) -> None:
        client = FakeStreamClient(
            FakeStreamResponse(
                b"server echoed TOP_SECRET and benchmark content", status_code=401
            )
        )
        with patch("mini_agent.providers.httpx.AsyncClient", return_value=client):
            with self.assertRaises(ProviderError) as raised:
                await _post_json(
                    "https://provider.example/v1/responses",
                    headers={"Authorization": "Bearer TOP_SECRET"},
                    payload={"input": "benchmark content"},
                    timeout_seconds=1,
                )
        message = str(raised.exception)
        self.assertEqual(message, "provider HTTP 401")
        self.assertNotIn("TOP_SECRET", message)
        self.assertNotIn("benchmark", message)

    async def test_http_error_surfaces_only_allowlisted_error_codes(self) -> None:
        body = (
            b'{"error": {"type": "invalid_request_error", '
            b'"code": "invalid_api_key", '
            b'"message": "echoed TOP_SECRET and benchmark content"}}'
        )
        client = FakeStreamClient(FakeStreamResponse(body, status_code=401))
        with patch("mini_agent.providers.httpx.AsyncClient", return_value=client):
            with self.assertRaises(ProviderError) as raised:
                await _post_json(
                    "https://provider.example/v1/responses",
                    headers={"Authorization": "Bearer TOP_SECRET"},
                    payload={},
                    timeout_seconds=1,
                )
        message = str(raised.exception)
        self.assertEqual(
            message,
            "provider HTTP 401 (type=invalid_request_error, code=invalid_api_key)",
        )
        self.assertNotIn("TOP_SECRET", message)
        self.assertNotIn("benchmark", message)

    async def test_transport_streams_and_rejects_oversized_json(self) -> None:
        client = FakeStreamClient(FakeStreamResponse(b'{"too":"large"}'))
        with (
            patch("mini_agent.providers.httpx.AsyncClient", return_value=client),
            patch("mini_agent.providers.MAX_PROVIDER_RESPONSE_BYTES", 8),
        ):
            with self.assertRaisesRegex(ProviderError, "8-byte limit"):
                await _post_json(
                    "https://provider.example/v1/responses",
                    headers={},
                    payload={},
                    timeout_seconds=1,
                )


def _ok_response(payload: dict[str, Any]) -> FakeStreamResponse:
    return FakeStreamResponse(json.dumps(payload).encode(), chunk_size=1024)


class ProviderRetryTests(unittest.IsolatedAsyncioTestCase):
    async def _post(
        self,
        client: FakeSequenceClient,
        *,
        max_retries: int,
        sleeps: list[float] | None = None,
        counter: list[int] | None = None,
    ) -> Any:
        recorded = sleeps if sleeps is not None else []

        async def fake_sleep(seconds: float) -> None:
            recorded.append(seconds)

        with (
            patch("mini_agent.providers.httpx.AsyncClient", return_value=client),
            patch("mini_agent.providers._retry_sleep", fake_sleep),
        ):
            return await _post_json(
                "https://provider.example/v1/responses",
                headers={},
                payload={},
                timeout_seconds=1,
                max_retries=max_retries,
                attempt_counter=counter,
            )

    async def test_retry_then_succeed(self) -> None:
        client = FakeSequenceClient(
            [
                FakeStreamResponse(b"", status_code=429),
                _ok_response({"ok": True}),
            ]
        )
        sleeps: list[float] = []
        counter: list[int] = []
        data = await self._post(
            client, max_retries=2, sleeps=sleeps, counter=counter
        )
        self.assertEqual(data, {"ok": True})
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(counter, [2])

    async def test_retry_after_seconds_is_honored_and_capped(self) -> None:
        first = FakeStreamResponse(b"", status_code=429)
        first.headers["retry-after"] = "7"
        second = FakeStreamResponse(b"", status_code=429)
        second.headers["retry-after"] = "9999"
        client = FakeSequenceClient([first, second, _ok_response({"ok": 1})])
        sleeps: list[float] = []
        await self._post(client, max_retries=2, sleeps=sleeps)
        self.assertEqual(sleeps, [7.0, 30.0])

    async def test_invalid_retry_after_falls_back_to_bounded_backoff(self) -> None:
        first = FakeStreamResponse(b"", status_code=503)
        first.headers["retry-after"] = "Wed, 21 Oct 2026 07:28:00 GMT"
        client = FakeSequenceClient([first, _ok_response({"ok": 1})])
        sleeps: list[float] = []
        await self._post(client, max_retries=1, sleeps=sleeps)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 0.0)
        self.assertLessEqual(sleeps[0], 1.0)

    async def test_auth_errors_are_never_retried(self) -> None:
        client = FakeSequenceClient([FakeStreamResponse(b"", status_code=401)])
        sleeps: list[float] = []
        with self.assertRaises(ProviderError) as raised:
            await self._post(client, max_retries=3, sleeps=sleeps)
        self.assertEqual(client.calls, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(raised.exception.attempts, 1)
        self.assertEqual(str(raised.exception), "provider HTTP 401")

    async def test_exhausted_retries_report_attempt_count(self) -> None:
        client = FakeSequenceClient(
            [FakeStreamResponse(b"", status_code=500) for _ in range(3)]
        )
        sleeps: list[float] = []
        with self.assertRaisesRegex(ProviderError, "after 3 attempts"):
            try:
                await self._post(client, max_retries=2, sleeps=sleeps)
            except ProviderError as exc:
                self.assertEqual(exc.attempts, 3)
                raise
        self.assertEqual(client.calls, 3)
        self.assertEqual(len(sleeps), 2)

    async def test_transport_errors_are_retried(self) -> None:
        client = FakeSequenceClient(
            [httpx.ConnectError("boom"), _ok_response({"ok": 1})]
        )
        data = await self._post(client, max_retries=1)
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(client.calls, 2)

    async def test_zero_retries_disables_the_loop(self) -> None:
        client = FakeSequenceClient([FakeStreamResponse(b"", status_code=429)])
        sleeps: list[float] = []
        with self.assertRaises(ProviderError):
            await self._post(client, max_retries=0, sleeps=sleeps)
        self.assertEqual(client.calls, 1)
        self.assertEqual(sleeps, [])

    async def test_parsed_response_errors_are_never_retried(self) -> None:
        body = {
            "id": "response-1",
            "model": "test-model",
            "status": "failed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        backend = OpenAIResponsesBackend(
            model="test-model", api_key="key", max_retries=3
        )
        client = FakeSequenceClient([_ok_response(body)])

        async def fail_sleep(seconds: float) -> None:
            raise AssertionError("parsed responses must never be retried")

        with (
            patch("mini_agent.providers.httpx.AsyncClient", return_value=client),
            patch("mini_agent.providers._retry_sleep", fail_sleep),
        ):
            with self.assertRaises(ProviderError) as raised:
                await backend.complete(
                    ModelRequest(prompt="task")
                )
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(raised.exception.usage)

    async def test_success_after_retry_reports_retries_on_the_response(self) -> None:
        body = {
            "id": "response-1",
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        backend = OpenAIResponsesBackend(
            model="test-model", api_key="key", max_retries=2
        )
        client = FakeSequenceClient(
            [FakeStreamResponse(b"", status_code=502), _ok_response(body)]
        )

        async def fake_sleep(seconds: float) -> None:
            return None

        with (
            patch("mini_agent.providers.httpx.AsyncClient", return_value=client),
            patch("mini_agent.providers._retry_sleep", fake_sleep),
            patch.dict("os.environ", {"OPENAI_API_KEY": "key"}),
        ):
            response = await backend.complete(
                ModelRequest(prompt="task")
            )
        self.assertEqual(response.retries, 1)
        self.assertEqual(response.usage.input_tokens, 5)

    def test_max_retries_is_validated_and_recorded(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_retries"):
            OpenAIResponsesBackend(model="test", max_retries=-1)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            OpenAIResponsesBackend(model="test", max_retries=True)  # type: ignore[arg-type]
        backend = AnthropicMessagesBackend(model="test", max_retries=5)
        self.assertEqual(backend.provenance()["max_retries"], 5)
        self.assertEqual(backend.provenance()["timeout_seconds"], 300)


def _chat_step_body(step: int, *, tool_call: bool) -> dict[str, Any]:
    if tool_call:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{step}",
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": "{}",
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": "done"}
        finish = "stop"
    return {
        "model": "test-model",
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _image_url(step: int) -> str:
    return f"data:image/png;base64,STEP{step}"


class ImageHistoryEvictionTests(unittest.IsolatedAsyncioTestCase):
    async def _run_chat_steps(
        self, backend: ChatCompletionsBackend, steps: int
    ) -> list[dict[str, Any]]:
        """Drive N tool-call steps, each returning one image; capture payloads."""

        payloads: list[dict[str, Any]] = []

        async def scripted_post(url: str, **kwargs: Any) -> dict[str, Any]:
            payloads.append(kwargs["payload"])
            step = len(payloads)
            return _chat_step_body(step, tool_call=step <= steps)

        model = BackendModel(backend)
        messages: list[Message] = [Message(role="user", content="task")]
        with patch(
            "mini_agent.providers._post_json", side_effect=scripted_post
        ):
            for step in range(1, steps + 2):
                response = await model.query(messages, ())
                if not response.tool_calls:
                    break
                messages.append(
                    Message(role="assistant", tool_calls=response.tool_calls)
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_results=(
                            ToolResult(
                                response.tool_calls[0].call_id,
                                "computer",
                                f"observation {step}",
                                image_data_url=_image_url(step),
                            ),
                        ),
                    )
                )
        return payloads

    @staticmethod
    def _chat_images(payload: dict[str, Any]) -> list[str]:
        found: list[str] = []
        for message in payload["messages"]:
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, list):
                for block in content:
                    if block.get("type") == "image_url":
                        found.append(block["image_url"]["url"])
        return found

    @staticmethod
    def _chat_placeholders(payload: dict[str, Any]) -> int:
        count = 0
        for message in payload["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        block.get("type") == "text"
                        and block.get("text") == "[earlier screenshot elided]"
                    ):
                        count += 1
        return count

    async def test_chat_image_history_is_bounded_to_the_newest_k(self) -> None:
        backend = ChatCompletionsBackend(
            model="test-model", api_key="key", max_history_images=2
        )
        payloads = await self._run_chat_steps(backend, steps=5)
        final = payloads[-1]
        self.assertEqual(
            self._chat_images(final), [_image_url(4), _image_url(5)]
        )
        self.assertEqual(self._chat_placeholders(final), 3)

    async def test_chat_current_step_image_always_survives(self) -> None:
        backend = ChatCompletionsBackend(
            model="test-model", api_key="key", max_history_images=0
        )
        payloads = await self._run_chat_steps(backend, steps=3)
        final = payloads[-1]
        self.assertEqual(self._chat_images(final), [_image_url(3)])
        self.assertEqual(self._chat_placeholders(final), 2)

    async def test_chat_unlimited_history_preserves_current_behavior(self) -> None:
        backend = ChatCompletionsBackend(
            model="test-model", api_key="key", max_history_images=None
        )
        payloads = await self._run_chat_steps(backend, steps=4)
        self.assertEqual(
            self._chat_images(payloads[-1]),
            [_image_url(step) for step in range(1, 5)],
        )

    async def test_anthropic_image_history_is_bounded(self) -> None:
        backend = AnthropicMessagesBackend(
            model="test-model", api_key="key", max_history_images=1
        )
        payloads: list[dict[str, Any]] = []

        def anthropic_body(step: int, *, tool_call: bool) -> dict[str, Any]:
            if tool_call:
                content: list[dict[str, Any]] = [
                    {
                        "type": "tool_use",
                        "id": f"call-{step}",
                        "name": "computer",
                        "input": {},
                    }
                ]
                stop = "tool_use"
            else:
                content = [{"type": "text", "text": "done"}]
                stop = "end_turn"
            return {
                "type": "message",
                "model": "test-model",
                "role": "assistant",
                "content": content,
                "stop_reason": stop,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        async def scripted_post(url: str, **kwargs: Any) -> dict[str, Any]:
            payloads.append(kwargs["payload"])
            step = len(payloads)
            return anthropic_body(step, tool_call=step <= 3)

        model = BackendModel(backend)
        messages: list[Message] = [Message(role="user", content="task")]
        with patch(
            "mini_agent.providers._post_json", side_effect=scripted_post
        ):
            for step in range(1, 5):
                response = await model.query(messages, ())
                if not response.tool_calls:
                    break
                messages.append(
                    Message(role="assistant", tool_calls=response.tool_calls)
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_results=(
                            ToolResult(
                                response.tool_calls[0].call_id,
                                "computer",
                                f"observation {step}",
                                image_data_url=_image_url(step),
                            ),
                        ),
                    )
                )
        final = payloads[-1]
        images: list[str] = []
        placeholders = 0
        tool_use_ids: list[str] = []
        for message in final["messages"]:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_result":
                    tool_use_ids.append(block["tool_use_id"])
                    for inner in block["content"]:
                        if inner.get("type") == "image":
                            images.append(inner["source"]["data"])
                        elif (
                            inner.get("type") == "text"
                            and inner["text"] == "[earlier screenshot elided]"
                        ):
                            placeholders += 1
        self.assertEqual(images, ["STEP3"])
        self.assertEqual(placeholders, 2)
        self.assertEqual(tool_use_ids, ["call-1", "call-2", "call-3"])

    async def test_stored_continuation_is_bounded_and_deterministic(self) -> None:
        backend = ChatCompletionsBackend(
            model="test-model", api_key="key", max_history_images=1
        )
        model = BackendModel(backend)
        payloads: list[dict[str, Any]] = []

        async def scripted_post(url: str, **kwargs: Any) -> dict[str, Any]:
            payloads.append(kwargs["payload"])
            step = len(payloads)
            return _chat_step_body(step, tool_call=step <= 3)

        messages: list[Message] = [Message(role="user", content="task")]
        with patch(
            "mini_agent.providers._post_json", side_effect=scripted_post
        ):
            for step in range(1, 4):
                response = await model.query(messages, ())
                messages.append(
                    Message(role="assistant", tool_calls=response.tool_calls)
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_results=(
                            ToolResult(
                                response.tool_calls[0].call_id,
                                "computer",
                                f"observation {step}",
                                image_data_url=_image_url(step),
                            ),
                        ),
                    )
                )
        continuation = model._continuation
        stored_images = sum(
            1
            for message in continuation
            if message.get("role") == "user"
            and isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        )
        self.assertLessEqual(stored_images, 1)
        request = ModelRequest(
            prompt="",
            tool_results=(
                ToolResult(
                    "call-3", "computer", "obs", image_data_url=_image_url(9)
                ),
            ),
            continuation=tuple(continuation),
        )
        from mini_agent.providers import _chat_transcript

        first = _chat_transcript(request, 1)
        second = _chat_transcript(request, 1)
        self.assertEqual(first, second)

    def test_eviction_is_declared_and_recorded(self) -> None:
        from mini_agent.models import translation_losses_for

        chat_fields = [
            loss.field
            for loss in translation_losses_for("openai", "chat-completions")
        ]
        anthropic_fields = [
            loss.field for loss in translation_losses_for("anthropic")
        ]
        responses_fields = [
            loss.field for loss in translation_losses_for("openai", "responses")
        ]
        self.assertIn("tool_result_image_history", chat_fields)
        self.assertIn("tool_result_image_history", anthropic_fields)
        self.assertNotIn("tool_result_image_history", responses_fields)
        backend = ChatCompletionsBackend(model="test", max_history_images=7)
        self.assertEqual(backend.provenance()["max_history_images"], 7)
        unlimited = AnthropicMessagesBackend(
            model="test", max_history_images=None
        )
        self.assertIsNone(unlimited.provenance()["max_history_images"])

    def test_build_model_threads_and_rejects_the_history_bound(self) -> None:
        chat = build_model(
            "openai/test",
            protocol="chat-completions",
            max_history_images=2,
            max_retries=1,
            timeout_seconds=30,
        ).backend
        self.assertEqual(chat.max_history_images, 2)
        self.assertEqual(chat.max_retries, 1)
        self.assertEqual(chat.timeout_seconds, 30)
        anthropic = build_model(
            "anthropic/test", max_history_images=None
        ).backend
        self.assertIsNone(anthropic.max_history_images)
        with self.assertRaisesRegex(ValueError, "transcript-replay"):
            build_model("openai/test", max_history_images=3)
        with self.assertRaisesRegex(ValueError, "max_history_images"):
            ChatCompletionsBackend(model="test", max_history_images=-1)


if __name__ == "__main__":
    unittest.main()
