import unittest
from unittest.mock import AsyncMock, patch

from scaffoldlab.harnesses import (
    OpenAIHostedMultiAgentHarness,
    XAIHostedMultiAgentHarness,
)
from scaffoldlab.providers import (
    OpenAIResponsesBackend,
    ProviderError,
    XAIResponsesBackend,
)
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelRequest, Task, ToolDefinition


OPENAI_RESULT = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "agent": {"agent_name": "/root"},
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "answer"}],
        }
    ],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

XAI_RESULT = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "answer"}],
        }
    ],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}


class HostedHarnessToolMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_harness_emits_allowlisted_hosted_tool_metadata(self) -> None:
        backend = ScriptedBackend({"/root": ["answer"]})
        harness = OpenAIHostedMultiAgentHarness(hosted_tools=["web_search"])

        result = await harness.run(
            Task(task_id="task", prompt="research"),
            backend,
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(
            backend.requests[0].metadata["openai_hosted_tools"], ["web_search"]
        )
        self.assertEqual(result.metadata["hosted_tools"], ["web_search"])
        self.assertTrue(result.metadata["built_in_tools_enabled"])

    async def test_xai_harness_emits_allowlisted_hosted_tool_metadata(self) -> None:
        backend = ScriptedBackend({"/xai-hosted/leader": ["answer"]})
        harness = XAIHostedMultiAgentHarness(
            agent_count=4,
            hosted_tools=["web_search", "x_search"],
        )

        result = await harness.run(
            Task(task_id="task", prompt="research"),
            backend,
            BudgetLimits(max_model_calls=1),
        )

        self.assertEqual(
            backend.requests[0].metadata["xai_hosted_tools"],
            ["web_search", "x_search"],
        )
        self.assertEqual(result.metadata["hosted_tools"], ["web_search", "x_search"])
        self.assertTrue(result.metadata["built_in_tools_enabled"])

    def test_harnesses_reject_unknown_or_duplicate_hosted_tools(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports 'web_search'"):
            OpenAIHostedMultiAgentHarness(hosted_tools=["file_search"])
        with self.assertRaisesRegex(ValueError, "duplicate hosted tool"):
            OpenAIHostedMultiAgentHarness(hosted_tools=["web_search", "web_search"])
        with self.assertRaisesRegex(ValueError, "only supports"):
            XAIHostedMultiAgentHarness(hosted_tools=["function"])


class HostedBackendToolPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_combines_client_and_hosted_tools_in_payload(self) -> None:
        post = AsyncMock(return_value=OPENAI_RESULT)
        backend = OpenAIResponsesBackend(model="gpt-5.6-sol", api_key="test")
        request = ModelRequest(
            agent_id="/root",
            role="hosted_multi_agent",
            prompt="research",
            metadata={
                "openai_multi_agent": True,
                "openai_hosted_tools": ["web_search"],
            },
            tools=(ToolDefinition(name="local_lookup"),),
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            await backend.complete(request)

        tools = post.await_args.kwargs["payload"]["tools"]
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["name"], "local_lookup")
        self.assertEqual(tools[1], {"type": "web_search"})

    async def test_openai_rejects_unallowlisted_hosted_tool_before_http(self) -> None:
        post = AsyncMock(return_value=OPENAI_RESULT)
        backend = OpenAIResponsesBackend(model="gpt-5.6-sol", api_key="test")
        request = ModelRequest(
            agent_id="/root",
            role="hosted_multi_agent",
            prompt="research",
            metadata={
                "openai_multi_agent": True,
                "openai_hosted_tools": ["file_search"],
            },
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            with self.assertRaisesRegex(ProviderError, "unsupported OpenAI hosted"):
                await backend.complete(request)
        post.assert_not_awaited()

    async def test_xai_sends_documented_server_search_tools(self) -> None:
        post = AsyncMock(return_value=XAI_RESULT)
        backend = XAIResponsesBackend(model="grok-4.20-multi-agent", api_key="test")
        request = ModelRequest(
            agent_id="/xai-hosted/leader",
            role="xai_hosted_multi_agent",
            prompt="research",
            metadata={
                "xai_multi_agent": True,
                "reasoning_effort": "low",
                "xai_hosted_tools": ["web_search", "x_search"],
            },
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            await backend.complete(request)

        self.assertEqual(
            post.await_args.kwargs["payload"]["tools"],
            [{"type": "web_search"}, {"type": "x_search"}],
        )

    async def test_xai_still_rejects_developer_tools_before_http(self) -> None:
        post = AsyncMock(return_value=XAI_RESULT)
        backend = XAIResponsesBackend(model="grok-4.20-multi-agent", api_key="test")
        request = ModelRequest(
            agent_id="/xai-hosted/leader",
            role="xai_hosted_multi_agent",
            prompt="research",
            metadata={
                "xai_multi_agent": True,
                "reasoning_effort": "low",
                "xai_hosted_tools": ["web_search"],
            },
            tools=(ToolDefinition(name="developer_function"),),
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            with self.assertRaisesRegex(ProviderError, "client tools"):
                await backend.complete(request)
        post.assert_not_awaited()

    async def test_xai_rejects_unallowlisted_hosted_tool_before_http(self) -> None:
        post = AsyncMock(return_value=XAI_RESULT)
        backend = XAIResponsesBackend(model="grok-4.20-multi-agent", api_key="test")
        request = ModelRequest(
            agent_id="/xai-hosted/leader",
            role="xai_hosted_multi_agent",
            prompt="research",
            metadata={
                "xai_multi_agent": True,
                "reasoning_effort": "low",
                "xai_hosted_tools": ["function"],
            },
        )

        with patch("scaffoldlab.providers._post_json", new=post):
            with self.assertRaisesRegex(ProviderError, "unsupported xAI hosted"):
                await backend.complete(request)
        post.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
