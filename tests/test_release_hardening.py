import asyncio
import time
import unittest

from scaffoldlab.providers import (
    AnthropicMessagesBackend,
    OpenAIResponsesBackend,
    ProviderError,
    XAIResponsesBackend,
)
from scaffoldlab.runtime import RunContext, ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelRequest


class ProvenanceTests(unittest.TestCase):
    def test_behavior_affecting_provider_settings_are_recorded(self) -> None:
        openai = OpenAIResponsesBackend(
            model="gpt-5.6-sol",
            api_key="test",
            timeout_seconds=123.0,
        )
        xai = XAIResponsesBackend(
            model="grok-4.20-multi-agent",
            api_key="test",
            timeout_seconds=456.0,
        )
        anthropic = AnthropicMessagesBackend(
            model="test-model",
            api_key="test",
            timeout_seconds=789.0,
            default_max_output_tokens=321,
        )

        self.assertEqual(openai.provenance()["timeout_seconds"], 123.0)
        self.assertEqual(xai.provenance()["timeout_seconds"], 456.0)
        self.assertEqual(anthropic.provenance()["timeout_seconds"], 789.0)
        self.assertEqual(anthropic.provenance()["default_max_output_tokens"], 321)


class HostedBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_hosted_beta_rejects_unsupported_model_before_http(
        self,
    ) -> None:
        backend = OpenAIResponsesBackend(model="not-gpt-5.6", api_key="test")
        with self.assertRaisesRegex(ProviderError, "requires a GPT-5.6 model"):
            await backend.complete(
                ModelRequest(
                    agent_id="/root",
                    role="hosted",
                    prompt="question",
                    metadata={"openai_multi_agent": True},
                )
            )

    async def test_backend_active_interval_excludes_ledger_delay(self) -> None:
        context = RunContext(
            ScriptedBackend({"/root": ["answer"]}),
            BudgetLimits(wall_time_seconds=2),
        )
        original_record = context.ledger.record

        async def delayed_record(usage):
            await asyncio.sleep(0.1)
            await original_record(usage)

        context.ledger.record = delayed_record  # type: ignore[method-assign]
        started = time.perf_counter()
        await context.call(
            ModelRequest(agent_id="/root", role="solver", prompt="question")
        )
        elapsed = time.perf_counter() - started

        self.assertGreater(elapsed, 0.09)
        self.assertLess(context.trace.backend_active_union_seconds, elapsed / 2)


if __name__ == "__main__":
    unittest.main()
