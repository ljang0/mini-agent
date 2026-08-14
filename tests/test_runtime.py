from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from mini_agent.agent import MiniAgent
from mini_agent.models import ScriptedModel
from mini_agent.providers import ProviderError, TokenPricing
from mini_agent.runtime import BudgetLedger, RunContext, TraceRecorder
from mini_agent.types import (
    BudgetExceeded,
    BudgetLimits,
    InfrastructureError,
    InvalidAction,
    Message,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolResult,
    Usage,
    strict_json_loads,
)


from support import BlockingModel, EchoEnvironment


class InvalidActionEnvironment(EchoEnvironment):
    async def execute(self, action: ToolCall) -> ToolExecution:
        del action
        raise InvalidAction("model argument is invalid")


class IncompatibleEnvironment(EchoEnvironment):
    async def execute(self, action: ToolCall) -> object:  # type: ignore[override]
        del action
        return object()


class DuplicateToolEnvironment(EchoEnvironment):
    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition("echo"), ToolDefinition("echo"))


class FailingModel:
    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        raise RuntimeError("provider failed")


class InvalidResponseModel:
    async def query(self, messages: Any, tools: Any) -> object:
        del messages, tools
        return object()


class UsageFailingModel:
    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        raise ProviderError(
            "reported failure", usage=Usage(input_tokens=3, output_tokens=2)
        )


def action(call_id: str = "call") -> ModelResponse:
    return ModelResponse(
        text="thinking",
        usage=Usage(input_tokens=2, output_tokens=1),
        tool_calls=(ToolCall(call_id, "echo", {"value": "observed"}),),
    )


class JsonContractTests(unittest.TestCase):
    def test_strict_json_rejects_ambiguous_or_nonfinite_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            strict_json_loads('{"key": 1, "key": 2}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            strict_json_loads('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            strict_json_loads('{"value": "\\ud800"}')
        self.assertEqual(strict_json_loads('{"value": 1}'), {"value": 1})


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_observation_is_charged_before_the_model_call(self) -> None:
        class InitialEnvironment(EchoEnvironment):
            async def initial_observation(self) -> ToolExecution:
                return ToolExecution("12345")

        model = ScriptedModel([ModelResponse("must not run")])
        context = RunContext(
            BudgetLimits(max_tool_output_bytes=4), trace=TraceRecorder()
        )
        with self.assertRaisesRegex(BudgetExceeded, "tool-output byte budget"):
            await MiniAgent(
                model=model,
                environment=InitialEnvironment(),
                context=context,
            ).run("task")
        self.assertEqual(context.ledger.tool_output_bytes, 5)
        self.assertEqual(context.ledger.calls, 0)
        self.assertEqual(len(model.queries), 0)
        self.assertEqual(context.trace.events[-1].event, "initial_observation_failed")

    async def test_ambiguous_tool_names_and_call_ids_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "tool names must be unique"):
            await MiniAgent(
                model=ScriptedModel([ModelResponse("unused")]),
                environment=DuplicateToolEnvironment(),
            ).run("task")

        duplicate_calls = (
            ToolCall("same", "echo", {"value": "one"}),
            ToolCall("same", "echo", {"value": "two"}),
        )
        with self.assertRaisesRegex(ProtocolError, "call ids must be unique"):
            await MiniAgent(
                model=ScriptedModel([ModelResponse("", tool_calls=duplicate_calls)]),
                environment=EchoEnvironment(),
            ).run("task")

    async def test_linear_history_and_accounting(self) -> None:
        environment = EchoEnvironment()
        model = ScriptedModel(
            [
                action(),
                ModelResponse(
                    text="final", usage=Usage(input_tokens=4, output_tokens=2)
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

        self.assertEqual(result.answer, "final")
        self.assertEqual(
            [message.role for message in result.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(result.messages[3].tool_results[0].output, "observed")
        self.assertEqual(context.ledger.calls, 2)
        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertEqual(context.ledger.usage.input_tokens, 6)
        self.assertTrue(environment.finished)

    async def test_invalid_action_is_charged_and_repairable(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    "",
                    tool_calls=(ToolCall("bad", "missing", {}),),
                ),
                ModelResponse("recovered"),
            ]
        )
        context = RunContext(BudgetLimits(max_tool_calls=2))
        result = await MiniAgent(
            model=model, environment=EchoEnvironment(), context=context
        ).run("task")
        observed = result.messages[2].tool_results[0]
        self.assertTrue(observed.is_error)
        self.assertIn("unknown tool", observed.output)
        self.assertEqual(context.ledger.tool_calls, 1)
        self.assertGreater(context.ledger.tool_output_bytes, 0)

    async def test_schema_error_does_not_reach_environment(self) -> None:
        environment = EchoEnvironment()
        context = RunContext()
        result = await context.execute(
            environment,
            ToolCall("bad", "echo", {"extra": "x"}),
            tuple(environment.tools()),
            agent_id="/root",
            role="solver",
        )
        self.assertTrue(result.is_error)
        self.assertIn("missing required", result.output)
        self.assertEqual(environment.calls, [])

    async def test_environment_action_and_return_failures_are_distinct(self) -> None:
        context = RunContext()
        action = ToolCall("call", "echo", {"value": "valid"})
        tools = tuple(EchoEnvironment().tools())

        repairable = await context.execute(
            InvalidActionEnvironment(),
            action,
            tools,
            agent_id="/root",
            role="solver",
        )
        self.assertTrue(repairable.is_error)
        self.assertIn("Invalid action: model argument is invalid", repairable.output)

        with self.assertRaisesRegex(InfrastructureError, "compatible ToolExecution"):
            await context.execute(
                IncompatibleEnvironment(),
                action,
                tools,
                agent_id="/root",
                role="solver",
            )
        failed = [
            event for event in context.trace.events if event.event == "tool_call_failed"
        ]
        self.assertEqual(failed[-1].data["error"], "InfrastructureError")

    async def test_malformed_tool_schema_is_infrastructure_not_model_feedback(
        self,
    ) -> None:
        context = RunContext()
        malformed = ToolDefinition(
            "echo",
            input_schema={"type": "array", "properties": {}},
        )
        environment = EchoEnvironment()
        with self.assertRaisesRegex(InfrastructureError, "schema root"):
            await context.execute(
                environment,
                ToolCall("call", "echo", {"value": "valid"}),
                (malformed,),
                agent_id="/root",
                role="solver",
            )
        self.assertEqual(environment.calls, [])
        failed = [
            event for event in context.trace.events if event.event == "tool_call_failed"
        ]
        self.assertEqual(failed[-1].data["error"], "InfrastructureError")

    async def test_tool_enum_violation_is_repairable_before_environment(self) -> None:
        context = RunContext()
        constrained = ToolDefinition(
            "echo",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["allowed"]}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
        environment = EchoEnvironment()
        result = await context.execute(
            environment,
            ToolCall("call", "echo", {"value": "other"}),
            (constrained,),
            agent_id="/root",
            role="solver",
        )
        self.assertTrue(result.is_error)
        self.assertIn("must be one of", result.output)
        self.assertEqual(environment.calls, [])

    async def test_tool_output_budget_crossing_is_charged_and_traced(self) -> None:
        environment = EchoEnvironment()
        context = RunContext(BudgetLimits(max_tool_output_bytes=1))
        with self.assertRaisesRegex(BudgetExceeded, "tool-output byte budget"):
            await context.execute(
                environment,
                ToolCall("large", "echo", {"value": "too large"}),
                tuple(environment.tools()),
                agent_id="/root",
                role="solver",
            )
        self.assertEqual(context.ledger.tool_output_bytes, len(b"too large"))
        failed = [
            event for event in context.trace.events if event.event == "tool_call_failed"
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].data["error"], "BudgetExceeded")

    async def test_stopping_conditions_are_explicit(self) -> None:
        with self.assertRaisesRegex(BudgetExceeded, "max_steps"):
            await MiniAgent(
                model=ScriptedModel([action()]),
                environment=EchoEnvironment(),
                max_steps=1,
            ).run("task")

        context = RunContext(BudgetLimits(max_model_calls=1))
        with self.assertRaisesRegex(BudgetExceeded, "model-call"):
            await MiniAgent(
                model=ScriptedModel([action(), ModelResponse("never")]),
                environment=EchoEnvironment(),
                max_steps=2,
                context=context,
            ).run("task")

        with self.assertRaises(ProtocolError):
            await MiniAgent(
                model=ScriptedModel([ModelResponse("")]),
                environment=EchoEnvironment(),
            ).run("task")

    async def test_provider_failure_marks_unknown_usage(self) -> None:
        context = RunContext()
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await MiniAgent(
                model=FailingModel(),
                environment=EchoEnvironment(),
                context=context,
            ).run("task")
        self.assertFalse(context.ledger.usage.complete)
        self.assertFalse(context.ledger.usage.cost_known)
        self.assertIn(
            "model_call_failed", [event.event for event in context.trace.events]
        )

    async def test_invalid_model_response_marks_unknown_usage(self) -> None:
        context = RunContext()
        with self.assertRaisesRegex(ProtocolError, "must return ModelResponse"):
            await MiniAgent(
                model=InvalidResponseModel(),
                environment=EchoEnvironment(),
                context=context,
            ).run("task")
        self.assertEqual(context.ledger.calls, 1)
        self.assertFalse(context.ledger.usage.complete)
        self.assertFalse(context.ledger.usage.cost_known)

    async def test_provider_reported_usage_is_preserved(self) -> None:
        context = RunContext()
        with self.assertRaises(ProviderError):
            await MiniAgent(
                model=UsageFailingModel(),
                environment=EchoEnvironment(),
                context=context,
            ).run("task")
        self.assertEqual(context.ledger.usage.input_tokens, 3)
        self.assertTrue(context.ledger.usage.complete)

    async def test_provider_error_usage_can_cross_budget_without_losing_trace(
        self,
    ) -> None:
        context = RunContext(BudgetLimits(max_input_tokens=2))
        with self.assertRaisesRegex(BudgetExceeded, "input-token budget"):
            await MiniAgent(
                model=UsageFailingModel(),
                environment=EchoEnvironment(),
                context=context,
            ).run("task")
        self.assertEqual(context.ledger.usage.input_tokens, 3)
        self.assertIn(
            "model_call_failed", [event.event for event in context.trace.events]
        )

    async def test_cancellation_is_traced_and_usage_is_incomplete(self) -> None:
        model = BlockingModel()
        context = RunContext()
        running = asyncio.create_task(
            MiniAgent(
                model=model,
                environment=EchoEnvironment(),
                context=context,
            ).run("task")
        )
        await model.started.wait()
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertFalse(context.ledger.usage.complete)
        self.assertIn(
            "model_call_cancelled",
            [event.event for event in context.trace.events],
        )

    async def test_wall_time_is_checked_before_backend_call(self) -> None:
        trace = TraceRecorder()
        trace.started -= 2
        context = RunContext(
            ledger=BudgetLedger(BudgetLimits(wall_time_seconds=1)),
            trace=trace,
        )
        model = ScriptedModel([ModelResponse("never")])
        with self.assertRaisesRegex(BudgetExceeded, "wall-time"):
            await MiniAgent(
                model=model, environment=EchoEnvironment(), context=context
            ).run("task")
        self.assertEqual(model.queries, [])

    async def test_per_agent_wall_time_starts_at_configuration(self) -> None:
        ledger = BudgetLedger(BudgetLimits(wall_time_seconds=60))
        ledger.configure_agent("/child", BudgetLimits(wall_time_seconds=0.001))
        await asyncio.sleep(0.01)
        context = RunContext(ledger=ledger)
        model = ScriptedModel([ModelResponse("never")])
        with self.assertRaisesRegex(BudgetExceeded, "per-agent wall-time"):
            await context.query(
                model,
                [Message(role="user", content="task")],
                (),
                agent_id="/child",
                role="solver",
            )
        self.assertEqual(model.queries, [])

    async def test_trace_delay_cannot_start_backend_after_wall_deadline(self) -> None:
        class DelayedTrace(TraceRecorder):
            async def emit(self, event: str, **kwargs: Any) -> None:
                await super().emit(event, **kwargs)
                if event in {"model_call_started", "tool_call_started"}:
                    await asyncio.sleep(0.02)

        trace = DelayedTrace()
        model = ScriptedModel([ModelResponse("never")])
        context = RunContext(
            ledger=BudgetLedger(BudgetLimits(wall_time_seconds=0.01)),
            trace=trace,
        )
        with self.assertRaisesRegex(BudgetExceeded, "wall-time"):
            await context.query(
                model,
                [Message(role="user", content="task")],
                (),
                agent_id="/root",
                role="solver",
            )
        self.assertEqual(model.queries, [])

        trace = DelayedTrace()
        environment = EchoEnvironment()
        context = RunContext(
            ledger=BudgetLedger(BudgetLimits(wall_time_seconds=0.01)),
            trace=trace,
        )
        with self.assertRaisesRegex(BudgetExceeded, "wall-time"):
            await context.execute(
                environment,
                ToolCall("call", "echo", {"value": "never"}),
                tuple(environment.tools()),
                agent_id="/root",
                role="solver",
            )
        self.assertEqual(environment.calls, [])


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_snapshot_uses_an_agent_path_boundary(self) -> None:
        ledger = BudgetLedger(BudgetLimits())
        await ledger.reserve_call("/eval/task/root")
        await ledger.record(Usage(input_tokens=1), "/eval/task/root")
        await ledger.reserve_call("/eval/task-other/root")
        await ledger.record(Usage(input_tokens=2), "/eval/task-other/root")
        snapshot = ledger.snapshot(prefix="/eval/task")
        self.assertEqual(snapshot["model_calls"], 1)
        self.assertEqual(snapshot["usage"]["input_tokens"], 1)

    async def test_agent_ids_include_an_agent_that_never_spent(self) -> None:
        """A spawned agent that did nothing is a coordination finding, not a gap."""

        ledger = BudgetLedger(BudgetLimits())
        ledger.configure_agent("/eval/task/root/1", BudgetLimits())
        await ledger.reserve_call("/eval/task/root")
        self.assertEqual(
            ledger.agent_ids(prefix="/eval/task"),
            ("/eval/task/root", "/eval/task/root/1"),
        )
        self.assertEqual(ledger.agent_ids(prefix="/eval/other"), ())

    async def test_including_idle_agents_does_not_move_any_total(self) -> None:
        """Widening the id set must not change what a prefix snapshot reports."""

        ledger = BudgetLedger(BudgetLimits())
        await ledger.reserve_call("/eval/task/root")
        await ledger.record(Usage(input_tokens=3), "/eval/task/root")
        before = ledger.snapshot(prefix="/eval/task")
        ledger.configure_agent("/eval/task/root/1", BudgetLimits())
        self.assertEqual(ledger.snapshot(prefix="/eval/task"), before)

    def test_restore_rejects_a_non_object_snapshot(self) -> None:
        ledger = BudgetLedger(BudgetLimits())
        with self.assertRaisesRegex(ValueError, "must be an object"):
            ledger.restore([])  # type: ignore[arg-type]

    async def test_concurrent_global_and_agent_accounting_is_exact(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=100))
        ledger.configure_agent(
            "/a", BudgetLimits(max_model_calls=50, max_concurrency=2)
        )

        async def charge(index: int) -> None:
            agent_id = "/a" if index % 2 else "/b"
            await ledger.reserve_call(agent_id)
            await ledger.record(Usage(input_tokens=1), agent_id)

        await asyncio.gather(*(charge(index) for index in range(40)))
        self.assertEqual(ledger.calls, 40)
        self.assertEqual(ledger.usage.input_tokens, 40)
        self.assertEqual(ledger.agent_snapshot("/a")["model_calls"], 20)
        self.assertEqual(ledger.snapshot(prefix="/")["model_calls"], 40)

    async def test_crossing_a_budget_charges_then_stops(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_input_tokens=2, max_model_calls=3))
        await ledger.reserve_call("/root")
        with self.assertRaisesRegex(BudgetExceeded, "input-token budget exceeded"):
            await ledger.record(Usage(input_tokens=3), "/root")
        self.assertEqual(ledger.usage.input_tokens, 3)
        with self.assertRaises(BudgetExceeded):
            await ledger.reserve_call("/root")

    async def test_restore_preserves_paid_accounting(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_model_calls=3))
        ledger.restore(
            {
                "model_calls": 2,
                "tool_calls": 1,
                "tool_output_bytes": 7,
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_known": True,
                    "complete": True,
                },
            }
        )
        await ledger.reserve_call("/root")
        with self.assertRaises(BudgetExceeded):
            await ledger.reserve_call("/root")
        self.assertEqual(ledger.calls, 3)

    async def test_restore_preserves_incomplete_usage_fail_closed(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_input_tokens=100))
        ledger.restore(
            {
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "cost_known": False,
                    "complete": False,
                }
            }
        )
        with self.assertRaisesRegex(BudgetExceeded, "usage is incomplete"):
            await ledger.reserve_call("/root")

    def test_context_rejects_two_budget_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "either limits or ledger"):
            RunContext(BudgetLimits(), ledger=BudgetLedger(BudgetLimits()))

    async def test_unknown_cost_blocks_configured_cost_budget(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_cost_usd=1.0))
        await ledger.reserve_call("/root")
        with self.assertRaisesRegex(BudgetExceeded, "cost is unknown"):
            await ledger.record(Usage(cost_known=False), "/root")


class TraceAndPricingTests(unittest.IsolatedAsyncioTestCase):
    def test_trace_creation_syncs_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mini_agent.runtime.os.fsync"
        ) as sync:
            TraceRecorder(Path(temporary) / "trace.jsonl")
        self.assertEqual(sync.call_count, 2)

    async def test_model_start_is_durable_before_backend_execution(self) -> None:
        ordering: list[str] = []

        class OrderingModel:
            async def query(
                self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
            ) -> ModelResponse:
                del messages, tools
                ordering.append("backend")
                return ModelResponse("done")

        with tempfile.TemporaryDirectory() as temporary:
            trace = TraceRecorder(Path(temporary) / "trace.jsonl")

            def synced(_descriptor: int) -> None:
                ordering.append("fsync")

            with patch("mini_agent.runtime.os.fsync", side_effect=synced):
                await RunContext(trace=trace).query(
                    OrderingModel(), (), (), agent_id="/root", role="solver"
                )

        self.assertEqual(ordering, ["fsync", "backend"])

    async def test_tool_start_is_durable_before_environment_execution(self) -> None:
        ordering: list[str] = []

        class OrderingEnvironment(EchoEnvironment):
            async def execute(self, action: ToolCall) -> ToolExecution:
                ordering.append("environment")
                return await super().execute(action)

        environment = OrderingEnvironment()
        with tempfile.TemporaryDirectory() as temporary:
            trace = TraceRecorder(Path(temporary) / "trace.jsonl")

            def synced(_descriptor: int) -> None:
                ordering.append("fsync")

            with patch("mini_agent.runtime.os.fsync", side_effect=synced):
                await RunContext(trace=trace).execute(
                    environment,
                    ToolCall("call", "echo", {"value": "observed"}),
                    environment.tools(),
                    agent_id="/root",
                    role="solver",
                )

        self.assertEqual(ordering, ["fsync", "environment"])

    async def test_trace_always_binds_exact_tool_definitions(self) -> None:
        first = RunContext()
        second = RunContext()
        await first.query(
            ScriptedModel([ModelResponse("done")]),
            (),
            (ToolDefinition("browser", description="search only"),),
            agent_id="/root",
            role="solver",
        )
        await second.query(
            ScriptedModel([ModelResponse("done")]),
            (),
            (ToolDefinition("browser", description="search and open"),),
            agent_id="/root",
            role="solver",
        )

        first_data = first.trace.events[0].data
        second_data = second.trace.events[0].data
        self.assertEqual(len(first_data["tools_sha256"]), 64)
        self.assertNotEqual(
            first_data["tools_sha256"], second_data["tools_sha256"]
        )
        self.assertNotIn("tools", first_data)

    async def test_trace_hashes_the_resolved_provider_model(self) -> None:
        context = RunContext()
        await context.query(
            ScriptedModel(
                [ModelResponse("done", resolved_model="snapshot-2026-08-01")]
            ),
            (),
            (),
            agent_id="/root",
            role="solver",
        )

        completed = next(
            event for event in context.trace.events
            if event.event == "model_call_completed"
        )
        self.assertEqual(len(completed.data["resolved_model_sha256"]), 64)
        self.assertNotIn("snapshot-2026-08-01", json.dumps(completed.data))

    def test_trace_does_not_chmod_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "shared"
            parent.mkdir()
            parent.chmod(0o1777)
            TraceRecorder(parent / "trace.jsonl")
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o1777)

    async def test_trace_is_private_before_first_write_under_permissive_umask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            previous = os.umask(0)
            try:
                trace = TraceRecorder(path)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                await trace.emit("secret", data={"value": "sensitive"})
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            finally:
                os.umask(previous)

    def test_trace_rejects_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("keep", encoding="utf-8")
            link = root / "trace.jsonl"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                TraceRecorder(link)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    async def test_public_json_values_are_copied_and_native_objects_stay_out_of_trace(
        self,
    ) -> None:
        arguments = {"value": "before"}
        call = ToolCall("call", "echo", arguments)
        arguments["value"] = "after"
        self.assertEqual(call.arguments["value"], "before")
        result = ToolResult("call", "echo", "ok")
        context = RunContext(capture_content=True)
        await context.query(
            ScriptedModel([ModelResponse("done")]),
            [
                Message(role="assistant", tool_calls=(call,)),
                Message(role="tool", tool_results=(result,)),
            ],
            (),
            agent_id="/root",
            role="solver",
        )
        queued = context.trace.events[0].data
        encoded = json.dumps(queued)
        self.assertNotIn("raw", encoded)

        with self.assertRaisesRegex(ValueError, "finite JSON"):
            ToolCall("bad", "echo", {"value": float("nan")})
        with self.assertRaisesRegex(ValueError, "JSON values"):
            ToolCall("bad", "echo", {"value": object()})
        with self.assertRaisesRegex(ValueError, "image data URL"):
            ToolExecution("bad", image_data_url=1)  # type: ignore[arg-type]

    async def test_trace_streams_and_redacts_nested_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            trace = TraceRecorder(path, secrets=("explicit-secret",))
            await trace.emit(
                "event",
                data={
                    "authorization": "Bearer hidden",
                    "nested": {
                        "url": "https://example.test/?api_key=value",
                        "text": "before explicit-secret after",
                        "token": "nested-token",
                        "x-api-key": "nested-key",
                    },
                    "image": "data:image/png;base64,AAAA",
                },
            )
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["data"]["authorization"], "<redacted>")
            self.assertNotIn("value", value["data"]["nested"]["url"])
            self.assertNotIn("explicit-secret", value["data"]["nested"]["text"])
            self.assertEqual(value["data"]["nested"]["token"], "<redacted>")
            self.assertEqual(value["data"]["nested"]["x-api-key"], "<redacted>")
            self.assertIn("<image sha256=", value["data"]["image"])

    async def test_trace_redacts_overlapping_secrets_longest_first(self) -> None:
        trace = TraceRecorder(secrets=("prefix-secret", "prefix-secret-tail"))
        await trace.emit(
            "fixture",
            data={"text": "before prefix-secret-tail after"},
        )
        redacted = trace.events[0].data["text"]
        self.assertEqual(redacted, "before <redacted> after")
        self.assertNotIn("tail", redacted)

    def test_trace_rejects_non_string_secrets(self) -> None:
        with self.assertRaisesRegex(ValueError, "tuple of strings"):
            TraceRecorder(secrets=("ok", 1))  # type: ignore[arg-type]

    async def test_trace_rejects_non_string_nested_keys(self) -> None:
        trace = TraceRecorder()
        with self.assertRaisesRegex(ValueError, "keys must be strings"):
            await trace.emit("bad", data={"nested": {1: "value"}})

    def test_zero_cache_price_is_not_treated_as_missing(self) -> None:
        pricing = TokenPricing(10, 20, cache_read_per_million=0)
        usage = Usage(
            input_tokens=10,
            output_tokens=0,
            cache_read_input_tokens=10,
        )
        self.assertEqual(pricing.cost(usage), 0)

    def test_invalid_usage_and_pricing_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            Usage(input_tokens=1, cache_read_input_tokens=2)
        with self.assertRaises(ValueError):
            TokenPricing(float("nan"), 1)


class RetryTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_records_retry_and_attempt_counts(self) -> None:
        context = RunContext(trace=TraceRecorder())
        await context.query(
            ScriptedModel([ModelResponse("done", retries=2)]),
            [Message(role="user", content="task")],
            (),
            agent_id="/root",
            role="solver",
        )
        completed = [
            event
            for event in context.trace.events
            if event.event == "model_call_completed"
        ][-1]
        self.assertEqual(completed.data["retries"], 2)

        class AttemptFailingModel:
            async def query(self, messages: Any, tools: Any) -> ModelResponse:
                del messages, tools
                raise ProviderError("provider HTTP 500", attempts=3)

        with self.assertRaises(ProviderError):
            await context.query(
                AttemptFailingModel(),
                [Message(role="user", content="task")],
                (),
                agent_id="/root",
                role="solver",
            )
        failed = [
            event
            for event in context.trace.events
            if event.event == "model_call_failed"
        ][-1]
        self.assertEqual(failed.data["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
