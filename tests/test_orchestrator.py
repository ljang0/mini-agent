from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, Sequence

from mini_agent.agent import MiniAgent
from mini_agent.models import ScriptedModel
from mini_agent.orchestrator import CommunicationEnvironment, Orchestrator
from mini_agent.runtime import RunContext, TraceRecorder
from mini_agent.types import (
    BudgetExceeded,
    BudgetLimits,
    ModelResponse,
    ProtocolError,
    ToolCall,
)

from support import BlockingModel, IsolatedEnvironment


def call(call_id: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        "",
        tool_calls=(ToolCall(call_id, "agent", arguments),),
    )


class ScriptThenBlockModel:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)

    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        if self.responses:
            return self.responses.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def orchestrator_for(
    models: dict[str, Any],
    *,
    environments: dict[str, IsolatedEnvironment] | None = None,
    max_active: int = 4,
    max_total: int = 8,
    per_agent_limits: BudgetLimits | None = None,
    shared_identity: bool = False,
    fail_root_close: bool = False,
    fail_close_ids: set[str] | None = None,
    max_message_bytes: int = 64 * 1024,
    harness: Any = None,
) -> Orchestrator:
    captured = environments if environments is not None else {}

    async def environment_factory(agent_id: str) -> IsolatedEnvironment:
        environment = IsolatedEnvironment(
            agent_id,
            identity="shared" if shared_identity else None,
            fail_close=(fail_root_close and agent_id == "/root")
            or agent_id in (fail_close_ids or set()),
        )
        captured[agent_id] = environment
        return environment

    def agent_builder(
        agent_id: str,
        environment: Any,
        context: RunContext,
    ) -> MiniAgent:
        return MiniAgent(
            model=models[agent_id],
            environment=environment,
            context=context,
            agent_id=agent_id,
            max_steps=12,
        )

    return Orchestrator(
        agent_builder=agent_builder,
        environment_factory=environment_factory,
        context=RunContext(BudgetLimits(max_model_calls=50, max_tool_calls=50)),
        max_active_agents=max_active,
        max_total_agents=max_total,
        per_agent_limits=per_agent_limits,
        max_message_bytes=max_message_bytes,
        harness=harness,
    )


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_spawn_reservations_obey_exact_capacity(self) -> None:
        orchestrator = Orchestrator(
            agent_builder=lambda *args: None,  # type: ignore[arg-type,return-value]
            environment_factory=lambda *args: None,
            context=RunContext(),
            max_active_agents=4,
            max_total_agents=20,
        )
        root = await orchestrator._reserve(None, "/root")
        root.status = "running"

        async def reserve() -> str:
            try:
                return (await orchestrator._reserve("/root")).agent_id
            except ProtocolError:
                return "capacity"

        results = await asyncio.gather(*(reserve() for _ in range(12)))
        self.assertEqual(
            sorted(result for result in results if result != "capacity"),
            ["/root/1", "/root/2", "/root/3"],
        )
        self.assertEqual(results.count("capacity"), 9)
        self.assertEqual(len(orchestrator.records), 4)

    async def test_message_state_is_unchanged_when_trace_persistence_fails(
        self,
    ) -> None:
        class ToggleTrace(TraceRecorder):
            fail_event: str | None = "message_sent"

            async def emit(self, event: str, **kwargs: Any) -> None:
                if event == self.fail_event:
                    raise RuntimeError(f"{event} trace failed")
                await super().emit(event, **kwargs)

        trace = ToggleTrace()
        orchestrator = Orchestrator(
            agent_builder=lambda *args: None,  # type: ignore[arg-type,return-value]
            environment_factory=lambda *args: None,
            context=RunContext(trace=trace),
        )
        await orchestrator._reserve(None, "/root")
        child = await orchestrator._reserve("/root")
        with self.assertRaisesRegex(RuntimeError, "message_sent trace failed"):
            await orchestrator.send_message("/root", child.agent_id, "hello")
        self.assertEqual(child.inbox_bytes, 0)
        self.assertEqual(list(child.inbox), [])

        trace.fail_event = None
        await orchestrator.send_message("/root", child.agent_id, "hello")
        trace.fail_event = "messages_read"
        with self.assertRaisesRegex(RuntimeError, "messages_read trace failed"):
            await orchestrator.read_messages(child.agent_id)
        self.assertEqual(child.inbox_bytes, len("hello".encode("utf-8")))
        self.assertEqual(len(child.inbox), 1)

    async def test_communication_wrapper_retries_failed_cleanup(self) -> None:
        class RetryCloseEnvironment(IsolatedEnvironment):
            def __init__(self) -> None:
                super().__init__("/root")
                self.close_calls = 0

            async def close(self) -> None:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("transient cleanup failure")

        base = RetryCloseEnvironment()
        environment = CommunicationEnvironment(base, object(), "/root")
        with self.assertRaisesRegex(RuntimeError, "transient cleanup failure"):
            await environment.close()
        await environment.close()
        self.assertEqual(base.close_calls, 2)

    async def test_a_failed_child_reports_to_its_parent_without_killing_the_run(
        self,
    ) -> None:
        """A child's failure is the parent's news, not the run's cause of death.

        Multi-agent work fails children routinely -- a bad tool call, an
        exhausted budget, a crashed environment. The parent has to hear about
        it so it can react, and the run has to survive so it can.
        """

        class FailingModel:
            async def query(self, messages: Any, tools: Any) -> ModelResponse:
                del messages, tools
                raise RuntimeError("child exploded")

        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "doomed"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call("inbox", {"action": "inbox"}),
                    ModelResponse("root survived"),
                ]
            ),
            "/root/1": FailingModel(),
        }
        orchestrator = orchestrator_for(models)

        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "root survived")
        self.assertEqual(orchestrator.records["/root/1"].status, "failed")
        inbox_output = models["/root"].queries[3][0][-1].tool_results[0].output
        messages = json.loads(inbox_output)
        self.assertEqual([message["kind"] for message in messages], ["error"])
        self.assertIn("child exploded", messages[0]["content"])

    async def test_spawn_message_wait_and_inbox(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "research"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call("inbox", {"action": "inbox"}),
                    ModelResponse("root answer"),
                ]
            ),
            "/root/1": ScriptedModel(
                [
                    call(
                        "send",
                        {
                            "action": "send",
                            "agent_id": "/root",
                            "message": "working",
                        },
                    ),
                    ModelResponse("child answer"),
                ]
            ),
        }
        environments: dict[str, IsolatedEnvironment] = {}
        orchestrator = orchestrator_for(models, environments=environments)
        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "root answer")
        self.assertEqual(orchestrator.records["/root/1"].result.answer, "child answer")
        inbox_output = models["/root"].queries[3][0][-1].tool_results[0].output
        messages = json.loads(inbox_output)
        self.assertEqual(
            [message["kind"] for message in messages], ["message", "result"]
        )
        self.assertIn("child answer", messages[-1]["content"])
        self.assertTrue(all(item.closed for item in environments.values()))
        events = [event.event for event in orchestrator.context.trace.events]
        self.assertIn("agent_spawned", events)
        self.assertIn("message_sent", events)
        child_agent_tool = next(
            tool for tool in models["/root/1"].queries[0][1] if tool.name == "agent"
        )
        root_agent_tool = next(
            tool for tool in models["/root"].queries[0][1] if tool.name == "agent"
        )
        self.assertIn("have no parent", root_agent_tool.description)
        self.assertIn("'/root/1'", child_agent_tool.description)
        self.assertIn("'/root'", child_agent_tool.description)
        explicit_message = next(
            event
            for event in orchestrator.context.trace.events
            if event.event == "message_sent" and event.data["kind"] == "message"
        )
        self.assertEqual(explicit_message.data["content_bytes"], len(b"working"))
        self.assertEqual(
            explicit_message.data["content_sha256"],
            "dd5ace9e018dbd62336cbe039916ad7c817d0e9bc9934f70d67f3ff75544da88",
        )

    async def test_blocking_inbox_has_no_delivery_race(self) -> None:
        orchestrator = Orchestrator(
            agent_builder=lambda *args: None,  # type: ignore[arg-type,return-value]
            environment_factory=lambda *args: None,
            context=RunContext(),
        )
        await orchestrator._reserve(None, "/root")
        child = await orchestrator._reserve("/root")

        waiting = asyncio.create_task(
            orchestrator.read_messages(child.agent_id, wait=True)
        )
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        await orchestrator.send_message("/root", child.agent_id, "after wait")
        messages = await asyncio.wait_for(waiting, 1)
        self.assertEqual([message.content for message in messages], ["after wait"])

        await orchestrator.send_message("/root", child.agent_id, "before wait")
        messages = await asyncio.wait_for(
            orchestrator.read_messages(child.agent_id, wait=True), 1
        )
        self.assertEqual([message.content for message in messages], ["before wait"])
        self.assertEqual(child.inbox_bytes, 0)
        self.assertFalse(child.inbox_event.is_set())

    async def test_recursive_descendants_have_no_topology_depth_cap(self) -> None:
        depth = 12
        models: dict[str, Any] = {}
        agent_id = "/root"
        for level in range(depth - 1):
            child_id = agent_id + "/1"
            models[agent_id] = ScriptedModel(
                [
                    call(
                        f"spawn-{level}",
                        {"action": "spawn", "task": f"level {level + 1}"},
                    ),
                    call(
                        f"wait-{level}",
                        {"action": "wait", "agent_ids": [child_id]},
                    ),
                    ModelResponse(f"level {level}"),
                ]
            )
            agent_id = child_id
        models[agent_id] = ScriptedModel([ModelResponse("leaf")])
        orchestrator = orchestrator_for(
            models, max_active=depth, max_total=depth
        )
        result = await orchestrator.run("lead")
        self.assertEqual(result.answer, "level 0")
        self.assertEqual(len(orchestrator.records), depth)
        self.assertIn("/root" + "/1" * (depth - 1), orchestrator.records)

    async def test_terminal_agent_cannot_spawn_send_wait_or_adopt(self) -> None:
        orchestrator = orchestrator_for(
            {"/root": ScriptedModel([ModelResponse("done")])}
        )
        await orchestrator.run("lead")
        with self.assertRaisesRegex(ProtocolError, "running agent.*spawn"):
            await orchestrator.spawn("/root", "late")
        with self.assertRaisesRegex(ProtocolError, "running agent.*send"):
            await orchestrator.send_message("/root", "/root", "late")
        with self.assertRaisesRegex(ProtocolError, "running agent.*wait"):
            await orchestrator.wait("/root", [])

    async def test_long_child_result_is_delivered_within_message_limit(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call("inbox", {"action": "inbox"}),
                    ModelResponse("root"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("x" * 100)]),
        }
        orchestrator = orchestrator_for(models, max_message_bytes=32)
        await orchestrator.run("lead")
        wait_status = json.loads(
            models["/root"].queries[2][0][-1].tool_results[0].output
        )
        self.assertEqual(
            wait_status, [{"agent_id": "/root/1", "status": "completed"}]
        )
        inbox = json.loads(models["/root"].queries[3][0][-1].tool_results[0].output)
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["kind"], "result")
        self.assertLessEqual(len(inbox[0]["content"].encode()), 32)
        self.assertTrue(inbox[0]["content"].endswith("[truncated]"))

    async def test_adopt_is_explicit_and_descendant_only(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "branch"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call("adopt", {"action": "adopt", "agent_id": "/root/1"}),
                    ModelResponse("selected"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("candidate")]),
        }
        environments: dict[str, IsolatedEnvironment] = {}
        orchestrator = orchestrator_for(models, environments=environments)
        await orchestrator.run("lead")
        self.assertEqual(environments["/root"].adoptions, ["/root/1"])
        self.assertEqual(orchestrator.records["/root"].state, "/root/1")
        self.assertEqual(orchestrator.records["/root"].adopted_from, "/root/1")
        self.assertEqual(
            orchestrator.records["/root"].adoption_history, ["/root/1"]
        )
        with self.assertRaisesRegex(ProtocolError, "descendant"):
            await orchestrator.adopt("/root/1", "/root")

    async def test_stop_cancels_a_descendant_subtree_and_frees_capacity(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn-one", {"action": "spawn", "task": "child"}),
                    call("receive", {"action": "inbox", "wait": True}),
                    call("stop", {"action": "stop", "agent_id": "/root/1"}),
                    call("spawn-two", {"action": "spawn", "task": "replacement"}),
                    call("wait-two", {"action": "wait", "agent_ids": ["/root/2"]}),
                    ModelResponse("root"),
                ]
            ),
            "/root/1": ScriptThenBlockModel(
                [
                    call("spawn-grandchild", {"action": "spawn", "task": "leaf"}),
                    call(
                        "ready",
                        {
                            "action": "send",
                            "agent_id": "/root",
                            "message": "subtree ready",
                        },
                    ),
                ]
            ),
            "/root/1/1": BlockingModel(),
            "/root/2": ScriptedModel([ModelResponse("replacement")]),
        }
        environments: dict[str, IsolatedEnvironment] = {}
        orchestrator = orchestrator_for(
            models,
            environments=environments,
            max_active=3,
            max_total=4,
        )
        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "root")
        self.assertEqual(orchestrator.records["/root/1"].status, "cancelled")
        self.assertEqual(orchestrator.records["/root/1/1"].status, "cancelled")
        self.assertEqual(orchestrator.records["/root/2"].status, "completed")
        self.assertTrue(
            all(environment.closed for environment in environments.values())
        )
        stopped = next(
            tool_result
            for message in result.messages
            for tool_result in message.tool_results
            if tool_result.call_id == "stop"
        )
        self.assertEqual(
            json.loads(stopped.output)["agents"],
            [
                {"agent_id": "/root/1", "status": "cancelled"},
                {"agent_id": "/root/1/1", "status": "cancelled"},
            ],
        )
        events = [event.event for event in orchestrator.context.trace.events]
        self.assertIn("agent_stop_requested", events)
        self.assertIn("agent_subtree_stopped", events)

    async def test_send_to_terminal_agent_is_a_recoverable_invalid_action(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call(
                        "late-send",
                        {
                            "action": "send",
                            "agent_id": "/root/1",
                            "message": "too late",
                        },
                    ),
                    ModelResponse("recovered"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("done")]),
        }
        result = await orchestrator_for(models).run("lead")
        invalid = next(
            tool_result
            for message in result.messages
            for tool_result in message.tool_results
            if tool_result.call_id == "late-send"
        )
        self.assertTrue(invalid.is_error)
        self.assertIn("already terminal", invalid.output)

    async def test_active_limit_is_a_recoverable_invalid_action(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("one", {"action": "spawn", "task": "one"}),
                    call("two", {"action": "spawn", "task": "two"}),
                    ModelResponse("recovered"),
                ]
            ),
            "/root/1": BlockingModel(),
        }
        orchestrator = orchestrator_for(models, max_active=2, max_total=3)
        result = await orchestrator.run("lead")
        invalid = result.messages[-2].tool_results[0]
        self.assertTrue(invalid.is_error)
        self.assertIn("active agent limit", invalid.output)
        self.assertEqual(result.answer, "recovered")
        self.assertEqual(orchestrator.records["/root/1"].status, "cancelled")

    async def test_total_limit_applies_after_a_child_finishes(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("one", {"action": "spawn", "task": "one"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    call("two", {"action": "spawn", "task": "two"}),
                    ModelResponse("done"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("one")]),
        }
        orchestrator = orchestrator_for(models, max_active=2, max_total=2)
        result = await orchestrator.run("lead")
        invalid = result.messages[-2].tool_results[0]
        self.assertTrue(invalid.is_error)
        self.assertIn("total agent limit", invalid.output)

    async def test_invalid_arguments_are_returned_to_the_model(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call(
                        "bad",
                        {"action": "inbox", "message": "not accepted"},
                    ),
                    ModelResponse("fixed"),
                ]
            )
        }
        result = await orchestrator_for(models).run("lead")
        self.assertEqual(result.answer, "fixed")
        self.assertTrue(result.messages[-2].tool_results[0].is_error)
        self.assertIn("unexpected arguments", result.messages[-2].content)

    async def test_resource_reuse_is_a_hard_isolation_failure(self) -> None:
        models = {
            "/root": ScriptedModel(
                [call("spawn", {"action": "spawn", "task": "child"})]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        with self.assertRaisesRegex(ValueError, "resource identity"):
            await orchestrator_for(models, shared_identity=True).run("lead")

    async def test_close_failure_changes_completion_to_failure(self) -> None:
        models = {"/root": ScriptedModel([ModelResponse("answer")])}
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            await orchestrator_for(models, fail_root_close=True).run("lead")

    async def test_cancelled_provisioning_is_awaited_then_closed(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        environment = IsolatedEnvironment("/root")

        async def environment_factory(agent_id: str) -> IsolatedEnvironment:
            del agent_id
            started.set()
            await release.wait()
            return environment

        orchestrator = Orchestrator(
            agent_builder=lambda agent_id, wrapped, context: MiniAgent(
                model=ScriptedModel([ModelResponse("unused")]),
                environment=wrapped,
                context=context,
                agent_id=agent_id,
            ),
            environment_factory=environment_factory,
            context=RunContext(),
        )
        running = asyncio.create_task(orchestrator.run("lead"))
        await started.wait()
        running.cancel()
        await asyncio.sleep(0)
        self.assertFalse(running.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertTrue(environment.closed)

    async def test_terminal_trace_failure_changes_completion_to_failure(self) -> None:
        class BrokenTrace(TraceRecorder):
            async def emit(self, event: str, **kwargs: Any) -> None:
                if event == "agent_completed":
                    raise RuntimeError("terminal trace failed")
                await super().emit(event, **kwargs)

        environment = IsolatedEnvironment("/root")
        context = RunContext(trace=BrokenTrace())
        orchestrator = Orchestrator(
            agent_builder=lambda agent_id, wrapped, shared: MiniAgent(
                model=ScriptedModel([ModelResponse("answer")]),
                environment=wrapped,
                context=shared,
                agent_id=agent_id,
            ),
            environment_factory=lambda agent_id: environment,
            context=context,
        )
        with self.assertRaisesRegex(RuntimeError, "terminal trace failed"):
            await orchestrator.run("lead")
        self.assertEqual(orchestrator.records["/root"].status, "failed")
        self.assertTrue(environment.closed)

    async def test_child_terminal_trace_failure_fails_the_overall_run(self) -> None:
        class BrokenTrace(TraceRecorder):
            async def emit(self, event: str, **kwargs: Any) -> None:
                if event == "agent_completed" and kwargs.get("agent_id") == "/root/1":
                    raise RuntimeError("child terminal trace failed")
                await super().emit(event, **kwargs)

        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    ModelResponse("root would otherwise succeed"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        orchestrator = orchestrator_for(models)
        orchestrator.context.trace = BrokenTrace()
        with self.assertRaisesRegex(
            RuntimeError, "agent terminal handling failed.*child terminal trace failed"
        ):
            await orchestrator.run("lead")
        self.assertIsInstance(
            orchestrator.records["/root/1"].terminal_error, RuntimeError
        )

    async def test_child_result_delivery_trace_failure_fails_the_run(self) -> None:
        class BrokenTrace(TraceRecorder):
            async def emit(self, event: str, **kwargs: Any) -> None:
                if event == "message_sent" and kwargs.get("agent_id") == "/root/1":
                    raise RuntimeError("child delivery trace failed")
                await super().emit(event, **kwargs)

        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    ModelResponse("root would otherwise succeed"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        orchestrator = orchestrator_for(models)
        orchestrator.context.trace = BrokenTrace()
        with self.assertRaisesRegex(
            RuntimeError, "agent terminal handling failed.*child delivery trace failed"
        ):
            await orchestrator.run("lead")

    async def test_start_failure_retains_trace_failure(self) -> None:
        class BrokenTrace(TraceRecorder):
            async def emit(self, event: str, **kwargs: Any) -> None:
                if event == "agent_start_failed":
                    raise RuntimeError("trace exploded")
                await super().emit(event, **kwargs)

        environment = IsolatedEnvironment("/root")
        context = RunContext(trace=BrokenTrace())
        orchestrator = Orchestrator(
            agent_builder=lambda agent_id, wrapped, shared: object(),
            environment_factory=lambda agent_id: environment,
            context=context,
        )
        with self.assertRaisesRegex(
            RuntimeError, "agent_builder.*failure trace.*trace exploded"
        ):
            await orchestrator.run("lead")
        self.assertTrue(environment.closed)

    async def test_descendant_cleanup_failure_fails_the_run(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    ModelResponse("root"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        with self.assertRaisesRegex(RuntimeError, "environment cleanup failed"):
            await orchestrator_for(models, fail_close_ids={"/root/1"}).run("lead")

    async def test_root_and_descendant_failures_are_both_reported(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "scripted model has no response left.*cleanup also failed.*close failed",
        ):
            await orchestrator_for(models, fail_close_ids={"/root/1"}).run("lead")

    async def test_per_agent_budget_includes_root(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("inbox", {"action": "inbox"}),
                    ModelResponse("never"),
                ]
            )
        }
        orchestrator = orchestrator_for(
            models,
            per_agent_limits=BudgetLimits(max_model_calls=1),
        )
        with self.assertRaisesRegex(BudgetExceeded, "agent model-call"):
            await orchestrator.run("lead")

    async def test_wait_rejects_non_descendants(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    call("spawn", {"action": "spawn", "task": "child"}),
                    call("wait", {"action": "wait", "agent_ids": ["/root/1"]}),
                    ModelResponse("root"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("child")]),
        }
        orchestrator = orchestrator_for(models)
        await orchestrator.run("lead")
        with self.assertRaisesRegex(ProtocolError, "descendants"):
            await orchestrator.wait("/root/1", ["/root"])

    async def test_stop_rejects_non_descendants(self) -> None:
        orchestrator = Orchestrator(
            agent_builder=lambda *args: None,  # type: ignore[arg-type,return-value]
            environment_factory=lambda *args: None,
            context=RunContext(),
        )
        await orchestrator._reserve(None, "/root")
        child = await orchestrator._reserve("/root")
        with self.assertRaisesRegex(ProtocolError, "descendant subtree"):
            await orchestrator.stop(child.agent_id, "/root")

    def test_agent_ids_and_limits_fail_closed(self) -> None:
        models = {"/root": ScriptedModel([ModelResponse("x")])}
        with self.assertRaises(ValueError):
            Orchestrator(
                agent_builder=lambda *_: None,  # type: ignore[arg-type]
                environment_factory=lambda *_: None,
                context=RunContext(),
                root_id="/bad\nid",
            )
        with self.assertRaises(ValueError):
            orchestrator_for(models, max_active=3, max_total=2)


if __name__ == "__main__":
    unittest.main()


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    """Each harness's structural contract, which is what makes them comparable.

    These assert shape rather than quality: how many agents exist and when,
    which tools each role holds, whether a call blocks, and whether an agent
    survives answering. Those are the properties an experiment varies.
    """

    async def test_a_fixed_team_exists_before_the_lead_takes_a_turn(self) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("fixed-team")
        peers = ("/root/peer-2", "/root/peer-3")
        models: dict[str, Any] = {
            "/root": ScriptedModel(
                [
                    call("inbox", {"action": "inbox", "wait": True}),
                    ModelResponse("lead answer"),
                ]
            )
        }
        for peer in peers:
            models[peer] = ScriptThenBlockModel(
                [
                    call(
                        "s",
                        {
                            "action": "send",
                            "agent_id": "/root",
                            "message": f"{peer} up",
                        },
                    ),
                    ModelResponse(f"{peer} done"),
                ]
            )
        orchestrator = orchestrator_for(models, max_active=3, max_total=3,
                                        harness=harness)

        result = await orchestrator.run(
            "shared task", seeds=harness.seeds(size=3, task="shared task")
        )

        self.assertEqual(result.answer, "lead answer")
        self.assertEqual(sorted(orchestrator.records), ["/root", *peers])
        # Peers hold the same tools as the lead and cannot spawn.
        for peer in peers:
            environment = orchestrator.records[peer].environment
            assert environment is not None
            actions = environment.tools()[-1].input_schema["properties"]["action"]
            self.assertEqual(actions["enum"], ["inbox", "send"])

    async def test_delegation_blocks_and_returns_the_subagent_answer(self) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("orchestrator")
        models = {
            "/root": ScriptedModel(
                [
                    call("d", {"action": "delegate", "task": "do the work"}),
                    ModelResponse("orchestrator answer"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse("subagent answer")]),
        }
        orchestrator = orchestrator_for(models, harness=harness)

        result = await orchestrator.run("coordinate")

        self.assertEqual(result.answer, "orchestrator answer")
        delegate_output = models["/root"].queries[1][0][-1].tool_results[0].output
        self.assertEqual(json.loads(delegate_output)["answer"], "subagent answer")
        self.assertEqual(orchestrator.records["/root/1"].status, "completed")
        # The orchestrator holds no domain tools, so it can only delegate.
        environment = orchestrator.records["/root"].environment
        assert environment is not None
        self.assertEqual([tool.name for tool in environment.tools()], ["agent"])

    async def test_an_async_subagent_idles_then_resumes_then_releases(self) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("async-subagents")
        models = {
            "/root": ScriptedModel(
                [
                    call("sp", {"action": "spawn", "task": "first"}),
                    call("in", {"action": "inbox", "wait": True}),
                    call(
                        "sd",
                        {"action": "send", "agent_id": "/root/1", "message": "second"},
                    ),
                    call("in2", {"action": "inbox", "wait": True}),
                    call("rel", {"action": "release", "agent_id": "/root/1"}),
                    ModelResponse("lead answer"),
                ]
            ),
            "/root/1": ScriptedModel(
                [ModelResponse("first done"), ModelResponse("second done")]
            ),
        }
        orchestrator = orchestrator_for(models, harness=harness)

        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "lead answer")
        first = json.loads(models["/root"].queries[2][0][-1].tool_results[0].output)
        second = json.loads(models["/root"].queries[4][0][-1].tool_results[0].output)
        self.assertEqual(first[0]["content"], "first done")
        self.assertEqual(second[0]["content"], "second done")
        # It answered twice on one conversation, then ended cleanly enough to
        # have exported state -- which `stop` would have discarded.
        record = orchestrator.records["/root/1"]
        self.assertEqual(record.status, "completed")
        self.assertIsNotNone(record.state)


class ReleaseGuardTests(unittest.IsolatedAsyncioTestCase):
    """Release retires an idle agent; it must not interrupt a working one."""

    async def test_releasing_a_busy_agent_is_refused(self) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("async-subagents")
        models = {
            "/root": ScriptedModel(
                [
                    call("sp", {"action": "spawn", "task": "work"}),
                    call("rel", {"action": "release", "agent_id": "/root/1"}),
                    ModelResponse("lead answer"),
                ]
            ),
            # Never answers, so it is running rather than idle when released.
            "/root/1": BlockingModel(),
        }
        orchestrator = orchestrator_for(models, harness=harness)

        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "lead answer")
        release_output = models["/root"].queries[2][0][-1].tool_results[0].output
        self.assertIn("not idle", release_output)

    async def test_releasing_an_agent_that_is_not_a_descendant_is_refused(
        self,
    ) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("async-subagents")
        models = {
            "/root": ScriptedModel(
                [
                    call("rel", {"action": "release", "agent_id": "/root"}),
                    ModelResponse("lead answer"),
                ]
            ),
        }
        orchestrator = orchestrator_for(models, harness=harness)

        await orchestrator.run("lead")

        release_output = models["/root"].queries[1][0][-1].tool_results[0].output
        self.assertIn("descendant", release_output)
