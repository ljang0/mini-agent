from __future__ import annotations

import asyncio
import unittest
from typing import Any, Sequence

from mini_agent import (
    BudgetExceeded,
    BudgetLimits,
    MiniAgent,
    ModelResponse,
    Orchestrator,
    RunContext,
    ScriptedModel,
    ToolCall,
    ToolDefinition,
)
from mini_agent.environments.base import BaseEnvironment
from mini_agent.types import ToolExecution


class _IsolatedEnvironment(BaseEnvironment):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition(name="identity"),)

    async def execute(self, action: ToolCall) -> ToolExecution:
        return ToolExecution(output=self.agent_id)

    async def close(self) -> None:
        self.closed = True


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        text="", tool_calls=(ToolCall(call_id, name, arguments),)
    )


class MiniOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stable_resource_identity_rejects_distinct_shared_wrappers(self) -> None:
        class Shared(_IsolatedEnvironment):
            def resource_identity(self) -> str:
                return "swe-workspace:/same"

        root = ScriptedModel([_call("spawn", "spawn_agent", {"task": "child"})])

        def builder(
            agent_id: str, env: Any, shared: RunContext, profile: str | None
        ) -> MiniAgent:
            del profile
            return MiniAgent(
                model=root,
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: Shared(agent_id),
            context=RunContext(BudgetLimits(max_model_calls=4, max_tool_calls=4)),
            max_agents=2,
        )
        with self.assertRaisesRegex(ValueError, "resource identity"):
            await orchestrator.run("lead")

    async def test_child_profile_allowlist_is_enforced_as_an_observation(self) -> None:
        root = ScriptedModel(
            [
                _call(
                    "spawn",
                    "spawn_agent",
                    {"task": "child", "profile": "not-allowed"},
                ),
                ModelResponse(text="recovered"),
            ]
        )

        def builder(
            agent_id: str, env: Any, shared: RunContext, profile: str | None
        ) -> MiniAgent:
            return MiniAgent(
                model=root,
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        result = await Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: _IsolatedEnvironment(agent_id),
            context=RunContext(BudgetLimits(max_model_calls=4, max_tool_calls=4)),
            max_agents=2,
            allowed_child_profiles=("approved",),
        ).run("lead")
        self.assertEqual(result.answer, "recovered")
        self.assertIn("not allowlisted", result.messages[-2].content)

    async def test_spawn_peer_message_wait_and_root_only_submission(self) -> None:
        environments: dict[str, _IsolatedEnvironment] = {}
        models = {
            "/root": ScriptedModel(
                [
                    _call("spawn", "spawn_agent", {"task": "research"}),
                    _call("wait", "wait", {"agent_ids": ["/root/1"]}),
                    _call("read", "read_messages", {}),
                    ModelResponse(text="root answer"),
                ]
            ),
            "/root/1": ScriptedModel(
                [
                    _call(
                        "message",
                        "send_message",
                        {"agent_id": "/root", "message": "working"},
                    ),
                    ModelResponse(text="child answer"),
                ]
            ),
        }
        context = RunContext(BudgetLimits(max_model_calls=8, max_tool_calls=8))

        async def environment_factory(
            agent_id: str, _profile: str | None
        ) -> _IsolatedEnvironment:
            environment = _IsolatedEnvironment(agent_id)
            environments[agent_id] = environment
            return environment

        def agent_builder(
            agent_id: str, environment: Any, shared: RunContext, profile: str | None
        ) -> MiniAgent:
            del profile
            return MiniAgent(
                model=models[agent_id],
                environment=environment,
                context=shared,
                agent_id=agent_id,
                max_steps=6,
            )

        orchestrator = Orchestrator(
            agent_builder=agent_builder,
            environment_factory=environment_factory,
            context=context,
            max_agents=2,
        )
        result = await orchestrator.run("lead")

        self.assertEqual(result.answer, "root answer")
        self.assertEqual(orchestrator.records["/root/1"].result.answer, "child answer")
        self.assertEqual(set(environments), {"/root", "/root/1"})
        self.assertIsNot(environments["/root"], environments["/root/1"])
        self.assertTrue(all(environment.closed for environment in environments.values()))
        root_read = models["/root"].queries[3][0][-1].tool_results[0].output
        self.assertIn("working", root_read)
        self.assertIn("child answer", root_read)
        events = [event.event for event in context.trace.events]
        self.assertIn("agent_spawned", events)
        self.assertIn("message_sent", events)

    async def test_maximum_agent_count_is_a_recoverable_tool_error(self) -> None:
        models = {
            "/root": ScriptedModel(
                [
                    _call("first", "spawn_agent", {"task": "one"}),
                    _call("second", "spawn_agent", {"task": "two"}),
                    ModelResponse(text="done"),
                ]
            ),
            "/root/1": ScriptedModel([ModelResponse(text="one")]),
        }
        context = RunContext(BudgetLimits(max_model_calls=8, max_tool_calls=8))

        def builder(agent_id: str, env: Any, shared: RunContext, profile: str | None) -> MiniAgent:
            del profile
            return MiniAgent(
                model=models[agent_id],
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: _IsolatedEnvironment(agent_id),
            context=context,
            max_agents=2,
        )
        result = await orchestrator.run("lead")
        self.assertEqual(result.answer, "done")
        second_result = result.messages[-2].tool_results[0]
        self.assertTrue(second_result.is_error)
        self.assertIn("maximum agent count", second_result.output)
        self.assertEqual(set(orchestrator.records), {"/root", "/root/1"})

    async def test_child_failure_is_reported_without_becoming_root_submission(self) -> None:
        class Fails:
            tool_family = "generic"

            async def query(self, messages: Any, tools: Any) -> ModelResponse:
                del messages, tools
                raise RuntimeError("child exploded")

        root = ScriptedModel(
            [
                _call("spawn", "spawn_agent", {"task": "fail"}),
                _call("wait", "wait", {"agent_ids": ["/root/1"]}),
                _call("read", "read_messages", {}),
                ModelResponse(text="root recovered"),
            ]
        )
        context = RunContext(BudgetLimits(max_model_calls=8, max_tool_calls=8))

        def builder(agent_id: str, env: Any, shared: RunContext, profile: str | None) -> MiniAgent:
            del profile
            return MiniAgent(
                model=root if agent_id == "/root" else Fails(),
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: _IsolatedEnvironment(agent_id),
            context=context,
            max_agents=2,
        )
        result = await orchestrator.run("lead")
        self.assertEqual(result.answer, "root recovered")
        self.assertEqual(orchestrator.records["/root/1"].status, "failed")
        observed = root.queries[3][0][-1].content
        self.assertIn("child exploded", observed)

    async def test_all_agents_share_the_global_model_call_budget(self) -> None:
        root = ScriptedModel(
            [
                _call("spawn", "spawn_agent", {"task": "child"}),
                ModelResponse(text="would need a second call"),
            ]
        )
        child = ScriptedModel([ModelResponse(text="child")])
        context = RunContext(
            BudgetLimits(max_model_calls=1, max_concurrency=1, max_tool_calls=4)
        )

        def builder(agent_id: str, env: Any, shared: RunContext, profile: str | None) -> MiniAgent:
            del profile
            return MiniAgent(
                model=root if agent_id == "/root" else child,
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: _IsolatedEnvironment(agent_id),
            context=context,
            max_agents=2,
        )
        with self.assertRaisesRegex(BudgetExceeded, "model-call budget"):
            await orchestrator.run("lead")
        self.assertEqual(context.ledger.calls, 1)

    async def test_per_agent_budget_stops_root_before_global_budget(self) -> None:
        root = ScriptedModel(
            [
                _call("identity", "identity", {}),
                ModelResponse(text="never"),
            ]
        )
        context = RunContext(BudgetLimits(max_model_calls=8, max_tool_calls=8))

        def builder(
            agent_id: str, env: Any, shared: RunContext, profile: str | None
        ) -> MiniAgent:
            del profile
            return MiniAgent(
                model=root,
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=lambda agent_id, _profile: _IsolatedEnvironment(agent_id),
            context=context,
            per_agent_limits=BudgetLimits(max_model_calls=1, max_tool_calls=4),
        )
        with self.assertRaisesRegex(BudgetExceeded, "agent model-call budget"):
            await orchestrator.run("lead")
        self.assertEqual(context.ledger.calls, 1)
        self.assertEqual(context.ledger.agent_snapshot("/root")["model_calls"], 1)

    async def test_root_completion_cancels_and_closes_unfinished_children(self) -> None:
        class BlockingModel:
            tool_family = "generic"

            async def query(self, messages: Any, tools: Any) -> ModelResponse:
                del messages, tools
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        root = ScriptedModel(
            [
                _call("spawn", "spawn_agent", {"task": "keep working"}),
                ModelResponse(text="root is done"),
            ]
        )
        environments: dict[str, _IsolatedEnvironment] = {}
        context = RunContext(BudgetLimits(max_model_calls=4, max_tool_calls=4))

        def environment_factory(
            agent_id: str, _profile: str | None
        ) -> _IsolatedEnvironment:
            environment = _IsolatedEnvironment(agent_id)
            environments[agent_id] = environment
            return environment

        def builder(agent_id: str, env: Any, shared: RunContext, profile: str | None) -> MiniAgent:
            del profile
            return MiniAgent(
                model=root if agent_id == "/root" else BlockingModel(),
                environment=env,
                context=shared,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=builder,
            environment_factory=environment_factory,
            context=context,
            max_agents=2,
        )
        result = await orchestrator.run("lead")
        self.assertEqual(result.answer, "root is done")
        self.assertEqual(orchestrator.records["/root/1"].status, "cancelled")
        self.assertTrue(environments["/root/1"].closed)


if __name__ == "__main__":
    unittest.main()
