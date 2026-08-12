from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from mini_agent.agent import MiniAgent
from mini_agent.models import ScriptedModel
from mini_agent.runtime import RunContext
from mini_agent.types import (
    BudgetExceeded,
    ModelResponse,
    ProtocolError,
    ToolCall,
)

from support import EchoEnvironment, EmptyEnvironment


def echo_call(call_id: str, value: str) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=(ToolCall(call_id, "echo", {"value": value}),),
    )


class AgentInvariantTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_exhaustion_raises_instead_of_returning(self) -> None:
        model = ScriptedModel([echo_call("one", "a"), echo_call("two", "b")])
        agent = MiniAgent(
            model=model, environment=EchoEnvironment(), max_steps=2
        )
        with self.assertRaisesRegex(BudgetExceeded, r"max_steps \(2\)"):
            await agent.run("task")
        self.assertEqual(len(model.queries), 2)

    async def test_empty_response_without_tool_calls_is_a_protocol_error(self) -> None:
        agent = MiniAgent(
            model=ScriptedModel([ModelResponse("")]),
            environment=EchoEnvironment(),
        )
        with self.assertRaisesRegex(
            ProtocolError, "neither final text nor tool calls"
        ):
            await agent.run("task")

    async def test_text_alongside_tool_calls_does_not_terminate(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    text="thinking out loud",
                    tool_calls=(ToolCall("one", "echo", {"value": "a"}),),
                ),
                ModelResponse("final"),
            ]
        )
        environment = EchoEnvironment()
        result = await MiniAgent(model=model, environment=environment).run("task")
        self.assertEqual(result.answer, "final")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(environment.calls), 1)

    async def test_tool_calls_execute_sequentially_in_model_order(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    tool_calls=(
                        ToolCall("one", "echo", {"value": "first"}),
                        ToolCall("two", "echo", {"value": "second"}),
                    ),
                ),
                ModelResponse("done"),
            ]
        )
        environment = EchoEnvironment()
        result = await MiniAgent(model=model, environment=environment).run("task")
        self.assertEqual(
            [call.arguments["value"] for call in environment.calls],
            ["first", "second"],
        )
        tool_message = result.messages[-2]
        self.assertEqual(tool_message.role, "tool")
        self.assertEqual(tool_message.content, "first\nsecond")
        self.assertEqual(len(tool_message.tool_results), 2)

    async def test_finish_runs_on_success_but_not_on_exhaustion(self) -> None:
        environment = EchoEnvironment()
        await MiniAgent(
            model=ScriptedModel([ModelResponse("done")]),
            environment=environment,
        ).run("task")
        self.assertTrue(environment.finished)

        exhausted = EchoEnvironment()
        with self.assertRaises(BudgetExceeded):
            await MiniAgent(
                model=ScriptedModel([echo_call("one", "a")]),
                environment=exhausted,
                max_steps=1,
            ).run("task")
        self.assertFalse(exhausted.finished)

    async def test_result_carries_transcript_and_agent_identity(self) -> None:
        result = await MiniAgent(
            model=ScriptedModel([ModelResponse("answer")]),
            environment=EmptyEnvironment(),
            system_prompt="be brief",
            agent_id="/root/7",
        ).run("task")
        self.assertEqual(result.metadata, {"agent_id": "/root/7"})
        self.assertEqual(
            [message.role for message in result.messages],
            ["system", "user", "assistant"],
        )

    async def test_run_resets_message_history_between_invocations(self) -> None:
        agent = MiniAgent(
            model=ScriptedModel([ModelResponse("one"), ModelResponse("two")]),
            environment=EmptyEnvironment(),
        )
        first = await agent.run("task")
        second = await agent.run("task")
        self.assertEqual(len(first.messages), len(second.messages))

    def test_constructor_rejects_invalid_configuration(self) -> None:
        model = ScriptedModel([])
        environment = EmptyEnvironment()
        with self.assertRaisesRegex(ValueError, "max_steps"):
            MiniAgent(model=model, environment=environment, max_steps=0)
        with self.assertRaisesRegex(ValueError, "query"):
            MiniAgent(model=object(), environment=environment)
        with self.assertRaisesRegex(ValueError, "tools"):
            MiniAgent(model=model, environment=object())
        with self.assertRaisesRegex(ValueError, "agent_id"):
            MiniAgent(model=model, environment=environment, agent_id=" ")
        with self.assertRaisesRegex(ValueError, "context"):
            MiniAgent(model=model, environment=environment, context=object())

    async def test_run_rejects_blank_tasks(self) -> None:
        agent = MiniAgent(
            model=ScriptedModel([ModelResponse("unused")]),
            environment=EmptyEnvironment(),
        )
        with self.assertRaisesRegex(ValueError, "task"):
            await agent.run("  ")

    async def test_shared_context_is_used_verbatim(self) -> None:
        context = RunContext()
        agent = MiniAgent(
            model=ScriptedModel([ModelResponse("done")]),
            environment=EmptyEnvironment(),
            context=context,
        )
        await agent.run("task")
        self.assertIs(agent.context, context)
        self.assertEqual(context.ledger.calls, 1)


class AgentImportGraphTests(unittest.TestCase):
    def test_agent_module_imports_no_environments_or_transport(self) -> None:
        probe = (
            "import sys\n"
            "import mini_agent.agent\n"
            "loaded = set(sys.modules)\n"
            "assert 'httpx' not in loaded, 'httpx'\n"
            "assert not any(name.startswith('mini_agent.environments.')\n"
            "               and name != 'mini_agent.environments.base'\n"
            "               for name in loaded), sorted(loaded)\n"
        )
        repo_root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            (sys.executable, "-c", probe),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root / "src")},
            cwd=str(repo_root),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
