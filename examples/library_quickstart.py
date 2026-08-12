"""Run the mini-agent loop offline: no API key, no network, no assets.

python examples/library_quickstart.py
"""

import asyncio
from typing import Sequence

from mini_agent import (
    BudgetLimits,
    MiniAgent,
    ModelResponse,
    RunContext,
    ScriptedModel,
    ToolCall,
    ToolDefinition,
    ToolExecution,
)
from mini_agent.environments.base import BaseEnvironment


class GreeterEnvironment(BaseEnvironment):
    """A minimal custom environment: one tool, one behavior."""

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                "greet",
                "Greet someone by name.",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            ),
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        return ToolExecution(f"Hello, {action.arguments['name']}!")


async def main() -> None:
    # ScriptedModel stands in for a real provider; swap in
    # mini_agent.models.build_model("openai/<model>") for live runs.
    model = ScriptedModel(
        [
            ModelResponse(
                "",
                tool_calls=(ToolCall("call-1", "greet", {"name": "world"}),),
            ),
            ModelResponse("The environment replied with a greeting."),
        ]
    )
    context = RunContext(limits=BudgetLimits(max_model_calls=4, max_tool_calls=4))
    agent = MiniAgent(
        model=model,
        environment=GreeterEnvironment(),
        system_prompt="Use the greet tool, then summarize.",
        max_steps=4,
        context=context,
    )
    result = await agent.run("Greet the world.")
    print("answer:", result.answer)
    print("steps:", result.steps)
    print("accounting:", dict(context.ledger.snapshot()))


if __name__ == "__main__":
    asyncio.run(main())
