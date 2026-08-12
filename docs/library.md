# Using mini-agent as a library

The package ships typed (`py.typed`) with a deliberately small public surface:
`MiniAgent`, `RunContext`, `BudgetLimits`, `Orchestrator`, `ScriptedModel`,
`build_model`, `AgentSpecV1`, and the provider-neutral types
(`Message`, `ToolDefinition`, `ToolCall`, `ToolResult`, `ToolExecution`,
`ModelResponse`, `Usage`). Everything below is importable from `mini_agent`.

A complete runnable version of this page is
[`examples/library_quickstart.py`](../examples/library_quickstart.py) — it
needs no API key.

## Build a model

```python
from mini_agent import build_model

model = build_model(
    "anthropic/MODEL",              # or "openai/MODEL", "meta/MODEL"
    max_retries=3,                  # transient 408/429/5xx and transport errors
    timeout_seconds=300,
    # Transcript-replay codecs only (chat-completions / Anthropic Messages):
    # keep the newest K screenshots when replaying history (default 4).
    # max_history_images=4,
    expected_resolved_model="MODEL-SNAPSHOT",  # optional snapshot pin
)
```

Credentials come from environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `MODEL_API_KEY`), never from arguments. For deterministic
offline work, `ScriptedModel([ModelResponse(...)])` satisfies the same `Model`
protocol.

## Write an environment

An environment exposes `tools()` and `execute()`; subclass `BaseEnvironment`
for the optional lifecycle hooks (`initial_observation`, `finish`, `close`,
`export_state`, `adopt_state`):

```python
from typing import Sequence
from mini_agent import ToolCall, ToolDefinition, ToolExecution
from mini_agent.environments.base import BaseEnvironment


class GreeterEnvironment(BaseEnvironment):
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
```

Tool arguments are schema-validated before `execute` is called; raise
`InvalidAction` for model-repairable failures (the model sees the message) and
any other exception for infrastructure failures (the run stops).

## Run with budgets and read the accounting

```python
import asyncio
from mini_agent import BudgetLimits, MiniAgent, RunContext

context = RunContext(
    limits=BudgetLimits(
        max_model_calls=32,
        max_tool_calls=64,
        wall_time_seconds=600,
        # max_cost_usd requires pricing on the model backend.
    )
)
agent = MiniAgent(
    model=model,
    environment=GreeterEnvironment(),
    system_prompt="Use the greet tool, then summarize.",
    max_steps=16,
    context=context,
)
result = asyncio.run(agent.run("Greet the world."))
print(result.answer, result.steps)
print(context.ledger.snapshot())   # model/tool calls, tokens, cost
```

Crossing any budget raises `BudgetExceeded`. Passing a `TraceRecorder` to
`RunContext(trace=...)` streams a secret-redacted JSONL event log.

## Bind a portable spec (optional, recommended for evaluations)

`AgentSpecV1` freezes the domain, model identity, prompt, step cap, budget, and
tool surface into a canonical fingerprint; `spec.bind(...)` verifies the live
model/environment against it before constructing the agent — this is how the
CLI enforces its manifests:

```python
from mini_agent import AgentSpecV1, BudgetLimits

spec = AgentSpecV1(
    environment="web",
    model="anthropic/MODEL",
    profile="custom",
    system_prompt="Use evidence.",
    max_steps=16,
    budget=BudgetLimits(max_model_calls=32),
    tool_capabilities=("greet",),
    communication_capabilities=(),
)
agent = spec.bind(
    model=model,
    environment=GreeterEnvironment(),
    model_id="anthropic/MODEL",
    environment_id="web",
)
```

## Multi-agent

`Orchestrator` schedules ordinary `MiniAgent` workers behind one `agent` tool
(`spawn`/`send`/`inbox`/`wait`/`stop`/`adopt`); every child gets an isolated
environment from your factory and shares the same ledger. See
[architecture.md](architecture.md) for the contracts.
