# mini-agent

[![CI](https://github.com/ljang0/mini-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ljang0/mini-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-agent-cmu)](https://pypi.org/project/mini-agent-cmu/)
[![Python](https://img.shields.io/pypi/pyversions/mini-agent-cmu)](https://pypi.org/project/mini-agent-cmu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

One small agent loop, and a way to compare how you coordinate several of them.

The loop handles three environment families — software engineering, web
research, and computer use. Everything else (provider codecs, benchmark
loaders, graders, storage, scheduling) stays outside it, so a multi-agent
experiment changes the topology and nothing else.

Alpha software. It is infrastructure for controlled experiments, not a claim
that one normalized prompt reproduces any provider's proprietary harness.

## Install

```bash
pip install mini-agent-cmu
```

The distribution is `mini-agent-cmu` — PyPI blocks the bare name as too similar
to an unrelated project — but the import is `mini_agent` and the command is
`mini-agent`. Python 3.10–3.13.

## Run something

No VM, no benchmark checkout, no search key — retrieval is deterministic
in-process BM25 over a local file:

```bash
export OPENAI_API_KEY=...
mini-agent run --environment web --web-backend jsonl \
  --corpus examples/corpus.jsonl \
  --task 'Which document explains when the wind turbine furls?' \
  --model openai/MODEL
```

The run prints its output directory on start; `tail -f <output>/trace.jsonl`
streams every model and tool event live.

Or drive the loop as a library with no API key at all —
[`examples/library_quickstart.py`](examples/library_quickstart.py) runs as is:

```python
import asyncio
from mini_agent import MiniAgent, RunContext, ScriptedModel, ModelResponse

agent = MiniAgent(
    model=ScriptedModel([ModelResponse("done")]),  # or build_model("openai/MODEL")
    environment=my_environment,                    # any BaseEnvironment
    system_prompt="Solve the task with the available tools.",
    context=RunContext(),
)
result = asyncio.run(agent.run("the task"))
```

## Run several agents

`--harness` picks a named topology, so two runs can differ only in how the
agents coordinate:

```bash
mini-agent eval --benchmark programbench --harness fixed-team --team-size 3 ...
```

| Harness | Shape |
|---|---|
| `single` | one agent |
| `fixed-team` | 3, 5, or 10 peers sharing the task, one designated lead |
| `orchestrator` | a coordinator with no task tools, delegating to blocking subagents |
| `async-subagents` | long-lived subagents that idle between instructions |
| `message-board` | peers sharing one append-only log instead of mailboxes |
| `recursive` | the free-form mesh `--multi-agent` selects |

Every result records what each agent spent and how much of it went into
coordination — messages, idle time, and work two agents did twice. See
[docs/harnesses.md](docs/harnesses.md).

`mini-agent report --runs A B C` puts those runs side by side in one table, so
the comparison the harnesses exist for is a command rather than a spreadsheet.

## Benchmarks

SWE-bench, ProgramBench, BrowseComp, BrowseComp-Plus, and OSWorld v1/v2, each
pinned to an exact upstream revision and image digest. Generation happens here;
scoring is always the official grader's job.

Four competition-mathematics sets — AIME, MATH-500, OlympiadBench, MinervaMath —
are the exception in both directions: they need no container, index, or VM, and
their score is computed locally because these datasets publish problems and
answers rather than an evaluation harness. That makes them comparable across
`mini-agent` runs, not against a published leaderboard.

The exact invocation for each is in [docs/cli.md](docs/cli.md); what is pinned
and where this deviates from upstream is in
[docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [CLI reference](docs/cli.md) — every command, storage layout, environment
  variables, and what a result means
- [Harnesses](docs/harnesses.md) — the topologies, adding one, capacity limits
- [Architecture](docs/architecture.md) — the loop, environments, accounting
- [Library usage](docs/library.md) — custom environments, budgets, spec binding
- [Benchmark fidelity](docs/benchmarks.md) — pins and intentional differences
- [Validation](docs/validation.md) — what has actually been run, and what has not

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m ruff check src tests
python -m mypy src/mini_agent
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the naming-honesty rules this project
holds itself to: a claim in a name or a docstring has to be one the code keeps.
