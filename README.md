# mini-agent

One small agent loop for controlled SWE, web, CUA, and multi-agent experiments.

The core idea is intentionally narrow: give a model a task and a small legal toolset, keep a linear history, and otherwise get out of the model's way. The same [`MiniAgent`](src/mini_agent/agent.py) runs every application. Applications are environments, model-specific behavior is a profile, and multi-agent execution adds only bounded tasks and communication tools.

```python
from mini_agent import MiniAgent, RunContext

agent = MiniAgent(
    model=model,
    environment=environment,
    system_prompt=system_prompt,
    max_steps=64,
    context=RunContext(),
)
result = await agent.run(task)
```

`agent.py` is the complete inference loop and is kept near 100 lines. It has no planner, memory framework, topology registry, domain branches, or dynamic privileged-tool registration.

## Applications

| Application | Minimal tools | Default reproducible boundary |
|---|---|---|
| SWE | `bash(command)` | Isolated workspace copy; one stateless shell per action |
| Web | `search(query)` | BrowseComp-Plus fixed local corpus and BM25 index |
| CUA | `computer(actions)` | cua-speed-run `observe/step/done` contract |

SWE follows the simplicity of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/tree/a83fcae82d2a08f0ee0c688f9d137b3566c097f8) and keeps full [SWE-agent](https://github.com/SWE-agent/SWE-agent/tree/3ea751c087f32b16e039a2233dd6eefecef325d5) as a reference boundary. Web uses [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402). CUA first targets [cua-speed-run](https://github.com/Pranjal2041/cua-speed-run/tree/7230223cbc57df68331cad32889adf01f3601651) and includes an evaluator-free bridge for [OSWorld](https://github.com/xlang-ai/OSWorld/tree/091f5ef1d5544bc74953c77875d5feb5bed30108).

## Install and run

```bash
python3 -m pip install -e .
```

Inspect the exact resolved profile before a run:

```bash
mini-agent profile --application web --profile default --model openai/gpt-5.4
```

Run SWE against an isolated copy of a repository:

```bash
mini-agent run \
  --application swe \
  --model openai/gpt-5.4 \
  --profile default \
  --workspace /path/to/repository \
  --task "Fix the failing test and verify the change" \
  --output runs/swe-example
```

Run web against either the canonical BrowseComp-Plus Lucene index or a small deterministic JSONL corpus:

```bash
mini-agent run \
  --application web \
  --model anthropic/claude-sonnet-4-6 \
  --profile anthropic \
  --index-path /path/to/browsecomp-plus-index \
  --task "Research the question and cite document IDs"
```

JSONL fixture documents use `{"docid": "...", "text": "..."}`. The pure-Python backend is for deterministic tests and small corpora; official BrowseComp-Plus measurements use its pinned Lucene index and Pyserini.

Run CUA through a cua-speed-run gateway:

```bash
mini-agent run \
  --application cua \
  --model openai/gpt-5.4 \
  --profile openai-gpt54 \
  --env-url http://127.0.0.1:8000 \
  --task "Complete the visible desktop task"
```

The CUA agent can access only its model and `observe`, `step`, and `done`. It cannot reset, snapshot, open a shell, or invoke the hidden verifier.

## Profiles and fidelity

Profiles live under [`src/mini_agent/profiles`](src/mini_agent/profiles). A resolved manifest records the complete system prompt, tool list, limits, provider settings, benchmark, source revision, and fidelity gaps.

- **Baseline:** a model running in the minimal wrapper.
- **Profile:** the wrapper configured from published prompt/tool/API details.
- **Reference:** a pinned upstream or hosted runtime executed outside the wrapper.

A profile is not automatically a reproduction of a proprietary product, trained search policy, or full upstream agent. Known differences stay machine-readable in `fidelity_gaps`.

## Multi-agent is communication

[`Orchestrator`](src/mini_agent/orchestrator.py) creates ordinary `MiniAgent` instances with separate environments and one shared budget/trace context. It adds four tools:

```text
spawn_agent(task, profile=None) -> agent_id
send_message(agent_id, message) -> acknowledgement
read_messages() -> messages
wait(agent_ids=None) -> statuses
```

Any known agent may message another. The wrapper owns only IDs, inboxes, async lifecycle, bounded spawning, cancellation, and root submission. Child results are delivered to their parent as messages; only `/root` produces the run result.

Each agent receives a separate environment instance from the caller's factory. Reusing an environment is rejected unless `allow_shared_environment=True` is explicitly selected. Immutable retrieval backends can still be shared behind separate lightweight environment objects, while SWE workspaces and CUA sessions default to isolation.

## Reproducibility and tests

Every model and tool call uses the shared budget ledger and trace recorder. `run.json` contains the resolved manifest and aggregate usage; `trace.jsonl` contains the event trace when `--output` is supplied.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
```

Deterministic tests cover history ordering, invalid actions, budgets, cancellation, workspace isolation, retrieval, CUA translation, hidden-verifier separation, spawning, messaging, waiting, failure, and root-only submission. Smoke tests validate plumbing; they are not evidence that an agent is a release winner.

## Repository transition

The former Scaffold Lab v0.2 state is preserved by the `scaffoldlab-v0.2-handoff` tag. The `scaffoldlab` package and CLI remain temporarily available for compatibility with the prior harness/reference catalog, but `mini_agent` and `mini-agent` are the authoritative interfaces for new work. See [`HANDOFF.md`](HANDOFF.md) and the [`documentation index`](docs/README.md).
