# Architecture

## One loop

`MiniAgent` owns one linear sequence:

```text
task -> model -> tool actions -> observations -> model -> final answer
```

The model supplies `query(messages, tools)`. The environment supplies `tools()` and `execute(action)`. Optional `initial_observation`, `finish`, and `close` hooks support stateful CUA and resource cleanup without adding domain branches to the loop.

The loop lives entirely in `src/mini_agent/agent.py`. Provider parsing, benchmark loading, grading, planning, memory policies, and orchestration do not belong in that file.

## Accounting boundary

`RunContext` wraps model and tool operations with the existing concurrency-safe `BudgetLedger` and `TraceRecorder`. One context can be shared by a single agent or an entire orchestrated group. It accounts for:

- Model calls, tokens, known cost, concurrency, and wall time.
- Tool calls and tool-output bytes.
- Queued, started, completed, failed, and cancelled events.
- Communication lifecycle events when the orchestrator is active.

Invalid tool names and domain protocol errors become bounded error observations so a model may repair its next action. Unexpected environment/provider failures and cancellations propagate.

## Applications are environments

- `BashEnvironment` exposes one stateless bash process per call over a persistent isolated workspace copy.
- `WebEnvironment` exposes fixed-corpus search and optional document retrieval. The canonical backend wraps BrowseComp-Plus's Lucene index; pure-Python JSONL BM25 exists for deterministic offline tests.
- `CUAEnvironment` exposes one batched computer tool over `observe/step/done`. The client does not expose reset, snapshots, shell, expected answers, or verification.
- `OSWorldClient` adapts a live DesktopEnv through a narrow pyautogui/time action encoder and deliberately leaves evaluator ownership with the outer OSWorld runner.

Models may synthesize scripts only through the selected environment's legal tools. The framework never creates a new privileged tool from model output.

## Profiles, not copied loops

Profiles select the system prompt, legal tools, limits, provider generation settings, screenshot/coordinate policy, history policy, response parser, benchmark, and pinned source. The resolved manifest includes full prompt text and hashes.

Fidelity labels distinguish minimal baselines, source-informed wrapper profiles, and externally executed references. Unsupported source behavior must appear in `fidelity_gaps`.

## Catalog and reference boundary

`catalog.py` is an immutable mini-agent view of the complete prior audit. It maps
only application names (`browser` to `web`, `computer-use` to `cua`) and preserves
all 55 IDs, statuses, fidelity claims, source pins, exact components, and known
gaps. Its execution modes are deliberately narrow:

- `reference`: one of 18 exact public protocols or pinned upstream runtimes.
- `study`: one of 28 controlled reconstructions, simulations, or baselines.
- `unavailable`: one of nine source-backed gaps.

`references.py` delegates exact references to the preserved evaluator without a
shell and without translating their prompts, tools, continuation state, or
lifecycle. Provider, implementation, and config identity are checked before the
delegation. This path still uses the existing shared budget ledger and trace
recorder.

The same module exposes studies through a distinct `StudyRuntime`. It uses the
preserved evaluator but cannot serialize as a reference, and exact references
cannot serialize as studies. Both paths bind the UTF-8 configuration bytes by
SHA-256 and recheck the expected catalog key inside the evaluator before any
backend is constructed.

Reference runtimes are not `MiniAgent` workers. An upstream runtime that owns its
loop cannot be embedded in `MiniAgent.run` or spawned by its orchestrator without
changing the implementation being measured. Studies and gaps cannot resolve as
references.

## Multi-agent adds communication

`Orchestrator` wraps each base environment with four communication tools and starts the same `MiniAgent` class for every participant. It maintains agent IDs, inboxes, task status, a maximum-agent bound, shared accounting, cancellation, and root-only submission.

Each agent receives a separate environment instance from the caller's factory. Sharing is therefore explicit: immutable retrieval data can be shared safely, while SWE workspaces and CUA machines default to isolation.

The wrapper does not define planners, roles, teams, DAGs, debate rounds, or topology-specific agent classes. Those are experiment policies expressed through tasks, prompts, and communication unless evidence justifies another primitive.
