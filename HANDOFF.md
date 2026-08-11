# mini-agent handoff

## Current direction

The repository is now centered on one domain-neutral agent loop for SWE, web, and CUA. The previous Scaffold Lab v0.2 handoff is preserved at tag `scaffoldlab-v0.2-handoff`; the `scaffoldlab` Python package remains as a compatibility layer for its existing adapters and regression suite.

New work should use:

- Package: `mini_agent`
- CLI: `mini-agent`
- Core loop: [`src/mini_agent/agent.py`](src/mini_agent/agent.py)
- Communication wrapper: [`src/mini_agent/orchestrator.py`](src/mini_agent/orchestrator.py)
- Profiles: [`src/mini_agent/profiles`](src/mini_agent/profiles)
- Migrated catalog: [`src/mini_agent/catalog.py`](src/mini_agent/catalog.py)
- Exact references: [`src/mini_agent/references.py`](src/mini_agent/references.py)

## Implemented

- A roughly 100-line `MiniAgent` loop with linear history and no domain logic.
- Shared, concurrency-safe model/tool budgets and traces.
- Existing OpenAI, Anthropic, and OpenAI-compatible provider codecs behind `BackendModel`.
- First-turn image input plus screenshot tool-result continuation.
- One-tool SWE environment with per-agent workspace copies and stateless bash actions.
- BrowseComp-Plus Lucene adapter plus deterministic JSONL BM25 fixtures.
- cua-speed-run `observe/step/done` client and an evaluator-free OSWorld environment bridge.
- Resolved YAML profiles with prompts, tools, limits, generation/observation/history policy, source pins, fidelity labels, and gaps.
- Minimal multi-agent orchestration through `spawn_agent`, `send_message`, `read_messages`, and `wait`.
- Complete 1:1 catalog view: 55 profiles, 18 labs, and all 54 lab/application status cells.
- Exact delegation for all 18 runnable references through the preserved evaluator.
- `mini-agent` commands for catalog/frontier inspection and reference validation/evaluation.
- Deterministic offline tests for all new control-flow and environment boundaries.

## Fidelity boundaries

- `baseline` means a model in the wrapper.
- `profile` means published details were applied where the wrapper supports them; inspect `fidelity_gaps`.
- `reference` is reserved for executing a pinned upstream or hosted runtime.
- The 28 studies retain their original non-exact labels; the nine gaps remain unavailable.
- Reference runtimes own their loops and cannot be `MiniAgent` workers without changing the measured implementation.
- The BrowseComp-Plus JSONL backend is a deterministic test backend, not an official-score replacement for its Lucene index.
- The OSWorld bridge never invokes the evaluator; the outer benchmark runner retains termination and grading ownership.
- Training methods, benchmark datasets, graders, and verifiers remain separate from the inference harness.

## Next research work

1. Run single-agent pilots in the order SWE, BrowseComp-Plus, cua-speed-run.
2. Add benchmark-owned task loaders and graders without importing them into `MiniAgent`.
3. Add native model profiles only from pinned public sources and record unsupported behavior as gaps.
4. Run multi-agent experiments only after the matching single-agent profile passes deterministic and benchmark-level validation.
5. Compare communication policies without adding topology-specific agent classes unless results require them.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
PYTHONPATH=src python3 -m mini_agent profile \
  --application web --profile default --model openai/test-model
```
