# Scaffold Lab contributor guidance

- Preserve the distinction between inference harnesses, training methods, and
  evaluation environments.
- Never call `flat_parallel` best-of-N; a selector is required for best-of-N.
- Never call `recursive_delegation` faithful RAO without reproducing policy training.
- Never call `external_context_json_search` an RLM or REPL.
- Treat Fable/Opus variants as topology simulations until tools, limits, compaction,
  exact message timing, and worktrees are implemented.
- Every new harness must use the shared budget ledger and trace recorder.
- Add deterministic offline tests for scheduling, messaging, stopping, and invalid
  actions before running paid model experiments.
- A passing smoke test is internal QA, not evidence for a release winner.

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider openai-responses
```
