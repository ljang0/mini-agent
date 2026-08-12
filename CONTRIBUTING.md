# Contributing

Keep one domain-neutral inference loop. Domain behavior belongs in environments,
provider translation in codecs, evaluation behavior in benchmark adapters, and
communication in the orchestrator. Do not add benchmark branches, provider
conditionals, planners, or topology classes to `agent.py`.

Provider-specific prompts and harness behavior should normally be downstream.
The maintained `default` profiles are intentionally small baselines, not a catalog
of copied agents.

Every new run or evaluation harness must use `RunContext`'s shared budget ledger
and trace recorder. New scheduling, messaging, stopping, invalid-action, timeout,
cleanup, resume, or artifact behavior needs a deterministic offline test before
any paid model run.

Preserve research labels:

- inference harnesses, training methods, and evaluation environments are distinct;
- parallel samples are not best-of-N without a selector;
- recursive delegation is not a faithful recursive-agent-optimization
  reproduction without that method's policy training;
- JSON search over external context is not a research language model (RLM)
  or a REPL; and
- proprietary team variants remain topology simulations until their tools,
  limits, compaction, message timing, and worktree behavior are reproduced.

Run the repository contract before opening a change:

```bash
python3 --version  # must be 3.10 through 3.13
python3 -m pip install -e '.[dev]'
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
PYTHONPATH=src python3 -m mini_agent profile \
  --application web --profile default --model openai/test-model
```

## Proposing a change

Open an issue first for anything larger than a bug fix so the design fit can be
discussed before you write code. Then fork, branch, and open a pull request
against `main` with the checklist in the PR template completed. There is no CLA;
contributions are accepted under the repository's Apache-2.0 license, and every
change must keep the verification commands above green on Python 3.10 through
3.13 (CI runs the full matrix).

For release changes, also build and install both artifacts in clean environments:

```bash
umask 0022  # archive members must be readable by downstream installers
python3 -m build
python3 -m twine check --strict dist/*
```

A passing smoke test is internal QA. Report benchmark results only with the exact
model, prompt, tool protocol, limits, dataset revision, evaluator, environment,
sample size, and artifact provenance.
