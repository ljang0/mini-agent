# Contributing

`mini-agent` keeps one domain-neutral inference loop. Domain behavior belongs in
environments, provider behavior in model adapters, benchmark behavior in eval
modules, and communication in the orchestrator. Do not add planners, topology
classes, graders, or benchmark-specific branches to `agent.py`.

Profiles must be executable configuration: every behavior field must alter a
runtime component or be rejected. Label a normalized implementation as a
baseline or profile; use a pinned external reference when an upstream runtime
owns a materially different loop.

Before opening a change, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
ruff check src tests
mypy src/mini_agent
python3 -m build
twine check dist/*
```

New scheduling, messaging, stopping, tool, and artifact behavior needs a
deterministic offline test. Paid smoke runs are internal QA, not benchmark
claims.
