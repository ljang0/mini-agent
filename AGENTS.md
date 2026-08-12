# Agent quick reference

Design rules, naming-honesty requirements, and the contribution process live in
[CONTRIBUTING.md](CONTRIBUTING.md). Read it before changing `src/`.

Verification (run before proposing any change):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
PYTHONPATH=src python3 -m mini_agent profile \
  --application web --profile default --model openai/test-model
```

Python 3.10 through 3.13 is required; an unqualified `python3` below 3.10 will
fail. A passing suite is internal QA, not evidence of benchmark quality.
