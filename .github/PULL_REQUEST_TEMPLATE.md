**What this changes and why**

**Checklist**

- [ ] `PYTHONPATH=src python3 -m unittest discover -s tests` passes (3.10–3.13)
- [ ] `python3 -m ruff check src tests` and `python3 -m mypy src/mini_agent` pass
- [ ] New behavior has deterministic offline tests
- [ ] No new branches in `src/mini_agent/agent.py` (see CONTRIBUTING.md)
- [ ] Docs updated where claims changed
