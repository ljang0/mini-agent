# mini-agent handoff

Version 0.3.0 is a breaking rebuild. `mini_agent` is self-contained and is the
only installed package and CLI. The pre-0.3 Scaffold Lab implementation is
preserved at tag `scaffoldlab-v0.2-handoff` and archived under
`research/scaffoldlab`; normal imports and wheels do not load it.

Implemented boundaries:

- `agent.py`: one 108-line domain-neutral inference loop.
- `runtime.py`: shared global/per-agent budgets, concurrency, cancellation, and
  content-addressed traces.
- SWE: safe local copies, private Git baseline/HOME, Docker SWE-bench instances,
  binary patches before cleanup, batch resume, predictions, and official grader.
- Web: inference-safe task loading, canonical BrowseComp-Plus retrieval policy,
  accounting, all eight published client families, official records and grader.
- CUA: validated PNG/actions/retries, coordinate translation/resizing, guaranteed
  `/done`, exact submission export with an embedded hash-checked 0.3.0 wheel and
  closed dependency-wheel set installed with `--no-index`, all 18 template
  mappings, and OSWorld runner. The pinned external runner owns the clock and
  verifier; its executable is required to live inside that checkout.
- Multi-agent: ordinary MiniAgents plus four communication tools, explicit
  allowlists, stable resource identities, shared/per-agent budgets, and root-only
  submission.
- References: read-only 18-lab catalog in the default package; exact old/upstream
  loops cross a lazy optional external-runtime boundary. Source checkouts may
  load the archived evaluator only when an explicit reference command runs;
  clean wheels require that preserved evaluator and the selected pinned runtime
  to be supplied separately.

Release gates are the root `tests` suite, compile, Ruff, Mypy, sdist/wheel build,
Twine metadata validation, clean-wheel installation, and CLI smokes. Real domain
smokes are intentionally conditional on `mini-agent doctor`; they are internal QA
and are not benchmark claims.

SWE and web batch generation accept `--mode multi` and keep the same official
artifact schemas. For CUA, mode is embedded by `mini-agent export`; the upstream
runner then executes that exact two-file submission. BrowseComp grading requires
a resolved local judge-model snapshot rather than a mutable hosted model name.

Future work should add new environments or profiles without copying the loop.
Only add a new primitive when an experiment cannot be expressed as prompt,
provider adapter, environment tool, or communication policy.
