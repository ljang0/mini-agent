# Scaffold Lab handoff

Status date: 2026-08-11

Package version: `0.2.0`

Working branch at handoff: `codex/scaffold-lab-v0.2`

## Executive summary

Scaffold Lab is an application-first research runtime for comparing agent
scaffolds across browser, computer-use, and SWE tasks. The shared runtime,
provider adapters, source/CLI adapters, application catalogs, budget ledger,
trace recorder, local environments, and deterministic offline test suite are in
place. The repository can run generic JSONL experiments through supported APIs
and pinned external runtimes.

This is not yet a benchmark release. The canonical BrowserGym, OSWorld 2, and
SWE-bench loaders, resets, assets, and evaluators are not implemented, and no
paid controlled model comparison has been run. A passing smoke test establishes
internal plumbing quality, not a winning scaffold.

Start with the [main README](README.md), the [documentation index](docs/README.md),
and the [reference and benchmark runbook](docs/reference-and-benchmark-runbook.md).

## What is implemented

| Layer | Handoff state |
| --- | --- |
| Shared runtime | Provider-neutral model and harness interfaces, budget ledger, trace recorder, stopping/cancellation, tools, environment isolation, and result artifacts. |
| Harness families | Single-agent control, selector-backed best-of-N, distinct-task flat fan-out, blocking manager, fixed team, async subagents, MACU-like DAG, recursive inference, restricted RLM-like REPL, and pinned upstream wrappers. |
| API boundaries | OpenAI Responses, Anthropic Messages, OpenAI-compatible Responses/Chat, OpenAI hosted multi-agent, xAI hosted multi-agent, and Anthropic Managed Agents. |
| External runtimes | Browser-Use, MACU, RLM, Prime Agent, Grok Build, Kimi Code, OpenAI Codex, and the official Claude Code Agent Teams distribution. |
| Application surfaces | Browser, screenshot/action computer use, isolated SWE workspaces, and SWE combined with computer tools. |
| Generic evaluation | JSONL task loading, repeats, shuffled trials, local text/JSON evaluators, manifests, traces, Wilson intervals, cost fields, and bounded SWE patch export. |
| Offline QA | Deterministic coverage of scheduling, messaging, stopping, invalid actions, continuations, isolation, and accounting. |

The exact claim for every profile is deliberately narrow. An implementation is
either a published protocol boundary, a revision-pinned public source runtime,
or an audited official distribution. Reconstructed mechanisms and system-card
topologies remain studies. In particular:

- `flat_parallel` is distinct-task concurrency, not best-of-N;
- `recursive_delegation` and Platoon-shaped inference are not faithful RAO
  without the policy-training method and checkpoint;
- `external_context_json_search` is an ablation, not an RLM or REPL;
- Fable, Mythos, and Opus system-card profiles are topology simulations until
  their exact tools, limits, compaction, message timing, and worktrees exist.

The per-profile evidence, source pins, prerequisites, run modes, and limitations
are recorded in the [runbook](docs/reference-and-benchmark-runbook.md). The
[source audit](docs/source-audit.md) is the authority for fidelity caveats.

## Repository map

| Path | Purpose |
| --- | --- |
| [`browser/`](browser/) | Browser implementations, studies, gaps, and application notes. |
| [`computer-use/`](computer-use/) | Computer-use implementations, studies, gaps, and application notes. |
| [`swe/`](swe/) | SWE implementations, studies, gaps, and application notes. |
| [`src/scaffoldlab/`](src/scaffoldlab/) | Shared runtime, harnesses, backends, tools, environments, catalog, runner, and CLI. |
| [`benchmarks/`](benchmarks/) | Deterministic smoke task sets; these are not canonical frontier benchmarks. |
| [`configs/`](configs/) | Backward-compatible controls and smoke configurations. Release-facing work should select an application profile instead. |
| [`tests/`](tests/) | Offline unit and integration tests. |
| [`docs/`](docs/) | Architecture, evidence, experiment protocol, coverage, and benchmark runbook. |
| `runs/` | Ignored local outputs created by experiments. Do not commit credentials or sensitive traces. |

## Reproduce the handoff

Use Python 3.10 or newer. The baseline checks do not require paid model calls:

```bash
python3 -m pip install -e '.[dev]'
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider openai-responses
```

Then inspect the catalog:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli list-applications
PYTHONPATH=src python3 -m scaffoldlab.cli list-implementations --json
PYTHONPATH=src python3 -m scaffoldlab.cli list-studies
PYTHONPATH=src python3 -m scaffoldlab.cli list-gaps
PYTHONPATH=src python3 -m scaffoldlab.cli list-frontier-sources --json
```

The runbook contains the provider-specific credentials, pinned source checkouts,
runtime flags, and sandboxing requirements. External-runtime adapters intentionally
do not install arbitrary upstream dependencies during a trial.

## Known gaps and risks

1. **Canonical benchmark bridges:** BrowserGym, OSWorld 2, and SWE-bench need
   pinned task loaders, clean resets, official evaluators, exact assets/images,
   and contamination tests.
2. **Release evidence:** no paid, repeated, counterbalanced comparison has been
   run with pinned model snapshots and prices.
3. **Closed internals:** hosted OpenAI, Anthropic, xAI, and system-card systems
   expose only documented boundaries. Their private schedulers cannot be
   reconstructed 1:1 from public evidence.
4. **Whole-tree accounting:** several upstream and hosted systems expose only
   aggregate or lower-bound usage. Those runs must not be treated as
   equal-compute release evidence until accounting is complete or the comparison
   protocol explicitly accommodates the limitation.
5. **Sandbox strength:** in-process Python restrictions, command allowlists, and
   Playwright isolation are not security boundaries. Paid or adversarial runs
   need disposable workspaces or VMs, scoped credentials, and network controls.
6. **Dependency identity:** some caller-built runtimes validate source and lock
   evidence without proving every generated dependency or binary byte. Keep the
   published fidelity label and artifact manifest attached to every result.

## Recommended next milestone

Implement benchmark parity before adding more topology names:

1. add a benchmark adapter interface that keeps task loading, reset, and scoring
   separate from inference;
2. land one bridge each for BrowserGym, OSWorld 2, and SWE-bench with pinned
   revisions and deterministic failure/reset tests;
3. add CI for the full offline suite and catalog validation;
4. run small credentialed canaries for every direct API and external runtime;
5. freeze model IDs, prices, environment images, task order, seeds, and budgets;
6. run the controlled matrix defined in the
   [experiment protocol](docs/experiment-protocol.md), publish complete artifacts,
   and only then nominate a release candidate.

## Release acceptance checklist

- [ ] Every candidate resolves through the application registry and uses the
      shared budget ledger and trace recorder.
- [ ] Every exact claim has a source/protocol pin and a documented unavailable-
      internals boundary.
- [ ] All required offline checks pass from a clean checkout.
- [ ] The named benchmark's official reset and evaluator pass contamination and
      failure-path tests.
- [ ] Credentials, raw sensitive content, workspaces, and run outputs are absent
      from Git history.
- [ ] Live runs use pinned models, prices, code revision, dependencies, seeds,
      budgets, and task order.
- [ ] Whole-tree token/cost accounting is complete or explicitly excluded from
      equal-compute conclusions.
- [ ] Repeated scores, uncertainty intervals, failures, and full manifests are
      published; smoke results are not presented as a winner.

## Maintainer notes

- Keep the application-first top level: `browser/`, `computer-use/`, and `swe/`.
- Put new exact profiles under `implementations/`, reconstructions and ablations
  under `studies/`, and unavailable targets under `gaps/`.
- Do not silently widen a fidelity label when implementation details are missing.
- Add deterministic scheduling, messaging, stopping, accounting, and invalid-
  action tests before any paid experiment.
- Update the application README, catalog, runbook, source audit, and tests together
  when adding or reclassifying a profile.
