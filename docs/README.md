# Scaffold Lab documentation

Use this page as the documentation entry point. The project deliberately keeps
inference harnesses, provider/runtime adapters, training methods, and evaluation
environments separate.

## Start here

| Document | Use it for |
| --- | --- |
| [Project handoff](../HANDOFF.md) | Current state, repository map, reproduction steps, risks, priorities, and release checklist. |
| [Main README](../README.md) | Installation, CLI usage, provider/runtime examples, safety notes, and output contract. |
| [Reference and benchmark runbook](reference-and-benchmark-runbook.md) | Per-profile source references, implementation status, prerequisites, run modes, and benchmark gaps. |

## Design and evidence

| Document | Authority |
| --- | --- |
| [Architecture](architecture.md) | Runtime boundaries, shared primitives, artifact flow, extension points, and remaining fidelity gaps. |
| [Source and fidelity audit](source-audit.md) | Exactness criteria, source pins, unavailable internals, accounting caveats, and claims the project will not make. |
| [Frontier-lab coverage](frontier-lab-coverage.md) | Survey coverage across named labs and the distinction between runnable systems and documentary evidence. |
| [Experiment protocol](experiment-protocol.md) | Candidate labels, task strata, compute regimes, controls, metrics, and release gate. |

## Application catalogs

The application directories are the operational entry points:

- [Browser](../browser/README.md)
- [Computer use](../computer-use/README.md)
- [SWE and SWE with computer tools](../swe/README.md)

Within each application, `implementations/` contains published protocol or pinned
runtime boundaries, `studies/` contains local reconstructions and controlled
ablations, and `gaps/` records targets that are intentionally not executable.

## Source of truth order

When documents appear to overlap, use this order:

1. registry definitions and validation code under `src/scaffoldlab/`;
2. application profile JSON and application README;
3. source and fidelity audit;
4. reference and benchmark runbook;
5. root README and handoff summary.

Any change to a profile's fidelity or runnable state should update all affected
layers in the same change.
