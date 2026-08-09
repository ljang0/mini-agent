# SWE application

This is the application-first entry point for software-engineering runs, including
the composite SWE-with-computer surface. `implementations/` contains only exact
hosted/CLI protocol boundaries or clean revision-pinned upstream runtimes. `studies/` contains
baselines, clean-room variants, and topology simulations. Every config omits
`harnesses`; the registry supplies the registered signature and rejects drift.

## Reproducible implementations

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`anthropic-managed-agents.json`](implementations/anthropic-managed-agents.json) | `anthropic-managed-agents` | published protocol boundary | Managed session/events/cleanup/aggregate usage with a pinned resolved coordinator version, non-empty roster, and recorded snapshot digest. Anthropic still owns the scheduler and mutable environment definition. |
| [`prime-agent-0.7.1.json`](implementations/prime-agent-0.7.1.json) | `prime-agent` | published runtime protocol boundary | Mandatory 0.7.1 check and exact JSON v3 session/event/`agent_end` framing. The installed executable is not identified unless `--prime-agent-executable-sha256` is supplied; runtime internals, daemon/schedule/continual state, and full-tree accounting remain outside the claim. |
| [`grok-build-1.0.0.json`](implementations/grok-build-1.0.0.json) | `grok-build` | published runtime protocol boundary | Mandatory 1.0.0 check and exact headless JSON invocation/result boundary. The installed npm/executable is not identified unless `--grok-executable-sha256` is supplied; npm/source revisions disagree, and scheduler internals and some usage remain outside the claim. |
| [`rlm-0.1.3-upstream.json`](implementations/rlm-0.1.3-upstream.json) | `rlm-upstream` | pinned upstream runtime | Official 0.1.3 prompts and selected REPL from a clean commit. The safe adapter defaults to Docker/1,500 seconds; recursion requires `max_depth >= 2`. The bridge covers string context and selected settings, not every public RLM option. It injects no SWE tools and is not SWE-bench parity. |

## Studies, not implementations

| Config | Classification | Boundary |
| --- | --- | --- |
| [`parallel-best-of-3.json`](studies/parallel-best-of-3.json) | controlled baseline | Three independent candidates plus an explicit judge. This is best-of-N; `flat_parallel` is not. |
| [`openai-hosted-shell.json`](studies/openai-hosted-shell.json) | source-matched composition | Public multi-agent and local-shell protocols are composed, but the combined Codex product harness and hosted scheduler are not public or live-verified. |
| [`openai-hosted-swe-computer.json`](studies/openai-hosted-swe-computer.json) | source-matched composition | Public multi-agent/computer protocols plus SWE functions; no published or live-verified combined boundary. |
| [`rlm-0.1.3.json`](studies/rlm-0.1.3.json) | local RLM subset | Restricted persistent external-context loop; not the upstream REPL/runtime. |
| [`platoon-recursive-inference.json`](studies/platoon-recursive-inference.json) | inference-shape study | Restricted local REPL and caller model, not the pinned Platoon runtime or RAO training/checkpoint. |
| [`macu-swe-computer.json`](studies/macu-swe-computer.json) | cross-domain study | MACU-inspired DAG over SWE/computer tools; MACU itself is a CUA artifact and the upstream runtime is absent. |
| [`fable-team-3-simulation.json`](studies/fable-team-3-simulation.json) | Mythos 5 topology simulation | The card's multi-agent evaluation is Mythos-only; exact prompts, timing, per-agent limits, compaction, messaging, and Git worktrees are not reproduced. |
| [`opus-team-5-simulation.json`](studies/opus-team-5-simulation.json) | Opus 5 topology simulation | Unreleased effort, exact prompts, timing, per-agent limits, messaging, and worktrees are not reproduced. |

Catalog-only entries are intentionally not runnable configs:

- `swe/rao-policy-training`: training method. The pinned Platoon snapshot publishes
  inference and Tinker/AReaL training pipelines, reward logic, prompts, configs, and
  task assets, but Scaffold Lab does not execute them and the paper-trained
  checkpoints are not identified.
- `swe/xai-hosted-multi-agent-tools`: xAI has not published a client
  shell/editor continuation for its hosted team.
- `swe/meta-muse-spark-1.1-coding-orchestration`: no public runnable scheduler
  contract.
- `swe/swe-bench-4.1.0`: evaluation environment; pinned evaluator images are not
  bundled.

## Workspace and patch boundary

Every local SWE or `swe_computer` config uses per-agent copied workspaces and sets
`export_patch: true` with a one-MiB `max_patch_bytes` cap. A task must provide a
frozen source checkout through `metadata.environment.workspace`. Run outputs must
remain disjoint from that checkout. Patch export preserves a reviewable artifact;
it does not replace the benchmark's own evaluator.

Native OpenAI local shell is enabled only in the dedicated hosted-shell config to
match that published protocol. It bypasses the command allowlist and therefore
requires an outer disposable container or VM with credentials removed. Other
local configs keep native shell disabled and use a fixed executable allowlist,
which is still not an OS sandbox.

## Validation

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/swe_smoke.jsonl \
  --config swe/studies/parallel-best-of-3.json \
  --provider openai-responses
```

The local RLM profile requires tasks with non-empty `context`. The upstream RLM
profile treats `context` as the external value when present and otherwise uses the
task prompt as that value. Upstream RLM, Prime Agent, and Grok Build require one
task, one profile, and one isolated invocation; Prime/Grok also require disposable
worktrees.
