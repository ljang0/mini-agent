# SWE application

This directory is the application-first entry point for software-engineering
experiments, including the composite SWE-with-computer surface. Every config uses
the canonical nested `application` selection and omits `harnesses`; the registry
auto-fills the exact harness name and options and rejects drift.

## Runnable representatives

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`parallel-best-of-3.json`](configs/parallel-best-of-3.json) | any listed local provider | controlled baseline | Three independent candidates plus an explicit judge selector. This is best-of-N; `flat_parallel` is not. |
| [`openai-hosted-shell.json`](configs/openai-hosted-shell.json) | `openai-responses` | source-matched reimplementation | Published hosted multi-agent and developer-tool continuation plus the documented local-shell protocol. The combined Codex product harness and hosted scheduler are not public. |
| [`openai-hosted-swe-computer.json`](configs/openai-hosted-swe-computer.json) | `openai-responses` | source-matched reimplementation | Composes documented multi-agent and GA computer protocols with SWE functions. No published or live-verified combined SWE/computer contract exists. |
| [`anthropic-managed-agents.json`](configs/anthropic-managed-agents.json) | `anthropic-managed-agents` | exact public protocol | Managed session, event-list, cleanup, roster/shared-environment contract, and aggregate usage. The hosted scheduler and exact remote snapshot remain unavailable. |
| [`prime-agent-0.7.1.json`](configs/prime-agent-0.7.1.json) | `prime-agent` | pinned upstream runtime adapter | Released JSON-mode Prime Agent runtime with persistent Python control and subagents. Full child-tree usage and a security sandbox are unavailable. |
| [`grok-build-1.0.0.json`](configs/grok-build-1.0.0.json) | `grok-build` | pinned upstream runtime adapter | Released headless Grok Build runtime, native subagents, and worktree semantics. Some usage and npm/source bit identity remain incomplete. |
| [`rlm-0.1.3.json`](configs/rlm-0.1.3.json) | any listed local provider | source-matched reimplementation | Persistent external-context code loop with batched/recursive subcalls. This is not the unrestricted upstream REPL and domain tools are withheld from the controller. |
| [`rlm-0.1.3-upstream.json`](configs/rlm-0.1.3-upstream.json) | `rlm-upstream` | pinned upstream runtime adapter | Official 0.1.3 prompts and REPL through a clean pinned checkout and bounded process. It injects no SWE tools; recursive-child usage is absent from the root summary. |
| [`platoon-recursive-inference.json`](configs/platoon-recursive-inference.json) | any listed local provider | inference-only reimplementation | Recursive sequential/parallel child-control shape only. It is not faithful RAO because shared-policy training, rewards, data, and checkpoint are absent. |
| [`macu-swe-computer.json`](configs/macu-swe-computer.json) | `openai-responses` or `anthropic-messages` | source-matched reimplementation | Local MACU-inspired DAG with SWE and computer tools. MACU is a CUA artifact, not an SWE system-card harness, and its upstream VM/file runtime is absent. |
| [`fable-team-3-simulation.json`](configs/fable-team-3-simulation.json) | any listed local provider | topology simulation | Disclosed three-agent coding topology only; prompts, timing, context enforcement, and Git worktrees are not reproduced. |
| [`opus-team-5-simulation.json`](configs/opus-team-5-simulation.json) | any listed local provider | topology simulation | Disclosed five-agent topology only; unreleased effort, prompts, timing, context limits, and worktrees are not reproduced. |

Catalog-only entries are intentionally not runnable configs:

- `swe/rao-policy-training`: training method; no training pipeline, rewards, data,
  or trained checkpoint.
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
  --config swe/configs/parallel-best-of-3.json \
  --provider openai-responses
```

The local RLM profile requires tasks with non-empty `context`. The upstream RLM
profile treats `context` as the external value when present and otherwise uses the
task prompt as that value. Upstream RLM, Prime Agent, and Grok Build require one
task, one profile, and one isolated invocation; Prime/Grok also require disposable
worktrees.
