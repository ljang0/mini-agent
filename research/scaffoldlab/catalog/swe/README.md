# SWE application

This is the application-first entry point for software-engineering runs, including
the composite SWE-with-computer surface. `implementations/` contains only exact
hosted/CLI protocol boundaries, clean revision-pinned upstream source runtimes, or an
audited official distribution when matching source is unavailable. `studies/` contains
baselines, clean-room variants, and topology simulations. Every config omits
`harnesses`; the registry supplies the registered signature and rejects drift.

## Reproducible implementations

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`anthropic-managed-agents.json`](implementations/anthropic-managed-agents.json) | `anthropic-managed-agents` | published protocol boundary | Managed session/events/cleanup/aggregate usage with a pinned resolved coordinator version, non-empty roster, and recorded snapshot digest. Anthropic still owns the scheduler and mutable environment definition. |
| [`prime-agent-0.7.1.json`](implementations/prime-agent-0.7.1.json) | `prime-agent` | published runtime protocol boundary | Mandatory 0.7.1 check and exact JSON v3 session/event/`agent_end` framing. The installed executable is not identified unless `--prime-agent-executable-sha256` is supplied; runtime internals, daemon/schedule/continual state, and full-tree accounting remain outside the claim. |
| [`grok-build-1.0.0.json`](implementations/grok-build-1.0.0.json) | `grok-build` | published runtime protocol boundary | Mandatory 1.0.0 check and exact headless JSON invocation/result boundary. The installed npm/executable is not identified unless `--grok-executable-sha256` is supplied; npm/source revisions disagree, and scheduler internals and some usage remain outside the claim. |
| [`grok-build-source-1.0.0.json`](implementations/grok-build-source-1.0.0.json) | `grok-build-source` | pinned upstream source runtime | Builds a verified private archive of the public revision with isolated Git/Cargo/rustc state, runs official native subagents, and exports a bounded binary patch. Hosted model/service state, bit-reproducible build inputs, and complete accounting remain explicit gaps. |
| [`kimi-code-0.34.0-upstream.json`](implementations/kimi-code-0.34.0-upstream.json) | `kimi-code-upstream` | pinned upstream source runtime | Runs the official TypeScript v2 `Agent`/`AgentSwarm`, prompts, tools, permissions, compaction, and stream-JSON protocol after matching every tracked worktree blob to the pinned commit. It still executes caller-installed pnpm dependencies from that worktree; ignored/generated dependency content, hosted services, and whole-tree usage are unavailable, and `--prompt` exposes task text to local process inspection. |
| [`openai-codex-source-0.147.0.json`](implementations/openai-codex-source-0.147.0.json) | `codex-source` | pinned upstream source runtime | Builds a private archive of the pinned Codex commit with isolated Git/Cargo/rustc state, runs native `codex exec` collaboration, and exports a bounded binary patch. Remote model/catalog/service policy and full-tree usage are unpinned; v1 is requested but not claimed effective without evidence. |
| [`claude-code-agent-teams-2.1.226.json`](implementations/claude-code-agent-teams-2.1.226.json) | `claude-code-agent-teams` | audited official distribution adapter | Verifies the official Darwin arm64 executable/package identity and current implicit-team lifecycle, cross-checks named Agent calls against a live session-derived roster and persistent tasks, verifies automatic config cleanup, and exports a patch. Matching runtime source and server-managed policy are unavailable. |
| [`rlm-0.1.3-upstream.json`](implementations/rlm-0.1.3-upstream.json) | `rlm-upstream` | pinned upstream runtime | Official 0.1.3 prompts and selected REPL from a clean commit. The safe adapter defaults to Docker/1,500 seconds; recursion requires `max_depth >= 2`. The bridge covers string context and selected settings, not every public RLM option. It injects no SWE tools and is not SWE-bench parity. |

## Studies, not implementations

| Config | Classification | Boundary |
| --- | --- | --- |
| [`prime-agent-source-0.7.1.json`](studies/prime-agent-source-0.7.1.json) | caller-built runtime study | Executes a private immutable copy of caller-provided bundle bytes beside pinned source/lockfile evidence. No adapter-owned build or authoritative bundle digest proves source parity. |
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

Codex source, Grok source, and the Claude Code distribution own their external
workspace loops rather than using `config.environment`. Each strips inherited Git
administration, creates a fresh standalone baseline, captures a bounded
`git diff --binary --full-index` including untracked files, and returns it through the
same private patch payload that the matrix externalizes under `patches/` before the
temporary workspace is removed. Prime/Kimi boundaries differ; read their profile
metadata before treating an answer as a durable code change.

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
task prompt as that value. Upstream RLM, Prime Agent, Grok Build, Kimi Code, Codex
source, and Claude Code require
one task, one profile, and one isolated invocation. The Prime source study requires a
caller-built clean checkout; source-native Grok builds a private archive with its
locked toolchain inputs; all coding runtimes require disposable or copied workspaces.
