# Reference and benchmark runbook

As-of date: 2026-08-10.

This page answers two separate questions for every Scaffold Lab candidate:

1. Is the multi-agent harness or provider/runtime adapter implemented?
2. Is the named benchmark's canonical task loader, reset, and evaluator also
   implemented?

Those are different boundaries. A harness can be runnable against JSONL tasks while
BrowserGym, OSWorld, or SWE-bench parity is still unavailable.

## Current implementation status

| Layer | Status | What is present |
| --- | --- | --- |
| Shared agent runtime | Implemented | `ModelBackend`, `Harness`, shared budget ledger, trace recorder, cancellation, stopping limits, provider-neutral tools, and per-agent environments. |
| Multi-agent harnesses | Implemented | Single agent, best-of-N with selector, flat fan-out, blocking manager, fixed team, async subagents, mutable MACU-like DAG, restricted RLM, recursive Platoon inference, and upstream-runtime wrappers. |
| API adapters | Implemented | OpenAI Responses, Anthropic Messages, OpenAI-compatible Responses/Chat, xAI hosted multi-agent, and Anthropic Managed Agents. |
| External source/CLI adapters | Implemented | Browser-Use, MACU, RLM, Prime Agent, Grok Build, Kimi Code, OpenAI Codex, and the official Claude Code distribution. |
| Generic benchmark matrix | Implemented | JSONL task loading, repeats, shuffled trial order, exact/contains/regex/JSON evaluators, manifests, result records, traces, Wilson intervals, cost fields, and SWE patch artifacts. |
| Browser and computer task surfaces | Implemented | Semantic Playwright browser tools and a screenshot/action computer loop. |
| SWE task surface | Implemented | Per-agent copied workspaces, file tools, bounded command execution, optional provider-native shell/editor tools, and binary patch export. |
| BrowserGym benchmark adapter | Not implemented | Browser tasks can run, but BrowserGym's canonical reset, task set, and evaluator are not bundled. |
| OSWorld 1/2 benchmark adapter | Not implemented | MACU can run its generic upstream VM path, but Scaffold Lab does not supply the canonical domain/UUID loader, benchmark setup, assets, or evaluator. |
| SWE-bench 4.1 evaluator | Not implemented | SWE agents can generate patches, but official images, repository reset, test execution, and scoring are not bundled. |
| Paid release comparison | Not run | Offline tests and smoke tasks validate plumbing only; no release winner has been established. |

The catalog uses four execution states:

- **Direct API:** runnable after supplying a provider credential, model, and any
  required remote resource IDs.
- **External runtime:** runnable after installing or checking out the pinned upstream
  artifact and supplying the provider-specific CLI flags.
- **Local study:** runnable through an API backend, but its name is a baseline,
  source-matched subset, or topology simulation rather than a parity claim.
- **Catalog gap:** documentation-only and intentionally rejected as an executable
  application profile.

## Common benchmark runner

Install and validate without making a model request:

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider openai-responses
```

Run an API-backed profile:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/implementations/openai-hosted-web-search.json \
  --provider openai-responses \
  --model gpt-5.6-sol \
  --output runs/openai-browser
```

Every task is one JSON object per line. Supported local evaluator types are `exact`,
`contains`, `regex`, and `json_equal`:

```json
{"task_id":"example","prompt":"Return the project code.","context":"The code is ORBIT.","evaluator":{"type":"contains","value":"ORBIT"},"split":"smoke"}
```

SWE tasks can select a frozen source workspace through task metadata:

```json
{"task_id":"repair-1","prompt":"Repair the failing test.","metadata":{"environment":{"workspace":"/absolute/path/to/frozen/repository"},"evaluator":{"type":"contains","value":"expected text"}}}
```

For a real SWE benchmark, the local text evaluator above must be replaced or followed
by that benchmark's own containerized test evaluator. Patch generation alone is not a
benchmark score.

Each run writes `manifest.json`, `results.jsonl`, `summary.json`, per-trial JSONL
traces, and—when enabled—bounded binary patches. See [Outputs](../README.md#outputs)
for the exact artifact and privacy contract.

## Browser application

All runnable rows below have an implemented harness and backend. None currently
includes a BrowserGym evaluator.

| Profile or config | Run mode | Implemented boundary | Primary references | Benchmark state |
| --- | --- | --- | --- | --- |
| [`openai-hosted-multi-agent-functions`](../browser/implementations/openai-hosted-browser-functions.json) | Direct API | Hosted Responses multi-agent plus client-executed semantic browser functions and HTTP continuation. | [OpenAI multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent), [GPT-5.6 system card](https://deploymentsafety.openai.com/gpt-5-6) | Generic browser JSONL tasks only. |
| [`openai-hosted-web-search`](../browser/implementations/openai-hosted-web-search.json) | Direct API | Hosted multi-agent request with server-side `web_search`. | [OpenAI multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent) | Generic web-research tasks only. |
| [`xai-hosted-web-research-4`](../browser/implementations/xai-hosted-web-research-4.json) and [`-16`](../browser/implementations/xai-hosted-web-research-16.json) | Direct API | Documented four- or sixteen-agent hosted request with `web_search`; leader response and aggregate usage are recorded. | [xAI multi-agent API](https://docs.x.ai/developers/model-capabilities/text/multi-agent), [Grok 4.20 model card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf) | Generic web-research tasks only. |
| [`anthropic-managed-web-research`](../browser/implementations/anthropic-managed-web-research.json) | Direct managed API | Managed session, pinned resolved coordinator snapshot, primary event list, cleanup, and aggregate usage. | [Anthropic Managed Agents orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) | Remote environment and task evaluator are caller-provisioned. |
| [`browser-use-0.13.7-upstream`](../browser/implementations/browser-use-0.13.7-upstream.json) | External runtime | Actual pinned `Agent`, `Browser`, provider wrapper, prompts, history, actions, and loop run from a private source archive. | [Pinned Browser-Use 0.13.7 source](https://github.com/browser-use/browser-use/tree/f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc), [parallel template](https://docs.browser-use.com/open-source/examples/templates/parallel-browser) | Browser-Use runtime is real; no BrowserGym task/evaluator bridge. |
| [`macu-upstream`](../browser/implementations/macu-upstream.json) | External runtime | Pinned released manager, graph scheduler, CUA workers, VM variants, replanning, and artifact ingestion for a generic task. | [MACU paper](https://arxiv.org/abs/2606.01533), [pinned MACU source](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105) | Generic task path only; no named browser benchmark evaluator. |
| [`browser-use-parallel-pattern`](../browser/studies/browser-use-parallel-pattern.json) | Local study + API | Distinct tasks fan out across isolated browser contexts. It has no selector and is not best-of-N. | [Browser-Use parallel template](https://docs.browser-use.com/open-source/examples/templates/parallel-browser) | Runnable scheduling ablation. |
| [`macu-text-dag`](../browser/studies/macu-text-dag.json) | Local study + API | Mutable DAG and ready-frontier scheduling over local browser tools. | [MACU paper](https://arxiv.org/abs/2606.01533), [pinned MACU source](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105) | Runnable source-matched subset, not upstream parity. |
| Mythos/Fable card variants | Local study + API | Fixed 3/5/10-agent teams, blocking manager, and async BrowseComp topology variants. | [Fable 5 and Mythos 5 system card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) | Topology simulations; representative config: [`fable-team-3-simulation.json`](../browser/studies/fable-team-3-simulation.json). |
| Opus 5 card variants | Local study + API | Fixed 5/10-agent teams and async topology variants. | [Claude Opus 5 system card](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) | Topology simulations; representative config: [`opus-team-5-simulation.json`](../browser/studies/opus-team-5-simulation.json). |
| Meta Muse Spark 1.1 | Catalog gap | Public roles and evaluation evidence are recorded; no scheduler API is implemented. | [Muse Spark release](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/), [evaluation report](https://ai.meta.com/static-resource/muse-spark-1-1-evaluation-report) | Not runnable as Muse orchestration. |

Registered Mythos/Fable browser study IDs are
`anthropic-fable5-team-3`, `anthropic-fable5-team-5`,
`anthropic-fable5-team-10`, `anthropic-fable5-blocking`, and
`anthropic-fable5-async-browsecomp`. Registered Opus study IDs are
`anthropic-opus5-team-5`, `anthropic-opus5-team-10`, and
`anthropic-opus5-async`. The checked-in JSON files are representative presets; copy
one and change only `application.study` to select another registered variant. The
registry injects and validates its harness signature. The exact catalog gap IDs are
`meta-muse-spark-1.1-orchestration` and `anthropic-opus5-cowork-safety`.

## Computer-use application

| Profile or config | Run mode | Implemented boundary | Primary references | Benchmark state |
| --- | --- | --- | --- | --- |
| [`openai-ga-computer-single`](../computer-use/implementations/openai-ga-single.json) | Direct API | Ordered GA `computer_call.actions`, action execution, screenshot capture, and `computer_call_output` continuation. | [OpenAI computer-use guide](https://developers.openai.com/api/docs/guides/tools-computer-use), [GPT-5.6 system card](https://deploymentsafety.openai.com/gpt-5-6) | Runnable in a Playwright viewport; not OSWorld. |
| [`anthropic-computer-20251124-single`](../computer-use/implementations/anthropic-20251124-single.json) | Direct API | Published `computer_20251124` action and tool-result protocol. | [Anthropic computer-use guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) | Runnable in a Playwright viewport; not Anthropic's Linux reference environment or OSWorld. |
| [`openai-hosted-multi-agent-computer`](../computer-use/studies/openai-hosted-multi-agent.json) | Local/API composition study | Hosted multi-agent request combined with the published computer loop. | [OpenAI multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent), [computer use](https://developers.openai.com/api/docs/guides/tools-computer-use) | Runnable study; the combined contract has not been published or live-verified. |
| [`macu-text-dag`](../computer-use/studies/macu-text-dag.json) | Local study + API | MACU-like DAG over the local screenshot/action environment. | [MACU paper](https://arxiv.org/abs/2606.01533) | Runnable study; not released MACU or OSWorld parity. |
| [`macu-upstream-generic-vm`](../computer-use/implementations/macu-upstream-generic-vm.json) | External runtime | Released MACU manager/CUA/VM loop for one generic blank-start task. | [Pinned MACU source](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105) | Runtime implemented; canonical OSWorld task mapping and evaluator missing. |
| `macu-osworld1-benchmark-parity` | Catalog gap | Records the missing domain/UUID loader, setup, assets, evaluator, and task pins. | [MACU paper](https://arxiv.org/abs/2606.01533) | Not runnable. |
| `osworld2-2026-06-24` | Catalog gap | Records the pinned benchmark release. | [OSWorld 2.0 release](https://github.com/xlang-ai/OSWorld-V2/releases/tag/v2026.06.24) | Gated evaluator assets and frozen VM are not bundled. |
| `meta-muse-spark-1.1-computer-use` | Catalog gap | Public description only; no action/scheduler contract. | [Muse Spark release](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) | Not runnable as Muse. |

## SWE application

| Profile or config | Run mode | Implemented boundary | Primary references | Benchmark state |
| --- | --- | --- | --- | --- |
| `single-agent-control` | Local study + API | One solver in an isolated copied repository. | [SWE-bench 4.1.0](https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0) | Runnable control; official evaluator missing. |
| [`parallel-best-of-3`](../swe/studies/parallel-best-of-3.json) | Local study + API | Three isolated candidate agents followed by an explicit judge. | Scaffold Lab controlled baseline | Runnable best-of-N control; official evaluator missing. |
| [`openai-hosted-multi-agent-shell`](../swe/studies/openai-hosted-shell.json) | Local/API composition study | Hosted multi-agent plus client-executed local shell and editor continuation. | [OpenAI multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent), [shell](https://developers.openai.com/api/docs/guides/tools-shell) | Runnable study; combined product boundary and official SWE evaluator missing. |
| [`openai-hosted-multi-agent-swe-computer`](../swe/studies/openai-hosted-swe-computer.json) | Local/API composition study | Hosted multi-agent with SWE functions and the computer loop. | [OpenAI multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent), [computer use](https://developers.openai.com/api/docs/guides/tools-computer-use) | Runnable study; combined boundary not live-verified. |
| [`openai-codex-source-0.147.0`](../swe/implementations/openai-codex-source-0.147.0.json) | External source runtime | Builds a private archive of pinned Codex source, executes native `codex exec` collaboration, requires completed spawn evidence, and exports a patch. | [Pinned OpenAI Codex source](https://github.com/openai/codex/tree/be6e8eac029b183056b7e4402879f15d2c85f61b), [OpenAI multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent) | Coding runtime implemented; official SWE-bench evaluator missing. |
| [`claude-code-agent-teams-2.1.226`](../swe/implementations/claude-code-agent-teams-2.1.226.json) | External official distribution | Audited Darwin arm64 bytes, implicit session-derived team, live roster/task evidence, automatic config cleanup, and patch export. | [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams), [tool reference](https://code.claude.com/docs/en/tools-reference) | Coding runtime implemented; system-card experiment and SWE-bench evaluator not reproduced. |
| [`anthropic-managed-agents`](../swe/implementations/anthropic-managed-agents.json) | Direct managed API | Managed session/events/cleanup/aggregate usage with pinned resolved coordinator snapshot. | [Anthropic Managed Agents](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) | Remote tools/environment/evaluator are caller-provisioned. |
| [`prime-agent-0.7.1`](../swe/implementations/prime-agent-0.7.1.json) | External installed runtime | Exact version-checked JSON v3 session/event/`agent_end` CLI boundary. | [Prime Agent launch](https://www.primeintellect.ai/blog/prime-agent), [pinned 0.7.1 source](https://github.com/PrimeIntellect-ai/prime-agent/tree/95afd319a78ae017a41241d50b013d656a0685ce), [multi-agent analysis](https://www.primeintellect.ai/blog/multi-agent-systems) | Runtime protocol implemented; official SWE evaluator and full-tree accounting missing. |
| [`prime-agent-source-0.7.1`](../swe/studies/prime-agent-source-0.7.1.json) | Caller-built runtime study | Copies and hashes a caller bundle beside pinned source/lock evidence. | [Pinned Prime source](https://github.com/PrimeIntellect-ai/prime-agent/tree/95afd319a78ae017a41241d50b013d656a0685ce) | Runnable study; source-to-bundle parity is not proven. |
| [`grok-build-1.0.0`](../swe/implementations/grok-build-1.0.0.json) | External installed runtime | Version-checked headless JSON CLI boundary. | [Grok Build source/docs](https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c) | Runtime protocol implemented; official SWE evaluator missing. |
| [`grok-build-source-1.0.0`](../swe/implementations/grok-build-source-1.0.0.json) | External source runtime | Builds locked pinned source, runs native subagents, and exports a patch. | [Pinned Grok Build source](https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c), [Grok 4.20 card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf) | Coding runtime implemented; build-byte and evaluator gaps remain. |
| [`kimi-code-0.34.0-upstream`](../swe/implementations/kimi-code-0.34.0-upstream.json) | External source runtime | Executes pinned TypeScript `Agent`/`AgentSwarm`, tools, permissions, compaction, and stream JSON after checking every tracked blob/mode. | [Pinned Kimi Code source](https://github.com/MoonshotAI/kimi-code/tree/f0614c53e59f7e1e257412063b059b9eb82764cf) | Coding runtime implemented; generated dependencies, full accounting, and evaluator missing. |
| [`rlm-0.1.3-contract`](../swe/studies/rlm-0.1.3.json) | Local study + API | Restricted persistent Python/REPL loop with external context and batched/recursive subcalls. | [RLM paper](https://arxiv.org/abs/2512.24601), [official RLM 0.1.3](https://github.com/alexzhang13/rlm/releases/tag/v0.1.3) | Runnable RLM subset; no upstream REPL or SWE tools. |
| [`rlm-0.1.3-upstream`](../swe/implementations/rlm-0.1.3-upstream.json) | External source runtime | Pinned official prompts and selected REPL through bounded JSON stdin; recursion requires depth two or greater. | [Official RLM 0.1.3](https://github.com/alexzhang13/rlm/releases/tag/v0.1.3) | Official generic RLM runtime runs; no Scaffold Lab SWE tools or SWE evaluator. |
| [`platoon-recursive-inference`](../swe/studies/platoon-recursive-inference.json) | Local study + API | Restricted recursive sequential/parallel inference shape. | [RAO paper](https://arxiv.org/abs/2605.06639), [Platoon snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe) | Runnable inference study; RAO training and paper checkpoint missing. |
| [`macu-swe-computer-experiment`](../swe/studies/macu-swe-computer.json) | Local study + API | MACU-like DAG over combined SWE and browser-viewport computer tools. | [MACU paper](https://arxiv.org/abs/2606.01533) | Runnable cross-domain study; not an upstream MACU SWE system. |
| Mythos/Fable card variants | Local study + API | Fixed 3/5/10-agent, blocking, and async ProgramBench topologies. | [Fable 5 and Mythos 5 system card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) | Topology simulations; representative config: [`fable-team-3-simulation.json`](../swe/studies/fable-team-3-simulation.json). |
| Opus 5 card variants | Local study + API | Fixed 5/10-agent and async topologies. | [Claude Opus 5 system card](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) | Topology simulations; representative config: [`opus-team-5-simulation.json`](../swe/studies/opus-team-5-simulation.json). |
| `rao-policy-training` | Catalog gap | Records the training method and public snapshot without pretending inference shape is the trained policy. | [RAO paper](https://arxiv.org/abs/2605.06639), [Platoon snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe) | Not runnable; no identified paper-trained checkpoint. |
| `swe-bench-4.1.0` | Catalog gap | Records the benchmark artifact separately from inference. | [SWE-bench 4.1.0](https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0) | Official images/reset/evaluator not bundled. |
| Meta Muse and xAI hosted SWE tools | Catalog gaps | Records public disclosures where no matching client scheduler/tool continuation exists. | [Muse Spark](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/), [xAI multi-agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent) | Not runnable as the named systems. |

Registered Mythos/Fable SWE variants are `anthropic-fable5-team-3`,
`anthropic-fable5-team-5`, `anthropic-fable5-team-10`,
`anthropic-fable5-blocking`, and `anthropic-fable5-async-programbench`. Registered
Opus variants are `anthropic-opus5-team-5`, `anthropic-opus5-team-10`, and
`anthropic-opus5-async`. As in the browser application, the checked-in JSON files are
representative presets; the application registry owns the exact harness signature for
every registered variant. The other exact catalog gap IDs are
`meta-muse-spark-1.1-coding-orchestration` and
`xai-hosted-multi-agent-tools`.

## External runtime prerequisites

The detailed flags and safety boundaries are maintained in the main
[README run examples](../README.md#application-environment-and-provider-coverage).
At minimum:

| Provider | Required caller-owned inputs |
| --- | --- |
| `browser-use-upstream` | Clean pinned checkout, matching Python environment/dependencies, LLM/browser JSON, scoped credentials. |
| `macu-upstream` | Clean pinned MACU checkout, OSWorld-derived runtime root, isolated result directory, manager/CUA model settings, scoped credentials. |
| `rlm-upstream` | Clean pinned checkout, Python/dependencies, selected REPL environment (Docker by default), model/provider settings, scoped credentials. |
| `prime-agent` | Installed 0.7.1 executable, disposable worktree, provider/model, scoped credentials; optional executable digest. |
| `prime-agent-source` | Clean pinned source, caller-built bundle and dependencies, Node/npm, disposable worktree, scoped credentials. |
| `grok-build` | Installed 1.0.0 CLI, disposable worktree, allow/deny policy, scoped xAI credential; optional executable digest. |
| `grok-build-source` | Clean pinned source, seed workspace, concrete Cargo/rustc/Git tools, build dependencies, scoped xAI credential. |
| `kimi-code-upstream` | Clean pinned tracked source, installed frozen-lockfile pnpm dependencies, Node/tsx, disposable worktree, provider credential. |
| `codex-source` | Clean pinned source, clean seed workspace, concrete Cargo/rustc/Git tools, model, scoped OpenAI credential. |
| `claude-code-agent-teams` | Official 2.1.226 distribution, supported audited platform, clean seed workspace, model, scoped Anthropic credential. |

These adapters intentionally do not install arbitrary upstream dependencies during a
trial. Provisioning is caller-controlled so it can be pinned, reviewed, and isolated.

## What must be added for benchmark parity

The inference layer is ready for benchmark integration. A release-grade bridge for a
named benchmark still needs all of the following:

1. a pinned dataset/task revision and deterministic loader;
2. a clean reset for every trial, including browser state, VM snapshot, or repository;
3. the benchmark's official evaluator and exact versioned assets/images;
4. task-specific network, credential, and human-confirmation policy;
5. conversion from benchmark task IDs into Scaffold Lab task/environment metadata;
6. deterministic offline tests for setup, reset, evaluator failure, timeout, and
   contamination;
7. repeated live trials with pinned models and prices before any release ranking.

The next useful implementation milestone is therefore three evaluator bridges—one
each for BrowserGym, OSWorld 2, and SWE-bench—rather than adding another topology
name to the harness registry.

## Related audit documents

- [Source and fidelity audit](source-audit.md) contains the exact/unavailable
  component analysis and accounting caveats.
- [Frontier-lab coverage](frontier-lab-coverage.md) catalogs all 18 surveyed labs,
  including model-only and framework-only evidence that is not runnable.
- [Architecture](architecture.md) defines inference harness, provider adapter,
  training method, and evaluation environment as separate artifact kinds.
- [Experiment protocol](experiment-protocol.md) defines budgets, controls, metrics,
  and the release gate.
