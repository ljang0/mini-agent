# Browser application

This is the application-first entry point for browser and web-research runs.
`implementations/` contains only exact published protocol boundaries or adapters to
pinned upstream runtimes. `studies/` contains useful local reconstructions and
system-card topology experiments that are not 1:1.

An implementation uses:

```json
{
  "application": {
    "name": "browser",
    "implementation": "profile-id"
  }
}
```

Studies use the same shape with `"study"` in place of `"implementation"`. Configs
omit `harnesses`; Scaffold Lab resolves the registered signature and rejects drift.

## Reproducible implementations

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`openai-hosted-browser-functions.json`](implementations/openai-hosted-browser-functions.json) | `openai-responses` | published protocol boundary | Exact `responses_multi_agent=v1` HTTP request/continuation with client browser functions. The server scheduler remains closed and is not reimplemented. |
| [`openai-hosted-web-search.json`](implementations/openai-hosted-web-search.json) | `openai-responses` | published protocol boundary | Exact hosted multi-agent plus `web_search` request. |
| [`anthropic-managed-web-research.json`](implementations/anthropic-managed-web-research.json) | `anthropic-managed-agents` | published protocol boundary | Exact managed-session, primary-event-list, and aggregate-usage boundary; validates the pinned resolved coordinator and records a snapshot digest. Application-specific remote tools and the mutable environment remain caller-provisioned. |
| [`xai-hosted-web-research-4.json`](implementations/xai-hosted-web-research-4.json) | `xai-responses` | published protocol boundary | Exact Responses request for the documented four-agent (`reasoning.effort=low`) hosted team with `web_search`. |
| [`xai-hosted-web-research-16.json`](implementations/xai-hosted-web-research-16.json) | `xai-responses` | published protocol boundary | Exact Responses request for the documented sixteen-agent (`reasoning.effort=high`) hosted team with `web_search`. |
| [`browser-use-0.13.7-upstream.json`](implementations/browser-use-0.13.7-upstream.json) | `browser-use-upstream` | pinned upstream source runtime | Executes the official 0.13.7 `Agent`, `Browser`, provider wrapper, prompts, history, actions, and model loop from a bounded private archive of the pinned commit. Git/Python identities are recorded; dependency/browser/model identity and complete cost remain caller-pinned gaps. |
| [`macu-upstream.json`](implementations/macu-upstream.json) | `macu-upstream` | pinned upstream runtime | Executes the clean released MACU checkout at the recorded commit. Dependencies, VM image, tasks, and model snapshots must also be pinned for experimental identity. |

## Studies, not implementations

| Config | What it preserves | Why it is not 1:1 |
| --- | --- | --- |
| [`browser-use-parallel-pattern.json`](studies/browser-use-parallel-pattern.json) | Distinct tasks in separate browser contexts with flat async fan-out. | It does not execute Browser-Use's Agent policy, prompts, history, actions, or browser runtime. It is not best-of-N because it has no selector. |
| [`macu-text-dag.json`](studies/macu-text-dag.json) | Mutable DAG and ready-frontier scheduling. | It omits upstream CUA processes, VMs, file handoffs, prompts, and evaluator. |
| [`fable-team-3-simulation.json`](studies/fable-team-3-simulation.json) | Disclosed Mythos 5 three-agent topology; the stable filename predates the model-name correction. | The card's multi-agent evaluation is Mythos-only. Per-agent limits are public but not enforced; exact prompts, timing, compaction behavior, tools, and checkout/Git semantics are not reproduced. |
| [`opus-team-5-simulation.json`](studies/opus-team-5-simulation.json) | Disclosed five-agent topology. | The 1M total-token limit per agent is public but not enforced; the card used unreleased effort, and exact prompts, timing, tools, and worktrees are unavailable. |

Meta Muse Spark 1.1 is registry-only (`browser/meta-muse-spark-1.1-orchestration`)
because no runnable public scheduler contract is available.

## Validation

Use `benchmarks/browser_smoke.jsonl` for browser-context profiles and
`benchmarks/flat_parallel_smoke.jsonl` for the Browser-Use parallel pattern. For
example:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/implementations/openai-hosted-browser-functions.json \
  --provider openai-responses
```

Hosted and upstream providers need their provider-specific run flags. Browser-Use
runs one task per invocation and owns the browser/tool loop; the flat-parallel study
remains a separate scheduler experiment. MACU must
run one generic task per invocation against a disposable, pinned VM/CUA stack;
that adapter path is not a claim of parity with a named browser benchmark.
