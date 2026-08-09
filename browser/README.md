# Browser application

This directory is the application-first entry point for browser and web-research
experiments. Each config selects exactly one registered implementation with the
canonical nested form:

```json
{
  "application": {
    "name": "browser",
    "implementation": "profile-id"
  }
}
```

The configs deliberately omit `harnesses`. Scaffold Lab resolves the profile's
exact harness name and options from the application registry and rejects drift.

## Runnable representatives

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`openai-hosted-browser-functions.json`](configs/openai-hosted-browser-functions.json) | `openai-responses` | exact public protocol | Published hosted multi-agent request and HTTP developer-function continuation; hosted scheduling and injected prompts remain closed. |
| [`openai-hosted-web-search.json`](configs/openai-hosted-web-search.json) | `openai-responses` | exact public protocol | Published hosted multi-agent and `web_search` request boundary; hosted search implementation remains closed. |
| [`anthropic-managed-web-research.json`](configs/anthropic-managed-web-research.json) | `anthropic-managed-agents` | exact public protocol | Managed session, primary-event-list, and aggregate-usage boundary. The remote agent owns its tools, roster, and environment. |
| [`xai-hosted-web-research-4.json`](configs/xai-hosted-web-research-4.json) | `xai-responses` | exact public protocol | Published four-agent request, hosted `web_search`, and aggregate usage. Server scheduling and child plaintext are unavailable. |
| [`browser-use-parallel-pattern.json`](configs/browser-use-parallel-pattern.json) | any listed local provider | source-matched reimplementation | Runs distinct tasks concurrently in separate browser contexts. `flat_parallel` is not best-of-N because there is no selector. |
| [`macu-text-dag.json`](configs/macu-text-dag.json) | any listed local provider | source-matched reimplementation | Local text-DAG subset only; it does not reproduce MACU's CUA processes, VM cloning, file handoffs, prompts, or evaluator. |
| [`macu-upstream.json`](configs/macu-upstream.json) | `macu-upstream` | pinned upstream runtime adapter | Executes the pinned released MACU checkout. Reproduction still requires pinned dependencies, VM image, tasks, and model snapshots. |
| [`fable-team-3-simulation.json`](configs/fable-team-3-simulation.json) | any listed local provider | topology simulation | Reproduces only the disclosed three-agent topology, not hidden prompts, timing, compaction, tools, or checkout behavior. |
| [`opus-team-5-simulation.json`](configs/opus-team-5-simulation.json) | any listed local provider | topology simulation | Reproduces only the disclosed five-agent topology, not unreleased effort, prompts, tools, timing, or worktrees. |

Meta Muse Spark 1.1 is registry-only (`browser/meta-muse-spark-1.1-orchestration`)
because no runnable public scheduler contract is available.

## Validation

Use `benchmarks/browser_smoke.jsonl` for browser-context profiles and
`benchmarks/flat_parallel_smoke.jsonl` for the Browser-Use parallel pattern. For
example:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/configs/openai-hosted-browser-functions.json \
  --provider openai-responses
```

Hosted and upstream providers need their provider-specific run flags. MACU must
run one task per invocation against a disposable, pinned OSWorld 1.x stack.
