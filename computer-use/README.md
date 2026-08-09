# Computer-use application

This directory is the application-first entry point for pixel-action computer
experiments. Every config uses the canonical nested application selection and
omits `harnesses`, allowing the registry to inject and validate the exact profile
signature.

## Runnable representatives

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`openai-ga-single.json`](configs/openai-ga-single.json) | `openai-responses` | exact public protocol | Ordered GA `computer_call.actions`, one updated original-detail screenshot, and `computer_call_output` continuation. Model training, full confirmation prompt, classifiers, and product runtime are not public. |
| [`openai-hosted-multi-agent.json`](configs/openai-hosted-multi-agent.json) | `openai-responses` | source-matched reimplementation | Composes two published client protocols, hosted multi-agent and GA computer use. A combined compatibility contract has not been published or live-verified, so this is not labeled exact. |
| [`anthropic-20251124-single.json`](configs/anthropic-20251124-single.json) | `anthropic-messages` | exact public protocol | `computer-use-2025-11-24`, `computer_20251124`, assistant `tool_use`, user `tool_result`, and the documented action vocabulary. Zoom stays disabled. |
| [`macu-text-dag.json`](configs/macu-text-dag.json) | `openai-responses` or `anthropic-messages` | source-matched reimplementation | Local text-DAG scheduling over computer tools; released MACU prompts, CUA subprocesses, VM cloning, files, and evaluator are absent. |
| [`macu-upstream-osworld1.json`](configs/macu-upstream-osworld1.json) | `macu-upstream` | pinned upstream runtime adapter | Executes the released MACU runtime on its OSWorld 1.x stack. Full identity still requires pinned dependencies, VM image, task assets, and model snapshots. |

The local `computer` environment is a Playwright browser viewport, not a full
desktop VM and not OSWorld. OSWorld 2.0 is separately cataloged as the
non-runnable evaluation-environment profile
`computer-use/osworld2-2026-06-24`; its gated evaluator assets and pinned VM image
are not bundled. Do not mix OSWorld 2.0 scores with MACU's OSWorld 1.x stack.

Anthropic Managed Agents is not listed here because its built-in managed toolset
does not publish native `computer_20251124` as a server-owned tool. A client-defined
GUI tool would be a different boundary.

## Validation

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/computer_smoke.jsonl \
  --config computer-use/configs/openai-ga-single.json \
  --provider openai-responses
```

`allowed_hosts` constrains requested top-level navigation only. Redirects,
subresources, and URLs typed through pixels still require an outer network
sandbox. Consequential actions require an independent human-confirmation gate.
