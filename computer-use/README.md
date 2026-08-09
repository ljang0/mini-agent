# Computer-use application

This is the application-first entry point for pixel-action computer runs.
`implementations/` contains exact published client protocols or pinned upstream
runtimes. `studies/` contains composed or clean-room experiments that have an
unverified boundary and therefore cannot be called 1:1.

## Reproducible implementations

| Config | Provider | Fidelity | Boundary |
| --- | --- | --- | --- |
| [`openai-ga-single.json`](implementations/openai-ga-single.json) | `openai-responses` | published protocol boundary | Ordered GA `computer_call.actions`, one updated original-detail screenshot, and `computer_call_output` continuation. |
| [`anthropic-20251124-single.json`](implementations/anthropic-20251124-single.json) | `anthropic-messages` | published protocol boundary | Exact `computer-use-2025-11-24` wire schema, request/tool-result ordering, and `computer_20251124` action vocabulary. Actions execute in Scaffold Lab's Playwright viewport, not Anthropic's reference Linux X11/VNC environment. |
| [`macu-upstream-generic-vm.json`](implementations/macu-upstream-generic-vm.json) | `macu-upstream` | pinned upstream runtime | Executes the released MACU manager/CUA/VM runtime for a generic blank-start task. It does not load an OSWorld domain/UUID task or invoke the benchmark evaluator. |

Exact client-protocol runs fail before network access when `--model` is outside the
currently documented families: GPT-5.4/GPT-5.6 for OpenAI GA computer, and Opus 5,
Sonnet 5, Opus 4.8/4.7/4.6/4.5, or Sonnet 4.6 for Anthropic
`computer_20251124`.

## Studies, not implementations

- [`openai-hosted-multi-agent.json`](studies/openai-hosted-multi-agent.json)
  composes two documented protocols, but the combined hosted-multi-agent/computer
  compatibility boundary is not published or live-verified.
- [`macu-text-dag.json`](studies/macu-text-dag.json) preserves local DAG scheduling
  while omitting MACU's released prompts, CUA subprocesses, VMs, files, and evaluator.

The local `computer` environment is a Playwright browser viewport, not a full
desktop VM and not OSWorld. The upstream MACU implementation uses its released
OSWorld-derived VM/CUA substrate, but its Scaffold Lab task boundary is generic.
`computer-use/macu-osworld1-benchmark-parity` records the missing domain/UUID task
map, canonical loader/setup, evaluator, data-directory pin, and VM/task assets.
OSWorld 2.0 is separately cataloged as `computer-use/osworld2-2026-06-24`; its
gated evaluator assets and pinned VM image are not bundled.

Anthropic Managed Agents is not listed here because its built-in managed toolset
does not publish native `computer_20251124` as a server-owned tool. A client-defined
GUI tool would be a different boundary.

## Validation

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/computer_smoke.jsonl \
  --config computer-use/implementations/openai-ga-single.json \
  --provider openai-responses
```

`allowed_hosts` constrains requested top-level navigation only. Redirects,
subresources, and URLs typed through pixels still require an outer network
sandbox. Consequential actions require an independent human-confirmation gate.
