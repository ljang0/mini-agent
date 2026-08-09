# Scaffold Lab

Scaffold Lab is an application-first, provider-neutral experiment runtime for
comparing multi-agent systems under shared model/tool budgets, evaluators, traces,
and fidelity labels. The only top-level applications are:

| Application | Entry point | Exact implementations, studies, and gaps |
| --- | --- | --- |
| Browser | [`browser/`](browser/) | Exact OpenAI/xAI/Anthropic hosted boundaries and pinned MACU live in `implementations/`; Browser-Use/MACU clean-room variants and card topologies live in `studies/`. |
| Computer use | [`computer-use/`](computer-use/) | Exact OpenAI and Anthropic client protocols plus pinned generic MACU VM runtime; unverified compositions are studies and OSWorld parity is a gap. |
| SWE | [`swe/`](swe/) | Exact hosted/CLI boundaries and clean pinned upstream runtimes are implementations; best-of-N, local RLM/Platoon/MACU, and system-card topologies are studies; training/evaluators remain gaps. |

The word `implementation` is reserved. It means either an exact published API/tool or
CLI protocol boundary, or execution of a clean revision-pinned upstream runtime. Local
reconstructions, baselines, and topology simulations must use `study` instead. The
registry injects the registered harness signature and rejects category, provider,
environment, or harness drift:

```json
{
  "application": {
    "name": "browser",
    "implementation": "openai-hosted-web-search"
  }
}
```

```json
{
  "application": {
    "name": "swe",
    "study": "parallel-best-of-3"
  }
}
```

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli list-applications
PYTHONPATH=src python3 -m scaffoldlab.cli list-implementations \
  --application browser --json
PYTHONPATH=src python3 -m scaffoldlab.cli list-studies --application browser
PYTHONPATH=src python3 -m scaffoldlab.cli list-gaps --application browser
```

“1:1” is always scoped. `exact_public_protocol` means the observable published API
wire/client loop or released CLI stream contract, not the scheduler behind it.
`upstream_runtime_adapter` means a clean source checkout is actually executed at its
recorded revision. Closed schedulers, withheld prompts, unverified executable/package
identity, missing evaluator images, and incomplete accounting remain visible rather
than being guessed.

The root `configs/` directory remains for backward-compatible controls and smoke
tests. Those unselected runs are intentionally **uncataloged**: they do not receive
an implementation identity or a 1:1 claim. Release-facing runs should select a
profile from an application's `implementations/` or `studies/` directory.

## Shared harness primitives

The application profiles resolve to these reusable mechanisms:

| Harness | Mechanism | Fidelity label |
| --- | --- | --- |
| `single` | One solver | Baseline |
| `flat_parallel` | Distinct independent tasks through flat fan-out | Browser-Use scheduling pattern with optional isolated browser profiles; no selector |
| `parallel_best_of_n` | Same task N times, followed by an explicit judge | Scaffold Lab baseline |
| `blocking_orchestrator` | Tool-less manager, fresh tool-capable workers, one whole-round barrier | Mythos-5-card-inspired control simulation; no card-exact compaction or prompts |
| `fixed_agent_team` | Long-lived logical peers, full task visibility, JSON messaging, lead submission | Mythos/Opus-card topology simulation |
| `async_subagents` | Dynamic logical workers with wait/wake/delete/status | Mythos/Opus-card topology simulation |
| `macu_dynamic_dag` | Mutable text DAG, ready-frontier scheduling, replan after observations | Source-matched MACU subset with domain tools |
| `macu_upstream` | Released MACU manager, graph, CUA, VM, replanning, and artifact runtime | Pinned upstream adapter; released accounting omits initial graph-generation usage |
| `recursive_delegation` | One policy recursively delegates through JSON actions | Generic recursion control; neither RAO inference/training nor code-first RAH |
| `rlm_repl` | Persistent restricted Python namespace with external context and recursive/batched subcalls | Source-matched restricted RLM subset; not the upstream runtime |
| `rlm_upstream` | Released `rlms` RLM/REPL runtime in a bounded external process | Pinned 0.1.3 adapter; root `UsageSummary` is a lower bound for recursive trees |
| `platoon_recursive_inference` | REPL-driven sequential/parallel child agents with bounded recursion | Platoon/RAO inference shape only; explicitly not RAO policy training |
| `external_context_json_search` | Context stays external; controller inspects, searches, and makes bounded subcalls | RLM-motivated ablation; explicitly not an RLM or REPL |
| `openai_hosted_multi_agent` | Responses hosted multi-agent beta with optional developer tools | Exact public stateless HTTP request/continuation boundary; server scheduler closed |
| `anthropic_managed_agents` | Managed Agents coordinator session | Exact public session, primary-event-list, and aggregate-usage boundary; server scheduler and sandbox hosted |
| `prime_agent` | Released Prime Agent JSON runtime | Exact JSON v3 CLI boundary at version 0.7.1; executable identity is optional and runtime internals remain outside the claim |
| `grok_build` | Released xAI Grok Build runtime with native subagents | Exact headless-JSON CLI boundary at version 1.0.0; npm/source identity and scheduler internals remain outside the claim |
| `xai_hosted_multi_agent` | xAI hosted 4- or 16-agent team with optional hosted web/X search | Exact public request boundary; plaintext state hidden, encrypted continuation unimplemented |

No paid live-model comparison has been run in this repository yet. The included
offline suite validates scheduling, messaging, stopping, invalid actions, provider
continuations, environment isolation, and accounting. Smoke tasks still cannot
identify a release winner.

`configs/fable_card_simulations.json` and
`configs/opus5_card_simulations.json` preserve the source-specific topology sets.
The former keeps its legacy filename for compatibility, but §8.15's multi-agent
evaluation uses Claude Mythos 5; it is not a Fable 5 result.
They remain simulations: public Messages tool schemas are available, but the local
runtime does not reproduce the cards' exact tool configuration, source-specific
per-agent token/context limits, compaction, message-injection timing, or Git
worktrees. In the Mythos profile, `blocking_orchestrator.max_workers=4` is a
local control choice; the card's four-concurrent/twenty-total async limit is specific
to ProgramBench, while its BrowseComp async configuration discloses no spawn cap.
The Opus async simulation likewise uses a local safety cap although the card states
none. Both card-profile configs intentionally set `capture_content: true` for trace
audit; change that before using sensitive tasks.

## Application, environment, and provider coverage

Eligible in-process profiles can be paired with one of four local environment
types. Anthropic Managed Agents, xAI hosted multi-agent, Prime Agent, Grok Build,
MACU upstream, and RLM upstream own their execution environment and reject a local
`config.environment`; OpenAI hosted multi-agent may use local developer tools
through its HTTP continuation loop.

| Application surface | Representative config | State boundary |
| --- | --- | --- |
| Browser | `browser/implementations/openai-hosted-browser-functions.json` | Semantic tools on an isolated Playwright context per agent |
| Computer use | `computer-use/implementations/openai-ga-single.json` | Provider-native client protocol over a Playwright browser viewport |
| SWE implementation | `swe/implementations/prime-agent-0.7.1.json` | Published JSON v3 runtime protocol over a caller-isolated workspace |
| SWE study | `swe/studies/parallel-best-of-3.json` | Independent workspace copy per agent, fresh Git baseline, portable file tools, and bounded patch export |
| SWE with computer tools study | `swe/studies/openai-hosted-swe-computer.json` | Composite repository and pixel environment per logical agent |

Provider adapters cover OpenAI Responses, Anthropic Messages, generic
OpenAI-compatible Responses and Chat Completions, xAI's hosted multi-agent API,
Anthropic Managed Agents, Prime Agent, Grok Build, pinned MACU, and pinned RLM.
“OpenAI-compatible” means portable JSON function calling; it does not imply
OpenAI-native computer/shell tools or the model vendor's hosted scheduler.

The Playwright computer driver is useful for protocol tests and browser-contained
computer tasks. Use an actual VM plus the pinned OSWorld evaluator before making an
OSWorld claim. Likewise, release-quality browser and SWE scores require the intended
BrowserGym or SWE-bench task reset and evaluator, not merely these tool surfaces.

Native tool support is model-specific. In particular, Anthropic's documented
`computer_20251124` compatibility list includes Opus 5 but not Fable 5 or Mythos 5.
Exact computer runs enforce the documented model-family prefixes. The Fable/Mythos
topology configs therefore do not imply native-computer compatibility.
The OpenAI hosted adapter's developer-function path is documented; native computer or
shell combined with hosted multi-agent remains a separate compatibility surface to
validate live before making an exact combined-protocol claim.

The RLM and Platoon candidates execute restricted Python in-process. Their AST and
capability checks block known namespace/I/O escapes, but they cannot reliably stop
memory exhaustion or expensive finite computation. Put adversarial or paid runs in
an outer process/container sandbox. Browser `allowed_hosts` validates requested
top-level navigations; redirects and subresources still require network-level egress
controls.

The SWE example configs deliberately omit a config-level workspace; their smoke tasks
point to `fixtures/swe_smoke_repo`. Never use `workspace: "."` for a paid or holdout
run. The runner rejects an output directory that overlaps any configured source
workspace before writing the manifest. Copy mode excludes inherited Git metadata and
commits the copied snapshot as a fresh, clean baseline, so pre-existing dirty and
untracked source files are treated as task input rather than candidate edits. With
`export_patch: true`, each successful trial writes a bounded binary Git patch before
the temporary copy is removed; patch bytes are not embedded in results or traces.
The source can still contain old artifacts, evaluators, or reference answers, so use a
clean frozen repository. Native shell is disabled in most examples; the portable
executable allowlist is still not a security sandbox—`python3` can execute arbitrary
code. Native shell or bash requires explicit opt-in plus an outer process/network
sandbox.

## Quick start

Run the offline tests and validate the example matrix:

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider openai-responses
```

Validate an application profile without manually specifying its harness:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/implementations/openai-hosted-browser-functions.json \
  --provider openai-responses
```

Install the optional browser runtime when running browser or computer environments:

```bash
python3 -m pip install -e '.[browser]'
playwright install chromium
```

Validate each domain without spending model credits:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/browser_smoke.jsonl \
  --config configs/domain_browser.json \
  --provider openai-responses
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/swe_smoke.jsonl \
  --config configs/domain_swe.json \
  --provider anthropic-messages
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/computer_smoke.jsonl \
  --config configs/domain_computer.json \
  --provider openai-responses
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/swe_computer_smoke.jsonl \
  --config configs/domain_swe_computer.json \
  --provider anthropic-messages
```

The distinct-task Browser-Use-style throughput profile uses its own benchmark shape:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli validate \
  --tasks benchmarks/flat_parallel_smoke.jsonl \
  --config configs/flat_parallel.json \
  --provider openai-responses
```

Run a model-backed matrix with explicit pricing so cost is comparable:

```bash
export OPENAI_API_KEY=...
export SCAFFOLDLAB_SOURCE_REVISION="$(git rev-parse HEAD)"
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider openai-responses \
  --model YOUR_PINNED_MODEL_ID \
  --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION \
  --output runs/smoke-openai
```

When cached input has different pricing, also pass `--cache-read-price` and/or
`--cache-write-price`; otherwise the input rate is used for those classes. Cache
token classes are preserved separately in result artifacts.

For Meta, Qwen, Kimi, Mistral, DeepSeek, MiniMax, Cohere, local, or routed models
that publish an OpenAI-compatible function-calling endpoint, use
`openai-compatible-responses` or `openai-compatible-chat` with an explicit
`--base-url` and `--api-key-env`. This runs the selected Scaffold Lab topology; it
does not reproduce that model vendor's private product coordinator.

Anthropic Messages is also supported:

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/smoke.jsonl \
  --config configs/smoke.json \
  --provider anthropic-messages \
  --model YOUR_PINNED_MODEL_ID \
  --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION \
  --output runs/smoke-anthropic
```

Anthropic Managed Agents uses a separately created and versioned remote coordinator
and environment. The referenced agent owns its roster, tools, prompts, and model;
Scaffold Lab owns the outer task, session observation, local post-session accounting,
and artifacts. Only `--managed-budget-cents` creates a provider-enforced session cap;
the other Scaffold Lab limits do not constrain server-side turns, tools, depth, or
thread concurrency:

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/smoke.jsonl \
  --config swe/implementations/anthropic-managed-agents.json \
  --provider anthropic-managed-agents \
  --managed-agent-id agent_... \
  --managed-agent-version 1 \
  --managed-environment-id env_... \
  --managed-budget-cents 2000 \
  --managed-cleanup retain \
  --output runs/anthropic-managed
```

The adapter supports server-owned tools configured with automatic permission. It
fails closed on custom-tool and confirmation `requires_action` events because their
application callbacks are not implemented in this release.

The example keeps `capture_content: false`, so it retains event counts and local
lifecycle summaries rather than raw Managed primary events. Enable content capture
only for controlled non-sensitive tasks; even then, child-thread event streams are not
fetched by this adapter.

Managed session budgets are shared across all session threads and use public-list-price
`list_cost`. Enforcement occurs between model requests, so the request that crosses the
cap finishes and can leave the final total slightly above it. Release runs must pass
`--managed-agent-version`, and Scaffold Lab rejects an unversioned Managed invocation.
The current manifest records the agent/environment identifiers and resolved agent
version, but not a sanitized hash of the complete remote model, prompts, tools, roster,
or environment definition. Export and hash those definitions separately until artifact
capture gains that snapshot. `retain` preserves a resumable session for audit.
`archive` prevents new events while retaining history, and `delete` permanently removes
the session record, events, and sandbox. A session that stops at `budget_reached` is
recorded as an incomplete/error trial; the adapter does not automatically raise the cap
or resume it.

The documented OpenAI hosted multi-agent beta is represented at its public request
boundary, including stateless developer-tool continuation and agent attribution.
The server scheduler remains closed, so this is not a local reimplementation of its
internal coordination loop:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/implementations/openai-hosted-web-search.json \
  --provider openai-responses \
  --model YOUR_PINNED_GPT_5_6_MODEL_ID \
  --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION \
  --output runs/openai-hosted
```

This command selects the hosted `web_search` request boundary. The separate
`browser/implementations/openai-hosted-browser-functions.json` profile pairs the same
harness with semantic browser developer functions and allows the additional outer calls
required for continuation. Both profiles have deterministic offline protocol coverage;
neither is evidence from a paid live-model comparison, and the developer-function
profile does not claim a combined native computer/shell surface.

The adapter currently uses non-streaming stateless HTTP. OpenAI recommends WebSocket
for tool-heavy or long-running workflows; over HTTP, the response waits until all active
agents finish or pause, after which the client must return every pending function result
in a new request. The current local loop executes those pending developer calls
sequentially. Treat HTTP latency as a separate transport condition. Local
`max_concurrency` and `max_depth` do not cap the hosted tree: the provider's
`max_concurrent_subagents` field caps active descendants, while the public beta documents
no fixed tree-depth or total-subagent limit.

To test the released Prime Agent runtime, install the audited 0.7.1 executable and
pass short-lived provider credentials explicitly; cached login state is not inherited.
The CLI checks 0.7.1 by default. Use exactly one task in a caller-provisioned
disposable worktree. The worktree and output directory must be disjoint:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/external_smoke.jsonl \
  --config swe/implementations/prime-agent-0.7.1.json \
  --provider prime-agent \
  --prime-agent-provider openai \
  --model YOUR_PINNED_MODEL_ID \
  --prime-agent-cwd /path/to/disposable-worktree \
  --prime-agent-pass-env OPENAI_API_KEY \
  --prime-agent-allow-sensitive-environment \
  --prime-agent-expected-version 0.7.1 \
  --output runs/prime-agent
```

Prime Agent executes model-generated Python and shell commands with the current
user's permissions. Its worker lifecycle is not a security sandbox. Use scoped
credentials and an outer network/process sandbox. For stronger executable identity,
also pass `--prime-agent-executable-sha256` with the SHA-256 of the resolved binary.
The adapter requires the documented JSON v3 session header and terminal `agent_end`
event. Every call uses upstream `--no-session` with fresh home/config directories;
Python and child-agent state persists within that call, but daemon, schedule, and
cross-invocation continual state are not reproduced. The legacy
`--prime-agent-persist-session` flag is rejected instead of claiming persistence that
the isolated lifecycle cannot provide.
The adapter preserves observed root-message tokens/cost as lower bounds but marks
whole-tree usage incomplete and cost unknown; therefore a successful exploratory
answer still exits nonzero at Scaffold Lab's clean-release accounting gate.

To run the released Grok Build harness, install a pinned CLI (the audit target is
`@xai-official/grok@1.0.0`) and pass authentication explicitly. Scaffold Lab uses a
private prompt file, a fresh ephemeral `GROK_HOME`, disables memory/update checks,
and defaults to xAI's `strict` sandbox plus `dontAsk` permission mode:

```bash
export XAI_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/external_smoke.jsonl \
  --config swe/implementations/grok-build-1.0.0.json \
  --provider grok-build \
  --model grok-build \
  --grok-cwd /path/to/disposable-worktree \
  --grok-pass-env XAI_API_KEY \
  --grok-expected-version 1.0.0 \
  --grok-allow 'Read(**)' \
  --grok-allow 'Grep(**)' \
  --output runs/grok-build
```

This is also a one-task, one-variant, one-repeat invocation in a caller-provisioned
workspace. Terminal execution is disabled by default with upstream tool filtering.
If a coding task requires it, add `--grok-enable-terminal` and
`--grok-allow-sensitive-environment`, and run the CLI inside an outer sandbox that
prevents child processes from inheriting or exfiltrating credentials. The upstream
`strict` profile is recorded but is not a substitute for that outer boundary. For a
stronger executable pin, add `--grok-executable-sha256`.

Add write permission rules only when a task needs them. The upstream JSON
usage includes completed subagents, but may exclude compaction, side-model, and
unfinished nested calls; artifacts label that scope instead of claiming exact
whole-tree accounting. The adapter is faithful to the documented public CLI
surface, not a reimplementation of the upstream orchestration policy. As with Prime,
that conservative accounting makes exploratory runs fail the clean-release gate even
when the upstream runtime returns a usable answer.

The pinned MACU adapter runs the released manager, graph scheduler, CUA workers,
VM variants, replanning loop, and result ingestion for one generic blank-start
task. Supply an immutable checkout at the recorded revision, a compatible
OSWorld-derived VM/CUA stack, and a separate result directory:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/external_smoke.jsonl \
  --config computer-use/implementations/macu-upstream-generic-vm.json \
  --provider macu-upstream \
  --macu-checkout /path/to/pinned-macu \
  --macu-osworld-root /path/to/pinned-osworld \
  --macu-result-dir /path/to/macu-results \
  --macu-manager-provider anthropic \
  --macu-manager-model YOUR_PINNED_MANAGER_MODEL \
  --macu-cua-provider openai \
  --macu-pass-env ANTHROPIC_API_KEY \
  --macu-pass-env OPENAI_API_KEY \
  --macu-allow-sensitive-environment \
  --output runs/macu-upstream
```

The default revision pin is `5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105`.
Pass any model-specific released CUA flags as repeated `--macu-cua-arg` values.
MACU's published summary omits initial graph-generation usage, so the adapter keeps
whole-tree accounting incomplete. This generic path does not supply an OSWorld
domain/UUID task map, enter the benchmark's canonical task setup, or invoke its
evaluator. That missing boundary is recorded separately as
`computer-use/macu-osworld1-benchmark-parity` rather than treating a VM dependency
as an evaluation result.

The separate upstream RLM profile executes the official 0.1.3 prompts and selected
REPL from a clean checkout. Scaffold Lab deliberately defaults this adapter to the
upstream Docker REPL with a 1,500-second RLM timeout; the library itself defaults to
the local REPL with no timeout. Pin the Python executable, container, and dependencies
as part of any reproducible run:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/external_smoke.jsonl \
  --config swe/implementations/rlm-0.1.3-upstream.json \
  --provider rlm-upstream \
  --rlm-checkout /path/to/pinned-rlm \
  --rlm-provider openai \
  --rlm-model YOUR_PINNED_MODEL_ID \
  --rlm-pass-env OPENAI_API_KEY \
  --rlm-allow-sensitive-environment \
  --output runs/rlm-upstream
```

The default revision pin is `72d6940142ddfb84ee6be573dc999a37e633e671`.
At the adapter default `max_depth=1`, `rlm_query` falls back to `llm_query`; configure
`--rlm-max-depth 2` or greater to create recursive child RLMs.
The v0.1.3 root usage summary does not merge recursively created child handlers, so
reported calls, tokens, and cost remain lower bounds. This adapter deliberately
injects no Scaffold Lab SWE/browser/computer tools and therefore makes no SWE patch
parity claim. It covers the official runtime's string-context path and selected
backend/environment/limit settings; structured context, custom tools, compaction,
persistence, alternate backends, and sampling/orchestrator options are not exposed.

xAI's separate hosted research team is also represented at its exact public API
boundary. The config below runs the documented four-agent (`low`) setting with hosted
`web_search`; `browser/implementations/xai-hosted-web-research-16.json` provides the
sixteen-agent (`high`) counterpart, and the harness also supports `x_search`. The API
returns the leader result and aggregate usage, while plaintext intermediate state is
hidden. xAI documents encrypted continuation state, but continuation is not
implemented in this adapter:

```bash
export XAI_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/browser_smoke.jsonl \
  --config browser/implementations/xai-hosted-web-research-4.json \
  --provider xai-responses \
  --model grok-4.20-multi-agent-0309 \
  --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION \
  --output runs/xai-hosted
```

## Outputs

Each matrix writes:

- `manifest.json`: tasks, evaluators, harness variants, configured provider settings
  and, where selected locally, model/pricing, behavior-affecting timeouts, limits,
  selected dependency versions, the
  `src/scaffoldlab/**/*.py` package-source hash, environment configuration, external
  executable/workspace provenance, and run fingerprint;
- `results.jsonl`: one durable, atomically replaced record set with one entry per
  task, harness, and repeat; completed entries include a machine-readable fidelity
  label plus model/tool counts and tool-output bytes;
- `traces/*.jsonl`: locally observable model-call, tool-call, message, and harness
  lifecycle events; hosted child-thread details appear only when the provider exposes
  them and the adapter persists them;
- `patches/*.patch`: bounded `git diff --binary` artifacts for local SWE profiles
  that enable `export_patch`, with path, byte count, and SHA-256 stored in result
  metadata rather than inline patch content;
- `summary.json`: pass rate with Wilson intervals, errors, tokens, cost, raw wall
  time, and the union of backend-active intervals.

Traces contain content hashes by default. Set `capture_content: true` in the config
or pass `--capture-content` to retain prompts, responses, and raw provider events for
human audit; those artifacts may contain sensitive task data. Output directories are
not overwritten unless `--overwrite` is explicit. Per-trial limits and the optional
`matrix_max_cost_usd` are reported separately. Token and dollar ceilings stop future
calls after observed usage crosses them; they cannot prevent a single provider response
from overshooting a ceiling.

`cost_usd` is a normalized experiment field, not always an invoice charge. OpenAI and
Anthropic Messages use provider-returned cost only when available; otherwise the CLI
price flags produce an estimate. Managed Agents reports cumulative public-list-price
`list_cost`. Interpret comparisons only with the recorded accounting scope and price
source.

For in-process harnesses, `max_model_calls` counts every root and child backend
invocation. For OpenAI hosted multi-agent, Anthropic Managed Agents, xAI hosted
multi-agent, Prime Agent, Grok Build, MACU upstream, and RLM upstream, one Scaffold
Lab call is one outer request/session that may contain many closed or
upstream-managed calls. Anthropic
reports authoritative aggregate session token usage and public-list-price cost, but the
primary event list is a condensed view rather than a complete child-thread trace. Other
candidates must not enter an equal-total-compute claim unless their complete inner tree
is observable. External CLI runs also record the caller workspace's Git state plus
pre/post tree hashes, and are restricted to one trial per invocation to prevent
filesystem contamination.

The summary always says `HUMAN_REVIEW_REQUIRED`. A validator pass is internal QA,
not release approval.

## Research notes

- [Source and fidelity audit](docs/source-audit.md)
- [18-lab implementation coverage](docs/frontier-lab-coverage.md)
- [Controlled experiment protocol](docs/experiment-protocol.md)
- [Architecture and extension points](docs/architecture.md)

The audit incorporates the user-supplied August 2026 frontier-harness brief, then
checks material claims against first-party papers, repositories, documentation,
and system cards. One correction to our initial read is important: the brief was
right about Claude Opus 5. Anthropic publishes a direct official
[Opus 5 system-card PDF](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf);
it is audited separately because its disclosed harness suite differs from the
Mythos 5 evaluation in the joint Fable/Mythos card.
