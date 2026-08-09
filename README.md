# Scaffold Lab

Scaffold Lab is a provider-neutral experiment runtime for comparing multi-agent
inference topologies across browser, SWE, computer-use, and hybrid SWE+computer
tasks under shared model/tool budgets, evaluators, and traces.

The repository starts with runnable versions of the mechanisms that are commonly
collapsed into the phrase “parallel agents”:

| Harness | Mechanism | Fidelity label |
| --- | --- | --- |
| `single` | One solver | Baseline |
| `flat_parallel` | Distinct independent tasks through flat fan-out | Browser-Use scheduling pattern with optional isolated browser profiles; no selector |
| `parallel_best_of_n` | Same task N times, followed by an explicit judge | Scaffold Lab baseline |
| `blocking_orchestrator` | Tool-less manager, fresh tool-capable workers, one whole-round barrier | Fable-inspired control simulation; no card-exact compaction or prompts |
| `fixed_agent_team` | Long-lived logical peers, full task visibility, JSON messaging, lead submission | Fable/Opus-card topology simulation |
| `async_subagents` | Dynamic logical workers with wait/wake/delete/status | Fable/Opus-card topology simulation |
| `macu_dynamic_dag` | Mutable text DAG, ready-frontier scheduling, replan after observations | MACU-inspired DAG with domain tools; not pinned MACU/OSWorld artifact parity |
| `recursive_delegation` | One policy recursively delegates through JSON actions | Generic recursion control; neither RAO inference/training nor code-first RAH |
| `rlm_repl` | Persistent restricted Python namespace with external context and recursive/batched subcalls | Clean-room implementation of the public RLM algorithm; not byte-identical upstream `rlms` |
| `platoon_recursive_inference` | REPL-driven sequential/parallel child agents with bounded recursion | Platoon/RAO inference shape only; explicitly not RAO policy training |
| `external_context_json_search` | Context stays external; controller inspects, searches, and makes bounded subcalls | RLM-motivated ablation; explicitly not an RLM or REPL |
| `openai_hosted_multi_agent` | Responses hosted multi-agent beta with optional developer tools | Exact public stateless HTTP request/continuation boundary; server scheduler closed |
| `anthropic_managed_agents` | Managed Agents coordinator session | Exact public session, primary-event-list, and aggregate-usage boundary; server scheduler and sandbox hosted |
| `prime_agent` | Released Prime Agent JSON runtime | Upstream CLI wrapper; observed root-stream usage is a lower bound, full-tree accounting unverified |
| `grok_build` | Released xAI Grok Build runtime with native subagents | Upstream headless-JSON CLI wrapper; reported usage is a lower bound, upstream scheduler retained |
| `xai_hosted_multi_agent` | xAI hosted 4- or 16-agent team with a designated leader | Exact public no-tool request boundary; plaintext state hidden, encrypted continuation unimplemented |

No paid live-model comparison has been run in this repository yet. The included
offline suite validates scheduling, messaging, stopping, invalid actions, provider
continuations, environment isolation, and accounting. Smoke tasks still cannot
identify a release winner.

`configs/fable_card_simulations.json` and
`configs/opus5_card_simulations.json` preserve the source-specific topology sets.
They remain simulations: public Messages tool schemas are available, but the local
runtime does not reproduce the cards' exact tool configuration, source-specific
per-agent token/context limits, compaction, message-injection timing, or Git
worktrees. In the Fable profile, `blocking_orchestrator.max_workers=4` is a
local control choice; the card's four-concurrent/twenty-total async limit is specific
to ProgramBench, while its BrowseComp async configuration discloses no spawn cap.
The Opus async simulation likewise uses a local safety cap although the card states
none. Both card-profile configs intentionally set `capture_content: true` for trace
audit; change that before using sensitive tasks.

## Domain and provider coverage

Eligible in-process harnesses can be paired with one of four local environment
configs. Anthropic Managed Agents, xAI hosted multi-agent, Prime Agent, and Grok
Build own their tool environment and reject `config.environment`; OpenAI hosted
multi-agent may use local developer tools through its HTTP continuation loop.

| Domain | Config | State boundary |
| --- | --- | --- |
| Browser | `configs/domain_browser.json` | Semantic tools on an isolated Playwright context per agent |
| SWE | `configs/domain_swe.json` | Configured workspace copy per agent; portable file tools plus supported provider-native schemas |
| Computer use | `configs/domain_computer.json` | Provider-native client protocol for supported model/tool-version pairs over a Playwright browser viewport |
| SWE + computer | `configs/domain_swe_computer.json` | Composite repository and pixel environment per logical agent |

Provider adapters cover OpenAI Responses, Anthropic Messages, generic
OpenAI-compatible Responses and Chat Completions, xAI's hosted multi-agent API,
Anthropic Managed Agents, Prime Agent, and Grok Build. “OpenAI-compatible” means
portable JSON function calling; it does not imply OpenAI-native computer/shell tools
or the hosted multi-agent scheduler.

The Playwright computer driver is useful for protocol tests and browser-contained
computer tasks. Use an actual VM plus the pinned OSWorld evaluator before making an
OSWorld claim. Likewise, release-quality browser and SWE scores require the intended
BrowserGym or SWE-bench task reset and evaluator, not merely these tool surfaces.

Native tool support is model-specific. In particular, Anthropic's documented
`computer_20251124` compatibility list includes Opus 5 but not Fable 5 or Mythos 5.
The Fable/Mythos topology configs therefore do not imply native-computer compatibility.
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
workspace before writing the manifest. Copy mode still copies the selected source tree
verbatim, so a dirty source can carry older artifacts, evaluators, or reference answers
into an agent workspace. Provision a clean frozen task repository disjoint from the
run-artifact directory. Native shell is
disabled in the examples; their portable executable allowlist is still not a security
sandbox—`python3` can execute arbitrary code. Native shell or bash requires explicit
opt-in plus an outer process/network sandbox.

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
  --config configs/anthropic_managed_agents.json \
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
  --tasks benchmarks/smoke.jsonl \
  --config configs/openai_hosted_multi_agent.json \
  --provider openai-responses \
  --model YOUR_PINNED_GPT_5_6_MODEL_ID \
  --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION \
  --output runs/openai-hosted
```

This command uses `configs/openai_hosted_multi_agent.json`, a tool-less one-call
request-boundary smoke profile. `configs/openai_hosted_developer_tools.json` pairs the
same harness with semantic browser developer functions and allows the additional outer
calls required for continuation. Both profiles have deterministic offline protocol
coverage; neither is evidence from a paid live-model comparison, and the developer-tool
profile does not claim a combined native computer/shell surface.

The adapter currently uses non-streaming stateless HTTP. OpenAI recommends WebSocket
for tool-heavy or long-running workflows; over HTTP, the response waits until all active
agents finish or pause, after which the client must return every pending function result
in a new request. The current local loop executes those pending developer calls
sequentially. Treat HTTP latency as a separate transport condition. Local
`max_concurrency` and `max_depth` do not cap the hosted tree: the provider's
`max_concurrent_subagents` field caps active descendants, while the public beta documents
no fixed tree-depth or total-subagent limit.

To test the released Prime Agent runtime, install a pinned executable and pass
short-lived provider credentials explicitly; cached login state is not inherited.
Use exactly one task in a caller-provisioned disposable worktree. The worktree and
output directory must be disjoint:

```bash
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/external_smoke.jsonl \
  --config configs/prime_agent.json \
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
  --config configs/grok_build.json \
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

xAI's separate hosted research team is also represented at its exact public API
boundary. This config runs the documented four-agent (`low`) and sixteen-agent
(`high`) settings without built-in tools, so unimplemented tool fees do not pollute
the cost comparison. The API returns the leader result and aggregate usage; plaintext
intermediate agent state is hidden. xAI documents encrypted continuation state, but
continuation is not implemented in this adapter:

```bash
export XAI_API_KEY=...
PYTHONPATH=src python3 -m scaffoldlab.cli run \
  --tasks benchmarks/smoke.jsonl \
  --config configs/xai_hosted_multi_agent.json \
  --provider xai-responses \
  --model grok-4.20-multi-agent \
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
multi-agent, Prime Agent, and Grok Build, one Scaffold Lab call is one outer
request/session that may contain many closed or upstream-managed calls. Anthropic
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
right about Claude Opus 5. Anthropic's official system-card index includes the July
2026 Opus 5 card; it is audited separately from Fable/Mythos because their harness
suites differ.
