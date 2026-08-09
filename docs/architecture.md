# Architecture

Scaffold Lab is application-first. The stable top-level registry contains exactly
`browser`, `computer-use`, and `swe`. An application profile binds a cited source,
fidelity/status label, harness signature, compatible provider, and environment type.
Profiles are partitioned into `implementations/`, `studies/`, and source-backed
`gaps`. The registry rejects category drift, catalog-only execution, and any provider,
environment, or harness that does not match the selected profile.

The runtime still keeps the experimental objects separate because changing any one
of them changes the experiment:

```text
Application (browser | computer-use | swe)
          |
          v
Implementation or study profile + source/fidelity claim
          |
          v
Task + frozen evaluator
          |
          v
Harness / coordination policy
          |
          v
Provider protocol adapter ----> model or hosted agent service
          |
          v
Tool environment -------------> browser, repository, computer, or hybrid
          |
          v
Shared budget ledger + trace recorder
```

The canonical config selector is:

```json
{
  "application": {
    "name": "browser",
    "implementation": "openai-hosted-web-search"
  }
}
```

A non-exact comparison replaces `implementation` with `study`. An item is exposed as
an implementation only when its fidelity is `exact_public_protocol` or
`upstream_runtime_adapter`; this is enforced in the data model and CLI rather than
left to documentation convention.

When `harnesses` is omitted, the profile supplies its declared signature. An
explicit harness list must match exactly. `list-implementations` exposes only exact
published boundaries and pinned upstream runtimes. `list-studies`, `list-gaps`, and
`list-profiles` expose the remaining tiers. A profile marked `catalog_only` documents
a source or gap but cannot be run.

- A **harness** decides who works, what each agent sees, when work runs, how agents
  communicate, and who selects or synthesizes the answer.
- A **provider adapter** translates the common request contract into OpenAI
  Responses, Anthropic Messages, an OpenAI-compatible endpoint, or a released
  external runtime. It does not silently add an orchestration policy.
- A **tool environment** owns state and executes structured client tool calls. It is
  independent of the task's evaluator.
- An **evaluation environment** supplies task reset and scoring. BrowserGym,
  SWE-bench, and OSWorld belong here even when they also expose an execution API.

Profiles use seven deliberately narrow fidelity labels:

| Label | Meaning |
| --- | --- |
| `exact_public_protocol` | The documented API request/action/response or released CLI stream boundary is implemented at the named scope; scheduler/runtime internals and executable identity are included only when verified. |
| `upstream_runtime_adapter` | A clean source checkout at the cataloged revision owns its scheduler; Scaffold Lab wraps its non-interactive boundary and records only observable accounting. |
| `source_matched_reimplementation` | Public mechanics are reimplemented, with named missing runtime details. |
| `topology_simulation` | Only disclosed roles, limits, or communication shape are approximated. |
| `inference_only_reimplementation` | An inference control is implemented without the named training procedure or checkpoint. |
| `controlled_baseline` | A deliberately simple experimental control with no named-system parity claim. |
| `documented_gap` | A source, training method, or evaluation target is cataloged but not runnable as that artifact. |

The first two labels are the only implementation labels. The middle four are studies,
even when runnable. `documented_gap` is catalog-only. Exact public protocols reproduce
only the named public boundary; upstream adapters reproduce the released runtime only
under the recorded source revision and caller-supplied dependency/model/environment
pins. For an exact application selection, the applicable public identity is
authoritative: CLI versions, source revisions, beta versions, and documented
computer-model families must match it. Exact MACU
profiles also require a clean checkout; its dirty-checkout escape hatch remains
available only to legacy configs that make no catalog claim.

This separation prevents a Playwright browser from being called BrowserGym, a
SWE-bench Docker evaluator from being called an inference harness, or a provider's
native computer schema from being confused with a desktop VM.

## Domain runtime

`ModelRequest` and `ModelResponse` carry provider-neutral tool definitions, calls,
results, continuation state, and logical agent attribution. `RunContext.call` runs
the client-side tool loop. For in-process harnesses, every root and child model turn
and every client-executed tool call passes through the same `BudgetLedger` and
`TraceRecorder`. A hosted or external boundary contributes one observable outer call;
its internal turns and tools are governed and exposed by that upstream runtime.

The built-in environments are:

| Config type | State | Public protocol mapping | Important boundary |
| --- | --- | --- | --- |
| `browser` | Isolated Playwright context per agent or one shared context | Portable semantic function tools | Playwright is an execution substrate, not BrowserGym or Browser-Use policy parity |
| `swe` | Copy of the configured source workspace per agent by default | Portable tools; supported OpenAI local-shell or Anthropic bash/text-editor schemas when explicitly enabled | An executable allowlist is not a process sandbox; provider-native shell requires an outer sandbox |
| `computer` | Pixel actions and screenshot on a Playwright browser viewport | OpenAI GA `computer` and Anthropic `computer_20251124` for supported model/tool-version pairs | This is not a full desktop VM and therefore not OSWorld parity |
| `swe_computer` | Composite repository and computer sessions | Supported schemas from both domains | One config does not reproduce a lab's hidden prompts, model compatibility, or scheduler |

`workspace_mode: copy` is the default for SWE state. With `isolation: per_agent`,
each logical agent receives a distinct copy. The copy excludes the source `.git`
directory and creates its own deterministic, clean Git baseline commit, so patch
capture does not depend on the parent repository's history or staged state. This is
still not a filtered or frozen benchmark reset: the implementation copies the rest of
the configured live directory verbatim.
The example configs omit a global workspace and their smoke tasks explicitly select
`fixtures/swe_smoke_repo`. Do not point a paid or holdout task at `workspace: "."`.
Before writing the manifest, matrix preflight rejects an output directory that overlaps
any selected source workspace. This prevents the current artifact tree from being copied
back into the trial, but copy mode remains unfiltered: older artifacts or evaluator data
already present in a dirty source are still visible. Paid and holdout runs require a
clean frozen task repository and a disjoint output directory.
`workspace_mode: direct` is allowed only with shared isolation and is intentionally
marked in provenance because mutations then persist. Matrix preflight permits direct
mode only for exactly one planned trial; use copy mode for every multi-trial matrix.

`export_patch: true` captures each SWE session's bounded
`git diff --binary --full-index` against that fresh baseline, including modified,
deleted, binary, and previously untracked files. The matrix runner writes durable
per-trial files under `patches/` before removing temporary workspaces. Result metadata
contains only the absolute artifact path, SHA-256, byte count, and format; patch bytes
are not serialized inline into results or traces. `max_patch_bytes` bounds each
capture. This is an artifact pipeline, not the SWE-bench Docker evaluator.

The portable `run_command` allowlist constrains only the executable name and cannot
make an interpreter such as `python3` safe. Provider-native shell/bash accepts command
strings and is disabled by default through `allow_native_shell: false`; enabling it
requires an outer process, filesystem, credential, and network sandbox.

Browser navigation through semantic tools checks each requested top-level URL
against `allowed_hosts`. Redirects, page subresources, and a pixel agent typing a
new address can still reach other hosts, so production browser/computer-use runs
require an outer network sandbox or proxy; `allowed_hosts` is not an egress boundary.

## Hosted and external boundaries

OpenAI Responses multi-agent and Anthropic Managed Agents expose exact public
request/session boundaries while their schedulers remain hosted and closed. OpenAI
developer-function calls preserve provider agent attribution so Scaffold Lab can route
a subagent's call into its own local environment and return every pending result in a
stateless HTTP continuation. The adapter is non-streaming HTTP, not WebSocket; HTTP
pauses at a response-wide tool barrier, and the current client loop executes pending
developer calls sequentially. This is a separate latency condition. The
tool-less `openai_hosted_multi_agent.json` and developer-function
`openai_hosted_developer_tools.json` profiles cover those two boundaries separately.
Native computer/shell combined with hosted multi-agent has not been validated live.
The OpenAI hosted adapter can declare the server-side `web_search` tool without
pretending to execute search through the local browser environment. The dedicated
xAI hosted adapter likewise supports the documented `web_search` and `x_search`
server-tool declarations; it does not accept arbitrary developer functions or reveal
plaintext child trajectories.

Anthropic Managed Agents instead owns the environment and session tree. The adapter
polls the session, lists the condensed primary event stream, and consumes authoritative
aggregate session token usage and public-list-price `list_cost`; it does not fetch a
complete child-thread trace. Local concurrency, depth, turn, and tool limits do not
constrain server execution. A Managed session budget is the proactive shared cap and
may be crossed by the final in-flight model request.
A `budget_reached` stop is recorded as an incomplete/error trial; the adapter does not
raise the provider cap or resume the session automatically.

xAI's hosted team, Prime Agent, and Grok Build are also outer boundaries. One
Scaffold Lab model call can represent many internal calls. Prime/Grok are exact only
at their version-checked public CLI protocols; installed executable/package and
scheduler identity require an optional digest and remain outside the default claim.
Their usage is conservatively incomplete where public output does not establish full
tree attribution. Prime Agent runs JSON v3 in fresh `--no-session` state, so its
daemon, schedule, and continual cross-invocation lifecycle is also outside the
implemented boundary.

The pinned MACU adapter verifies a clean checkout at commit
`5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105`, keeps checkout, OSWorld, and result
paths disjoint, passes the task through a private JSON file rather than process
arguments, and bounds/cancels the released process tree. It records the selected
manager/CUA models, limits, Git identity, and parsed release artifacts. The audited
release does not append the initial graph-generation call to `manager_usages`, so its
summary token/cost totals and manager-call count are observed lower bounds. MACU runs
therefore remain `usage_complete: false` and `cost_known: false` for release gates.

The local `rlm_repl` is a source-matched restricted subset with a shared Scaffold Lab
ledger/trace, not the upstream REPL. The separate `swe/rlm-0.1.3-upstream` profile
selects the `rlm_upstream` adapter, which verifies the official `rlms` v0.1.3 release
commit
`72d6940142ddfb84ee6be573dc999a37e633e671`, invokes its selected Python runtime over
bounded JSON stdin, and defaults to the upstream Docker environment. It injects no
Scaffold Lab domain tools and makes no SWE parity claim. Docker plus a 1,500-second
RLM timeout are adapter defaults; upstream library defaults are local plus no timeout.
At `max_depth=1`, `rlm_query` falls back to a direct subcall, and recursive child RLMs
begin only at depth two. In v0.1.3, those child RLMs have separate `LMHandler`
summaries that are not merged into the root
`UsageSummary`; reported calls, tokens, and optional cost are consequently lower
bounds with incomplete/unknown accounting. The bridge covers string context and
selected backend/environment/limit settings, not structured context, custom tools,
compaction, persistence, alternate backends, or sampling/orchestrator configuration.

External CLI adapters run only in a caller-provisioned single-trial workspace.
Scaffold Lab records resolved executable identity, version, Git state, and pre/post
workspace hashes, but does not claim that a version string proves bit identity with a
source snapshot.

## Artifacts and privacy

Each matrix writes `manifest.json`, `results.jsonl`, `traces/*.jsonl`, and
`summary.json`; SWE runs with patch export also write `patches/*.patch`. The manifest
records configured provider settings, environment
configuration, budgets, selected package versions, task/harness data, and a hash of
`src/scaffoldlab/**/*.py`. That hash is not a hash of the entire checkout, dependency
lock, Playwright browser binary, or remote hosted-agent definition. In particular, the
Managed adapter currently records agent/environment IDs and the resolved agent version
without persisting a sanitized complete model/prompt/tool/roster/environment snapshot.
Results include model/tool counts and tool-output bytes at the scope observable to the
adapter.

Content is hashed by default. `capture_content: true` stores prompts, responses,
tool arguments, tool outputs, and raw provider events for audit and may therefore
contain credentials, screenshots, proprietary code, or personal data.

## Extension points

- Implement `ModelBackend.complete` for another provider protocol.
- Subclass `Harness` and implement `_execute` for another coordination policy.
- Implement `ToolEnvironment`, then add it to an `EnvironmentFactory`, for another
  stateful tool surface.
- Add deterministic evaluators separately from candidate models. A model judge must
  be pinned and charged as its own experimental component.
- Prefer an upstream-runtime adapter when the exact released scheduler exists. A
  clean-room topology should not be presented as more faithful than that runtime.

## Remaining fidelity gaps

- `macu_dynamic_dag` remains a local text-DAG subset. `macu_upstream` runs the pinned
  scheduler and prompts, but exact experiment parity still requires pinned
  dependencies, model snapshots, task assets, and OSWorld VM image; its released
  usage summary omits initial graph generation.
- Mythos 5 and Opus 5 system-card variants remain topology simulations until exact
  prompts, message injection timing, context limits, compaction, and Git worktrees
  are reproduced.
- The local Platoon-style candidate is recursive inference only. RAO's trained policy
  and reward procedure are a separate training artifact.
- The restricted local RLM does not become an RLM merely by searching JSON and is not
  byte-for-byte upstream. The pinned upstream adapter preserves the released runtime
  but omits Scaffold Lab browser/SWE/computer tools, and v0.1.3 root accounting omits
  recursive-child summaries.
- Per-agent context limits and automatic compaction are not yet enforced local
  policies.
- Hosted `max_model_calls`, concurrency, depth, turn, and tool limits describe the
  observable outer adapter unless an explicit provider-side field says otherwise.
- OpenAI hosted developer tools use stateless non-streaming HTTP; WebSocket transport
  and live combined native computer/shell compatibility remain unimplemented.
- Managed Agents artifacts do not yet include a sanitized complete remote agent and
  environment definition, and primary events are not a complete child-thread trace.
- `backend_active_union_seconds` is an interval union, not a causal or
  token-rate-normalized critical path.

These labels are part of the result contract. A successful smoke test verifies the
runtime, not a scientific winner or release candidate.
