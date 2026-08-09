# Source and fidelity audit

As-of date: 2026-08-10.

The central distinction is between an inference scaffold, a training method, and
an evaluation environment. They are separate experimental objects.

The public catalog is organized first by application: `browser`, `computer-use`,
and `swe`. Each entry records its artifact kind, sources, runtime owner, runnable
status, exact components, and unavailable components. Selecting a profile is not a
marketing alias: configuration resolution enforces its exact harness signature and
compatible provider/environment, while `catalog_only` training/evaluation/gap entries
cannot execute.

The user-supplied frontier-harness brief was correct that Claude Opus 5 has a
published [official system-card PDF](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf).
This audit therefore treats Opus 5 separately from the Mythos 5 multi-agent
evaluation in the joint Fable/Mythos card rather than assuming the Opus name was
speculative.

## Named systems

| System | Publicly supported mechanics | Candidate treatment in Scaffold Lab |
| --- | --- | --- |
| [MACU paper](https://arxiv.org/abs/2606.01533), [project](https://jykoh.com/multi-agent-computer-use/), [pinned code](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105) | Manager creates and continually mutates a validated DAG; ready frontier runs in parallel isolated CUA environments; observations and selected artifacts pass through dependencies. Apache-2.0. | `macu_dynamic_dag` is a MACU-inspired local text-DAG subset. `macu_upstream` separately executes the clean pinned release for a generic blank-start task and ingests its artifacts. It does not enter the domain/UUID OSWorld benchmark path or evaluator. The released summary also omits initial graph-generation usage, so observed tokens, cost, and manager calls remain lower bounds. |
| [Browser-Use parallel template](https://docs.browser-use.com/open-source/examples/templates/parallel-browser), [0.13.7 release](https://github.com/browser-use/browser-use/releases/tag/0.13.7) | Different tasks run in separate browser profiles with flat async fan-out and no manager, communication, aggregation, or selector. MIT. | `flat_parallel` reproduces that scheduling pattern and must never be labelled best-of-N. The built-in browser domain adds isolated Playwright contexts but does not copy Browser-Use's agent policy. |
| [RLM paper](https://arxiv.org/abs/2512.24601), [official `rlms` 0.1.3](https://github.com/alexzhang13/rlm/releases/tag/v0.1.3), [minimal code](https://github.com/alexzhang13/rlm-minimal) | Context is an external REPL variable; the root writes code to inspect/transform it and calls `llm_query`, batched subcalls, and recursive RLM queries on selected data. The upstream runtime supports local, Docker, and hosted sandboxes. MIT. | `swe/rlm-0.1.3-contract` selects the source-matched restricted `rlm_repl`, whose subcalls use Scaffold Lab's ledger/trace. `swe/rlm-0.1.3-upstream` selects `rlm_upstream`, verifies release commit `72d6940142ddfb84ee6be573dc999a37e633e671`, and executes the upstream package in a bounded external process. Scaffold Lab selects Docker and a 1,500-second timeout, while the library defaults are local and no timeout. Recursive child RLMs require `max_depth >= 2`. It injects no domain tools and is not SWE parity. `external_context_json_search` remains a separate non-RLM ablation. |
| [RAO project](https://apga.github.io/RAO/), [paper](https://arxiv.org/abs/2605.06639), [Platoon 0.1.0 paper snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe) | A Python/IPython-REPL policy launches copies of one shared policy sequentially or concurrently. RAO jointly trains every tree node using node-local success plus an optional immediate-child delegation bonus, a root-group leave-one-out baseline, and inverse depth-frequency weighting. The MIT snapshot publishes inference and Tinker/AReaL training pipelines, reward code, prompts, configs, and task assets. | `platoon_recursive_inference` implements only the recursive control shape over a restricted local REPL and caller-selected model. It is never called faithful RAO. No Scaffold Lab adapter executes the pinned upstream runtime or training pipeline, and no paper-trained checkpoint is identified in the paper or snapshot. `recursive_delegation` remains a simpler JSON control. |
| [Recursive Agent Harnesses](https://arxiv.org/abs/2606.13643) | The parent writes executable code that spawns full tool-using agent harnesses in parallel, uses structured calls for small subtasks, and allows children to recurse to a bounded depth. The paper describes established primitives and a controlled evaluation, but does not publish a reference scheduler. | `recursive_delegation` is only a control: it recursively asks one backend policy for JSON actions and has no generated launcher code, filesystem tools, or full child harnesses. It is not RAH. |
| [Prime Agent launch](https://www.primeintellect.ai/blog/prime-agent), [0.7.1 release](https://github.com/PrimeIntellect-ai/prime-agent/releases/tag/v0.7.1), [tagged source](https://github.com/PrimeIntellect-ai/prime-agent/tree/95afd319a78ae017a41241d50b013d656a0685ce), [JSON v3 contract](https://github.com/PrimeIntellect-ai/prime-agent/blob/95afd319a78ae017a41241d50b013d656a0685ce/packages/coding-agent/docs/json.md) | Persistent IPython parent; asynchronous full child sessions; explicit messages/files; daemon persistence, compaction, goals, schedules, bounded autonomous mode, and continual harness state. The release tag resolves to commit `95afd31`; GitHub reports SHA-256 `d68612c…` for `prime-agent-0.7.1.tgz`. MIT. | `prime_agent` checks version 0.7.1 and exactly implements the published JSON v3 CLI framing: session header, object events, and terminal `agent_end`. It runs each call with `--no-session` and fresh home/config directories. The release-asset digest is recorded as provenance, but installed-package identity is outside the exact boundary unless an executable SHA is supplied. Cross-invocation state is absent, and root assistant usage/cost is a lower bound because events do not prove complete child-tree attribution. |
| [Grok Build source](https://github.com/xai-org/grok-build), [pinned headless contract](https://github.com/xai-org/grok-build/blob/8a14c91d88875a831a38b3a066b1683116bcb31c/crates/codegen/xai-grok-pager/docs/user-guide/14-headless-mode.md), [npm 1.0.0](https://www.npmjs.com/package/@xai-official/grok/v/1.0.0), [sandbox docs](https://docs.x.ai/build/features/sandbox) | Released coding harness with native subagents, tools, permissions, sandbox profiles, worktrees, compaction, and a single-object headless JSON result. The npm `gitHead` (`3cd0d0c…`), public-repository commit (`8a14c91…`), and repository `SOURCE_REV` (`27b3c66…`) differ, so bit identity is not established. | `grok_build` invokes the upstream CLI with native subagents enabled, a fresh session/user home but the caller's persistent trial workspace, explicit permissions/sandbox, and runtime version/executable checks. Reported usage is retained as a lower bound because it can exclude compaction, side-model, and unfinished nested calls. |
| [Prime-RL multi-agent post](https://www.primeintellect.ai/blog/multi-agent-systems), [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) | User-authored async environments control several agents and assign environment-specific rewards/credit. | Evaluation/training infrastructure, not an inference-topology candidate. Integrate later as a benchmark executor. |
| [Prime RLM harness](https://github.com/PrimeIntellect-ai/rlm-harness) | The repository explicitly scopes itself to RLM-style rollouts for RL training; Verifiers also ships RLM evaluation environments. | Evidence for a future upstream training/evaluation integration, not a released inference CLI adapter in this matrix. |

The fidelity labels are deliberately narrow and correspond to registry values:

- **`exact_public_protocol`:** the documented API request/response or released CLI
  stream contract is reproduced at the stated boundary; the scheduler/runtime behind
  that boundary and executable identity are included only when explicitly verified.
- **`upstream_runtime_adapter`:** a clean source checkout at the cataloged revision
  owns the scheduler; Scaffold Lab records caller-provisioned isolation, artifacts,
  and only the accounting exposed by that runtime.
- **`source_matched_reimplementation`:** disclosed source mechanics are reimplemented,
  with all missing runtime behavior stated explicitly.
- **`topology_simulation`:** only disclosed roles and communication/scheduling shape
  are approximated; source prompts, tools, compaction, and timing are unavailable.
- **`inference_only_reimplementation`:** inference/control flow is reproduced without
  the named training method or trained checkpoint.
- **`controlled_baseline`:** a testable control with no named-system parity claim.
- **`documented_gap`:** a source, training artifact, or evaluation target is recorded
  but not runnable as that system.

Only the first two labels are exposed by `list-implementations` or allowed under an
`application.implementation` selector. The four local/reimplementation labels are
exposed as studies, and documented gaps are catalog-only.

## Application coverage

| Top-level application | Exact implementations | Studies and documented gaps |
| --- | --- | --- |
| `browser` | OpenAI hosted multi-agent with client functions or hosted `web_search`; xAI 4/16-agent hosted web research; Anthropic Managed Agents; pinned MACU runtime | Browser-Use flat parallel pattern, local MACU, and Mythos/Opus topology studies; Meta Muse scheduler and BrowserGym evaluator gaps |
| `computer-use` | OpenAI GA computer wire protocol, Anthropic `computer_20251124` wire protocol, and pinned generic-task MACU VM runtime | Hosted OpenAI composition and local MACU studies; MACU OSWorld 1 parity, Meta Muse action/scheduler, and OSWorld 2 evaluator gaps |
| `swe` | Anthropic Managed Agents plus Prime Agent JSON v3 and Grok Build headless CLI boundaries, and the clean pinned upstream RLM runtime | Local controls, composed OpenAI protocols, restricted RLM, Platoon inference, MACU, and card-topology studies; RAO training, Meta Muse scheduling, xAI developer tools, and SWE-bench evaluator gaps |

## Frontier-lab disclosures

| Lab | Strongest reproducible public evidence | Classification |
| --- | --- | --- |
| OpenAI | [GPT-5.6 Responses multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent) publishes the hosted invocation, six coordination actions, injected prompts, context inheritance, independent compaction, recommended concurrency, and stateless HTTP/WebSocket continuation. The [GA computer guide](https://developers.openai.com/api/docs/guides/tools-computer-use) and [shell guide](https://developers.openai.com/api/docs/guides/tools-shell) publish client execution protocols. [Codex source](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/multi_agents.rs) exposes related local handlers, not the hosted scheduler. | Scaffold Lab implements the public beta request and agent-attributed developer-function continuation over non-streaming stateless HTTP, plus a separate server-side `web_search` profile. GA computer and local shell are separately implemented protocols. Their combination with hosted multi-agent is source-matched but not yet live-validated; WebSocket transport and the hosted scheduler remain unimplemented. |
| Anthropic | The [joint Fable/Mythos 5 system card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) §8.15.3 evaluates **Mythos 5** with blocking, fixed-team, and async harnesses. The [Opus 5 system card](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) §8.11.3 specifies fixed N-agent and async harnesses. Separately, [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) publishes a hosted coordinator roster, one delegation level, isolated thread contexts, and shared sandbox/filesystem/vault state. The [Messages computer guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) publishes `computer_20251124`, `text_editor_20250728`, and `bash_20250124`. | System-card local variants remain topology simulations. Public limits include Mythos blocking workers at 200K context, manager compaction at 100K, fixed/async agents at 1M total tokens, ProgramBench async at four concurrent/twenty total, Opus agents at 1M tokens, and no disclosed BrowseComp/Opus async spawn cap. Scaffold Lab does not yet enforce those per-agent limits, and exact prompts, timing, tool surfaces, compaction behavior, and Git sharing remain unavailable. `anthropic_managed_agents` uses the public hosted session boundary and validates the returned pinned coordinator snapshot; Anthropic still owns scheduling, child streams, and the mutable environment. Messages uses the published client schema over a local Playwright executor only for supported model/tool-version pairs. |
| xAI | The [pinned Grok Build snapshot](https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c) releases subagent source and worktree/capability semantics. Its [current headless interface](https://docs.x.ai/build/cli/headless-scripting) supports scriptable, machine-readable external invocation. The [Grok multi-agent API](https://docs.x.ai/developers/model-capabilities/text/multi-agent) exposes a team total of 4 agents for low/medium effort or 16 for high/xhigh, including a designated leader and hosted `web_search`/`x_search` tools. | `grok_build` exactly implements the version-1.0.0 headless CLI boundary, not npm/source or scheduler identity unless separately hashed; those revisions publicly disagree. `xai_hosted_multi_agent` pins model snapshot `grok-4.20-multi-agent-0309` and reproduces the documented 4/16-agent HTTP boundary and server-tool declarations. Plaintext intermediate state and scheduler/prompt internals remain closed; encrypted continuation state is exposed by the API but is not implemented locally. Developer tools remain unsupported. |
| Meta | [Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) describes a main agent that plans/delegates and subagents that escalate, plus context compaction. Muse 1.1 is available through an OpenAI-compatible Meta Model API, but the public launch material does not specify a hosted multi-agent toggle or scheduler contract. [Meta ARE](https://github.com/facebookresearch/meta-agents-research-environments) releases agent evaluation infrastructure. | Meta and other OpenAI-compatible models can use the generic Responses/Chat backends and local harnesses. That is model portability, not evidence for an exact Muse scheduler. The browser, computer-use, and SWE Muse profiles are catalog-only. ARE is evaluation infrastructure, not the deployed inference scaffold. |

### Hosted-boundary caveats

OpenAI multi-agent is currently a GPT-5.6 beta gated by
`OpenAI-Beta: responses_multi_agent=v1`. The provider's
`max_concurrent_subagents` applies across all descendants and excludes the root; the
public guide documents no fixed tree-depth or total-subagent limit. Scaffold Lab's
local concurrency and depth fields do not replace that provider setting. The public
guide also says `/responses/compact`, `reasoning.summary`, and request-level
`max_tool_calls` are unsupported in multi-agent mode, while automatic per-agent
compaction is implicit. Over HTTP, the response waits for all active agents to finish
or pause and the client returns every pending function result in a new request;
WebSocket can inject results without that global response barrier. Scaffold Lab's
current HTTP client also executes pending developer calls sequentially, so its latency
must not be compared as if it used the WebSocket path.

Managed Agents session endpoints use `managed-agents-2026-04-01`, including when a
memory store is attached as a session resource. The separate memory-store endpoints
use `agent-memory-2026-07-22` **instead**; the headers must not be combined. The
published built-in agent toolset type is `agent_toolset_20260401`. The session-level
event list is the condensed primary view;
complete subagent inspection requires separate thread event endpoints, which the
adapter does not fetch. Session usage is authoritative for aggregate tokens and
public-list-price `list_cost`, not necessarily the invoiced charge; independently
rounded per-thread figures omit runtime cost and need not sum to the session value.
A session budget is one shared cap across threads and is enforced between model
requests, allowing the crossing request to finish slightly above the cap.
A `budget_reached` stop is an incomplete/error trial locally; the adapter neither
raises the cap nor resumes the paused session.

Anthropic's public API resolves a bare agent ID to its latest version, so Scaffold Lab
requires `--managed-agent-version` and rejects an unversioned invocation. The create
and retrieve responses expose the resolved agent snapshot. The adapter validates its
ID/version, requires `multiagent.type=coordinator` with a non-empty roster, fails if
the snapshot changes while polling, and records a canonical SHA-256 digest. The
remote environment definition is not returned as an immutable session snapshot and
must still be exported and hashed separately. The adapter fails closed on
custom-tool and permission-confirmation `requires_action` events. `retain` preserves a
resumable session; `archive` blocks new events while retaining history; `delete`
permanently removes the session record, events, and sandbox.

### Pinned-runtime accounting caveats

The MACU adapter runs only from a clean checkout at
`5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105` and keeps the checkout, OSWorld root,
and result directory disjoint. It passes task content through a private JSON task file,
bounds stdout/stderr and artifact ingestion, terminates the process group on timeout or
cancellation, and records explicit manager/CUA providers, models, and scheduler limits.
This preserves the released scheduler and prompts, subject to the caller's dependency,
model, and VM pins. The adapter materializes the upstream generic list-task shape with
`no_initial_setup`, so it does not load an OSWorld domain/UUID canonical task or run
the evaluator; OSWorld scores require a separate adapter plus data/task/VM pins. Code
audit found that the release's initial manager call for
graph generation is not appended to `manager_usages`; `summary.json` therefore omits
that call from manager counts, tokens, and cost. Scaffold Lab retains the reported
values only as incomplete, cost-unknown lower bounds.

The upstream RLM adapter runs the official v0.1.3 commit
`72d6940142ddfb84ee6be573dc999a37e633e671` from a clean checkout using an explicitly
selected Python (the release requires Python 3.11 or newer). Input travels through a
bounded JSON-stdin bridge rather than command arguments, process output and lifetime
are bounded. Scaffold Lab selects the upstream Docker REPL and a 1,500-second RLM
timeout; those are adapter choices, not the library defaults of local execution and no
timeout. With the adapter default `max_depth=1`, `rlm_query` falls back to
`llm_query`; recursive child RLM control begins at `max_depth >= 2`. The upstream root
`UsageSummary` describes its own `LMHandler`, while each recursive child RLM creates a
separate handler whose summary is not merged into the root in v0.1.3. Calls, tokens,
and optional costs are therefore lower bounds with `usage_complete: false` and
`cost_known: false`. This adapter preserves the released RLM runtime; it does not add
Scaffold Lab SWE/browser/computer tools or claim domain-task parity. Its bridge covers
string context and selected backend/environment/limit settings; it does not expose
structured context, custom tools, compaction, persistence, alternate backends, or
sampling/orchestrator configuration.

## Tool and evaluation environments

The local domain layer is deliberately smaller than the benchmark integrations it
can eventually drive:

| Artifact | Public role | Scaffold Lab treatment |
| --- | --- | --- |
| [Playwright 1.60.0](https://github.com/microsoft/playwright/releases/tag/v1.60.0) | Browser automation substrate. | Optional implementation behind semantic browser tools and the browser-viewport computer driver. It is not itself a benchmark or agent policy. |
| [BrowserGym 0.14.3](https://github.com/ServiceNow/BrowserGym/releases/tag/v0.14.3) | Browser task environments, observation/action spaces, resets, and evaluators. | Pinned target for a future exact evaluation adapter. The built-in browser environment makes no BrowserGym parity claim. |
| [Browser-Use 0.13.7](https://github.com/browser-use/browser-use/releases/tag/0.13.7) | Browser agent runtime and examples. | Only its flat parallel scheduling example informs `flat_parallel`; local tools do not reproduce its policy. |
| [SWE-bench 4.1.0](https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0) | Dataset and Docker-based patch evaluator. | Evaluation target, not an inference harness. Local SWE tools can produce a patch, but a release score still requires the pinned SWE-bench evaluator/images. |
| [OpenHands Agent SDK 1.23.1](https://github.com/OpenHands/software-agent-sdk/releases/tag/v1.23.1) | Released coding/browser agent runtime. | External comparator/integration target. It is not silently substituted for a system-card harness. |
| [OSWorld](https://github.com/xlang-ai/OSWorld) | Desktop-computer task environment and evaluator. | Required external VM/evaluator for an OSWorld claim. The Playwright viewport driver is only protocol QA. |

OpenAI's current computer response contains an ordered `actions[]` batch; the client
executes every action in order and returns one original-detail screenshot. Its
`keypress` uses ordered `keys[]`, and mouse actions can carry `keys[]` modifiers.
Anthropic returns one `tool_use` action, and the client appends the assistant content
block followed by a user `tool_result`; its keyboard shortcut field is singular
`key`. Scaffold Lab implements the `computer_20251124` request/result schema and
action vocabulary, including triple click,
left mouse down/up, modifier-aware mouse/scroll actions, `hold_key` with bounded
duration, and failure-safe release of pressed keys/buttons during composite actions.
Zoom remains deliberately disabled. Among these Anthropic Messages client tools, only
computer use needs the `computer-use-2025-11-24` beta header. Actions execute in a
local Playwright browser viewport, not Anthropic's reference Linux X11/VNC executor;
therefore timing, screenshot, and desktop behavior are not upstream-runtime parity.
Managed Agents uses its separate beta header described above. Anthropic documents
`computer_20251124` for Opus 5, Sonnet 5, Opus 4.8/4.7/4.6/4.5, and Sonnet 4.6—not
Fable 5, Mythos 5, or Haiku 4.5. Exact application runs enforce those model-family
prefixes. OpenAI exact computer runs likewise accept only the currently documented
GPT-5.4 and GPT-5.6 families. Scaffold Lab tests action ordering, schema differences,
modifier release, and recovery screenshots offline with fake drivers.

The SWE example profiles set `allow_native_shell: false`. They expose the portable
`run_command` fallback through an executable-name allowlist, a fixed system PATH, and a
minimal subprocess environment that omits provider credentials. This is capability
reduction, not a sandbox: an allowlisted interpreter can still execute arbitrary code.
Enabling provider-native OpenAI shell or Anthropic bash requires an explicit opt-in and
an outer process/network sandbox.

The SWE configs also omit a global workspace; smoke tasks select a dedicated fixture.
Matrix preflight rejects overlap between run artifacts and every selected SWE source
before writing the manifest, and direct mode is restricted to a single planned trial.
Copy mode excludes the parent `.git` directory and initializes a deterministic clean
Git baseline inside each temporary workspace. With `export_patch: true`, the runner
captures a `git diff --binary --full-index` that includes modified, deleted, binary,
and untracked files, enforces `max_patch_bytes`, externalizes it to a per-trial
`patches/*.patch` artifact, and replaces private in-memory bytes with path/hash/size
metadata before trace/result serialization and temporary cleanup. Copy mode remains
otherwise unfiltered, so release tasks still require a clean frozen source without
old artifacts or evaluator data. Durable patch capture is not SWE-bench scoring; the
pinned Docker evaluator/images remain a separate requirement.

## Provider portability is not scheduler parity

The generic OpenAI-compatible Responses and Chat Completions adapters allow local
harnesses to run models served by compatible lab or self-hosted endpoints. The
supported contract is JSON function calling only. A compatible endpoint does not
thereby gain OpenAI's hosted multi-agent beta, native computer/shell tools, or
Responses continuation semantics.

xAI's hosted multi-agent endpoint is intentionally restricted to its dedicated
harness because its continuation/tool semantics differ. Anthropic Managed Agents is
likewise restricted to its session harness. These compatibility checks prevent a
model endpoint from being presented as a reproduction of its lab's internal
coordinator.

## Anthropic controlled profiles

The joint Fable/Mythos card evaluates **Mythos 5** with three topologies. It does
not report Fable 5 as the model for this multi-agent comparison:

1. Blocking orchestrator: manager has only spawn capability; fresh 200K workers
   have task tools; manager compacts at 100K and waits for the whole dispatched
   round. No overall manager token cap is disclosed.
2. Fixed team: 3/5/10 long-lived peers see the complete task, share identical
   tools, send/wait for messages, and use a designated lead. Coding agents use
   separate checkouts and share through Git; every agent has the same 1M total-token
   limit.
3. Async subagents: the lead retains task tools and dynamically manages long-lived
   workers. Workers see only their delegation, message peers/lead, idle on
   completion, and can be resumed. The lead and every worker have a 1M-token limit
   without compaction. ProgramBench used four concurrent and twenty total workers.
   The BrowseComp async profile does not disclose a corresponding spawn cap.
   Scaffold Lab's blocking profile uses four workers only as a local
   controlled choice; the card does not prescribe that number for blocking rounds.

Opus 5 differs materially: §8.11.3 compares only N-agent teams (N=5 or 10) and
async subagents; every agent has a 1M-token limit, and no async spawn cap is stated.
For a pre-release Opus 5 configuration, the card reports a 93.6% 10-agent
BrowseComp score, 5.6x/5.9x speedups for N=5/10 against its 10M-token single-agent
baseline, and an async score 2.8 percentage points above that same baseline. Mythos,
by contrast, includes blocking orchestration, N=3/5/10, and a ProgramBench async cap
of four concurrent/twenty total workers.

Anthropic's production Research post is a separate disclosure, not evidence that a
system-card benchmark topology is deployed unchanged. It describes a lead that
dispatches synchronous blocking batches, persistent planning, and a final citation
agent; it does not establish deployment of the card's async scheduler.

Both cards report total tokens across all agents. Their latency is a normalized
critical-path estimate derived from fixed prefill/decode rates plus tool time, not
raw wall-clock time. Scaffold Lab currently reports raw wall time and the union of
live backend-call spans, explicitly named `backend_active_union_seconds`. A
paper-parity normalized metric is still required.

## Claims we will not make

- Flat parallelism as best-of-N; it has no selector.
- Recursive inference as a reproduction of RAO training.
- Long context, memory, or compaction alone as an RLM implementation.
- The local restricted `rlm_repl` is the upstream `rlms` package, or the pinned
  upstream adapter provides SWE/browser/computer tool parity.
- A public API reproduces a closed server scheduler.
- A version string alone proves that an installed CLI is bit-identical to an audited
  source tree or package artifact.
- A public-card topology simulation is “1:1” when prompts, tools, message timing,
  compaction, or runtime source are unavailable.
