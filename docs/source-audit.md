# Source and fidelity audit

As-of date: 2026-08-10.

The central distinction is between an inference scaffold, a training method, and
an evaluation environment. They are separate experimental objects.

The user-supplied frontier-harness brief was correct that Claude Opus 5 has a
published system card. Anthropic's [official system-card index](https://www.anthropic.com/system-cards)
lists the July 2026 card. This audit therefore treats Opus 5 and Fable/Mythos 5 as
separate disclosures rather than assuming the Opus name was speculative.

## Named systems

| System | Publicly supported mechanics | Candidate treatment in Scaffold Lab |
| --- | --- | --- |
| [MACU paper](https://arxiv.org/abs/2606.01533), [project](https://jykoh.com/multi-agent-computer-use/), [pinned code](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105) | Manager creates and continually mutates a validated DAG; ready frontier runs in parallel isolated CUA environments; observations and selected artifacts pass through dependencies. Apache-2.0. | `macu_dynamic_dag` is a MACU-inspired text-DAG subset. It omits CUA sessions, screenshots/state/files, upstream mutation semantics, and OSWorld/VM infrastructure. |
| [Browser-Use parallel template](https://docs.browser-use.com/open-source/examples/templates/parallel-browser), [0.13.7 release](https://github.com/browser-use/browser-use/releases/tag/0.13.7) | Different tasks run in separate browser profiles with flat async fan-out and no manager, communication, aggregation, or selector. MIT. | `flat_parallel` reproduces that scheduling pattern and must never be labelled best-of-N. The built-in browser domain adds isolated Playwright contexts but does not copy Browser-Use's agent policy. |
| [RLM paper](https://arxiv.org/abs/2512.24601), [official `rlms` 0.1.3](https://github.com/alexzhang13/rlm/releases/tag/v0.1.3), [minimal code](https://github.com/alexzhang13/rlm-minimal) | Context is an external REPL variable; the root writes code to inspect/transform it and calls `llm_query`, batched subcalls, and recursive RLM queries on selected data. The upstream runtime supports local, Docker, and hosted sandboxes. MIT. | `rlm_repl` is a clean-room restricted-REPL implementation whose subcalls use Scaffold Lab's ledger/trace. It reproduces the algorithmic boundary, not the upstream package byte-for-byte. `external_context_json_search` remains a separate non-RLM ablation. |
| [RAO project](https://apga.github.io/RAO/), [paper](https://arxiv.org/abs/2605.06639), [Platoon snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe) | A Python-REPL policy launches subagents sequentially or concurrently; RAO trains one shared policy across the tree with hierarchical/local rewards. MIT code. | `platoon_recursive_inference` implements the public recursive inference/control shape only. It is never called faithful RAO because the trained policy and reward procedure are absent. `recursive_delegation` remains a simpler JSON control. |
| [Recursive Agent Harnesses](https://arxiv.org/abs/2606.13643) | The parent writes executable code that spawns full tool-using agent harnesses in parallel, uses structured calls for small subtasks, and allows children to recurse to a bounded depth. The paper describes established primitives and a controlled evaluation, but does not publish a reference scheduler. | `recursive_delegation` is only a control: it recursively asks one backend policy for JSON actions and has no generated launcher code, filesystem tools, or full child harnesses. It is not RAH. |
| [Prime Agent launch](https://www.primeintellect.ai/blog/prime-agent), [0.7.1 release](https://github.com/PrimeIntellect-ai/prime-agent/releases/tag/v0.7.1), [tagged source](https://github.com/PrimeIntellect-ai/prime-agent/tree/v0.7.1) | Persistent IPython parent; asynchronous full child sessions; explicit messages/files; daemon persistence, compaction, goals, schedules, bounded autonomous mode, and continual harness state. The release tag resolves to commit `95afd31`. MIT. | `prime_agent` invokes the released JSON runtime. It preserves observed root assistant usage/cost as a lower bound, but v0.7.1 JSON events do not prove complete child-tree attribution. One Scaffold Lab “call” is an external session tree, so usage is always marked incomplete/cost unknown for release gating. |
| [Grok Build source](https://github.com/xai-org/grok-build), [pinned headless contract](https://github.com/xai-org/grok-build/blob/8a14c91d88875a831a38b3a066b1683116bcb31c/crates/codegen/xai-grok-pager/docs/user-guide/14-headless-mode.md), [npm 1.0.0](https://www.npmjs.com/package/@xai-official/grok/v/1.0.0), [sandbox docs](https://docs.x.ai/build/features/sandbox) | Released coding harness with native subagents, tools, permissions, sandbox profiles, worktrees, compaction, and a single-object headless JSON result. The npm `gitHead` (`3cd0d0c…`), public-repository commit (`8a14c91…`), and repository `SOURCE_REV` (`27b3c66…`) differ, so bit identity is not established. | `grok_build` invokes the upstream CLI with native subagents enabled, a fresh session/user home but the caller's persistent trial workspace, explicit permissions/sandbox, and runtime version/executable checks. Reported usage is retained as a lower bound because it can exclude compaction, side-model, and unfinished nested calls. |
| [Prime-RL multi-agent post](https://www.primeintellect.ai/blog/multi-agent-systems), [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) | User-authored async environments control several agents and assign environment-specific rewards/credit. | Evaluation/training infrastructure, not an inference-topology candidate. Integrate later as a benchmark executor. |
| [Prime RLM harness](https://github.com/PrimeIntellect-ai/rlm-harness) | The repository explicitly scopes itself to RLM-style rollouts for RL training; Verifiers also ships RLM evaluation environments. | Evidence for a future upstream training/evaluation integration, not a released inference CLI adapter in this matrix. |

The fidelity labels are deliberately narrow:

- **Exact public boundary:** the documented request/response shape and named transport
  are reproduced at the stated scope, but a closed hosted scheduler is not.
- **Upstream runtime adapter:** the released executable owns the scheduler; Scaffold
  Lab records caller-provisioned isolation, artifact capture, and only the accounting
  exposed by that runtime.
- **Topology simulation:** only disclosed roles and communication/scheduling shape
  are approximated; source prompts, tools, compaction, and timing are unavailable.
- **Inspired baseline/ablation:** a testable mechanism related to a paper, with no
  claim of reproducing the named system.

## Frontier-lab disclosures

| Lab | Strongest reproducible public evidence | Classification |
| --- | --- | --- |
| OpenAI | [GPT-5.6 Responses multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent) publishes the hosted invocation, six coordination actions, injected prompts, context inheritance, independent compaction, recommended concurrency, and stateless HTTP/WebSocket continuation. The [GA computer guide](https://developers.openai.com/api/docs/guides/tools-computer-use) and [shell guide](https://developers.openai.com/api/docs/guides/tools-shell) publish client execution protocols. [Codex source](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/multi_agents.rs) exposes related local handlers, not the hosted scheduler. | Scaffold Lab implements the public beta request and agent-attributed developer-function continuation over non-streaming stateless HTTP. `openai_hosted_multi_agent.json` is tool-less; `openai_hosted_developer_tools.json` exercises developer functions. Computer and local shell are separately implemented protocols, but their combination with hosted multi-agent has not been validated live. The WebSocket transport and hosted scheduler remain unimplemented. |
| Anthropic | [Fable/Mythos 5 system card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) §8.15.3 specifies blocking, fixed-team, and async harnesses. The [Opus 5 system card](https://anthropic.com/claude-opus-5-system-card) §8.11.3 specifies fixed N-agent and async harnesses. Separately, [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) publishes a hosted coordinator roster, one delegation level, isolated thread contexts, and shared sandbox/filesystem/vault state. The [Messages computer guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) publishes `computer_20251124`, `text_editor_20250728`, and `bash_20250124`. | System-card local variants remain topology simulations because exact prompts, timing, compaction, limits, and worktrees are absent. `anthropic_managed_agents` uses the public hosted session, primary-event-list, and aggregate-usage boundary; Anthropic still owns its scheduler, child-thread streams, and environment. Messages uses client tool schemas over Scaffold Lab environments only for model/tool-version pairs supported by Anthropic. |
| xAI | The [pinned Grok Build snapshot](https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c) releases subagent source and worktree/capability semantics. Its [current headless interface](https://docs.x.ai/build/cli/headless-scripting) supports scriptable, machine-readable external invocation. The [Grok multi-agent API](https://docs.x.ai/developers/model-capabilities/text/multi-agent) exposes a team total of 4 agents for low/medium effort or 16 for high/xhigh, including a designated leader. | `grok_build` is an upstream-runtime adapter. `xai_hosted_multi_agent` reproduces the documented no-tool HTTP request for both team sizes. Plaintext intermediate state and scheduler/prompt internals remain closed; encrypted continuation state is exposed by the API but is not implemented locally. |
| Meta | [Muse Spark](https://ai.meta.com/blog/introducing-muse-spark-msl/) discloses parallel multi-agent inference, while [Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) describes a main agent that plans/delegates and subagents that escalate, plus context compaction. Muse 1.1 is available through an OpenAI-compatible Meta Model API, but the public launch material does not specify a hosted multi-agent toggle or scheduler contract. [Meta ARE](https://github.com/facebookresearch/meta-agents-research-environments) releases agent evaluation infrastructure. | Meta and other OpenAI-compatible models can use the generic Responses/Chat backends and local harnesses. That is model portability, not evidence for an exact Muse scheduler. ARE is evaluation infrastructure, not the deployed inference scaffold. |

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

Managed Agents uses `managed-agents-2026-04-01`; attaching a memory store also adds
`agent-memory-2026-07-22`, and the published built-in agent toolset type is
`agent_toolset_20260401`. The session-level event list is the condensed primary view;
complete subagent inspection requires separate thread event endpoints, which the
adapter does not fetch. Session usage is authoritative for aggregate tokens and
public-list-price `list_cost`, not necessarily the invoiced charge; independently
rounded per-thread figures omit runtime cost and need not sum to the session value.
A session budget is one shared cap across threads and is enforced between model
requests, allowing the crossing request to finish slightly above the cap.
A `budget_reached` stop is an incomplete/error trial locally; the adapter neither
raises the cap nor resumes the paused session.

Anthropic's public API resolves a bare agent ID to its latest version, so Scaffold Lab
requires `--managed-agent-version` and rejects an unversioned invocation. Current
artifacts do not persist a sanitized complete snapshot of
the remote model, system prompt, tools, roster, permissions, or environment definition;
those definitions must be exported and hashed separately. The adapter fails closed on
custom-tool and permission-confirmation `requires_action` events. `retain` preserves a
resumable session; `archive` blocks new events while retaining history; `delete`
permanently removes the session record, events, and sandbox.

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
executes all actions and returns one original-detail screenshot. Anthropic returns
`tool_use`, and the client appends an assistant content block followed by a user
`tool_result`; among these Anthropic Messages client tools, only computer use needs the
`computer-use-2025-11-24` beta header. Managed Agents uses its separate beta headers
described above. Anthropic's documented `computer_20251124` model list includes Opus 5
but not Fable 5 or Mythos 5.
Scaffold Lab implements both sequences and tests them offline with fake drivers.

The SWE example profiles set `allow_native_shell: false`. They expose the portable
`run_command` fallback through an executable-name allowlist, a fixed system PATH, and a
minimal subprocess environment that omits provider credentials. This is capability
reduction, not a sandbox: an allowlisted interpreter can still execute arbitrary code.
Enabling provider-native OpenAI shell or Anthropic bash requires an explicit opt-in and
an outer process/network sandbox.

The SWE configs also omit a global workspace; smoke tasks select a dedicated fixture.
Matrix preflight rejects overlap between run artifacts and every selected SWE source
before writing the manifest, and direct mode is restricted to a single planned trial.
Copy mode is intentionally simple and unfiltered, so release tasks still require a
clean frozen source without old artifacts or evaluator data.

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

The Fable/Mythos card compares three topologies:

1. Blocking orchestrator: manager has only spawn capability; fresh 200K workers
   have task tools; manager compacts at 100K and waits for the whole dispatched
   round.
2. Fixed team: 3/5/10 long-lived peers see the complete task, share identical
   tools, send/wait for messages, and use a designated lead. Coding agents use
   separate checkouts and share through Git.
3. Async subagents: the lead retains task tools and dynamically manages long-lived
   workers. Workers see only their delegation, message peers/lead, idle on
   completion, and can be resumed. ProgramBench used four concurrent and twenty
   total workers. The BrowseComp async profile does not disclose a corresponding
   spawn cap. Scaffold Lab's blocking profile uses four workers only as a local
   controlled choice; the card does not prescribe that number for blocking rounds.

Opus 5 differs materially: §8.11.3 compares only N-agent teams (N=5 or 10) and
async subagents; every agent has a 1M-token limit, and no async spawn cap is stated.
For a pre-release Opus 5 configuration, the card reports a 93.6% 10-agent
BrowseComp score, 5.6x/5.9x speedups for N=5/10 against its 10M-token single-agent
baseline, and an async score 2.8 percentage points above that same baseline. Fable,
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

- Flat parallelism is best-of-N.
- A recursive inference wrapper reproduces RAO training.
- Long context, memory, or compaction is automatically an RLM.
- A public API reproduces a closed server scheduler.
- A version string alone proves that an installed CLI is bit-identical to an audited
  source tree or package artifact.
- A public-card topology simulation is “1:1” when prompts, tools, message timing,
  compaction, or runtime source are unavailable.
