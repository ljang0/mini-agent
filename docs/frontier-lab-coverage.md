# Frontier-lab coverage

This table turns the user-supplied 18-lab survey into an implementation decision.
“Model runnable” and “named harness reproduced” are intentionally different claims.

The executable/catalog surface is application-first:

| Application | What belongs underneath it | Important non-equivalence |
| --- | --- | --- |
| `browser` | Hosted OpenAI multi-agent functions or `web_search`; hosted xAI 4/16-agent web research; Anthropic Managed web research; Browser-Use flat fan-out; MACU local/upstream; Mythos/Opus simulations | Playwright tools are not BrowserGym, Browser-Use agent-policy parity, or a hosted search implementation |
| `computer-use` | Exact public OpenAI GA and Anthropic `computer_20251124` request/result schemas; source-matched hosted combinations; MACU local study and pinned generic upstream VM runtime | The built-in driver is a local browser viewport, not Anthropic's reference Linux executor or an OSWorld VM; the MACU adapter does not enter the OSWorld domain/UUID task and evaluator path |
| `swe` | Local controls and system-card simulations; hosted OpenAI/Anthropic boundaries; Prime Agent, Grok Build, and RLM upstream adapters; restricted local RLM; Platoon inference; SWE with computer tools | Patch generation is not SWE-bench evaluation; Platoon inference is not RAO training |

The xAI hosted adapter allowlists both `web_search` and `x_search`; the registered
4/16-agent web-research profiles select `web_search`. Each implementation is labeled
`exact_public_protocol`, `upstream_runtime_adapter`,
`source_matched_reimplementation`, `topology_simulation`,
`inference_only_reimplementation`, `controlled_baseline`, or `documented_gap`.
Only runnable/simulation entries bind a harness, provider, and environment;
`catalog_only` entries cannot execute.

| Lab | Strongest public artifact in the survey | Scaffold Lab path | Fidelity/status |
| --- | --- | --- | --- |
| Anthropic | [Fable/Mythos](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf) and [Opus 5](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) system cards; [Messages computer tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool); [Managed Agents](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) | Mythos-only Browser/SWE card simulations; exact public `anthropic-messages` request/result schema over a local Playwright executor; hosted `anthropic-managed-agents` browser/SWE profiles | Card variants are topology simulations. The joint card's multi-agent evaluation uses Mythos 5, not Fable 5. Public per-agent limits are documented but not locally enforced. Managed validates and hashes the resolved pinned coordinator snapshot; child streams, scheduling, and the mutable environment remain hosted. |
| OpenAI | [Responses multi-agent beta](https://developers.openai.com/api/docs/guides/responses-multi-agent); [GA computer](https://developers.openai.com/api/docs/guides/tools-computer-use); [shell](https://developers.openai.com/api/docs/guides/tools-shell); Codex source | Browser hosted functions/`web_search`; exact single-agent computer protocol; source-matched hosted computer/SWE combinations; local harnesses via `openai-responses` | Stateless HTTP and agent-attributed developer continuation are implemented. Computer executes ordered `actions[]` (`keypress.keys[]`, modifier `keys[]`) then returns one original-detail screenshot. WebSocket and the server scheduler are not reproduced; hosted computer/shell combinations need live validation. |
| Google DeepMind | ADK and A2A | Any local harness through a compatible model endpoint; upstream ADK/A2A remains an external comparator | No claim that local topologies reproduce Gemini Deep Research/Deep Think. No dedicated ADK bridge yet. |
| xAI | [Grok Build source](https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c) and [hosted multi-agent API](https://docs.x.ai/developers/model-capabilities/text/multi-agent) | `grok_build`; 4/16-agent `xai_hosted_multi_agent` browser profiles with server `web_search`; adapter also allowlists `x_search` | Exact headless-CLI and hosted request/tool-declaration boundaries. Installed npm/source identity, developer functions, child plaintext, and hosted schedulers remain outside those scopes. |
| Microsoft | AutoGen and Microsoft Agent Framework | Local fixed/async teams as controlled baselines; upstream framework is an external comparator | No AutoGen/MAF parity claim. A source-pinned upstream adapter is preferable to reimplementing their full runtimes. |
| Alibaba/Qwen | Qwen-Agent GroupChat and Qwen Code | Compatible model endpoint plus local topologies | Model portability only. The moderated GroupChat runtime is not yet wrapped. |
| Moonshot/Kimi | [Kimi Code](https://github.com/MoonshotAI/kimi-code) and AgentSwarm disclosures | Compatible model endpoint; upstream CLI is an external-adapter target | No hosted AgentSwarm parity claim. |
| Mistral | Vibe subagents and managed durable agents/workflows | Compatible endpoint plus local topologies | Model portability only; no Vibe/durable-runtime wrapper yet. |
| Meta | [Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) and [Meta ARE](https://github.com/facebookresearch/meta-agents-research-environments) | Catalog-only Muse profiles under browser, computer-use, and SWE; compatible Meta endpoint plus local topologies | Public descriptions do not provide a scheduler API, prompts, limits, or timing. ARE is evaluation infrastructure, not the deployed inference scaffold. |
| AWS | Bedrock multi-agent collaboration | External managed-service target | No Bedrock provider/session adapter yet. |
| NVIDIA | [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) | External framework/comparator target | Integration and optimization framework, not one canonical frontier harness. |
| ByteDance | DeerFlow | External upstream-runtime target | No claim that DeerFlow is the private Seed/Doubao coordinator. |
| Tencent | YunqueAgent | External upstream-runtime target | Public research framework, not Hunyuan product parity. |
| Z.ai/GLM | Synapse | Compatible model endpoint; external upstream-runtime target | No GLM product-scheduler parity claim. |
| Baidu | LoongFlow and Qianfan AppBuilder | Compatible model endpoint; external upstream-runtime target | No ERNIE product-scheduler parity claim. |
| DeepSeek | Model weights/APIs; no qualifying first-party team runtime in the survey | Compatible model endpoint plus any local harness | Worker-model support, not a DeepSeek-authored team reproduction. |
| MiniMax | Mini-Agent, explicitly single-agent | Compatible model endpoint plus any local harness | Worker-model support; no first-party multi-agent runtime to reproduce. |
| Cohere | Tool-use API and framework integrations | Compatible endpoint where function calling matches; otherwise a future native provider | Worker-model support; orchestration belongs to the selected external/local harness. |

## What “all models” means in this repository

The local coordination policies are model-independent. A model can participate when
one of these protocol adapters matches its endpoint:

- OpenAI Responses, with current developer-function continuation, hosted
  `web_search`, and separately implemented computer and local-shell client protocols.
  Their combination with the hosted multi-agent beta requires separate live
  compatibility validation;
- Anthropic Messages, with current bash, text-editor, and computer client schemas when
  the selected model supports that tool version. The documented
  `computer_20251124` list includes Opus 5 but not Fable 5 or Mythos 5;
- generic OpenAI-compatible Responses or Chat Completions function calling;
- dedicated xAI hosted, Anthropic Managed, or released-runtime adapters.

That gives broad model coverage without pretending that an OpenAI-compatible model
endpoint exposes its vendor's private orchestration stack. If a public upstream
runtime owns essential scheduler state, prompts, retries, compaction, or message
delivery, the faithful next step is a pinned external adapter, not a similarly named
clean-room class.

## Source-pinned cross-lab runtimes

- [MACU](https://github.com/kohjingyu/multi-agent-computer-use/tree/5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105)
  has two intentionally distinct treatments. `macu_dynamic_dag` is a local
  source-matched text-DAG subset; `macu_upstream` runs the pinned released scheduler,
  prompts, CUA subprocesses, and result protocol on a generic blank-start task.
  It does not supply the domain/UUID input, canonical loader/setup, or evaluator
  needed for an OSWorld score; that boundary is a separate catalog gap. Exact
  experiment identity still needs pinned dependencies/models/VM assets. The release
  omits initial graph-generation usage from `manager_usages`, so accounting remains
  a lower bound.
- [RLM v0.1.3](https://github.com/alexzhang13/rlm/releases/tag/v0.1.3) likewise has
  two treatments. `swe/rlm-0.1.3-contract` selects the restricted local `rlm_repl`
  with shared ledger/trace; `swe/rlm-0.1.3-upstream` selects `rlm_upstream` and runs
  official commit
  `72d6940142ddfb84ee6be573dc999a37e633e671` over bounded JSON stdin, using the
  upstream Docker REPL and a 1,500-second RLM timeout by adapter default. The library
  itself defaults to local execution and no timeout. At default `max_depth=1`,
  `rlm_query` falls back to `llm_query`; child RLMs require `max_depth >= 2`. The
  adapter injects no Scaffold Lab domain tools. Recursive child RLMs have independent
  handlers in v0.1.3, so the root
  `UsageSummary` is incomplete and cost-unknown for release comparisons. The bridge
  exposes string context and selected backend/environment/limit settings, not every
  public option such as structured context, custom tools, compaction, persistence,
  alternate backends, or sampling/orchestrator configuration.
- [Platoon 0.1.0's RAO paper snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe)
  publishes the domain-specific inference runtime and Tinker/AReaL training
  pipelines, reward code, prompts, configs, and task assets. Scaffold Lab has no
  pinned external adapter for that runtime: `platoon_recursive_inference` is only
  a restricted inference-shape study. A paper-equivalent RAO result also needs a
  paper-trained checkpoint and exact backend, model, dataset, judge, tool, and
  compute snapshots; the paper and snapshot do not identify such a checkpoint.
- [Prime Agent 0.7.1](https://github.com/PrimeIntellect-ai/prime-agent/releases/tag/v0.7.1)
  and Grok Build 1.0.0 are exact only at their published CLI protocol boundaries,
  not installed executable/package or scheduler identity unless a digest is supplied.
  Prime checks 0.7.1 by default and enforces its JSON v3 header/terminal event. Its
  fresh `--no-session` lifecycle preserves within-call Python/subagent state but not
  daemon, schedule, or cross-invocation continual state. Public output does not
  establish complete child-tree accounting.

SWE copy-mode runs create a fresh Git baseline independent of the parent repository.
When `export_patch` is enabled, modified/new/deleted/binary changes are captured in a
bounded `git diff --binary --full-index`, written as durable per-trial patch files, and
represented in result metadata only by path, SHA-256, size, and format. This enables
later evaluation but does not reproduce
[SWE-bench 4.1.0](https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0) or turn
smoke-test success into release evidence.

## Adapter acceptance checklist

An external runtime becomes a release candidate only after its adapter pins and
records:

1. source/package revision and executable hash;
2. exact non-interactive input/output protocol and transport mode;
3. a resolved snapshot or content hash of prompt, model, effort, tools, roster,
   permissions, and environment configuration—not only mutable resource IDs;
4. process, filesystem, network, and credential boundaries;
5. complete root/child token and cost accounting, its price source (provider-returned,
   public list price, or local estimate), or an explicit lower-bound label;
6. cancellation, timeout, partial-output, and workspace-contamination behavior;
7. deterministic offline protocol tests before any paid benchmark run.

The current release prioritizes exact public boundaries and clean controlled
baselines. Rows marked as external targets are documented gaps, not hidden stubs. A
passing smoke test is internal QA, not evidence for a release winner.
