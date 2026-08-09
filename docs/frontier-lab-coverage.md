# Frontier-lab coverage

This table turns the 18-lab survey in
`frontier_multi_agent_harnesses_2026_readable.pdf` into an implementation decision.
“Model runnable” and “named harness reproduced” are intentionally different claims.

| Lab | Strongest public artifact in the survey | Scaffold Lab path | Fidelity/status |
| --- | --- | --- | --- |
| Anthropic | Fable/Mythos and Opus system cards; Messages tools; Managed Agents | `blocking_orchestrator`, `fixed_agent_team`, `async_subagents`; `anthropic-messages`; `anthropic-managed-agents` | Card variants are topology simulations. Supported Messages client-tool protocols and the Managed session/primary-event-list/aggregate-usage boundary are implemented from public docs; child-thread traces and the scheduler remain hosted. |
| OpenAI | Responses multi-agent beta; Codex; Agents SDK | `openai_hosted_multi_agent`; `openai-responses`; all local harnesses | Public stateless HTTP request and developer-function continuation boundaries are implemented in separate tool-less and developer-tool profiles. The WebSocket transport and server scheduler are not. |
| Google DeepMind | ADK and A2A | Any local harness through a compatible model endpoint; upstream ADK/A2A remains an external comparator | No claim that local topologies reproduce Gemini Deep Research/Deep Think. No dedicated ADK bridge yet. |
| xAI | Grok Build source and hosted multi-agent API | `grok_build`; `xai_hosted_multi_agent` | Upstream CLI adapter plus exact public hosted request boundary. Hosted scheduler remains closed. |
| Microsoft | AutoGen and Microsoft Agent Framework | Local fixed/async teams as controlled baselines; upstream framework is an external comparator | No AutoGen/MAF parity claim. A source-pinned upstream adapter is preferable to reimplementing their full runtimes. |
| Alibaba/Qwen | Qwen-Agent GroupChat and Qwen Code | Compatible model endpoint plus local topologies | Model portability only. The moderated GroupChat runtime is not yet wrapped. |
| Moonshot/Kimi | [Kimi Code](https://github.com/MoonshotAI/kimi-code) and AgentSwarm disclosures | Compatible model endpoint; upstream CLI is an external-adapter target | No hosted AgentSwarm parity claim. |
| Mistral | Vibe subagents and managed durable agents/workflows | Compatible endpoint plus local topologies | Model portability only; no Vibe/durable-runtime wrapper yet. |
| Meta | [MATRIX](https://github.com/facebookresearch/matrix) and Muse disclosures | Compatible Meta endpoint plus local topologies | MATRIX is a distributed data-generation runtime; Muse's deployed scheduler remains closed. |
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

- OpenAI Responses, with current developer-function continuation and separately
  implemented computer and local-shell client protocols. Their combination with the
  hosted multi-agent beta requires separate live compatibility validation;
- Anthropic Messages, with current bash, text-editor, and computer client schemas when
  the selected model supports that tool version. The documented
  `computer_20251124` list includes Opus 5 but not Fable 5 or Mythos 5;
- generic OpenAI-compatible Responses or Chat Completions function calling;
- a dedicated hosted or released-runtime adapter.

That gives broad model coverage without pretending that an OpenAI-compatible model
endpoint exposes its vendor's private orchestration stack. If a public upstream
runtime owns essential scheduler state, prompts, retries, compaction, or message
delivery, the faithful next step is a pinned external adapter, not a similarly named
clean-room class.

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
baselines. Rows marked as external targets are documented gaps, not hidden stubs.
