# Frontier migration

The migration preserves the complete audited inventory under the `mini-agent`
interface without promoting unavailable systems or controlled studies into exact
implementations.

| Surface | Web | CUA | SWE | Total |
|---|---:|---:|---:|---:|
| Exact references | 7 | 3 | 8 | 18 |
| Studies | 10 | 2 | 16 | 28 |
| Documented gaps | 2 | 3 | 4 | 9 |
| All catalog entries | 19 | 8 | 28 | 55 |

The frontier manifest contains 18 labs and 54 application-status cells. Ten
frontier links resolve to exact references; xAI has two distinct web variants.

| Lab | Web | CUA | SWE |
|---|---|---|---|
| Anthropic | protocol | protocol | distribution |
| OpenAI | protocol | protocol | source |
| xAI | protocol (2) | gap | source |
| Moonshot AI / Kimi | gap | gap | source |
| Google DeepMind | gap | gap | gap |
| Microsoft | gap | gap | gap |
| Alibaba / Qwen | gap | gap | gap |
| Mistral AI | gap | gap | gap |
| Meta | gap | gap | gap |
| Amazon Web Services | gap | gap | gap |
| NVIDIA | gap | gap | gap |
| ByteDance | gap | gap | gap |
| Tencent | gap | gap | gap |
| Z.ai / GLM | gap | gap | gap |
| Baidu | gap | gap | gap |
| DeepSeek | model only | model only | model only |
| MiniMax | model only | model only | model only |
| Cohere | model only | model only | model only |

“Protocol,” “source,” and “distribution” describe the public boundary that can be
executed. They do not claim parity with a private flagship product. A compatible
model endpoint is also not evidence that the vendor's agent implementation has
been reproduced.

The authoritative machine-readable view is:

```bash
mini-agent catalog --json
mini-agent frontiers --json
mini-agent applications --json
mini-agent harnesses
```

All 18 exact references and all 28 studies are validateable and runnable through
the `mini-agent` CLI. They remain separate execution modes; a study never acquires
an exactness claim by being runnable. The nine gaps have no execution command.

Exact reference execution requires the same credentials, checked-out source or
installed distribution, environment, and task/evaluator configuration as the
preserved runtime. Missing public prompts, schedulers, compaction, timing, model
snapshots, and managed-service policy remain explicit limitations.
