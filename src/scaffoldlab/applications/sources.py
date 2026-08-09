from __future__ import annotations

from .base import SourceArtifact


OPENAI_MULTI_AGENT = SourceArtifact(
    title="OpenAI Responses multi-agent guide",
    url="https://developers.openai.com/api/docs/guides/responses-multi-agent",
    published="2026",
    version="responses_multi_agent=v1",
)
OPENAI_COMPUTER = SourceArtifact(
    title="OpenAI computer use guide",
    url="https://developers.openai.com/api/docs/guides/tools-computer-use",
    published="2026",
    version="GA computer tool",
)
OPENAI_SHELL = SourceArtifact(
    title="OpenAI shell tool guide",
    url="https://developers.openai.com/api/docs/guides/tools-shell",
    published="2026",
    version="local shell tool",
)
OPENAI_GPT56 = SourceArtifact(
    title="OpenAI GPT-5.6 model guide",
    url="https://developers.openai.com/api/docs/guides/latest-model.md",
    published="2026-07",
    version="gpt-5.6",
)
ANTHROPIC_FABLE = SourceArtifact(
    title="Claude Fable 5 and Mythos 5 system card",
    url=(
        "https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/"
        "Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf"
    ),
    published="2026",
    version="section 8.15.3 (multi-agent evaluation uses Claude Mythos 5)",
)
ANTHROPIC_OPUS5 = SourceArtifact(
    title="Claude Opus 5 system card",
    url=(
        "https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/"
        "Claude%20Opus%205%20System%20Card.pdf"
    ),
    published="2026",
    version="section 8.11.3",
)
ANTHROPIC_MANAGED = SourceArtifact(
    title="Anthropic Managed Agents multi-agent orchestration",
    url=("https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration"),
    published="2026-04",
    version="managed-agents-2026-04-01",
)
ANTHROPIC_COMPUTER = SourceArtifact(
    title="Anthropic computer use tool guide",
    url=(
        "https://platform.claude.com/docs/en/agents-and-tools/tool-use/"
        "computer-use-tool"
    ),
    published="2025-11-24",
    version="computer_20251124",
)
BROWSER_USE = SourceArtifact(
    title="Browser-Use parallel agents template and runtime",
    url=(
        "https://github.com/browser-use/browser-use/tree/"
        "f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc"
    ),
    published="2026",
    version="0.13.7",
    revision="f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc",
)
MACU = SourceArtifact(
    title="Multi-Agent Computer Use released runtime",
    url=(
        "https://github.com/kohjingyu/multi-agent-computer-use/tree/"
        "5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105"
    ),
    published="2026-06-01",
    revision="5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105",
)
MACU_PAPER = SourceArtifact(
    title="Multi-Agent Computer Use paper",
    url="https://arxiv.org/abs/2606.01533",
    published="2026-06-01",
    version="arXiv:2606.01533v1",
)
RLM = SourceArtifact(
    title="Official Recursive Language Models runtime",
    url="https://github.com/alexzhang13/rlm/releases/tag/v0.1.3",
    published="2026-06-26",
    version="0.1.3",
    revision="72d6940142ddfb84ee6be573dc999a37e633e671",
)
RAO_PAPER = SourceArtifact(
    title="Recursive Agent Optimization paper",
    url="https://arxiv.org/abs/2605.06639",
    published="2026-05-07",
    version="arXiv:2605.06639v1",
)
PLATOON_RAO_SNAPSHOT = SourceArtifact(
    title="Platoon RAO paper snapshot",
    url=(
        "https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe"
    ),
    published="2026-07-27",
    version="0.1.0",
    revision="d9c5857d3a0a056ebc9b047241a2a0c9515aafbe",
)
PRIME_AGENT = SourceArtifact(
    title="Prime Agent released runtime",
    url=(
        "https://github.com/PrimeIntellect-ai/prime-agent/tree/"
        "95afd319a78ae017a41241d50b013d656a0685ce"
    ),
    published="2026-08-07",
    version="0.7.1",
    revision="95afd319a78ae017a41241d50b013d656a0685ce",
)
GROK_BUILD = SourceArtifact(
    title="xAI Grok Build released runtime",
    url=(
        "https://github.com/xai-org/grok-build/tree/"
        "8a14c91d88875a831a38b3a066b1683116bcb31c"
    ),
    published="2026",
    version="1.0.0",
    revision="8a14c91d88875a831a38b3a066b1683116bcb31c",
)
XAI_MULTI_AGENT = SourceArtifact(
    title="xAI hosted multi-agent API",
    url="https://docs.x.ai/developers/model-capabilities/text/multi-agent",
    published="2026-07-02",
    version="grok-4.20-multi-agent-0309",
)
META_MUSE = SourceArtifact(
    title="Meta Muse Spark 1.1 release and Model API preview",
    url="https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/",
    published="2026-07-09",
    version="Muse Spark 1.1",
)
META_ARE = SourceArtifact(
    title="Meta Agents Research Environments",
    url="https://github.com/facebookresearch/meta-agents-research-environments",
    published="2026",
    version="public repository",
)
OSWORLD2 = SourceArtifact(
    title="OSWorld 2.0 release",
    url="https://github.com/xlang-ai/OSWorld-V2/releases/tag/v2026.06.24",
    published="2026-06-24",
    version="v2026.06.24",
    revision="2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6",
)
SWE_BENCH = SourceArtifact(
    title="SWE-bench evaluation runtime",
    url="https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0",
    published="2026",
    version="4.1.0",
)
