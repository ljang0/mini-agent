from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .base import APPLICATION_NAMES


FRONTIER_MANIFEST_AS_OF = "2026-08-10"


@dataclass(frozen=True)
class FrontierApplicationStatus:
    """Scaffold Lab coverage for one lab in one application."""

    application: str
    status: str
    implementation_ids: tuple[str, ...] = ()
    boundary: str = ""

    def __post_init__(self) -> None:
        if self.application not in APPLICATION_NAMES:
            raise ValueError("frontier manifest contains an unknown application")
        if self.status not in {
            "source_executed",
            "distribution_executed",
            "protocol_executed",
            "catalog_gap",
            "model_only",
        }:
            raise ValueError(f"invalid Scaffold Lab status {self.status!r}")
        executed = self.status in {
            "source_executed",
            "distribution_executed",
            "protocol_executed",
        }
        if executed != bool(self.implementation_ids):
            raise ValueError(
                "executed application statuses require implementation ids and gaps "
                "or model-only statuses cannot name runnable implementations"
            )
        if any(not value or "/" in value for value in self.implementation_ids):
            raise ValueError("frontier implementation ids must be local profile ids")
        if not self.boundary:
            raise ValueError("frontier application statuses require a boundary")

    def as_dict(self) -> dict[str, Any]:
        return {
            "application": self.application,
            "status": self.status,
            "implementation_ids": list(self.implementation_ids),
            "boundary": self.boundary,
        }


def _uniform_statuses(
    applications: tuple[str, ...], status: str, *, boundary: str
) -> tuple[FrontierApplicationStatus, ...]:
    return tuple(
        FrontierApplicationStatus(
            application=application,
            status=status,
            boundary=boundary,
        )
        for application in applications
    )


@dataclass(frozen=True)
class FrontierDistributionIdentity:
    """Audited package/executable identity for a non-source distribution."""

    platform: str
    wrapper_package: str
    wrapper_integrity: str
    native_package: str
    native_integrity: str
    executable_sha256: str
    public_tag_revision: str = ""
    public_repository_is_runtime_source: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.platform,
                self.wrapper_package,
                self.wrapper_integrity,
                self.native_package,
                self.native_integrity,
            )
        ):
            raise ValueError("distribution identity fields must be non-empty")
        if not self.wrapper_integrity.startswith(
            "sha512-"
        ) or not self.native_integrity.startswith("sha512-"):
            raise ValueError("distribution package integrities must be sha512 values")
        if len(self.executable_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.executable_sha256
        ):
            raise ValueError("distribution executable identity requires SHA-256")
        if self.public_tag_revision and (
            len(self.public_tag_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.public_tag_revision
            )
        ):
            raise ValueError("public tag revision must be a Git commit")
        if self.public_repository_is_runtime_source:
            raise ValueError(
                "public distributions cannot mark a non-source tag as runtime source"
            )


@dataclass(frozen=True)
class FrontierLabRecord:
    """Audited first-party runtime and model/card evidence for one lab.

    ``runtime_revision`` is the dereferenced commit when the recorded artifact has
    one, including model-only source. It is empty for managed protocols. A source
    record is discovery evidence, not a runnable Scaffold Lab implementation.
    Coverage is application-specific so a public SWE source adapter cannot silently
    promote an unrelated hosted browser or computer-use boundary.
    """

    lab: str
    model_families: tuple[str, ...]
    runtime_kind: str
    runtime_title: str
    runtime_url: str
    runtime_version: str
    runtime_revision: str
    runtime_entrypoint: str
    evidence_title: str
    evidence_url: str
    application_statuses: tuple[FrontierApplicationStatus, ...]
    flagship_exact: str
    limitation: str
    distribution_identity: FrontierDistributionIdentity | None = None
    audited_at: str = FRONTIER_MANIFEST_AS_OF

    def __post_init__(self) -> None:
        if self.runtime_kind not in {
            "source_runtime",
            "public_distribution",
            "managed_protocol",
            "single_agent_source",
            "no_team_runtime",
        }:
            raise ValueError(f"invalid runtime kind {self.runtime_kind!r}")
        if self.flagship_exact not in {"no", "partial"}:
            raise ValueError(f"invalid flagship exactness {self.flagship_exact!r}")
        if self.audited_at != FRONTIER_MANIFEST_AS_OF:
            raise ValueError("frontier manifest records must use the shared audit date")
        if not self.runtime_url.startswith(
            "https://"
        ) or not self.evidence_url.startswith("https://"):
            raise ValueError("frontier evidence URLs must use HTTPS")
        if self.runtime_kind == "source_runtime" and (
            len(self.runtime_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.runtime_revision
            )
        ):
            raise ValueError("source runtimes require a 40-character Git revision")
        if self.runtime_kind == "public_distribution":
            if self.runtime_revision:
                raise ValueError(
                    "public distributions must not put a non-source tag in "
                    "runtime_revision"
                )
            if self.distribution_identity is None:
                raise ValueError(
                    "public distributions require an audited distribution identity"
                )
        elif self.distribution_identity is not None:
            raise ValueError(
                "distribution identity is valid only for public distributions"
            )
        applications = [item.application for item in self.application_statuses]
        if len(applications) != len(set(applications)):
            raise ValueError(
                "frontier records require unique application-specific statuses"
            )
        if set(applications) != set(APPLICATION_NAMES):
            raise ValueError(
                "frontier records require status for browser, computer-use, and swe"
            )

    @property
    def applications(self) -> tuple[str, ...]:
        return tuple(item.application for item in self.application_statuses)

    @property
    def scaffoldlab_status(self) -> str:
        """Compatibility aggregate; machine output uses per-application statuses."""

        statuses = {item.status for item in self.application_statuses}
        return next(iter(statuses)) if len(statuses) == 1 else "mixed"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_families"] = list(self.model_families)
        value["applications"] = list(self.applications)
        value["application_statuses"] = [
            item.as_dict() for item in self.application_statuses
        ]
        if self.distribution_identity is None:
            value.pop("distribution_identity", None)
        return value


FRONTIER_LABS: tuple[FrontierLabRecord, ...] = (
    FrontierLabRecord(
        lab="Anthropic",
        model_families=("Claude Opus 5", "Claude Fable 5", "Claude Mythos 5"),
        runtime_kind="public_distribution",
        runtime_title="Claude Code Agent Teams and subagents",
        runtime_url="https://www.npmjs.com/package/@anthropic-ai/claude-code/v/2.1.226",
        runtime_version="Claude Code 2.1.226",
        runtime_revision="",
        runtime_entrypoint="Claude Code product sessions",
        evidence_title="Claude Fable 5 and Mythos 5 system card",
        evidence_url=(
            "https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/"
            "Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf"
        ),
        application_statuses=(
            FrontierApplicationStatus(
                application="browser",
                status="protocol_executed",
                implementation_ids=("anthropic-managed-web-research",),
                boundary="Managed Agents protocol implementation; not the Claude Code distribution or a system-card experiment.",
            ),
            FrontierApplicationStatus(
                application="computer-use",
                status="protocol_executed",
                implementation_ids=("anthropic-computer-20251124-single",),
                boundary="Published computer-use wire protocol; not Claude Code Agent Teams.",
            ),
            FrontierApplicationStatus(
                application="swe",
                status="distribution_executed",
                implementation_ids=("claude-code-agent-teams-2.1.226",),
                boundary="Audited official Darwin arm64 distribution and public CLI; matching runtime source and server-managed policy are unavailable.",
            ),
        ),
        flagship_exact="partial",
        limitation="The audited Claude Code distribution runs for SWE and separate managed/browser and computer wire contracts run; matching executable source, server policy, and system-card prompts, timing, compaction, safeguards, and evaluator state remain unavailable.",
        distribution_identity=FrontierDistributionIdentity(
            platform="darwin-arm64",
            wrapper_package="@anthropic-ai/claude-code@2.1.226",
            wrapper_integrity=(
                "sha512-mUkA81SbzATHFsHNz/rPy3Itw0D0S9kQMsIUJ3qPGwpNJMqPePyDP6xnWHI0"
                "jfFlspVjs8r/GfolMUyiy8P1FQ=="
            ),
            native_package="@anthropic-ai/claude-code-darwin-arm64@2.1.226",
            native_integrity=(
                "sha512-/vIgn1GB6SiOHMcx7zVDZej2Vk+hDr2qkd4aKTryoPm2THorWW3lPpCkzoa4OArg"
                "5na1K+eNhGdenhefWthtsw=="
            ),
            executable_sha256=(
                "013a1cf17df5ff1dcc189d5d6fd3fdd5f097ddc3cd41aa9992e99805574febbe"
            ),
            public_tag_revision="2bb60696142b493eafaeacfe00eac51d16c50c4f",
            public_repository_is_runtime_source=False,
        ),
    ),
    FrontierLabRecord(
        lab="OpenAI",
        model_families=("GPT-5.6", "Codex"),
        runtime_kind="source_runtime",
        runtime_title="OpenAI Codex",
        runtime_url="https://github.com/openai/codex/tree/be6e8eac029b183056b7e4402879f15d2c85f61b",
        runtime_version="rust-v0.147.0",
        runtime_revision="be6e8eac029b183056b7e4402879f15d2c85f61b",
        runtime_entrypoint="codex CLI and app-server",
        evidence_title="GPT-5.6 system card",
        evidence_url="https://deploymentsafety.openai.com/gpt-5-6",
        application_statuses=(
            FrontierApplicationStatus(
                application="browser",
                status="protocol_executed",
                implementation_ids=("openai-hosted-multi-agent-functions",),
                boundary="Hosted Responses multi-agent protocol with browser functions; hosted scheduler source is closed.",
            ),
            FrontierApplicationStatus(
                application="computer-use",
                status="protocol_executed",
                implementation_ids=("openai-ga-computer-single",),
                boundary="Published GA computer-use protocol; not the local Codex source runtime.",
            ),
            FrontierApplicationStatus(
                application="swe",
                status="source_executed",
                implementation_ids=("openai-codex-source-0.147.0",),
                boundary="Pinned public Codex source plus a separate hosted Responses protocol; private service routing and model state remain unavailable.",
            ),
        ),
        flagship_exact="partial",
        limitation="The pinned local Codex source and separate hosted Responses/browser and GA computer protocols run; remote model state, service routing, cloud/managed policy, complete tree accounting, and private ChatGPT internals remain unavailable.",
    ),
    FrontierLabRecord(
        lab="Google DeepMind",
        model_families=("Gemini 3.x",),
        runtime_kind="source_runtime",
        runtime_title="Google Agent Development Kit",
        runtime_url="https://github.com/google/adk-python/tree/0b55dcf9d32e22d4c8b303c3da1c275c135682bf",
        runtime_version="2.6.3",
        runtime_revision="0b55dcf9d32e22d4c8b303c3da1c275c135682bf",
        runtime_entrypoint="google.adk.agents.LlmAgent(sub_agents=...)",
        evidence_title="Google DeepMind model-card index",
        evidence_url="https://deepmind.google/models/model-cards/",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="ADK and model-card evidence are cataloged; no application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="ADK is executable generic framework source, not the undisclosed Gemini Deep Research or Deep Think coordinator.",
    ),
    FrontierLabRecord(
        lab="xAI",
        model_families=("Grok 4.x", "Grok 4.20"),
        runtime_kind="source_runtime",
        runtime_title="Grok Build",
        runtime_url="https://github.com/xai-org/grok-build/tree/8a14c91d88875a831a38b3a066b1683116bcb31c",
        runtime_version="1.0.0 audited source snapshot",
        runtime_revision="8a14c91d88875a831a38b3a066b1683116bcb31c",
        runtime_entrypoint="xai-grok-pager --output-format json",
        evidence_title="Grok 4.20 model card",
        evidence_url="https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf",
        application_statuses=(
            FrontierApplicationStatus(
                application="browser",
                status="protocol_executed",
                implementation_ids=(
                    "xai-hosted-web-research-4",
                    "xai-hosted-web-research-16",
                ),
                boundary="Hosted 4/16-agent web-research protocol; Grok Build source is a separate SWE artifact.",
            ),
            FrontierApplicationStatus(
                application="computer-use",
                status="catalog_gap",
                boundary="No xAI computer-use team protocol or Scaffold Lab adapter is registered.",
            ),
            FrontierApplicationStatus(
                application="swe",
                status="source_executed",
                implementation_ids=("grok-build-source-1.0.0",),
                boundary="Pinned public Grok Build source and native subagents; hosted scheduler and model state are unavailable.",
            ),
        ),
        flagship_exact="partial",
        limitation="The audited public source, historical headless contract, and hosted 4/16-agent API run; hosted scheduling, encrypted continuation, package identity, and complete accounting remain unavailable.",
    ),
    FrontierLabRecord(
        lab="Microsoft",
        model_families=("MAI models",),
        runtime_kind="source_runtime",
        runtime_title="AutoGen AgentChat",
        runtime_url="https://github.com/microsoft/autogen/tree/83afbf5857aac683340d4c692194e548b1e8edda",
        runtime_version="python-v0.7.5",
        runtime_revision="83afbf5857aac683340d4c692194e548b1e8edda",
        runtime_entrypoint="autogen_agentchat.teams",
        evidence_title="Microsoft responsible AI transparency report",
        evidence_url="https://www.microsoft.com/en-us/corporate-responsibility/responsible-ai-transparency-report/",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="AutoGen source is cataloged; no MAI/Copilot application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="AutoGen is a runnable general framework, not the proprietary Copilot or MAI product/evaluation scaffold.",
    ),
    FrontierLabRecord(
        lab="Alibaba / Qwen",
        model_families=("Qwen 3.x", "Qwen Code"),
        runtime_kind="source_runtime",
        runtime_title="Qwen-Agent GroupChat",
        runtime_url="https://github.com/QwenLM/Qwen-Agent/tree/37e7e5fe6053b0640381063df40560a85aacc697",
        runtime_version="0.0.26",
        runtime_revision="37e7e5fe6053b0640381063df40560a85aacc697",
        runtime_entrypoint="qwen_agent.agents.GroupChat",
        evidence_title="Qwen3.6 model repository and report",
        evidence_url="https://github.com/QwenLM/Qwen3.6",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="Qwen-Agent source is cataloged; no Qwen production application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="The moderated group-chat source is public, but no Qwen production coordinator or benchmark scaffold is released.",
    ),
    FrontierLabRecord(
        lab="Moonshot AI / Kimi",
        model_families=("Kimi K2/K3", "Kimi Code"),
        runtime_kind="source_runtime",
        runtime_title="Kimi Code Agent and AgentSwarm",
        runtime_url="https://github.com/MoonshotAI/kimi-code/tree/f0614c53e59f7e1e257412063b059b9eb82764cf",
        runtime_version="0.34.0",
        runtime_revision="f0614c53e59f7e1e257412063b059b9eb82764cf",
        runtime_entrypoint="apps/kimi-code/src/main.ts --output-format stream-json",
        evidence_title="Kimi K3 technical report",
        evidence_url="https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf",
        application_statuses=(
            FrontierApplicationStatus(
                application="browser",
                status="catalog_gap",
                boundary="Kimi Code source is cataloged for SWE; no Kimi browser team adapter is registered.",
            ),
            FrontierApplicationStatus(
                application="computer-use",
                status="catalog_gap",
                boundary="Kimi Code source is cataloged for SWE; no Kimi computer-use adapter is registered.",
            ),
            FrontierApplicationStatus(
                application="swe",
                status="source_executed",
                implementation_ids=("kimi-code-0.34.0-upstream",),
                boundary=(
                    "Pinned public Kimi Code Agent/AgentSwarm tracked source is "
                    "verified before execution; ignored/generated dependencies, "
                    "hosted product services, and model state are unavailable."
                ),
            ),
        ),
        flagship_exact="partial",
        limitation=(
            "Pinned tracked Agent/AgentSwarm source runs with caller-installed "
            "dependencies; hosted product scheduling, search, private prompts, "
            "and complete usage remain unavailable."
        ),
    ),
    FrontierLabRecord(
        lab="Mistral AI",
        model_families=("Mistral Large", "Mistral Medium"),
        runtime_kind="source_runtime",
        runtime_title="Mistral Vibe subagents",
        runtime_url="https://github.com/mistralai/mistral-vibe/tree/b78b451c39eab9213393ad2f45908e8562a5c5e7",
        runtime_version="2.24.0",
        runtime_revision="b78b451c39eab9213393ad2f45908e8562a5c5e7",
        runtime_entrypoint="vibe -p with task(..., agent=...)",
        evidence_title="Mistral Vibe source documentation",
        evidence_url="https://github.com/mistralai/mistral-vibe",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="Mistral Vibe source is cataloged but not executed by a Scaffold Lab adapter.",
        ),
        flagship_exact="no",
        limitation="Vibe source is runnable; managed Durable Agents and private evaluation/runtime settings remain separate.",
    ),
    FrontierLabRecord(
        lab="Meta",
        model_families=("Llama", "Muse Spark 1.1"),
        runtime_kind="source_runtime",
        runtime_title="Meta Matrix",
        runtime_url="https://github.com/facebookresearch/matrix/tree/280c96ab2cea1e3a4e83bdaea4995c8fafa19bfb",
        runtime_version="0.2.6",
        runtime_revision="280c96ab2cea1e3a4e83bdaea4995c8fafa19bfb",
        runtime_entrypoint="Ray-backed peer-to-peer workflows",
        evidence_title="Muse Spark 1.1 evaluation report",
        evidence_url="https://ai.meta.com/static-resource/muse-spark-1-1-evaluation-report",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="Matrix and Muse evidence are cataloged; no Muse application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="Matrix is a research framework, not Muse's hosted scheduler; public Muse prompts, scheduler API, limits, and child traces are absent.",
    ),
    FrontierLabRecord(
        lab="Amazon Web Services",
        model_families=("Amazon Nova",),
        runtime_kind="managed_protocol",
        runtime_title="Amazon Bedrock multi-agent collaboration",
        runtime_url="https://docs.aws.amazon.com/bedrock/latest/userguide/create-multi-agent-collaboration.html",
        runtime_version="managed service",
        runtime_revision="",
        runtime_entrypoint="Bedrock supervisor and collaborator APIs",
        evidence_title="Amazon Nova 2 Lite service card",
        evidence_url="https://docs.aws.amazon.com/pdfs/ai/responsible-ai/nova-2-lite/nova-2-lite.pdf",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="Bedrock collaboration is cataloged as a managed protocol; no application adapter is registered.",
        ),
        flagship_exact="partial",
        limitation="The service contract is public, but the server supervisor, prompts, model routing, and scheduler source are closed.",
    ),
    FrontierLabRecord(
        lab="NVIDIA",
        model_families=("Nemotron",),
        runtime_kind="source_runtime",
        runtime_title="NVIDIA NeMo Agent Toolkit",
        runtime_url="https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/3c44584ef5de2531e2ff548408f0e4658b755a69",
        runtime_version="1.8.0",
        runtime_revision="3c44584ef5de2531e2ff548408f0e4658b755a69",
        runtime_entrypoint="nat start console --config_file ...",
        evidence_title="NeMo Agent Toolkit agent components",
        evidence_url="https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/index.html",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="NeMo Agent Toolkit is cataloged; no canonical application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="NAT is an integration and optimization toolkit rather than one canonical frontier-model harness.",
    ),
    FrontierLabRecord(
        lab="ByteDance",
        model_families=("Seed", "Doubao"),
        runtime_kind="source_runtime",
        runtime_title="DeerFlow",
        runtime_url="https://github.com/bytedance/deer-flow/tree/7e7f0410797693cf882594555ba414e0361d4c6f",
        runtime_version="2.0.0",
        runtime_revision="7e7f0410797693cf882594555ba414e0361d4c6f",
        runtime_entrypoint="make_lead_agent -> task -> SubagentExecutor",
        evidence_title="DeerFlow 2.0 release",
        evidence_url="https://github.com/bytedance/deer-flow/releases/tag/v2.0.0",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="DeerFlow source is cataloged; no Seed/Doubao application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="DeerFlow is strong runnable manager-worker source, but it is not confirmed as the private Seed/Doubao product coordinator.",
    ),
    FrontierLabRecord(
        lab="Tencent",
        model_families=("Hunyuan",),
        runtime_kind="source_runtime",
        runtime_title="YunqueAgent",
        runtime_url="https://github.com/Tencent-BAC/YunqueAgent/tree/83653cb0bd7884a54a7f10dedeff7d1c62ebf823",
        runtime_version="2026-08-10 audited snapshot",
        runtime_revision="83653cb0bd7884a54a7f10dedeff7d1c62ebf823",
        runtime_entrypoint="bash run.sh",
        evidence_title="Yunque DeepResearch technical report",
        evidence_url="https://arxiv.org/abs/2601.19578",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="YunqueAgent research source is cataloged but not executed by a browser adapter.",
        ),
        flagship_exact="no",
        limitation="The reproducible research framework is not Hunyuan's production runtime and depends on external search/model services.",
    ),
    FrontierLabRecord(
        lab="Z.ai / GLM",
        model_families=("GLM-5",),
        runtime_kind="source_runtime",
        runtime_title="Synapse",
        runtime_url="https://github.com/zai-org/Synapse/tree/50bec2a744dcbd39611aa2b291c6aade070adf2e",
        runtime_version="0.29.1",
        runtime_revision="50bec2a744dcbd39611aa2b291c6aade070adf2e",
        runtime_entrypoint="conversation actors and remote-agent daemon",
        evidence_title="GLM-5 model artifact",
        evidence_url="https://github.com/zai-org/GLM-5",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="Synapse source is cataloged; no GLM application runtime adapter is registered.",
        ),
        flagship_exact="no",
        limitation="Synapse is an early conversation workspace, not a released GLM product or benchmark coordinator.",
    ),
    FrontierLabRecord(
        lab="Baidu",
        model_families=("ERNIE",),
        runtime_kind="source_runtime",
        runtime_title="LoongFlow",
        runtime_url="https://github.com/baidu-baige/LoongFlow/tree/cd11477c2f92531412d1917f55e583c39a4608e8",
        runtime_version="0.0.2",
        runtime_revision="cd11477c2f92531412d1917f55e583c39a4608e8",
        runtime_entrypoint="run_general.sh PES/evolution loop",
        evidence_title="LoongFlow paper",
        evidence_url="https://arxiv.org/abs/2512.24077",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "catalog_gap",
            boundary="LoongFlow research source is cataloged but not executed by an application adapter.",
        ),
        flagship_exact="no",
        limitation="LoongFlow is public expert/evolutionary agent source, not ERNIE Bot's production multi-agent scheduler.",
    ),
    FrontierLabRecord(
        lab="DeepSeek",
        model_families=("DeepSeek V3", "DeepSeek R1"),
        runtime_kind="no_team_runtime",
        runtime_title="DeepSeek-V3 model and inference source",
        runtime_url="https://github.com/deepseek-ai/DeepSeek-V3/tree/f6e34dd26772dd4a216be94a8899276c5dca9e43",
        runtime_version="1.0.0",
        runtime_revision="f6e34dd26772dd4a216be94a8899276c5dca9e43",
        runtime_entrypoint="external harness required",
        evidence_title="DeepSeek-V3 technical report",
        evidence_url="https://arxiv.org/abs/2412.19437",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "model_only",
            boundary="Model weights and inference source are cataloged; no first-party team runtime is available.",
        ),
        flagship_exact="no",
        limitation="Open model weights and inference are not a first-party general multi-agent harness.",
    ),
    FrontierLabRecord(
        lab="MiniMax",
        model_families=("MiniMax M2.x",),
        runtime_kind="single_agent_source",
        runtime_title="Mini-Agent",
        runtime_url="https://github.com/MiniMax-AI/Mini-Agent/tree/d76a4f6389688cabda39c224a6cdfa274215d47c",
        runtime_version="2026-08-10 audited snapshot",
        runtime_revision="d76a4f6389688cabda39c224a6cdfa274215d47c",
        runtime_entrypoint="single-agent demo",
        evidence_title="MiniMax-M2 model artifact",
        evidence_url="https://github.com/MiniMax-AI/MiniMax-M2",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "model_only",
            boundary="The first-party public demo is single-agent; no first-party team runtime is available.",
        ),
        flagship_exact="no",
        limitation="The first-party demo explicitly implements one agent; a team requires an external harness.",
    ),
    FrontierLabRecord(
        lab="Cohere",
        model_families=("Command A", "Command R"),
        runtime_kind="no_team_runtime",
        runtime_title="Cohere Toolkit and external framework integrations",
        runtime_url="https://github.com/cohere-ai/cohere-toolkit/tree/654588c4dbcdf9cb54597f19a2321cf3bd476048",
        runtime_version="1.1.7",
        runtime_revision="654588c4dbcdf9cb54597f19a2321cf3bd476048",
        runtime_entrypoint="external orchestration required",
        evidence_title="Cohere model documentation",
        evidence_url="https://docs.cohere.com/docs/models",
        application_statuses=_uniform_statuses(
            ("browser", "computer-use", "swe"),
            "model_only",
            boundary="Tool-capable models and integrations are cataloged; no canonical first-party team runtime is available.",
        ),
        flagship_exact="no",
        limitation="Cohere publishes tool-capable models and integrations, not a canonical first-party general team runtime.",
    ),
)


FRONTIER_LABS_BY_NAME: Mapping[str, FrontierLabRecord] = MappingProxyType(
    {record.lab: record for record in FRONTIER_LABS}
)

if len(FRONTIER_LABS) != 18 or len(FRONTIER_LABS_BY_NAME) != 18:
    raise RuntimeError("frontier source manifest must contain exactly 18 unique labs")


def list_frontier_labs() -> tuple[FrontierLabRecord, ...]:
    return FRONTIER_LABS
