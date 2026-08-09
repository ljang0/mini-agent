from __future__ import annotations

from .base import HarnessSignature, ImplementationProfile, profile
from .sources import (
    ANTHROPIC_COMPUTER,
    MACU,
    MACU_PAPER,
    META_MUSE,
    OPENAI_COMPUTER,
    OPENAI_MULTI_AGENT,
    OSWORLD2,
)


def _h(name: str, **options: object) -> HarnessSignature:
    return HarnessSignature.create(name, options)


PROFILES: tuple[ImplementationProfile, ...] = (
    profile(
        application="computer-use",
        profile_id="openai-ga-computer-single",
        title="OpenAI GA computer protocol, single agent",
        artifact_kind="provider_protocol",
        fidelity="exact_public_protocol",
        runtime_owner="scaffoldlab_client",
        harnesses=(_h("single"),),
        providers=("openai-responses",),
        environment_types=("computer",),
        model_families=("GPT-5.4", "GPT-5.6"),
        exact_components=(
            "ordered computer_call.actions execution",
            "one original-detail screenshot and computer_call_output continuation",
        ),
        unavailable_components=(
            "model training, full confirmation prompt, classifiers, and product runtime",
        ),
        sources=(OPENAI_COMPUTER,),
    ),
    profile(
        application="computer-use",
        profile_id="openai-hosted-multi-agent-computer",
        title="OpenAI hosted multi-agent with GA computer tool",
        artifact_kind="provider_protocol",
        fidelity="source_matched_reimplementation",
        runtime_owner="openai_hosted",
        harnesses=(_h("openai_hosted_multi_agent", max_concurrent_subagents=3),),
        providers=("openai-responses",),
        environment_types=("computer",),
        model_families=("GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna"),
        exact_components=(
            "documented multi-agent request plus GA computer client loop",
            "agent-attributed calls routed to isolated logical environments",
        ),
        unavailable_components=(
            "hosted scheduler source, injected prompts, and a published or live-verified combined multi-agent/computer compatibility contract",
        ),
        sources=(OPENAI_MULTI_AGENT, OPENAI_COMPUTER),
    ),
    profile(
        application="computer-use",
        profile_id="anthropic-computer-20251124-single",
        title="Anthropic computer_20251124 wire schema with local browser executor",
        artifact_kind="provider_protocol",
        fidelity="exact_public_protocol",
        runtime_owner="scaffoldlab_client",
        harnesses=(_h("single"),),
        providers=("anthropic-messages",),
        environment_types=("computer",),
        model_families=(
            "Claude Opus 5",
            "Claude Sonnet 5",
            "Claude Opus 4.8/4.7/4.6/4.5",
            "Claude Sonnet 4.6",
        ),
        exact_components=(
            "computer-use-2025-11-24 beta header and tool schema",
            "assistant tool_use followed by user tool_result request ordering",
            "published 20251124 action vocabulary with zoom disabled",
        ),
        unavailable_components=(
            "Anthropic reference Linux X11/VNC executor, action timing, and screenshot implementation; actions run in a local Playwright browser viewport",
            "injected computer prompt, classifier, model training, and product runtime",
        ),
        sources=(ANTHROPIC_COMPUTER,),
    ),
    profile(
        application="computer-use",
        profile_id="meta-muse-spark-1.1-computer-use",
        title="Meta Muse Spark 1.1 computer-use orchestration",
        artifact_kind="provider_protocol",
        fidelity="documented_gap",
        status="catalog_only",
        runtime_owner="meta_hosted",
        providers=(),
        environment_types=(),
        model_families=("Muse Spark 1.1",),
        exact_components=(
            "public description of script/UI choice and batched computer actions",
            "public description of parallel subagent delegation",
        ),
        unavailable_components=(
            "public computer action schema, scheduler API, prompts, limits, and timing",
        ),
        sources=(META_MUSE,),
    ),
    profile(
        application="computer-use",
        profile_id="macu-text-dag",
        title="MACU-inspired text DAG over computer tools",
        fidelity="source_matched_reimplementation",
        runtime_owner="scaffoldlab_local",
        harnesses=(_h("macu_dynamic_dag", max_workers=4),),
        providers=("openai-responses", "anthropic-messages"),
        environment_types=("computer",),
        model_families=("provider-selected manager and CUA model",),
        exact_components=(
            "mutable DAG, ready-frontier parallelism, continuous replanning",
        ),
        unavailable_components=(
            "released MACU prompts, VM cloning, CUA subprocesses, files, and evaluator",
        ),
        sources=(MACU_PAPER, MACU),
    ),
    profile(
        application="computer-use",
        profile_id="macu-upstream-generic-vm",
        title="Pinned released MACU generic-task VM runtime",
        fidelity="upstream_runtime_adapter",
        runtime_owner="macu_upstream",
        harnesses=(_h("macu_upstream"),),
        providers=("macu-upstream",),
        environment_types=("none",),
        model_families=("GPT-5.4 CUA", "Qwen 3.6 CUA", "supported manager models"),
        exact_components=(
            "released prompts, graph validation retries, async scheduler and replanning",
            "VM variants/init_from, CUA subprocesses, final selection and result ingestion",
            "released generic no-initial-setup task-file path",
        ),
        unavailable_components=(
            "OSWorld benchmark domain/UUID loading, canonical task setup, and evaluator are not entered by this adapter",
            "bit identity unless dependencies, VM image, and model snapshots are pinned; released summary omits initial graph-generation usage",
        ),
        sources=(MACU, MACU_PAPER),
    ),
    profile(
        application="computer-use",
        profile_id="macu-osworld1-benchmark-parity",
        title="MACU OSWorld 1.x benchmark parity",
        artifact_kind="evaluation_environment",
        fidelity="documented_gap",
        status="catalog_only",
        runtime_owner="macu_osworld_upstream",
        providers=(),
        environment_types=(),
        model_families=("MACU-supported manager and CUA models",),
        exact_components=(
            "released domain/UUID task map, canonical OSWorld loader/setup, and evaluator integration",
        ),
        unavailable_components=(
            "the current adapter materializes a generic no-initial-setup list task rather than a domain/UUID task map",
            "OSWorld data-directory, task/evaluator assets, and VM image identity require explicit pins",
        ),
        sources=(MACU, MACU_PAPER),
    ),
    profile(
        application="computer-use",
        profile_id="osworld2-2026-06-24",
        title="OSWorld 2.0 evaluation environment",
        artifact_kind="evaluation_environment",
        fidelity="documented_gap",
        status="catalog_only",
        runtime_owner="osworld_upstream",
        providers=(),
        environment_types=(),
        model_families=("environment only; model-independent",),
        exact_components=(
            "published release manifest and reset/step/evaluate boundary",
        ),
        unavailable_components=(
            "gated task/evaluator assets and pinned VM image are not bundled",
        ),
        sources=(OSWORLD2,),
    ),
)
