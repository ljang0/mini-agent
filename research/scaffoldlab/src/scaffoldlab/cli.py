from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .applications import (
    ApplicationSelection,
    list_applications as catalog_applications,
    list_frontier_labs as catalog_frontier_labs,
    list_gaps as catalog_gaps,
    list_implementations as catalog_implementations,
    list_profiles as catalog_profiles,
    list_studies as catalog_studies,
    resolve_application_config,
)
from .browser_use_external import BrowserUseUpstreamBackend
from .claude_code_source import (
    CLAUDE_CODE_DISTRIBUTION_VERSION,
    ClaudeCodeAgentTeamsDistributionBackend,
)
from .codex_source import CODEX_SOURCE_REVISION, CodexSourceBackend
from .evaluation import (
    MatrixRunner,
    _atomic_write_text,
    load_tasks,
    validate_harness_variants,
)
from .environments import build_environment_factory
from .external import (
    GrokBuildJSONBackend,
    MACUUpstreamBackend,
    PrimeAgentJSONBackend,
    RLMUpstreamBackend,
)
from .grok_source import GROK_BUILD_PUBLIC_REVISION, GrokBuildSourceBackend
from .kimi_upstream import KimiCodeUpstreamBackend
from .prime_source import PRIME_AGENT_SOURCE_REVISION, PrimeAgentSourceBackend
from .harnesses import (
    AnthropicManagedAgentsHarness,
    AsyncSubagentsHarness,
    BlockingOrchestratorHarness,
    BrowserUseUpstreamHarness,
    ClaudeCodeAgentTeamsDistributionHarness,
    CodexSourceHarness,
    FixedAgentTeamHarness,
    ExternalContextJSONSearchHarness,
    FlatParallelHarness,
    GrokBuildHarness,
    GrokBuildSourceHarness,
    Harness,
    KimiCodeUpstreamHarness,
    MACUHarness,
    MACUUpstreamHarness,
    OpenAIHostedMultiAgentHarness,
    ParallelBestOfNHarness,
    PlatoonRecursiveInferenceHarness,
    PrimeAgentHarness,
    PrimeAgentSourceHarness,
    RLMREPLHarness,
    RLMUpstreamHarness,
    RecursiveDelegationHarness,
    SingleAgentHarness,
    XAIHostedMultiAgentHarness,
)
from .providers import (
    AnthropicManagedAgentsBackend,
    AnthropicMessagesBackend,
    OpenAICompatibleChatBackend,
    OpenAIResponsesBackend,
    TokenPricing,
    XAIResponsesBackend,
)
from .types import BudgetLimits


HARNESS_TYPES: Mapping[str, type[Harness]] = {
    "anthropic_managed_agents": AnthropicManagedAgentsHarness,
    "single": SingleAgentHarness,
    "flat_parallel": FlatParallelHarness,
    "grok_build": GrokBuildHarness,
    "grok_build_source": GrokBuildSourceHarness,
    "kimi_code_upstream": KimiCodeUpstreamHarness,
    "parallel_best_of_n": ParallelBestOfNHarness,
    "blocking_orchestrator": BlockingOrchestratorHarness,
    "browser_use_upstream": BrowserUseUpstreamHarness,
    "claude_code_agent_teams_distribution": (ClaudeCodeAgentTeamsDistributionHarness),
    "codex_source": CodexSourceHarness,
    "fixed_agent_team": FixedAgentTeamHarness,
    "async_subagents": AsyncSubagentsHarness,
    "macu_dynamic_dag": MACUHarness,
    "macu_upstream": MACUUpstreamHarness,
    "recursive_delegation": RecursiveDelegationHarness,
    "platoon_recursive_inference": PlatoonRecursiveInferenceHarness,
    "rlm_repl": RLMREPLHarness,
    "rlm_upstream": RLMUpstreamHarness,
    "external_context_json_search": ExternalContextJSONSearchHarness,
    "openai_hosted_multi_agent": OpenAIHostedMultiAgentHarness,
    "prime_agent": PrimeAgentHarness,
    "prime_agent_source": PrimeAgentSourceHarness,
    "xai_hosted_multi_agent": XAIHostedMultiAgentHarness,
}


# Exact application profiles make their cited source pin authoritative. These
# bindings translate provider-specific CLI controls back to the corresponding
# SourceArtifact field so an override cannot silently change the catalog claim.
_EXACT_RUNTIME_PIN_BINDINGS: Mapping[str, tuple[str, str, str]] = {
    "anthropic-managed-agents": (
        "managed_beta_version",
        "version",
        "Managed Agents beta version",
    ),
    "prime-agent": (
        "prime_agent_expected_version",
        "version",
        "Prime Agent version",
    ),
    "prime-agent-source": (
        "prime_source_expected_revision",
        "revision",
        "Prime Agent source revision",
    ),
    "grok-build": (
        "grok_expected_version",
        "version",
        "Grok Build version",
    ),
    "grok-build-source": (
        "grok_source_expected_revision",
        "revision",
        "Grok Build source revision",
    ),
    "macu-upstream": (
        "macu_expected_checkout_revision",
        "revision",
        "MACU checkout revision",
    ),
    "rlm-upstream": (
        "rlm_expected_checkout_revision",
        "revision",
        "RLM checkout revision",
    ),
    "kimi-code-upstream": (
        "kimi_expected_checkout_revision",
        "revision",
        "Kimi Code checkout revision",
    ),
    "browser-use-upstream": (
        "browser_use_expected_checkout_revision",
        "revision",
        "Browser-Use checkout revision",
    ),
    "codex-source": (
        "codex_source_expected_revision",
        "revision",
        "OpenAI Codex source revision",
    ),
    "claude-code-agent-teams": (
        "claude_code_expected_version",
        "version",
        "Claude Code distribution version",
    ),
}

_EXACT_RUNTIME_SOURCE_TITLES: Mapping[str, str] = {
    "claude-code-agent-teams": "Claude Code Agent Teams documentation",
}

_EXACT_PROVIDER_ORIGINS: Mapping[str, str] = {
    "openai-responses": "https://api.openai.com/v1",
    "anthropic-messages": "https://api.anthropic.com/v1",
    "anthropic-managed-agents": "https://api.anthropic.com/v1",
    "xai-responses": "https://api.x.ai/v1",
}

_EXACT_MODEL_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "computer-use/openai-ga-computer-single": ("gpt-5.4", "gpt-5.6"),
    "computer-use/anthropic-computer-20251124-single": (
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
    ),
}


def _matches_model_prefix(model: str, prefix: str) -> bool:
    return model == prefix or model.startswith(prefix + "-")


def _enforce_expected_application_key(
    expected: str | None, application: ApplicationSelection | None
) -> None:
    """Bind delegated runs to the catalog selection loaded by this process."""

    if expected is None:
        return
    actual = None if application is None else application.profile.key
    if actual != expected:
        raise ValueError(
            f"config selects application profile {actual!r}, not expected {expected!r}"
        )


def _enforce_exact_application_identity(
    args: argparse.Namespace, application: ApplicationSelection | None
) -> None:
    """Fail closed when a catalog selection drifts from its recorded identity.

    Raw legacy configs deliberately have no application selection and retain their
    existing override behavior. Provider origins and model allowlists apply only to
    exact implementations; recorded runtime pins apply to every selected profile.
    """

    if application is None:
        return

    profile = application.profile
    exact_implementation = application.selection_kind == "implementation"
    if exact_implementation:
        official_origin = _EXACT_PROVIDER_ORIGINS.get(args.provider)
        selected_origin = getattr(args, "base_url", None)
        if (
            official_origin is not None
            and selected_origin is not None
            and selected_origin.rstrip("/") != official_origin
        ):
            raise ValueError(
                f"exact implementation {profile.key!r} requires official provider "
                f"origin {official_origin!r}; --base-url={selected_origin!r} is a "
                "different experimental condition. Use an uncataloged compatible "
                "provider or study for proxies and alternate endpoints."
            )

        allowed_model_prefixes = _EXACT_MODEL_PREFIXES.get(profile.key)
        if allowed_model_prefixes is not None:
            selected_model = getattr(args, "model", None)
            if not isinstance(selected_model, str) or not any(
                _matches_model_prefix(selected_model, prefix)
                for prefix in allowed_model_prefixes
            ):
                raise ValueError(
                    f"exact implementation {profile.key!r} requires a documented "
                    "compatible model prefix from "
                    f"{list(allowed_model_prefixes)!r}; --model={selected_model!r} "
                    "is outside that boundary"
                )

    binding = _EXACT_RUNTIME_PIN_BINDINGS.get(args.provider)
    if binding is not None:
        argument_name, source_field, label = binding
        selection_label = (
            "exact implementation" if exact_implementation else "catalog study"
        )
        if not hasattr(args, argument_name):
            raise ValueError(
                f"{selection_label} {profile.key!r} requires runtime pin "
                f"argument {argument_name!r}"
            )
        selected_pin = getattr(args, argument_name)
        sources = profile.sources
        source_title = _EXACT_RUNTIME_SOURCE_TITLES.get(args.provider)
        if source_title is not None:
            sources = tuple(
                source for source in sources if source.title == source_title
            )
        source_pins = {
            getattr(source, source_field)
            for source in sources
            if getattr(source, source_field)
        }
        if len(source_pins) != 1:
            raise ValueError(
                f"{selection_label} {profile.key!r} must cite exactly one "
                f"{label} pin; found {sorted(source_pins)!r}"
            )
        recorded_pin = next(iter(source_pins))
        if selected_pin != recorded_pin:
            option = "--" + argument_name.replace("_", "-")
            raise ValueError(
                f"{selection_label} {profile.key!r} is pinned to {label} "
                f"{recorded_pin!r}; {option}={selected_pin!r} would change the "
                "cataloged runtime boundary"
            )

    if (
        exact_implementation
        and args.provider == "macu-upstream"
        and getattr(args, "macu_allow_dirty_checkout", False)
    ):
        raise ValueError(
            f"exact implementation {profile.key!r} requires a clean MACU checkout; "
            "--macu-allow-dirty-checkout is available only to unclassified legacy "
            "configs"
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _load_config(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    encoded = path.read_bytes()
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"config SHA-256 changed: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    raw = json.loads(encoded.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a JSON object")
    return raw


def _json_object_argument(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return parsed


def _build_harnesses(config: Mapping[str, Any]) -> list[Harness]:
    raw_specs = config.get("harnesses")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("config requires a non-empty harnesses list")
    harnesses: list[Harness] = []
    for raw in raw_specs:
        name: Any
        options: Any
        if isinstance(raw, str):
            name, options = raw, {}
        elif isinstance(raw, dict):
            name = raw.get("name")
            options = raw.get("options", {})
        else:
            raise ValueError("harness specs must be strings or objects")
        if not isinstance(name, str) or name not in HARNESS_TYPES:
            raise ValueError(f"unknown harness {name!r}")
        if not isinstance(options, dict):
            raise ValueError(f"options for {name!r} must be an object")
        try:
            harnesses.append(HARNESS_TYPES[name](**options))
        except TypeError as exc:
            raise ValueError(f"invalid options for harness {name!r}: {exc}") from exc
    validate_harness_variants(harnesses)
    return harnesses


def _build_limits(config: Mapping[str, Any]) -> BudgetLimits:
    raw = config.get("limits", {})
    if not isinstance(raw, dict):
        raise ValueError("config limits must be an object")
    try:
        return BudgetLimits(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid budget limits: {exc}") from exc


def _matrix_controls(config: Mapping[str, Any]) -> tuple[int, int, bool, Any]:
    repeats = config.get("repeats", 1)
    random_seed = config.get("random_seed", 0)
    capture_content = config.get("capture_content", False)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")
    if not isinstance(capture_content, bool):
        raise ValueError("capture_content must be a boolean")
    matrix_cap = config.get("matrix_max_cost_usd")
    if matrix_cap is not None and (
        not isinstance(matrix_cap, (int, float))
        or isinstance(matrix_cap, bool)
        or not math.isfinite(matrix_cap)
        or matrix_cap < 0
    ):
        raise ValueError("matrix_max_cost_usd must be finite and non-negative or null")
    return (
        repeats,
        random_seed,
        capture_content,
        matrix_cap,
    )


def _pricing(args: argparse.Namespace) -> TokenPricing | None:
    if args.input_price is None and args.output_price is None:
        if args.cache_read_price is not None or args.cache_write_price is not None:
            raise ValueError("cache prices require --input-price and --output-price")
        return None
    if args.input_price is None or args.output_price is None:
        raise ValueError("both --input-price and --output-price are required together")
    return TokenPricing(
        args.input_price,
        args.output_price,
        cache_read_per_million_usd=args.cache_read_price,
        cache_write_per_million_usd=args.cache_write_price,
    )


def _build_backend(args: argparse.Namespace, config: Mapping[str, Any]) -> Any:
    extra_body = config.get("provider_extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("provider_extra_body must be an object")
    if args.provider == "openai-responses":
        if not args.model:
            raise ValueError("--model is required for OpenAI Responses")
        return OpenAIResponsesBackend(
            model=args.model,
            api_key_env=args.api_key_env or "OPENAI_API_KEY",
            base_url=args.base_url or "https://api.openai.com/v1",
            pricing=_pricing(args),
            extra_body=extra_body,
        )
    if args.provider == "openai-compatible-responses":
        if not args.model:
            raise ValueError("--model is required for compatible Responses")
        if not args.base_url:
            raise ValueError("--base-url is required for compatible Responses")
        return OpenAIResponsesBackend(
            model=args.model,
            api_key_env=args.api_key_env or "MODEL_API_KEY",
            base_url=args.base_url,
            pricing=_pricing(args),
            extra_body=extra_body,
            tool_family="generic",
            provider_label="openai-compatible-responses",
        )
    if args.provider == "openai-compatible-chat":
        if not args.model:
            raise ValueError("--model is required for compatible Chat Completions")
        if not args.base_url:
            raise ValueError("--base-url is required for compatible Chat Completions")
        return OpenAICompatibleChatBackend(
            model=args.model,
            api_key_env=args.api_key_env or "MODEL_API_KEY",
            base_url=args.base_url,
            pricing=_pricing(args),
            extra_body=extra_body,
        )
    if args.provider == "anthropic-messages":
        if not args.model:
            raise ValueError("--model is required for Anthropic Messages")
        return AnthropicMessagesBackend(
            model=args.model,
            api_key_env=args.api_key_env or "ANTHROPIC_API_KEY",
            base_url=args.base_url or "https://api.anthropic.com/v1",
            pricing=_pricing(args),
            extra_body=extra_body,
        )
    if args.provider == "anthropic-managed-agents":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Anthropic Managed Agents"
            )
        if args.model:
            raise ValueError(
                "--model is not used by Anthropic Managed Agents; pin the model "
                "on the referenced agent version"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Anthropic Managed Agents reports authoritative session list_cost; "
                "remove token-price flags"
            )
        if not args.managed_agent_id:
            raise ValueError(
                "--managed-agent-id is required for Anthropic Managed Agents"
            )
        if args.managed_agent_version is None:
            raise ValueError(
                "--managed-agent-version is required for Anthropic Managed Agents"
            )
        if not args.managed_environment_id:
            raise ValueError(
                "--managed-environment-id is required for Anthropic Managed Agents"
            )
        return AnthropicManagedAgentsBackend(
            agent_id=args.managed_agent_id,
            agent_version=args.managed_agent_version,
            environment_id=args.managed_environment_id,
            api_key_env=args.api_key_env or "ANTHROPIC_API_KEY",
            base_url=args.base_url or "https://api.anthropic.com/v1",
            beta_version=args.managed_beta_version,
            timeout_seconds=args.managed_timeout_seconds,
            poll_interval_seconds=args.managed_poll_interval_seconds,
            budget_cents=args.managed_budget_cents,
            resources=args.managed_resource_json,
            vault_ids=args.managed_vault_id,
            cleanup=args.managed_cleanup,
        )
    if args.provider == "prime-agent":
        if args.prime_agent_cwd is None:
            raise ValueError(
                "--prime-agent-cwd is required; use an explicit disposable worktree"
            )
        if args.prime_agent_expected_version != "0.7.1":
            raise ValueError("the audited Prime Agent adapter requires version 0.7.1")
        if args.prime_agent_persist_session:
            raise ValueError(
                "--prime-agent-persist-session is unsupported: the exact adapter "
                "uses a fresh HOME and XDG_CONFIG_HOME for every backend call"
            )
        if _paths_overlap(args.prime_agent_cwd, args.output):
            raise ValueError(
                "--output and --prime-agent-cwd must be disjoint directories"
            )
        return PrimeAgentJSONBackend(
            cwd=args.prime_agent_cwd,
            executable=args.prime_agent_executable,
            provider=args.prime_agent_provider,
            model=args.model,
            no_session=True,
            pass_env=args.prime_agent_pass_env,
            expected_version=args.prime_agent_expected_version,
            expected_executable_sha256=args.prime_agent_executable_sha256,
            max_output_bytes=args.prime_agent_max_output_bytes,
            allow_sensitive_environment=(args.prime_agent_allow_sensitive_environment),
        )
    if args.provider == "prime-agent-source":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Prime Agent source"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Prime Agent source usage does not prove complete child-tree "
                "accounting; remove token-price flags"
            )
        if args.prime_source_checkout is None:
            raise ValueError("--prime-source-checkout is required")
        if args.prime_source_cwd is None:
            raise ValueError(
                "--prime-source-cwd is required; use an explicit disposable worktree"
            )
        for path, label in (
            (args.prime_source_checkout, "--prime-source-checkout"),
            (args.prime_source_cwd, "--prime-source-cwd"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        return PrimeAgentSourceBackend(
            checkout=args.prime_source_checkout,
            cwd=args.prime_source_cwd,
            node_executable=args.prime_source_node_executable,
            npm_executable=args.prime_source_npm_executable,
            provider=args.prime_source_provider,
            model=args.model,
            expected_revision=args.prime_source_expected_revision,
            expected_node_sha256=args.prime_source_node_sha256,
            expected_npm_sha256=args.prime_source_npm_sha256,
            expected_bundle_sha256=args.prime_source_bundle_sha256,
            timeout_seconds=args.prime_source_timeout_seconds,
            pass_env=args.prime_source_pass_env,
            max_output_bytes=args.prime_source_max_output_bytes,
            allow_sensitive_environment=(args.prime_source_allow_sensitive_environment),
        )
    if args.provider == "browser-use-upstream":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Browser-Use upstream"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Browser-Use reports incomplete token lower bounds; remove "
                "token-price flags"
            )
        if args.browser_use_checkout is None:
            raise ValueError("--browser-use-checkout is required")
        if not args.model:
            raise ValueError("--model is required for Browser-Use upstream")
        if _paths_overlap(args.browser_use_checkout, args.output):
            raise ValueError(
                "--output and --browser-use-checkout must be disjoint directories"
            )
        return BrowserUseUpstreamBackend(
            checkout=args.browser_use_checkout,
            provider=args.browser_use_llm_provider,
            model=args.model,
            python_executable=args.browser_use_python_executable,
            llm_kwargs=args.browser_use_llm_json,
            browser_kwargs=args.browser_use_browser_json,
            agent_kwargs=args.browser_use_agent_json,
            max_steps=args.browser_use_max_steps,
            process_timeout_seconds=args.browser_use_timeout_seconds,
            pass_env=args.browser_use_pass_env,
            expected_checkout_revision=(args.browser_use_expected_checkout_revision),
            expected_python_sha256=args.browser_use_python_sha256,
            allow_sensitive_environment=(args.browser_use_allow_sensitive_environment),
            max_input_bytes=args.browser_use_max_input_bytes,
            max_output_bytes=args.browser_use_max_output_bytes,
        )
    if args.provider == "grok-build":
        if args.grok_cwd is None:
            raise ValueError(
                "--grok-cwd is required; use an explicit disposable worktree"
            )
        if _paths_overlap(args.grok_cwd, args.output):
            raise ValueError("--output and --grok-cwd must be disjoint directories")
        if not args.model:
            raise ValueError("--model is required for Grok Build")
        if (
            args.grok_enable_terminal
            and args.grok_pass_env
            and not args.grok_allow_sensitive_environment
        ):
            raise ValueError(
                "--grok-enable-terminal with passed environment values requires "
                "--grok-allow-sensitive-environment and an outer sandbox"
            )
        return GrokBuildJSONBackend(
            cwd=args.grok_cwd,
            executable=args.grok_executable,
            model=args.model,
            sandbox=args.grok_sandbox,
            permission_mode=args.grok_permission_mode,
            max_turns=args.grok_max_turns,
            pass_env=args.grok_pass_env,
            allow_rules=args.grok_allow,
            deny_rules=args.grok_deny,
            disallowed_tools=(
                () if args.grok_enable_terminal else ("run_terminal_cmd",)
            ),
            expected_version=args.grok_expected_version,
            expected_executable_sha256=args.grok_executable_sha256,
            max_output_bytes=args.grok_max_output_bytes,
        )
    if args.provider == "grok-build-source":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Grok Build source"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Grok Build source reports incomplete lower-bound usage; remove "
                "token-price flags"
            )
        if args.grok_source_checkout is None:
            raise ValueError("--grok-source-checkout is required")
        if args.grok_source_workspace is None:
            raise ValueError(
                "--grok-source-workspace is required; use a clean seed workspace"
            )
        if not args.model:
            raise ValueError("--model is required for Grok Build source")
        for path, label in (
            (args.grok_source_checkout, "--grok-source-checkout"),
            (args.grok_source_workspace, "--grok-source-workspace"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        return GrokBuildSourceBackend(
            checkout=args.grok_source_checkout,
            workspace=args.grok_source_workspace,
            model=args.model,
            cargo_executable=args.grok_source_cargo_executable,
            rustc_executable=args.grok_source_rustc_executable,
            git_executable=args.grok_source_git_executable,
            sandbox=args.grok_source_sandbox,
            permission_mode=args.grok_source_permission_mode,
            max_turns=args.grok_source_max_turns,
            timeout_seconds=args.grok_source_timeout_seconds,
            build_timeout_seconds=args.grok_source_build_timeout_seconds,
            pass_env=args.grok_source_pass_env,
            allow_sensitive_environment=(args.grok_source_allow_sensitive_environment),
            expected_checkout_revision=args.grok_source_expected_revision,
            expected_executable_sha256=args.grok_source_executable_sha256,
            expected_cargo_sha256=args.grok_source_cargo_sha256,
            expected_rustc_sha256=args.grok_source_rustc_sha256,
            expected_git_sha256=args.grok_source_git_sha256,
            max_output_bytes=args.grok_source_max_output_bytes,
        )
    if args.provider == "macu-upstream":
        if extra_body:
            raise ValueError("provider_extra_body is not supported by MACU upstream")
        if args.model:
            raise ValueError(
                "--model is not used by MACU upstream; use --macu-manager-model "
                "and --macu-cua-arg=--model/--macu-cua-arg=MODEL"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "MACU imports whole-tree cost from summary.json; remove token-price flags"
            )
        if args.macu_checkout is None:
            raise ValueError("--macu-checkout is required")
        if args.macu_result_dir is None:
            raise ValueError("--macu-result-dir is required")
        if args.macu_osworld_root is None:
            raise ValueError("--macu-osworld-root is required")
        for path, label in (
            (args.macu_checkout, "--macu-checkout"),
            (args.macu_osworld_root, "--macu-osworld-root"),
            (args.macu_result_dir, "--macu-result-dir"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        return MACUUpstreamBackend(
            checkout=args.macu_checkout,
            result_dir=args.macu_result_dir,
            osworld_root=args.macu_osworld_root,
            manager_provider=args.macu_manager_provider,
            manager_model=args.macu_manager_model,
            cua_provider=args.macu_cua_provider,
            python_executable=args.macu_python_executable,
            max_parallelism=args.macu_max_parallelism,
            max_replans=args.macu_max_replans,
            max_task_timeout_seconds=args.macu_max_task_timeout_seconds,
            timeout_seconds=args.macu_timeout_seconds,
            cua_args=args.macu_cua_arg,
            pass_env=args.macu_pass_env,
            expected_checkout_revision=args.macu_expected_checkout_revision,
            expected_python_sha256=args.macu_python_sha256,
            allow_dirty_checkout=args.macu_allow_dirty_checkout,
            allow_sensitive_environment=args.macu_allow_sensitive_environment,
            max_output_bytes=args.macu_max_output_bytes,
            max_artifact_bytes=args.macu_max_artifact_bytes,
        )
    if args.provider == "rlm-upstream":
        if extra_body:
            raise ValueError("provider_extra_body is not supported by RLM upstream")
        if args.model:
            raise ValueError("--model is not used by RLM upstream; use --rlm-model")
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "RLM v0.1.3 usage is a recursive lower bound; remove token-price flags"
            )
        if args.rlm_checkout is None:
            raise ValueError("--rlm-checkout is required")
        if not args.rlm_model:
            raise ValueError("--rlm-model is required")
        if _paths_overlap(args.rlm_checkout, args.output):
            raise ValueError("--output and --rlm-checkout must be disjoint directories")
        return RLMUpstreamBackend(
            checkout=args.rlm_checkout,
            provider=args.rlm_provider,
            model=args.rlm_model,
            python_executable=args.rlm_python_executable,
            environment=args.rlm_environment,
            backend_kwargs=args.rlm_backend_json,
            environment_kwargs=args.rlm_environment_json,
            max_depth=args.rlm_max_depth,
            max_iterations=args.rlm_max_iterations,
            max_budget_usd=args.rlm_max_budget_usd,
            max_timeout_seconds=args.rlm_max_timeout_seconds,
            max_tokens=args.rlm_max_tokens,
            max_errors=args.rlm_max_errors,
            max_concurrent_subcalls=args.rlm_max_concurrent_subcalls,
            process_timeout_seconds=args.rlm_process_timeout_seconds,
            pass_env=args.rlm_pass_env,
            expected_checkout_revision=args.rlm_expected_checkout_revision,
            expected_python_sha256=args.rlm_python_sha256,
            allow_sensitive_environment=args.rlm_allow_sensitive_environment,
            max_input_bytes=args.rlm_max_input_bytes,
            max_output_bytes=args.rlm_max_output_bytes,
        )
    if args.provider == "kimi-code-upstream":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Kimi Code upstream"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Kimi stream-json omits whole-tree usage; remove token-price flags"
            )
        if args.kimi_checkout is None:
            raise ValueError("--kimi-checkout is required")
        if args.kimi_cwd is None:
            raise ValueError(
                "--kimi-cwd is required; use an explicit disposable worktree"
            )
        if not args.model:
            raise ValueError("--model is required for Kimi Code upstream")
        for path, label in (
            (args.kimi_checkout, "--kimi-checkout"),
            (args.kimi_cwd, "--kimi-cwd"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        return KimiCodeUpstreamBackend(
            checkout=args.kimi_checkout,
            cwd=args.kimi_cwd,
            model=args.model,
            api_key_env=args.api_key_env or "KIMI_API_KEY",
            provider_type=args.kimi_provider_type,
            base_url=args.base_url,
            expected_revision=args.kimi_expected_checkout_revision,
            node_executable=args.kimi_node_executable,
            tsx_executable=args.kimi_tsx_executable,
            expected_node_sha256=args.kimi_node_sha256,
            expected_tsx_sha256=args.kimi_tsx_sha256,
            timeout_seconds=args.kimi_timeout_seconds,
            max_output_bytes=args.kimi_max_output_bytes,
            max_swarm_concurrency=args.kimi_max_swarm_concurrency,
            max_steps_per_turn=args.kimi_max_steps_per_turn,
            subagent_timeout_seconds=args.kimi_subagent_timeout_seconds,
            pass_env=args.kimi_pass_env,
            allow_sensitive_environment=(args.kimi_allow_sensitive_environment),
            allow_insecure_base_url=args.kimi_allow_insecure_base_url,
        )
    if args.provider == "codex-source":
        if extra_body:
            raise ValueError("provider_extra_body is not supported by Codex source")
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Codex source does not prove complete child-tree accounting; "
                "remove token-price flags"
            )
        if args.codex_source_checkout is None:
            raise ValueError("--codex-source-checkout is required")
        if args.codex_source_workspace is None:
            raise ValueError("--codex-source-workspace is required")
        if not args.model:
            raise ValueError("--model is required for Codex source")
        for path, label in (
            (args.codex_source_checkout, "--codex-source-checkout"),
            (args.codex_source_workspace, "--codex-source-workspace"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        return CodexSourceBackend(
            checkout=args.codex_source_checkout,
            workspace=args.codex_source_workspace,
            model=args.model,
            reasoning_effort=args.codex_source_reasoning_effort,
            api_key_env=args.api_key_env or "OPENAI_API_KEY",
            auth_target_env=args.codex_source_auth_target_env,
            cargo_executable=args.codex_source_cargo_executable,
            rustc_executable=args.codex_source_rustc_executable,
            git_executable=args.codex_source_git_executable,
            multi_agent_version=args.codex_source_multi_agent_version,
            max_subagents=args.codex_source_max_subagents,
            max_depth=args.codex_source_max_depth,
            max_wait_seconds=args.codex_source_max_wait_seconds,
            timeout_seconds=args.codex_source_timeout_seconds,
            build_timeout_seconds=args.codex_source_build_timeout_seconds,
            max_output_bytes=args.codex_source_max_output_bytes,
            max_prompt_bytes=args.codex_source_max_prompt_bytes,
            max_patch_bytes=args.codex_source_max_patch_bytes,
            pass_env=args.codex_source_pass_env,
            allow_sensitive_environment=(args.codex_source_allow_sensitive_environment),
            expected_revision=args.codex_source_expected_revision,
            expected_executable_sha256=args.codex_source_executable_sha256,
            expected_cargo_sha256=args.codex_source_cargo_sha256,
            expected_rustc_sha256=args.codex_source_rustc_sha256,
            expected_git_sha256=args.codex_source_git_sha256,
        )
    if args.provider == "claude-code-agent-teams":
        if extra_body:
            raise ValueError(
                "provider_extra_body is not supported by Claude Code Agent Teams"
            )
        if any(
            value is not None
            for value in (
                args.input_price,
                args.output_price,
                args.cache_read_price,
                args.cache_write_price,
            )
        ):
            raise ValueError(
                "Claude Code does not document authoritative whole-team usage; "
                "remove token-price flags"
            )
        if args.claude_code_distribution_root is None:
            raise ValueError("--claude-code-distribution-root is required")
        if args.claude_code_workspace is None:
            raise ValueError("--claude-code-workspace is required")
        if args.claude_code_max_budget_usd is None:
            raise ValueError("--claude-code-max-budget-usd is required")
        if not args.model:
            raise ValueError("--model is required for Claude Code Agent Teams")
        for path, label in (
            (
                args.claude_code_distribution_root,
                "--claude-code-distribution-root",
            ),
            (args.claude_code_workspace, "--claude-code-workspace"),
        ):
            if _paths_overlap(path, args.output):
                raise ValueError(f"--output and {label} must be disjoint directories")
        backend = ClaudeCodeAgentTeamsDistributionBackend(
            distribution_root=args.claude_code_distribution_root,
            workspace=args.claude_code_workspace,
            model=args.model,
            max_budget_usd=args.claude_code_max_budget_usd,
            api_key_env=args.api_key_env or "ANTHROPIC_API_KEY",
            expected_version=args.claude_code_expected_version,
            expected_executable_sha256=args.claude_code_executable_sha256,
            native_package_name=args.claude_code_native_package,
            git_executable=args.claude_code_git_executable,
            expected_git_sha256=args.claude_code_git_sha256,
            max_turns=args.claude_code_max_turns,
            permission_mode=args.claude_code_permission_mode,
            effort=args.claude_code_effort,
            timeout_seconds=args.claude_code_timeout_seconds,
            tool_timeout_seconds=args.claude_code_tool_timeout_seconds,
            max_output_bytes=args.claude_code_max_output_bytes,
            max_patch_bytes=args.claude_code_max_patch_bytes,
            pass_env=args.claude_code_pass_env,
            allow_sensitive_environment=(args.claude_code_allow_sensitive_environment),
            require_team_evidence=True,
        )
        if not backend.official_distribution_verified:
            raise ValueError(
                "this released Claude Code adapter requires a bundled audited "
                "official platform digest; caller-pinned unknown platforms are "
                "not release implementations"
            )
        return backend
    if args.provider == "xai-responses":
        if not args.model:
            raise ValueError("--model is required for xAI Responses")
        return XAIResponsesBackend(
            model=args.model,
            api_key_env=args.api_key_env or "XAI_API_KEY",
            base_url=args.base_url or "https://api.x.ai/v1",
            pricing=_pricing(args),
            extra_body=extra_body,
        )
    raise ValueError(f"unknown provider {args.provider!r}")


def _validate_compatibility(
    harnesses: Sequence[Harness],
    provider: str,
    task_path: Path,
    tasks: Sequence[Any],
    repeats: int = 1,
    environment_enabled: bool = False,
) -> list[str]:
    warnings: list[str] = []
    names = {harness.name for harness in harnesses}
    if "openai_hosted_multi_agent" in names and provider != "openai-responses":
        raise ValueError(
            "openai_hosted_multi_agent requires --provider openai-responses"
        )
    if "prime_agent" in names and provider != "prime-agent":
        raise ValueError("prime_agent harness requires --provider prime-agent")
    if "prime_agent_source" in names and provider != "prime-agent-source":
        raise ValueError(
            "prime_agent_source harness requires --provider prime-agent-source"
        )
    if "grok_build" in names and provider != "grok-build":
        raise ValueError("grok_build harness requires --provider grok-build")
    if "grok_build_source" in names and provider != "grok-build-source":
        raise ValueError(
            "grok_build_source harness requires --provider grok-build-source"
        )
    if "xai_hosted_multi_agent" in names and provider != "xai-responses":
        raise ValueError("xai_hosted_multi_agent requires --provider xai-responses")
    if "anthropic_managed_agents" in names and provider != "anthropic-managed-agents":
        raise ValueError(
            "anthropic_managed_agents requires --provider anthropic-managed-agents"
        )
    if "macu_upstream" in names and provider != "macu-upstream":
        raise ValueError("macu_upstream harness requires --provider macu-upstream")
    if "rlm_upstream" in names and provider != "rlm-upstream":
        raise ValueError("rlm_upstream harness requires --provider rlm-upstream")
    if "kimi_code_upstream" in names and provider != "kimi-code-upstream":
        raise ValueError(
            "kimi_code_upstream harness requires --provider kimi-code-upstream"
        )
    if "browser_use_upstream" in names and provider != "browser-use-upstream":
        raise ValueError(
            "browser_use_upstream harness requires --provider browser-use-upstream"
        )
    if "codex_source" in names and provider != "codex-source":
        raise ValueError("codex_source harness requires --provider codex-source")
    if (
        "claude_code_agent_teams_distribution" in names
        and provider != "claude-code-agent-teams"
    ):
        raise ValueError(
            "claude_code_agent_teams_distribution harness requires --provider "
            "claude-code-agent-teams"
        )
    if "flat_parallel" in names:
        missing = [
            task.task_id
            for task in tasks
            if not isinstance(task.metadata.get("parallel_tasks"), list)
            or not task.metadata.get("parallel_tasks")
        ]
        if missing:
            raise ValueError(
                "flat_parallel requires non-empty parallel_tasks for every task; "
                f"missing on {missing}"
            )
        warnings.append("flat_parallel is distinct-task throughput, not best-of-N")
    external_context_harnesses = names.intersection(
        {"external_context_json_search", "rlm_repl"}
    )
    if external_context_harnesses:
        missing = [task.task_id for task in tasks if not task.context]
        if missing:
            raise ValueError(
                f"{sorted(external_context_harnesses)} require non-empty context for "
                "every task; "
                f"missing on {missing}"
            )
    if "platoon_recursive_inference" in names:
        warnings.append(
            "platoon_recursive_inference is inference-only and does not reproduce RAO policy training"
        )
    if names.intersection({"rlm_repl", "platoon_recursive_inference"}):
        warnings.append(
            "restricted Python is an in-process capability layer, not an OS sandbox; "
            "use an outer process/container sandbox for adversarial or paid runs"
        )
    if provider == "prime-agent":
        warnings.append(
            "Prime Agent may execute commands and modify files in --prime-agent-cwd; use a disposable worktree"
        )
        incompatible = sorted(names - {"prime_agent"})
        if incompatible:
            raise ValueError(
                "--provider prime-agent may only run the prime_agent harness; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "prime-agent-source":
        warnings.append(
            "Prime Agent source study executes caller-provided bundle bytes and "
            "source-owned tools in --prime-source-cwd; use a disposable outer sandbox"
        )
        warnings.append(
            "The source revision and lockfile are recorded, but no adapter-owned "
            "build or authoritative bundle digest proves source-runtime parity; "
            "Node, npm, and bundle SHA pins only stabilize caller bytes"
        )
        incompatible = sorted(names - {"prime_agent_source"})
        if incompatible:
            raise ValueError(
                "--provider prime-agent-source may only run the "
                "prime_agent_source harness; incompatible harnesses: "
                f"{incompatible}"
            )
    if provider == "grok-build":
        warnings.append(
            "Grok Build can execute tools in --grok-cwd; strict sandboxing still requires a disposable worktree"
        )
        warnings.append(
            "Grok usage includes completed subagents but can exclude compaction, side-model, and unfinished nested calls"
        )
        incompatible = sorted(names - {"grok_build"})
        if incompatible:
            raise ValueError(
                "--provider grok-build may only run the grok_build harness; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "grok-build-source":
        warnings.append(
            "Grok Build source compiles a verified private git-archive export, then "
            "runs its native tools and subagents in a disposable workspace copy and "
            "exports a bounded binary patch"
        )
        warnings.append(
            "Terminal JSON usage remains a lower bound; reproducible binary identity "
            "also requires explicit Git, Cargo, rustc, and executable SHA pins"
        )
        incompatible = sorted(names - {"grok_build_source"})
        if incompatible:
            raise ValueError(
                "--provider grok-build-source may only run the grok_build_source "
                f"harness; incompatible harnesses: {incompatible}"
            )
    if provider in {
        "prime-agent",
        "prime-agent-source",
        "grok-build",
        "grok-build-source",
        "macu-upstream",
        "rlm-upstream",
        "kimi-code-upstream",
        "browser-use-upstream",
        "codex-source",
        "claude-code-agent-teams",
    }:
        if environment_enabled:
            raise ValueError(
                f"--provider {provider} owns its tools/environment; remove config.environment"
            )
        planned_trials = len(tasks) * len(harnesses) * repeats
        if planned_trials != 1:
            raise ValueError(
                f"--provider {provider} requires exactly one trial per invocation "
                "so filesystem mutations cannot contaminate another trial; provide "
                "one task, one harness variant, and repeats=1 in a fresh workspace"
            )
    if provider == "macu-upstream":
        warnings.append(
            "MACU upstream creates VM-backed workers and persistent result artifacts; "
            "use a pinned checkout, disposable VM pool, and isolated network boundary"
        )
        incompatible = sorted(names - {"macu_upstream"})
        if incompatible:
            raise ValueError(
                "--provider macu-upstream may only run the macu_upstream harness; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "rlm-upstream":
        warnings.append(
            "Scaffold Lab defaults the RLM adapter to the upstream Docker REPL "
            "(the library default is local); non-Docker environments need an "
            "independently enforced outer sandbox"
        )
        warnings.append(
            "rlms v0.1.3 root UsageSummary omits recursive child handlers, so calls, "
            "tokens, and reported cost are lower bounds"
        )
        incompatible = sorted(names - {"rlm_upstream"})
        if incompatible:
            raise ValueError(
                "--provider rlm-upstream may only run the rlm_upstream harness; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "kimi-code-upstream":
        warnings.append(
            "Kimi Code executes source-owned shell/file/subagent tools in --kimi-cwd; "
            "use a disposable outer sandbox and a scoped model credential"
        )
        warnings.append(
            "Kimi stream-json omits whole-tree tokens and cost, so accounting is "
            "unknown rather than release-comparable"
        )
        warnings.append(
            "Kimi's official non-interactive interface carries the task in the "
            "--prompt process argument; local process inspection can observe it"
        )
        incompatible = sorted(names - {"kimi_code_upstream"})
        if incompatible:
            raise ValueError(
                "--provider kimi-code-upstream may only run the "
                "kimi_code_upstream harness; incompatible harnesses: "
                f"{incompatible}"
            )
    if provider == "browser-use-upstream":
        warnings.append(
            "Browser-Use owns its Agent, Browser, provider wrapper, prompts, and "
            "actions; use a scoped credential and isolated browser environment"
        )
        warnings.append(
            "Browser-Use TokenCost entries are incomplete lower bounds and cannot "
            "establish release-comparable cost"
        )
        incompatible = sorted(names - {"browser_use_upstream"})
        if incompatible:
            raise ValueError(
                "--provider browser-use-upstream may only run the "
                "browser_use_upstream harness; incompatible harnesses: "
                f"{incompatible}"
            )
    if provider == "codex-source":
        warnings.append(
            "Codex source builds a private archive of the pinned revision and runs "
            "native tools in a disposable Git baseline; use a scoped credential "
            "and isolated outer runner"
        )
        warnings.append(
            "Source and protocol identity do not pin the remote model, service "
            "routing, managed/cloud policy, or complete child-tree accounting"
        )
        incompatible = sorted(names - {"codex_source"})
        if incompatible:
            raise ValueError(
                "--provider codex-source may only run the codex_source harness; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "claude-code-agent-teams":
        warnings.append(
            "Claude Code Agent Teams executes an official binary distribution, "
            "not public implementation source, with native tools and a shared model "
            "credential in a disposable workspace"
        )
        warnings.append(
            "Endpoint-managed files are rejected, but server-managed policy, model "
            "snapshot, and authoritative whole-team accounting remain unobservable"
        )
        incompatible = sorted(names - {"claude_code_agent_teams_distribution"})
        if incompatible:
            raise ValueError(
                "--provider claude-code-agent-teams may only run the "
                "claude_code_agent_teams_distribution harness; incompatible "
                f"harnesses: {incompatible}"
            )
    if provider == "xai-responses":
        if environment_enabled:
            raise ValueError(
                "xAI hosted multi-agent client-tool continuation is not implemented"
            )
        warnings.append(
            "xAI exposes leader plaintext and aggregate usage; its hosted scheduler "
            "is closed and encrypted-state continuation is not implemented"
        )
        incompatible = sorted(names - {"xai_hosted_multi_agent"})
        if incompatible:
            raise ValueError(
                "--provider xai-responses may only run xai_hosted_multi_agent; "
                f"incompatible harnesses: {incompatible}"
            )
    if provider == "anthropic-managed-agents":
        if environment_enabled:
            raise ValueError(
                "Anthropic Managed Agents owns its remote environment; remove "
                "config.environment"
            )
        incompatible = sorted(names - {"anthropic_managed_agents"})
        if incompatible:
            raise ValueError(
                "--provider anthropic-managed-agents may only run the "
                "anthropic_managed_agents harness; incompatible harnesses: "
                f"{incompatible}"
            )
        warnings.append(
            "Anthropic Managed Agents uses the pinned remote agent topology, tools, "
            "and environment; local harness topology options do not apply"
        )
        warnings.append(
            "Managed custom-tool results and permission confirmations are not "
            "implemented; configure server-owned tools with always-allow policy"
        )
    if task_path.name.startswith("smoke"):
        warnings.append(
            "smoke tasks validate plumbing; they cannot select a release winner"
        )
    return warnings


async def _run(args: argparse.Namespace) -> int:
    config, application, application_warnings = resolve_application_config(
        _load_config(
            args.config,
            expected_sha256=getattr(args, "expected_config_sha256", None),
        ),
        provider=args.provider,
    )
    _enforce_expected_application_key(
        getattr(args, "expected_application_key", None), application
    )
    _enforce_exact_application_identity(args, application)
    tasks = load_tasks(args.tasks)
    harnesses = _build_harnesses(config)
    limits = _build_limits(config)
    repeats, random_seed, capture_content, configured_matrix_cap = _matrix_controls(
        config
    )
    environment_factory = build_environment_factory(config.get("environment"))
    warnings = application_warnings + _validate_compatibility(
        harnesses,
        args.provider,
        args.tasks,
        tasks,
        repeats,
        environment_enabled=environment_factory is not None,
    )
    if args.provider == "grok-build" and args.grok_enable_terminal:
        warnings.append(
            "Grok terminal access is enabled; use an outer sandbox that prevents "
            "tool children from inheriting or exfiltrating credentials"
        )
    backend = _build_backend(args, config)
    if isinstance(
        backend,
        (GrokBuildJSONBackend, KimiCodeUpstreamBackend, PrimeAgentJSONBackend),
    ):
        await backend.verify_version()
    run_metadata: dict[str, Any] = {
        "config_path": str(args.config.resolve()),
        "task_path": str(args.tasks.resolve()),
        "config_metadata": config.get("metadata"),
    }
    if application is not None:
        run_metadata["application"] = application.as_dict()
    runner = MatrixRunner(
        backend=backend,
        limits=limits,
        output_dir=args.output,
        repeats=repeats,
        random_seed=random_seed,
        capture_content=args.capture_content or capture_content,
        matrix_max_cost_usd=(
            args.matrix_max_cost_usd
            if args.matrix_max_cost_usd is not None
            else configured_matrix_cap
        ),
        overwrite=args.overwrite,
        run_metadata=run_metadata,
        environment_factory=environment_factory,
    )
    _, summary = await runner.run(tasks, harnesses)
    summary = {**summary, "warnings": warnings}
    if application is not None:
        summary["application"] = application.as_dict()
    _atomic_write_text(
        args.output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    clean = (
        summary["matrix_completed"]
        and summary["total_errors"] == 0
        and summary["all_usage_complete"]
        and summary["all_cost_known"]
    )
    return 0 if clean else 1


def _validate(args: argparse.Namespace) -> int:
    config, application, application_warnings = resolve_application_config(
        _load_config(
            args.config,
            expected_sha256=getattr(args, "expected_config_sha256", None),
        ),
        provider=args.provider,
    )
    _enforce_expected_application_key(
        getattr(args, "expected_application_key", None), application
    )
    tasks = load_tasks(args.tasks)
    harnesses = _build_harnesses(config)
    limits = _build_limits(config)
    repeats, random_seed, capture_content, matrix_cap = _matrix_controls(config)
    environment_factory = build_environment_factory(config.get("environment"))
    warnings = application_warnings + _validate_compatibility(
        harnesses,
        args.provider,
        args.tasks,
        tasks,
        repeats,
        environment_enabled=environment_factory is not None,
    )
    result = {
        "tasks": len(tasks),
        "harnesses": [harness.name for harness in harnesses],
        "harness_variants": [
            {"name": harness.name, "options": dict(harness.__dict__)}
            for harness in harnesses
        ],
        "limits": limits.__dict__,
        "limits_scope": "per_trial",
        "repeats": repeats,
        "random_seed": random_seed,
        "capture_content": capture_content,
        "matrix_max_cost_usd": matrix_cap,
        "environment": (
            dict(environment_factory.provenance())
            if environment_factory is not None
            else None
        ),
        "warnings": warnings,
        "valid": True,
    }
    if application is not None:
        result["application"] = application.as_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-application-key", help=argparse.SUPPRESS)
    parser.add_argument("--expected-config-sha256", help=argparse.SUPPRESS)
    parser.add_argument(
        "--provider",
        choices=(
            "openai-responses",
            "openai-compatible-responses",
            "openai-compatible-chat",
            "anthropic-messages",
            "anthropic-managed-agents",
            "prime-agent",
            "prime-agent-source",
            "browser-use-upstream",
            "grok-build",
            "grok-build-source",
            "xai-responses",
            "macu-upstream",
            "rlm-upstream",
            "kimi-code-upstream",
            "codex-source",
            "claude-code-agent-teams",
        ),
        default="openai-responses",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scaffoldlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-harnesses")
    list_parser.set_defaults(handler=lambda args: _list_harnesses())

    applications_parser = subparsers.add_parser("list-applications")
    applications_parser.add_argument("--json", action="store_true")
    applications_parser.set_defaults(
        handler=lambda args: _list_applications(json_output=args.json)
    )

    frontier_parser = subparsers.add_parser("list-frontier-sources")
    frontier_parser.add_argument("--json", action="store_true")
    frontier_parser.set_defaults(
        handler=lambda args: _list_frontier_sources(json_output=args.json)
    )

    implementations_parser = subparsers.add_parser("list-implementations")
    implementations_parser.add_argument("--application", choices=catalog_applications())
    implementations_parser.add_argument("--json", action="store_true")
    implementations_parser.set_defaults(
        handler=lambda args: _list_implementations(
            application=args.application, json_output=args.json
        )
    )

    studies_parser = subparsers.add_parser("list-studies")
    studies_parser.add_argument("--application", choices=catalog_applications())
    studies_parser.add_argument("--json", action="store_true")
    studies_parser.set_defaults(
        handler=lambda args: _list_studies(
            application=args.application, json_output=args.json
        )
    )

    gaps_parser = subparsers.add_parser("list-gaps")
    gaps_parser.add_argument("--application", choices=catalog_applications())
    gaps_parser.add_argument("--json", action="store_true")
    gaps_parser.set_defaults(
        handler=lambda args: _list_gaps(
            application=args.application, json_output=args.json
        )
    )

    profiles_parser = subparsers.add_parser("list-profiles")
    profiles_parser.add_argument("--application", choices=catalog_applications())
    profiles_parser.add_argument("--json", action="store_true")
    profiles_parser.set_defaults(
        handler=lambda args: _list_profiles(
            application=args.application, json_output=args.json
        )
    )

    validate_parser = subparsers.add_parser("validate", allow_abbrev=False)
    _add_common_arguments(validate_parser)
    validate_parser.set_defaults(handler=_validate)

    run_parser = subparsers.add_parser("run", allow_abbrev=False)
    _add_common_arguments(run_parser)
    run_parser.add_argument("--model")
    run_parser.add_argument("--api-key-env")
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--input-price", type=float)
    run_parser.add_argument("--output-price", type=float)
    run_parser.add_argument("--cache-read-price", type=float)
    run_parser.add_argument("--cache-write-price", type=float)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--capture-content", action="store_true")
    run_parser.add_argument("--matrix-max-cost-usd", type=float)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--managed-agent-id")
    run_parser.add_argument("--managed-agent-version", type=int)
    run_parser.add_argument("--managed-environment-id")
    run_parser.add_argument(
        "--managed-beta-version", default="managed-agents-2026-04-01"
    )
    run_parser.add_argument("--managed-budget-cents", type=int)
    run_parser.add_argument(
        "--managed-resource-json",
        action="append",
        type=_json_object_argument,
        default=[],
        metavar="JSON",
    )
    run_parser.add_argument(
        "--managed-vault-id", action="append", default=[], metavar="ID"
    )
    run_parser.add_argument(
        "--managed-cleanup",
        choices=("retain", "archive", "delete"),
        default="retain",
    )
    run_parser.add_argument("--managed-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument("--managed-poll-interval-seconds", type=float, default=1.0)
    run_parser.add_argument("--prime-agent-cwd", type=Path)
    run_parser.add_argument("--prime-agent-executable", default="prime-agent")
    run_parser.add_argument("--prime-agent-provider")
    run_parser.add_argument(
        "--prime-agent-pass-env",
        action="append",
        default=[],
        metavar="NAME",
    )
    run_parser.add_argument(
        "--prime-agent-persist-session",
        action="store_true",
        help=(
            "unsupported compatibility flag; the adapter rejects cross-call session "
            "persistence because each call receives fresh state"
        ),
    )
    run_parser.add_argument("--prime-agent-expected-version", default="0.7.1")
    run_parser.add_argument("--prime-agent-executable-sha256")
    run_parser.add_argument(
        "--prime-agent-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--prime-agent-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Prime Agent code can inspect passed environment values"
        ),
    )
    run_parser.add_argument("--prime-source-checkout", type=Path)
    run_parser.add_argument("--prime-source-cwd", type=Path)
    run_parser.add_argument("--prime-source-node-executable", default="node")
    run_parser.add_argument("--prime-source-npm-executable", default="npm")
    run_parser.add_argument("--prime-source-provider")
    run_parser.add_argument(
        "--prime-source-expected-revision",
        default=PRIME_AGENT_SOURCE_REVISION,
    )
    run_parser.add_argument("--prime-source-node-sha256")
    run_parser.add_argument("--prime-source-npm-sha256")
    run_parser.add_argument("--prime-source-bundle-sha256")
    run_parser.add_argument(
        "--prime-source-timeout-seconds", type=float, default=1800.0
    )
    run_parser.add_argument(
        "--prime-source-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--prime-source-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--prime-source-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Prime Agent source-owned tools and children can "
            "inspect passed environment values"
        ),
    )
    run_parser.add_argument("--browser-use-checkout", type=Path)
    run_parser.add_argument(
        "--browser-use-llm-provider",
        choices=(
            "anthropic",
            "azure-openai",
            "browser-use",
            "google",
            "groq",
            "litellm",
            "mistral",
            "oci-raw",
            "ollama",
            "openai",
            "vercel",
        ),
        default="openai",
    )
    run_parser.add_argument("--browser-use-python-executable", default="python3")
    run_parser.add_argument("--browser-use-python-sha256")
    run_parser.add_argument(
        "--browser-use-expected-checkout-revision",
        default="f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc",
    )
    run_parser.add_argument(
        "--browser-use-llm-json",
        type=_json_object_argument,
        default={},
        metavar="JSON",
    )
    run_parser.add_argument(
        "--browser-use-browser-json",
        type=_json_object_argument,
        default={},
        metavar="JSON",
    )
    run_parser.add_argument(
        "--browser-use-agent-json",
        type=_json_object_argument,
        default={},
        metavar="JSON",
    )
    run_parser.add_argument("--browser-use-max-steps", type=int, default=100)
    run_parser.add_argument("--browser-use-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument(
        "--browser-use-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--browser-use-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Browser-Use and visited pages can inspect passed "
            "environment values"
        ),
    )
    run_parser.add_argument(
        "--browser-use-max-input-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--browser-use-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument("--grok-cwd", type=Path)
    run_parser.add_argument("--grok-executable", default="grok")
    run_parser.add_argument("--grok-sandbox", default="strict")
    run_parser.add_argument("--grok-permission-mode", default="dontAsk")
    run_parser.add_argument("--grok-max-turns", type=int, default=64)
    run_parser.add_argument(
        "--grok-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument("--grok-allow", action="append", default=[], metavar="RULE")
    run_parser.add_argument("--grok-deny", action="append", default=[], metavar="RULE")
    run_parser.add_argument("--grok-expected-version", default="1.0.0")
    run_parser.add_argument("--grok-executable-sha256")
    run_parser.add_argument(
        "--grok-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--grok-enable-terminal",
        action="store_true",
        help=(
            "allow Grok's terminal tool; requires an outer credential-isolating "
            "sandbox because tool children can inherit API-key environment variables"
        ),
    )
    run_parser.add_argument(
        "--grok-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge credential exposure when terminal tools inherit environment"
        ),
    )
    run_parser.add_argument("--grok-source-checkout", type=Path)
    run_parser.add_argument("--grok-source-workspace", type=Path)
    run_parser.add_argument("--grok-source-cargo-executable", default="cargo")
    run_parser.add_argument("--grok-source-rustc-executable")
    run_parser.add_argument("--grok-source-git-executable", default="git")
    run_parser.add_argument("--grok-source-sandbox", default="strict")
    run_parser.add_argument("--grok-source-permission-mode", default="dontAsk")
    run_parser.add_argument("--grok-source-max-turns", type=int, default=64)
    run_parser.add_argument(
        "--grok-source-expected-revision",
        default=GROK_BUILD_PUBLIC_REVISION,
    )
    run_parser.add_argument("--grok-source-executable-sha256")
    run_parser.add_argument("--grok-source-cargo-sha256")
    run_parser.add_argument("--grok-source-rustc-sha256")
    run_parser.add_argument("--grok-source-git-sha256")
    run_parser.add_argument("--grok-source-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument(
        "--grok-source-build-timeout-seconds", type=float, default=1800.0
    )
    run_parser.add_argument(
        "--grok-source-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--grok-source-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--grok-source-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Grok Build source-owned tools and children can inspect "
            "passed environment values"
        ),
    )
    run_parser.add_argument("--macu-checkout", type=Path)
    run_parser.add_argument("--macu-result-dir", type=Path)
    run_parser.add_argument("--macu-osworld-root", type=Path)
    run_parser.add_argument(
        "--macu-manager-provider",
        choices=("openai", "anthropic", "huggingface", "google"),
        default="anthropic",
    )
    run_parser.add_argument("--macu-manager-model", default="claude-opus-4-6")
    run_parser.add_argument(
        "--macu-cua-provider", choices=("qwen", "openai"), default="openai"
    )
    run_parser.add_argument("--macu-python-executable", default="python3")
    run_parser.add_argument("--macu-max-parallelism", type=int, default=4)
    run_parser.add_argument("--macu-max-replans", type=int, default=10)
    run_parser.add_argument("--macu-max-task-timeout-seconds", type=int, default=5400)
    run_parser.add_argument("--macu-timeout-seconds", type=float, default=6000.0)
    run_parser.add_argument(
        "--macu-cua-arg", action="append", default=[], metavar="ARG"
    )
    run_parser.add_argument(
        "--macu-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--macu-expected-checkout-revision",
        default="5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105",
    )
    run_parser.add_argument("--macu-python-sha256")
    run_parser.add_argument("--macu-allow-dirty-checkout", action="store_true")
    run_parser.add_argument(
        "--macu-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that MACU and its CUA children can inspect every passed "
            "environment value"
        ),
    )
    run_parser.add_argument(
        "--macu-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--macu-max-artifact-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument("--rlm-checkout", type=Path)
    run_parser.add_argument(
        "--rlm-provider",
        choices=(
            "openai",
            "portkey",
            "openrouter",
            "vercel",
            "vllm",
            "anthropic",
            "azure_openai",
            "gemini",
        ),
        default="openai",
    )
    run_parser.add_argument("--rlm-model")
    run_parser.add_argument("--rlm-python-executable", default="python3")
    run_parser.add_argument(
        "--rlm-environment",
        choices=("local", "ipython", "docker", "modal", "prime", "daytona", "e2b"),
        default="docker",
    )
    run_parser.add_argument(
        "--rlm-backend-json",
        type=_json_object_argument,
        default={},
        metavar="JSON",
    )
    run_parser.add_argument(
        "--rlm-environment-json",
        type=_json_object_argument,
        default={},
        metavar="JSON",
    )
    run_parser.add_argument("--rlm-max-depth", type=int, default=1)
    run_parser.add_argument("--rlm-max-iterations", type=int, default=30)
    run_parser.add_argument("--rlm-max-budget-usd", type=float)
    run_parser.add_argument("--rlm-max-timeout-seconds", type=float, default=1500.0)
    run_parser.add_argument("--rlm-max-tokens", type=int)
    run_parser.add_argument("--rlm-max-errors", type=int)
    run_parser.add_argument("--rlm-max-concurrent-subcalls", type=int, default=4)
    run_parser.add_argument("--rlm-process-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument(
        "--rlm-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--rlm-expected-checkout-revision",
        default="72d6940142ddfb84ee6be573dc999a37e633e671",
    )
    run_parser.add_argument("--rlm-python-sha256")
    run_parser.add_argument(
        "--rlm-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that the RLM and its REPL can inspect every passed "
            "environment value"
        ),
    )
    run_parser.add_argument("--rlm-max-input-bytes", type=int, default=64 * 1024 * 1024)
    run_parser.add_argument(
        "--rlm-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument("--kimi-checkout", type=Path)
    run_parser.add_argument("--kimi-cwd", type=Path)
    run_parser.add_argument(
        "--kimi-provider-type",
        choices=("kimi", "anthropic", "openai"),
        default="kimi",
    )
    run_parser.add_argument("--kimi-tsx-executable", type=Path)
    run_parser.add_argument("--kimi-node-executable", default="node")
    run_parser.add_argument("--kimi-node-sha256")
    run_parser.add_argument("--kimi-tsx-sha256")
    run_parser.add_argument(
        "--kimi-expected-checkout-revision",
        default="f0614c53e59f7e1e257412063b059b9eb82764cf",
    )
    run_parser.add_argument(
        "--kimi-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument("--kimi-max-swarm-concurrency", type=int, default=8)
    run_parser.add_argument("--kimi-max-steps-per-turn", type=int, default=64)
    run_parser.add_argument(
        "--kimi-subagent-timeout-seconds", type=float, default=1200.0
    )
    run_parser.add_argument("--kimi-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument(
        "--kimi-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--kimi-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Kimi source-owned shell and subagent tools inherit "
            "the scoped model credential"
        ),
    )
    run_parser.add_argument(
        "--kimi-allow-insecure-base-url",
        action="store_true",
        help=(
            "acknowledge cleartext model-credential transport to a non-loopback "
            "HTTP --base-url; HTTPS remains the default requirement"
        ),
    )
    run_parser.add_argument("--codex-source-checkout", type=Path)
    run_parser.add_argument("--codex-source-workspace", type=Path)
    run_parser.add_argument("--codex-source-cargo-executable", default="cargo")
    run_parser.add_argument("--codex-source-rustc-executable")
    run_parser.add_argument("--codex-source-git-executable", default="git")
    run_parser.add_argument(
        "--codex-source-expected-revision", default=CODEX_SOURCE_REVISION
    )
    run_parser.add_argument("--codex-source-executable-sha256")
    run_parser.add_argument("--codex-source-cargo-sha256")
    run_parser.add_argument("--codex-source-rustc-sha256")
    run_parser.add_argument("--codex-source-git-sha256")
    run_parser.add_argument("--codex-source-reasoning-effort")
    run_parser.add_argument(
        "--codex-source-auth-target-env",
        choices=("CODEX_API_KEY",),
        default="CODEX_API_KEY",
    )
    run_parser.add_argument(
        "--codex-source-multi-agent-version", choices=("v1", "v2"), default="v1"
    )
    run_parser.add_argument("--codex-source-max-subagents", type=int)
    run_parser.add_argument("--codex-source-max-depth", type=int, default=1)
    run_parser.add_argument("--codex-source-max-wait-seconds", type=float)
    run_parser.add_argument(
        "--codex-source-timeout-seconds", type=float, default=1800.0
    )
    run_parser.add_argument(
        "--codex-source-build-timeout-seconds", type=float, default=3600.0
    )
    run_parser.add_argument(
        "--codex-source-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--codex-source-max-prompt-bytes", type=int, default=1024 * 1024
    )
    run_parser.add_argument(
        "--codex-source-max-patch-bytes", type=int, default=8 * 1024 * 1024
    )
    run_parser.add_argument(
        "--codex-source-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--codex-source-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that native Codex and its subagents use the scoped model "
            "credential; shell children receive a filtered environment"
        ),
    )
    run_parser.add_argument("--claude-code-distribution-root", type=Path)
    run_parser.add_argument("--claude-code-workspace", type=Path)
    run_parser.add_argument(
        "--claude-code-expected-version", default=CLAUDE_CODE_DISTRIBUTION_VERSION
    )
    run_parser.add_argument("--claude-code-executable-sha256")
    run_parser.add_argument("--claude-code-native-package")
    run_parser.add_argument("--claude-code-git-executable", default="git")
    run_parser.add_argument("--claude-code-git-sha256")
    run_parser.add_argument("--claude-code-max-budget-usd", type=float)
    run_parser.add_argument("--claude-code-max-turns", type=int, default=64)
    run_parser.add_argument(
        "--claude-code-permission-mode",
        choices=(
            "acceptEdits",
            "auto",
            "bypassPermissions",
            "manual",
            "dontAsk",
            "plan",
        ),
        default="dontAsk",
    )
    run_parser.add_argument(
        "--claude-code-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
    )
    run_parser.add_argument("--claude-code-timeout-seconds", type=float, default=1800.0)
    run_parser.add_argument(
        "--claude-code-tool-timeout-seconds", type=float, default=600.0
    )
    run_parser.add_argument(
        "--claude-code-max-output-bytes", type=int, default=16 * 1024 * 1024
    )
    run_parser.add_argument(
        "--claude-code-max-patch-bytes", type=int, default=8 * 1024 * 1024
    )
    run_parser.add_argument(
        "--claude-code-pass-env", action="append", default=[], metavar="NAME"
    )
    run_parser.add_argument(
        "--claude-code-allow-sensitive-environment",
        action="store_true",
        help=(
            "acknowledge that Claude Code teammates and tools inherit the scoped "
            "model credential in an outer disposable sandbox"
        ),
    )
    run_parser.set_defaults(handler=_run)
    return parser


def _list_harnesses() -> int:
    for name in HARNESS_TYPES:
        print(name)
    return 0


def _list_applications(*, json_output: bool = False) -> int:
    applications = catalog_applications()
    if json_output:
        payload = [
            {
                "name": name,
                "implementation_count": len(catalog_implementations(name)),
                "study_count": len(catalog_studies(name)),
                "gap_count": len(catalog_gaps(name)),
                "profile_count": len(catalog_profiles(name)),
                "simulation_count": sum(
                    profile.status == "simulation" for profile in catalog_studies(name)
                ),
            }
            for name in applications
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name in applications:
            print(name)
    return 0


def _list_frontier_sources(*, json_output: bool = False) -> int:
    records = catalog_frontier_labs()
    if json_output:
        print(
            json.dumps(
                [record.as_dict() for record in records],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for record in records:
            print(record.lab)
    return 0


def _list_implementations(
    *, application: str | None = None, json_output: bool = False
) -> int:
    return _print_profiles(
        catalog_implementations(application), json_output=json_output
    )


def _list_studies(*, application: str | None = None, json_output: bool = False) -> int:
    return _print_profiles(catalog_studies(application), json_output=json_output)


def _list_gaps(*, application: str | None = None, json_output: bool = False) -> int:
    return _print_profiles(catalog_gaps(application), json_output=json_output)


def _list_profiles(*, application: str | None = None, json_output: bool = False) -> int:
    return _print_profiles(catalog_profiles(application), json_output=json_output)


def _print_profiles(profiles: Sequence[Any], *, json_output: bool) -> int:
    if json_output:
        print(
            json.dumps(
                [profile.as_dict() for profile in profiles],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for profile in profiles:
            print(profile.key)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return int(result)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
