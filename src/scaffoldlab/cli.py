from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .applications import (
    list_applications as catalog_applications,
    list_implementations as catalog_implementations,
    resolve_application_config,
)
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
from .harnesses import (
    AnthropicManagedAgentsHarness,
    AsyncSubagentsHarness,
    BlockingOrchestratorHarness,
    FixedAgentTeamHarness,
    ExternalContextJSONSearchHarness,
    FlatParallelHarness,
    GrokBuildHarness,
    Harness,
    MACUHarness,
    MACUUpstreamHarness,
    OpenAIHostedMultiAgentHarness,
    ParallelBestOfNHarness,
    PlatoonRecursiveInferenceHarness,
    PrimeAgentHarness,
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
    "parallel_best_of_n": ParallelBestOfNHarness,
    "blocking_orchestrator": BlockingOrchestratorHarness,
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
    "xai_hosted_multi_agent": XAIHostedMultiAgentHarness,
}


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
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
            memory_beta_version=args.managed_memory_beta_version,
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
        if _paths_overlap(args.prime_agent_cwd, args.output):
            raise ValueError(
                "--output and --prime-agent-cwd must be disjoint directories"
            )
        return PrimeAgentJSONBackend(
            cwd=args.prime_agent_cwd,
            executable=args.prime_agent_executable,
            provider=args.prime_agent_provider,
            model=args.model,
            no_session=not args.prime_agent_persist_session,
            pass_env=args.prime_agent_pass_env,
            expected_version=args.prime_agent_expected_version,
            expected_executable_sha256=args.prime_agent_executable_sha256,
            max_output_bytes=args.prime_agent_max_output_bytes,
            allow_sensitive_environment=(args.prime_agent_allow_sensitive_environment),
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
    if "grok_build" in names and provider != "grok-build":
        raise ValueError("grok_build harness requires --provider grok-build")
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
    if provider in {
        "prime-agent",
        "grok-build",
        "macu-upstream",
        "rlm-upstream",
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
            "RLM upstream defaults to its Docker REPL; non-Docker environments need "
            "an independently enforced outer sandbox"
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
        _load_config(args.config), provider=args.provider
    )
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
    if isinstance(backend, (GrokBuildJSONBackend, PrimeAgentJSONBackend)):
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
        _load_config(args.config), provider=args.provider
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
    parser.add_argument(
        "--provider",
        choices=(
            "openai-responses",
            "openai-compatible-responses",
            "openai-compatible-chat",
            "anthropic-messages",
            "anthropic-managed-agents",
            "prime-agent",
            "grok-build",
            "xai-responses",
            "macu-upstream",
            "rlm-upstream",
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

    implementations_parser = subparsers.add_parser("list-implementations")
    implementations_parser.add_argument("--application", choices=catalog_applications())
    implementations_parser.add_argument("--json", action="store_true")
    implementations_parser.set_defaults(
        handler=lambda args: _list_implementations(
            application=args.application, json_output=args.json
        )
    )

    validate_parser = subparsers.add_parser("validate")
    _add_common_arguments(validate_parser)
    validate_parser.set_defaults(handler=_validate)

    run_parser = subparsers.add_parser("run")
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
    run_parser.add_argument(
        "--managed-memory-beta-version", default="agent-memory-2026-07-22"
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
    run_parser.add_argument("--prime-agent-persist-session", action="store_true")
    run_parser.add_argument("--prime-agent-expected-version")
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
    run_parser.add_argument("--grok-expected-version")
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
                "runnable_count": sum(
                    profile.status == "runnable"
                    for profile in catalog_implementations(name)
                ),
                "simulation_count": sum(
                    profile.status == "simulation"
                    for profile in catalog_implementations(name)
                ),
                "catalog_only_count": sum(
                    profile.status == "catalog_only"
                    for profile in catalog_implementations(name)
                ),
            }
            for name in applications
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name in applications:
            print(name)
    return 0


def _list_implementations(
    *, application: str | None = None, json_output: bool = False
) -> int:
    implementations = catalog_implementations(application)
    if json_output:
        print(
            json.dumps(
                [implementation.as_dict() for implementation in implementations],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for implementation in implementations:
            print(implementation.key)
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
