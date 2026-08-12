"""Command-line boundary for the maintained minimal-agent paths."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import __version__
from ._hash import stable_file_sha256
from .agent import MiniAgent
from .benchmarks.base import (
    EvaluationRunner,
    atomic_bytes,
    atomic_json,
    harness_identity,
    raise_after_cleanup,
    spec_bound_agent,
    task_agent_prefix,
)
from .doctor import _doctor
from .environments.base import AgentEnvironment
from .grading import _grade, _required_path
from .models import BackendModel, Model, build_model
from .orchestrator import AgentBuilder, Orchestrator
from .profiles import load_profile, prompt_for
from .providers import TokenPricing, _validate_endpoint
from .runtime import RunContext, TraceRecorder, redact_artifact
from .specs import AgentSpecV1
from .storage import StorageLayout
from .types import AgentResult, BudgetLimits, strict_json_loads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="One minimal loop for SWE, web research, and computer use.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mini-agent {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile", help="Resolve one maintained default profile."
    )
    profile.add_argument(
        "--application",
        "--environment",
        dest="application",
        choices=("swe", "web", "computer", "cua"),
        required=True,
    )
    profile.add_argument("--profile", default="default")
    profile.add_argument("--model", default="openai/test-model")
    profile.add_argument("--multi-agent", action="store_true")
    profile.add_argument(
        "--format",
        choices=("profile", "agent-spec", "translation-report"),
        default="profile",
    )

    run = commands.add_parser("run", help="Run one task with a domain environment.")
    run.add_argument("--environment", choices=("swe", "web", "computer"), required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--workspace", type=Path)
    run.add_argument(
        "--web-backend",
        choices=("serpapi", "jsonl", "browsecomp-plus"),
        default="serpapi",
    )
    run.add_argument("--corpus", type=Path)
    run.add_argument("--index", type=Path)
    run.add_argument("--index-sha256")
    run.add_argument("--anserini-jar", type=Path)
    run.add_argument("--page-reader", choices=("http", "playwright"), default="http")
    run.add_argument("--top-k", type=_positive_int, default=5)
    run.add_argument("--snippet-tokens", type=_positive_int)
    run.add_argument("--snippet-tokenizer", default="Qwen/Qwen3-0.6B")
    run.add_argument("--snippet-tokenizer-revision")
    run.add_argument("--env-url")
    run.add_argument("--env-token-env")
    _add_execution_arguments(run)
    _add_storage_arguments(run)

    evaluate = commands.add_parser("eval", help="Run a pinned benchmark adapter.")
    evaluate.add_argument(
        "--benchmark",
        choices=(
            "swebench",
            "browsecomp",
            "browsecomp-plus",
            "osworld-v1",
            "osworld-v2",
            "cua-speed-run",
        ),
        required=True,
    )
    evaluate.add_argument("--dataset", type=Path)
    selection = evaluate.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=_positive_int, default=1)
    selection.add_argument("--all", action="store_true")
    evaluate.add_argument("--sample-seed", type=int, default=0)
    evaluate.add_argument("--max-workers", type=_positive_int, default=1)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--progress", action="store_true")
    evaluate.add_argument(
        "--runtime", choices=("docker", "apptainer"), default="docker"
    )
    evaluate.add_argument("--container-runtime", nargs="+", default=["docker"])
    evaluate.add_argument("--apptainer-executable", default="apptainer")
    evaluate.add_argument("--overlay-size-mib", type=_positive_int, default=16 * 1024)
    _add_provider_arguments(evaluate, prefix="grader-")
    evaluate.add_argument("--index", type=Path)
    evaluate.add_argument("--index-sha256")
    evaluate.add_argument("--anserini-jar", type=Path)
    evaluate.add_argument(
        "--page-reader", choices=("http", "playwright"), default="http"
    )
    evaluate.add_argument("--top-k", type=_positive_int, default=5)
    evaluate.add_argument("--snippet-tokens", type=_positive_int, default=512)
    evaluate.add_argument("--snippet-tokenizer", default="Qwen/Qwen3-0.6B")
    evaluate.add_argument("--snippet-tokenizer-revision")
    evaluate.add_argument("--checkout", type=Path)
    evaluate.add_argument("--task-list", type=Path)
    evaluate.add_argument("--provider-name", default="docker")
    evaluate.add_argument("--path-to-vm")
    evaluate.add_argument("--osworld-apptainer-image", type=Path)
    evaluate.add_argument("--headed", action="store_true")
    evaluate.add_argument("--screen-width", type=_positive_int, default=1920)
    evaluate.add_argument("--screen-height", type=_positive_int, default=1080)
    evaluate.add_argument("--enable-proxy", action="store_true")
    evaluate.add_argument("--client-password-env")
    evaluate.add_argument("--include-gitlab", action="store_true")
    evaluate.add_argument("--benchmark-path", type=Path)
    evaluate.add_argument("--backend", default="gym-anything-qemu-apptainer")
    evaluate.add_argument("--qemu-cache", type=Path)
    evaluate.add_argument("--seed", type=int, default=0)
    _add_execution_arguments(evaluate)
    _add_storage_arguments(evaluate)

    grade = commands.add_parser(
        "grade", help="Invoke an official benchmark grader on generated artifacts."
    )
    grade.add_argument(
        "--benchmark", choices=("swebench", "browsecomp-plus"), required=True
    )
    grade.add_argument("--evaluation", type=Path, required=True)
    grade.add_argument("--output", type=Path)
    grade.add_argument("--checkout", type=Path)
    grade.add_argument("--dataset", type=Path)
    grade.add_argument("--ground-truth", type=Path)
    grade.add_argument("--qrel-evidence", type=Path)
    grade.add_argument("--eval-dir", type=Path)
    grade.add_argument("--run-id", default="mini-agent")
    grade.add_argument("--max-workers", type=_positive_int, default=1)
    grade.add_argument("--python-executable", default=sys.executable)
    grade.add_argument("--judge-model")
    grade.add_argument("--tensor-parallel-size", type=_positive_int, default=1)

    doctor = commands.add_parser(
        "doctor", help="Report prerequisites without running paid inference."
    )
    doctor.add_argument(
        "--target",
        choices=("all", "storage", "swebench", "web", "computer"),
        default="all",
    )
    doctor.add_argument("--checkout", type=Path)
    doctor.add_argument("--osworld-version", choices=("v1", "v2"))
    doctor.add_argument("--container-runtime", nargs="+", default=["docker"])
    doctor.add_argument("--runtime", choices=("docker", "apptainer"), default="docker")
    doctor.add_argument("--apptainer-executable", default="apptainer")
    doctor.add_argument("--overlay-size-mib", type=_positive_int, default=16 * 1024)
    doctor.add_argument("--web-mode", choices=("live", "fixed"), default="live")
    doctor.add_argument("--page-reader", choices=("http", "playwright"), default="http")
    doctor.add_argument("--index", type=Path)
    doctor.add_argument("--anserini-jar", type=Path)
    doctor.add_argument("--benchmark-path", type=Path)
    doctor.add_argument("--backend", default="gym-anything-qemu-apptainer")
    doctor.add_argument("--path-to-vm")
    doctor.add_argument("--osworld-apptainer-image", type=Path)
    doctor.add_argument("--qemu-cache", type=Path)
    doctor.add_argument("--min-durable-free-gib", type=_nonnegative_float, default=1.0)
    doctor.add_argument("--min-scratch-free-gib", type=_nonnegative_float, default=1.0)
    doctor.add_argument("--home", type=Path)
    doctor.add_argument("--scratch", type=Path)
    return parser


def _add_provider_arguments(
    parser: argparse.ArgumentParser, *, prefix: str = "", require_model: bool = False
) -> None:
    """Declare one provider option set; the grader gets the same set prefixed."""

    parser.add_argument(f"--{prefix}model", required=require_model)
    parser.add_argument(f"--{prefix}expected-provider-model")
    parser.add_argument(f"--{prefix}base-url")
    parser.add_argument(f"--{prefix}api-key-env")
    parser.add_argument(f"--{prefix}provider-body", default="{}")
    parser.add_argument(
        f"--{prefix}protocol", choices=("responses", "chat-completions")
    )
    parser.add_argument(
        f"--{prefix}provider-header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
    )
    parser.add_argument(f"--{prefix}max-output-tokens", type=_positive_int)
    parser.add_argument(f"--{prefix}provider-retries", type=_nonnegative_int)
    parser.add_argument(f"--{prefix}provider-timeout", type=_positive_float)
    parser.add_argument(f"--{prefix}max-history-images", type=_history_images)
    parser.add_argument(f"--{prefix}input-price", type=_nonnegative_float)
    parser.add_argument(f"--{prefix}output-price", type=_nonnegative_float)
    parser.add_argument(f"--{prefix}cache-read-price", type=_nonnegative_float)
    parser.add_argument(f"--{prefix}cache-write-price", type=_nonnegative_float)


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    _add_provider_arguments(parser, require_model=True)
    parser.add_argument("--system-prompt")
    parser.add_argument("--max-steps", type=_positive_int, default=64)
    parser.add_argument("--max-model-calls", type=_positive_int, default=64)
    parser.add_argument("--model-concurrency", type=_positive_int, default=4)
    parser.add_argument("--max-input-tokens-budget", type=_nonnegative_int)
    parser.add_argument("--max-output-tokens-budget", type=_nonnegative_int)
    parser.add_argument("--max-cost-usd", type=_nonnegative_float)
    parser.add_argument("--wall-time", type=_positive_float, default=900.0)
    parser.add_argument("--max-tool-calls", type=_positive_int, default=256)
    parser.add_argument(
        "--max-tool-output-bytes", type=_positive_int, default=8 * 1024 * 1024
    )
    parser.add_argument("--multi-agent", action="store_true")
    parser.add_argument("--max-active-agents", type=_positive_int, default=4)
    parser.add_argument("--max-total-agents", type=_positive_int, default=8)
    parser.add_argument("--per-agent-model-calls", type=_positive_int)
    parser.add_argument("--capture-content", action="store_true")


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            resolved = load_profile(args.application, args.profile, model=args.model)
            if args.format == "profile":
                value = resolved.as_dict()
            elif args.format == "agent-spec":
                spec = resolved.to_agent_spec(multi_agent=args.multi_agent)
                value = {**spec.as_dict(), "fingerprint": spec.fingerprint}
            else:
                value = resolved.translation_report(
                    multi_agent=args.multi_agent
                ).as_dict()
            print(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "eval":
            return asyncio.run(_evaluate(args))
        if args.command == "grade":
            return _grade(args)
        if args.command == "doctor":
            return asyncio.run(_doctor(args))
        parser.error(f"unknown command {args.command!r}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"mini-agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 2


async def _run(args: argparse.Namespace) -> int:
    if not isinstance(args.task, str) or not args.task.strip():
        raise ValueError("--task must be non-empty")
    _validate_runtime_credentials(args)
    _validate_topology_arguments(args)
    output, work, layout = _new_output(args, "run")
    print(f"mini-agent run output: {output}", file=sys.stderr)
    limits = _limits(args)
    trace = TraceRecorder(output / "trace.jsonl", secrets=_secrets(args))
    context = RunContext(
        limits=limits, trace=trace, capture_content=bool(args.capture_content)
    )
    model_factory = _model_factory(args)
    system_prompt = args.system_prompt or prompt_for(
        args.environment, multi_agent=args.multi_agent
    )
    spec = _resolved_agent_spec(args, args.environment, system_prompt)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "mini-agent-run-v2",
            "harness": harness_identity(),
            "environment": args.environment,
            "model": args.model,
            "task_sha256": hashlib.sha256(args.task.encode("utf-8")).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "multi_agent": bool(args.multi_agent),
            "agent_spec": spec.identity_dict(),
            "topology": _topology_config(args),
            "execution": _execution_config(args),
            "limits": asdict(limits),
            "storage": {
                "durable": str(layout.root),
                "scratch": str(layout.scratch),
            },
        },
    )

    if args.environment == "swe":
        answer, metadata = await _run_swe(
            args, context, model_factory, spec, output, work
        )
    elif args.environment == "web":
        answer, metadata = await _run_web(args, context, model_factory, spec)
    else:
        answer, metadata = await _run_computer(args, context, model_factory, spec)
    result = redact_artifact(
        {
            "answer": answer,
            "metadata": metadata,
            "accounting": context.ledger.snapshot(),
            "elapsed_seconds": trace.elapsed,
            "backend_active_union_seconds": trace.backend_active_union_seconds,
        },
        _secrets(args),
    )
    atomic_json(output / "result.json", result)
    print(json.dumps({"output": str(output), **result}, indent=2, sort_keys=True))
    return 0


async def _run_swe(
    args: argparse.Namespace,
    context: RunContext,
    model_factory: Callable[[str], Model],
    spec: AgentSpecV1,
    output: Path,
    work: Path,
) -> tuple[str, Mapping[str, Any]]:
    system_prompt = spec.system_prompt
    from .environments.swe import BashEnvironment, SWEPatchState

    if args.workspace is None:
        raise ValueError(
            "--workspace is required for swe runs: the bash agent edits the "
            "selected directory as the current user with no sandbox; pass an "
            "isolated workspace directory"
        )
    workspace = args.workspace.expanduser().resolve()
    if args.multi_agent:

        async def environment_factory(agent_id: str) -> BashEnvironment:
            del agent_id
            return await BashEnvironment.isolated(
                workspace, scratch_root=work / "agents"
            )

        orchestrator = Orchestrator(
            agent_builder=_agent_builder(
                model_factory, system_prompt, args.max_steps, agent_spec=spec
            ),
            environment_factory=environment_factory,
            context=context,
            max_active_agents=args.max_active_agents,
            max_total_agents=args.max_total_agents,
            per_agent_limits=_per_agent_limits(args),
        )
        result = await orchestrator.run(args.task)
        state = orchestrator.records[orchestrator.root_id].state
        if not isinstance(state, SWEPatchState):
            raise RuntimeError("root SWE agent did not produce a patch")
        atomic_bytes(output / "patch.diff", state.patch)
        return result.answer, {
            "mode": "multi",
            "workspace_modified": False,
            "patch": "patch.diff",
            "agents": {
                agent_id: record.status
                for agent_id, record in orchestrator.records.items()
            },
        }

    environment = BashEnvironment(workspace)
    result = await _run_single_agent(
        environment,
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        context=context,
        task=args.task,
        label="direct SWE run",
        agent_spec=spec,
    )
    return result.answer, {
        "mode": "single",
        "workspace_modified": True,
        "environment": environment.provenance(),
        "steps": result.steps,
    }


async def _run_web(
    args: argparse.Namespace,
    context: RunContext,
    model_factory: Callable[[str], Model],
    spec: AgentSpecV1,
) -> tuple[str, Mapping[str, Any]]:
    system_prompt = spec.system_prompt
    tokenizer = _load_tokenizer(args) if args.snippet_tokens is not None else None

    async def environment_factory(agent_id: str) -> Any:
        del agent_id
        return _browser_environment(args, tokenizer=tokenizer)

    if args.multi_agent:
        orchestrator = Orchestrator(
            agent_builder=_agent_builder(
                model_factory, system_prompt, args.max_steps, agent_spec=spec
            ),
            environment_factory=environment_factory,
            context=context,
            max_active_agents=args.max_active_agents,
            max_total_agents=args.max_total_agents,
            per_agent_limits=_per_agent_limits(args),
        )
        result = await orchestrator.run(args.task)
        return result.answer, {
            "mode": "multi",
            "agents": {
                agent_id: record.status
                for agent_id, record in orchestrator.records.items()
            },
            "browsers": {
                agent_id: {
                    "accounting": record.environment.base.accounting(),
                    "provenance": record.environment.base.provenance(),
                }
                for agent_id, record in orchestrator.records.items()
                if record.environment is not None
            },
        }

    environment = await environment_factory("/root")
    result = await _run_single_agent(
        environment,
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        context=context,
        task=args.task,
        label="direct web run",
        agent_spec=spec,
    )
    return result.answer, {
        "mode": "single",
        "steps": result.steps,
        "accounting": environment.accounting(),
        "environment": environment.provenance(),
    }


async def _run_computer(
    args: argparse.Namespace,
    context: RunContext,
    model_factory: Callable[[str], Model],
    spec: AgentSpecV1,
) -> tuple[str, Mapping[str, Any]]:
    system_prompt = spec.system_prompt
    from .environments.cua import CUAEnvironment, CUASpeedRunClient

    if args.multi_agent:
        raise ValueError(
            "direct computer multi-agent runs need an isolated machine factory; "
            "use eval with OSWorld or cua-speed-run"
        )
    if not args.env_url:
        raise ValueError("computer run requires --env-url")
    token = None
    if args.env_token_env:
        token = os.environ.get(args.env_token_env)
        if not token:
            raise ValueError(
                f"CUA gateway token environment variable {args.env_token_env!r} "
                "is unset or empty"
            )
    environment = CUAEnvironment(CUASpeedRunClient(args.env_url, bearer_token=token))
    result = await _run_single_agent(
        environment,
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        context=context,
        task=args.task,
        label="direct computer run",
        agent_spec=spec,
    )
    return result.answer, {
        "mode": "single",
        "steps": result.steps,
        "environment": environment.provenance(),
    }


async def _run_single_agent(
    environment: Any,
    *,
    model_factory: Callable[[str], Model],
    system_prompt: str,
    max_steps: int,
    context: RunContext,
    task: str,
    label: str,
    agent_spec: AgentSpecV1 | None = None,
) -> AgentResult:
    result: AgentResult | None = None
    operation_error: BaseException | None = None
    try:
        if agent_spec is not None:
            agent = spec_bound_agent(
                agent_spec,
                model=model_factory("/root"),
                environment=environment,
                context=context,
                agent_id="/root",
                system_prompt=system_prompt,
                max_steps=max_steps,
            )
        else:
            agent = MiniAgent(
                model=model_factory("/root"),
                environment=environment,
                system_prompt=system_prompt,
                max_steps=max_steps,
                context=context,
            )
        result = await agent.run(task)
    except BaseException as exc:
        operation_error = exc
    cleanup_error: BaseException | None = None
    try:
        await environment.close()
    except BaseException as exc:
        cleanup_error = exc
    raise_after_cleanup(label, operation_error, cleanup_error)
    if result is None:
        raise AssertionError("successful direct run has no result")
    return result


async def _evaluate(args: argparse.Namespace) -> int:
    _validate_runtime_credentials(args)
    _validate_topology_arguments(args)
    if args.capture_content:
        raise ValueError(
            "benchmark traces are content-redacted; --capture-content is for run only"
        )
    output, work, layout = _evaluation_output(args)
    print(f"mini-agent eval output: {output}", file=sys.stderr)
    limit = None if args.all else args.limit
    model_factory = _model_factory(args)
    benchmark = args.benchmark
    domain = _benchmark_domain(benchmark)
    resolved_prompt = args.system_prompt or prompt_for(
        domain, multi_agent=args.multi_agent
    )
    agent_spec = _resolved_agent_spec(args, domain, resolved_prompt)
    checkout_cwd: Path | None = None

    if benchmark == "swebench":
        from .benchmarks.swebench import (
            load_swebench,
            prepare_swebench_image_bindings,
            run_swebench_task,
        )

        dataset = _required_path(args.dataset, "--dataset")
        tasks = load_swebench(dataset, limit=limit)
        image_bindings = await prepare_swebench_image_bindings(
            tasks,
            runtime=args.runtime,
            container_runtime=tuple(args.container_runtime),
            apptainer_executable=args.apptainer_executable,
            apptainer_image_cache=layout.assets / "apptainer-images",
        )
        args._swebench_image_bindings = {
            task_id: dict(binding.manifest_identity())
            for task_id, binding in sorted(image_bindings.items())
        }

        async def worker(task: Any, context: RunContext, directory: Path) -> Any:
            return await run_swebench_task(
                task,
                context,
                directory,
                model_factory=model_factory,
                system_prompt=resolved_prompt,
                max_steps=args.max_steps,
                agent_spec=agent_spec,
                runtime=args.runtime,
                model_name=args.model,
                scratch_root=work / "swebench",
                container_runtime=tuple(args.container_runtime),
                apptainer_executable=args.apptainer_executable,
                apptainer_image_cache=layout.assets / "apptainer-images",
                image_binding=image_bindings[task.task_id],
                overlay_size_mib=args.overlay_size_mib,
                multi_agent=args.multi_agent,
                max_active_agents=args.max_active_agents,
                max_total_agents=args.max_total_agents,
                per_agent_limits=_per_agent_limits(args),
            )

    elif benchmark in {"browsecomp", "browsecomp-plus"}:
        from .benchmarks.web import (
            grade_browsecomp,
            load_browsecomp,
            load_browsecomp_plus,
            run_web_task,
        )

        dataset = _required_path(args.dataset, "--dataset")
        if benchmark == "browsecomp":
            tasks = load_browsecomp(dataset, limit=limit, sample_seed=args.sample_seed)
            if not args.grader_model:
                raise ValueError("BrowseComp requires --grader-model")
            grader_factory = _grader_model_factory(args)

            def browser_factory(agent_id: str) -> Any:
                del agent_id
                return _live_browser(args)

            async def worker(task: Any, context: RunContext, directory: Path) -> Any:
                outcome = await run_web_task(
                    task,
                    context,
                    directory,
                    browser_factory=browser_factory,
                    model_factory=model_factory,
                    system_prompt=resolved_prompt,
                    max_steps=args.max_steps,
                    agent_spec=agent_spec,
                    multi_agent=args.multi_agent,
                    max_active_agents=args.max_active_agents,
                    max_total_agents=args.max_total_agents,
                    model_name=args.model,
                    per_agent_limits=_per_agent_limits(args),
                )
                grader = grader_factory(task_agent_prefix(task.task_id) + "/grader")
                score, raw = await grade_browsecomp(
                    task=task,
                    response=outcome.answer,
                    grader=grader,
                    context=context,
                )
                grader_path = directory / "private" / "grader.json"
                atomic_json(
                    grader_path,
                    {
                        "contains_hidden_benchmark_data": True,
                        "grader_model": args.grader_model,
                        "output": raw,
                    },
                )
                return replace(
                    outcome,
                    score=score,
                    metadata={
                        **dict(outcome.metadata),
                        "grader_model": args.grader_model,
                        "private_grader_artifact": "private/grader.json",
                        "private_grader_sha256": hashlib.sha256(
                            grader_path.read_bytes()
                        ).hexdigest(),
                    },
                )

        else:
            if args.index is None:
                raise ValueError("BrowseComp-Plus requires --index")
            if args.anserini_jar is None:
                raise ValueError("BrowseComp-Plus requires --anserini-jar")
            if args.top_k != 5 or args.snippet_tokens != 512:
                raise ValueError(
                    "BrowseComp-Plus evaluation requires --top-k 5 and "
                    "--snippet-tokens 512"
                )
            if args.snippet_tokenizer != "Qwen/Qwen3-0.6B":
                raise ValueError(
                    "BrowseComp-Plus evaluation requires "
                    "--snippet-tokenizer Qwen/Qwen3-0.6B"
                )
            from .environments.web import (
                directory_sha256,
                validate_anserini_jar,
            )

            observed_index_sha = directory_sha256(args.index.expanduser())
            if (
                args.index_sha256 is not None
                and args.index_sha256.casefold() != observed_index_sha
            ):
                raise ValueError("--index-sha256 does not match the Lucene index")
            args.index_sha256 = observed_index_sha
            _, args.anserini_jar_sha256 = validate_anserini_jar(args.anserini_jar)
            tokenizer = _load_tokenizer(args, require_revision=True)
            tasks = load_browsecomp_plus(dataset, limit=limit)

            def browser_factory(agent_id: str) -> Any:
                del agent_id
                return _fixed_browser(args, tokenizer)

            async def worker(task: Any, context: RunContext, directory: Path) -> Any:
                return await run_web_task(
                    task,
                    context,
                    directory,
                    browser_factory=browser_factory,
                    model_factory=model_factory,
                    system_prompt=resolved_prompt,
                    max_steps=args.max_steps,
                    agent_spec=agent_spec,
                    multi_agent=args.multi_agent,
                    max_active_agents=args.max_active_agents,
                    max_total_agents=args.max_total_agents,
                    model_name=args.model,
                    per_agent_limits=_per_agent_limits(args),
                )

    elif benchmark in {"osworld-v1", "osworld-v2"}:
        from .benchmarks.osworld import (
            UpstreamDesktopFactory,
            load_osworld,
            run_osworld_task,
        )

        version = benchmark.rsplit("-", 1)[1]
        checkout = _required_path(args.checkout, "--checkout").resolve()
        if args.provider_name.casefold().strip() == "docker":
            default_image = checkout / "docker_vm_data" / "Ubuntu.qcow2"
            selected_image = (
                Path(args.path_to_vm).expanduser()
                if args.path_to_vm is not None
                else default_image
            )
            if not selected_image.is_file():
                raise ValueError(
                    "OSWorld Docker evaluation requires a pre-provisioned "
                    "Ubuntu.qcow2 via --path-to-vm or "
                    f"{default_image}"
                )
            args.path_to_vm = str(selected_image.resolve())
        if args.runtime == "apptainer":
            if args.provider_name.casefold().strip() != "docker":
                raise ValueError(
                    "OSWorld --runtime apptainer requires --provider-name docker"
                )
            if args.osworld_apptainer_image is None:
                raise ValueError(
                    "OSWorld --runtime apptainer requires --osworld-apptainer-image"
                )
        elif args.osworld_apptainer_image is not None:
            raise ValueError("--osworld-apptainer-image requires --runtime apptainer")
        checkout_cwd = checkout
        with _working_directory(checkout):
            tasks = load_osworld(
                checkout,
                version=version,
                task_list=args.task_list,
                limit=limit,
                exclude_gitlab=not args.include_gitlab,
            )
        password = (
            os.environ.get(args.client_password_env, "")
            if args.client_password_env
            else ""
        )
        desktop_factory = UpstreamDesktopFactory(
            checkout,
            version=version,
            provider_name=args.provider_name,
            path_to_vm=args.path_to_vm,
            headless=not args.headed,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            enable_proxy=args.enable_proxy,
            client_password=password,
            apptainer_image=args.osworld_apptainer_image,
            apptainer_executable=args.apptainer_executable,
        )
        args._osworld_factory_provenance = dict(desktop_factory.provenance())

        async def worker(task: Any, context: RunContext, directory: Path) -> Any:
            return await run_osworld_task(
                task,
                context,
                directory,
                desktop_factory=desktop_factory,
                model_factory=model_factory,
                system_prompt=resolved_prompt,
                max_steps=args.max_steps,
                agent_spec=agent_spec,
                multi_agent=args.multi_agent,
                max_active_agents=args.max_active_agents,
                max_total_agents=args.max_total_agents,
                per_agent_limits=_per_agent_limits(args),
            )

    else:
        from .benchmarks.cua_speedrun import (
            load_cua_speedrun,
            prepare_cua_speedrun_backend,
            run_cua_speedrun_task,
        )

        _configure_cua_runtime(args.qemu_cache, work)
        checkout = _required_path(args.checkout, "--checkout")
        benchmark_path = _required_path(args.benchmark_path, "--benchmark-path")
        tasks = load_cua_speedrun(
            checkout,
            benchmark_path,
            seed=args.seed,
            limit=limit,
        )
        args._cua_backend_preflight = dict(
            prepare_cua_speedrun_backend(
                checkout,
                benchmark_path,
                backend_name=args.backend,
            )
        )

        async def worker(task: Any, context: RunContext, directory: Path) -> Any:
            return await run_cua_speedrun_task(
                task,
                context,
                directory,
                backend_name=args.backend,
                model_factory=model_factory,
                system_prompt=resolved_prompt,
                max_steps=args.max_steps,
                agent_spec=agent_spec,
                multi_agent=args.multi_agent,
                max_active_agents=args.max_active_agents,
                max_total_agents=args.max_total_agents,
                per_agent_limits=_per_agent_limits(args),
            )

    selected_worker = (
        _progress_worker(worker, benchmark) if args.progress else worker
    )

    limits = _limits(args)
    runner = EvaluationRunner(
        benchmark=benchmark,
        tasks=tasks,
        output=output,
        config=_evaluation_config(args, layout),
        limits=limits,
        max_workers=args.max_workers,
        capture_content=bool(args.capture_content),
        secrets=_secrets(args),
    )
    if checkout_cwd is None:
        summary = await runner.run(selected_worker, resume=args.resume)
    else:
        with _working_directory(checkout_cwd):
            summary = await runner.run(selected_worker, resume=args.resume)
    if benchmark == "browsecomp-plus":
        assert args.index is not None
        from .environments.web import directory_sha256

        final_index_sha = directory_sha256(args.index.expanduser())
        if final_index_sha != args.index_sha256:
            raise RuntimeError("Lucene index changed during BrowseComp-Plus evaluation")
    if benchmark == "swebench":
        from .benchmarks.swebench import collect_predictions

        count = collect_predictions(output, output / "predictions.jsonl")
        summary = {**summary, "predictions": count}
    elif benchmark == "browsecomp-plus":
        from .benchmarks.web import collect_browsecomp_plus_runs

        count = collect_browsecomp_plus_runs(output, output / "official_runs")
        summary = {**summary, "official_runs": count}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 and summary["blocked"] == 0 else 1


def _browser_environment(args: argparse.Namespace, *, tokenizer: Any = None) -> Any:
    from .environments.web import (
        BrowserEnvironment,
        BrowseCompPlusBackend,
        JsonlSearchBackend,
    )

    if args.web_backend == "serpapi":
        return _live_browser(args, tokenizer=tokenizer)
    if args.web_backend == "jsonl":
        corpus = _required_path(args.corpus, "--corpus")
        backend: Any = JsonlSearchBackend(corpus)
    else:
        index = _required_path(args.index, "--index")
        backend = BrowseCompPlusBackend(
            index,
            _required_path(args.anserini_jar, "--anserini-jar"),
            expected_sha256=args.index_sha256,
        )
    return BrowserEnvironment(
        backend,
        top_k=args.top_k,
        snippet_tokens=args.snippet_tokens,
        tokenizer=tokenizer,
    )


def _live_browser(args: argparse.Namespace, *, tokenizer: Any = None) -> Any:
    from .environments.web import (
        BrowserEnvironment,
        HttpPageReader,
        PlaywrightPageReader,
        SerpAPIBackend,
    )

    reader = (
        PlaywrightPageReader() if args.page_reader == "playwright" else HttpPageReader()
    )
    return BrowserEnvironment(
        SerpAPIBackend(page_reader=reader),
        top_k=args.top_k,
        snippet_tokens=args.snippet_tokens if tokenizer is not None else None,
        tokenizer=tokenizer,
    )


def _fixed_browser(args: argparse.Namespace, tokenizer: Any) -> Any:
    from .environments.web import BrowserEnvironment, BrowseCompPlusBackend

    assert args.index is not None
    return BrowserEnvironment(
        BrowseCompPlusBackend(
            args.index,
            _required_path(args.anserini_jar, "--anserini-jar"),
            expected_sha256=args.index_sha256,
        ),
        top_k=args.top_k,
        max_observation_chars=None,
        snippet_tokens=args.snippet_tokens,
        tokenizer=tokenizer,
        allow_open=False,
    )


class _TokenizersSnippetAdapter:
    """Expose the two upstream snippet operations without a model framework."""

    def __init__(
        self,
        backend: Any,
        *,
        name: str,
        revision: str | None,
        tokenizer_json_sha256: str,
    ) -> None:
        self._backend = backend
        self.name_or_path = name
        self.init_kwargs = {"_commit_hash": revision}
        self.tokenizer_json_sha256 = tokenizer_json_sha256

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        return list(
            self._backend.encode(text, add_special_tokens=add_special_tokens).ids
        )

    def decode(self, tokens: Sequence[Any], *, skip_special_tokens: bool) -> str:
        return self._backend.decode(
            list(tokens), skip_special_tokens=skip_special_tokens
        )


def _load_tokenizer(args: argparse.Namespace, *, require_revision: bool = False) -> Any:
    requested = args.snippet_tokenizer_revision
    if require_revision and not (
        isinstance(requested, str) and re.fullmatch(r"[0-9a-fA-F]{40}", requested)
    ):
        raise ValueError(
            "BrowseComp-Plus evaluation requires "
            "--snippet-tokenizer-revision as a full 40-character commit"
        )
    if require_revision:
        from .environments.web import (
            HUGGINGFACE_HUB_VERSION,
            TOKENIZERS_VERSION,
        )

        for name, expected in (
            ("huggingface-hub", HUGGINGFACE_HUB_VERSION),
            ("tokenizers", TOKENIZERS_VERSION),
        ):
            try:
                observed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"BrowseComp-Plus evaluation requires {name}=={expected}"
                ) from exc
            if observed != expected:
                raise RuntimeError(
                    f"BrowseComp-Plus evaluation requires {name}=={expected}, "
                    f"found {observed}"
                )
    try:
        from huggingface_hub import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            hf_hub_download,
        )
        from tokenizers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            Tokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "token-bounded fixed retrieval requires mini-agent[web-fixed]"
        ) from exc
    tokenizer_path = Path(
        hf_hub_download(
            repo_id=args.snippet_tokenizer,
            filename="tokenizer.json",
            revision=requested,
        )
    )
    snapshot_revision = (
        tokenizer_path.parent.name.casefold()
        if tokenizer_path.parent.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-fA-F]{40}", tokenizer_path.parent.name)
        else None
    )
    exact_requested = (
        requested.casefold()
        if isinstance(requested, str) and re.fullmatch(r"[0-9a-fA-F]{40}", requested)
        else None
    )
    if (
        exact_requested is not None
        and snapshot_revision is not None
        and snapshot_revision != exact_requested
    ):
        raise RuntimeError("downloaded tokenizer revision does not match the request")
    resolved = snapshot_revision or exact_requested
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    try:
        tokenizer_json = tokenizer_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("downloaded tokenizer.json is not UTF-8") from exc
    tokenizer = _TokenizersSnippetAdapter(
        # Parse the same immutable bytes that were hashed. Reloading this path
        # would leave a hash/load race in a shared Hugging Face cache.
        Tokenizer.from_str(tokenizer_json),
        name=args.snippet_tokenizer,
        revision=resolved,
        tokenizer_json_sha256=tokenizer_sha256,
    )
    setattr(args, "_resolved_snippet_tokenizer_revision", resolved)
    setattr(args, "_snippet_tokenizer_json_sha256", tokenizer_sha256)
    return tokenizer


def _agent_builder(
    model_factory: Callable[[str], Model],
    system_prompt: str,
    max_steps: int,
    *,
    agent_spec: AgentSpecV1 | None = None,
) -> AgentBuilder:
    def build(
        agent_id: str,
        environment: AgentEnvironment,
        context: RunContext,
    ) -> MiniAgent:
        if agent_spec is not None:
            return spec_bound_agent(
                agent_spec,
                model=model_factory(agent_id),
                environment=environment,
                context=context,
                agent_id=agent_id,
                system_prompt=system_prompt,
                max_steps=max_steps,
            )
        return MiniAgent(
            model=model_factory(agent_id),
            environment=environment,
            system_prompt=system_prompt,
            max_steps=max_steps,
            context=context,
            agent_id=agent_id,
        )

    return build


def _provider_option(args: argparse.Namespace, prefix: str, name: str) -> Any:
    return getattr(args, f"{prefix}{name}")


def _model_factory(
    args: argparse.Namespace, *, prefix: str = ""
) -> Callable[[str], BackendModel]:
    model = _provider_option(args, prefix, "model")
    if prefix and not model:
        raise ValueError("grader model is required")
    body = _provider_body(_provider_option(args, prefix, "provider_body"))
    headers = _provider_headers(_provider_option(args, prefix, "provider_header"))
    pricing = _pricing(args, prefix=prefix)

    transport: dict[str, Any] = {}
    if _provider_option(args, prefix, "provider_retries") is not None:
        transport["max_retries"] = _provider_option(args, prefix, "provider_retries")
    if _provider_option(args, prefix, "provider_timeout") is not None:
        transport["timeout_seconds"] = _provider_option(
            args, prefix, "provider_timeout"
        )
    history = _provider_option(args, prefix, "max_history_images")
    if history is not None:
        transport["max_history_images"] = None if history == "unlimited" else history

    def create(agent_id: str) -> BackendModel:
        return build_model(
            model,
            base_url=_provider_option(args, prefix, "base_url"),
            api_key_env=_provider_option(args, prefix, "api_key_env"),
            max_output_tokens=_provider_option(args, prefix, "max_output_tokens"),
            default_body=body,
            default_headers=headers,
            pricing=pricing,
            protocol=_provider_option(args, prefix, "protocol"),
            agent_id=agent_id,
            expected_resolved_model=_provider_option(
                args, prefix, "expected_provider_model"
            ),
            **transport,
        )

    return create


def _grader_model_factory(
    args: argparse.Namespace,
) -> Callable[[str], BackendModel]:
    return _model_factory(args, prefix="grader_")


def _pricing(
    args: argparse.Namespace, *, prefix: str = ""
) -> TokenPricing | None:
    values = tuple(
        _provider_option(args, prefix, name)
        for name in (
            "input_price",
            "output_price",
            "cache_read_price",
            "cache_write_price",
        )
    )
    option = f"--{prefix.replace('_', '-')}"
    if all(value is None for value in values):
        if args.max_cost_usd is not None:
            raise ValueError(
                f"--max-cost-usd requires {option}input-price and "
                f"{option}output-price"
            )
        return None
    if values[0] is None or values[1] is None:
        raise ValueError(
            f"token pricing requires {option}input-price and {option}output-price"
        )
    return TokenPricing(*values)


def _limits(args: argparse.Namespace) -> BudgetLimits:
    return BudgetLimits(
        max_model_calls=args.max_model_calls,
        max_concurrency=args.model_concurrency,
        max_input_tokens=args.max_input_tokens_budget,
        max_output_tokens=args.max_output_tokens_budget,
        max_cost_usd=args.max_cost_usd,
        wall_time_seconds=args.wall_time,
        max_tool_calls=args.max_tool_calls,
        max_tool_output_bytes=args.max_tool_output_bytes,
    )


def _validate_topology_arguments(args: argparse.Namespace) -> None:
    if args.per_agent_model_calls is not None and not args.multi_agent:
        raise ValueError("--per-agent-model-calls requires --multi-agent")


def _per_agent_limits(args: argparse.Namespace) -> BudgetLimits | None:
    if args.per_agent_model_calls is None:
        return None
    return BudgetLimits(
        max_model_calls=args.per_agent_model_calls,
        max_concurrency=1,
        max_input_tokens=args.max_input_tokens_budget,
        max_output_tokens=args.max_output_tokens_budget,
        max_cost_usd=args.max_cost_usd,
        wall_time_seconds=args.wall_time,
        max_tool_calls=args.max_tool_calls,
        max_tool_output_bytes=args.max_tool_output_bytes,
    )


def _new_output(
    args: argparse.Namespace, prefix: str
) -> tuple[Path, Path, StorageLayout]:
    output, work, layout = _output_paths(args, prefix)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")
    if work.exists() and not work.is_dir():
        raise ValueError(f"scratch work path is not a directory: {work}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output is not empty: {output}")
    if work.exists() and any(work.iterdir()):
        raise ValueError(f"scratch work directory is not empty: {work}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    work.chmod(0o700)
    return output, work, layout


def _evaluation_output(
    args: argparse.Namespace,
) -> tuple[Path, Path, StorageLayout]:
    output, work, layout = _output_paths(args, args.benchmark)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")
    if work.exists() and not work.is_dir():
        raise ValueError(f"scratch work path is not a directory: {work}")
    if not args.resume and work.exists() and any(work.iterdir()):
        raise ValueError(f"scratch work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    work.chmod(0o700)
    return output, work, layout


def _output_paths(
    args: argparse.Namespace, prefix: str
) -> tuple[Path, Path, StorageLayout]:
    layout = StorageLayout.resolve(args.home, args.scratch)
    layout.ensure()
    run_id = args.run_id or (
        f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else layout.run(run_id)
    )
    work = layout.work(run_id)
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("durable output and scratch work paths must not overlap")
    return output, work, layout


def _configure_cua_runtime(qemu_cache: Path | None, work: Path) -> None:
    if qemu_cache is not None:
        expanded = qemu_cache.expanduser()
        if expanded.is_symlink():
            raise ValueError("--qemu-cache must not be a symlink")
        resolved_cache = expanded.resolve()
        if resolved_cache.exists() and not resolved_cache.is_dir():
            raise ValueError("--qemu-cache must be a directory")
        os.environ["GYM_ANYTHING_QEMU_CACHE"] = str(resolved_cache)
    qemu_work = work / "cua-speed-run-qemu"
    if qemu_work.is_symlink():
        raise ValueError("cua-speed-run QEMU work directory must not be a symlink")
    os.environ["GYM_ANYTHING_QEMU_WORK_DIR"] = str(qemu_work.resolve())


def _progress_worker(
    worker: Callable[[Any, RunContext, Path], Any], benchmark: str
) -> Callable[[Any, RunContext, Path], Any]:
    async def wrapped(task: Any, context: RunContext, directory: Path) -> Any:
        try:
            outcome = await worker(task, context, directory)
        except BaseException as exc:
            print(
                f"mini-agent eval [{benchmark}] {task.task_id}: "
                f"failed ({type(exc).__name__})",
                file=sys.stderr,
            )
            raise
        snapshot = context.ledger.snapshot()
        usage = snapshot["usage"]
        score = f" score={outcome.score}" if outcome.score is not None else ""
        print(
            f"mini-agent eval [{benchmark}] {task.task_id}: {outcome.status}"
            f"{score} | calls={snapshot['model_calls']}"
            f" tools={snapshot['tool_calls']}"
            f" cost_usd={usage['cost_usd']:.4f}",
            file=sys.stderr,
        )
        return outcome

    return wrapped


def _benchmark_domain(benchmark: str) -> str:
    return (
        "swe"
        if benchmark == "swebench"
        else "web"
        if benchmark in {"browsecomp", "browsecomp-plus"}
        else "computer"
    )


def _evaluation_config(
    args: argparse.Namespace, layout: StorageLayout
) -> Mapping[str, Any]:
    domain = _benchmark_domain(args.benchmark)
    resolved_prompt = args.system_prompt or prompt_for(
        domain, multi_agent=args.multi_agent
    )
    assets: dict[str, Any] = {}
    for name in ("dataset", "task_list", "anserini_jar"):
        value = getattr(args, name, None)
        if value is not None:
            expanded = value.expanduser()
            path = expanded.resolve()
            assets[name] = {
                "path": str(path),
                "sha256": _sha256_file(expanded) if path.is_file() else None,
            }
    for name in ("index", "checkout", "benchmark_path"):
        value = getattr(args, name, None)
        if value is not None:
            assets[name] = str(value.expanduser().resolve())
    topology = _topology_config(args)
    spec = _resolved_agent_spec(args, domain, resolved_prompt)
    if args.benchmark == "swebench":
        adapter: Mapping[str, Any] = {
            "runtime": args.runtime,
            **(
                {"container_runtime": list(args.container_runtime)}
                if args.runtime == "docker"
                else {
                    "apptainer_executable": args.apptainer_executable,
                    "overlay_size_mib": args.overlay_size_mib,
                }
            ),
        }
        prepared_images = getattr(args, "_swebench_image_bindings", None)
        if prepared_images is not None:
            adapter = {**adapter, "image_bindings": prepared_images}
    elif args.benchmark == "browsecomp":
        adapter = {
            "search": "serpapi",
            "page_reader": args.page_reader,
            "top_k": args.top_k,
            "sample_seed": args.sample_seed,
            "grader": _grader_execution_config(args),
        }
    elif args.benchmark == "browsecomp-plus":
        from .environments.web import BROWSECOMP_PLUS_INDEX_REVISION

        adapter = {
            "search": "lucene_bm25",
            "actions": ["search"],
            "upstream_query_template": "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "index_repository": "Tevatron/browsecomp-plus-indexes",
            "index_repository_revision": BROWSECOMP_PLUS_INDEX_REVISION,
            "top_k": args.top_k,
            "max_observation_chars": None,
            "index_sha256": args.index_sha256,
            "anserini_jar_sha256": getattr(args, "anserini_jar_sha256", None),
            "snippet_tokens": args.snippet_tokens,
            "snippet_tokenizer": args.snippet_tokenizer,
            "snippet_tokenizer_requested_revision": args.snippet_tokenizer_revision,
            "snippet_tokenizer_resolved_revision": getattr(
                args, "_resolved_snippet_tokenizer_revision", None
            ),
            "snippet_tokenizer_json_sha256": getattr(
                args, "_snippet_tokenizer_json_sha256", None
            ),
        }
    elif args.benchmark in {"osworld-v1", "osworld-v2"}:
        adapter = {
            "provider_name": args.provider_name,
            "container_runtime": (
                args.runtime
                if args.provider_name.casefold().strip() == "docker"
                else None
            ),
            "path_to_vm": args.path_to_vm,
            "osworld_apptainer_image": (
                str(args.osworld_apptainer_image.expanduser().resolve())
                if args.osworld_apptainer_image is not None
                else None
            ),
            "headless": not args.headed,
            "screen_size": [args.screen_width, args.screen_height],
            "enable_proxy": bool(args.enable_proxy),
            "client_password_env": args.client_password_env,
            "client_password_configured": bool(
                args.client_password_env and os.environ.get(args.client_password_env)
            ),
            "include_gitlab": bool(args.include_gitlab),
            "environment_factory": getattr(args, "_osworld_factory_provenance", None),
        }
    else:
        adapter = {
            "backend": args.backend,
            "seed": args.seed,
            "qemu_cache": (
                str(args.qemu_cache.expanduser().resolve())
                if args.qemu_cache is not None
                else os.environ.get("GYM_ANYTHING_QEMU_CACHE")
            ),
            "upstream_preflight": getattr(args, "_cua_backend_preflight", None),
        }
    return {
        "model": args.model,
        "agent_spec": spec.identity_dict(),
        "topology": topology,
        "system_prompt_sha256": hashlib.sha256(
            resolved_prompt.encode("utf-8")
        ).hexdigest(),
        "execution": _execution_config(args),
        "adapter": adapter,
        "assets": assets,
        "storage": {
            "durable": str(layout.root),
            "scratch": str(layout.scratch),
        },
    }


def _resolved_agent_spec(
    args: argparse.Namespace, domain: str, system_prompt: str
) -> AgentSpecV1:
    profile = load_profile(domain, model=args.model)
    return replace(
        profile.to_agent_spec(multi_agent=bool(args.multi_agent)),
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        budget=_limits(args),
    )


def _topology_config(args: argparse.Namespace) -> dict[str, Any]:
    topology: dict[str, Any] = {
        "mode": "multi" if args.multi_agent else "single",
        "max_steps": args.max_steps,
    }
    if args.multi_agent:
        per_agent = _per_agent_limits(args)
        topology.update(
            {
                "max_active_agents": args.max_active_agents,
                "max_total_agents": args.max_total_agents,
                "per_agent_limits": asdict(per_agent) if per_agent else None,
            }
        )
    return topology


def _provider_config(
    args: argparse.Namespace, *, prefix: str = ""
) -> dict[str, Any]:
    option = f"--{prefix.replace('_', '-')}base-url"
    _validate_cli_endpoint(_provider_option(args, prefix, "base_url"), option)
    body = _provider_body(_provider_option(args, prefix, "provider_body"))
    pricing = _pricing(args, prefix=prefix)
    headers = _provider_headers(_provider_option(args, prefix, "provider_header"))
    return {
        "base_url": _provider_option(args, prefix, "base_url"),
        "api_key_env": _provider_option(args, prefix, "api_key_env"),
        "expected_provider_model": _provider_option(
            args, prefix, "expected_provider_model"
        ),
        "protocol": _provider_option(args, prefix, "protocol"),
        "provider_body_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "provider_headers": _header_identities(headers),
        "max_output_tokens": _provider_option(args, prefix, "max_output_tokens"),
        "provider_retries": _provider_option(args, prefix, "provider_retries"),
        "provider_timeout": _provider_option(args, prefix, "provider_timeout"),
        "max_history_images": _provider_option(args, prefix, "max_history_images"),
        "pricing": asdict(pricing) if pricing is not None else None,
    }


def _execution_config(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        **_provider_config(args),
        "max_steps": args.max_steps,
        "per_agent_model_calls": args.per_agent_model_calls,
    }


def _grader_execution_config(args: argparse.Namespace) -> Mapping[str, Any] | None:
    if not args.grader_model:
        return None
    return {
        "model": args.grader_model,
        **_provider_config(args, prefix="grader_"),
    }


def _provider_body(raw: str) -> Mapping[str, Any]:
    value = strict_json_loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("--provider-body must decode to a JSON object")
    sensitive = {
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "authorization",
        "cookie",
        "password",
        "proxy-authorization",
        "secret",
        "token",
        "x-api-key",
    }

    def inspect_value(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in sensitive:
                    raise ValueError(
                        "provider secrets belong in --api-key-env, not --provider-body"
                    )
                inspect_value(child)
        elif isinstance(item, list):
            for child in item:
                inspect_value(child)

    inspect_value(value)
    return dict(value)


def _provider_headers(raw: Sequence[str]) -> Mapping[str, str] | None:
    if not raw:
        return None
    sensitive = {
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "authorization",
        "cookie",
        "password",
        "proxy-authorization",
        "secret",
        "token",
        "x-api-key",
        "anthropic-version",
        "content-type",
    }
    headers: dict[str, str] = {}
    normalized_names: set[str] = set()
    for entry in raw:
        name, separator, value = entry.partition("=")
        if not separator or not name.strip() or not value:
            raise ValueError("--provider-header entries must look like NAME=VALUE")
        name = name.strip()
        if name.casefold() in sensitive:
            raise ValueError(
                "provider credentials belong in --api-key-env, not --provider-header"
            )
        normalized = name.casefold()
        if normalized in normalized_names:
            raise ValueError(f"--provider-header repeats {name!r}")
        normalized_names.add(normalized)
        headers[name] = value
    return headers


def _validate_cli_endpoint(value: str | None, option: str) -> None:
    if value is None:
        return
    try:
        _validate_endpoint(value, 120.0)
    except ValueError as exc:
        raise ValueError(f"invalid {option}: {exc}") from None


def _environment_variable_name(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError(f"{option} must name an environment variable")
    return value


def _validate_runtime_credentials(args: argparse.Namespace) -> None:
    _validate_cli_endpoint(getattr(args, "base_url", None), "--base-url")
    _validate_cli_endpoint(getattr(args, "grader_base_url", None), "--grader-base-url")
    for attribute, option in (
        ("api_key_env", "--api-key-env"),
        ("grader_api_key_env", "--grader-api-key-env"),
        ("env_token_env", "--env-token-env"),
        ("client_password_env", "--client-password-env"),
    ):
        _environment_variable_name(getattr(args, attribute, None), option)
    for attribute, option in (
        ("expected_provider_model", "--expected-provider-model"),
        ("grader_expected_provider_model", "--grader-expected-provider-model"),
    ):
        value = getattr(args, attribute, None)
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or "\x00" in value
        ):
            raise ValueError(f"{option} must be a non-empty model identifier")
    if (
        getattr(args, "grader_expected_provider_model", None) is not None
        and not getattr(args, "grader_model", None)
    ):
        raise ValueError("--grader-expected-provider-model requires --grader-model")
    _require_model_credential(
        getattr(args, "model", None), getattr(args, "api_key_env", None)
    )
    _require_model_credential(
        getattr(args, "grader_model", None),
        getattr(args, "grader_api_key_env", None),
    )


_DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "meta": "MODEL_API_KEY",
}


def _require_model_credential(model: str | None, api_key_env: str | None) -> None:
    """Fail before any output, container, or VM exists when the key is unset."""

    if not model:
        return
    if api_key_env is None:
        provider = model.partition("/")[0]
        api_key_env = _DEFAULT_KEY_ENVS.get(provider)
        if api_key_env is None:
            return  # provider validity is reported by build_model
    if not os.environ.get(api_key_env):
        raise ValueError(
            f"model credential environment variable {api_key_env} is unset or "
            f"empty; export it before running (model {model!r})"
        )


def _header_identities(
    headers: Mapping[str, str] | None,
) -> list[Mapping[str, str]]:
    """Bind non-secret provider header values without persisting their contents."""

    if headers is None:
        return []
    return sorted(
        (
            {
                "name": name.casefold(),
                "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
            for name, value in headers.items()
        ),
        key=lambda item: item["name"],
    )


def _sha256_file(path: Path) -> str:
    return stable_file_sha256(path, label="asset")


def _secrets(args: argparse.Namespace) -> tuple[str, ...]:
    names = [
        args.api_key_env,
        getattr(args, "grader_api_key_env", None),
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_KEY",
        "SERPAPI_API_KEY",
        getattr(args, "client_password_env", None),
        getattr(args, "env_token_env", None),
    ]
    return tuple(
        value for name in names if name for value in [os.environ.get(name)] if value
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _history_images(value: str) -> int | str:
    if value.strip().casefold() == "unlimited":
        return "unlimited"
    return _nonnegative_int(value)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


__all__ = ["build_parser", "main"]
