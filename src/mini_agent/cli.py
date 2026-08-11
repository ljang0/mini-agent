from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .providers import (
    AnthropicMessagesBackend,
    OpenAICompatibleChatBackend,
    OpenAIResponsesBackend,
)

from .agent import MiniAgent
from .environments import (
    BashEnvironment,
    BrowseCompPlusBackend,
    CUASpeedRunClient,
    directory_identity,
    JsonlSearchBackend,
    WebEnvironment,
)
from .models import BackendModel
from .profiles import Profile, load_profile
from .runtime import RunContext
from .types import BudgetLimits


def _selected_provider(
    requested: str | None, profile: Profile, model: str
) -> tuple[str, str]:
    provider = requested or profile.provider
    model_id = model
    prefixes = {
        "openai/": "openai-responses",
        "anthropic/": "anthropic-messages",
    }
    for prefix, inferred in prefixes.items():
        if model.startswith(prefix):
            model_id = model[len(prefix) :]
            if provider and provider != inferred:
                raise ValueError(
                    f"model prefix {prefix[:-1]!r} conflicts with provider {provider!r}"
                )
            provider = inferred
            break
    if not provider:
        raise ValueError(
            "select --provider or use an openai/... or anthropic/... model name"
        )
    return provider, model_id


def _resolved_manifest(
    profile: Profile,
    *,
    model: str = "",
    provider: str | None = None,
) -> dict[str, Any]:
    selected_model = model or profile.model
    selected_provider = provider or profile.provider
    if selected_model and (
        selected_provider
        or selected_model.startswith("openai/")
        or selected_model.startswith("anthropic/")
    ):
        selected_provider, _ = _selected_provider(provider, profile, selected_model)
    return profile.manifest(
        selected_model=selected_model or None,
        selected_provider=selected_provider or None,
    )


def _build_backend(args: argparse.Namespace, profile: Profile) -> Any:
    provider, model = _selected_provider(args.provider, profile, args.model)
    extra_body = dict(profile.generation)
    extra_body.pop("max_output_tokens", None)
    if provider == "openai-responses":
        return OpenAIResponsesBackend(
            model=model,
            api_key_env=args.api_key_env or "OPENAI_API_KEY",
            base_url=args.base_url or "https://api.openai.com/v1",
            extra_body=extra_body,
        )
    if provider == "anthropic-messages":
        return AnthropicMessagesBackend(
            model=model,
            api_key_env=args.api_key_env or "ANTHROPIC_API_KEY",
            base_url=args.base_url or "https://api.anthropic.com/v1",
            extra_body=extra_body,
        )
    if provider == "openai-compatible-chat":
        if not args.base_url:
            raise ValueError("openai-compatible-chat requires --base-url")
        return OpenAICompatibleChatBackend(
            model=model,
            api_key_env=args.api_key_env or "MODEL_API_KEY",
            base_url=args.base_url,
            extra_body=extra_body,
        )
    raise ValueError(f"unsupported minimal provider {provider!r}")


def _build_limits(profile: Profile) -> tuple[BudgetLimits, int]:
    raw = dict(profile.limits)
    max_steps = raw.pop("max_steps", raw.get("max_model_calls", 64))
    if "max_tokens" in raw:
        if "max_output_tokens" in raw:
            raise ValueError("profile cannot set both max_tokens and max_output_tokens")
        raw["max_output_tokens"] = raw.pop("max_tokens")
    try:
        limits = BudgetLimits(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid profile limits: {exc}") from exc
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("profile limits.max_steps must be a positive integer")
    return limits, max_steps


def _build_model(
    args: argparse.Namespace, profile: Profile, *, agent_id: str = "/root"
) -> Any:
    configured_output = profile.generation.get("max_output_tokens")
    if configured_output is not None and (
        not isinstance(configured_output, int)
        or isinstance(configured_output, bool)
        or configured_output < 1
    ):
        raise ValueError("profile generation.max_output_tokens must be positive")
    history = dict(profile.history)
    if history.get("mode", "linear") != "linear":
        raise ValueError("history.mode must be linear")
    images_to_keep = history.get("images_to_keep")
    if images_to_keep is not None and not (
        isinstance(images_to_keep, int)
        and not isinstance(images_to_keep, bool)
        and images_to_keep >= 0
    ):
        raise ValueError("profile history.images_to_keep must be non-negative")
    if images_to_keep is not None and profile.provider != "anthropic-messages":
        raise ValueError(
            "history.images_to_keep is currently supported only by anthropic-messages"
        )
    image_removal_chunk = history.get("image_removal_chunk", 1)
    if not (
        isinstance(image_removal_chunk, int)
        and not isinstance(image_removal_chunk, bool)
        and image_removal_chunk >= 1
    ):
        raise ValueError("profile history.image_removal_chunk must be positive")
    backend = _build_backend(args, profile)
    metadata = {
        "images_to_keep": images_to_keep,
        "image_removal_chunk": image_removal_chunk,
    }
    if profile.response_parser in {
        "mini_swe_text",
        "mini_swe_backticks",
        "mini_swe_xml",
    }:
        from .evals.swebench import MiniSWETextActionModel

        return MiniSWETextActionModel(
            response_parser=profile.response_parser,
            backend=backend,
            max_output_tokens=configured_output,
            metadata=metadata,
            agent_id=agent_id,
        )
    if profile.application == "web":
        from .web_models import build_web_model

        return build_web_model(
            backend,
            response_parser=profile.response_parser,
            max_output_tokens=configured_output,
            metadata=metadata,
            agent_id=agent_id,
        )
    if profile.response_parser != "provider_tool_calls":
        raise ValueError(
            f"response parser {profile.response_parser!r} requires a registered "
            "downstream model adapter"
        )
    return BackendModel(
        backend,
        agent_id=agent_id,
        max_output_tokens=configured_output,
        metadata=metadata,
    )


async def _build_environment(args: argparse.Namespace, profile: Profile) -> Any:
    environment: Any
    if args.application == "swe":
        if args.workspace is None:
            raise ValueError("SWE runs require --workspace")
        environment = await BashEnvironment.isolated(args.workspace)
    elif args.application == "web":
        if bool(args.corpus) == bool(args.index_path):
            raise ValueError("web runs require exactly one of --corpus or --index-path")
        backend: Any = (
            JsonlSearchBackend(args.corpus)
            if args.corpus is not None
            else BrowseCompPlusBackend(args.index_path)
        )
        tokenizer = None
        tokenizer_identity: Mapping[str, Any] | None = None
        snippet_tokens = profile.observation.get("snippet_tokens")
        if snippet_tokens is not None:
            tokenizer_path = getattr(args, "tokenizer_path", None)
            if tokenizer_path is None:
                raise ValueError(
                    "token-based BrowseComp profiles require --tokenizer-path to a "
                    "resolved local snapshot"
                )
            try:
                from transformers import AutoTokenizer  # type: ignore[import]
            except ImportError as exc:
                raise ValueError(
                    "token-based BrowseComp profiles require the web extra"
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path.expanduser().resolve()), local_files_only=True
            )
            tokenizer_identity = directory_identity(tokenizer_path)
        environment = WebEnvironment.from_policy(
            backend,
            benchmark=profile.benchmark,
            observation=profile.observation,
            tools=profile.tools,
            tokenizer=tokenizer,
            tokenizer_identity=tokenizer_identity,
        )
    else:
        if not args.env_url:
            raise ValueError("CUA runs require --env-url")
        from .integrations.cua_speed_run import build_profile_environment

        environment = build_profile_environment(
            profile, CUASpeedRunClient(args.env_url)
        )
    available = {tool.name for tool in environment.tools()}
    requested = set(profile.tools)
    if available != requested:
        await environment.close()
        raise ValueError(
            f"profile tools {sorted(requested)} do not match environment tools "
            f"{sorted(available)}"
        )
    return environment


def _task_text(args: argparse.Namespace) -> str:
    if args.task_file is not None:
        return args.task_file.read_text(encoding="utf-8")
    return args.task


def _write_run(
    output: Path,
    payload: Mapping[str, Any],
    trace: Sequence[Any],
    *,
    patch_bytes: bytes | None = None,
) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / "run.json").write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (staging / "trace.jsonl").write_text(
            "".join(
                json.dumps(asdict(event), sort_keys=True, default=str) + "\n"
                for event in trace
            ),
            encoding="utf-8",
        )
        if patch_bytes is not None:
            artifacts = staging / "artifacts"
            artifacts.mkdir()
            (artifacts / "patch.diff").write_bytes(patch_bytes)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


async def _run(args: argparse.Namespace) -> int:
    profile = load_profile(args.application, args.profile)
    limits, configured_steps = _build_limits(profile)
    context = RunContext(limits, capture_content=args.capture_content)
    task_text = _task_text(args)
    manifest = _resolved_manifest(
        profile, model=args.model, provider=args.provider
    )
    manifest["mode"] = args.mode
    manifest["task"] = {
        "chars": len(task_text),
        "sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
    }
    patch_bytes: bytes | None = None
    if args.mode == "single":
        model = _build_model(args, profile)
        manifest["provider_runtime"] = dict(model.provenance())
        environment: Any = None
        failed = False
        try:
            environment = await _build_environment(args, profile)
            provenance = getattr(environment, "provenance", None)
            manifest["environment"] = (
                dict(provenance()) if provenance is not None else {}
            )
            agent = MiniAgent(
                model=model,
                environment=environment,
                system_prompt=profile.system_prompt,
                max_steps=args.max_steps or configured_steps,
                context=context,
            )
            result = await agent.run(task_text)
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_errors: list[BaseException] = []
            if environment is not None:
                export_patch = getattr(environment, "export_patch", None)
                if export_patch is not None:
                    try:
                        patch_bytes = await export_patch()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    await environment.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors and not failed:
                raise RuntimeError(
                    "; ".join(
                        f"{type(error).__name__}: {error}" for error in cleanup_errors
                    )
                ) from cleanup_errors[0]
    else:
        from .orchestrator import AdvisoryEnvironment, Orchestrator

        root_patch_environment: Any = None

        async def environment_factory(
            agent_id: str, selected_profile: str | None
        ) -> Any:
            nonlocal root_patch_environment
            if args.application == "cua" and agent_id != "/root":
                return AdvisoryEnvironment(f"cua-advisory:{agent_id}")
            resolved = (
                load_profile(args.application, selected_profile)
                if selected_profile is not None
                else profile
            )
            environment = await _build_environment(args, resolved)
            if args.application == "swe" and agent_id == "/root":
                # Orchestrator owns environment cleanup, so capture the root
                # workspace patch from its close hook before that cleanup.
                from .evals.swebench import _RootPatchCaptureEnvironment

                root_patch_environment = _RootPatchCaptureEnvironment(
                    environment, capture_patch=True
                )
                return root_patch_environment
            return environment

        def agent_builder(
            agent_id: str,
            environment: Any,
            shared_context: RunContext,
            selected_profile: str | None,
        ) -> MiniAgent:
            resolved = (
                load_profile(args.application, selected_profile)
                if selected_profile is not None
                else profile
            )
            selected_limits, steps = _build_limits(resolved)
            shared_context.configure_agent(agent_id, selected_limits)
            return MiniAgent(
                model=_build_model(args, resolved, agent_id=agent_id),
                environment=environment,
                system_prompt=resolved.system_prompt,
                max_steps=args.max_steps or steps,
                context=shared_context,
                agent_id=agent_id,
            )

        orchestrator = Orchestrator(
            agent_builder=agent_builder,
            environment_factory=environment_factory,
            context=context,
            max_agents=args.max_agents,
            allowed_child_profiles=tuple(args.child_profile),
            per_agent_limits=None,
        )
        result = await orchestrator.run(task_text, profile=None)
        if root_patch_environment is not None:
            patch_bytes = root_patch_environment.patch
        manifest["multi_agent"] = {
            "max_agents": args.max_agents,
            "allowed_child_profiles": list(args.child_profile),
            "cua_root_only_control": args.application == "cua",
        }
    payload = {
        "answer": result.answer,
        "status": result.status,
        "steps": result.steps,
        "usage": asdict(context.ledger.usage),
        "model_calls": context.ledger.calls,
        "tool_calls": context.ledger.tool_calls,
        "tool_output_bytes": context.ledger.tool_output_bytes,
        "manifest": manifest,
    }
    if patch_bytes is not None:
        payload["patch"] = {
            "bytes": len(patch_bytes),
            "sha256": hashlib.sha256(patch_bytes).hexdigest(),
        }
    if args.output is not None:
        _write_run(
            args.output,
            payload,
            context.trace.events,
            patch_bytes=patch_bytes,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _show_profile(args: argparse.Namespace) -> int:
    profile = load_profile(args.application, args.profile)
    print(
        json.dumps(
            _resolved_manifest(
                profile, model=args.model or "", provider=args.provider
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


async def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        profile = load_profile(args.application, args.profile)
    except ValueError as exc:
        check("profile", False, str(exc))
        profile = None
    else:
        assert profile is not None
        check(
            "profile",
            True,
            _resolved_manifest(
                profile, model=args.model or "", provider=args.provider
            ),
        )
    runtime_provider = ""
    if profile is not None:
        selected_model = args.model or profile.model
        try:
            if selected_model:
                runtime_provider, _ = _selected_provider(
                    args.provider, profile, selected_model
                )
            else:
                runtime_provider = args.provider or profile.provider
        except ValueError as exc:
            check("provider", False, str(exc))
        else:
            if runtime_provider:
                check("provider", True, runtime_provider)
    if runtime_provider:
        default_keys = {
            "openai-responses": "OPENAI_API_KEY",
            "anthropic-messages": "ANTHROPIC_API_KEY",
            "openai-compatible-chat": "MODEL_API_KEY",
        }
        key_name = args.api_key_env or default_keys.get(runtime_provider)
        if key_name is not None:
            check(
                "credentials",
                bool(os.environ.get(key_name)),
                {"environment_variable": key_name, "value_exposed": False},
            )
        if runtime_provider == "openai-compatible-chat":
            from urllib.parse import urlsplit

            parsed = urlsplit(args.base_url or "")
            check(
                "base_url",
                parsed.scheme in {"http", "https"} and bool(parsed.netloc),
                f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "missing",
            )
    if args.application == "swe":
        from .environments.swebench import swebench_doctor

        swe_report = await swebench_doctor(image=args.image)
        checks.extend(asdict(item) for item in swe_report.checks)
    elif args.application == "web":
        selected = args.index_path or args.corpus
        check("data", selected is not None and selected.exists(), str(selected or ""))
        if args.index_path is not None:
            check(
                "pyserini",
                importlib.util.find_spec("pyserini") is not None,
                "required for the canonical Lucene backend",
            )
            check("java", shutil.which("java") is not None, shutil.which("java") or "")
        if profile is not None and profile.observation.get("snippet_tokens") is not None:
            tokenizer = args.tokenizer_path
            check(
                "tokenizer_snapshot",
                tokenizer is not None and tokenizer.is_dir(),
                str(tokenizer or ""),
            )
    else:
        from .integrations.cua_speed_run import preflight, validate_env_url

        cua_report = preflight(submission=args.submission, benchmark=args.benchmark)
        cua_checks = cua_report.get("checks")
        if not isinstance(cua_checks, list) or not all(
            isinstance(item, Mapping) for item in cua_checks
        ):
            raise ValueError("CUA preflight returned invalid checks")
        checks.extend(dict(item) for item in cua_checks)
        if args.env_url:
            try:
                validated_url = validate_env_url(args.env_url)
            except ValueError as exc:
                check("gateway_url", False, str(exc))
            else:
                client = CUASpeedRunClient(
                    validated_url, timeout_seconds=5, connect_retries=0
                )
                try:
                    observation = await client.observe()
                except Exception as exc:
                    check(
                        "gateway_observe",
                        False,
                        {"error_type": type(exc).__name__, "url_exposed": False},
                    )
                else:
                    check(
                        "gateway_observe",
                        True,
                        {
                            **dict(client.provenance()),
                            "screenshot_bytes": len(observation.png),
                        },
                    )
                finally:
                    await client.close()
        else:
            benchmark_run = (
                args.submission is not None
                and args.benchmark is not None
                and args.checkout is not None
            )
            check(
                "execution_target",
                benchmark_run,
                (
                    "pinned checkout, benchmark submission, and benchmark provided"
                    if benchmark_run
                    else "provide --env-url for one task or --checkout, --submission, "
                    "and --benchmark for a benchmark run"
                ),
            )
            if benchmark_run:
                from .evals.cua import (
                    _python_from_console_script,
                    resolve_cua_speed_run_executable,
                    verify_cua_speed_run_checkout,
                )

                try:
                    revision = await verify_cua_speed_run_checkout(args.checkout)
                except (ValueError, RuntimeError) as exc:
                    check("checkout_revision", False, str(exc))
                else:
                    check("checkout_revision", True, revision)
                    try:
                        executable = resolve_cua_speed_run_executable(
                            args.checkout, args.cua_executable
                        )
                        interpreter = _python_from_console_script(
                            args.checkout, executable
                        )
                        source_cli = (
                            args.checkout.expanduser().resolve()
                            / "src"
                            / "cua_speedrun"
                            / "cli.py"
                        )
                        if not source_cli.is_file():
                            raise ValueError(
                                f"pinned cua-speed-run CLI source is missing: {source_cli}"
                            )
                    except ValueError as exc:
                        check("runner_executable", False, str(exc))
                    else:
                        check(
                            "runner_executable",
                            True,
                            {
                                "console_script": str(executable),
                                "interpreter": str(interpreter),
                                "source_cli": str(source_cli),
                            },
                        )
    ok = all(bool(item["ok"]) for item in checks)
    payload = {
        "schema": "mini-agent-doctor-v1",
        "application": args.application,
        "status": "ready" if ok else "blocked",
        "ok": ok,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if ok else 1


def _export(args: argparse.Namespace) -> int:
    if args.target != "cua-speed-run":
        raise ValueError("supported export target is cua-speed-run")
    from .integrations.cua_speed_run import export_submission

    profile = load_profile("cua", args.profile)
    provider = args.provider or profile.provider
    model = args.model or profile.model
    if not provider or not model:
        raise ValueError("CUA export requires a resolved provider and model")
    for child_profile in args.child_profile:
        load_profile("cua", child_profile)
    exported = export_submission(
        args.output,
        template=args.profile,
        model=model,
        provider=provider,
        required_environment=tuple(args.require_env),
        mode=args.mode,
        max_agents=args.max_agents,
        child_profiles=tuple(args.child_profile),
        wheel=args.wheel,
        dependency_wheels=tuple(args.dependency_wheel),
    )
    print(
        json.dumps(
            {
                "target": args.target,
                "directory": str(exported.directory),
                "template": exported.template,
                "profile": exported.profile,
                "source_revision": exported.source_revision,
                "mode": args.mode,
                "max_agents": args.max_agents if args.mode == "multi" else 1,
                "allowed_child_profiles": list(args.child_profile),
                "runtime_wheel_sha256": exported.runtime_wheel_sha256,
                "dependency_wheel_sha256": dict(
                    exported.dependency_wheel_sha256
                ),
                "files": dict(exported.files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


async def _grade(args: argparse.Namespace) -> int:
    if args.application == "swe":
        from .evals.swebench import run_official_grader

        if args.predictions is None or args.dataset_name is None or args.run_id is None:
            raise ValueError(
                "SWE grading requires --predictions, --dataset-name, and --run-id"
            )
        swe_result = await run_official_grader(
            dataset_name=args.dataset_name,
            predictions_path=args.predictions,
            run_id=args.run_id,
            max_workers=args.max_workers,
            split=args.split,
        )
        payload = {
            "application": "swe",
            "status": "completed" if swe_result.returncode == 0 else "failed",
            "returncode": swe_result.returncode,
            "output": swe_result.text(),
        }
    elif args.application == "web":
        from .environments.swe import LocalProcessRunner
        from .evals.browsecomp_plus import official_evaluator_argv

        required = {
            "checkout": args.checkout,
            "input_dir": args.input_dir,
            "ground_truth": args.ground_truth,
            "eval_dir": args.eval_dir,
            "judge_model": args.judge_model,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"web grading requires {missing}")
        argv = official_evaluator_argv(
            checkout=args.checkout,
            input_dir=args.input_dir,
            ground_truth=args.ground_truth,
            eval_dir=args.eval_dir,
            qrel_evidence=args.qrel_evidence,
            model=args.judge_model,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        web_result = await LocalProcessRunner().run(
            argv,
            cwd=args.checkout,
            timeout_seconds=args.timeout_seconds,
            max_output_bytes=16 * 1024 * 1024,
        )
        payload = {
            "application": "web",
            "status": "completed" if web_result.returncode == 0 else "failed",
            "returncode": web_result.returncode,
            "output": web_result.text(),
        }
    else:
        from .evals.cua import run_cua_speed_run_reference

        required = {
            "checkout": args.checkout,
            "submission": args.submission,
            "benchmark": args.benchmark,
            "eval_dir": args.eval_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"CUA grading requires {missing}")
        cua_result = await run_cua_speed_run_reference(
            source_root=args.checkout,
            executable=args.cua_executable,
            submission=args.submission,
            benchmark=args.benchmark,
            output_root=args.eval_dir,
            task_ids=tuple(args.instance_id),
            parallel_evaluations=args.max_workers,
            timeout_seconds=args.timeout_seconds,
        )
        payload = {
            "application": "cua",
            "status": "completed" if cua_result.returncode == 0 else "failed",
            "returncode": cua_result.returncode,
            "stdout": cua_result.stdout,
            "stderr": cua_result.stderr,
        }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "completed" else 1


def _load_tokenizer(profile: Profile, path: Path | None) -> Any:
    if profile.observation.get("snippet_tokens") is None:
        return None
    if path is None:
        raise ValueError(
            "token-based BrowseComp profiles require --tokenizer-path to a local snapshot"
        )
    try:
        from transformers import AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise ValueError("token-based BrowseComp profiles require the web extra") from exc
    return AutoTokenizer.from_pretrained(
        str(path.expanduser().resolve()), local_files_only=True
    )


def _validate_cua_eval_options(args: argparse.Namespace) -> None:
    """Reject generation options that the exported CUA submission already owns."""

    unsupported: list[str] = []
    values = (
        ("--model", bool(args.model)),
        ("--profile", args.profile != "default"),
        ("--provider", args.provider is not None),
        ("--base-url", args.base_url is not None),
        ("--api-key-env", args.api_key_env is not None),
        ("--tasks", args.tasks is not None),
        ("--resume", args.resume),
        ("--retry-errors", args.retry_errors),
        ("--capture-content", args.capture_content),
        ("--mode", args.mode != "single"),
        ("--max-agents", args.max_agents != 4),
        ("--child-profile", bool(args.child_profile)),
        ("--corpus", args.corpus is not None),
        ("--index-path", args.index_path is not None),
        ("--tokenizer-path", args.tokenizer_path is not None),
        ("--task-format", args.task_format is not None),
    )
    unsupported.extend(option for option, selected in values if selected)
    if unsupported:
        raise ValueError(
            "CUA eval executes an exported submission and does not accept generation "
            f"options {unsupported}; configure them with `mini-agent export`"
        )


async def _eval(args: argparse.Namespace) -> int:
    if args.application == "cua":
        _validate_cua_eval_options(args)
    profile = load_profile(args.application, args.profile)
    if args.application in {"swe", "web"}:
        if not args.model:
            raise ValueError(f"{args.application} eval requires --model")
        if args.tasks is None:
            raise ValueError(f"{args.application} eval requires --tasks")
    limits, max_steps = _build_limits(profile)
    manifest = _resolved_manifest(
        profile, model=args.model or "", provider=args.provider
    )
    manifest["mode"] = args.mode
    if args.mode == "multi":
        manifest["multi_agent"] = {
            "max_agents": args.max_agents,
            "allowed_child_profiles": list(args.child_profile),
            "per_agent_limits": asdict(limits),
        }
    if args.application == "swe":
        from .environments.swebench import DockerSWEEnvironment
        from .evals.swebench import (
            SWEbenchBatchRunner,
            load_swebench_jsonl,
            run_mini_agent_instance,
            run_multi_agent_instance,
            select_swebench_instances,
        )

        instances = select_swebench_instances(
            load_swebench_jsonl(args.tasks), instance_ids=tuple(args.instance_id)
        )

        async def swe_worker(instance: Any) -> Any:
            if args.mode == "multi":
                async def environment_factory(
                    item: Any, agent_id: str, name: str | None
                ) -> Any:
                    del agent_id
                    _ = load_profile("swe", name) if name is not None else profile
                    return await DockerSWEEnvironment.create(item.data)

                def agent_builder(
                    item: Any,
                    agent_id: str,
                    environment: Any,
                    shared_context: RunContext,
                    selected_profile: str | None,
                ) -> MiniAgent:
                    del item
                    resolved = (
                        load_profile("swe", selected_profile)
                        if selected_profile is not None
                        else profile
                    )
                    selected_limits, selected_steps = _build_limits(resolved)
                    shared_context.configure_agent(agent_id, selected_limits)
                    return MiniAgent(
                        model=_build_model(args, resolved, agent_id=agent_id),
                        environment=environment,
                        system_prompt=resolved.system_prompt,
                        max_steps=selected_steps,
                        context=shared_context,
                        agent_id=agent_id,
                    )

                return await run_multi_agent_instance(
                    instance,
                    agent_builder=agent_builder,
                    environment_factory=environment_factory,
                    limits=limits,
                    per_agent_limits=None,
                    max_agents=args.max_agents,
                    allowed_child_profiles=tuple(args.child_profile),
                    capture_content=args.capture_content,
                )
            return await run_mini_agent_instance(
                instance,
                model_factory=lambda _: _build_model(args, profile),
                environment_factory=lambda item: DockerSWEEnvironment.create(item.data),
                system_prompt=profile.system_prompt,
                max_steps=max_steps,
                limits=limits,
                capture_content=args.capture_content,
            )

        swe_summary = await SWEbenchBatchRunner(
            output_dir=args.output,
            model_name_or_path=args.model,
            worker=swe_worker,
            max_workers=args.max_workers,
            manifest=manifest,
        ).run(instances, resume=args.resume, retry_errors=args.retry_errors)
        payload = asdict(swe_summary)
    elif args.application == "web":
        from .evals.browsecomp_plus import (
            BrowseCompBatchRunner,
            load_browsecomp_tasks,
            run_mini_agent_task,
            run_multi_agent_task,
        )

        task_set = load_browsecomp_tasks(args.tasks, source_format=args.task_format)
        selected = tuple(
            task
            for task in task_set.tasks
            if not args.instance_id or task.query_id in set(args.instance_id)
        )
        if not selected:
            raise ValueError("BrowseComp selection is empty")
        if bool(args.corpus) == bool(args.index_path):
            raise ValueError("web eval requires exactly one of --corpus or --index-path")
        backend: Any = (
            JsonlSearchBackend(args.corpus)
            if args.corpus is not None
            else BrowseCompPlusBackend(args.index_path)
        )
        tokenizers: dict[Path, Any] = {
            profile.path: _load_tokenizer(profile, args.tokenizer_path)
        }
        tokenizer_identity = (
            directory_identity(args.tokenizer_path)
            if args.tokenizer_path is not None
            else None
        )

        def web_environment(_: Any, selected_profile: Profile = profile) -> WebEnvironment:
            if selected_profile.path not in tokenizers:
                tokenizers[selected_profile.path] = _load_tokenizer(
                    selected_profile, args.tokenizer_path
                )
            return WebEnvironment.from_policy(
                backend,
                benchmark=selected_profile.benchmark,
                observation=selected_profile.observation,
                tools=selected_profile.tools,
                tokenizer=tokenizers[selected_profile.path],
                tokenizer_identity=tokenizer_identity,
            )

        async def web_worker(task: Any) -> Any:
            if args.mode == "multi":
                def selected_profile(name: str | None) -> Profile:
                    return load_profile("web", name) if name is not None else profile

                return await run_multi_agent_task(
                    task,
                    model_factory=lambda _task, agent_id, name: _build_model(
                        args, selected_profile(name), agent_id=agent_id
                    ),
                    environment_factory=lambda selected_task, agent_id, name: web_environment(
                        selected_task, selected_profile(name)
                    ),
                    system_prompt=lambda name: selected_profile(name).system_prompt,
                    max_steps=lambda name: _build_limits(
                        selected_profile(name)
                    )[1],
                    limits=limits,
                    per_agent_limits=lambda name: _build_limits(
                        selected_profile(name)
                    )[0],
                    max_agents=args.max_agents,
                    allowed_child_profiles=tuple(args.child_profile),
                    capture_content=args.capture_content,
                )
            return await run_mini_agent_task(
                task,
                model_factory=lambda _: _build_model(
                    args, profile, agent_id=f"/browsecomp/{task.query_id}"
                ),
                environment_factory=web_environment,
                system_prompt=profile.system_prompt,
                max_steps=max_steps,
                limits=limits,
                capture_content=args.capture_content,
            )

        manifest["dataset"] = task_set.manifest()
        provenance = getattr(backend, "provenance", None)
        manifest["retrieval_backend"] = (
            dict(provenance()) if provenance is not None else {}
        )
        manifest["tokenizer_snapshot"] = dict(tokenizer_identity or {})
        web_summary = await BrowseCompBatchRunner(
            output_dir=args.output,
            model_name_or_path=args.model,
            worker=web_worker,
            max_workers=args.max_workers,
            manifest=manifest,
        ).run(selected, resume=args.resume, retry_errors=args.retry_errors)
        payload = asdict(web_summary)
    else:
        from .evals.cua import run_cua_speed_run_reference

        required = (args.checkout, args.submission, args.benchmark)
        if any(value is None for value in required):
            raise ValueError(
                "CUA eval delegates to the pinned runner and requires --checkout, "
                "--submission, and --benchmark"
            )
        result = await run_cua_speed_run_reference(
            source_root=args.checkout,
            executable=args.cua_executable,
            submission=args.submission,
            benchmark=args.benchmark,
            output_root=args.output,
            task_ids=tuple(args.instance_id),
            parallel_evaluations=args.max_workers,
            timeout_seconds=args.timeout_seconds,
        )
        payload = {
            **asdict(result),
            "status": (
                "failed"
                if result.returncode != 0 or result.timed_out
                else "completed"
            ),
        }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.application == "cua" and payload.get("status") != "completed":
        return 1
    return 0


def _list_catalog(args: argparse.Namespace) -> int:
    if args.frontiers:
        return _list_frontiers(args)
    from .catalog import (
        get_profile,
        list_gaps,
        list_implementations,
        list_profiles,
        list_studies,
    )

    loaders = {
        "all": list_profiles,
        "implementation": list_implementations,
        "study": list_studies,
        "gap": list_gaps,
    }
    profiles: Sequence[Any]
    if args.name is not None:
        if args.application is None:
            raise ValueError("catalog --name requires --application")
        profiles = (get_profile(args.application, args.name),)
    else:
        profiles = loaders[args.kind](args.application)
    if args.json:
        print(json.dumps([profile.as_dict() for profile in profiles], indent=2))
    else:
        for profile in profiles:
            print(profile.key)
    return 0


def _list_frontiers(args: argparse.Namespace) -> int:
    from .catalog import get_frontier_source, list_frontier_sources

    sources: Sequence[Any] = (
        (get_frontier_source(args.lab),)
        if args.lab is not None
        else list_frontier_sources()
    )
    if args.json:
        payload = []
        for source in sources:
            item = source.as_dict()
            if args.application is not None:
                item["application_statuses"] = [
                    status
                    for status in item["application_statuses"]
                    if status["application"] == args.application
                ]
                item["applications"] = [args.application]
            payload.append(item)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for source in sources:
        statuses = (
            source.application_statuses
            if args.application is None
            else tuple(
                status
                for status in source.application_statuses
                if status.application == args.application
            )
        )
        summary = ", ".join(
            f"{status.application}={status.status}" for status in statuses
        )
        print(f"{source.lab}\t{summary}")
    return 0


def _validate_reference(args: argparse.Namespace) -> int:
    from .references import get_reference

    reference = get_reference(args.application, args.implementation)
    return reference.validate(
        tasks=args.tasks,
        config=args.config,
        provider=args.provider,
    )


def _run_reference(args: argparse.Namespace) -> int:
    from .references import get_reference

    reference = get_reference(args.application, args.implementation)
    extra = tuple(args.runtime_arguments)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return reference.run(
        tasks=args.tasks,
        config=args.config,
        output=args.output,
        provider=args.provider,
        arguments=extra,
    )


def _list_references(args: argparse.Namespace) -> int:
    from .references import list_references

    values = list_references(args.application)
    if args.json:
        print(json.dumps([value.manifest() for value in values], indent=2))
    else:
        for value in values:
            print(value.key)
    return 0


def _add_preserved_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    selector: str,
    output: bool,
) -> None:
    parser.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    parser.add_argument(f"--{selector}", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider")
    if output:
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument(
            "runtime_arguments",
            nargs=argparse.REMAINDER,
            help="runtime-specific arguments after --",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one minimal agent")
    run.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    run.add_argument("--model", required=True)
    run.add_argument("--profile", default="default")
    run.add_argument(
        "--provider",
        choices=("openai-responses", "anthropic-messages", "openai-compatible-chat"),
    )
    task = run.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    run.add_argument("--workspace", type=Path)
    run.add_argument("--corpus", type=Path)
    run.add_argument("--index-path", type=Path)
    run.add_argument("--tokenizer-path", type=Path)
    run.add_argument("--env-url")
    run.add_argument("--base-url")
    run.add_argument("--api-key-env")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--capture-content", action="store_true")
    run.add_argument("--output", type=Path)
    run.add_argument("--mode", choices=("single", "multi"), default="single")
    run.add_argument("--max-agents", type=int, default=4)
    run.add_argument("--child-profile", action="append", default=[])
    run.set_defaults(handler=_run)

    evaluate_native = subparsers.add_parser(
        "eval", help="run a deterministic benchmark generation batch"
    )
    evaluate_native.add_argument(
        "--application", required=True, choices=("swe", "web", "cua")
    )
    evaluate_native.add_argument("--model", default="")
    evaluate_native.add_argument("--profile", default="default")
    evaluate_native.add_argument("--provider")
    evaluate_native.add_argument("--base-url")
    evaluate_native.add_argument("--api-key-env")
    evaluate_native.add_argument("--tasks", type=Path)
    evaluate_native.add_argument("--output", type=Path, required=True)
    evaluate_native.add_argument("--max-workers", type=int, default=1)
    evaluate_native.add_argument("--instance-id", action="append", default=[])
    evaluate_native.add_argument("--resume", action="store_true")
    evaluate_native.add_argument("--retry-errors", action="store_true")
    evaluate_native.add_argument("--capture-content", action="store_true")
    evaluate_native.add_argument("--mode", choices=("single", "multi"), default="single")
    evaluate_native.add_argument("--max-agents", type=int, default=4)
    evaluate_native.add_argument("--child-profile", action="append", default=[])
    evaluate_native.add_argument("--corpus", type=Path)
    evaluate_native.add_argument("--index-path", type=Path)
    evaluate_native.add_argument("--tokenizer-path", type=Path)
    evaluate_native.add_argument("--task-format", choices=("jsonl", "tsv"))
    evaluate_native.add_argument("--checkout", type=Path)
    evaluate_native.add_argument("--submission", type=Path)
    evaluate_native.add_argument("--benchmark", type=Path)
    evaluate_native.add_argument("--cua-executable", default="cua-speedrun")
    evaluate_native.add_argument("--timeout-seconds", type=float, default=14400)
    evaluate_native.set_defaults(handler=_eval)

    grade = subparsers.add_parser("grade", help="invoke a pinned official grader")
    grade.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    grade.add_argument("--predictions", type=Path)
    grade.add_argument("--dataset-name")
    grade.add_argument("--run-id")
    grade.add_argument("--split", default="test")
    grade.add_argument("--max-workers", type=int, default=1)
    grade.add_argument("--checkout", type=Path)
    grade.add_argument("--input-dir", type=Path)
    grade.add_argument("--ground-truth", type=Path)
    grade.add_argument("--eval-dir", type=Path)
    grade.add_argument("--qrel-evidence", type=Path)
    grade.add_argument("--judge-model", type=Path)
    grade.add_argument("--tensor-parallel-size", type=int, default=1)
    grade.add_argument("--submission", type=Path)
    grade.add_argument("--benchmark", type=Path)
    grade.add_argument("--instance-id", action="append", default=[])
    grade.add_argument("--cua-executable", default="cua-speedrun")
    grade.add_argument("--timeout-seconds", type=float, default=14400)
    grade.set_defaults(handler=_grade)

    doctor = subparsers.add_parser("doctor", help="report local run prerequisites")
    doctor.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    doctor.add_argument("--profile", default="default")
    doctor.add_argument("--model", default="")
    doctor.add_argument(
        "--provider",
        choices=("openai-responses", "anthropic-messages", "openai-compatible-chat"),
    )
    doctor.add_argument("--base-url")
    doctor.add_argument("--api-key-env")
    doctor.add_argument("--image")
    doctor.add_argument("--corpus", type=Path)
    doctor.add_argument("--index-path", type=Path)
    doctor.add_argument("--tokenizer-path", type=Path)
    doctor.add_argument("--submission", type=Path)
    doctor.add_argument("--benchmark", type=Path)
    doctor.add_argument("--checkout", type=Path)
    doctor.add_argument("--cua-executable", default="cua-speedrun")
    doctor.add_argument("--env-url")
    doctor.set_defaults(handler=_doctor)

    export = subparsers.add_parser("export", help="package an upstream submission")
    export.add_argument("--target", required=True, choices=("cua-speed-run",))
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--profile", required=True)
    export.add_argument("--model", default="")
    export.add_argument("--provider", default="")
    export.add_argument("--require-env", action="append", default=[])
    export.add_argument("--mode", choices=("single", "multi"), default="single")
    export.add_argument("--max-agents", type=int, default=4)
    export.add_argument("--child-profile", action="append", default=[])
    export.add_argument("--wheel", type=Path)
    export.add_argument("--dependency-wheel", type=Path, action="append", default=[])
    export.set_defaults(handler=_export)

    reference = subparsers.add_parser(
        "reference", help="list, validate, or run a pinned external runtime"
    )
    reference_commands = reference.add_subparsers(dest="reference_command", required=True)
    reference_list = reference_commands.add_parser("list")
    reference_list.add_argument("--application", choices=("swe", "web", "cua"))
    reference_list.add_argument("--json", action="store_true")
    reference_list.set_defaults(handler=_list_references)
    reference_validate = reference_commands.add_parser("validate", allow_abbrev=False)
    _add_preserved_runtime_arguments(
        reference_validate, selector="implementation", output=False
    )
    reference_validate.set_defaults(handler=_validate_reference)
    reference_run = reference_commands.add_parser("run", allow_abbrev=False)
    _add_preserved_runtime_arguments(reference_run, selector="implementation", output=True)
    reference_run.set_defaults(handler=_run_reference)

    show = subparsers.add_parser("profile", help="print a resolved profile manifest")
    show.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    show.add_argument("--profile", default="default")
    show.add_argument("--model", default="")
    show.add_argument("--provider")
    show.set_defaults(handler=_show_profile)

    catalog = subparsers.add_parser(
        "catalog", help="list migrated implementations, studies, and gaps"
    )
    catalog.add_argument("--application", choices=("swe", "web", "cua"))
    catalog.add_argument(
        "--kind",
        choices=("all", "implementation", "study", "gap"),
        default="all",
    )
    catalog.add_argument("--json", action="store_true")
    catalog.add_argument("--name")
    catalog.add_argument("--frontiers", action="store_true")
    catalog.add_argument("--lab")
    catalog.set_defaults(handler=_list_catalog)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        return asyncio.run(result) if asyncio.iscoroutine(result) else int(result)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
