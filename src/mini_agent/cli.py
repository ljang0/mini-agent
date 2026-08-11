from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scaffoldlab.providers import (
    AnthropicMessagesBackend,
    OpenAICompatibleChatBackend,
    OpenAIResponsesBackend,
)

from .agent import MiniAgent
from .environments import (
    BashEnvironment,
    BrowseCompPlusBackend,
    CUAEnvironment,
    CUASpeedRunClient,
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
        top_k = profile.benchmark.get("top_k", 5)
        snippet_chars = profile.observation.get("snippet_chars", 4096)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("profile benchmark.top_k must be an integer")
        if not isinstance(snippet_chars, int) or isinstance(snippet_chars, bool):
            raise ValueError("profile observation.snippet_chars must be an integer")
        environment = WebEnvironment(
            backend,
            top_k=top_k,
            snippet_chars=snippet_chars,
            include_get_document="get_document" in profile.tools,
        )
    else:
        if not args.env_url:
            raise ValueError("CUA runs require --env-url")
        environment = CUAEnvironment(CUASpeedRunClient(args.env_url))
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


def _write_run(output: Path, payload: Mapping[str, Any], trace: Sequence[Any]) -> None:
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(asdict(event), sort_keys=True, default=str) + "\n" for event in trace),
        encoding="utf-8",
    )


async def _run(args: argparse.Namespace) -> int:
    profile = load_profile(args.application, args.profile)
    backend = _build_backend(args, profile)
    limits, configured_steps = _build_limits(profile)
    context = RunContext(limits, capture_content=args.capture_content)
    environment = await _build_environment(args, profile)
    configured_output = profile.generation.get("max_output_tokens")
    if configured_output is not None and (
        not isinstance(configured_output, int)
        or isinstance(configured_output, bool)
        or configured_output < 1
    ):
        raise ValueError("profile generation.max_output_tokens must be positive")
    model = BackendModel(backend, max_output_tokens=configured_output)
    task_text = _task_text(args)
    manifest = profile.manifest(selected_model=args.model)
    manifest["task"] = {
        "chars": len(task_text),
        "sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
    }
    manifest["provider_runtime"] = dict(model.provenance())
    provenance = getattr(environment, "provenance", None)
    manifest["environment"] = dict(provenance()) if provenance is not None else {}
    try:
        agent = MiniAgent(
            model=model,
            environment=environment,
            system_prompt=profile.system_prompt,
            max_steps=args.max_steps or configured_steps,
            context=context,
        )
        result = await agent.run(task_text)
    finally:
        await environment.close()
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
    if args.output is not None:
        _write_run(args.output, payload, context.trace.events)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _show_profile(args: argparse.Namespace) -> int:
    profile = load_profile(args.application, args.profile)
    print(
        json.dumps(
            profile.manifest(selected_model=args.model or None),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    run.add_argument("--env-url")
    run.add_argument("--base-url")
    run.add_argument("--api-key-env")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--capture-content", action="store_true")
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=_run)

    show = subparsers.add_parser("profile", help="print a resolved profile manifest")
    show.add_argument("--application", required=True, choices=("swe", "web", "cua"))
    show.add_argument("--profile", default="default")
    show.add_argument("--model", default="")
    show.set_defaults(handler=_show_profile)
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
