"""JSON-stdin bridge into a verified Browser-Use source checkout.

This module deliberately imports no Scaffold Lab code.  The parent adapter launches
it with an isolated Python interpreter, prepends the verified checkout to
``sys.path``, and receives exactly one marker-prefixed JSON result.  Browser-Use owns
the Agent policy, prompts, Browser runtime, action loop, history, and model calls.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RESULT_MARKER = "__SCAFFOLDLAB_BROWSER_USE_RESULT_V1__="

LLM_CLASSES = {
    "anthropic": "ChatAnthropic",
    "azure-openai": "ChatAzureOpenAI",
    "browser-use": "ChatBrowserUse",
    "google": "ChatGoogle",
    "groq": "ChatGroq",
    "litellm": "ChatLiteLLM",
    "mistral": "ChatMistral",
    "oci-raw": "ChatOCIRaw",
    "ollama": "ChatOllama",
    "openai": "ChatOpenAI",
    "vercel": "ChatVercel",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): child for key, child in value.items()}


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sys.stdout.write(RESULT_MARKER + serialized + "\n")
    sys.stdout.flush()


def _load_payload() -> dict[str, Any]:
    payload = _object(json.load(sys.stdin), "runner payload")
    if payload.get("schema_version") != 1:
        raise ValueError("runner payload schema_version must equal 1")
    return payload


def _source_file(value: Any, checkout: Path, label: str) -> str:
    try:
        candidate = inspect.getsourcefile(value) or inspect.getfile(value)
    except (TypeError, OSError) as exc:
        raise RuntimeError(f"could not resolve {label} source file") from exc
    path = Path(_string(candidate, f"{label} source file")).resolve()
    if not path.is_relative_to(checkout):
        raise RuntimeError(f"imported {label} is outside the verified checkout")
    return str(path)


def _model_object(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: Any = dict(value)
    else:
        model_dump = getattr(value, "model_dump", None)
        if not callable(model_dump):
            raise RuntimeError(f"{label} is not serializable")
        try:
            result = model_dump(mode="json")
        except TypeError:
            result = model_dump()
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{label} serialization returned a non-object")
    # Round-trip here so the marker can never fail after the upstream session has run.
    try:
        copied = json.loads(json.dumps(dict(result), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} contains non-JSON values") from exc
    if not isinstance(copied, dict):
        raise RuntimeError(f"{label} serialization returned a non-object")
    return copied


async def _usage_summary(history: Any, agent: Any) -> dict[str, Any] | None:
    summary = getattr(history, "usage", None) if history is not None else None
    if summary is None and agent is not None:
        token_service = getattr(agent, "token_cost_service", None)
        get_summary = getattr(token_service, "get_usage_summary", None)
        if callable(get_summary):
            summary = get_summary()
            if inspect.isawaitable(summary):
                summary = await summary
    return _model_object(summary, "Browser-Use usage summary")


def _history_value(history: Any, method_name: str) -> Any:
    method = getattr(history, method_name, None)
    if not callable(method):
        return None
    return method()


async def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(_string(payload.get("checkout"), "checkout")).resolve()
    if not checkout.is_dir():
        raise ValueError("checkout is not a directory")

    # The parent also supplies -B and these environment settings.  Keep the guards
    # for direct runner use and prevent telemetry/version checks from adding hidden
    # network work outside the requested Browser-Use task.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["BROWSER_USE_SETUP_LOGGING"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.environ["BROWSER_USE_CLOUD_SYNC"] = "false"
    os.environ["BROWSER_USE_VERSION_CHECK"] = "false"
    os.environ["BROWSER_USE_CALCULATE_COST"] = "false"
    sys.path.insert(0, str(checkout))

    browser_use = importlib.import_module("browser_use")
    module_file = Path(
        _string(getattr(browser_use, "__file__", None), "browser_use.__file__")
    ).resolve()
    if not module_file.is_relative_to(checkout):
        raise RuntimeError(
            "imported browser_use package is outside the verified checkout"
        )

    agent_type = getattr(browser_use, "Agent", None)
    browser_type = getattr(browser_use, "Browser", None)
    if not callable(agent_type) or not callable(browser_type):
        raise RuntimeError("verified checkout must export callable Agent and Browser")

    provider = _string(payload.get("provider"), "provider")
    llm_class_name = LLM_CLASSES.get(provider)
    if llm_class_name is None:
        raise ValueError(f"unsupported Browser-Use provider {provider!r}")
    llm_type = getattr(browser_use, llm_class_name, None)
    if not callable(llm_type):
        raise RuntimeError(
            f"verified checkout does not export callable {llm_class_name}"
        )

    source_files = {
        "browser_use": str(module_file),
        "agent": _source_file(agent_type, checkout, "browser_use.Agent"),
        "browser": _source_file(browser_type, checkout, "browser_use.Browser"),
        "llm": _source_file(llm_type, checkout, f"browser_use.{llm_class_name}"),
    }

    llm_kwargs = _object(payload.get("llm_kwargs", {}), "llm_kwargs")
    if "model" in llm_kwargs:
        raise ValueError("set model explicitly instead of llm_kwargs.model")
    llm = llm_type(model=_string(payload.get("model"), "model"), **llm_kwargs)

    browser_kwargs = _object(payload.get("browser_kwargs", {}), "browser_kwargs")
    if browser_kwargs.get("keep_alive") is True:
        raise ValueError("Browser-Use upstream sessions require keep_alive=false")
    browser_kwargs.setdefault("headless", True)
    browser_kwargs["keep_alive"] = False
    browser = browser_type(**browser_kwargs)

    agent_kwargs = _object(payload.get("agent_kwargs", {}), "agent_kwargs")
    reserved_agent_keys = {
        "browser",
        "browser_session",
        "calculate_cost",
        "enable_signal_handler",
        "extend_system_message",
        "llm",
        "override_system_message",
        "task",
    }
    reserved = reserved_agent_keys & agent_kwargs.keys()
    if reserved:
        raise ValueError(
            f"agent_kwargs cannot override adapter-owned fields: {sorted(reserved)}"
        )
    system_extension = payload.get("system_extension")
    if system_extension is not None and not isinstance(system_extension, str):
        raise ValueError("system_extension must be a string or null")
    if system_extension:
        agent_kwargs["extend_system_message"] = system_extension
    # Cost is kept unknown rather than fetching mutable pricing data.  Token usage is
    # still captured by Browser-Use's TokenCost wrapper.
    agent_kwargs["calculate_cost"] = False
    agent_kwargs["enable_signal_handler"] = False
    task_id = payload.get("task_id")
    if task_id is not None:
        agent_kwargs["task_id"] = _string(task_id, "task_id")

    agent = agent_type(
        task=_string(payload.get("task"), "task"),
        llm=llm,
        browser=browser,
        **agent_kwargs,
    )
    max_steps = _positive_int(payload.get("max_steps"), "max_steps")
    started = time.perf_counter()
    try:
        history = await agent.run(max_steps=max_steps)
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "usage_summary": await _usage_summary(None, agent),
            "execution_time_seconds": time.perf_counter() - started,
            "provider": provider,
            "model": payload.get("model"),
            "llm_class": llm_class_name,
            "source_files": source_files,
            "cost_tracking_enabled": False,
        }

    answer = _history_value(history, "final_result")
    if not isinstance(answer, str) or not answer:
        raise RuntimeError("Browser-Use history contained no final result text")
    done = _history_value(history, "is_done")
    successful = _history_value(history, "is_successful")
    steps = _history_value(history, "number_of_steps")
    if not isinstance(done, bool):
        done = None
    if successful is not None and not isinstance(successful, bool):
        successful = None
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        try:
            steps = len(history)
        except (TypeError, AttributeError):
            steps = None

    return {
        "schema_version": 1,
        "ok": True,
        "response": answer,
        "usage_summary": await _usage_summary(history, agent),
        "history": {
            "steps": steps,
            "is_done": done,
            "is_successful": successful,
        },
        "execution_time_seconds": time.perf_counter() - started,
        "provider": provider,
        "model": payload.get("model"),
        "llm_class": llm_class_name,
        "source_files": source_files,
        "browser_effective_options": {
            "headless": browser_kwargs.get("headless"),
            "keep_alive": False,
        },
        "agent_effective_options": {
            "calculate_cost": False,
            "enable_signal_handler": False,
            "max_steps": max_steps,
        },
        "cost_tracking_enabled": False,
        "python_version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ],
    }


async def _main() -> int:
    try:
        result = await _run(_load_payload())
    except Exception as exc:
        result = {
            "schema_version": 1,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    _emit(result)
    return 0 if result.get("ok") is True else 1


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
