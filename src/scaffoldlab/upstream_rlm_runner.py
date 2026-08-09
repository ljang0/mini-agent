"""Minimal JSON-stdin bridge into an exact upstream ``rlms`` checkout.

This file intentionally imports no Scaffold Lab modules. It is launched with an
explicit Python interpreter in an isolated external process, prepends the verified
checkout to ``sys.path``, executes one public ``RLM.completion`` call, and emits one
marker-prefixed JSON result.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RESULT_MARKER = "__SCAFFOLDLAB_RLM_RESULT_V1__="


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): child for key, child in value.items()}


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    sys.stdout.write(RESULT_MARKER + serialized + "\n")
    sys.stdout.flush()


def _load_payload() -> dict[str, Any]:
    value = json.load(sys.stdin)
    payload = _object(value, "runner payload")
    if payload.get("schema_version") != 1:
        raise ValueError("runner payload schema_version must equal 1")
    return payload


def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(_string(payload.get("checkout"), "checkout")).resolve()
    if not checkout.is_dir():
        raise ValueError("checkout is not a directory")

    # -B is also supplied by the parent process; keep this guard for direct use.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(checkout))
    rlm_module = importlib.import_module("rlm")
    source_file = Path(
        _string(getattr(rlm_module, "__file__", None), "rlm.__file__")
    ).resolve()
    if not source_file.is_relative_to(checkout):
        raise RuntimeError("imported rlm package is outside the verified checkout")
    rlm_type = getattr(rlm_module, "RLM", None)
    if not callable(rlm_type):
        raise RuntimeError("verified checkout does not export callable rlm.RLM")

    provider = _string(payload.get("provider"), "provider")
    model = _string(payload.get("model"), "model")
    environment = _string(payload.get("environment"), "environment")
    backend_kwargs = _object(payload.get("backend_kwargs", {}), "backend_kwargs")
    environment_kwargs = _object(
        payload.get("environment_kwargs", {}), "environment_kwargs"
    )
    limits = _object(payload.get("limits", {}), "limits")
    backend_kwargs["model_name"] = model

    # Two v0.1.3 clients require an explicit constructor key rather than reading
    # it themselves. The parent allowlists exactly which environment values cross.
    provider_key_names = {
        "anthropic": "ANTHROPIC_API_KEY",
        "portkey": "PORTKEY_API_KEY",
    }
    provider_key_name = provider_key_names.get(provider)
    if provider_key_name and "api_key" not in backend_kwargs:
        provider_key = os.environ.get(provider_key_name)
        if provider_key:
            backend_kwargs["api_key"] = provider_key

    instance = rlm_type(
        backend=provider,
        backend_kwargs=backend_kwargs,
        environment=environment,
        environment_kwargs=environment_kwargs,
        max_depth=limits["max_depth"],
        max_iterations=limits["max_iterations"],
        max_budget=limits.get("max_budget_usd"),
        max_timeout=limits.get("max_timeout_seconds"),
        max_tokens=limits.get("max_tokens"),
        max_errors=limits.get("max_errors"),
        max_concurrent_subcalls=limits["max_concurrent_subcalls"],
        verbose=False,
        persistent=False,
        logger=None,
        custom_tools=None,
        custom_sub_tools=None,
    )
    try:
        root_prompt = payload.get("root_prompt")
        if root_prompt is not None and not isinstance(root_prompt, str):
            raise ValueError("root_prompt must be a string or null")
        completion = instance.completion(
            _string(payload.get("context"), "context"),
            root_prompt=root_prompt,
        )
        usage_summary = completion.usage_summary.to_dict()
        if not isinstance(usage_summary, Mapping):
            raise RuntimeError("upstream UsageSummary.to_dict() returned non-object")
        response = completion.response
        if not isinstance(response, str) or not response:
            raise RuntimeError("upstream completion returned no response text")
        completion_error = getattr(completion, "error", None)
        if completion_error is not None:
            raise RuntimeError("upstream completion carried an error")
        return {
            "schema_version": 1,
            "ok": True,
            "response": response,
            "root_model": getattr(completion, "root_model", model),
            "execution_time_seconds": getattr(completion, "execution_time", None),
            "usage_summary": dict(usage_summary),
            "environment": environment,
            "rlm_source_file": str(source_file),
            "python_version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        }
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            close()


def main() -> int:
    try:
        result = _run(_load_payload())
    except Exception as exc:
        _emit(
            {
                "schema_version": 1,
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        return 1
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
