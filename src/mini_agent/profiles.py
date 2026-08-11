from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import]


_PROFILE_KEYS = frozenset(
    {
        "application",
        "model",
        "provider",
        "system_prompt",
        "tools",
        "limits",
        "benchmark",
        "fidelity",
        "source",
        "generation",
        "observation",
        "history",
        "response_parser",
        "fidelity_gaps",
    }
)

_CUA_PROFILE_BY_TEMPLATE = {
    "qwen_vllm": "qwen_vllm",
    "qwen3vl": "qwen3vl",
    "qwen35vl": "qwen35vl",
    "gemini35": "gemini35",
    "meta": "meta",
    "gemini3_flash_preview": "gemini3_flash_preview",
    "glm5v_turbo": "glm5v_turbo",
    "kimi_k3": "kimi_k3",
    "minimax_m3": "minimax_m3",
    "glm_cua": "glm_cua",
    "claude": "claude",
    "gpt54": "gpt54",
    "qwen_vl_remote": "qwen_vl_remote",
}


def _validate_declared_policy(
    *,
    application: str,
    profile_name: str,
    tools: tuple[str, ...],
    benchmark: Mapping[str, Any],
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
    response_parser: str,
) -> None:
    """Reject manifest policy that has no implemented execution meaning."""

    unknown_history = set(history) - {
        "mode",
        "images_to_keep",
        "image_removal_chunk",
    }
    if unknown_history:
        raise ValueError(f"unsupported history fields: {sorted(unknown_history)}")
    if history.get("mode", "linear") != "linear":
        raise ValueError("mini-agent profiles currently support only linear history")

    if application == "swe":
        if tools != ("bash",):
            raise ValueError("SWE profiles must expose exactly the bash tool")
        unknown = set(observation) - {"truncation"}
        if unknown:
            raise ValueError(f"unsupported SWE observation fields: {sorted(unknown)}")
        if observation.get("truncation", "head_tail") != "head_tail":
            raise ValueError("SWE observation truncation must be head_tail")
        allowed_benchmark = {"name", "evaluator", "evaluator_revision", "workdir"}
        unknown = set(benchmark) - allowed_benchmark
        if unknown:
            raise ValueError(f"unsupported SWE benchmark fields: {sorted(unknown)}")
        expected = {
            "name": "swe_bench",
            "evaluator": "swebench.harness.run_evaluation",
            "evaluator_revision": "v4.1.0",
            "workdir": "/testbed",
        }
        for key, value in benchmark.items():
            if value != expected[key]:
                raise ValueError(f"SWE benchmark {key} must be {expected[key]!r}")
        if response_parser not in {
            "provider_tool_calls",
            "mini_swe_text",
            "mini_swe_backticks",
            "mini_swe_xml",
        }:
            raise ValueError(f"unsupported SWE response parser {response_parser!r}")
        return

    if application == "web":
        if tools not in {("search",), ("search", "get_document")}:
            raise ValueError("web tools must be search with optional get_document")
        unknown = set(benchmark) - {"name", "retrieval", "top_k"}
        if unknown:
            raise ValueError(f"unsupported web benchmark fields: {sorted(unknown)}")
        if benchmark.get("name", "browsecomp_plus") != "browsecomp_plus":
            raise ValueError("web benchmark name must be browsecomp_plus")
        if benchmark.get("retrieval", "bm25") != "bm25":
            raise ValueError("web benchmark retrieval must be bm25")
        unknown = set(observation) - {"snippet_chars", "snippet_tokens"}
        if unknown:
            raise ValueError(f"unsupported web observation fields: {sorted(unknown)}")
        return

    if tools != ("computer",):
        raise ValueError("CUA profiles must expose exactly the computer tool")
    if response_parser != "provider_tool_calls":
        raise ValueError("CUA profiles require provider_tool_calls")
    unknown = set(benchmark) - {"name", "template", "tool_protocol", "coordinate_mode"}
    if unknown:
        raise ValueError(f"unsupported CUA benchmark fields: {sorted(unknown)}")
    benchmark_name = benchmark.get("name", "cua_speed_run")
    if benchmark_name not in {"cua_speed_run", "osworld"}:
        raise ValueError("CUA benchmark name must be cua_speed_run or osworld")
    template = benchmark.get("template")
    if template is not None:
        if benchmark_name != "cua_speed_run" or not isinstance(template, str):
            raise ValueError("CUA templates require the cua_speed_run benchmark")
        expected_profile = _CUA_PROFILE_BY_TEMPLATE.get(template)
        if expected_profile is None:
            raise ValueError(f"unknown cua-speed-run template {template!r}")
        if profile_name != expected_profile:
            raise ValueError(
                f"cua-speed-run template {template!r} requires profile "
                f"{expected_profile!r}"
            )


@dataclass(frozen=True)
class Profile:
    path: Path
    application: str
    model: str
    provider: str
    system_prompt: str
    tools: tuple[str, ...]
    limits: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    fidelity: str
    source: Mapping[str, Any]
    generation: Mapping[str, Any]
    observation: Mapping[str, Any]
    history: Mapping[str, Any]
    response_parser: str
    fidelity_gaps: tuple[str, ...]
    raw: Mapping[str, Any]

    def manifest(
        self,
        *,
        selected_model: str | None = None,
        selected_provider: str | None = None,
    ) -> dict[str, Any]:
        prompt_bytes = self.system_prompt.encode("utf-8")
        return {
            "application": self.application,
            "model": selected_model or self.model,
            "provider": (
                self.provider if selected_provider is None else selected_provider
            ),
            "fidelity": self.fidelity,
            "tools": list(self.tools),
            "limits": dict(self.limits),
            "benchmark": dict(self.benchmark),
            "source": dict(self.source),
            "generation": dict(self.generation),
            "observation": dict(self.observation),
            "history": dict(self.history),
            "response_parser": self.response_parser,
            "fidelity_gaps": list(self.fidelity_gaps),
            "profile_path": str(self.path),
            "profile_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "system_prompt": self.system_prompt,
            "system_prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        }


def _builtin_profile(application: str, name: str) -> Path:
    return Path(__file__).parent / "profiles" / application / f"{name}.yaml"


def load_profile(
    application: str, profile: str | Path | None = None
) -> Profile:
    if application not in {"swe", "web", "cua"}:
        raise ValueError("application must be swe, web, or cua")
    if profile is None:
        path = _builtin_profile(application, "default")
    else:
        candidate = Path(profile).expanduser()
        path = candidate if candidate.is_file() else _builtin_profile(application, str(profile))
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"profile does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("profile must be a YAML object")
    unknown = sorted(set(raw).difference(_PROFILE_KEYS))
    if unknown:
        raise ValueError(f"unsupported profile fields: {unknown}")
    selected_application = raw.get("application")
    if selected_application != application:
        raise ValueError(
            f"profile application {selected_application!r} does not match {application!r}"
        )
    model = raw.get("model", "")
    provider = raw.get("provider", "")
    tools = raw.get("tools")
    limits = raw.get("limits", {})
    benchmark = raw.get("benchmark", {})
    fidelity = raw.get("fidelity", "baseline")
    source = raw.get("source", {})
    generation = raw.get("generation", {})
    observation = raw.get("observation", {})
    history = raw.get("history", {})
    response_parser = raw.get("response_parser", "provider_tool_calls")
    fidelity_gaps = raw.get("fidelity_gaps", [])
    prompt_value = raw.get("system_prompt", "")
    if not isinstance(model, str) or not isinstance(provider, str):
        raise ValueError("profile model and provider must be strings")
    if not isinstance(tools, list) or not tools or not all(
        isinstance(tool, str) and tool for tool in tools
    ):
        raise ValueError("profile tools must be a non-empty string list")
    if len(tools) != len(set(tools)):
        raise ValueError("profile tools must be unique")
    if not isinstance(limits, Mapping) or not isinstance(benchmark, Mapping):
        raise ValueError("profile limits and benchmark must be objects")
    if fidelity not in {"baseline", "profile", "reference"}:
        raise ValueError("profile fidelity must be baseline, profile, or reference")
    if not isinstance(source, Mapping):
        raise ValueError("profile source must be an object")
    if not all(
        isinstance(value, Mapping) for value in (generation, observation, history)
    ):
        raise ValueError("profile generation, observation, and history must be objects")
    if not isinstance(response_parser, str) or not response_parser:
        raise ValueError("profile response_parser must be a non-empty string")
    if not isinstance(fidelity_gaps, list) or not all(
        isinstance(gap, str) and gap for gap in fidelity_gaps
    ):
        raise ValueError("profile fidelity_gaps must be a string list")
    if not isinstance(prompt_value, str):
        raise ValueError("profile system_prompt must be a string")
    prompt_path = (path.parent / prompt_value).resolve()
    system_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_value and prompt_path.is_file()
        else prompt_value
    )
    normalized_tools = tuple(tools)
    _validate_declared_policy(
        application=application,
        profile_name=path.stem,
        tools=normalized_tools,
        benchmark=benchmark,
        observation=observation,
        history=history,
        response_parser=response_parser,
    )
    return Profile(
        path=path,
        application=application,
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=normalized_tools,
        limits=dict(limits),
        benchmark=dict(benchmark),
        fidelity=fidelity,
        source=dict(source),
        generation=dict(generation),
        observation=dict(observation),
        history=dict(history),
        response_parser=response_parser,
        fidelity_gaps=tuple(fidelity_gaps),
        raw=dict(raw),
    )


__all__ = ["Profile", "load_profile"]
