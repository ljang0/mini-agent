from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import]


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

    def manifest(self, *, selected_model: str | None = None) -> dict[str, Any]:
        prompt_bytes = self.system_prompt.encode("utf-8")
        return {
            "application": self.application,
            "model": selected_model or self.model,
            "provider": self.provider,
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
    return Profile(
        path=path,
        application=application,
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=tuple(tools),
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
