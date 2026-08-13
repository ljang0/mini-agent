"""Small resolved defaults, not a catalog of copied provider harnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .models import SUPPORTED_PROVIDERS, parse_model_spec, translation_losses_for
from .specs import AgentSpecV1, TranslationReport
from .types import BudgetLimits, _require_bool, _require_str


_PROMPTS = {
    "swe": (
        "You are a software engineering agent. Inspect the repository, reproduce "
        "the issue, make the smallest correct change, and verify it. Use the bash "
        "tool for all workspace interaction. Each call starts a fresh shell, but "
        "filesystem changes persist. Finish with a concise summary and tests run."
    ),
    "web": (
        "You are a research agent. Use the available browser actions to gather "
        "evidence. Triangulate claims, distinguish evidence from inference, and "
        "finish with a direct answer that cites the result references you relied on."
    ),
    "computer": (
        "You are a computer-use agent. Use only the computer tool and native pixel "
        "coordinates from the latest screenshot. Prefer short, verifiable action "
        "batches. Finish only after the requested visible state is present."
    ),
}

_TOOLS = {"swe": "bash", "web": "browser", "computer": "computer"}
_COMMUNICATION_CAPABILITIES = (
    "adopt",
    "inbox",
    "send",
    "spawn",
    "stop",
    "wait",
)
_COMMUNICATION_PROMPT = (
    " You may also use the agent tool to delegate bounded subtasks, exchange "
    "messages, block for inbox delivery, wait for or stop descendant work, or "
    "explicitly adopt supported child state."
)


def _resolved_prompt(prompt: str, *, multi_agent: bool) -> str:
    return prompt + _COMMUNICATION_PROMPT if multi_agent else prompt


@dataclass(frozen=True)
class Profile:
    environment: str
    name: str
    model: str
    system_prompt: str
    max_steps: int
    limits: BudgetLimits

    def as_dict(self) -> Mapping[str, Any]:
        value = asdict(self)
        value["fidelity"] = "minimal_baseline"
        return value

    def to_agent_spec(self, *, multi_agent: bool = False) -> AgentSpecV1:
        """Resolve this maintained profile to the stable provider-neutral spec."""

        _require_bool(multi_agent, "multi_agent")
        tools: tuple[str, ...] = (_TOOLS[self.environment],)
        communication: tuple[str, ...] = ()
        if multi_agent:
            tools = (*tools, "agent")
            communication = _COMMUNICATION_CAPABILITIES
        prompt = _resolved_prompt(self.system_prompt, multi_agent=multi_agent)
        return AgentSpecV1(
            environment=self.environment,
            model=self.model,
            profile=self.name,
            system_prompt=prompt,
            max_steps=self.max_steps,
            budget=self.limits,
            tool_capabilities=tools,
            communication_capabilities=communication,
            fidelity="minimal_baseline",
        )

    def translation_report(
        self, *, multi_agent: bool = False, protocol: str | None = None
    ) -> TranslationReport:
        """Report the declared-field mapping plus the codec's declared losses.

        The profile-to-spec mapping itself is lossless; the losses come from
        the provider codec that ``build_model`` would select for this model
        (``protocol=None`` selects each provider's default protocol).
        """

        provider, _ = parse_model_spec(self.model)
        return TranslationReport(
            source_format="mini-agent.profile/v1",
            spec=self.to_agent_spec(multi_agent=multi_agent),
            losses=tuple(translation_losses_for(provider, protocol)),
        )


def load_profile(
    application: str,
    profile: str = "default",
    *,
    model: str = "openai/test-model",
) -> Profile:
    for value, label in ((application, "application"), (profile, "profile")):
        _require_str(value, label, stripped=True)
    _require_str(model, "model", stripped=True)
    environment = {
        "cua": "computer",
        "computer-use": "computer",
        "browser": "web",
    }.get(application, application)
    if environment not in _PROMPTS:
        raise ValueError(f"unknown environment {application!r}")
    if profile != "default":
        raise ValueError(
            "provider-specific profiles were removed; customize the model adapter "
            "or system prompt downstream"
        )
    try:
        provider, _ = parse_model_spec(model)
    except ValueError:
        provider = ""
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "model must use openai/model, anthropic/model, or meta/model syntax"
        )
    return Profile(
        environment=environment,
        name="default",
        model=model,
        system_prompt=_PROMPTS[environment],
        max_steps=64,
        limits=BudgetLimits(),
    )


def prompt_for(environment: str, *, multi_agent: bool = False) -> str:
    _require_bool(multi_agent, "multi_agent")
    profile = load_profile(environment)
    return _resolved_prompt(profile.system_prompt, multi_agent=multi_agent)


__all__ = ["Profile", "load_profile", "prompt_for"]
