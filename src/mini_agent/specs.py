"""Versioned, provider-neutral agent configuration contracts.

The spec describes inputs to a mini-agent runtime.  It does not claim that two
provider harnesses, policies, or benchmark executions are behaviorally
equivalent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Tuple

from .types import BudgetLimits, ToolDefinition, strict_json_loads

if TYPE_CHECKING:
    from .agent import MiniAgent
    from .environments.base import AgentEnvironment
    from .models import Model
    from .runtime import RunContext


AGENT_SPEC_SCHEMA = "mini-agent.agent-spec/v1"
TRANSLATION_REPORT_SCHEMA = "mini-agent.translation-report/v1"
TRANSLATION_CLAIM_SCOPE = "declared_fields_only"

_CAPABILITY = re.compile(r"^[a-z][a-z0-9_.-]*$")
_LOSS_KINDS = frozenset({"approximated", "dropped", "unsupported"})


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if value != value.strip() or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier} without surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8 text") from exc
    return value


def _capabilities(values: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    normalized: list[str] = []
    for value in values:
        item = _text(value, f"{label} item")
        if _CAPABILITY.fullmatch(item) is None:
            raise ValueError(f"invalid {label} item {item!r}")
        normalized.append(item)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(normalized))


def _budget_dict(limits: BudgetLimits) -> dict[str, Any]:
    # Normalize numeric aliases so 900 and 900.0 have the same identity.
    max_cost = limits.max_cost_usd
    normalized_cost = None if max_cost is None else float(max_cost)
    return {
        "max_concurrency": limits.max_concurrency,
        "max_cost_usd": normalized_cost,
        "max_input_tokens": limits.max_input_tokens,
        "max_model_calls": limits.max_model_calls,
        "max_output_tokens": limits.max_output_tokens,
        "max_tool_calls": limits.max_tool_calls,
        "max_tool_output_bytes": limits.max_tool_output_bytes,
        "wall_time_seconds": float(limits.wall_time_seconds),
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class AgentSpecV1:
    """The small, stable configuration surface shared by agent adapters."""

    environment: str
    model: str
    profile: str
    system_prompt: str
    max_steps: int
    budget: BudgetLimits
    tool_capabilities: Tuple[str, ...]
    communication_capabilities: Tuple[str, ...] = ()
    fidelity: str = "minimal_baseline"

    schema: ClassVar[str] = AGENT_SPEC_SCHEMA

    def __post_init__(self) -> None:
        for name in ("environment", "model", "profile", "fidelity"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "system_prompt",
            _text(self.system_prompt, "system_prompt", allow_empty=True),
        )
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.budget, BudgetLimits):
            raise ValueError("budget must be BudgetLimits")
        object.__setattr__(
            self,
            "tool_capabilities",
            _capabilities(self.tool_capabilities, "tool_capabilities"),
        )
        object.__setattr__(
            self,
            "communication_capabilities",
            _capabilities(
                self.communication_capabilities, "communication_capabilities"
            ),
        )
        if self.communication_capabilities and "agent" not in self.tool_capabilities:
            raise ValueError(
                "communication capabilities require the agent tool capability"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the complete JSON representation without a derived fingerprint."""

        return {
            "budget": _budget_dict(self.budget),
            "communication_capabilities": list(self.communication_capabilities),
            "environment": self.environment,
            "fidelity": self.fidelity,
            "max_steps": self.max_steps,
            "model": self.model,
            "profile": self.profile,
            "schema": self.schema,
            "system_prompt": self.system_prompt,
            "tool_capabilities": list(self.tool_capabilities),
        }

    def canonical_json(self) -> str:
        """Serialize with the deterministic UTF-8 JSON rules used for identity."""

        return _canonical_json(self.as_dict())

    def identity_dict(self) -> dict[str, Any]:
        """Return manifest-safe identity without persisting the prompt contents."""

        value = self.as_dict()
        prompt = value.pop("system_prompt")
        value["system_prompt_sha256"] = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        value["fingerprint"] = self.fingerprint
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentSpecV1":
        """Load one v1 spec, rejecting missing, extra, or newer-version fields."""

        if not isinstance(value, Mapping):
            raise ValueError("agent spec must be an object")
        if not all(isinstance(key, str) for key in value):
            raise ValueError("agent spec keys must be strings")
        expected = {
            "budget",
            "communication_capabilities",
            "environment",
            "fidelity",
            "max_steps",
            "model",
            "profile",
            "schema",
            "system_prompt",
            "tool_capabilities",
        }
        keys = set(value)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"agent spec fields do not match v1 (missing={missing}, extra={extra})"
            )
        if value["schema"] != cls.schema:
            raise ValueError(f"unsupported agent spec schema {value['schema']!r}")
        budget_value = value["budget"]
        if not isinstance(budget_value, Mapping):
            raise ValueError("agent spec budget must be an object")
        if not all(isinstance(key, str) for key in budget_value):
            raise ValueError("agent spec budget keys must be strings")
        budget_fields = {
            "max_concurrency",
            "max_cost_usd",
            "max_input_tokens",
            "max_model_calls",
            "max_output_tokens",
            "max_tool_calls",
            "max_tool_output_bytes",
            "wall_time_seconds",
        }
        if set(budget_value) != budget_fields:
            raise ValueError("agent spec budget fields do not match v1")
        tool_capabilities = value["tool_capabilities"]
        communication_capabilities = value["communication_capabilities"]
        if not isinstance(tool_capabilities, (list, tuple)):
            raise ValueError("tool_capabilities must be an array")
        if not isinstance(communication_capabilities, (list, tuple)):
            raise ValueError("communication_capabilities must be an array")
        try:
            budget = BudgetLimits(**dict(budget_value))
        except TypeError as exc:
            raise ValueError("invalid agent spec budget") from exc
        return cls(
            environment=value["environment"],
            model=value["model"],
            profile=value["profile"],
            system_prompt=value["system_prompt"],
            max_steps=value["max_steps"],
            budget=budget,
            tool_capabilities=tuple(tool_capabilities),
            communication_capabilities=tuple(communication_capabilities),
            fidelity=value["fidelity"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "AgentSpecV1":
        """Load a canonical spec document and verify its optional fingerprint."""

        decoded = strict_json_loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("agent spec document must be an object")
        document = dict(decoded)
        claimed = document.pop("fingerprint", None)
        spec = cls.from_dict(document)
        if claimed is not None and (
            not isinstance(claimed, str)
            or not hmac.compare_digest(claimed, spec.fingerprint)
        ):
            raise ValueError("agent spec fingerprint does not match its contents")
        return spec

    def bind(
        self,
        *,
        model: "Model",
        environment: "AgentEnvironment",
        model_id: str,
        environment_id: str,
        context: "RunContext | None" = None,
        agent_id: str = "/root",
        role: str = "solver",
    ) -> "MiniAgent":
        """Validate declared capabilities and construct the ordinary mini-agent.

        Provider endpoint/authentication and environment assets remain downstream
        constructor concerns. Explicit identifiers prevent an opaque adapter from
        being silently bound under a different declared model or domain.
        """

        from .agent import MiniAgent
        from .runtime import RunContext

        if model_id != self.model:
            raise ValueError("model identifier does not match the agent spec")
        if environment_id != self.environment:
            raise ValueError("environment identifier does not match the agent spec")
        if not callable(getattr(environment, "tools", None)):
            raise ValueError("environment must expose tools")
        tools = tuple(environment.tools())
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise ValueError("environment tools must be ToolDefinition values")
        names = tuple(sorted(tool.name for tool in tools))
        if len(names) != len(set(names)):
            raise ValueError("environment tool names must be unique")
        if names != self.tool_capabilities:
            raise ValueError(
                "environment tool capabilities do not match the agent spec"
            )
        agent_tool = next((tool for tool in tools if tool.name == "agent"), None)
        observed_communication: tuple[str, ...] = ()
        if agent_tool is not None:
            properties = agent_tool.input_schema.get("properties")
            action = (
                properties.get("action") if isinstance(properties, Mapping) else None
            )
            choices = action.get("enum") if isinstance(action, Mapping) else None
            if not isinstance(choices, (list, tuple)) or not all(
                isinstance(choice, str) for choice in choices
            ):
                raise ValueError(
                    "agent tool must declare its communication action enum"
                )
            observed_communication = tuple(sorted(choices))
        if observed_communication != self.communication_capabilities:
            raise ValueError(
                "environment communication capabilities do not match the agent spec"
            )
        if context is None:
            context = RunContext(limits=self.budget)
        elif not isinstance(context, RunContext):
            raise ValueError("context must be RunContext or None")
        elif context.ledger.limits != self.budget:
            raise ValueError("context budget does not match the agent spec")
        return MiniAgent(
            model=model,
            environment=environment,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            context=context,
            agent_id=agent_id,
            role=role,
        )


@dataclass(frozen=True)
class TranslationLoss:
    """One source requirement that the target spec does not preserve exactly."""

    field: str
    reason: str
    kind: str = "unsupported"

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _text(self.field, "loss field"))
        object.__setattr__(self, "reason", _text(self.reason, "loss reason"))
        object.__setattr__(self, "kind", _text(self.kind, "loss kind"))
        if self.kind not in _LOSS_KINDS:
            raise ValueError(f"unsupported translation loss kind {self.kind!r}")

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "kind": self.kind, "reason": self.reason}


@dataclass(frozen=True)
class TranslationReport:
    """Auditable result of translating declared fields into :class:`AgentSpecV1`.

    ``exact`` only means that the translator declared no field-level losses.  Its
    scope is deliberately limited to declared fields; it is not a behavioral,
    policy-training, timing, tool, or benchmark-fidelity claim.
    """

    source_format: str
    spec: AgentSpecV1
    losses: Tuple[TranslationLoss, ...] = ()

    schema: ClassVar[str] = TRANSLATION_REPORT_SCHEMA
    claim_scope: ClassVar[str] = TRANSLATION_CLAIM_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_format", _text(self.source_format, "source_format")
        )
        if not isinstance(self.spec, AgentSpecV1):
            raise ValueError("translation spec must be AgentSpecV1")
        if not isinstance(self.losses, tuple) or not all(
            isinstance(loss, TranslationLoss) for loss in self.losses
        ):
            raise ValueError("translation losses must be a tuple of TranslationLoss")
        ordered = tuple(
            sorted(self.losses, key=lambda loss: (loss.field, loss.kind, loss.reason))
        )
        if len(ordered) != len(set(ordered)):
            raise ValueError("translation losses must not contain duplicates")
        object.__setattr__(self, "losses", ordered)

    @property
    def exact(self) -> bool:
        return not self.losses

    @property
    def status(self) -> str:
        return "exact" if self.exact else "lossy"

    def require_exact(self) -> AgentSpecV1:
        """Return the target spec or fail rather than silently accepting loss."""

        if not self.exact:
            fields = ", ".join(loss.field for loss in self.losses)
            raise ValueError(f"translation is lossy for: {fields}")
        return self.spec

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_scope": self.claim_scope,
            "exact": self.exact,
            "losses": [loss.as_dict() for loss in self.losses],
            "schema": self.schema,
            "source_format": self.source_format,
            "status": self.status,
            "target_fingerprint": self.spec.fingerprint,
            "target_format": self.spec.schema,
            "target_spec": self.spec.as_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


__all__ = [
    "AGENT_SPEC_SCHEMA",
    "TRANSLATION_CLAIM_SCOPE",
    "TRANSLATION_REPORT_SCHEMA",
    "AgentSpecV1",
    "TranslationLoss",
    "TranslationReport",
]
