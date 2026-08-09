from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .base import (
    APPLICATION_NAMES,
    ApplicationSelection,
    HarnessSignature,
    ImplementationProfile,
    normalize_harness_specs,
)
from .browser import PROFILES as BROWSER_PROFILES
from .computer_use import PROFILES as COMPUTER_USE_PROFILES
from .swe import PROFILES as SWE_PROFILES


_PROFILE_GROUPS: dict[str, tuple[ImplementationProfile, ...]] = {
    "browser": BROWSER_PROFILES,
    "computer-use": COMPUTER_USE_PROFILES,
    "swe": SWE_PROFILES,
}


def _build_profile_index() -> dict[str, ImplementationProfile]:
    missing = set(APPLICATION_NAMES) - set(_PROFILE_GROUPS)
    extra = set(_PROFILE_GROUPS) - set(APPLICATION_NAMES)
    if missing or extra:
        raise RuntimeError(
            "application registry groups do not match APPLICATION_NAMES: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    index: dict[str, ImplementationProfile] = {}
    for application in APPLICATION_NAMES:
        for implementation in _PROFILE_GROUPS[application]:
            if implementation.application != application:
                raise RuntimeError(
                    f"profile {implementation.key!r} is registered under "
                    f"{application!r}"
                )
            if implementation.key in index:
                raise RuntimeError(
                    f"duplicate implementation profile {implementation.key!r}"
                )
            index[implementation.key] = implementation
    return index


APPLICATIONS: Mapping[str, tuple[ImplementationProfile, ...]] = MappingProxyType(
    _PROFILE_GROUPS
)
IMPLEMENTATIONS: tuple[ImplementationProfile, ...] = tuple(
    implementation
    for application in APPLICATION_NAMES
    for implementation in _PROFILE_GROUPS[application]
)
IMPLEMENTATIONS_BY_KEY: Mapping[str, ImplementationProfile] = MappingProxyType(
    _build_profile_index()
)


def list_applications() -> tuple[str, ...]:
    """Return the stable top-level application names."""

    return APPLICATION_NAMES


def list_implementations(
    application: str | None = None,
) -> tuple[ImplementationProfile, ...]:
    """Return catalog profiles in deterministic declaration order."""

    if application is None:
        return IMPLEMENTATIONS
    if not isinstance(application, str) or application not in APPLICATIONS:
        raise ValueError(
            f"unknown application {application!r}; choose from "
            f"{', '.join(APPLICATION_NAMES)}"
        )
    return APPLICATIONS[application]


def get_implementation(application: str, implementation: str) -> ImplementationProfile:
    """Resolve an application-local id or a fully-qualified profile key."""

    if not isinstance(application, str) or application not in APPLICATIONS:
        raise ValueError(
            f"unknown application {application!r}; choose from "
            f"{', '.join(APPLICATION_NAMES)}"
        )
    if not isinstance(implementation, str) or not implementation:
        raise ValueError("implementation must be a non-empty string")
    if "/" in implementation:
        prefix, separator, profile_id = implementation.partition("/")
        if not separator or not profile_id:
            raise ValueError(f"invalid implementation key {implementation!r}")
        if prefix != application:
            raise ValueError(
                f"implementation {implementation!r} belongs to application "
                f"{prefix!r}, not {application!r}"
            )
        key = implementation
    else:
        key = f"{application}/{implementation}"
    try:
        return IMPLEMENTATIONS_BY_KEY[key]
    except KeyError as exc:
        available = ", ".join(
            profile.profile_id for profile in APPLICATIONS[application]
        )
        raise ValueError(
            f"unknown implementation {implementation!r} for application "
            f"{application!r}; choose from {available}"
        ) from exc


def _parse_selection(
    config: Mapping[str, Any],
) -> ApplicationSelection | None:
    has_application = "application" in config
    has_implementation = "implementation" in config
    if not has_application and not has_implementation:
        return None
    if not has_application:
        raise ValueError("config implementation requires config.application")

    raw_application = config.get("application")
    top_level_implementation = config.get("implementation")
    application: Any
    implementation: Any
    if isinstance(raw_application, str):
        application = raw_application
        implementation = top_level_implementation
    elif isinstance(raw_application, Mapping):
        unknown = set(raw_application) - {"name", "implementation"}
        if unknown:
            raise ValueError(
                "config application object has unknown fields: "
                f"{sorted(str(item) for item in unknown)}"
            )
        application = raw_application.get("name")
        nested_implementation = raw_application.get("implementation")
        if (
            has_implementation
            and nested_implementation is not None
            and nested_implementation != top_level_implementation
        ):
            raise ValueError(
                "top-level implementation conflicts with application.implementation"
            )
        implementation = (
            top_level_implementation if has_implementation else nested_implementation
        )
    else:
        raise ValueError("config application must be a string or object")

    if not isinstance(application, str) or not application:
        raise ValueError("config application name must be a non-empty string")
    if not isinstance(implementation, str) or not implementation:
        raise ValueError(
            "config application selection requires a non-empty implementation"
        )
    profile = get_implementation(application, implementation)
    return ApplicationSelection(name=application, implementation=profile)


def _thaw(value: Any) -> Any:
    if not isinstance(value, tuple):
        return value
    if value and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ):
        return {key: _thaw(child) for key, child in value}
    return [_thaw(child) for child in value]


def _harness_config(signature: HarnessSignature) -> dict[str, Any]:
    return {
        "name": signature.name,
        "options": {key: _thaw(value) for key, value in signature.options},
    }


def _environment_type(config: Mapping[str, Any]) -> str:
    raw = config.get("environment")
    if raw is None:
        return "none"
    if not isinstance(raw, Mapping):
        raise ValueError("config environment must be an object or null")
    environment_type = raw.get("type")
    if not isinstance(environment_type, str) or not environment_type:
        raise ValueError("config environment.type must be a non-empty string")
    if environment_type == "none":
        raise ValueError(
            "environment type 'none' is a catalog sentinel; omit "
            "config.environment or set it to null"
        )
    return environment_type


def _format_harnesses(harnesses: Sequence[HarnessSignature]) -> list[dict[str, Any]]:
    return [_harness_config(signature) for signature in harnesses]


def resolve_application_config(
    config: Mapping[str, Any], *, provider: str
) -> tuple[dict[str, Any], ApplicationSelection | None, list[str]]:
    """Resolve and validate an optional catalog selection.

    The canonical shape is ``{"application": {"name": "browser",
    "implementation": "profile-id"}}``. The top-level ``application`` plus
    ``implementation`` spelling is accepted for compatibility with early configs.

    Legacy configs containing neither ``application`` nor ``implementation`` are
    returned unchanged. A selected runnable profile supplies its exact harness list
    when that field is omitted. If a harness list is present, it must match the
    profile exactly so the catalog fidelity claim cannot silently drift.
    """

    if not isinstance(config, Mapping):
        raise ValueError("experiment config must be a JSON object")
    resolved = dict(config)
    selection = _parse_selection(config)
    if selection is None:
        return resolved, None, []

    profile = selection.implementation
    if profile.status == "catalog_only":
        raise ValueError(
            f"implementation {profile.key!r} is catalog-only and cannot be run"
        )
    if provider not in profile.providers:
        raise ValueError(
            f"implementation {profile.key!r} does not support provider "
            f"{provider!r}; choose from {', '.join(profile.providers)}"
        )

    actual_environment = _environment_type(resolved)
    if actual_environment not in profile.environment_types:
        expected = ", ".join(profile.environment_types)
        raise ValueError(
            f"implementation {profile.key!r} requires environment type "
            f"{expected}; config selects {actual_environment}"
        )

    if "harnesses" not in resolved:
        resolved["harnesses"] = _format_harnesses(profile.harnesses)
    else:
        configured_harnesses = normalize_harness_specs(resolved.get("harnesses"))
        if configured_harnesses != profile.harnesses:
            raise ValueError(
                f"implementation {profile.key!r} requires exact harnesses "
                f"{_format_harnesses(profile.harnesses)!r}; config selects "
                f"{_format_harnesses(configured_harnesses)!r}"
            )

    warnings: list[str] = []
    if profile.status == "simulation":
        warnings.append(
            f"{profile.key} is a topology simulation, not a 1:1 reproduction; "
            f"unavailable components: {'; '.join(profile.unavailable_components)}"
        )
    elif profile.fidelity == "exact_public_protocol" and profile.unavailable_components:
        warnings.append(
            f"{profile.key} is exact only at the published protocol boundary; "
            f"unavailable components: {'; '.join(profile.unavailable_components)}"
        )
    elif (
        profile.fidelity == "upstream_runtime_adapter"
        and profile.unavailable_components
    ):
        warnings.append(
            f"{profile.key} executes a pinned upstream runtime, but full experimental "
            f"identity still requires: {'; '.join(profile.unavailable_components)}"
        )
    elif profile.fidelity in {
        "source_matched_reimplementation",
        "inference_only_reimplementation",
    }:
        warnings.append(
            f"{profile.key} is a {profile.fidelity.replace('_', ' ')} rather than "
            f"the complete source artifact; unavailable components: "
            f"{'; '.join(profile.unavailable_components)}"
        )
    return resolved, selection, warnings


__all__ = [
    "APPLICATIONS",
    "IMPLEMENTATIONS",
    "IMPLEMENTATIONS_BY_KEY",
    "get_implementation",
    "list_applications",
    "list_implementations",
    "resolve_application_config",
]
