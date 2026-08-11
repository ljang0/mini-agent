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
PROFILES: tuple[ImplementationProfile, ...] = tuple(
    profile
    for application in APPLICATION_NAMES
    for profile in _PROFILE_GROUPS[application]
)
PROFILES_BY_KEY: Mapping[str, ImplementationProfile] = MappingProxyType(
    _build_profile_index()
)
IMPLEMENTATIONS: tuple[ImplementationProfile, ...] = tuple(
    profile for profile in PROFILES if profile.catalog_kind == "implementation"
)
STUDIES: tuple[ImplementationProfile, ...] = tuple(
    profile for profile in PROFILES if profile.catalog_kind == "study"
)
GAPS: tuple[ImplementationProfile, ...] = tuple(
    profile for profile in PROFILES if profile.catalog_kind == "gap"
)
IMPLEMENTATIONS_BY_KEY: Mapping[str, ImplementationProfile] = MappingProxyType(
    {profile.key: profile for profile in IMPLEMENTATIONS}
)
STUDIES_BY_KEY: Mapping[str, ImplementationProfile] = MappingProxyType(
    {profile.key: profile for profile in STUDIES}
)
GAPS_BY_KEY: Mapping[str, ImplementationProfile] = MappingProxyType(
    {profile.key: profile for profile in GAPS}
)


def list_applications() -> tuple[str, ...]:
    """Return the stable top-level application names."""

    return APPLICATION_NAMES


def list_implementations(
    application: str | None = None,
) -> tuple[ImplementationProfile, ...]:
    """Return only exact public-boundary or pinned-upstream implementations."""

    return _list_kind("implementation", application)


def list_studies(
    application: str | None = None,
) -> tuple[ImplementationProfile, ...]:
    """Return non-exact runnable comparisons and topology simulations."""

    return _list_kind("study", application)


def list_gaps(
    application: str | None = None,
) -> tuple[ImplementationProfile, ...]:
    """Return source-backed artifacts that cannot currently run faithfully."""

    return _list_kind("gap", application)


def list_profiles(
    application: str | None = None,
) -> tuple[ImplementationProfile, ...]:
    """Return the complete catalog in deterministic declaration order."""

    _validate_application(application)
    if application is None:
        return PROFILES
    return APPLICATIONS[application]


def _validate_application(application: str | None) -> None:
    if application is not None and (
        not isinstance(application, str) or application not in APPLICATIONS
    ):
        raise ValueError(
            f"unknown application {application!r}; choose from "
            f"{', '.join(APPLICATION_NAMES)}"
        )


def _list_kind(kind: str, application: str | None) -> tuple[ImplementationProfile, ...]:
    _validate_application(application)
    source = PROFILES if application is None else APPLICATIONS[application]
    return tuple(profile for profile in source if profile.catalog_kind == kind)


def _profile_key(application: str, profile_id: str) -> str:
    _validate_application(application)
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile id must be a non-empty string")
    if "/" in profile_id:
        prefix, separator, local_id = profile_id.partition("/")
        if not separator or not local_id:
            raise ValueError(f"invalid profile key {profile_id!r}")
        if prefix != application:
            raise ValueError(
                f"profile {profile_id!r} belongs to application "
                f"{prefix!r}, not {application!r}"
            )
        return profile_id
    return f"{application}/{profile_id}"


def get_profile(application: str, profile_id: str) -> ImplementationProfile:
    """Resolve any implementation, study, or gap profile."""

    key = _profile_key(application, profile_id)
    try:
        return PROFILES_BY_KEY[key]
    except KeyError as exc:
        available = ", ".join(
            profile.profile_id for profile in APPLICATIONS[application]
        )
        raise ValueError(
            f"unknown profile {profile_id!r} for application {application!r}; "
            f"choose from {available}"
        ) from exc


def _get_kind(
    application: str, profile_id: str, expected_kind: str
) -> ImplementationProfile:
    profile = get_profile(application, profile_id)
    if profile.catalog_kind != expected_kind:
        article = "an" if expected_kind == "implementation" else "a"
        raise ValueError(
            f"profile {profile.key!r} is a {profile.catalog_kind}, not {article} "
            f"{expected_kind}; select it with application.{profile.catalog_kind}"
        )
    return profile


def get_study(application: str, study: str) -> ImplementationProfile:
    return _get_kind(application, study, "study")


def get_implementation(application: str, implementation: str) -> ImplementationProfile:
    """Resolve an exact published-boundary or pinned-upstream implementation."""

    return _get_kind(application, implementation, "implementation")


def _parse_selection(
    config: Mapping[str, Any],
) -> ApplicationSelection | None:
    has_application = "application" in config
    has_implementation = "implementation" in config
    has_study = "study" in config
    if not has_application and not has_implementation and not has_study:
        return None
    if not has_application:
        raise ValueError("config implementation or study requires config.application")

    raw_application = config.get("application")
    top_level = {
        "implementation": config.get("implementation") if has_implementation else None,
        "study": config.get("study") if has_study else None,
    }
    application: Any
    nested: dict[str, Any] = {"implementation": None, "study": None}
    if isinstance(raw_application, str):
        application = raw_application
    elif isinstance(raw_application, Mapping):
        unknown = set(raw_application) - {"name", "implementation", "study"}
        if unknown:
            raise ValueError(
                "config application object has unknown fields: "
                f"{sorted(str(item) for item in unknown)}"
            )
        application = raw_application.get("name")
        nested = {
            "implementation": raw_application.get("implementation"),
            "study": raw_application.get("study"),
        }
    else:
        raise ValueError("config application must be a string or object")

    if not isinstance(application, str) or not application:
        raise ValueError("config application name must be a non-empty string")
    # mini-agent's public domain names are accepted at the preserved evaluator
    # boundary; the resolved selection retains its original legacy identity.
    application = {"web": "browser", "cua": "computer-use"}.get(
        application, application
    )

    selections: list[tuple[str, Any]] = []
    for kind in ("implementation", "study"):
        top_value = top_level[kind]
        nested_value = nested[kind]
        if (
            top_value is not None
            and nested_value is not None
            and top_value != nested_value
        ):
            raise ValueError(f"top-level {kind} conflicts with application.{kind}")
        value = top_value if top_value is not None else nested_value
        if value is not None:
            selections.append((kind, value))
    if len(selections) != 1:
        raise ValueError(
            "config application selection requires exactly one non-empty "
            "implementation or study"
        )
    kind, profile_id = selections[0]
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError(f"config application {kind} must be a non-empty string")
    profile = (
        get_implementation(application, profile_id)
        if kind == "implementation"
        else get_study(application, profile_id)
    )
    return ApplicationSelection(name=application, profile=profile)


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

    The canonical exact shape is ``{"application": {"name": "browser",
    "implementation": "profile-id"}}``. Non-exact runnable comparisons use
    ``study`` instead of ``implementation``. The equivalent top-level spellings are
    accepted for compatibility with early configs.

    Legacy configs containing none of ``application``, ``implementation``, or
    ``study`` are returned unchanged. A selected runnable profile supplies its exact
    harness list when that field is omitted. If a harness list is present, it must
    match the profile exactly so the catalog fidelity claim cannot silently drift.
    """

    if not isinstance(config, Mapping):
        raise ValueError("experiment config must be a JSON object")
    resolved = dict(config)
    selection = _parse_selection(config)
    if selection is None:
        return resolved, None, []

    profile = selection.profile
    if profile.status == "catalog_only":
        raise ValueError(f"profile {profile.key!r} is catalog-only and cannot be run")

    metadata = resolved.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("config metadata must be an object or null")
        configured_fidelity = metadata.get("fidelity")
        if configured_fidelity is not None and configured_fidelity != profile.fidelity:
            raise ValueError(
                f"profile {profile.key!r} requires fidelity {profile.fidelity!r}; "
                f"config metadata selects {configured_fidelity!r}"
            )
        if profile.fidelity == "upstream_runtime_adapter":
            for pin_field in ("version", "revision"):
                configured_pin = metadata.get(pin_field)
                if configured_pin is None:
                    continue
                source_pins = {
                    getattr(source, pin_field)
                    for source in profile.sources
                    if getattr(source, pin_field)
                }
                if configured_pin not in source_pins:
                    raise ValueError(
                        f"profile {profile.key!r} does not cite metadata "
                        f"{pin_field} {configured_pin!r}; choose from "
                        f"{sorted(source_pins)!r}"
                    )
    if provider not in profile.providers:
        raise ValueError(
            f"profile {profile.key!r} does not support provider "
            f"{provider!r}; choose from {', '.join(profile.providers)}"
        )

    actual_environment = _environment_type(resolved)
    if actual_environment not in profile.environment_types:
        expected = ", ".join(profile.environment_types)
        raise ValueError(
            f"profile {profile.key!r} requires environment type "
            f"{expected}; config selects {actual_environment}"
        )

    if "harnesses" not in resolved:
        resolved["harnesses"] = _format_harnesses(profile.harnesses)
    else:
        configured_harnesses = normalize_harness_specs(resolved.get("harnesses"))
        if configured_harnesses != profile.harnesses:
            raise ValueError(
                f"profile {profile.key!r} requires exact harnesses "
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
        "caller_built_runtime_study",
        "source_matched_reimplementation",
        "inference_only_reimplementation",
    }:
        warnings.append(
            f"{profile.key} is a {profile.fidelity.replace('_', ' ')} rather than "
            f"the complete source artifact; unavailable components: "
            f"{'; '.join(profile.unavailable_components)}"
        )
    elif profile.catalog_kind == "study":
        warnings.append(
            f"{profile.key} is a controlled study, not a 1:1 implementation"
        )
    return resolved, selection, warnings


__all__ = [
    "APPLICATIONS",
    "GAPS",
    "GAPS_BY_KEY",
    "IMPLEMENTATIONS",
    "IMPLEMENTATIONS_BY_KEY",
    "PROFILES",
    "PROFILES_BY_KEY",
    "STUDIES",
    "STUDIES_BY_KEY",
    "get_implementation",
    "get_profile",
    "get_study",
    "list_applications",
    "list_gaps",
    "list_implementations",
    "list_profiles",
    "list_studies",
    "resolve_application_config",
]
