"""Thin access to preserved Scaffold Lab runtimes behind an optional boundary.

References and studies are deliberately not translated into ``MiniAgent`` loops.
They run the preserved evaluator in-process so its catalog checks, provider
adapters, budget ledger, trace recorder, lifecycle, and artifacts stay authoritative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from .research_catalog import (
    ImplementationProfile,
    resolve_application_config,
)

from .catalog import (
    LEGACY_TO_APPLICATION,
    get_implementation as get_catalog_implementation,
    get_study as get_catalog_study,
    list_implementations as list_catalog_implementations,
    list_studies as list_catalog_studies,
)


_RESERVED_ARGUMENTS = frozenset(
    {
        "--tasks",
        "--config",
        "--provider",
        "--output",
        "--expected-application-key",
        "--expected-config-sha256",
    }
)


@dataclass(frozen=True)
class ReferenceRuntime:
    """An exact public protocol or pinned upstream runtime.

    ``profile`` remains the legacy catalog object because it owns the fidelity
    claim.  This class only gives that object mini-agent application names and a
    small delegation API.
    """

    application: str
    profile: ImplementationProfile
    catalog_kind: ClassVar[str] = "implementation"
    execution_mode: ClassVar[str] = "reference"
    runtime_label: ClassVar[str] = "reference"

    def __post_init__(self) -> None:
        expected_application = LEGACY_TO_APPLICATION.get(self.profile.application)
        if expected_application != self.application:
            raise ValueError(
                f"reference profile {self.profile.key!r} belongs to "
                f"{expected_application!r}, not {self.application!r}"
            )
        if (
            self.profile.catalog_kind != self.catalog_kind
            or self.profile.status != "runnable"
        ):
            raise ValueError(
                f"profile {self.profile.key!r} is a {self.profile.catalog_kind}, "
                f"not a runnable {self.runtime_label}"
            )

    @property
    def name(self) -> str:
        return self.profile.profile_id

    @property
    def key(self) -> str:
        return f"{self.application}/{self.name}"

    @property
    def providers(self) -> tuple[str, ...]:
        return self.profile.providers

    def manifest(self) -> dict[str, Any]:
        payload = self.profile.as_dict()
        payload["application"] = self.application
        payload["key"] = self.key
        payload["legacy_application"] = self.profile.application
        payload["legacy_key"] = self.profile.key
        payload["execution_mode"] = self.execution_mode
        payload["delegate"] = "scaffoldlab.cli"
        return payload

    def validate(
        self,
        *,
        tasks: Path,
        config: Path,
        provider: str | None = None,
    ) -> int:
        """Run the legacy config/task validation boundary in-process."""

        selected_provider, config_sha256 = self._prepare(config, provider)
        return _legacy_main(
            (
                "validate",
                "--tasks",
                str(tasks),
                "--config",
                str(config),
                "--provider",
                selected_provider,
                "--expected-application-key",
                self.profile.key,
                "--expected-config-sha256",
                config_sha256,
            )
        )

    def run(
        self,
        *,
        tasks: Path,
        config: Path,
        output: Path,
        provider: str | None = None,
        arguments: Sequence[str] = (),
    ) -> int:
        """Run the preserved target through the complete evaluation pipeline.

        ``arguments`` holds provider-specific options such as ``--model`` or a
        pinned checkout.  They are argv tokens, never shell text.  Identity-bearing
        paths and the provider are appended by this adapter and cannot be replaced.
        """

        selected_provider, config_sha256 = self._prepare(config, provider)
        extra = _safe_arguments(arguments)
        return _legacy_main(
            (
                "run",
                *extra,
                "--tasks",
                str(tasks),
                "--config",
                str(config),
                "--provider",
                selected_provider,
                "--expected-application-key",
                self.profile.key,
                "--expected-config-sha256",
                config_sha256,
                "--output",
                str(output),
            )
        )

    def _prepare(self, config: Path, provider: str | None) -> tuple[str, str]:
        selected_provider = _select_provider(self.profile, provider)
        config_sha256 = _require_selected_profile(
            config,
            self.profile,
            selected_provider,
            expected_kind=self.catalog_kind,
        )
        return selected_provider, config_sha256


class StudyRuntime(ReferenceRuntime):
    """A preserved non-exact study, never promoted to an exact reference."""

    catalog_kind = "study"
    execution_mode = "study"
    runtime_label = "study"

    def __post_init__(self) -> None:
        expected_application = LEGACY_TO_APPLICATION.get(self.profile.application)
        if expected_application != self.application:
            raise ValueError(
                f"study profile {self.profile.key!r} belongs to "
                f"{expected_application!r}, not {self.application!r}"
            )
        if self.profile.catalog_kind != "study" or self.profile.status not in {
            "runnable",
            "simulation",
        }:
            raise ValueError(
                f"profile {self.profile.key!r} is a {self.profile.catalog_kind}, "
                "not a runnable study"
            )


def list_references(application: str | None = None) -> tuple[ReferenceRuntime, ...]:
    """List only exact legacy implementations using mini-agent domain names."""

    if application is None:
        profiles = list_catalog_implementations()
    else:
        if application not in {"swe", "web", "cua"}:
            raise ValueError("application must be swe, web, or cua")
        profiles = list_catalog_implementations(application)
    return tuple(
        ReferenceRuntime(profile.application, profile.legacy) for profile in profiles
    )


def get_reference(application: str, name: str) -> ReferenceRuntime:
    """Resolve an exact runtime; studies and catalog gaps fail closed."""

    profile = get_catalog_implementation(application, name)
    return ReferenceRuntime(profile.application, profile.legacy)


def list_study_runtimes(application: str | None = None) -> tuple[StudyRuntime, ...]:
    """List preserved studies without weakening their non-exact fidelity labels."""

    if application is None:
        profiles = list_catalog_studies()
    else:
        if application not in {"swe", "web", "cua"}:
            raise ValueError("application must be swe, web, or cua")
        profiles = list_catalog_studies(application)
    return tuple(
        StudyRuntime(profile.application, profile.legacy) for profile in profiles
    )


def get_study_runtime(application: str, name: str) -> StudyRuntime:
    profile = get_catalog_study(application, name)
    return StudyRuntime(profile.application, profile.legacy)


def _select_provider(profile: ImplementationProfile, provider: str | None) -> str:
    label = "reference" if profile.catalog_kind == "implementation" else "study"
    if provider is None:
        if len(profile.providers) != 1:
            raise ValueError(
                f"{label} {profile.key!r} requires an explicit provider from "
                f"{list(profile.providers)!r}"
            )
        return profile.providers[0]
    if provider not in profile.providers:
        raise ValueError(
            f"{label} {profile.key!r} does not support provider {provider!r}; "
            f"choose from {', '.join(profile.providers)}"
        )
    return provider


def _require_selected_profile(
    path: Path,
    expected: ImplementationProfile,
    provider: str,
    *,
    expected_kind: str,
) -> str:
    encoded = path.read_bytes()
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reference config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("reference config must be a JSON object")
    _, selection, _ = resolve_application_config(raw, provider=provider)
    if selection is None or selection.selection_kind != expected_kind:
        raise ValueError(
            f"{expected_kind} config must select application.{expected_kind}"
        )
    if selection.profile.key != expected.key:
        raise ValueError(
            f"config selects {selection.profile.key!r}, not {expected_kind} "
            f"profile {expected.key!r}"
        )
    return hashlib.sha256(encoded).hexdigest()


def _safe_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for argument in arguments:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError("delegated arguments must be strings without NUL bytes")
        option = argument.split("=", 1)[0] if argument.startswith("--") else ""
        if argument == "--" or (
            option
            and any(reserved.startswith(option) for reserved in _RESERVED_ARGUMENTS)
        ):
            raise ValueError(f"reference argument {option!r} is owned by the adapter")
        result.append(argument)
    return tuple(result)


def _legacy_main(arguments: Sequence[str]) -> int:
    # Listing manifests never imports the archived harness dependency graph.
    from .references_runtime import preserved_scaffold_main

    return preserved_scaffold_main(tuple(arguments))


__all__ = [
    "ReferenceRuntime",
    "StudyRuntime",
    "get_reference",
    "get_study_runtime",
    "list_references",
    "list_study_runtimes",
]
