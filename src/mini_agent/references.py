"""Thin access to exact runtimes retained in :mod:`scaffoldlab`.

References are deliberately not translated into ``MiniAgent`` profiles.  They run
the legacy CLI in-process so its catalog checks, provider adapters, budget ledger,
trace recorder, environment lifecycle, and evaluation artifacts stay authoritative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scaffoldlab.applications import (
    ImplementationProfile,
    resolve_application_config,
)

from .catalog import (
    get_implementation as get_catalog_implementation,
    list_implementations as list_catalog_implementations,
)


_RESERVED_ARGUMENTS = frozenset(
    {"--tasks", "--config", "--provider", "--output"}
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

        selected_provider = self._prepare(config, provider)
        return _legacy_main(
            (
                "validate",
                "--tasks",
                str(tasks),
                "--config",
                str(config),
                "--provider",
                selected_provider,
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
        """Run the reference through the complete legacy evaluation pipeline.

        ``arguments`` holds provider-specific options such as ``--model`` or a
        pinned checkout.  They are argv tokens, never shell text.  Identity-bearing
        paths and the provider are appended by this adapter and cannot be replaced.
        """

        selected_provider = self._prepare(config, provider)
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
                "--output",
                str(output),
            )
        )

    def _prepare(self, config: Path, provider: str | None) -> str:
        selected_provider = _select_provider(self.profile, provider)
        _require_selected_reference(config, self.profile, selected_provider)
        return selected_provider


def list_references(application: str | None = None) -> tuple[ReferenceRuntime, ...]:
    """List only exact legacy implementations using mini-agent domain names."""

    if application is None:
        profiles = list_catalog_implementations()
    else:
        if application not in {"swe", "web", "cua"}:
            raise ValueError("application must be swe, web, or cua")
        profiles = list_catalog_implementations(application)
    return tuple(
        ReferenceRuntime(profile.application, profile.legacy)
        for profile in profiles
    )


def get_reference(application: str, name: str) -> ReferenceRuntime:
    """Resolve an exact runtime; studies and catalog gaps fail closed."""

    profile = get_catalog_implementation(application, name)
    return ReferenceRuntime(profile.application, profile.legacy)


def _select_provider(
    profile: ImplementationProfile, provider: str | None
) -> str:
    if provider is None:
        if len(profile.providers) != 1:
            raise ValueError(
                f"reference {profile.key!r} requires an explicit provider from "
                f"{list(profile.providers)!r}"
            )
        return profile.providers[0]
    if provider not in profile.providers:
        raise ValueError(
            f"reference {profile.key!r} does not support provider {provider!r}; "
            f"choose from {', '.join(profile.providers)}"
        )
    return provider


def _require_selected_reference(
    path: Path, expected: ImplementationProfile, provider: str
) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reference config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("reference config must be a JSON object")
    resolved, selection, _ = resolve_application_config(raw, provider=provider)
    if selection is None or selection.selection_kind != "implementation":
        raise ValueError("reference config must select application.implementation")
    if selection.profile.key != expected.key:
        raise ValueError(
            f"config selects {selection.profile.key!r}, not reference {expected.key!r}"
        )
    return resolved


def _safe_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for argument in arguments:
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ValueError("reference arguments must be non-empty argv strings")
        option = argument.split("=", 1)[0]
        if option in _RESERVED_ARGUMENTS:
            raise ValueError(f"reference argument {option!r} is owned by the adapter")
        result.append(argument)
    return tuple(result)


def _legacy_main(arguments: Sequence[str]) -> int:
    # Import lazily: listing reference manifests does not need the provider/runtime
    # dependency graph, and tests can prove delegation without starting a process.
    from scaffoldlab.cli import main

    return main(tuple(arguments))


__all__ = ["ReferenceRuntime", "get_reference", "list_references"]
