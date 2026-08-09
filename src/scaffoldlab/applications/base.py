from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


APPLICATION_NAMES = ("browser", "computer-use", "swe")

EXACT_IMPLEMENTATION_FIDELITIES = frozenset(
    {"exact_public_protocol", "upstream_runtime_adapter"}
)
STUDY_FIDELITIES = frozenset(
    {
        "source_matched_reimplementation",
        "topology_simulation",
        "inference_only_reimplementation",
        "controlled_baseline",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(child)) for key, child in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class SourceArtifact:
    title: str
    url: str
    published: str
    version: str = ""
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.title or not self.url.startswith("https://") or not self.published:
            raise ValueError("application sources require a title, HTTPS URL, and date")

    def as_dict(self) -> dict[str, str]:
        result = {
            "title": self.title,
            "url": self.url,
            "published": self.published,
        }
        if self.version:
            result["version"] = self.version
        if self.revision:
            result["revision"] = self.revision
        return result


@dataclass(frozen=True)
class HarnessSignature:
    name: str
    options: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls, name: str, options: Mapping[str, Any] | None = None
    ) -> "HarnessSignature":
        if not name:
            raise ValueError("harness signature name must be non-empty")
        frozen = _freeze(options or {})
        if not isinstance(frozen, tuple):
            raise TypeError("frozen harness options must be a tuple")
        return cls(name=name, options=frozen)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "options": dict(self.options)}


@dataclass(frozen=True)
class ImplementationProfile:
    application: str
    profile_id: str
    title: str
    artifact_kind: str
    fidelity: str
    status: str
    runtime_owner: str
    harnesses: tuple[HarnessSignature, ...]
    providers: tuple[str, ...]
    environment_types: tuple[str, ...]
    model_families: tuple[str, ...]
    exact_components: tuple[str, ...]
    unavailable_components: tuple[str, ...]
    sources: tuple[SourceArtifact, ...]

    def __post_init__(self) -> None:
        if self.application not in APPLICATION_NAMES:
            raise ValueError(f"unknown application {self.application!r}")
        if not self.profile_id or not self.title:
            raise ValueError("implementation profile id and title must be non-empty")
        if self.artifact_kind not in {
            "inference_harness",
            "provider_protocol",
            "runtime_protocol",
            "evaluation_environment",
            "training_method",
        }:
            raise ValueError(f"invalid artifact kind {self.artifact_kind!r}")
        if self.fidelity not in {
            "exact_public_protocol",
            "upstream_runtime_adapter",
            "source_matched_reimplementation",
            "topology_simulation",
            "inference_only_reimplementation",
            "controlled_baseline",
            "documented_gap",
        }:
            raise ValueError(f"invalid fidelity {self.fidelity!r}")
        if self.status not in {"runnable", "simulation", "catalog_only"}:
            raise ValueError(f"invalid status {self.status!r}")
        if not self.sources:
            raise ValueError("every implementation profile requires a primary source")
        for label, values in (
            ("providers", self.providers),
            ("environment_types", self.environment_types),
            ("model_families", self.model_families),
            ("exact_components", self.exact_components),
            ("unavailable_components", self.unavailable_components),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{label} entries must be non-empty strings")
        if self.status != "catalog_only" and not self.harnesses:
            raise ValueError("runnable and simulation profiles require a harness")
        if self.status != "catalog_only" and (
            not self.providers or not self.environment_types
        ):
            raise ValueError(
                "runnable and simulation profiles require providers and environments"
            )
        if self.fidelity == "upstream_runtime_adapter" and not any(
            source.revision for source in self.sources
        ):
            raise ValueError(
                "upstream runtime adapters require a pinned source revision"
            )
        if self.fidelity in {"topology_simulation", "documented_gap"} and not (
            self.unavailable_components
        ):
            raise ValueError("simulations and gaps must name unavailable components")
        if self.status == "catalog_only" and (
            self.harnesses or self.providers or self.environment_types
        ):
            raise ValueError(
                "catalog-only profiles cannot declare harnesses, providers, or environments"
            )
        if (
            self.fidelity in EXACT_IMPLEMENTATION_FIDELITIES
            and self.status != "runnable"
        ):
            raise ValueError("exact implementations must be runnable")
        if self.fidelity in STUDY_FIDELITIES and self.status == "catalog_only":
            raise ValueError("studies cannot be catalog-only")
        if self.fidelity == "documented_gap" and self.status != "catalog_only":
            raise ValueError("documented gaps must be catalog-only")
        if self.fidelity == "exact_public_protocol" and self.artifact_kind not in {
            "provider_protocol",
            "runtime_protocol",
        }:
            raise ValueError(
                "exact public protocols must declare artifact_kind as "
                "'provider_protocol' or 'runtime_protocol'"
            )
        if (
            self.fidelity in EXACT_IMPLEMENTATION_FIDELITIES
            and self.runtime_owner == "scaffoldlab_local"
        ):
            raise ValueError("exact implementations cannot be locally reconstructed")

    @property
    def key(self) -> str:
        return f"{self.application}/{self.profile_id}"

    @property
    def catalog_kind(self) -> str:
        """Return the claim-bearing catalog tier for this profile.

        ``implementation`` is deliberately narrow: it means either the exact public
        protocol boundary or an adapter that executes a pinned upstream runtime.
        Clean-room subsets, topology reconstructions, and baselines are studies;
        non-runnable disclosures are gaps.
        """

        if self.fidelity in EXACT_IMPLEMENTATION_FIDELITIES:
            return "implementation"
        if self.fidelity in STUDY_FIDELITIES:
            return "study"
        return "gap"

    @property
    def exactness_scope(self) -> str | None:
        if self.fidelity == "exact_public_protocol":
            if self.artifact_kind == "runtime_protocol":
                return "published_runtime_protocol_boundary"
            return "published_protocol_boundary"
        if self.fidelity == "upstream_runtime_adapter":
            return "pinned_upstream_runtime"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "application": self.application,
            "id": self.profile_id,
            "key": self.key,
            "title": self.title,
            "artifact_kind": self.artifact_kind,
            "catalog_kind": self.catalog_kind,
            "fidelity": self.fidelity,
            "exactness_scope": self.exactness_scope,
            "status": self.status,
            "runtime_owner": self.runtime_owner,
            "harnesses": [signature.as_dict() for signature in self.harnesses],
            "providers": list(self.providers),
            "environment_types": list(self.environment_types),
            "model_families": list(self.model_families),
            "exact_components": list(self.exact_components),
            "unavailable_components": list(self.unavailable_components),
            "sources": [source.as_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class ApplicationSelection:
    name: str
    profile: ImplementationProfile

    @property
    def selection_kind(self) -> str:
        return self.profile.catalog_kind

    @property
    def implementation(self) -> ImplementationProfile:
        """Return the exact implementation selected by this application config.

        The compatibility name is retained for callers written before the catalog
        split, but it must not let a study masquerade as an implementation.
        """

        if self.selection_kind != "implementation":
            raise ValueError(
                f"selected profile {self.profile.key!r} is a "
                f"{self.selection_kind}, not an implementation"
            )
        return self.profile

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            self.selection_kind: self.profile.as_dict(),
        }


def normalize_harness_specs(raw_specs: Any) -> tuple[HarnessSignature, ...]:
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("config requires a non-empty harnesses list")
    normalized: list[HarnessSignature] = []
    for raw in raw_specs:
        if isinstance(raw, str):
            normalized.append(HarnessSignature.create(raw))
            continue
        if not isinstance(raw, Mapping):
            raise ValueError("harness specs must be strings or objects")
        name = raw.get("name")
        options = raw.get("options", {})
        if not isinstance(name, str) or not name:
            raise ValueError("harness spec name must be a non-empty string")
        if not isinstance(options, Mapping):
            raise ValueError(f"options for {name!r} must be an object")
        normalized.append(HarnessSignature.create(name, options))
    return tuple(normalized)


def profile(
    *,
    application: str,
    profile_id: str,
    title: str,
    artifact_kind: str = "inference_harness",
    fidelity: str,
    status: str = "runnable",
    runtime_owner: str,
    harnesses: Sequence[HarnessSignature] = (),
    providers: Sequence[str] = (),
    environment_types: Sequence[str] = (),
    model_families: Sequence[str] = (),
    exact_components: Sequence[str] = (),
    unavailable_components: Sequence[str] = (),
    sources: Sequence[SourceArtifact],
) -> ImplementationProfile:
    return ImplementationProfile(
        application=application,
        profile_id=profile_id,
        title=title,
        artifact_kind=artifact_kind,
        fidelity=fidelity,
        status=status,
        runtime_owner=runtime_owner,
        harnesses=tuple(harnesses),
        providers=tuple(providers),
        environment_types=tuple(environment_types),
        model_families=tuple(model_families),
        exact_components=tuple(exact_components),
        unavailable_components=tuple(unavailable_components),
        sources=tuple(sources),
    )
