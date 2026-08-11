"""Read-only mini-agent view of the audited legacy implementation catalog.

This module deliberately does not instantiate agents or translate harnesses.  It
keeps catalog evidence separate from executable ``MiniAgent`` profiles and exposes
the old records 1:1 with only the application names normalized for mini-agent:
``browser`` becomes ``web`` and ``computer-use`` becomes ``cua``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .research_catalog import (
    FRONTIER_LABS as _LEGACY_FRONTIER_SOURCES,
    FRONTIER_MANIFEST_AS_OF,
    PROFILES as _LEGACY_PROFILES,
)
from .research_catalog.base import ImplementationProfile
from .research_catalog.frontier_manifest import (
    FrontierDistributionIdentity,
    FrontierLabRecord,
)


APPLICATIONS = ("web", "cua", "swe")
LEGACY_TO_APPLICATION: Mapping[str, str] = MappingProxyType(
    {"browser": "web", "computer-use": "cua", "swe": "swe"}
)
APPLICATION_TO_LEGACY: Mapping[str, str] = MappingProxyType(
    {value: key for key, value in LEGACY_TO_APPLICATION.items()}
)


def _application(value: str) -> str:
    try:
        return LEGACY_TO_APPLICATION[value]
    except KeyError as exc:
        raise ValueError(f"unknown legacy application {value!r}") from exc


@dataclass(frozen=True)
class CatalogProfile:
    """A legacy catalog profile exposed under mini-agent domain names.

    ``legacy`` is retained so consumers can inspect the original immutable record
    without this compatibility view copying or weakening any fidelity claim.
    """

    legacy: ImplementationProfile

    def __getattr__(self, name: str) -> Any:
        """Delegate unchanged metadata fields to the immutable source record."""

        return getattr(self.legacy, name)

    @property
    def application(self) -> str:
        return _application(self.legacy.application)

    @property
    def legacy_application(self) -> str:
        return self.legacy.application

    @property
    def profile_id(self) -> str:
        return self.legacy.profile_id

    @property
    def key(self) -> str:
        return f"{self.application}/{self.profile_id}"

    @property
    def legacy_key(self) -> str:
        return self.legacy.key

    @property
    def title(self) -> str:
        return self.legacy.title

    @property
    def status(self) -> str:
        return self.legacy.status

    @property
    def fidelity(self) -> str:
        return self.legacy.fidelity

    @property
    def exactness_scope(self) -> str | None:
        return self.legacy.exactness_scope

    @property
    def catalog_kind(self) -> str:
        return self.legacy.catalog_kind

    @property
    def execution_mode(self) -> str:
        """Describe the only honest execution boundary for this catalog entry."""

        if self.catalog_kind == "implementation":
            return "reference"
        if self.catalog_kind == "study":
            return "study"
        return "unavailable"

    def as_dict(self) -> dict[str, Any]:
        value = self.legacy.as_dict()
        value.update(
            {
                "application": self.application,
                "key": self.key,
                "legacy_application": self.legacy_application,
                "legacy_key": self.legacy_key,
                "execution_mode": self.execution_mode,
            }
        )
        return value


@dataclass(frozen=True)
class FrontierApplicationStatus:
    application: str
    legacy_application: str
    status: str
    implementation_ids: tuple[str, ...]
    boundary: str

    @property
    def implementation_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.application}/{value}" for value in self.implementation_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "application": self.application,
            "legacy_application": self.legacy_application,
            "status": self.status,
            "implementation_ids": list(self.implementation_ids),
            "implementation_keys": list(self.implementation_keys),
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class FrontierSource:
    """Audited frontier evidence; catalog presence is not runtime availability."""

    lab: str
    model_families: tuple[str, ...]
    runtime_kind: str
    runtime_title: str
    runtime_url: str
    runtime_version: str
    runtime_revision: str
    runtime_entrypoint: str
    evidence_title: str
    evidence_url: str
    application_statuses: tuple[FrontierApplicationStatus, ...]
    flagship_exact: str
    limitation: str
    distribution_identity: FrontierDistributionIdentity | None
    audited_at: str
    legacy: FrontierLabRecord

    def __post_init__(self) -> None:
        scalar_fields = (
            "lab",
            "model_families",
            "runtime_kind",
            "runtime_title",
            "runtime_url",
            "runtime_version",
            "runtime_revision",
            "runtime_entrypoint",
            "evidence_title",
            "evidence_url",
            "flagship_exact",
            "limitation",
            "distribution_identity",
            "audited_at",
        )
        for name in scalar_fields:
            if getattr(self, name) != getattr(self.legacy, name):
                raise ValueError(
                    f"frontier source field {name!r} differs from legacy evidence"
                )
        if len(self.application_statuses) != len(self.legacy.application_statuses):
            raise ValueError("frontier source has incomplete application statuses")
        for status, legacy_status in zip(
            self.application_statuses,
            self.legacy.application_statuses,
            strict=True,
        ):
            expected = (
                _application(legacy_status.application),
                legacy_status.application,
                legacy_status.status,
                legacy_status.implementation_ids,
                legacy_status.boundary,
            )
            actual = (
                status.application,
                status.legacy_application,
                status.status,
                status.implementation_ids,
                status.boundary,
            )
            if actual != expected:
                raise ValueError(
                    "frontier application status differs from legacy evidence"
                )

    @property
    def applications(self) -> tuple[str, ...]:
        return tuple(value.application for value in self.application_statuses)

    @property
    def status(self) -> str:
        statuses = {value.status for value in self.application_statuses}
        return next(iter(statuses)) if len(statuses) == 1 else "mixed"

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "lab": self.lab,
            "model_families": list(self.model_families),
            "runtime_kind": self.runtime_kind,
            "runtime_title": self.runtime_title,
            "runtime_url": self.runtime_url,
            "runtime_version": self.runtime_version,
            "runtime_revision": self.runtime_revision,
            "runtime_entrypoint": self.runtime_entrypoint,
            "evidence_title": self.evidence_title,
            "evidence_url": self.evidence_url,
            "application_statuses": [
                item.as_dict() for item in self.application_statuses
            ],
            "applications": list(self.applications),
            "flagship_exact": self.flagship_exact,
            "limitation": self.limitation,
            "audited_at": self.audited_at,
        }
        if self.distribution_identity is not None:
            value["distribution_identity"] = asdict(self.distribution_identity)
        return value


def _frontier_source(record: FrontierLabRecord) -> FrontierSource:
    return FrontierSource(
        lab=record.lab,
        model_families=record.model_families,
        runtime_kind=record.runtime_kind,
        runtime_title=record.runtime_title,
        runtime_url=record.runtime_url,
        runtime_version=record.runtime_version,
        runtime_revision=record.runtime_revision,
        runtime_entrypoint=record.runtime_entrypoint,
        evidence_title=record.evidence_title,
        evidence_url=record.evidence_url,
        application_statuses=tuple(
            FrontierApplicationStatus(
                application=_application(item.application),
                legacy_application=item.application,
                status=item.status,
                implementation_ids=item.implementation_ids,
                boundary=item.boundary,
            )
            for item in record.application_statuses
        ),
        flagship_exact=record.flagship_exact,
        limitation=record.limitation,
        distribution_identity=record.distribution_identity,
        audited_at=record.audited_at,
        legacy=record,
    )


PROFILES = tuple(CatalogProfile(value) for value in _LEGACY_PROFILES)
PROFILES_BY_KEY: Mapping[str, CatalogProfile] = MappingProxyType(
    {value.key: value for value in PROFILES}
)
PROFILES_BY_APPLICATION: Mapping[str, tuple[CatalogProfile, ...]] = MappingProxyType(
    {
        application: tuple(
            value for value in PROFILES if value.application == application
        )
        for application in APPLICATIONS
    }
)
IMPLEMENTATIONS = tuple(
    value for value in PROFILES if value.catalog_kind == "implementation"
)
STUDIES = tuple(value for value in PROFILES if value.catalog_kind == "study")
GAPS = tuple(value for value in PROFILES if value.catalog_kind == "gap")
IMPLEMENTATIONS_BY_KEY: Mapping[str, CatalogProfile] = MappingProxyType(
    {value.key: value for value in IMPLEMENTATIONS}
)
STUDIES_BY_KEY: Mapping[str, CatalogProfile] = MappingProxyType(
    {value.key: value for value in STUDIES}
)
GAPS_BY_KEY: Mapping[str, CatalogProfile] = MappingProxyType(
    {value.key: value for value in GAPS}
)
IMPLEMENTATIONS_BY_APPLICATION: Mapping[str, tuple[CatalogProfile, ...]] = (
    MappingProxyType(
        {
            application: tuple(
                value for value in IMPLEMENTATIONS if value.application == application
            )
            for application in APPLICATIONS
        }
    )
)
STUDIES_BY_APPLICATION: Mapping[str, tuple[CatalogProfile, ...]] = MappingProxyType(
    {
        application: tuple(
            value for value in STUDIES if value.application == application
        )
        for application in APPLICATIONS
    }
)
GAPS_BY_APPLICATION: Mapping[str, tuple[CatalogProfile, ...]] = MappingProxyType(
    {
        application: tuple(value for value in GAPS if value.application == application)
        for application in APPLICATIONS
    }
)

FRONTIER_SOURCES = tuple(_frontier_source(value) for value in _LEGACY_FRONTIER_SOURCES)
FRONTIER_SOURCES_BY_LAB: Mapping[str, FrontierSource] = MappingProxyType(
    {value.lab: value for value in FRONTIER_SOURCES}
)


def _list(
    values: tuple[CatalogProfile, ...], application: str | None
) -> tuple[CatalogProfile, ...]:
    if application is None:
        return values
    if application not in APPLICATIONS:
        raise ValueError(
            f"unknown application {application!r}; choose from {', '.join(APPLICATIONS)}"
        )
    return tuple(value for value in values if value.application == application)


def list_profiles(application: str | None = None) -> tuple[CatalogProfile, ...]:
    return _list(PROFILES, application)


def list_implementations(
    application: str | None = None,
) -> tuple[CatalogProfile, ...]:
    return _list(IMPLEMENTATIONS, application)


def list_studies(application: str | None = None) -> tuple[CatalogProfile, ...]:
    return _list(STUDIES, application)


def list_gaps(application: str | None = None) -> tuple[CatalogProfile, ...]:
    return _list(GAPS, application)


def _profile_key(application: str, profile_id: str) -> str:
    if application not in APPLICATIONS:
        raise ValueError(
            f"unknown application {application!r}; choose from {', '.join(APPLICATIONS)}"
        )
    if "/" in profile_id:
        prefix, _, local_id = profile_id.partition("/")
        normalized_prefix = LEGACY_TO_APPLICATION.get(prefix, prefix)
        if normalized_prefix != application:
            raise ValueError(
                f"profile {profile_id!r} belongs to {normalized_prefix!r}, "
                f"not {application!r}"
            )
        profile_id = local_id
    return f"{application}/{profile_id}"


def get_profile(application: str, profile_id: str) -> CatalogProfile:
    key = _profile_key(application, profile_id)
    try:
        return PROFILES_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile_id!r} for application {application!r}"
        ) from exc


def _get_kind(application: str, profile_id: str, kind: str) -> CatalogProfile:
    value = get_profile(application, profile_id)
    if value.catalog_kind != kind:
        article = "an" if kind == "implementation" else "a"
        raise ValueError(
            f"profile {value.key!r} is a {value.catalog_kind}, not {article} {kind}"
        )
    return value


def get_implementation(application: str, profile_id: str) -> CatalogProfile:
    return _get_kind(application, profile_id, "implementation")


def get_study(application: str, profile_id: str) -> CatalogProfile:
    return _get_kind(application, profile_id, "study")


def get_gap(application: str, profile_id: str) -> CatalogProfile:
    return _get_kind(application, profile_id, "gap")


def list_frontier_sources() -> tuple[FrontierSource, ...]:
    return FRONTIER_SOURCES


def get_frontier_source(lab: str) -> FrontierSource:
    try:
        return FRONTIER_SOURCES_BY_LAB[lab]
    except KeyError as exc:
        raise ValueError(f"unknown frontier source {lab!r}") from exc


def _validate_catalog_links() -> None:
    for source in FRONTIER_SOURCES:
        if set(source.applications) != set(APPLICATIONS):
            raise RuntimeError(
                f"frontier source {source.lab!r} has incomplete coverage"
            )
        for status in source.application_statuses:
            for profile_id in status.implementation_ids:
                implementation = get_implementation(status.application, profile_id)
                if implementation.status != "runnable":
                    raise RuntimeError(
                        f"frontier link {implementation.key!r} is not runnable"
                    )


_validate_catalog_links()


__all__ = [
    "APPLICATIONS",
    "APPLICATION_TO_LEGACY",
    "FRONTIER_MANIFEST_AS_OF",
    "FRONTIER_SOURCES",
    "FRONTIER_SOURCES_BY_LAB",
    "GAPS",
    "GAPS_BY_APPLICATION",
    "GAPS_BY_KEY",
    "IMPLEMENTATIONS",
    "IMPLEMENTATIONS_BY_APPLICATION",
    "IMPLEMENTATIONS_BY_KEY",
    "LEGACY_TO_APPLICATION",
    "PROFILES",
    "PROFILES_BY_APPLICATION",
    "PROFILES_BY_KEY",
    "STUDIES",
    "STUDIES_BY_APPLICATION",
    "STUDIES_BY_KEY",
    "CatalogProfile",
    "FrontierApplicationStatus",
    "FrontierSource",
    "get_frontier_source",
    "get_gap",
    "get_implementation",
    "get_profile",
    "get_study",
    "list_frontier_sources",
    "list_gaps",
    "list_implementations",
    "list_profiles",
    "list_studies",
]
