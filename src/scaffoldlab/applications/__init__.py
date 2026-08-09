"""Domain-first application and implementation catalog."""

from .base import (
    APPLICATION_NAMES,
    ApplicationSelection,
    HarnessSignature,
    ImplementationProfile,
    SourceArtifact,
    normalize_harness_specs,
)
from .registry import (
    APPLICATIONS,
    IMPLEMENTATIONS,
    IMPLEMENTATIONS_BY_KEY,
    get_implementation,
    list_applications,
    list_implementations,
    resolve_application_config,
)

__all__ = [
    "APPLICATION_NAMES",
    "APPLICATIONS",
    "IMPLEMENTATIONS",
    "IMPLEMENTATIONS_BY_KEY",
    "ApplicationSelection",
    "HarnessSignature",
    "ImplementationProfile",
    "SourceArtifact",
    "get_implementation",
    "list_applications",
    "list_implementations",
    "normalize_harness_specs",
    "resolve_application_config",
]
