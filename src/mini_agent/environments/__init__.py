"""Domain environments, re-exported lazily.

PEP 562 lazy loading keeps ``mini_agent.agent`` (and anything else that only
needs ``environments.base``) from importing every domain module and its
transport dependencies at package-import time.
"""

from __future__ import annotations

from typing import Any

from .base import BaseEnvironment, Environment

_LAZY_EXPORTS = {
    "AdapterLiveState": "cua",
    "CUAEnvironment": "cua",
    "CUASpeedRunAdapterClient": "cua",
    "CUASpeedRunClient": "cua",
    "ComputerObservation": "cua",
    "OSWorldClient": "cua",
    "OSWorldEnvironment": "cua",
    "OSWorldLiveState": "cua",
    "BashEnvironment": "swe",
    "SWEPatchState": "swe",
    "ApptainerSWEEnvironment": "swebench",
    "DockerSWEEnvironment": "swebench",
    "SWEbenchImageBinding": "swebench",
    "resolve_swebench_image_binding": "swebench",
    "swebench_doctor": "swebench",
    "swebench_image_name": "swebench",
    "BrowserEnvironment": "web",
    "BrowserSessionState": "web",
    "BrowseCompPlusBackend": "web",
    "HttpPageReader": "web",
    "JsonlSearchBackend": "web",
    "PlaywrightPageReader": "web",
    "SerpAPIBackend": "web",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


__all__ = ["BaseEnvironment", "Environment", *sorted(_LAZY_EXPORTS)]
