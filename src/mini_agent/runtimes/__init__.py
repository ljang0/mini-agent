"""Benchmark-neutral sandbox runtimes, re-exported lazily.

``base`` is the only eagerly imported module so that importing a runtime
protocol never drags in Docker or Apptainer provisioning code.
"""

from __future__ import annotations

from typing import Any

from .base import ProcessResult, ProcessRunner, SandboxRuntime

_LAZY_EXPORTS = {
    "ApptainerRuntime": "apptainer",
    "apptainer_exec_argv": "apptainer",
    "apptainer_image_identity": "apptainer",
    "materialize_apptainer_image": "apptainer",
    "DockerRuntime": "docker",
    "docker_doctor": "docker",
    "docker_image_id": "docker",
    "docker_security_is_rootless": "docker",
    "LocalProcessRunner": "local",
    "LocalRuntime": "local",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


__all__ = [
    "ProcessResult",
    "ProcessRunner",
    "SandboxRuntime",
    *sorted(_LAZY_EXPORTS),
]
