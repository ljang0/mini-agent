"""Shared fidelity policy for caller-forwarded process environment names."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


_RUNTIME_INJECTION_NAMES = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_WORK_TREE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTDOC",
        "RUSTFLAGS",
    }
)

_RUNTIME_INJECTION_PREFIXES = (
    "DYLD_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)


def reject_runtime_environment_overrides(
    names: Sequence[str],
    *,
    label: str,
    reserved_names: Iterable[str] = (),
    reserved_prefixes: Iterable[str] = (),
) -> None:
    """Reject variables that can replace code or silently change a pinned runtime."""

    exact = _RUNTIME_INJECTION_NAMES | {name.upper() for name in reserved_names}
    prefixes = _RUNTIME_INJECTION_PREFIXES + tuple(
        prefix.upper() for prefix in reserved_prefixes
    )
    rejected = sorted(
        {
            name
            for name in names
            if name.upper() in exact
            or any(name.upper().startswith(prefix) for prefix in prefixes)
        }
    )
    if rejected:
        raise ValueError(
            f"{label} pass_env cannot override runtime-control or loader variables: "
            + ", ".join(rejected)
        )
