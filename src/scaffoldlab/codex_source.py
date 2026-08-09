"""Source-first adapter for OpenAI Codex's native multi-agent runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import stat
import tarfile
import tempfile
import weakref
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from .external import (
    _CapturedOutput,
    _ProcessOutputLimitExceeded,
    _communicate_limited,
    _git_workspace_provenance,
    _prepare_executable,
    _process_environment,
    _terminate_process_tree,
    _workspace_tree_sha256,
    _workspace_tree_sha256_async,
)
from .environment_policy import reject_runtime_environment_overrides
from .environments.swe import SWEPatchPayload
from .providers import ProviderError
from .source_integrity import (
    copytree_ignore_git_metadata,
    reject_case_variant_git_metadata,
    require_standalone_git_checkout,
)
from .types import ModelRequest, ModelResponse, Usage


CODEX_SOURCE_VERSION = "0.147.0"
CODEX_SOURCE_REVISION = "be6e8eac029b183056b7e4402879f15d2c85f61b"
CODEX_SOURCE_REPOSITORY = "https://github.com/openai/codex"
CODEX_SOURCE_TAG = "rust-v0.147.0"
CODEX_CARGO_LOCK_SHA256 = (
    "eeab4e9d3466da54037032251e2f13ad1ed11eae18bb8ee7dd2c89dbb86f645d"
)
CODEX_RUST_TOOLCHAIN_SHA256 = (
    "570656042681cfd8795403a455baf9a33035331a07db0645e866bbcea89a3d64"
)
CODEX_RUST_TOOLCHAIN = "1.95.0"

_MAX_SUBAGENTS = 64
_MAX_DEPTH = 16
_MAX_WAIT_SECONDS = 3600.0
_V2_SOURCE_MIN_WAIT_MS = 10_000
_V2_SOURCE_MAX_WAIT_MS = 3_600_000
_V2_SOURCE_DEFAULT_WAIT_MS = 30_000
_DEFAULT_WORKSPACE_ENTRIES = 250_000
_DEFAULT_WORKSPACE_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_WORKSPACE_HASH_TIMEOUT = 30.0
_DEFAULT_SOURCE_ARCHIVE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_PATCH_BYTES = 8 * 1024 * 1024
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", re.IGNORECASE
)
_CODEX_SYSTEM_CONFIGURATION_PATHS = (
    Path("/etc/codex/config.toml"),
    Path("/etc/codex/requirements.toml"),
    Path("/etc/codex/managed_config.toml"),
)


def _is_rustup_proxy(identity: Mapping[str, Any]) -> bool:
    resolved = identity.get("resolved_path")
    return isinstance(resolved, str) and Path(resolved).name.casefold() in {
        "rustup",
        "rustup-init",
    }


def _present_codex_system_configuration() -> list[str]:
    return [
        str(path) for path in _CODEX_SYSTEM_CONFIGURATION_PATHS if os.path.lexists(path)
    ]


def _reject_codex_system_configuration(*, runtime_error: bool) -> None:
    present = _present_codex_system_configuration()
    if not present:
        return
    error_type = ProviderError if runtime_error else ValueError
    raise error_type(
        "Codex source runtime refuses host system/managed configuration layers: "
        + ", ".join(present)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Optional[str], name: str) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters or null")


def _validate_positive_finite(value: float, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _validate_embedded_git_metadata(root: Path, *, max_entries: int) -> None:
    """Validate worktree naming before all inherited Git metadata is stripped."""

    reject_case_variant_git_metadata(
        root, label="Codex seed workspace", max_entries=max_entries
    )


def _validate_source_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    cargo_lock: Path,
    expected_cargo_lock_sha256: str,
    rust_toolchain: Path,
    expected_rust_toolchain_sha256: str,
    runtime_error: bool = False,
) -> Mapping[str, Any]:
    error_type = ProviderError if runtime_error else ValueError
    try:
        require_standalone_git_checkout(checkout, label="Codex source checkout")
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    git_state = _git_workspace_provenance(checkout)
    if git_state.get("available") is not True:
        raise error_type("Codex source checkout must be a readable Git working tree")
    if git_state.get("head") != expected_revision:
        raise error_type(
            "Codex source checkout revision mismatch: expected "
            f"{expected_revision!r}, observed {git_state.get('head')!r}"
        )
    if git_state.get("dirty") is not False:
        raise error_type("Codex source checkout must remain clean")
    try:
        lock_sha256 = _file_sha256(cargo_lock)
        toolchain_sha256 = _file_sha256(rust_toolchain)
    except OSError as exc:
        raise error_type("Codex source identity files are unreadable") from exc
    if lock_sha256 != expected_cargo_lock_sha256:
        raise error_type(
            "Codex Cargo.lock SHA-256 mismatch: expected "
            f"{expected_cargo_lock_sha256}, observed {lock_sha256}"
        )
    if toolchain_sha256 != expected_rust_toolchain_sha256:
        raise error_type(
            "Codex rust-toolchain.toml SHA-256 mismatch: expected "
            f"{expected_rust_toolchain_sha256}, observed {toolchain_sha256}"
        )
    return {
        "git": dict(git_state),
        "cargo_lock_sha256": lock_sha256,
        "rust_toolchain_sha256": toolchain_sha256,
    }


def _extract_verified_git_archive(
    archive: Path,
    destination: Path,
    *,
    max_entries: int,
    max_bytes: int,
) -> None:
    """Extract a local ``git archive`` without permitting path traversal."""

    with tarfile.open(archive, mode="r:") as stream:
        members = stream.getmembers()
        if len(members) > max_entries:
            raise ProviderError("Codex Git archive exceeds its entry limit")
        if sum(member.size for member in members if member.isfile()) > max_bytes:
            raise ProviderError("Codex Git archive exceeds its expanded-byte limit")
        normalized_names: set[str] = set()
        symlink_names: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            normalized_name = posixpath.normpath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or member.name.rstrip("/") != normalized_name
                or normalized_name in normalized_names
            ):
                raise ProviderError("Codex Git archive contains an unsafe path")
            normalized_names.add(normalized_name)
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ProviderError("Codex Git archive contains a special file")
            if member.issym():
                symlink_names.add(normalized_name)
                link = PurePosixPath(member.linkname)
                target = path.parent / link
                normalized = PurePosixPath(posixpath.normpath(str(target)))
                if normalized.is_absolute() or (
                    normalized.parts and normalized.parts[0] == ".."
                ):
                    raise ProviderError(
                        "Codex Git archive contains an escaping link target"
                    )
        for member in members:
            normalized_name = posixpath.normpath(member.name)
            parts = PurePosixPath(normalized_name).parts
            if any(
                str(PurePosixPath(*parts[:index])) in symlink_names
                for index in range(1, len(parts))
            ):
                raise ProviderError(
                    "Codex Git archive places content beneath a symbolic link"
                )
            if member.issym():
                normalized_target_name = posixpath.normpath(
                    str(PurePosixPath(normalized_name).parent / member.linkname)
                )
                if normalized_target_name in symlink_names:
                    raise ProviderError(
                        "Codex Git archive contains a symbolic-link chain"
                    )
        stream.extractall(destination)


def _toml_string(path: Path, *, table: str, key: str, label: str) -> Optional[str]:
    """Read the simple pinned version/toolchain strings without a new dependency."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read Codex {label}") from exc
    active_table: Optional[str] = None
    table_pattern = re.compile(r"^\s*\[\s*([^\]]+)\s*\]\s*(?:#.*)?$")
    value_pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*(?:#.*)?$')
    for line in lines:
        table_match = table_pattern.fullmatch(line)
        if table_match is not None:
            active_table = table_match.group(1).strip()
            continue
        if active_table == table:
            value_match = value_pattern.fullmatch(line)
            if value_match is not None:
                return value_match.group(1)
    return None


def _usage_from_events(events: Sequence[Mapping[str, Any]]) -> Usage:
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not completed:
        return Usage(cost_known=False, complete=False)
    raw = completed[-1].get("usage")
    if not isinstance(raw, Mapping):
        raise ProviderError("Codex turn.completed usage must be an object")
    raw_usage: Mapping[str, Any] = raw

    def count(name: str) -> int:
        value = raw_usage.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProviderError(f"Codex emitted invalid {name!r} usage")
        return value

    input_tokens = count("input_tokens")
    cache_read = count("cached_input_tokens")
    cache_write = count("cache_write_input_tokens")
    if cache_read + cache_write > input_tokens:
        raise ProviderError("Codex cache token classes exceed input_tokens")
    return Usage(
        input_tokens=input_tokens,
        output_tokens=count("output_tokens"),
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cost_known=False,
        # exec reports the root turn total. Public source does not establish that
        # this is complete whole-tree accounting for every child and side call.
        complete=False,
    )


def _parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    decoded = data.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for line in decoded.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "Codex exec emitted a non-JSON stdout line",
                raw={"stdout": decoded},
            ) from exc
        if not isinstance(event, dict):
            raise ProviderError("Codex exec emitted a non-object JSONL event")
        events.append(event)
    if not events or events[0].get("type") != "thread.started":
        raise ProviderError("Codex exec JSONL did not start with thread.started")
    return events


def _successful_collaboration_calls(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "collab_tool_call":
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id in seen_ids:
            continue
        tool = item.get("tool")
        status = item.get("status")
        receiver_thread_ids = item.get("receiver_thread_ids")
        if not isinstance(tool, str) or not isinstance(status, str):
            raise ProviderError("Codex emitted a malformed collaboration item")
        if not isinstance(receiver_thread_ids, list) or any(
            not isinstance(thread_id, str) or not thread_id
            for thread_id in receiver_thread_ids
        ):
            raise ProviderError(
                "Codex emitted malformed collaboration receiver thread IDs"
            )
        if status == "completed":
            calls.append(
                {
                    "tool": tool,
                    "receiver_thread_ids": list(receiver_thread_ids),
                }
            )
        if isinstance(item_id, str):
            seen_ids.add(item_id)
    return calls


class CodexSourceBackend:
    """Build and run the pinned public Codex source without reconstructing it.

    One call is one native ``codex exec`` session. Codex owns its model loop,
    prompts, tools, compaction, and subagent scheduler. The adapter supplies a fresh
    copy of the seed workspace, a fresh CODEX_HOME, bounded process I/O, and exact
    source/build identity checks. Enabling collaboration tools does not prove that a
    run delegated: ``multi_agent_execution_observed`` is true only after an upstream
    completed ``spawn_agent`` event.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        workspace: Path,
        model: str,
        reasoning_effort: Optional[str] = None,
        api_key_env: str,
        auth_target_env: str = "CODEX_API_KEY",
        cargo_executable: str = "cargo",
        rustc_executable: Optional[str] = None,
        git_executable: str = "git",
        multi_agent_version: str = "v1",
        max_subagents: Optional[int] = None,
        max_depth: int = 1,
        max_wait_seconds: Optional[float] = None,
        timeout_seconds: float = 1800.0,
        build_timeout_seconds: float = 3600.0,
        max_output_bytes: int = 16 * 1024 * 1024,
        max_prompt_bytes: int = 1024 * 1024,
        max_source_archive_bytes: int = _DEFAULT_SOURCE_ARCHIVE_BYTES,
        max_patch_bytes: int = _DEFAULT_MAX_PATCH_BYTES,
        pass_env: Sequence[str] = (),
        allow_sensitive_environment: bool = False,
        expected_revision: str = CODEX_SOURCE_REVISION,
        expected_version: str = CODEX_SOURCE_VERSION,
        expected_cargo_lock_sha256: str = CODEX_CARGO_LOCK_SHA256,
        expected_rust_toolchain_sha256: str = CODEX_RUST_TOOLCHAIN_SHA256,
        expected_executable_sha256: Optional[str] = None,
        expected_cargo_sha256: Optional[str] = None,
        expected_rustc_sha256: Optional[str] = None,
        expected_git_sha256: Optional[str] = None,
        workspace_hash_max_entries: int = _DEFAULT_WORKSPACE_ENTRIES,
        workspace_hash_max_bytes: int = _DEFAULT_WORKSPACE_BYTES,
        workspace_hash_timeout_seconds: float = _DEFAULT_WORKSPACE_HASH_TIMEOUT,
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Codex source adapter currently requires POSIX process groups"
            )
        self.checkout = checkout.resolve()
        self.workspace = workspace.resolve()
        if not self.checkout.is_dir():
            raise ValueError(
                f"Codex source checkout is not a directory: {self.checkout}"
            )
        if not self.workspace.is_dir():
            raise ValueError(
                f"Codex seed workspace is not a directory: {self.workspace}"
            )
        if (
            self.checkout == self.workspace
            or self.checkout.is_relative_to(self.workspace)
            or self.workspace.is_relative_to(self.checkout)
        ):
            raise ValueError(
                "Codex source checkout and seed workspace must be disjoint"
            )
        for name, value in (
            ("model", model),
            ("api_key_env", api_key_env),
            ("cargo_executable", cargo_executable),
            ("git_executable", git_executable),
        ):
            if (
                not isinstance(value, str)
                or not value
                or "\x00" in value
                or "=" in value
            ):
                raise ValueError(f"{name} must be a non-empty safe string")
        if rustc_executable is not None and (
            not isinstance(rustc_executable, str)
            or not rustc_executable
            or "\x00" in rustc_executable
            or "=" in rustc_executable
        ):
            raise ValueError("rustc_executable must be a non-empty safe string or null")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", reasoning_effort) is None
        ):
            raise ValueError(
                "reasoning_effort must be a non-empty simple string or null"
            )
        if auth_target_env != "CODEX_API_KEY":
            raise ValueError(
                "auth_target_env must be CODEX_API_KEY for the pinned default "
                "OpenAI provider"
            )
        if multi_agent_version not in {"v1", "v2"}:
            raise ValueError("multi_agent_version must be v1 or v2")
        effective_max_subagents = (
            max_subagents
            if max_subagents is not None
            else (6 if multi_agent_version == "v1" else 3)
        )
        for limit_name, limit_value, maximum in (
            ("max_subagents", effective_max_subagents, _MAX_SUBAGENTS),
            ("max_depth", max_depth, _MAX_DEPTH),
        ):
            if (
                not isinstance(limit_value, int)
                or isinstance(limit_value, bool)
                or limit_value < 1
                or limit_value > maximum
            ):
                raise ValueError(
                    f"{limit_name} must be an integer from 1 through {maximum}"
                )
        self.timeout_seconds = _validate_positive_finite(
            timeout_seconds, "timeout_seconds"
        )
        self.build_timeout_seconds = _validate_positive_finite(
            build_timeout_seconds, "build_timeout_seconds"
        )
        self.max_wait_seconds = (
            _validate_positive_finite(max_wait_seconds, "max_wait_seconds")
            if max_wait_seconds is not None
            else None
        )
        if (
            self.max_wait_seconds is not None
            and self.max_wait_seconds > _MAX_WAIT_SECONDS
        ):
            raise ValueError(f"max_wait_seconds cannot exceed {_MAX_WAIT_SECONDS:g}")
        self.workspace_hash_timeout_seconds = _validate_positive_finite(
            workspace_hash_timeout_seconds, "workspace_hash_timeout_seconds"
        )
        for limit_name, limit_value, minimum in (
            ("max_output_bytes", max_output_bytes, 1024),
            ("max_prompt_bytes", max_prompt_bytes, 1),
            ("max_source_archive_bytes", max_source_archive_bytes, 1024 * 1024),
            ("max_patch_bytes", max_patch_bytes, 1024),
            ("workspace_hash_max_entries", workspace_hash_max_entries, 1),
            ("workspace_hash_max_bytes", workspace_hash_max_bytes, 1),
        ):
            if (
                not isinstance(limit_value, int)
                or isinstance(limit_value, bool)
                or limit_value < minimum
            ):
                raise ValueError(
                    f"{limit_name} must be an integer of at least {minimum}"
                )
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        sensitive_passed = [
            name for name in pass_env if _SENSITIVE_ENVIRONMENT_NAME.search(name)
        ]
        if sensitive_passed:
            raise ValueError(
                "pass_env cannot expose credential-like names to Codex shell tools: "
                + ", ".join(sorted(sensitive_passed))
            )
        if api_key_env in pass_env:
            raise ValueError(
                "pass_env cannot expose the Codex credential alias to shell tools"
            )
        reject_runtime_environment_overrides(
            pass_env,
            label="Codex source",
            reserved_prefixes=("CODEX_", "OPENAI_"),
        )
        if not allow_sensitive_environment:
            raise ValueError(
                "native Codex and its subagents use the model credential; acknowledge "
                "this only with an isolated outer runner"
            )
        _reject_codex_system_configuration(runtime_error=False)
        if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
            raise ValueError("expected_revision must be 40 lowercase hex characters")
        if re.fullmatch(r"\d+\.\d+\.\d+", expected_version) is None:
            raise ValueError("expected_version must be a canonical semantic version")
        for digest_value, digest_name in (
            (expected_cargo_lock_sha256, "expected_cargo_lock_sha256"),
            (expected_rust_toolchain_sha256, "expected_rust_toolchain_sha256"),
            (expected_executable_sha256, "expected_executable_sha256"),
            (expected_cargo_sha256, "expected_cargo_sha256"),
            (expected_rustc_sha256, "expected_rustc_sha256"),
            (expected_git_sha256, "expected_git_sha256"),
        ):
            _validate_sha256(digest_value, digest_name)

        self.codex_rs = self.checkout / "codex-rs"
        self.cargo_lock = self.codex_rs / "Cargo.lock"
        self.workspace_manifest = self.codex_rs / "Cargo.toml"
        self.cli_manifest = self.codex_rs / "cli" / "Cargo.toml"
        self.rust_toolchain = self.codex_rs / "rust-toolchain.toml"
        for path, label in (
            (self.cargo_lock, "codex-rs/Cargo.lock"),
            (self.workspace_manifest, "codex-rs/Cargo.toml"),
            (self.cli_manifest, "codex-rs/cli/Cargo.toml"),
            (self.rust_toolchain, "codex-rs/rust-toolchain.toml"),
        ):
            if not path.is_file():
                raise ValueError(f"Codex source checkout is missing {label}")

        workspace_version = _toml_string(
            self.workspace_manifest,
            table="workspace.package",
            key="version",
            label="workspace manifest",
        )
        if workspace_version != expected_version:
            raise ValueError(
                "Codex workspace version mismatch: expected "
                f"{expected_version!r}, observed {workspace_version!r}"
            )
        observed_toolchain = _toml_string(
            self.rust_toolchain,
            table="toolchain",
            key="channel",
            label="rust toolchain manifest",
        )
        if observed_toolchain != CODEX_RUST_TOOLCHAIN:
            raise ValueError(
                "Codex Rust toolchain mismatch: expected "
                f"{CODEX_RUST_TOOLCHAIN!r}, observed {observed_toolchain!r}"
            )

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key_env = api_key_env
        self.auth_target_env = auth_target_env
        self.cargo_executable = cargo_executable
        self.rustc_executable = rustc_executable
        self.git_executable = git_executable
        self.multi_agent_version = multi_agent_version
        self.max_subagents = effective_max_subagents
        self.max_depth = max_depth
        self.max_output_bytes = max_output_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_source_archive_bytes = max_source_archive_bytes
        self.max_patch_bytes = max_patch_bytes
        self.pass_env = tuple(pass_env)
        self.allow_sensitive_environment = allow_sensitive_environment
        self.expected_revision = expected_revision
        self.expected_version = expected_version
        self.expected_cargo_lock_sha256 = expected_cargo_lock_sha256
        self.expected_rust_toolchain_sha256 = expected_rust_toolchain_sha256
        self.expected_executable_sha256 = expected_executable_sha256
        self.expected_cargo_sha256 = expected_cargo_sha256
        self.expected_rustc_sha256 = expected_rustc_sha256
        self.expected_git_sha256 = expected_git_sha256
        self.workspace_hash_max_entries = workspace_hash_max_entries
        self.workspace_hash_max_bytes = workspace_hash_max_bytes
        _validate_embedded_git_metadata(
            self.workspace, max_entries=self.workspace_hash_max_entries
        )
        self._build_lock = asyncio.Lock()
        self._built = False
        self._cargo_identity: Mapping[str, Any] = {"available": False}
        self._rustc_identity: Mapping[str, Any] = {"available": False}
        self._git_identity: Mapping[str, Any] = {"available": False}
        self._executable_identity: Mapping[str, Any] = {"available": False}
        self._cargo_version_output: Optional[str] = None
        self._rustc_version_output: Optional[str] = None
        self._version_output: Optional[str] = None
        self._last_source_state = _validate_source_checkout(
            self.checkout,
            expected_revision=self.expected_revision,
            cargo_lock=self.cargo_lock,
            expected_cargo_lock_sha256=self.expected_cargo_lock_sha256,
            rust_toolchain=self.rust_toolchain,
            expected_rust_toolchain_sha256=self.expected_rust_toolchain_sha256,
        )
        self.checkout_git = dict(self._last_source_state["git"])
        self.workspace_tree_sha256 = _workspace_tree_sha256(
            self.workspace,
            max_entries=self.workspace_hash_max_entries,
            max_bytes=self.workspace_hash_max_bytes,
            timeout_seconds=self.workspace_hash_timeout_seconds,
        )
        self.git_workspace = {
            "available": os.path.lexists(self.workspace / ".git"),
            "metadata_inherited_into_trial": False,
        }
        # Never trust an ignored target/ artifact in the checkout. A new target
        # directory forces the pinned sources to produce the binary this adapter
        # executes, while keeping the checkout itself clean.
        self.build_root = Path(
            tempfile.mkdtemp(prefix="scaffoldlab-codex-build-")
        ).resolve()
        self._build_cleanup = weakref.finalize(
            self,
            shutil.rmtree,
            self.build_root,
            ignore_errors=True,
        )
        self.build_target_dir = self.build_root / "target"
        self.build_source_root = self.build_root / "source"
        self.build_codex_rs = self.build_source_root / "codex-rs"
        self.cargo_home = self.build_root / "cargo-home"
        self.build_home = self.build_root / "home"
        self.build_tmp = self.build_root / "tmp"
        self.build_xdg_config = self.build_root / "xdg-config"
        for directory in (
            self.build_target_dir,
            self.cargo_home,
            self.build_home,
            self.build_tmp,
            self.build_xdg_config,
        ):
            directory.mkdir(mode=0o700)
        self.executable = self.build_target_dir / "release" / "codex"
        self._source_exported = False
        self._source_export_tree_sha256: Optional[str] = None

    def close(self) -> None:
        """Remove the private Cargo target after the backend is no longer in use."""

        self._build_cleanup()

    async def prepare_for_manifest(self) -> None:
        """Resolve and build every lazy identity before run fingerprinting."""

        _reject_codex_system_configuration(runtime_error=True)
        await self._ensure_built()
        self._validate_source(runtime_error=True)
        _validate_embedded_git_metadata(
            self.workspace, max_entries=self.workspace_hash_max_entries
        )
        if (
            await self._workspace_hash(
                self.workspace, "the Codex seed workspace before manifest capture"
            )
            != self.workspace_tree_sha256
        ):
            raise ProviderError(
                "Codex seed workspace changed before manifest fingerprinting"
            )

    def provenance(self) -> Mapping[str, Any]:
        exact_public_pin = (
            self.expected_revision == CODEX_SOURCE_REVISION
            and self.expected_version == CODEX_SOURCE_VERSION
            and self.expected_cargo_lock_sha256 == CODEX_CARGO_LOCK_SHA256
            and self.expected_rust_toolchain_sha256 == CODEX_RUST_TOOLCHAIN_SHA256
        )
        reproducible_build_pins = all(
            value is not None
            for value in (
                self.expected_git_sha256,
                self.expected_cargo_sha256,
                self.expected_rustc_sha256,
                self.expected_executable_sha256,
            )
        )
        exact_runtime_build_verified = (
            exact_public_pin
            and reproducible_build_pins
            and self._source_exported
            and self._built
            and self._executable_identity.get("available") is True
        )
        return {
            "provider": "openai-codex-upstream-source",
            "repository": CODEX_SOURCE_REPOSITORY,
            "tag": CODEX_SOURCE_TAG,
            "checkout": str(self.checkout),
            "checkout_git": dict(self.checkout_git),
            "last_verified_source_state": dict(self._last_source_state),
            "expected_revision": self.expected_revision,
            "expected_version": self.expected_version,
            "expected_cargo_lock_sha256": self.expected_cargo_lock_sha256,
            "expected_rust_toolchain_sha256": self.expected_rust_toolchain_sha256,
            "rust_toolchain": CODEX_RUST_TOOLCHAIN,
            "runtime_source_identity_verified": (
                exact_public_pin and self._source_exported
            ),
            "source_or_protocol_pin_verified": exact_public_pin,
            "bit_reproducible_runtime_verified": False,
            "reproducible_build_pins_supplied": reproducible_build_pins,
            "exact_runtime_build_verified": exact_runtime_build_verified,
            # Cloud policy can still be supplied by the service and macOS managed
            # preferences are outside a portable filesystem attestation. Even a
            # fully hash-pinned local build is therefore not called an exact
            # end-to-end runtime boundary.
            "exact_runtime_boundary_verified": False,
            "flagship_system_card_parity_claimed": False,
            "build_command": [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--bin",
                "codex",
            ],
            "build_target_isolation": "fresh-temporary-CARGO_TARGET_DIR",
            "build_configuration_isolation": "fresh-CARGO_HOME-and-XDG_CONFIG_HOME",
            "build_source_isolation": "git-archive-of-pinned-commit",
            "build_source_tree_sha256": self._source_export_tree_sha256,
            "git": dict(self._git_identity),
            "expected_git_sha256": self.expected_git_sha256,
            "cargo": dict(self._cargo_identity),
            "expected_cargo_sha256": self.expected_cargo_sha256,
            "cargo_version_output": self._cargo_version_output,
            "rustc": dict(self._rustc_identity),
            "expected_rustc_sha256": self.expected_rustc_sha256,
            "rustc_version_output": self._rustc_version_output,
            "runtime_executable": dict(self._executable_identity),
            "expected_executable_sha256": self.expected_executable_sha256,
            "version_output": self._version_output,
            "workspace": str(self.workspace),
            "workspace_isolation": "fresh-disposable-copy-and-git-baseline-per-call",
            "workspace_git_baseline": (
                "fresh standalone repository; inherited Git metadata and history "
                "are stripped"
            ),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "model_catalog_snapshot_pinned": False,
            "system_configuration_preflight": "absent-at-construction-and-launch",
            "macos_managed_preferences_attested_absent": False,
            "cloud_configuration_attested_absent": False,
            "api_key_environment_name": self.api_key_env,
            "auth_target_environment_name": self.auth_target_env,
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "credential_removed_from_direct_shell_environment": True,
            "credential_process_isolation_guaranteed": False,
            "credential_isolation_scope": (
                "fresh process homes plus direct shell-environment exclusion; "
                "the caller must still isolate the outer runner"
            ),
            "headless_protocol": "codex exec --json --ephemeral",
            "requested_multi_agent_version": self.multi_agent_version,
            "effective_multi_agent_version": (
                "v2" if self.multi_agent_version == "v2" else None
            ),
            "effective_multi_agent_version_verified": (
                self.multi_agent_version == "v2"
            ),
            "effective_version_evidence": (
                "explicit V2 feature override wins model_info in pinned source"
                if self.multi_agent_version == "v2"
                else (
                    "unverified: remote model_info can select V2 despite the local "
                    "V1 feature request"
                )
            ),
            "native_multi_agent_tools_enabled": True,
            "max_subagents": self.max_subagents,
            "configured_v1_max_depth": self.max_depth,
            "max_depth": None,
            "v2_total_concurrency_including_root": (
                self.max_subagents + 1 if self.multi_agent_version == "v2" else None
            ),
            "configured_max_wait_seconds": self.max_wait_seconds,
            "effective_v2_min_wait_timeout_ms": (
                self._v2_wait_timeouts()[0]
                if self.multi_agent_version == "v2"
                else None
            ),
            "effective_v2_max_wait_timeout_ms": (
                self._v2_wait_timeouts()[1]
                if self.multi_agent_version == "v2"
                else None
            ),
            "effective_v2_default_wait_timeout_ms": (
                self._v2_wait_timeouts()[2]
                if self.multi_agent_version == "v2"
                else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "build_timeout_seconds": self.build_timeout_seconds,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_source_archive_bytes": self.max_source_archive_bytes,
            "max_patch_bytes": self.max_patch_bytes,
            "one_backend_call_is_external_session_tree": True,
            "usage_scope": (
                "root codex exec turn total; public source does not establish "
                "complete accounting for every nested or side call"
            ),
        }

    def _validate_source(self, *, runtime_error: bool = False) -> Mapping[str, Any]:
        state = _validate_source_checkout(
            self.checkout,
            expected_revision=self.expected_revision,
            cargo_lock=self.cargo_lock,
            expected_cargo_lock_sha256=self.expected_cargo_lock_sha256,
            rust_toolchain=self.rust_toolchain,
            expected_rust_toolchain_sha256=self.expected_rust_toolchain_sha256,
            runtime_error=runtime_error,
        )
        self._last_source_state = state
        return state

    async def _run_bounded_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        label: str,
        input_data: Optional[bytes] = None,
        output_limit: Optional[int] = None,
    ) -> tuple[asyncio.subprocess.Process, bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=dict(environment),
                stdin=(
                    asyncio.subprocess.PIPE
                    if input_data is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ProviderError(f"{label} executable was not found") from exc
        captured = _CapturedOutput()
        try:
            stdout, stderr = await asyncio.wait_for(
                _communicate_limited(
                    process,
                    input_data=input_data,
                    max_output_bytes=output_limit or self.max_output_bytes,
                    captured=captured,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_tree(process)
            raise ProviderError(
                f"{label} timed out",
                usage=Usage(cost_known=False, complete=False),
                raw={
                    "stdout": bytes(captured.stdout).decode("utf-8", errors="replace"),
                    "stderr": bytes(captured.stderr).decode("utf-8", errors="replace"),
                },
            ) from exc
        except asyncio.CancelledError as exc:
            await _terminate_process_tree(process)
            exc.usage = Usage(cost_known=False, complete=False)  # type: ignore[attr-defined]
            raise
        except _ProcessOutputLimitExceeded as exc:
            await _terminate_process_tree(process)
            raise ProviderError(
                f"{label} exceeded its output limit",
                usage=Usage(cost_known=False, complete=False),
                raw={
                    "stdout": bytes(captured.stdout).decode("utf-8", errors="replace"),
                    "stderr": bytes(captured.stderr).decode("utf-8", errors="replace"),
                },
            ) from exc
        except Exception:
            await _terminate_process_tree(process)
            raise
        await _terminate_process_tree(process)
        return process, stdout, stderr

    async def _ensure_source_export(self) -> None:
        if self._source_exported:
            return
        git, git_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Codex source Git",
        )
        if git_identity.get("available") is not True:
            raise ProviderError(
                f"Codex source Git executable was not found: {self.git_executable!r}"
            )
        invoked = git_identity.get("invoked_path")
        if isinstance(invoked, str):
            git = invoked
        self._git_identity = dict(git_identity)
        archive = self.build_root / "source.tar"
        environment = _process_environment(())
        environment.update(
            {
                "HOME": str(self.build_home),
                "XDG_CONFIG_HOME": str(self.build_xdg_config),
                "TMPDIR": str(self.build_tmp),
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        process, stdout, stderr = await self._run_bounded_process(
            [
                git,
                "-C",
                str(self.checkout),
                "archive",
                "--format=tar",
                self.expected_revision,
            ],
            cwd=self.build_root,
            environment=environment,
            timeout_seconds=min(self.build_timeout_seconds, 300.0),
            label="Codex pinned source export",
            output_limit=self.max_source_archive_bytes,
        )
        self._validate_source(runtime_error=True)
        if process.returncode != 0 or not stdout:
            raise ProviderError(
                "Codex could not export the pinned source commit",
                raw={
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        archive.write_bytes(stdout)
        archive.chmod(0o600)
        self.build_source_root.mkdir(mode=0o700)
        try:
            await asyncio.to_thread(
                _extract_verified_git_archive,
                archive,
                self.build_source_root,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.max_source_archive_bytes,
            )
        finally:
            archive.unlink(missing_ok=True)
        exported_lock = self.build_codex_rs / "Cargo.lock"
        exported_toolchain = self.build_codex_rs / "rust-toolchain.toml"
        exported_manifest = self.build_codex_rs / "Cargo.toml"
        for path, expected, label in (
            (
                exported_lock,
                self.expected_cargo_lock_sha256,
                "exported Cargo.lock",
            ),
            (
                exported_toolchain,
                self.expected_rust_toolchain_sha256,
                "exported rust-toolchain.toml",
            ),
        ):
            if not path.is_file() or _file_sha256(path) != expected:
                raise ProviderError(f"Codex {label} identity mismatch")
        if (
            _toml_string(
                exported_manifest,
                table="workspace.package",
                key="version",
                label="exported workspace manifest",
            )
            != self.expected_version
        ):
            raise ProviderError("Codex exported workspace version mismatch")
        for ancestor in self.build_source_root.parents:
            for config_name in ("config", "config.toml"):
                if (ancestor / ".cargo" / config_name).exists():
                    raise ProviderError(
                        "Codex build ancestor contains external Cargo configuration: "
                        f"{ancestor / '.cargo' / config_name}"
                    )
        self._source_export_tree_sha256 = await self._workspace_hash(
            self.build_source_root, "the exported Codex source tree"
        )
        self._source_exported = True

    async def _initialize_disposable_git(
        self, workspace: Path, environment: Mapping[str, str]
    ) -> None:
        _resolved_git, current_git_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Codex trial Git",
        )
        if current_git_identity.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or current_git_identity.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError("Codex source Git executable identity changed")
        git_identity = current_git_identity
        git = git_identity.get("invoked_path")
        if not isinstance(git, str):
            raise ProviderError("Codex source Git identity is unavailable")
        commands = (
            (git, "init", "--quiet"),
            (git, "add", "--force", "--all", "--", "."),
            (
                git,
                "-c",
                "user.name=Scaffold Lab",
                "-c",
                "user.email=scaffoldlab@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "--allow-empty",
                "--no-verify",
                "-m",
                "Scaffold Lab disposable baseline",
            ),
        )
        for index, command in enumerate(commands, start=1):
            process, stdout, stderr = await self._run_bounded_process(
                command,
                cwd=workspace,
                environment=environment,
                timeout_seconds=min(self.timeout_seconds, 60.0),
                label=f"Codex disposable Git baseline step {index}",
            )
            if process.returncode != 0:
                raise ProviderError(
                    f"Codex could not initialize disposable Git at step {index}",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )

    async def _capture_swe_patch(
        self, workspace: Path, environment: Mapping[str, str]
    ) -> bytes:
        """Capture tracked, untracked, deleted, and binary edits before cleanup."""

        _resolved_git, current_git_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Codex patch Git",
        )
        if current_git_identity.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or current_git_identity.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError("Codex source Git executable identity changed")
        git = current_git_identity.get("invoked_path")
        if not isinstance(git, str):
            raise ProviderError("Codex source Git identity is unavailable")
        intent_process, intent_stdout, intent_stderr = await self._run_bounded_process(
            [git, "add", "--intent-to-add", "--force", "--all", "--", "."],
            cwd=workspace,
            environment=environment,
            timeout_seconds=min(self.timeout_seconds, 60.0),
            label="Codex SWE patch untracked-file preparation",
        )
        if intent_process.returncode != 0:
            raise ProviderError(
                "Codex could not prepare untracked files for patch export",
                raw={
                    "stdout": intent_stdout.decode("utf-8", errors="replace"),
                    "stderr": intent_stderr.decode("utf-8", errors="replace"),
                },
            )
        patch_process, patch, patch_stderr = await self._run_bounded_process(
            [
                git,
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "HEAD",
                "--",
                ".",
            ],
            cwd=workspace,
            environment=environment,
            timeout_seconds=min(self.timeout_seconds, 120.0),
            label="Codex SWE patch export",
            output_limit=self.max_patch_bytes,
        )
        if patch_process.returncode != 0:
            raise ProviderError(
                "Codex could not capture its SWE patch",
                raw={"stderr": patch_stderr.decode("utf-8", errors="replace")},
            )
        return patch

    async def _ensure_built(self) -> None:
        async with self._build_lock:
            self._validate_source(runtime_error=True)
            if self._built:
                observed = (
                    _file_sha256(self.executable) if self.executable.is_file() else None
                )
                if observed != self._executable_identity.get("sha256"):
                    raise ProviderError("Codex source executable changed after build")
                return

            await self._ensure_source_export()
            cargo, cargo_identity = _prepare_executable(
                self.cargo_executable,
                self.expected_cargo_sha256,
                "Codex cargo",
            )
            self._cargo_identity = dict(cargo_identity)
            cargo_invoked = cargo_identity.get("invoked_path")
            if _is_rustup_proxy(cargo_identity):
                raise ProviderError(
                    "Codex cannot isolate a rustup Cargo proxy safely; pass the "
                    "concrete toolchain Cargo path returned by `rustup which cargo`"
                )
            if isinstance(cargo_invoked, str):
                cargo = cargo_invoked
            if cargo_identity.get("available") is not True:
                raise ProviderError(
                    f"Codex Cargo executable was not found: {self.cargo_executable!r}"
                )
            rustc_candidate = self.rustc_executable
            if rustc_candidate is None and isinstance(cargo_invoked, str):
                sibling = Path(cargo_invoked).parent / "rustc"
                if sibling.is_file():
                    rustc_candidate = str(sibling)
            if rustc_candidate is None:
                raise ProviderError(
                    "Codex needs a concrete rustc beside Cargo or an explicit "
                    "rustc_executable"
                )
            rustc, rustc_identity = _prepare_executable(
                rustc_candidate,
                self.expected_rustc_sha256,
                "Codex rustc",
            )
            if _is_rustup_proxy(rustc_identity):
                raise ProviderError(
                    "Codex rustc must be a concrete toolchain binary, not a rustup "
                    "proxy"
                )
            rustc_invoked = rustc_identity.get("invoked_path")
            if isinstance(rustc_invoked, str):
                rustc = rustc_invoked
            self._rustc_identity = dict(rustc_identity)
            build_environment = _process_environment(())
            build_environment.update(
                {
                    "HOME": str(self.build_home),
                    "CARGO_HOME": str(self.cargo_home),
                    "CARGO_TARGET_DIR": str(self.build_target_dir),
                    "CARGO_TERM_COLOR": "never",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                    "TMPDIR": str(self.build_tmp),
                    "XDG_CONFIG_HOME": str(self.build_xdg_config),
                    "RUSTC": rustc,
                }
            )
            cargo_process, cargo_stdout, cargo_stderr = await self._run_bounded_process(
                [cargo, "--version"],
                cwd=self.build_codex_rs,
                environment=build_environment,
                timeout_seconds=min(30.0, self.build_timeout_seconds),
                label="Codex Cargo version probe",
                output_limit=min(self.max_output_bytes, 1024 * 1024),
            )
            cargo_output = "\n".join(
                value
                for value in (
                    cargo_stdout.decode("utf-8", errors="replace").strip(),
                    cargo_stderr.decode("utf-8", errors="replace").strip(),
                )
                if value
            )
            if cargo_process.returncode != 0 or not re.search(
                rf"\bcargo\s+{re.escape(CODEX_RUST_TOOLCHAIN)}\b", cargo_output
            ):
                raise ProviderError(
                    "Codex concrete Cargo version does not match rust-toolchain.toml",
                    raw={"output": cargo_output},
                )
            self._cargo_version_output = cargo_output
            rustc_process, rustc_stdout, rustc_stderr = await self._run_bounded_process(
                [rustc, "--version"],
                cwd=self.build_codex_rs,
                environment=build_environment,
                timeout_seconds=min(30.0, self.build_timeout_seconds),
                label="Codex rustc version probe",
                output_limit=min(self.max_output_bytes, 1024 * 1024),
            )
            rustc_output = "\n".join(
                value
                for value in (
                    rustc_stdout.decode("utf-8", errors="replace").strip(),
                    rustc_stderr.decode("utf-8", errors="replace").strip(),
                )
                if value
            )
            if rustc_process.returncode != 0 or not re.search(
                rf"\brustc\s+{re.escape(CODEX_RUST_TOOLCHAIN)}\b", rustc_output
            ):
                raise ProviderError(
                    "Codex concrete rustc version does not match rust-toolchain.toml",
                    raw={"output": rustc_output},
                )
            self._rustc_version_output = rustc_output
            process, stdout, stderr = await self._run_bounded_process(
                [cargo, "build", "--locked", "--release", "--bin", "codex"],
                cwd=self.build_codex_rs,
                environment=build_environment,
                timeout_seconds=self.build_timeout_seconds,
                label="Codex source build",
            )
            self._validate_source(runtime_error=True)
            if process.returncode != 0:
                raise ProviderError(
                    f"Codex source build exited with status {process.returncode}",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )
            post_build_source_hash = await self._workspace_hash(
                self.build_source_root, "the exported Codex source after build"
            )
            if post_build_source_hash != self._source_export_tree_sha256:
                raise ProviderError("Codex build mutated its pinned source export")
            if not self.executable.is_file() or self.executable.is_symlink():
                raise ProviderError(
                    "Codex source build did not produce a regular "
                    "CARGO_TARGET_DIR/release/codex"
                )
            if not self.executable.resolve().is_relative_to(self.build_target_dir):
                raise ProviderError("Codex source executable escaped its build target")
            if not self.executable.stat().st_mode & (
                stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            ):
                raise ProviderError("Codex source executable is not executable")
            executable_sha256 = _file_sha256(self.executable)
            if (
                self.expected_executable_sha256 is not None
                and executable_sha256 != self.expected_executable_sha256
            ):
                raise ProviderError(
                    "Codex source executable SHA-256 mismatch",
                    raw={"observed_sha256": executable_sha256},
                )
            self._executable_identity = {
                "available": True,
                "path": str(self.executable),
                "sha256": executable_sha256,
            }

            version_home = self.build_root / "version-home"
            version_codex_home = self.build_root / "version-codex-home"
            version_xdg_config = self.build_root / "version-xdg-config"
            version_tmp = self.build_root / "version-tmp"
            for directory in (
                version_home,
                version_codex_home,
                version_xdg_config,
                version_tmp,
            ):
                directory.mkdir(mode=0o700, exist_ok=True)

            (
                version_process,
                version_stdout,
                version_stderr,
            ) = await self._run_bounded_process(
                [str(self.executable), "--version"],
                cwd=self.build_codex_rs,
                environment={
                    **_process_environment(()),
                    "HOME": str(version_home),
                    "CODEX_HOME": str(version_codex_home),
                    "XDG_CONFIG_HOME": str(version_xdg_config),
                    "TMPDIR": str(version_tmp),
                },
                timeout_seconds=min(30.0, self.timeout_seconds),
                label="Codex version probe",
                output_limit=min(self.max_output_bytes, 1024 * 1024),
            )
            version_output = "\n".join(
                value
                for value in (
                    version_stdout.decode("utf-8", errors="replace").strip(),
                    version_stderr.decode("utf-8", errors="replace").strip(),
                )
                if value
            )
            if version_process.returncode != 0:
                raise ProviderError(
                    f"Codex version probe exited with status {version_process.returncode}",
                    raw={"output": version_output},
                )
            versions = set(
                re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?![\d.])", version_output)
            )
            if (
                versions != {self.expected_version}
                or "codex" not in version_output.lower()
            ):
                raise ProviderError(
                    "Codex source version mismatch",
                    raw={"expected": self.expected_version, "output": version_output},
                )
            self._version_output = version_output
            if (
                await self._workspace_hash(
                    self.build_source_root,
                    "the exported Codex source after version probe",
                )
                != self._source_export_tree_sha256
            ):
                raise ProviderError("Codex version probe mutated its source export")
            self._built = True

    async def _workspace_hash(self, root: Path, label: str) -> str:
        try:
            return await _workspace_tree_sha256_async(
                root,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.workspace_hash_max_bytes,
                timeout_seconds=self.workspace_hash_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(f"could not hash {label}") from exc

    def _v2_wait_timeouts(self) -> tuple[int, int, int]:
        if self.max_wait_seconds is None:
            return (
                _V2_SOURCE_MIN_WAIT_MS,
                _V2_SOURCE_MAX_WAIT_MS,
                _V2_SOURCE_DEFAULT_WAIT_MS,
            )
        max_wait_ms = max(1, math.ceil(self.max_wait_seconds * 1000))
        min_wait_ms = min(_V2_SOURCE_MIN_WAIT_MS, max_wait_ms)
        default_wait_ms = min(_V2_SOURCE_DEFAULT_WAIT_MS, max_wait_ms)
        return min_wait_ms, max_wait_ms, default_wait_ms

    def _multi_agent_overrides(self) -> list[str]:
        common = [
            "agents.enabled=true",
            "features.multi_agent=true",
            f"agents.max_concurrent_threads_per_session={self.max_subagents}",
            f"agents.max_depth={self.max_depth}",
        ]
        if self.multi_agent_version == "v1":
            return [*common, "features.multi_agent_v2=false"]
        overrides = [
            *common,
            "features.multi_agent_v2.enabled=true",
            (
                "features.multi_agent_v2.max_concurrent_threads_per_session="
                f"{self.max_subagents + 1}"
            ),
        ]
        if self.max_wait_seconds is None:
            return overrides
        min_wait_ms, max_wait_ms, default_wait_ms = self._v2_wait_timeouts()
        return [
            *overrides,
            f"features.multi_agent_v2.min_wait_timeout_ms={min_wait_ms}",
            f"features.multi_agent_v2.max_wait_timeout_ms={max_wait_ms}",
            f"features.multi_agent_v2.default_wait_timeout_ms={default_wait_ms}",
        ]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if (
            request.system
            or request.tools
            or request.tool_results
            or request.continuation is not None
        ):
            raise ProviderError(
                "Codex source owns system prompts, tools, and continuation state"
            )
        prompt_bytes = request.prompt.encode("utf-8")
        if len(prompt_bytes) > self.max_prompt_bytes:
            raise ProviderError(
                f"Codex prompt exceeds {self.max_prompt_bytes} encoded bytes"
            )
        if self.api_key_env not in os.environ or not os.environ[self.api_key_env]:
            raise ProviderError(
                f"missing Codex credential environment {self.api_key_env!r}"
            )
        _reject_codex_system_configuration(runtime_error=True)
        await self._ensure_built()
        self._validate_source(runtime_error=True)
        _validate_embedded_git_metadata(
            self.workspace, max_entries=self.workspace_hash_max_entries
        )
        pre_base_hash = await self._workspace_hash(
            self.workspace, "the Codex seed workspace"
        )
        if pre_base_hash != self.workspace_tree_sha256:
            raise ProviderError(
                "Codex seed workspace changed after backend initialization"
            )

        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-codex-source-"
        ) as temp_dir:
            temp_root = Path(temp_dir)
            trial_workspace = temp_root / "workspace"
            await asyncio.to_thread(
                shutil.copytree,
                self.workspace,
                trial_workspace,
                symlinks=True,
                ignore=copytree_ignore_git_metadata,
            )
            # An absolute link into the seed resolves within the seed during the
            # initial hash but points back outside after copytree preserves it.
            # Re-hashing the copy catches that case before Codex can run.
            trial_seed_hash = await self._workspace_hash(
                trial_workspace, "the disposable Codex workspace before the session"
            )
            if trial_seed_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Codex disposable workspace copy does not match its seed"
                )
            codex_home = temp_root / "codex-home"
            sqlite_home = temp_root / "sqlite-home"
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            runtime_tmp = temp_root / "tmp"
            for directory in (codex_home, sqlite_home, home, xdg_config, runtime_tmp):
                directory.mkdir(mode=0o700)
            last_message_path = temp_root / "last-message.txt"

            git_environment = _process_environment(())
            git_environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                }
            )
            await self._initialize_disposable_git(trial_workspace, git_environment)
            post_git_baseline_hash = await self._workspace_hash(
                trial_workspace,
                "the disposable Codex workspace after Git initialization",
            )
            if post_git_baseline_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Codex disposable Git initialization changed task content"
                )

            command = [
                str(self.executable),
                "exec",
                "--strict-config",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--json",
                "--color",
                "never",
                "--sandbox",
                "workspace-write",
                "-C",
                str(trial_workspace),
                "--model",
                self.model,
                "--output-last-message",
                str(last_message_path),
            ]
            config_overrides = [
                "analytics.enabled=false",
                'otel.exporter="none"',
                'otel.trace_exporter="none"',
                'otel.metrics_exporter="none"',
                "check_for_update_on_startup=false",
                'shell_environment_policy.inherit="all"',
                ('shell_environment_policy.exclude=["OPENAI_API_KEY","CODEX_API_KEY"]'),
                "shell_environment_policy.ignore_default_excludes=false",
                "sandbox_workspace_write.exclude_slash_tmp=true",
                "sandbox_workspace_write.exclude_tmpdir_env_var=false",
                *self._multi_agent_overrides(),
            ]
            if self.reasoning_effort is not None:
                config_overrides.append(
                    f'model_reasoning_effort="{self.reasoning_effort}"'
                )
            for value in config_overrides:
                command.extend(("-c", value))
            command.append("-")

            environment = _process_environment(self.pass_env)
            environment.update(
                {
                    "CODEX_HOME": str(codex_home),
                    "CODEX_SQLITE_HOME": str(sqlite_home),
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "TMPDIR": str(runtime_tmp),
                    self.auth_target_env: os.environ[self.api_key_env],
                }
            )

            _reject_codex_system_configuration(runtime_error=True)
            usage = Usage(cost_known=False, complete=False)
            try:
                process, stdout, stderr = await self._run_bounded_process(
                    command,
                    cwd=trial_workspace,
                    environment=environment,
                    timeout_seconds=self.timeout_seconds,
                    label="Codex source session",
                    input_data=prompt_bytes,
                )
                events = _parse_jsonl(stdout)
                usage = _usage_from_events(events)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                post_trial_hash = await self._workspace_hash(
                    trial_workspace, "the disposable Codex workspace"
                )
                swe_patch = await self._capture_swe_patch(
                    trial_workspace, git_environment
                )
                post_patch_capture_hash = await self._workspace_hash(
                    trial_workspace,
                    "the disposable Codex workspace after patch capture",
                )
                if post_patch_capture_hash != post_trial_hash:
                    raise ProviderError(
                        "Codex SWE patch capture changed task workspace content",
                        usage=usage,
                    )
                collaboration_calls = _successful_collaboration_calls(events)
                collab_tools = [call["tool"] for call in collaboration_calls]
                successful_spawns = [
                    call
                    for call in collaboration_calls
                    if call["tool"] == "spawn_agent" and call["receiver_thread_ids"]
                ]
                spawn_calls = len(successful_spawns)
                spawned_child_thread_ids = sorted(
                    {
                        thread_id
                        for call in successful_spawns
                        for thread_id in call["receiver_thread_ids"]
                    }
                )
                runtime_provenance = self.provenance()
                raw: dict[str, Any] = {
                    "events": events,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "_scaffoldlab_workspace": {
                        "isolation": "fresh-disposable-copy",
                        "base_cwd": str(self.workspace),
                        "trial_cwd": str(trial_workspace),
                        "base_tree_sha256": self.workspace_tree_sha256,
                        "pre_trial_tree_sha256": trial_seed_hash,
                        "post_git_baseline_tree_sha256": post_git_baseline_hash,
                        "post_trial_tree_sha256": post_trial_hash,
                        "post_patch_capture_tree_sha256": post_patch_capture_hash,
                        "inherited_git_metadata": False,
                        "git_baseline": "fresh-standalone-commit",
                        "patch_artifact": SWEPatchPayload(swe_patch),
                        "patch_format": "git_diff_binary",
                        "patch_bytes": len(swe_patch),
                        "max_patch_bytes": self.max_patch_bytes,
                    },
                    "_scaffoldlab_source": {
                        "repository": CODEX_SOURCE_REPOSITORY,
                        "tag": CODEX_SOURCE_TAG,
                        "checkout": str(self.checkout),
                        "revision": self.expected_revision,
                        "version": self.expected_version,
                        "cargo_lock_sha256": self.expected_cargo_lock_sha256,
                        "rust_toolchain_sha256": self.expected_rust_toolchain_sha256,
                        "source_export_tree_sha256": (self._source_export_tree_sha256),
                        "git": dict(self._git_identity),
                        "cargo": dict(self._cargo_identity),
                        "cargo_version_output": self._cargo_version_output,
                        "rustc": dict(self._rustc_identity),
                        "rustc_version_output": self._rustc_version_output,
                        "executable": dict(self._executable_identity),
                        "official_public_pin_verified": (
                            self.expected_revision == CODEX_SOURCE_REVISION
                            and self.expected_version == CODEX_SOURCE_VERSION
                            and self.expected_cargo_lock_sha256
                            == CODEX_CARGO_LOCK_SHA256
                            and self.expected_rust_toolchain_sha256
                            == CODEX_RUST_TOOLCHAIN_SHA256
                        ),
                        "reproducible_build_pins_supplied": runtime_provenance.get(
                            "reproducible_build_pins_supplied"
                        ),
                        "exact_runtime_build_verified": runtime_provenance.get(
                            "exact_runtime_build_verified"
                        ),
                        "exact_runtime_boundary_verified": False,
                        "cloud_configuration_attested_absent": False,
                    },
                    "_scaffoldlab_codex": {
                        "requested_multi_agent_version": self.multi_agent_version,
                        "effective_multi_agent_version": (
                            "v2" if self.multi_agent_version == "v2" else None
                        ),
                        "effective_multi_agent_version_verified": (
                            self.multi_agent_version == "v2"
                        ),
                        "native_multi_agent_tools_enabled": True,
                        "completed_collaboration_tools": collab_tools,
                        "spawn_agent_calls_observed": spawn_calls,
                        "spawned_child_thread_ids_observed": (spawned_child_thread_ids),
                        "multi_agent_execution_observed": spawn_calls > 0,
                        "max_subagents": self.max_subagents,
                        "configured_v1_max_depth": self.max_depth,
                        "max_depth": None,
                        "v2_total_concurrency_including_root": (
                            self.max_subagents + 1
                            if self.multi_agent_version == "v2"
                            else None
                        ),
                        "v2_min_wait_timeout_ms": (
                            self._v2_wait_timeouts()[0]
                            if self.multi_agent_version == "v2"
                            else None
                        ),
                        "v2_max_wait_timeout_ms": (
                            self._v2_wait_timeouts()[1]
                            if self.multi_agent_version == "v2"
                            else None
                        ),
                        "v2_default_wait_timeout_ms": (
                            self._v2_wait_timeouts()[2]
                            if self.multi_agent_version == "v2"
                            else None
                        ),
                    },
                    "usage_is_incomplete": True,
                    "cost_is_unknown": True,
                    "usage_scope": (
                        "root codex exec turn total; whole-tree completeness unverified"
                    ),
                }
                if process.returncode != 0:
                    raise ProviderError(
                        f"Codex source exited with status {process.returncode}",
                        usage=usage,
                        raw=raw,
                    )
                failures = [
                    event
                    for event in events
                    if event.get("type") in {"turn.failed", "error"}
                ]
                if failures:
                    raise ProviderError(
                        "Codex source reported a failed turn", usage=usage, raw=raw
                    )
                if not any(event.get("type") == "turn.completed" for event in events):
                    raise ProviderError(
                        "Codex source emitted no turn.completed event",
                        usage=usage,
                        raw=raw,
                    )
                try:
                    answer = last_message_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise ProviderError(
                        "Codex source did not write its final-message file",
                        usage=usage,
                        raw=raw,
                    ) from exc
                if not answer:
                    raise ProviderError(
                        "Codex source final message was empty", usage=usage, raw=raw
                    )
                return ModelResponse(text=answer, usage=usage, raw=raw)
            except asyncio.CancelledError as exc:
                # Cancellation can arrive during hashing or patch capture after the
                # terminal event has already exposed billable usage.
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except ProviderError as exc:
                # Postprocessing helpers do not know the child session's usage.
                # Preserve their diagnostics while attaching the strongest terminal
                # lower bound parsed above.
                exc.usage = usage
                raise
            finally:
                try:
                    self._validate_source(runtime_error=True)
                    observed_executable_sha256 = (
                        _file_sha256(self.executable)
                        if self.executable.is_file()
                        else None
                    )
                    if observed_executable_sha256 != self._executable_identity.get(
                        "sha256"
                    ):
                        raise ProviderError(
                            "Codex source executable changed during the session",
                            usage=usage,
                        )
                    post_base_hash = await self._workspace_hash(
                        self.workspace, "the Codex seed workspace after the session"
                    )
                    if post_base_hash != self.workspace_tree_sha256:
                        raise ProviderError(
                            "Codex escaped its disposable workspace and changed the "
                            "seed workspace",
                            usage=usage,
                        )
                except ProviderError as exc:
                    exc.usage = usage
                    raise
