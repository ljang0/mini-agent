from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .external import (
    _CapturedOutput,
    _ProcessOutputLimitExceeded,
    _communicate_limited,
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
)
from .types import ModelRequest, ModelResponse, Usage


# This is an official executable distribution, not a public source checkout.
# npm registry metadata was independently checked on 2026-08-10.
CLAUDE_CODE_DISTRIBUTION_VERSION = "2.1.226"
CLAUDE_CODE_WRAPPER_NPM_INTEGRITY = (
    "sha512-mUkA81SbzATHFsHNz/rPy3Itw0D0S9kQMsIUJ3qPGwpNJMqPePyDP6xnWHI0"
    "jfFlspVjs8r/GfolMUyiy8P1FQ=="
)
CLAUDE_CODE_WRAPPER_NPM_SHASUM = "68b41535948dc06a7486716bd21353b3d48626c0"
CLAUDE_CODE_DARWIN_ARM64_NPM_INTEGRITY = (
    "sha512-/vIgn1GB6SiOHMcx7zVDZej2Vk+hDr2qkd4aKTryoPm2THorWW3lPpCkzoa4OArg"
    "5na1K+eNhGdenhefWthtsw=="
)
CLAUDE_CODE_DARWIN_ARM64_NPM_SHASUM = "a5130848073e9cda593a09e4cc1a89ea2d9506f8"
CLAUDE_CODE_DARWIN_ARM64_EXECUTABLE_SHA256 = (
    "013a1cf17df5ff1dcc189d5d6fd3fdd5f097ddc3cd41aa9992e99805574febbe"
)
CLAUDE_CODE_PUBLIC_TAG_REVISION = "2bb60696142b493eafaeacfe00eac51d16c50c4f"

_KNOWN_EXECUTABLE_SHA256: Mapping[str, str] = {
    "@anthropic-ai/claude-code-darwin-arm64": (
        CLAUDE_CODE_DARWIN_ARM64_EXECUTABLE_SHA256
    ),
}
_DEFAULT_MAX_WORKSPACE_ENTRIES = 250_000
_DEFAULT_MAX_WORKSPACE_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_PATCH_BYTES = 8 * 1024 * 1024
_CLAUDE_CODE_MANAGED_PATHS: tuple[Path, ...] = (
    Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
    Path("/Library/Application Support/ClaudeCode/managed-mcp.json"),
    Path("/Library/Application Support/ClaudeCode/managed-settings.d"),
    Path("/Library/Managed Preferences/com.anthropic.claudecode.plist"),
    Path("/Library/Preferences/com.anthropic.claudecode.plist"),
    Path("/etc/claude-code/managed-settings.json"),
    Path("/etc/claude-code/managed-mcp.json"),
    Path("/etc/claude-code/managed-settings.d"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Claude Code {label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Claude Code {label} must contain a JSON object")
    return value


def _assert_no_host_managed_settings(*, runtime_error: bool = False) -> None:
    """Fail closed when documented endpoint-managed configuration is present."""

    error_type = ProviderError if runtime_error else ValueError
    present: list[str] = []
    for path in _CLAUDE_CODE_MANAGED_PATHS:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise error_type(
                f"could not attest Claude Code managed-settings path {path}"
            ) from exc
        present.append(str(path))
    if present:
        raise error_type(
            "Claude Code endpoint-managed settings are present and cannot be "
            "disabled by --setting-sources: " + ", ".join(present)
        )


def _native_package_for_host() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    architecture = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x64",
        "x86_64": "x64",
    }.get(machine)
    if system not in {"darwin", "linux"} or architecture is None:
        raise ValueError(
            "cannot infer the Claude Code native package for this host; pass "
            "native_package_name explicitly"
        )
    # Musl installations must override this with the -musl package name. The
    # executable digest remains mandatory, so an incorrect inference fails closed.
    return f"@anthropic-ai/claude-code-{system}-{architecture}"


def _native_package_path(distribution_root: Path, package_name: str) -> Path:
    scope, name = package_name.split("/", 1)
    return distribution_root / "node_modules" / scope / name


def _assert_regular_workspace(root: Path, *, max_entries: int) -> None:
    """Reject links/special files after pruning Git administrative metadata."""

    entries = 0
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        kept_directories = [name for name in directory_names if name != ".git"]
        kept_files = [name for name in file_names if name != ".git"]
        directory_names[:] = kept_directories
        for name in [*kept_directories, *kept_files]:
            entries += 1
            if entries > max_entries:
                raise ValueError(
                    "Claude Code workspace validation exceeded its entry limit"
                )
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(
                    "Claude Code disposable workspaces reject symbolic links"
                )
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(
                    "Claude Code disposable workspaces reject special files"
                )


def _non_negative_int(raw: Mapping[str, Any], name: str) -> tuple[int, bool]:
    value = raw.get(name)
    if value is None:
        return 0, False
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderError(f"Claude Code emitted invalid {name!r} usage")
    return value, True


def _usage_lower_bound(result: Mapping[str, Any]) -> Usage:
    raw = result.get("usage")
    if not isinstance(raw, Mapping):
        return Usage(cost_known=False, complete=False)
    uncached, _ = _non_negative_int(raw, "input_tokens")
    output, _ = _non_negative_int(raw, "output_tokens")
    cache_read, _ = _non_negative_int(raw, "cache_read_input_tokens")
    cache_write, _ = _non_negative_int(raw, "cache_creation_input_tokens")
    cost = result.get("total_cost_usd")
    if cost is None:
        cost_usd = 0.0
    elif (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        cost_usd = float(cost)
    else:
        raise ProviderError("Claude Code emitted invalid total_cost_usd")
    # Claude Code documents a terminal usage object, but does not document it as
    # authoritative whole-team accounting. Preserve it only as a lower bound.
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=output,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        cost_usd=cost_usd,
        cost_known=False,
        complete=False,
    )


def _walk_json(value: Any):  # type: ignore[no-untyped-def]
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _team_evidence(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    named_teammates: set[str] = set()
    agent_tool_calls = 0
    send_message_calls = 0
    removed_team_create_calls = 0
    removed_team_delete_calls = 0
    deprecated_team_name_inputs = 0
    team_protocol_markers = 0
    for event in events:
        for value in _walk_json(event):
            if isinstance(value, Mapping) and value.get("type") == "tool_use":
                tool_name = value.get("name")
                tool_input = value.get("input")
                if tool_name == "Agent":
                    agent_tool_calls += 1
                    if isinstance(tool_input, Mapping):
                        name = tool_input.get("name")
                        if isinstance(name, str) and name.strip():
                            named_teammates.add(name.strip())
                        team_name = tool_input.get("team_name")
                        if isinstance(team_name, str) and team_name.strip():
                            deprecated_team_name_inputs += 1
                elif tool_name == "SendMessage":
                    send_message_calls += 1
                elif tool_name == "TeamCreate":
                    removed_team_create_calls += 1
                elif tool_name == "TeamDelete":
                    removed_team_delete_calls += 1
            elif isinstance(value, str) and (
                "<teammate-message" in value
                or "<teammate_message" in value
                or "TeammateIdle" in value
            ):
                team_protocol_markers += 1
    return {
        "agent_tool_calls": agent_tool_calls,
        "named_teammates": sorted(named_teammates),
        "named_teammate_count": len(named_teammates),
        "deprecated_team_name_inputs": deprecated_team_name_inputs,
        "removed_team_create_calls": removed_team_create_calls,
        "removed_team_delete_calls": removed_team_delete_calls,
        "send_message_calls": send_message_calls,
        "team_protocol_markers": team_protocol_markers,
        "native_team_observed": False,
    }


def _session_team_name(session_id: Any) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ProviderError("Claude Code result omitted its session_id")
    try:
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise ProviderError(
            "Claude Code result contained an invalid session_id"
        ) from exc
    canonical = str(parsed)
    if canonical != session_id.lower():
        raise ProviderError("Claude Code result session_id was not canonical")
    return f"session-{canonical[:8]}"


def _parse_live_team_config(config_path: Path) -> Mapping[str, Any] | None:
    """Return one complete live config snapshot, or None during an atomic rewrite."""

    try:
        file_stat = config_path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ProviderError(
                "Claude Code generated team config is not a regular file"
            )
        if file_stat.st_size > 1024 * 1024:
            raise ProviderError("Claude Code generated team config is too large")
        raw = config_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProviderError("Claude Code generated team config is unreadable") from exc
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # The runtime may rewrite config.json in place. A later polling pass must
        # capture a complete version before the process exits.
        return None
    if not isinstance(config, Mapping):
        return None
    members = config.get("members")
    if not isinstance(members, list) or not members:
        return None
    normalized_members: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping):
            return None
        name = member.get("name")
        agent_id = member.get("agentId")
        agent_type = member.get("agentType")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(agent_id, str)
            or not agent_id.strip()
            or (
                agent_type is not None
                and (not isinstance(agent_type, str) or not agent_type.strip())
            )
        ):
            return None
        normalized: dict[str, Any] = {
            "name": name.strip(),
            "agent_id": agent_id.strip(),
        }
        if isinstance(agent_type, str):
            normalized["agent_type"] = agent_type.strip()
        normalized_members.append(normalized)
    return {
        "members": normalized_members,
        "member_count": len(normalized_members),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
    }


async def _capture_live_team_configs(
    config_dir: Path,
    stop: asyncio.Event,
) -> Mapping[str, Mapping[str, Any]]:
    """Snapshot the largest valid config for each implicit team while it exists."""

    snapshots: dict[str, Mapping[str, Any]] = {}

    def capture_once() -> None:
        teams_root = config_dir / "teams"
        try:
            if teams_root.is_symlink():
                raise ProviderError("Claude Code teams directory is a symbolic link")
            candidates = sorted(teams_root.glob("session-*/config.json"))
        except OSError as exc:
            raise ProviderError("Claude Code teams directory is unreadable") from exc
        if len(candidates) > 8:
            raise ProviderError("Claude Code created too many team configs")
        for config_path in candidates:
            team_name = config_path.parent.name
            if re.fullmatch(r"session-[0-9a-f]{8}", team_name) is None:
                continue
            snapshot = _parse_live_team_config(config_path)
            if snapshot is None:
                continue
            previous = snapshots.get(team_name)
            if previous is None or int(snapshot["member_count"]) >= int(
                previous["member_count"]
            ):
                snapshots[team_name] = snapshot

    while True:
        capture_once()
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.01)
        except asyncio.TimeoutError:
            pass
    return snapshots


def _session_team_storage_evidence(
    config_dir: Path,
    *,
    session_id: Any,
    named_teammates: Sequence[str],
    live_snapshots: Mapping[str, Mapping[str, Any]],
    deprecated_team_name_inputs: int,
    removed_team_create_calls: int,
    removed_team_delete_calls: int,
) -> Mapping[str, Any]:
    team_name = _session_team_name(session_id)
    snapshot = live_snapshots.get(team_name)
    members = snapshot.get("members") if isinstance(snapshot, Mapping) else None
    normalized_members = members if isinstance(members, list) else []
    lead_members = [
        member
        for member in normalized_members
        if isinstance(member, Mapping) and member.get("agent_type") == "team-lead"
    ]
    configured_teammates = {
        str(member.get("name"))
        for member in normalized_members
        if isinstance(member, Mapping) and member.get("agent_type") != "team-lead"
    }
    observed_teammates = set(named_teammates)

    expected_config_dir = config_dir / "teams" / team_name
    config_removed = not expected_config_dir.exists()
    tasks_root = config_dir / "tasks" / team_name
    if tasks_root.is_symlink():
        raise ProviderError("Claude Code session task directory is a symbolic link")
    task_directory_present = tasks_root.is_dir()
    task_files = (
        sorted(
            str(path.relative_to(tasks_root))
            for path in tasks_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        if task_directory_present
        else []
    )

    legacy_tools_absent = (
        deprecated_team_name_inputs == 0
        and removed_team_create_calls == 0
        and removed_team_delete_calls == 0
    )
    member_roster_matches = (
        len(lead_members) == 1
        and bool(observed_teammates)
        and observed_teammates == configured_teammates
    )
    native_team_observed = (
        snapshot is not None
        and member_roster_matches
        and task_directory_present
        and config_removed
        and legacy_tools_absent
    )
    teams_root = config_dir / "teams"
    return {
        "session_derived_team_name": team_name,
        "live_team_config_snapshot": snapshot,
        "live_team_config_snapshot_count": len(live_snapshots),
        "live_member_names": sorted(
            str(member.get("name"))
            for member in normalized_members
            if isinstance(member, Mapping)
        ),
        "lead_member_count": len(lead_members),
        "member_roster_matches_agent_calls": member_roster_matches,
        "session_task_directory_present": task_directory_present,
        "session_team_config_removed_after_exit": config_removed,
        "teams_root_removed_after_exit": not teams_root.exists(),
        "persisted_task_file_count": len(task_files),
        "persisted_task_files": task_files,
        "native_team_observed": native_team_observed,
    }


class ClaudeCodeAgentTeamsDistributionBackend:
    """Execute the pinned official Claude Code binary with native Agent Teams.

    This is deliberately a *distribution* adapter. Anthropic does not publish the
    implementation source corresponding to the native executable. One outer model
    call represents the complete upstream lead-and-teammate session. Claude Code
    owns prompts, tools, scheduling, team mailboxes, task files, compaction, and
    shutdown behavior; Scaffold Lab only supplies the task and execution bounds.
    """

    def __init__(
        self,
        *,
        distribution_root: Path,
        workspace: Path,
        model: str,
        max_budget_usd: float,
        api_key_env: str = "ANTHROPIC_API_KEY",
        expected_version: str = CLAUDE_CODE_DISTRIBUTION_VERSION,
        expected_executable_sha256: Optional[str] = None,
        native_package_name: Optional[str] = None,
        git_executable: str = "git",
        expected_git_sha256: Optional[str] = None,
        max_turns: int = 64,
        permission_mode: str = "dontAsk",
        effort: Optional[str] = None,
        timeout_seconds: float = 1800.0,
        tool_timeout_seconds: float = 600.0,
        max_output_bytes: int = 16 * 1024 * 1024,
        max_patch_bytes: int = _DEFAULT_MAX_PATCH_BYTES,
        pass_env: Sequence[str] = (),
        allow_sensitive_environment: bool = False,
        require_team_evidence: bool = True,
        workspace_hash_max_entries: int = _DEFAULT_MAX_WORKSPACE_ENTRIES,
        workspace_hash_max_bytes: int = _DEFAULT_MAX_WORKSPACE_BYTES,
        workspace_hash_timeout_seconds: float = (
            _DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS
        ),
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Claude Code distribution adapter requires POSIX process groups"
            )
        self.distribution_root = distribution_root.resolve()
        self.workspace = workspace.resolve()
        if not self.distribution_root.is_dir():
            raise ValueError(
                "Claude Code distribution_root is not a directory: "
                f"{self.distribution_root}"
            )
        if not self.workspace.is_dir():
            raise ValueError(
                f"Claude Code workspace is not a directory: {self.workspace}"
            )
        if (
            self.distribution_root == self.workspace
            or self.distribution_root.is_relative_to(self.workspace)
            or self.workspace.is_relative_to(self.distribution_root)
        ):
            raise ValueError(
                "Claude Code distribution and task workspace must be disjoint"
            )
        if re.fullmatch(r"\d+\.\d+\.\d+", expected_version) is None:
            raise ValueError("expected_version must be a canonical semantic version")
        for name, value in (("model", model), ("git_executable", git_executable)):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{name} must be a non-empty string")
        if not api_key_env or "=" in api_key_env or "\x00" in api_key_env:
            raise ValueError("api_key_env must be a non-empty environment name")
        if (
            not isinstance(max_budget_usd, (int, float))
            or isinstance(max_budget_usd, bool)
            or not math.isfinite(max_budget_usd)
            or max_budget_usd <= 0
        ):
            raise ValueError("max_budget_usd must be positive and finite")
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer")
        if permission_mode not in {
            "acceptEdits",
            "auto",
            "bypassPermissions",
            "manual",
            "dontAsk",
            "plan",
        }:
            raise ValueError("permission_mode is not supported by Claude Code 2.1.226")
        if effort is not None and effort not in {
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("effort must be low, medium, high, xhigh, max, or null")
        for numeric_name, numeric_value in (
            ("timeout_seconds", timeout_seconds),
            ("tool_timeout_seconds", tool_timeout_seconds),
            ("workspace_hash_timeout_seconds", workspace_hash_timeout_seconds),
        ):
            if (
                not isinstance(numeric_value, (int, float))
                or isinstance(numeric_value, bool)
                or not math.isfinite(numeric_value)
                or numeric_value <= 0
            ):
                raise ValueError(f"{numeric_name} must be positive and finite")
        for integer_name, integer_value, minimum in (
            ("max_output_bytes", max_output_bytes, 1024),
            ("max_patch_bytes", max_patch_bytes, 1024),
            ("workspace_hash_max_entries", workspace_hash_max_entries, 1),
            ("workspace_hash_max_bytes", workspace_hash_max_bytes, 1),
        ):
            if (
                not isinstance(integer_value, int)
                or isinstance(integer_value, bool)
                or integer_value < minimum
            ):
                raise ValueError(f"{integer_name} must be an integer >= {minimum}")
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        reject_runtime_environment_overrides(
            pass_env,
            label="Claude Code distribution",
            reserved_prefixes=("ANTHROPIC_", "CLAUDE_"),
        )
        if not allow_sensitive_environment:
            raise ValueError(
                "Claude Code tools and teammates inherit the model credential; "
                "acknowledge this only for a disposable outer sandbox"
            )
        if not isinstance(require_team_evidence, bool):
            raise ValueError("require_team_evidence must be a boolean")

        self.native_package_name = native_package_name or _native_package_for_host()
        if (
            re.fullmatch(
                r"@anthropic-ai/claude-code-[a-z0-9-]+", self.native_package_name
            )
            is None
        ):
            raise ValueError("native_package_name is not a Claude Code native package")
        known_digest = _KNOWN_EXECUTABLE_SHA256.get(self.native_package_name)
        if (
            known_digest is not None
            and expected_executable_sha256 is not None
            and expected_executable_sha256 != known_digest
        ):
            raise ValueError(
                "expected_executable_sha256 cannot override the audited official "
                "digest for this Claude Code platform package"
            )
        self.expected_executable_sha256 = known_digest or expected_executable_sha256
        if self.expected_executable_sha256 is None:
            raise ValueError("expected_executable_sha256 is required for this platform")
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_executable_sha256) is None:
            raise ValueError(
                "expected_executable_sha256 must be 64 lowercase hex characters"
            )
        self.official_distribution_verified = known_digest is not None
        if (
            expected_git_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", expected_git_sha256) is None
        ):
            raise ValueError(
                "expected_git_sha256 must be 64 lowercase hex characters or null"
            )

        self.expected_version = expected_version
        self.git_executable = git_executable
        self.expected_git_sha256 = expected_git_sha256
        self.model = model
        self.max_budget_usd = float(max_budget_usd)
        self.api_key_env = api_key_env
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.effort = effort
        self.timeout_seconds = float(timeout_seconds)
        self.tool_timeout_seconds = float(tool_timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.max_patch_bytes = max_patch_bytes
        self.pass_env = tuple(pass_env)
        self.allow_sensitive_environment = allow_sensitive_environment
        self.require_team_evidence = require_team_evidence
        self.workspace_hash_max_entries = workspace_hash_max_entries
        self.workspace_hash_max_bytes = workspace_hash_max_bytes
        self.workspace_hash_timeout_seconds = float(workspace_hash_timeout_seconds)

        self.wrapper_manifest_path = self.distribution_root / "package.json"
        self.executable = self.distribution_root / "bin" / "claude.exe"
        self.native_package_root = _native_package_path(
            self.distribution_root, self.native_package_name
        )
        self.native_manifest_path = self.native_package_root / "package.json"
        self.native_executable = self.native_package_root / "claude"
        _assert_no_host_managed_settings()
        self._distribution_identity = self._validate_distribution()
        reject_case_variant_git_metadata(
            self.workspace,
            label="Claude Code task workspace",
            max_entries=self.workspace_hash_max_entries,
        )
        _assert_regular_workspace(
            self.workspace, max_entries=self.workspace_hash_max_entries
        )
        self.workspace_tree_sha256 = _workspace_tree_sha256(
            self.workspace,
            max_entries=self.workspace_hash_max_entries,
            max_bytes=self.workspace_hash_max_bytes,
            timeout_seconds=self.workspace_hash_timeout_seconds,
        )
        self.git_workspace: Mapping[str, Any] = {
            "inherited_metadata_used": False,
            "disposable_baseline": "fresh-standalone-repository",
        }
        self._git_identity: Mapping[str, Any] = {}
        self._observed_version: Optional[str] = None

    def _validate_distribution(
        self, *, runtime_error: bool = False
    ) -> Mapping[str, Any]:
        error_type = ProviderError if runtime_error else ValueError
        try:
            wrapper = _read_json_object(
                self.wrapper_manifest_path, "wrapper package.json"
            )
            native = _read_json_object(self.native_manifest_path, "native package.json")
            if wrapper.get("name") != "@anthropic-ai/claude-code":
                raise ValueError("Claude Code wrapper package name mismatch")
            if wrapper.get("version") != self.expected_version:
                raise ValueError("Claude Code wrapper package version mismatch")
            binary_map = wrapper.get("bin")
            if not isinstance(binary_map, Mapping) or binary_map.get("claude") != (
                "bin/claude.exe"
            ):
                raise ValueError("Claude Code wrapper executable mapping mismatch")
            optional = wrapper.get("optionalDependencies")
            if (
                not isinstance(optional, Mapping)
                or optional.get(self.native_package_name) != self.expected_version
            ):
                raise ValueError("Claude Code native dependency pin mismatch")
            if native.get("name") != self.native_package_name:
                raise ValueError("Claude Code native package name mismatch")
            if native.get("version") != self.expected_version:
                raise ValueError("Claude Code native package version mismatch")
            for path, label in (
                (self.executable, "wrapper executable"),
                (self.native_executable, "native executable"),
            ):
                if not path.is_file():
                    raise ValueError(f"Claude Code {label} is missing: {path}")
                if not path.stat().st_mode & (
                    stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                ):
                    raise ValueError(f"Claude Code {label} is not executable")
            executable_sha256 = _file_sha256(self.executable)
            native_sha256 = _file_sha256(self.native_executable)
            if executable_sha256 != self.expected_executable_sha256:
                raise ValueError("Claude Code executable SHA-256 mismatch")
            if native_sha256 != executable_sha256:
                raise ValueError(
                    "Claude Code wrapper executable differs from its native package"
                )
            return {
                "wrapper_manifest_sha256": _file_sha256(self.wrapper_manifest_path),
                "native_manifest_sha256": _file_sha256(self.native_manifest_path),
                "executable_sha256": executable_sha256,
                "native_executable_sha256": native_sha256,
            }
        except (OSError, ValueError) as exc:
            if isinstance(exc, error_type):
                raise
            raise error_type(str(exc)) from exc

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "anthropic-claude-code-agent-teams-distribution",
            "artifact_kind": "official_binary_distribution_not_public_source",
            "distribution_root": str(self.distribution_root),
            "expected_version": self.expected_version,
            "observed_version": self._observed_version,
            "version_verified": self._observed_version is not None,
            "native_package_name": self.native_package_name,
            "expected_executable_sha256": self.expected_executable_sha256,
            "distribution_identity": dict(self._distribution_identity),
            "known_official_platform_digest": (
                _KNOWN_EXECUTABLE_SHA256.get(self.native_package_name)
                == self.expected_executable_sha256
            ),
            "official_distribution_verified": self.official_distribution_verified,
            "official_wrapper_npm_integrity": CLAUDE_CODE_WRAPPER_NPM_INTEGRITY,
            "official_wrapper_npm_shasum": CLAUDE_CODE_WRAPPER_NPM_SHASUM,
            "public_repository_tag_revision": CLAUDE_CODE_PUBLIC_TAG_REVISION,
            "public_repository_is_runtime_source": False,
            "workspace": str(self.workspace),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "workspace_isolation": (
                "fresh-disposable-copy-no-links-inherited-git-stripped"
            ),
            "git_executable": self.git_executable,
            "expected_git_sha256": self.expected_git_sha256,
            "git_identity": dict(self._git_identity),
            "configuration_home": "ephemeral-per-call",
            "requested_setting_sources": [],
            "host_managed_settings_paths_checked": [
                str(path) for path in _CLAUDE_CODE_MANAGED_PATHS
            ],
            "host_managed_settings_absent": True,
            "server_managed_policy_observable": False,
            "default_system_prompt_retained": True,
            "bare_mode": False,
            "model": self.model,
            "api_key_environment_name": self.api_key_env,
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "agent_teams_environment_enabled": True,
            "teammate_mode": "in-process",
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
            "max_budget_usd": self.max_budget_usd,
            "effort": self.effort,
            "timeout_seconds": self.timeout_seconds,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "max_patch_bytes": self.max_patch_bytes,
            "require_team_evidence": self.require_team_evidence,
            "one_backend_call_is_external_session_tree": True,
            "usage_scope": (
                "terminal result is an observed lower bound; authoritative "
                "whole-team coverage is not documented"
            ),
            "exactness_scope": (
                "audited official executable and public noninteractive CLI "
                "invocation; server-managed policy remains unobservable"
                if self.official_distribution_verified
                else (
                    "caller-pinned unverified executable and CLI invocation; "
                    "server-managed policy remains unobservable"
                )
            ),
            "source_or_protocol_pin_verified": self.official_distribution_verified,
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }

    def _command(self) -> list[str]:
        command = [
            str(self.executable),
            "-p",
            "--input-format",
            "text",
            "--output-format",
            "stream-json",
            "--verbose",
            "--forward-subagent-text",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-chrome",
            "--teammate-mode",
            "in-process",
            "--model",
            self.model,
            "--permission-mode",
            self.permission_mode,
            "--max-turns",
            str(self.max_turns),
            "--max-budget-usd",
            format(self.max_budget_usd, ".12g"),
        ]
        if self.effort is not None:
            command.extend(("--effort", self.effort))
        return command

    async def prepare_for_manifest(self) -> None:
        """Attest the distribution and Git identity before fingerprinting."""

        await self.verify_version()
        _resolved_git, identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Claude Code trial Git",
        )
        if identity.get("available") is not True:
            raise ProviderError(
                "Claude Code trial Git executable was not found: "
                f"{self.git_executable!r}"
            )
        self._git_identity = dict(identity)

    async def _run_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        input_data: Optional[bytes],
        timeout_seconds: float,
        label: str,
        output_limit: Optional[int] = None,
    ) -> tuple[asyncio.subprocess.Process, bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=dict(environment),
                stdin=asyncio.subprocess.PIPE,
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
            exc.usage = Usage(  # type: ignore[attr-defined]
                cost_known=False, complete=False
            )
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

    async def verify_version(self) -> str:
        if self._observed_version is not None:
            return self._observed_version
        _assert_no_host_managed_settings(runtime_error=True)
        self._validate_distribution(runtime_error=True)
        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-claude-code-version-"
        ) as temp_dir:
            temp_root = Path(temp_dir)
            cwd = temp_root / "cwd"
            home = temp_root / "home"
            config = temp_root / "claude-config"
            xdg_config = temp_root / "xdg-config"
            xdg_cache = temp_root / "xdg-cache"
            process_temp = temp_root / "tmp"
            for directory in (
                cwd,
                home,
                config,
                xdg_config,
                xdg_cache,
                process_temp,
            ):
                directory.mkdir(mode=0o700)
            probe_environment = _process_environment(())
            probe_environment.update(
                {
                    "HOME": str(home),
                    "CLAUDE_CONFIG_DIR": str(config),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "XDG_CACHE_HOME": str(xdg_cache),
                    "TMPDIR": str(process_temp),
                    "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
                    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
                    "DISABLE_UPDATES": "1",
                    "DISABLE_TELEMETRY": "1",
                    "DISABLE_ERROR_REPORTING": "1",
                    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
                }
            )
            process, stdout, stderr = await self._run_process(
                [
                    str(self.executable),
                    "--teammate-mode",
                    "in-process",
                    "--version",
                ],
                cwd=cwd,
                environment=probe_environment,
                input_data=None,
                timeout_seconds=min(self.timeout_seconds, 30.0),
                label="Claude Code version and Agent Teams flag probe",
                output_limit=min(self.max_output_bytes, 1024 * 1024),
            )
        output = "\n".join(
            value
            for value in (
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            )
            if value
        )
        if process.returncode != 0:
            raise ProviderError(
                "Claude Code version and Agent Teams flag probe failed",
                raw={"output": output, "returncode": process.returncode},
            )
        versions = set(re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?![\d.])", output))
        if versions != {self.expected_version}:
            raise ProviderError(
                "Claude Code version mismatch",
                raw={"expected": self.expected_version, "output": output},
            )
        self._validate_distribution(runtime_error=True)
        _assert_no_host_managed_settings(runtime_error=True)
        self._observed_version = self.expected_version
        return self._observed_version

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

    async def _initialize_disposable_git(
        self, workspace: Path, environment: Mapping[str, str]
    ) -> None:
        _resolved_git, current_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Claude Code trial Git",
        )
        if current_identity.get("available") is not True:
            raise ProviderError(
                "Claude Code trial Git executable was not found: "
                f"{self.git_executable!r}"
            )
        if self._git_identity and (
            current_identity.get("resolved_path")
            != self._git_identity.get("resolved_path")
            or current_identity.get("sha256") != self._git_identity.get("sha256")
        ):
            raise ProviderError("Claude Code trial Git executable identity changed")
        invoked = current_identity.get("invoked_path")
        if not isinstance(invoked, str):
            raise ProviderError("Claude Code trial Git identity is unavailable")
        self._git_identity = dict(current_identity)
        commands = (
            (invoked, "init", "--quiet"),
            (invoked, "add", "--force", "--all", "--", "."),
            (
                invoked,
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
            process, stdout, stderr = await self._run_process(
                command,
                cwd=workspace,
                environment=environment,
                input_data=None,
                timeout_seconds=min(self.timeout_seconds, 60.0),
                label=f"Claude Code disposable Git baseline step {index}",
                output_limit=min(self.max_output_bytes, 1024 * 1024),
            )
            if process.returncode != 0:
                raise ProviderError(
                    "Claude Code could not create its standalone disposable Git "
                    f"baseline at step {index}",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )

    async def _export_disposable_patch(
        self, workspace: Path, environment: Mapping[str, str]
    ) -> SWEPatchPayload:
        _resolved_git, identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Claude Code patch-export Git",
        )
        if (
            identity.get("available") is not True
            or identity.get("resolved_path") != self._git_identity.get("resolved_path")
            or identity.get("sha256") != self._git_identity.get("sha256")
        ):
            raise ProviderError("Claude Code patch-export Git identity changed")
        invoked = identity.get("invoked_path")
        if not isinstance(invoked, str):
            raise ProviderError("Claude Code patch-export Git identity is unavailable")
        add_process, add_stdout, add_stderr = await self._run_process(
            (
                invoked,
                "add",
                "--intent-to-add",
                "--force",
                "--all",
                "--",
                ".",
            ),
            cwd=workspace,
            environment=environment,
            input_data=None,
            timeout_seconds=min(self.timeout_seconds, 60.0),
            label="Claude Code patch export untracked-file preparation",
            output_limit=min(self.max_output_bytes, 1024 * 1024),
        )
        if add_process.returncode != 0:
            raise ProviderError(
                "Claude Code could not prepare untracked files for patch export",
                raw={
                    "stdout": add_stdout.decode("utf-8", errors="replace"),
                    "stderr": add_stderr.decode("utf-8", errors="replace"),
                },
            )
        diff_process, patch, diff_stderr = await self._run_process(
            (
                invoked,
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "HEAD",
                "--",
            ),
            cwd=workspace,
            environment=environment,
            input_data=None,
            timeout_seconds=min(self.timeout_seconds, 120.0),
            label="Claude Code binary patch export",
            output_limit=self.max_patch_bytes,
        )
        if diff_process.returncode != 0:
            raise ProviderError(
                "Claude Code binary patch export failed",
                raw={"stderr": diff_stderr.decode("utf-8", errors="replace")},
            )
        return SWEPatchPayload(patch)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if (
            request.system
            or request.tools
            or request.tool_results
            or request.continuation is not None
        ):
            raise ProviderError(
                "Claude Code distribution owns its system prompt, tools, and "
                "continuation state"
            )
        self._validate_distribution(runtime_error=True)
        _assert_no_host_managed_settings(runtime_error=True)
        if self._observed_version is None:
            await self.verify_version()
        reject_case_variant_git_metadata(
            self.workspace,
            label="Claude Code task workspace",
            max_entries=self.workspace_hash_max_entries,
        )
        _assert_regular_workspace(
            self.workspace, max_entries=self.workspace_hash_max_entries
        )
        before = await self._workspace_hash(
            self.workspace, "the Claude Code base workspace"
        )
        if before != self.workspace_tree_sha256:
            raise ProviderError(
                "Claude Code base workspace changed after backend initialization"
            )
        credential = os.environ.get(self.api_key_env, "")
        if not credential:
            raise ProviderError(
                f"missing Claude Code API credential environment {self.api_key_env!r}"
            )

        usage = Usage(cost_known=False, complete=False)
        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-claude-code-teams-"
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
            ephemeral_user = temp_root / "user"
            config_dir = temp_root / "claude-config"
            xdg_config = temp_root / "xdg-config"
            xdg_cache = temp_root / "xdg-cache"
            process_temp = temp_root / "tmp"
            for directory in (
                ephemeral_user,
                config_dir,
                xdg_config,
                xdg_cache,
                process_temp,
            ):
                directory.mkdir(mode=0o700)
            git_environment = _process_environment(())
            git_environment.update(
                {
                    "HOME": str(ephemeral_user),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "TMPDIR": str(process_temp),
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_OPTIONAL_LOCKS": "0",
                }
            )
            copied_seed_hash = await self._workspace_hash(
                trial_workspace,
                "the disposable Claude Code workspace before Git initialization",
            )
            if copied_seed_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Claude Code disposable workspace copy does not match its seed"
                )
            await self._initialize_disposable_git(trial_workspace, git_environment)
            post_git_seed_hash = await self._workspace_hash(
                trial_workspace,
                "the disposable Claude Code workspace after Git initialization",
            )
            if post_git_seed_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Claude Code trial Git changed non-administrative workspace data"
                )
            environment = _process_environment(self.pass_env)
            environment.update(
                {
                    "HOME": str(ephemeral_user),
                    "CLAUDE_CONFIG_DIR": str(config_dir),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "XDG_CACHE_HOME": str(xdg_cache),
                    "TMPDIR": str(process_temp),
                    "ANTHROPIC_API_KEY": credential,
                    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
                    "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
                    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
                    "DISABLE_UPDATES": "1",
                    "DISABLE_TELEMETRY": "1",
                    "DISABLE_ERROR_REPORTING": "1",
                    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
                    "BASH_DEFAULT_TIMEOUT_MS": str(
                        int(self.tool_timeout_seconds * 1000)
                    ),
                    "BASH_MAX_TIMEOUT_MS": str(int(self.tool_timeout_seconds * 1000)),
                    "CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS": str(
                        int(self.tool_timeout_seconds * 1000)
                    ),
                }
            )
            capture_stop = asyncio.Event()
            capture_task = asyncio.create_task(
                _capture_live_team_configs(config_dir, capture_stop)
            )
            live_team_configs: Mapping[str, Mapping[str, Any]] = {}
            try:
                try:
                    process, stdout, stderr = await self._run_process(
                        self._command(),
                        cwd=trial_workspace,
                        environment=environment,
                        input_data=request.prompt.encode("utf-8"),
                        timeout_seconds=self.timeout_seconds,
                        label="Claude Code Agent Teams session",
                    )
                finally:
                    capture_stop.set()
                    live_team_configs = await capture_task
                decoded = stdout.decode("utf-8", errors="replace")
                events: list[dict[str, Any]] = []
                for line in decoded.splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            "Claude Code emitted a non-JSON stdout line",
                            usage=usage,
                            raw={
                                "stdout": decoded,
                                "stderr": stderr.decode("utf-8", errors="replace"),
                            },
                        ) from exc
                    if not isinstance(event, dict):
                        raise ProviderError(
                            "Claude Code emitted a non-object JSON event",
                            usage=usage,
                            raw=events,
                        )
                    events.append(event)
                if not events:
                    raise ProviderError(
                        "Claude Code stream-json output was empty",
                        usage=usage,
                        raw={"stderr": stderr.decode("utf-8", errors="replace")},
                    )
                first = events[0]
                if (
                    first.get("type") != "system"
                    or first.get("subtype") != "init"
                    or first.get("claude_code_version") != self.expected_version
                ):
                    raise ProviderError(
                        "Claude Code stream-json did not start with the pinned init "
                        "event",
                        usage=usage,
                        raw=events,
                    )
                results = [event for event in events if event.get("type") == "result"]
                if len(results) != 1 or events[-1] is not results[0]:
                    raise ProviderError(
                        "Claude Code stream-json must end with exactly one result event",
                        usage=usage,
                        raw=events,
                    )
                result = results[0]
                usage = _usage_lower_bound(result)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                evidence = dict(_team_evidence(events))
                evidence.update(
                    _session_team_storage_evidence(
                        config_dir,
                        session_id=result.get("session_id"),
                        named_teammates=evidence["named_teammates"],
                        live_snapshots=live_team_configs,
                        deprecated_team_name_inputs=evidence[
                            "deprecated_team_name_inputs"
                        ],
                        removed_team_create_calls=evidence["removed_team_create_calls"],
                        removed_team_delete_calls=evidence["removed_team_delete_calls"],
                    )
                )
                evidence["cleanup_deferred_to_ephemeral_boundary"] = True
                post_trial_hash = await self._workspace_hash(
                    trial_workspace, "the disposable Claude Code workspace"
                )
                patch_payload = await self._export_disposable_patch(
                    trial_workspace, git_environment
                )
                raw_result = {
                    "events": events,
                    "terminal_result": result,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "team_evidence": evidence,
                    "workspace": {
                        "isolation": (
                            "fresh-disposable-copy-no-links-inherited-git-stripped"
                        ),
                        "inherited_git_metadata_used": False,
                        "fresh_git_baseline": True,
                        "base_cwd": str(self.workspace),
                        "trial_cwd": str(trial_workspace),
                        "base_tree_sha256": self.workspace_tree_sha256,
                        "post_trial_tree_sha256": post_trial_hash,
                        "patch": patch_payload,
                    },
                    "distribution": {
                        "version": self.expected_version,
                        "native_package": self.native_package_name,
                        "executable_sha256": self.expected_executable_sha256,
                        "artifact_kind": "official_binary_distribution_not_source",
                        "official_distribution_verified": (
                            self.official_distribution_verified
                        ),
                    },
                    "usage_scope": (
                        "terminal result is a lower bound; whole-team coverage "
                        "is not documented"
                    ),
                }
                if process.returncode != 0:
                    raise ProviderError(
                        f"Claude Code exited with status {process.returncode}",
                        usage=usage,
                        raw=raw_result,
                    )
                if result.get("is_error") is not False or result.get("subtype") != (
                    "success"
                ):
                    raise ProviderError(
                        "Claude Code Agent Teams session did not end successfully",
                        usage=usage,
                        raw=raw_result,
                    )
                answer = result.get("result")
                if not isinstance(answer, str) or not answer:
                    raise ProviderError(
                        "Claude Code result contained no final text",
                        usage=usage,
                        raw=raw_result,
                    )
                if self.require_team_evidence and not evidence["native_team_observed"]:
                    raise ProviderError(
                        "Claude Code completed without observable native Agent "
                        "Teams evidence",
                        usage=usage,
                        raw=raw_result,
                    )
                return ModelResponse(text=answer, usage=usage, raw=raw_result)
            except asyncio.CancelledError as exc:
                # A cancellation can arrive while hashing or exporting the patch,
                # after the terminal result has already exposed billable usage.
                # Replace the process helper's generic zero-usage marker with the
                # strongest observation available to the shared ledger.
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except ProviderError as exc:
                # Postprocessing helpers do not know the child session's usage.
                # Preserve their diagnostics while attaching the terminal lower
                # bound parsed immediately after the child returned.
                exc.usage = usage
                raise
            finally:
                try:
                    _assert_no_host_managed_settings(runtime_error=True)
                    self._validate_distribution(runtime_error=True)
                    after = await self._workspace_hash(
                        self.workspace,
                        "the Claude Code base workspace after the session",
                    )
                    if after != self.workspace_tree_sha256:
                        raise ProviderError(
                            "Claude Code escaped its disposable workspace and changed "
                            "the base workspace",
                            usage=usage,
                        )
                except ProviderError as exc:
                    # Integrity validation runs in ``finally`` and can otherwise
                    # replace a post-session failure with an unaccounted one.
                    exc.usage = usage
                    raise
