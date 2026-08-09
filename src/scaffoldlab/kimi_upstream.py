from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from .external import (
    _CapturedOutput,
    _ProcessOutputLimitExceeded,
    _communicate_limited,
    _git_workspace_provenance,
    _prepare_executable,
    _process_environment,
    _resolved_executable_identity,
    _terminate_process_tree,
    _workspace_tree_sha256,
    _workspace_tree_sha256_async,
)
from .environment_policy import reject_runtime_environment_overrides
from .providers import ProviderError
from .source_integrity import require_standalone_git_checkout
from .types import ModelRequest, ModelResponse, Usage


KIMI_CODE_RELEASE_REVISION = "f0614c53e59f7e1e257412063b059b9eb82764cf"
_MAX_TRACKED_SOURCE_ENTRIES = 250_000
_MAX_TRACKED_SOURCE_BYTES = 16 * 1024 * 1024 * 1024
_TRACKED_SOURCE_TIMEOUT_SECONDS = 60.0
_MAX_LS_TREE_BYTES = 64 * 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_git_environment() -> dict[str, str]:
    environment = _process_environment(())
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _source_git_provenance(checkout: Path, *, git_executable: str) -> Mapping[str, Any]:
    environment = _source_git_environment()
    try:
        head = subprocess.run(
            [git_executable, "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            env=environment,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            [
                git_executable,
                "-C",
                str(checkout),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            env=environment,
            timeout=30,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False}
    return {
        "available": True,
        "head": head.decode("ascii", errors="replace"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _validate_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    git_executable: str,
    label: str = "Kimi Code",
) -> Mapping[str, Any]:
    require_standalone_git_checkout(checkout, label=f"{label} checkout")
    provenance = _source_git_provenance(
        checkout,
        git_executable=git_executable,
    )
    if provenance.get("available") is not True:
        raise ValueError(f"{label} checkout must be a Git working tree")
    observed = provenance.get("head")
    if observed != expected_revision:
        raise ValueError(
            f"{label} checkout revision mismatch: expected {expected_revision!r}, "
            f"observed {observed!r}"
        )
    if provenance.get("dirty") is not False:
        raise ValueError(f"{label} checkout must be clean")
    return provenance


def _tracked_tree_attestation(
    checkout: Path,
    *,
    expected_revision: str,
    git_executable: str,
) -> Mapping[str, Any]:
    """Compare every tracked worktree entry with the pinned commit's Git blob."""

    started = time.monotonic()
    try:
        listing = subprocess.run(
            [
                git_executable,
                "-C",
                str(checkout),
                "ls-tree",
                "-rz",
                "--full-tree",
                expected_revision,
            ],
            check=True,
            capture_output=True,
            env=_source_git_environment(),
            timeout=_TRACKED_SOURCE_TIMEOUT_SECONDS,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "Kimi Code could not enumerate the pinned source tree"
        ) from exc
    if len(listing) > _MAX_LS_TREE_BYTES:
        raise ValueError("Kimi Code tracked source listing exceeds its byte limit")

    records = [record for record in listing.split(b"\0") if record]
    if len(records) > _MAX_TRACKED_SOURCE_ENTRIES:
        raise ValueError("Kimi Code tracked source tree exceeds its entry limit")
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    total_bytes = 0
    for record in records:
        if time.monotonic() - started >= _TRACKED_SOURCE_TIMEOUT_SECONDS:
            raise ValueError("Kimi Code tracked source attestation timed out")
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, object_type, raw_object_id = metadata.split(b" ", 2)
            mode = int(raw_mode, 8)
            object_id = raw_object_id.decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise ValueError(
                "Kimi Code emitted a malformed tracked-tree record"
            ) from exc
        if object_type != b"blob" or mode not in {0o100644, 0o100755, 0o120000}:
            raise ValueError(
                "Kimi Code tracked source contains an unsupported Git entry type"
            )
        path_text = os.fsdecode(raw_path)
        relative = Path(path_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative in {Path("."), Path("..")}
            or ".." in relative.parts
        ):
            raise ValueError("Kimi Code tracked source contains an unsafe path")
        normalized = relative.as_posix()
        if normalized in seen or normalized.casefold() in seen_casefolded:
            raise ValueError(
                "Kimi Code tracked source contains duplicate or case-ambiguous paths"
            )
        seen.add(normalized)
        seen_casefolded.add(normalized.casefold())

        candidate = checkout.joinpath(*relative.parts)
        try:
            candidate_stat = candidate.lstat()
            if mode == 0o120000:
                if not stat.S_ISLNK(candidate_stat.st_mode):
                    raise ValueError(
                        f"Kimi Code tracked symbolic link changed: {normalized}"
                    )
                content = os.fsencode(os.readlink(candidate))
                total_bytes += len(content)
                if total_bytes > _MAX_TRACKED_SOURCE_BYTES:
                    raise ValueError(
                        "Kimi Code tracked source tree exceeds its byte limit"
                    )
                blob_digest = hashlib.sha1(  # noqa: S324 - Git object identity.
                    f"blob {len(content)}\0".encode("ascii") + content
                )
            else:
                if not stat.S_ISREG(candidate_stat.st_mode):
                    raise ValueError(
                        f"Kimi Code tracked regular file changed: {normalized}"
                    )
                executable = bool(stat.S_IMODE(candidate_stat.st_mode) & 0o111)
                if executable != (mode == 0o100755):
                    raise ValueError(
                        f"Kimi Code tracked executable mode changed: {normalized}"
                    )
                file_bytes = candidate_stat.st_size
                total_bytes += file_bytes
                if total_bytes > _MAX_TRACKED_SOURCE_BYTES:
                    raise ValueError(
                        "Kimi Code tracked source tree exceeds its byte limit"
                    )
                blob_digest = hashlib.sha1(  # noqa: S324 - Git object identity.
                    f"blob {file_bytes}\0".encode("ascii")
                )
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        if (
                            time.monotonic() - started
                            >= _TRACKED_SOURCE_TIMEOUT_SECONDS
                        ):
                            raise ValueError(
                                "Kimi Code tracked source attestation timed out"
                            )
                        blob_digest.update(chunk)
        except OSError as exc:
            raise ValueError(
                f"Kimi Code tracked source entry is unreadable: {normalized}"
            ) from exc
        blob = blob_digest.hexdigest()
        if blob != object_id:
            raise ValueError(
                f"Kimi Code tracked source content differs from the pinned commit: "
                f"{normalized}"
            )

    return {
        "verified": True,
        "revision": expected_revision,
        "entries": len(records),
        "content_bytes": total_bytes,
        "ls_tree_sha256": hashlib.sha256(listing).hexdigest(),
        "scope": "all Git blob entries in the pinned commit",
    }


class KimiCodeUpstreamBackend:
    """Execute the pinned Kimi Code TypeScript source in non-interactive mode.

    This adapter intentionally starts the official ``src/main.ts`` through the
    checkout's own ``tsx`` dependency. It does not reconstruct Agent, AgentSwarm,
    permissions, compaction, prompts, or tool execution. One Scaffold Lab model
    call therefore represents an entire upstream Kimi session tree.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        cwd: Path,
        model: str,
        api_key_env: str,
        provider_type: str = "kimi",
        base_url: Optional[str] = None,
        expected_revision: str = KIMI_CODE_RELEASE_REVISION,
        expected_version: str = "0.34.0",
        node_executable: str = "node",
        git_executable: str = "git",
        tsx_executable: Optional[Path] = None,
        expected_node_sha256: Optional[str] = None,
        expected_git_sha256: Optional[str] = None,
        expected_tsx_sha256: Optional[str] = None,
        timeout_seconds: float = 1800.0,
        max_output_bytes: int = 16 * 1024 * 1024,
        max_swarm_concurrency: int = 8,
        max_steps_per_turn: int = 64,
        subagent_timeout_seconds: float = 1200.0,
        pass_env: Sequence[str] = (),
        allow_sensitive_environment: bool = False,
        allow_insecure_base_url: bool = False,
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Kimi Code source adapter currently requires POSIX process groups"
            )
        self.checkout = checkout.resolve()
        self.cwd = cwd.resolve()
        if not self.checkout.is_dir():
            raise ValueError(f"Kimi Code checkout is not a directory: {self.checkout}")
        if not self.cwd.is_dir():
            raise ValueError(f"Kimi Code cwd is not a directory: {self.cwd}")
        if self.checkout == self.cwd or self.checkout.is_relative_to(self.cwd):
            raise ValueError("Kimi Code checkout and task cwd must be disjoint")
        if self.cwd.is_relative_to(self.checkout):
            raise ValueError("Kimi Code task cwd cannot be inside the source checkout")
        git_pointer = self.cwd / ".git"
        if git_pointer.is_file() or git_pointer.is_symlink():
            raise ValueError(
                "Kimi Code task cwd cannot be a linked Git worktree; use a "
                "standalone clone or a directory without inherited Git metadata"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
            raise ValueError("expected_revision must be 40 lowercase hex characters")
        if re.fullmatch(r"\d+\.\d+\.\d+", expected_version) is None:
            raise ValueError("expected_version must be a canonical semantic version")
        if not model:
            raise ValueError("model must be non-empty")
        if not node_executable or "\x00" in node_executable:
            raise ValueError("node_executable must be a non-empty string")
        if not git_executable or "\x00" in git_executable:
            raise ValueError("git_executable must be a non-empty string")
        if not api_key_env or "=" in api_key_env or "\x00" in api_key_env:
            raise ValueError("api_key_env must be a non-empty environment name")
        if provider_type not in {"kimi", "anthropic", "openai"}:
            raise ValueError("provider_type must be kimi, anthropic, or openai")
        if base_url is not None:
            parsed_base_url = urlsplit(base_url)
            if (
                parsed_base_url.scheme not in {"https", "http"}
                or parsed_base_url.hostname is None
                or parsed_base_url.username is not None
                or parsed_base_url.password is not None
                or "\x00" in base_url
            ):
                raise ValueError(
                    "base_url must be an HTTP(S) URL without embedded credentials"
                )
            is_loopback = parsed_base_url.hostname.casefold() == "localhost"
            try:
                is_loopback = (
                    is_loopback
                    or ipaddress.ip_address(parsed_base_url.hostname).is_loopback
                )
            except ValueError:
                pass
            if (
                parsed_base_url.scheme == "http"
                and not is_loopback
                and not allow_insecure_base_url
            ):
                raise ValueError(
                    "HTTP Kimi base URLs can expose the model credential; use HTTPS, "
                    "a loopback test endpoint, or explicitly acknowledge the risk"
                )
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if (
            not isinstance(subagent_timeout_seconds, (int, float))
            or isinstance(subagent_timeout_seconds, bool)
            or not math.isfinite(subagent_timeout_seconds)
            or subagent_timeout_seconds <= 0
        ):
            raise ValueError("subagent_timeout_seconds must be positive and finite")
        for name, value in (
            ("max_output_bytes", max_output_bytes),
            ("max_swarm_concurrency", max_swarm_concurrency),
            ("max_steps_per_turn", max_steps_per_turn),
        ):
            minimum = 1024 if name == "max_output_bytes" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        reject_runtime_environment_overrides(
            pass_env,
            label="Kimi Code",
            reserved_prefixes=("KIMI_",),
        )
        if not allow_sensitive_environment:
            raise ValueError(
                "Kimi Code tools inherit the model credential; acknowledge this with "
                "allow_sensitive_environment only for a disposable outer sandbox"
            )
        for digest_value, digest_label in (
            (expected_node_sha256, "expected_node_sha256"),
            (expected_git_sha256, "expected_git_sha256"),
            (expected_tsx_sha256, "expected_tsx_sha256"),
        ):
            if (
                digest_value is not None
                and re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
            ):
                raise ValueError(f"{digest_label} must be 64 lowercase hex characters")

        self.expected_revision = expected_revision
        self.expected_version = expected_version
        resolved_git, git_identity = _prepare_executable(
            git_executable,
            expected_git_sha256,
            "Kimi Code Git",
        )
        if git_identity.get("available") is not True:
            raise ValueError(f"Kimi Code Git executable not found: {git_executable!r}")
        self.git_executable = resolved_git
        self.expected_git_sha256 = expected_git_sha256
        self._git_identity = dict(git_identity)
        self.checkout_git = _validate_checkout(
            self.checkout,
            expected_revision=expected_revision,
            git_executable=self.git_executable,
        )
        self.entrypoint = self.checkout / "apps" / "kimi-code" / "src" / "main.ts"
        self.raw_text_loader = self.checkout / "build" / "register-raw-text-loader.mjs"
        self.lockfile = self.checkout / "pnpm-lock.yaml"
        self.package_json = self.checkout / "apps" / "kimi-code" / "package.json"
        for path, label in (
            (self.entrypoint, "source entrypoint"),
            (self.raw_text_loader, "raw-text loader"),
            (self.lockfile, "pnpm lockfile"),
            (self.package_json, "package manifest"),
        ):
            if not path.is_file():
                raise ValueError(f"Kimi Code {label} is missing: {path}")
        self.tracked_source_tree = _tracked_tree_attestation(
            self.checkout,
            expected_revision=self.expected_revision,
            git_executable=self.git_executable,
        )
        self.entrypoint_sha256 = _file_sha256(self.entrypoint)
        self.raw_text_loader_sha256 = _file_sha256(self.raw_text_loader)
        self.lockfile_sha256 = _file_sha256(self.lockfile)
        self.package_json_sha256 = _file_sha256(self.package_json)
        configured_tsx = tsx_executable or (
            self.checkout / "node_modules" / ".bin" / "tsx"
        )
        self.tsx_executable = str(configured_tsx)
        self.node_executable = node_executable
        self.expected_node_sha256 = expected_node_sha256
        self.expected_tsx_sha256 = expected_tsx_sha256
        self.model = model
        self.api_key_env = api_key_env
        self.provider_type = provider_type
        self.base_url = base_url
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.max_swarm_concurrency = max_swarm_concurrency
        self.max_steps_per_turn = max_steps_per_turn
        self.subagent_timeout_seconds = float(subagent_timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.allow_sensitive_environment = allow_sensitive_environment
        self.allow_insecure_base_url = allow_insecure_base_url
        self.workspace_tree_sha256 = _workspace_tree_sha256(self.cwd)
        self.git_workspace = _git_workspace_provenance(self.cwd)
        self._resolved_tsx: Optional[str] = None
        self._resolved_node: Optional[str] = None
        self._node_identity: Mapping[str, Any] = {"available": False}
        self._tsx_identity: Mapping[str, Any] = {"available": False}
        self._observed_version: Optional[str] = None

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "kimi-code-upstream-source",
            "checkout": str(self.checkout),
            "checkout_git": dict(self.checkout_git),
            "expected_revision": self.expected_revision,
            "source_or_protocol_pin_verified": True,
            "bit_reproducible_runtime_verified": False,
            "runtime_source_identity_verified": True,
            "runtime_source_identity_scope": (
                "all tracked Git blobs match the pinned commit; ignored/generated "
                "dependency content is excluded"
            ),
            "tracked_source_tree": dict(self.tracked_source_tree),
            "caller_worktree_executed": True,
            "private_source_export_executed": False,
            "adversarial_full_runtime_content_attestation_verified": False,
            "ignored_or_generated_dependency_content_verified": False,
            "expected_version": self.expected_version,
            "entrypoint": str(self.entrypoint.relative_to(self.checkout)),
            "entrypoint_sha256": self.entrypoint_sha256,
            "raw_text_loader": str(self.raw_text_loader.relative_to(self.checkout)),
            "raw_text_loader_sha256": self.raw_text_loader_sha256,
            "pnpm_lock_sha256": self.lockfile_sha256,
            "package_json_sha256": self.package_json_sha256,
            "git_runtime": {
                **self._git_identity,
                "expected_sha256": self.expected_git_sha256,
                "caller_sha256_pin_verified": self.expected_git_sha256 is not None,
            },
            "runtime_executable": dict(self._tsx_identity),
            "expected_tsx_sha256": self.expected_tsx_sha256,
            "node_runtime": dict(self._node_identity),
            "expected_node_sha256": self.expected_node_sha256,
            "observed_version": self._observed_version,
            "version_verified": self._observed_version is not None,
            "cwd": str(self.cwd),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "model": self.model,
            "provider_type": self.provider_type,
            "base_url": self.base_url,
            "insecure_base_url_acknowledged": self.allow_insecure_base_url,
            "api_key_environment_name": self.api_key_env,
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "output_format": "stream-json",
            "prompt_transport": "official --prompt command argument",
            "prompt_visible_to_local_process_inspection": True,
            "permission_mode": "upstream-print-mode-auto",
            "home": "ephemeral-per-call",
            "xdg_config_home": "ephemeral-per-call",
            "xdg_cache_home": "ephemeral-per-call",
            "tmpdir": "ephemeral-per-call",
            "max_swarm_concurrency": self.max_swarm_concurrency,
            "max_steps_per_turn": self.max_steps_per_turn,
            "subagent_timeout_seconds": self.subagent_timeout_seconds,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "one_backend_call_is_external_session_tree": True,
            "usage_scope": "not emitted by Kimi stream-json; unknown",
            "dependency_identity": "lockfile-recorded; installed tree not content-hashed",
            "flagship_system_card_parity_claimed": False,
        }

    async def prepare_for_manifest(self) -> None:
        """Resolve and version-check Node/tsx before run fingerprinting."""

        await self.verify_version()

    def _verify_runtime_identity(self, usage: Optional[Usage] = None) -> None:
        try:
            _validate_checkout(
                self.checkout,
                expected_revision=self.expected_revision,
                git_executable=self.git_executable,
                label="Kimi Code source",
            )
            tracked_source_tree = _tracked_tree_attestation(
                self.checkout,
                expected_revision=self.expected_revision,
                git_executable=self.git_executable,
            )
        except ValueError as exc:
            raise ProviderError(
                f"Kimi Code source identity violation: {exc}",
                usage=usage or Usage(cost_known=False, complete=False),
            ) from exc
        if tracked_source_tree != self.tracked_source_tree:
            raise ProviderError(
                "Kimi Code source identity violation: tracked source tree changed",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={"tracked_source_tree": dict(tracked_source_tree)},
            )
        for path, expected_sha256, label in (
            (self.entrypoint, self.entrypoint_sha256, "source entrypoint"),
            (self.raw_text_loader, self.raw_text_loader_sha256, "raw-text loader"),
            (self.lockfile, self.lockfile_sha256, "pnpm lockfile"),
            (self.package_json, self.package_json_sha256, "package manifest"),
        ):
            try:
                observed_sha256 = _file_sha256(path)
            except OSError as exc:
                raise ProviderError(
                    f"Kimi Code source identity violation: {label} became unreadable",
                    usage=usage or Usage(cost_known=False, complete=False),
                ) from exc
            if observed_sha256 != expected_sha256:
                raise ProviderError(
                    f"Kimi Code source identity violation: {label} changed",
                    usage=usage or Usage(cost_known=False, complete=False),
                )
        current_git = _resolved_executable_identity(self.git_executable)
        if current_git.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or current_git.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError(
                "Kimi Code source identity violation: Git executable changed",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={"git_runtime": dict(current_git)},
            )
        if self._resolved_tsx is None or self._resolved_node is None:
            return
        current_tsx = _resolved_executable_identity(self._resolved_tsx)
        current_node = _resolved_executable_identity(self._resolved_node)
        if current_tsx.get("resolved_path") != self._tsx_identity.get(
            "resolved_path"
        ) or current_tsx.get("sha256") != self._tsx_identity.get("sha256"):
            raise ProviderError(
                "Kimi Code source identity violation: tsx executable changed",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={"runtime_executable": dict(current_tsx)},
            )
        if current_node.get("resolved_path") != self._node_identity.get(
            "resolved_path"
        ) or current_node.get("sha256") != self._node_identity.get("sha256"):
            raise ProviderError(
                "Kimi Code source identity violation: Node executable changed",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={"node_runtime": dict(current_node)},
            )

    def _command(self) -> list[str]:
        return [
            self._resolved_node or self.node_executable,
            self._resolved_tsx or self.tsx_executable,
            "--import",
            str(self.raw_text_loader),
            str(self.entrypoint),
        ]

    async def verify_version(self) -> str:
        if self._observed_version is not None:
            return self._observed_version
        resolved, identity = _prepare_executable(
            self.tsx_executable,
            self.expected_tsx_sha256,
            "Kimi Code tsx",
        )
        self._resolved_tsx = resolved
        self._tsx_identity = identity
        resolved_node, node_identity = _prepare_executable(
            self.node_executable,
            self.expected_node_sha256,
            "Kimi Code Node",
        )
        if node_identity.get("available") is not True:
            raise ProviderError(
                f"Kimi Code Node executable was not found: {self.node_executable!r}"
            )
        self._resolved_node = resolved_node
        self._node_identity = node_identity
        resolved_path = identity.get("resolved_path")
        if self.expected_tsx_sha256 is None and (
            not isinstance(resolved_path, str)
            or not Path(resolved_path).is_relative_to(self.checkout)
        ):
            raise ProviderError(
                "Kimi Code tsx must come from the pinned checkout unless its "
                "executable SHA-256 is explicitly pinned"
            )
        with tempfile.TemporaryDirectory(prefix="scaffoldlab-kimi-version-") as root:
            version_root = Path(root)
            home = version_root / "home"
            config = version_root / "config"
            kimi_home = version_root / "kimi-code"
            for directory in (home, config, kimi_home):
                directory.mkdir(mode=0o700)
            version_environment = _process_environment(())
            version_environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(config),
                    "KIMI_CODE_HOME": str(kimi_home),
                    "KIMI_DISABLE_TELEMETRY": "1",
                    "KIMI_CODE_NO_AUTO_UPDATE": "1",
                }
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(),
                    "--version",
                    cwd=str(self.cwd),
                    env=version_environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    "Kimi Code source dependencies are not installed; run the pinned "
                    "checkout's frozen pnpm install first"
                ) from exc
            captured = _CapturedOutput()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data=None,
                        max_output_bytes=min(self.max_output_bytes, 1024 * 1024),
                        captured=captured,
                    ),
                    timeout=min(self.timeout_seconds, 30.0),
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process_tree(process)
                raise ProviderError("Kimi Code version probe timed out") from exc
            except asyncio.CancelledError:
                await _terminate_process_tree(process)
                raise
            except _ProcessOutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "Kimi Code version probe exceeded its output limit"
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise
            await _terminate_process_tree(process)
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
                f"Kimi Code version probe exited with status {process.returncode}",
                raw={"output": output},
            )
        versions = set(re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?![\d.])", output))
        if versions != {self.expected_version}:
            raise ProviderError(
                "Kimi Code version mismatch",
                raw={"expected": self.expected_version, "output": output},
            )
        self._observed_version = self.expected_version
        self._verify_runtime_identity()
        return self._observed_version

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if (
            request.system
            or request.tools
            or request.tool_results
            or request.continuation
        ):
            raise ProviderError(
                "Kimi Code source owns system prompts, tools, and continuation state"
            )
        _validate_checkout(
            self.checkout,
            expected_revision=self.expected_revision,
            git_executable=self.git_executable,
        )
        try:
            if self._observed_version is None:
                await self.verify_version()
            self._verify_runtime_identity()
            try:
                current_workspace_hash = await _workspace_tree_sha256_async(
                    self.cwd,
                    max_entries=250_000,
                    max_bytes=16 * 1024 * 1024 * 1024,
                    timeout_seconds=30.0,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "Kimi Code could not hash its task workspace before the session"
                ) from exc
            if current_workspace_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Kimi Code task workspace changed after backend initialization"
                )
            if self.api_key_env not in os.environ or not os.environ[self.api_key_env]:
                raise ProviderError(
                    f"missing Kimi Code model credential environment {self.api_key_env!r}"
                )

            with tempfile.TemporaryDirectory(
                prefix="scaffoldlab-kimi-home-"
            ) as temp_dir:
                temp_root = Path(temp_dir)
                home = temp_root / "home"
                xdg_config = temp_root / "xdg-config"
                xdg_cache = temp_root / "xdg-cache"
                process_temp = temp_root / "tmp"
                kimi_home = temp_root / "kimi-code"
                for directory in (
                    home,
                    xdg_config,
                    xdg_cache,
                    process_temp,
                    kimi_home,
                ):
                    directory.mkdir(mode=0o700)
                environment = _process_environment(self.pass_env)
                environment.update(
                    {
                        "HOME": str(home),
                        "XDG_CONFIG_HOME": str(xdg_config),
                        "XDG_CACHE_HOME": str(xdg_cache),
                        "TMPDIR": str(process_temp),
                        "KIMI_CODE_HOME": str(kimi_home),
                        "KIMI_DISABLE_TELEMETRY": "1",
                        "KIMI_CODE_NO_AUTO_UPDATE": "1",
                        "KIMI_MODEL_NAME": self.model,
                        "KIMI_MODEL_API_KEY": os.environ[self.api_key_env],
                        "KIMI_MODEL_PROVIDER_TYPE": self.provider_type,
                        "KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY": str(
                            self.max_swarm_concurrency
                        ),
                        "KIMI_LOOP_MAX_STEPS_PER_TURN": str(self.max_steps_per_turn),
                        "KIMI_SUBAGENT_TIMEOUT_MS": str(
                            int(self.subagent_timeout_seconds * 1000)
                        ),
                    }
                )
                if self.base_url is not None:
                    environment["KIMI_MODEL_BASE_URL"] = self.base_url
                response = await self._complete_with_environment(request, environment)
        except asyncio.CancelledError as exc:
            try:
                self._verify_runtime_identity(getattr(exc, "usage", None))
            except ProviderError as integrity_error:
                exc.source_integrity_error = str(  # type: ignore[attr-defined]
                    integrity_error
                )
            raise
        except Exception as exc:
            try:
                self._verify_runtime_identity(getattr(exc, "usage", None))
            except ProviderError as integrity_error:
                raise integrity_error from exc
            raise
        self._verify_runtime_identity(response.usage)
        return response

    async def _complete_with_environment(
        self, request: ModelRequest, environment: Mapping[str, str]
    ) -> ModelResponse:
        command = [
            *self._command(),
            "--prompt",
            request.prompt,
            "--output-format",
            "stream-json",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.cwd),
                env=dict(environment),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ProviderError("Kimi Code source launcher was not found") from exc
        captured = _CapturedOutput()
        try:
            stdout, stderr = await asyncio.wait_for(
                _communicate_limited(
                    process,
                    input_data=None,
                    max_output_bytes=self.max_output_bytes,
                    captured=captured,
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_tree(process)
            raise ProviderError(
                "Kimi Code session timed out",
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
                "Kimi Code session exceeded its output limit",
                usage=Usage(cost_known=False, complete=False),
            ) from exc
        except Exception:
            await _terminate_process_tree(process)
            raise
        await _terminate_process_tree(process)

        usage = Usage(cost_known=False, complete=False)
        events: list[dict[str, Any]] = []
        decoded_stdout = stdout.decode("utf-8", errors="replace")
        for line in decoded_stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "Kimi Code emitted a non-JSON stdout line",
                    usage=usage,
                    raw={
                        "stdout": decoded_stdout,
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                ) from exc
            if not isinstance(event, dict):
                raise ProviderError(
                    "Kimi Code emitted a non-object JSON event",
                    usage=usage,
                    raw=events,
                )
            events.append(event)
        if not events:
            raise ProviderError(
                "Kimi Code stream-json output was empty",
                usage=usage,
                raw={"stderr": stderr.decode("utf-8", errors="replace")},
            )
        first = events[0]
        if first != {
            "role": "meta",
            "type": "system.version",
            "version": self.expected_version,
        }:
            raise ProviderError(
                "Kimi Code stream-json did not start with the pinned version event",
                usage=usage,
                raw=events,
            )
        version_events = [
            event
            for event in events
            if event.get("role") == "meta" and event.get("type") == "system.version"
        ]
        if len(version_events) != 1:
            raise ProviderError(
                "Kimi Code stream-json must contain exactly one version event",
                usage=usage,
                raw=events,
            )
        if process.returncode != 0:
            raise ProviderError(
                f"Kimi Code exited with status {process.returncode}",
                usage=usage,
                raw={
                    "events": events,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        terminal = events[-1]
        if (
            terminal.get("role") != "meta"
            or terminal.get("type") != "session.resume_hint"
            or not isinstance(terminal.get("session_id"), str)
            or not terminal["session_id"]
            or not isinstance(terminal.get("command"), str)
            or not terminal["command"]
            or not isinstance(terminal.get("content"), str)
        ):
            raise ProviderError(
                "Kimi Code stream-json did not end with a valid session.resume_hint",
                usage=usage,
                raw=events,
            )
        answer_parts = [
            event["content"]
            for event in events
            if event.get("role") == "assistant"
            and isinstance(event.get("content"), str)
            and event["content"]
        ]
        if not answer_parts:
            raise ProviderError(
                "Kimi Code stream-json contained no assistant text",
                usage=usage,
                raw=events,
            )
        post_workspace_hash = await _workspace_tree_sha256_async(
            self.cwd,
            max_entries=250_000,
            max_bytes=16 * 1024 * 1024 * 1024,
            timeout_seconds=30.0,
        )
        raw: dict[str, Any] = {
            "events": events,
            "stderr": stderr.decode("utf-8", errors="replace"),
            "workspace": {
                "cwd": str(self.cwd),
                "pre_tree_sha256": self.workspace_tree_sha256,
                "post_tree_sha256": post_workspace_hash,
            },
            "usage_scope": "not emitted by Kimi stream-json; unknown",
        }
        return ModelResponse(text=answer_parts[-1], usage=usage, raw=raw)
