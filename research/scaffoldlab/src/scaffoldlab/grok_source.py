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
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from .external import (
    _CapturedOutput,
    _ProcessOutputLimitExceeded,
    _communicate_limited,
    _git_workspace_provenance,
    _mark_whole_tree_unverified,
    _prepare_executable,
    _process_environment,
    _terminate_process_tree,
    _usage_from_grok_result,
    _workspace_tree_sha256,
    _workspace_tree_sha256_async,
)
from .environment_policy import reject_runtime_environment_overrides
from .environments.swe import SWEPatchPayload
from .providers import ProviderError
from .source_integrity import require_standalone_git_checkout
from .types import ModelRequest, ModelResponse, Usage


GROK_BUILD_PUBLIC_REVISION = "8a14c91d88875a831a38b3a066b1683116bcb31c"
GROK_BUILD_SOURCE_REV = "27b3c66635e2c0bf213429a36ab916f25d59df20"
GROK_BUILD_CARGO_LOCK_SHA256 = (
    "285e13b019551e76680a21fe300dda5934aba9bcf1f7e7ad24b2ddee7fd3eb92"
)

_DEFAULT_MAX_WORKSPACE_ENTRIES = 250_000
_DEFAULT_MAX_WORKSPACE_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS = 30.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Optional[str], name: str) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters")


def _copytree_ignore_git_metadata(_directory: str, names: list[str]) -> set[str]:
    """Strip every nested Git administrative entry from a trial copy."""

    return {name for name in names if name.casefold() == ".git"}


def _is_rustup_proxy(identity: Mapping[str, Any]) -> bool:
    """Recognize both common resolved names of rustup's multicall proxy."""

    return any(
        isinstance(value, str)
        and Path(value).name.casefold() in {"rustup", "rustup-init"}
        for value in (
            identity.get("invoked_path"),
            identity.get("resolved_path"),
        )
    )


def _extract_verified_git_archive(
    archive: Path,
    destination: Path,
    *,
    max_entries: int,
    max_bytes: int,
) -> None:
    """Extract an uncompressed Git archive without trusting tar path handling.

    A caller-selected Git binary is not necessarily hash-pinned, so archive output is
    treated as hostile even though normal ``git archive`` output is simple. Hard links
    and special files are not emitted for Git tree entries and are rejected. Symlinks
    are created only after all directories and regular files, preventing a link entry
    from redirecting a later extraction outside ``destination``.
    """

    with tarfile.open(archive, mode="r:") as stream:
        members = stream.getmembers()
        if len(members) > max_entries:
            raise ProviderError("Grok Build Git archive exceeds the source entry limit")

        normalized: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        seen_paths: set[str] = set()
        casefolded_paths: set[str] = set()
        non_directory_paths: set[PurePosixPath] = set()
        total_bytes = 0
        for member in members:
            path = PurePosixPath(member.name)
            normalized_path = PurePosixPath(posixpath.normpath(member.name))
            if (
                path.is_absolute()
                or not path.parts
                or path != normalized_path
                or ".." in path.parts
                or path == PurePosixPath(".")
            ):
                raise ProviderError("Grok Build Git archive contains an unsafe path")
            path_text = path.as_posix()
            if path_text in seen_paths or path_text.casefold() in casefolded_paths:
                raise ProviderError(
                    "Grok Build Git archive contains duplicate or case-ambiguous paths"
                )
            seen_paths.add(path_text)
            casefolded_paths.add(path_text.casefold())

            if member.isdir():
                pass
            elif member.isfile():
                total_bytes += member.size
                if total_bytes > max_bytes:
                    raise ProviderError(
                        "Grok Build Git archive exceeds the source byte limit"
                    )
                non_directory_paths.add(path)
            elif member.issym():
                link_target = PurePosixPath(member.linkname)
                resolved_target = PurePosixPath(
                    posixpath.normpath(str(path.parent / link_target))
                )
                if (
                    link_target.is_absolute()
                    or not link_target.parts
                    or (resolved_target.parts and resolved_target.parts[0] == "..")
                ):
                    raise ProviderError(
                        "Grok Build Git archive contains an escaping symlink"
                    )
                non_directory_paths.add(path)
            else:
                raise ProviderError(
                    "Grok Build Git archive contains a link or special file"
                )
            normalized.append((member, path))

        for _member, path in normalized:
            if any(parent in non_directory_paths for parent in path.parents):
                raise ProviderError(
                    "Grok Build Git archive nests content below a non-directory"
                )

        destination.mkdir(mode=0o700)
        for member, path in sorted(normalized, key=lambda item: len(item[1].parts)):
            output_path = destination.joinpath(*path.parts)
            if member.isdir():
                output_path.mkdir(mode=member.mode & 0o777, parents=True, exist_ok=True)
                output_path.chmod(member.mode & 0o777)
                continue
            if member.issym():
                continue
            output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ProviderError(
                    "Grok Build Git archive regular file has no content"
                )
            remaining = member.size
            with output_path.open("xb") as output:
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ProviderError(
                            "Grok Build Git archive contains a truncated file"
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    raise ProviderError(
                        "Grok Build Git archive file exceeds its declared size"
                    )
            output_path.chmod(member.mode & 0o777)

        for member, path in normalized:
            if not member.issym():
                continue
            output_path = destination.joinpath(*path.parts)
            output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            output_path.symlink_to(member.linkname)


class GrokBuildSourceBackend:
    """Build and execute the pinned public Grok Build Rust source.

    The source checkout supplies Grok's prompts, tools, worktree behavior,
    compaction, scheduler, and native subagents. Scaffold Lab only verifies and
    builds that checkout, creates a fresh copy of the task workspace, invokes the
    documented headless JSON protocol, and projects the terminal result into the
    shared ledger. One ``complete`` call is an entire external agent tree.

    Grok's terminal JSON accounting is retained as an observed lower bound. It is
    not claimed to cover compaction, side-model work, unfinished calls, or every
    nested subagent call.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        workspace: Path,
        model: str,
        cargo_executable: str = "cargo",
        rustc_executable: Optional[str] = None,
        git_executable: str = "git",
        sandbox: str = "strict",
        permission_mode: str = "dontAsk",
        max_turns: int = 64,
        timeout_seconds: float = 1800.0,
        build_timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        allow_sensitive_environment: bool = False,
        expected_checkout_revision: str = GROK_BUILD_PUBLIC_REVISION,
        expected_source_rev: str = GROK_BUILD_SOURCE_REV,
        expected_cargo_lock_sha256: str = GROK_BUILD_CARGO_LOCK_SHA256,
        expected_executable_sha256: Optional[str] = None,
        expected_cargo_sha256: Optional[str] = None,
        expected_rustc_sha256: Optional[str] = None,
        expected_git_sha256: Optional[str] = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        workspace_hash_max_entries: int = _DEFAULT_MAX_WORKSPACE_ENTRIES,
        workspace_hash_max_bytes: int = _DEFAULT_MAX_WORKSPACE_BYTES,
        workspace_hash_timeout_seconds: float = (
            _DEFAULT_WORKSPACE_HASH_TIMEOUT_SECONDS
        ),
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Grok Build source adapter currently requires POSIX process groups"
            )
        self.checkout = checkout.resolve()
        self.workspace = workspace.resolve()
        if not self.checkout.is_dir():
            raise ValueError(
                f"Grok Build source checkout is not a directory: {self.checkout}"
            )
        if not self.workspace.is_dir():
            raise ValueError(
                f"Grok Build base workspace is not a directory: {self.workspace}"
            )
        if (
            self.checkout == self.workspace
            or self.checkout.is_relative_to(self.workspace)
            or self.workspace.is_relative_to(self.checkout)
        ):
            raise ValueError(
                "Grok Build source checkout and workspace must be disjoint"
            )
        git_entries = [
            entry.name
            for entry in self.workspace.iterdir()
            if entry.name.casefold() == ".git"
        ]
        if any(name != ".git" for name in git_entries):
            raise ValueError(
                "Grok Build workspace contains case-variant Git metadata; provide "
                "an unambiguous standalone workspace"
            )
        git_pointer = self.workspace / ".git"
        if git_pointer.is_file() or git_pointer.is_symlink():
            raise ValueError(
                "Grok Build workspace cannot be a linked Git worktree; use a normal "
                "clone or a directory without external Git metadata"
            )
        for name, string_value in (
            ("model", model),
            ("cargo_executable", cargo_executable),
            ("git_executable", git_executable),
            ("sandbox", sandbox),
            ("permission_mode", permission_mode),
        ):
            if (
                not isinstance(string_value, str)
                or not string_value
                or "\x00" in string_value
            ):
                raise ValueError(f"{name} must be a non-empty string")
        if rustc_executable is not None and (
            not isinstance(rustc_executable, str)
            or not rustc_executable
            or "\x00" in rustc_executable
        ):
            raise ValueError("rustc_executable must be a non-empty string or null")
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer")
        for name, numeric_value in (
            ("timeout_seconds", timeout_seconds),
            ("build_timeout_seconds", build_timeout_seconds),
            ("workspace_hash_timeout_seconds", workspace_hash_timeout_seconds),
        ):
            if (
                not isinstance(numeric_value, (int, float))
                or isinstance(numeric_value, bool)
                or not math.isfinite(numeric_value)
                or numeric_value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name, integer_value, minimum in (
            ("max_output_bytes", max_output_bytes, 1024),
            ("workspace_hash_max_entries", workspace_hash_max_entries, 1),
            ("workspace_hash_max_bytes", workspace_hash_max_bytes, 1),
        ):
            if (
                not isinstance(integer_value, int)
                or isinstance(integer_value, bool)
                or integer_value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        reject_runtime_environment_overrides(
            pass_env,
            label="Grok Build source",
            reserved_prefixes=("GROK_",),
        )
        if pass_env and not allow_sensitive_environment:
            raise ValueError(
                "Grok Build and its native subagents can inspect passed environment "
                "values; acknowledge this only for scoped credentials and an outer "
                "sandbox"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_checkout_revision) is None:
            raise ValueError(
                "expected_checkout_revision must be 40 lowercase hex characters"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_source_rev) is None:
            raise ValueError("expected_source_rev must be 40 lowercase hex characters")
        _validate_sha256(expected_cargo_lock_sha256, "expected_cargo_lock_sha256")
        _validate_sha256(expected_executable_sha256, "expected_executable_sha256")
        _validate_sha256(expected_cargo_sha256, "expected_cargo_sha256")
        _validate_sha256(expected_rustc_sha256, "expected_rustc_sha256")
        _validate_sha256(expected_git_sha256, "expected_git_sha256")

        self.cargo_lock = self.checkout / "Cargo.lock"
        self.source_rev_file = self.checkout / "SOURCE_REV"
        self.package_manifest = (
            self.checkout / "crates" / "codegen" / "xai-grok-pager-bin" / "Cargo.toml"
        )
        for path, label in (
            (self.cargo_lock, "Cargo.lock"),
            (self.source_rev_file, "SOURCE_REV"),
            (self.package_manifest, "xai-grok-pager-bin manifest"),
        ):
            if not path.is_file():
                raise ValueError(f"Grok Build checkout is missing {label}: {path}")

        self.model = model
        self.cargo_executable = cargo_executable
        self.rustc_executable = rustc_executable
        self.git_executable = git_executable
        self.sandbox = sandbox
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.timeout_seconds = float(timeout_seconds)
        self.build_timeout_seconds = float(build_timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.allow_sensitive_environment = allow_sensitive_environment
        self.expected_checkout_revision = expected_checkout_revision
        self.expected_source_rev = expected_source_rev
        self.expected_cargo_lock_sha256 = expected_cargo_lock_sha256
        self.expected_executable_sha256 = expected_executable_sha256
        self.expected_cargo_sha256 = expected_cargo_sha256
        self.expected_rustc_sha256 = expected_rustc_sha256
        self.expected_git_sha256 = expected_git_sha256
        self.max_output_bytes = max_output_bytes
        self.workspace_hash_max_entries = workspace_hash_max_entries
        self.workspace_hash_max_bytes = workspace_hash_max_bytes
        self.workspace_hash_timeout_seconds = float(workspace_hash_timeout_seconds)
        self._build_lock = asyncio.Lock()
        self._built = False
        self._cargo_identity: Mapping[str, Any] = {"available": False}
        self._rustc_identity: Mapping[str, Any] = {"available": False}
        self._git_identity: Mapping[str, Any] = {"available": False}
        self._executable_identity: Mapping[str, Any] = {"available": False}
        self._last_source_state: Mapping[str, Any] = {}

        self.checkout_git = self._validate_source()
        self.workspace_tree_sha256 = _workspace_tree_sha256(
            self.workspace,
            max_entries=self.workspace_hash_max_entries,
            max_bytes=self.workspace_hash_max_bytes,
            timeout_seconds=self.workspace_hash_timeout_seconds,
        )
        self.git_workspace = {
            "available": git_pointer.is_dir(),
            "metadata_kind": "standalone-directory" if git_pointer.is_dir() else None,
            "metadata_inherited_into_trial": False,
        }
        self.build_root = Path(
            tempfile.mkdtemp(prefix="scaffoldlab-grok-build-output-")
        ).resolve()
        self.build_target = self.build_root / "target"
        self.build_source_root = self.build_root / "source"
        self.build_home = self.build_root / "home"
        self.build_xdg_config = self.build_root / "xdg-config"
        self.build_cargo_home = self.build_root / "cargo-home"
        self.build_tmp = self.build_root / "tmp"
        for directory in (
            self.build_target,
            self.build_home,
            self.build_xdg_config,
            self.build_cargo_home,
            self.build_tmp,
        ):
            directory.mkdir(mode=0o700)
        self.executable = self.build_target / "release" / "xai-grok-pager"
        self._source_exported = False
        self._source_export_tree_sha256: Optional[str] = None

    def close(self) -> None:
        """Remove the private source export, Cargo state, and build outputs."""

        shutil.rmtree(self.build_root, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validate_source(self, *, runtime_error: bool = False) -> Mapping[str, Any]:
        error_type = ProviderError if runtime_error else ValueError
        try:
            require_standalone_git_checkout(
                self.checkout, label="Grok Build source checkout"
            )
        except ValueError as exc:
            raise error_type(str(exc)) from exc
        git_state = _git_workspace_provenance(self.checkout)
        if git_state.get("available") is not True:
            raise error_type("Grok Build source must be a readable Git checkout")
        if git_state.get("head") != self.expected_checkout_revision:
            raise error_type(
                "Grok Build source revision mismatch: expected "
                f"{self.expected_checkout_revision}, observed {git_state.get('head')}"
            )
        if git_state.get("dirty") is not False:
            raise error_type("Grok Build source checkout must remain clean")
        try:
            source_rev = self.source_rev_file.read_text(encoding="ascii").strip()
            lock_sha256 = _file_sha256(self.cargo_lock)
        except (OSError, UnicodeError) as exc:
            raise error_type("Grok Build source identity files are unreadable") from exc
        if source_rev != self.expected_source_rev:
            raise error_type(
                "Grok Build internal SOURCE_REV mismatch: expected "
                f"{self.expected_source_rev}, observed {source_rev!r}"
            )
        if lock_sha256 != self.expected_cargo_lock_sha256:
            raise error_type(
                "Grok Build Cargo.lock SHA-256 mismatch: expected "
                f"{self.expected_cargo_lock_sha256}, observed {lock_sha256}"
            )
        state = {
            "git": dict(git_state),
            "source_rev": source_rev,
            "cargo_lock_sha256": lock_sha256,
        }
        self._last_source_state = state
        return git_state

    def provenance(self) -> Mapping[str, Any]:
        exact_public_pin = (
            self.expected_checkout_revision == GROK_BUILD_PUBLIC_REVISION
            and self.expected_source_rev == GROK_BUILD_SOURCE_REV
            and self.expected_cargo_lock_sha256 == GROK_BUILD_CARGO_LOCK_SHA256
        )
        return {
            "provider": "grok-build-upstream-source",
            "checkout": str(self.checkout),
            "checkout_git": dict(self.checkout_git),
            "last_verified_source_state": dict(self._last_source_state),
            "expected_checkout_revision": self.expected_checkout_revision,
            "expected_source_rev": self.expected_source_rev,
            "expected_cargo_lock_sha256": self.expected_cargo_lock_sha256,
            "runtime_source_identity_verified": (
                exact_public_pin and self._source_exported
            ),
            "cargo_executable": self.cargo_executable,
            "expected_cargo_sha256": self.expected_cargo_sha256,
            "runtime_cargo": dict(self._cargo_identity),
            "rustc_executable": self.rustc_executable,
            "expected_rustc_sha256": self.expected_rustc_sha256,
            "runtime_rustc": dict(self._rustc_identity),
            "git_executable": self.git_executable,
            "expected_git_sha256": self.expected_git_sha256,
            "runtime_git": dict(self._git_identity),
            "build_command": [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "xai-grok-pager-bin",
            ],
            "built_executable": str(self.executable),
            "cargo_target": "ephemeral-private-per-backend",
            "checkout_cargo_target_used": False,
            "build_source_isolation": "git-archive-of-pinned-commit",
            "build_source_tree_sha256": self._source_export_tree_sha256,
            "build_environment": {
                "home": "ephemeral",
                "xdg_config_home": "ephemeral",
                "cargo_home": "ephemeral",
                "tmpdir": "ephemeral",
                "network_forced_offline": False,
                "cargo_build_scripts_execute": True,
            },
            "expected_executable_sha256": self.expected_executable_sha256,
            "runtime_executable": dict(self._executable_identity),
            "executable_hash_pinned": self.expected_executable_sha256 is not None,
            "workspace": str(self.workspace),
            "base_workspace_tree_sha256": self.workspace_tree_sha256,
            "git_workspace": dict(self.git_workspace),
            "workspace_isolation": "fresh-disposable-copy-per-call",
            "workspace_git_baseline": (
                "fresh standalone repository; inherited Git metadata and history "
                "are stripped"
            ),
            "model": self.model,
            "sandbox": self.sandbox,
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
            "timeout_seconds": self.timeout_seconds,
            "build_timeout_seconds": self.build_timeout_seconds,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "workspace_hash_limits": {
                "max_entries": self.workspace_hash_max_entries,
                "max_content_bytes": self.workspace_hash_max_bytes,
                "timeout_seconds": self.workspace_hash_timeout_seconds,
            },
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": (self.allow_sensitive_environment),
            "fresh_session": True,
            "grok_home": "ephemeral-per-call",
            "headless_output_format": "json",
            "native_subagents_enabled": True,
            "one_backend_call_is_external_session_tree": True,
            "usage_scope": (
                "terminal JSON totals are a lower bound; compaction, side-model, "
                "unfinished, and some nested calls may be absent"
            ),
            "audited_release": {
                "repository": "xai-org/grok-build",
                "version": "1.0.0",
                "revision": GROK_BUILD_PUBLIC_REVISION,
                "source_rev": GROK_BUILD_SOURCE_REV,
                "cargo_lock_sha256": GROK_BUILD_CARGO_LOCK_SHA256,
            },
        }

    async def _run_bounded_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        label: str,
    ) -> tuple[asyncio.subprocess.Process, bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=dict(environment),
                stdin=asyncio.subprocess.DEVNULL,
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
                    input_data=None,
                    max_output_bytes=self.max_output_bytes,
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

    def _reject_external_cargo_configuration(self) -> None:
        """Fail closed on Cargo configuration outside the pinned source export."""

        for ancestor in self.build_source_root.parents:
            for config_name in ("config", "config.toml"):
                candidate = ancestor / ".cargo" / config_name
                if candidate.exists() or candidate.is_symlink():
                    raise ProviderError(
                        "Grok Build ancestor contains external Cargo configuration: "
                        f"{candidate}"
                    )

    async def _ensure_source_export(self) -> None:
        if self._source_exported:
            return
        git, git_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Grok Build source Git",
        )
        if git_identity.get("available") is not True:
            raise ProviderError(
                "Grok Build source Git executable was not found: "
                f"{self.git_executable!r}"
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
                f"--output={archive}",
                self.expected_checkout_revision,
            ],
            cwd=self.build_root,
            environment=environment,
            timeout_seconds=min(self.build_timeout_seconds, 300.0),
            label="Grok Build pinned source export",
        )
        self._validate_source(runtime_error=True)
        if process.returncode != 0 or not archive.is_file() or archive.is_symlink():
            raise ProviderError(
                "Grok Build could not export the pinned source commit",
                raw={
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        try:
            await asyncio.to_thread(
                _extract_verified_git_archive,
                archive,
                self.build_source_root,
                max_entries=self.workspace_hash_max_entries,
                max_bytes=self.workspace_hash_max_bytes,
            )
        finally:
            archive.unlink(missing_ok=True)

        exported_lock = self.build_source_root / "Cargo.lock"
        exported_source_rev = self.build_source_root / "SOURCE_REV"
        exported_manifest = (
            self.build_source_root
            / "crates"
            / "codegen"
            / "xai-grok-pager-bin"
            / "Cargo.toml"
        )
        if (
            not exported_lock.is_file()
            or _file_sha256(exported_lock) != self.expected_cargo_lock_sha256
        ):
            raise ProviderError("Grok Build exported Cargo.lock identity mismatch")
        try:
            exported_revision = exported_source_rev.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise ProviderError("Grok Build exported SOURCE_REV is unreadable") from exc
        if exported_revision != self.expected_source_rev:
            raise ProviderError("Grok Build exported SOURCE_REV identity mismatch")
        if not exported_manifest.is_file():
            raise ProviderError("Grok Build exported package manifest is missing")

        self._reject_external_cargo_configuration()
        self._source_export_tree_sha256 = await self._workspace_hash(
            self.build_source_root, "the exported Grok Build source tree"
        )
        self._source_exported = True

    async def _ensure_built(self) -> None:
        async with self._build_lock:
            self._validate_source(runtime_error=True)
            if self._built:
                observed = (
                    _file_sha256(self.executable) if self.executable.is_file() else None
                )
                if self.executable.is_symlink():
                    observed = None
                recorded = self._executable_identity.get("sha256")
                if observed != recorded:
                    raise ProviderError(
                        "Grok Build source executable changed after it was built"
                    )
                return

            await self._ensure_source_export()
            cargo, cargo_identity = _prepare_executable(
                self.cargo_executable,
                self.expected_cargo_sha256,
                "Grok Build cargo",
            )
            self._cargo_identity = cargo_identity
            cargo_invoked = cargo_identity.get("invoked_path")
            if _is_rustup_proxy(cargo_identity):
                raise ProviderError(
                    "Grok Build cannot isolate a rustup cargo proxy safely; pass the "
                    "concrete toolchain Cargo path returned by `rustup which cargo`"
                )
            if isinstance(cargo_invoked, str):
                # Preserve argv[0] semantics for non-rustup multicall wrappers while
                # retaining the resolved-file digest in provenance.
                cargo = cargo_invoked
            rustc: Optional[str] = None
            if cargo_identity.get("available") is True:
                rustc_candidate = self.rustc_executable
                if rustc_candidate is None and isinstance(cargo_invoked, str):
                    sibling = Path(cargo_invoked).parent / "rustc"
                    if sibling.is_file():
                        rustc_candidate = str(sibling)
                if rustc_candidate is None:
                    raise ProviderError(
                        "Grok Build needs a concrete rustc beside Cargo or an explicit "
                        "--grok-source-rustc-executable"
                    )
                rustc, rustc_identity = _prepare_executable(
                    rustc_candidate,
                    self.expected_rustc_sha256,
                    "Grok Build rustc",
                )
                if _is_rustup_proxy(rustc_identity):
                    raise ProviderError(
                        "Grok Build rustc must be a concrete toolchain binary, not a "
                        "rustup proxy"
                    )
                rustc_invoked = rustc_identity.get("invoked_path")
                if isinstance(rustc_invoked, str):
                    rustc = rustc_invoked
                self._rustc_identity = rustc_identity
            command = [
                cargo,
                "build",
                "--locked",
                "--release",
                "-p",
                "xai-grok-pager-bin",
            ]
            self._reject_external_cargo_configuration()
            build_environment = _process_environment(())
            build_environment.update(
                {
                    "HOME": str(self.build_home),
                    "XDG_CONFIG_HOME": str(self.build_xdg_config),
                    "CARGO_HOME": str(self.build_cargo_home),
                    "CARGO_TARGET_DIR": str(self.build_target),
                    "TMPDIR": str(self.build_tmp),
                    "CARGO_TERM_COLOR": "never",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                }
            )
            if rustc is not None:
                build_environment["RUSTC"] = rustc
            process, stdout, stderr = await self._run_bounded_process(
                command,
                cwd=self.build_source_root,
                environment=build_environment,
                timeout_seconds=self.build_timeout_seconds,
                label="Grok Build source build",
            )
            self._validate_source(runtime_error=True)
            if process.returncode != 0:
                raise ProviderError(
                    f"Grok Build source build exited with status {process.returncode}",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )
            post_build_source_hash = await self._workspace_hash(
                self.build_source_root, "the exported Grok Build source after build"
            )
            if post_build_source_hash != self._source_export_tree_sha256:
                raise ProviderError("Grok Build build mutated its pinned source export")
            if not self.executable.is_file() or self.executable.is_symlink():
                raise ProviderError(
                    "Grok Build source build did not produce a regular private "
                    "target/release/xai-grok-pager executable"
                )
            for parent in self.executable.parents:
                if parent == self.build_root:
                    break
                if parent.is_symlink():
                    raise ProviderError(
                        "Grok Build private Cargo target contains a symlinked parent"
                    )
            if not self.executable.resolve().is_relative_to(self.build_root):
                raise ProviderError(
                    "Grok Build private executable escaped its build directory"
                )
            mode = self.executable.stat().st_mode
            if not mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                raise ProviderError("Grok Build source executable is not executable")
            executable_sha256 = _file_sha256(self.executable)
            if (
                self.expected_executable_sha256 is not None
                and executable_sha256 != self.expected_executable_sha256
            ):
                raise ProviderError(
                    "Grok Build source executable SHA-256 mismatch",
                    raw={"observed_sha256": executable_sha256},
                )
            self._executable_identity = {
                "available": True,
                "path": str(self.executable),
                "sha256": executable_sha256,
            }
            self._built = True

    async def prepare_for_manifest(self) -> None:
        """Resolve the lazy source, toolchain, and executable identity up front."""

        await self._ensure_built()

    async def _initialize_disposable_git(
        self, workspace: Path, environment: Mapping[str, str]
    ) -> str:
        git, git_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Grok Build trial Git",
        )
        if git_identity.get("available") is not True:
            raise ProviderError(
                f"Grok Build trial Git executable was not found: {self.git_executable!r}"
            )
        if self._git_identity.get("available") is True and (
            git_identity.get("resolved_path") != self._git_identity.get("resolved_path")
            or git_identity.get("sha256") != self._git_identity.get("sha256")
        ):
            raise ProviderError("Grok Build trial Git executable identity changed")
        self._git_identity = git_identity
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
                label=f"Grok Build disposable Git baseline step {index}",
            )
            if process.returncode != 0:
                raise ProviderError(
                    "Grok Build could not create its standalone disposable Git "
                    f"baseline at step {index}",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )
        process, stdout, stderr = await self._run_bounded_process(
            (git, "rev-parse", "--verify", "HEAD^{commit}"),
            cwd=workspace,
            environment=environment,
            timeout_seconds=min(self.timeout_seconds, 60.0),
            label="Grok Build disposable Git baseline identity",
        )
        baseline_revision = stdout.decode("ascii", errors="replace").strip()
        if (
            process.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", baseline_revision) is None
        ):
            raise ProviderError(
                "Grok Build could not resolve its disposable Git baseline",
                raw={
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        return baseline_revision

    async def _capture_swe_patch(
        self,
        workspace: Path,
        environment: Mapping[str, str],
        baseline_revision: str,
        usage: Usage,
    ) -> SWEPatchPayload:
        """Capture every trial change relative to the disposable seed baseline."""

        _resolved_git, current_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Grok Build patch Git",
        )
        if current_identity.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or current_identity.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError(
                "Grok Build patch Git executable identity changed", usage=usage
            )
        git = current_identity.get("invoked_path")
        if not isinstance(git, str):
            raise ProviderError(
                "Grok Build patch Git identity is unavailable", usage=usage
            )
        try:
            (
                intent_process,
                _intent_stdout,
                intent_stderr,
            ) = await self._run_bounded_process(
                (
                    git,
                    "add",
                    "--intent-to-add",
                    "--force",
                    "--all",
                    "--",
                    ".",
                ),
                cwd=workspace,
                environment=environment,
                timeout_seconds=min(self.timeout_seconds, 120.0),
                label="Grok Build SWE patch intent-to-add",
            )
            if intent_process.returncode != 0:
                raise ProviderError(
                    "Grok Build could not prepare untracked SWE files for export",
                    usage=usage,
                    raw={"stderr": intent_stderr.decode("utf-8", errors="replace")},
                )
            diff_process, patch, diff_stderr = await self._run_bounded_process(
                (
                    git,
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    baseline_revision,
                    "--",
                    ".",
                ),
                cwd=workspace,
                environment=environment,
                timeout_seconds=min(self.timeout_seconds, 120.0),
                label="Grok Build SWE patch",
            )
            if diff_process.returncode != 0:
                raise ProviderError(
                    "Grok Build could not capture its SWE patch",
                    usage=usage,
                    raw={"stderr": diff_stderr.decode("utf-8", errors="replace")},
                )
        except ProviderError as exc:
            exc.usage = usage
            raise
        _resolved_git, final_identity = _prepare_executable(
            self.git_executable,
            self.expected_git_sha256,
            "Grok Build patch Git",
        )
        if final_identity.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or final_identity.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError(
                "Grok Build patch Git executable changed during export", usage=usage
            )
        return SWEPatchPayload(patch)

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
            raise ProviderError(f"could not hash {label}: {exc}") from exc

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.tools or request.tool_results or request.continuation is not None:
            raise ProviderError(
                "Grok Build source owns tools and continuation state; client tool "
                "continuations are unsupported"
            )
        await self._ensure_built()
        self._validate_source(runtime_error=True)
        pre_base_hash = await self._workspace_hash(
            self.workspace, "the Grok Build base workspace"
        )
        if pre_base_hash != self.workspace_tree_sha256:
            raise ProviderError(
                "Grok Build base workspace changed after backend initialization"
            )
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"

        with tempfile.TemporaryDirectory(prefix="scaffoldlab-grok-source-") as temp_dir:
            temp_root = Path(temp_dir)
            trial_workspace = temp_root / "workspace"
            await asyncio.to_thread(
                shutil.copytree,
                self.workspace,
                trial_workspace,
                symlinks=True,
                ignore=_copytree_ignore_git_metadata,
            )
            # A source-root absolute symlink can legitimately resolve inside the
            # seed tree yet still point back to that seed after copytree preserves
            # it. Hash the copied tree before execution: relative internal links
            # retain the same digest, while absolute/back-reference links now escape
            # the disposable root and fail closed before the agent can mutate them.
            trial_seed_hash = await self._workspace_hash(
                trial_workspace,
                "the disposable Grok Build workspace before the session",
            )
            if trial_seed_hash != self.workspace_tree_sha256:
                raise ProviderError(
                    "Grok Build disposable workspace copy does not match its seed"
                )
            prompt_path = temp_root / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            prompt_path.chmod(0o600)
            grok_home = temp_root / "grok-home"
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            for directory in (grok_home, home, xdg_config):
                directory.mkdir(mode=0o700)

            git_environment = _process_environment(())
            git_environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_NO_REPLACE_OBJECTS": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_OPTIONAL_LOCKS": "0",
                }
            )
            baseline_revision = await self._initialize_disposable_git(
                trial_workspace, git_environment
            )

            command = [
                str(self.executable),
                "--cwd",
                str(trial_workspace),
                "--no-auto-update",
                "--output-format",
                "json",
                "--prompt-file",
                str(prompt_path),
                "--sandbox",
                self.sandbox,
                "--permission-mode",
                self.permission_mode,
                "--max-turns",
                str(self.max_turns),
                "--model",
                self.model,
            ]
            environment = _process_environment(self.pass_env)
            environment.update(
                {
                    "GROK_HOME": str(grok_home),
                    "GROK_DISABLE_AUTOUPDATER": "1",
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                }
            )
            process: Optional[asyncio.subprocess.Process] = None
            usage = Usage(cost_known=False, complete=False)
            try:
                process, stdout, stderr = await self._run_bounded_process(
                    command,
                    cwd=trial_workspace,
                    environment=environment,
                    timeout_seconds=self.timeout_seconds,
                    label="Grok Build source session",
                )
                decoded = stdout.decode("utf-8", errors="replace").strip()
                try:
                    result = json.loads(decoded) if decoded else None
                except json.JSONDecodeError as exc:
                    raise ProviderError(
                        "Grok Build source emitted invalid JSON",
                        usage=usage,
                        raw={
                            "stdout": decoded,
                            "stderr": stderr.decode("utf-8", errors="replace"),
                        },
                    ) from exc
                if not isinstance(result, dict):
                    raise ProviderError(
                        "Grok Build source emitted a non-object JSON result",
                        usage=usage,
                        raw=result,
                    )
                usage = _mark_whole_tree_unverified(_usage_from_grok_result(result))
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                post_trial_hash = await self._workspace_hash(
                    trial_workspace, "the disposable Grok Build workspace"
                )
                result = {
                    **result,
                    "_scaffoldlab_workspace": {
                        "isolation": "fresh-disposable-copy",
                        "base_cwd": str(self.workspace),
                        "trial_cwd": str(trial_workspace),
                        "base_tree_sha256": self.workspace_tree_sha256,
                        "pre_trial_tree_sha256": trial_seed_hash,
                        "post_trial_tree_sha256": post_trial_hash,
                        "inherited_git_metadata": False,
                        "git_baseline": "fresh-standalone-commit",
                        "git_baseline_revision": baseline_revision,
                    },
                    "_scaffoldlab_source": {
                        "checkout": str(self.checkout),
                        "revision": self.expected_checkout_revision,
                        "source_rev": self.expected_source_rev,
                        "cargo_lock_sha256": self.expected_cargo_lock_sha256,
                        "source_export_tree_sha256": (self._source_export_tree_sha256),
                        "source_archive_verified": (
                            self._source_exported
                            and isinstance(self._source_export_tree_sha256, str)
                        ),
                        "official_public_pin_verified": (
                            self.expected_checkout_revision
                            == GROK_BUILD_PUBLIC_REVISION
                            and self.expected_source_rev == GROK_BUILD_SOURCE_REV
                            and self.expected_cargo_lock_sha256
                            == GROK_BUILD_CARGO_LOCK_SHA256
                        ),
                        "git": dict(self._git_identity),
                        "cargo": dict(self._cargo_identity),
                        "rustc": dict(self._rustc_identity),
                        "executable_sha256": self._executable_identity.get("sha256"),
                        "executable_hash_pin_verified": (
                            self.expected_executable_sha256 is not None
                            and self._executable_identity.get("sha256")
                            == self.expected_executable_sha256
                        ),
                    },
                    "usage_is_incomplete": True,
                    "cost_is_partial": True,
                }
                if process.returncode != 0:
                    raise ProviderError(
                        f"Grok Build source exited with status {process.returncode}",
                        usage=usage,
                        raw={
                            "result": result,
                            "stderr": stderr.decode("utf-8", errors="replace"),
                        },
                    )
                stop_reason = result.get("stopReason")
                answer = result.get("text")
                if stop_reason != "end_turn":
                    raise ProviderError(
                        f"Grok Build source ended with stopReason={stop_reason!r}",
                        usage=usage,
                        raw=result,
                    )
                if not isinstance(answer, str) or not answer:
                    raise ProviderError(
                        "Grok Build source JSON contained no final text",
                        usage=usage,
                        raw=result,
                    )
                patch_payload = await self._capture_swe_patch(
                    trial_workspace,
                    git_environment,
                    baseline_revision,
                    usage,
                )
                result["_scaffoldlab_swe_patch"] = patch_payload
                result["_scaffoldlab_workspace"]["patch_bytes"] = len(
                    patch_payload.content
                )
                return ModelResponse(text=answer, usage=usage, raw=result)
            except asyncio.CancelledError as exc:
                # Cancellation can arrive during hashing or patch capture after the
                # terminal result has exposed billable usage.
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except ProviderError as exc:
                exc.usage = usage
                raise
            finally:
                # Fail closed if either supposedly immutable input was touched. The
                # disposable workspace itself is removed by TemporaryDirectory.
                try:
                    self._validate_source(runtime_error=True)
                    observed_executable_sha256 = (
                        _file_sha256(self.executable)
                        if self.executable.is_file()
                        and not self.executable.is_symlink()
                        else None
                    )
                    if observed_executable_sha256 != self._executable_identity.get(
                        "sha256"
                    ):
                        raise ProviderError(
                            "Grok Build source executable changed during the session",
                            usage=usage,
                        )
                    post_base_hash = await self._workspace_hash(
                        self.workspace,
                        "the Grok Build base workspace after the session",
                    )
                    if post_base_hash != self.workspace_tree_sha256:
                        raise ProviderError(
                            "Grok Build escaped its disposable workspace and changed "
                            "the base workspace",
                            usage=usage,
                        )
                except ProviderError as exc:
                    exc.usage = usage
                    raise
