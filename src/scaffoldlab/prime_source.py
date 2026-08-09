"""Source-first adapter for the audited Prime Agent v0.7.1 checkout.

The release repository does not commit ``dist``.  Callers install the exact
``package-lock.json`` and build the checkout before constructing this backend.  The
adapter then executes the generated official bundle while continuously enforcing the
source revision, lockfile, bundle, Node, and npm identities observed at construction.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .external import (
    PrimeAgentJSONBackend,
    _prepare_executable,
    _resolved_executable_identity,
)
from .providers import ProviderError
from .source_integrity import require_standalone_git_checkout
from .types import ModelRequest, ModelResponse, Usage


PRIME_AGENT_SOURCE_VERSION = "0.7.1"
PRIME_AGENT_SOURCE_REVISION = "95afd319a78ae017a41241d50b013d656a0685ce"
PRIME_AGENT_PACKAGE_LOCK_SHA256 = (
    "39ee303bca10c0933cf917275613c8f44099f50de1650a5c356e7cda02b701e8"
)
PRIME_AGENT_BUNDLE = Path("packages/coding-agent/dist/bundle/cli.js")
PRIME_AGENT_SOURCE_ENTRYPOINT = Path("packages/coding-agent/src/main.ts")
PRIME_AGENT_PACKAGE_MANIFEST = Path("packages/coding-agent/package.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_parameter(value: Optional[str], name: str) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters or null")


def _git_checkout(checkout: Path, *, git_executable: str) -> Mapping[str, Any]:
    environment = {
        name: os.environ[name]
        for name in ("LANG", "LC_ALL", "TERM")
        if name in os.environ
    }
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


def _clean_checkout(
    checkout: Path,
    expected_revision: str,
    *,
    git_executable: str,
) -> Mapping[str, Any]:
    require_standalone_git_checkout(checkout, label="Prime Agent source checkout")
    observed = _git_checkout(checkout, git_executable=git_executable)
    if observed.get("available") is not True:
        raise ValueError("Prime Agent source checkout must be a Git working tree")
    if observed.get("head") != expected_revision:
        raise ValueError(
            "Prime Agent source checkout revision mismatch: expected "
            f"{expected_revision!r}, observed {observed.get('head')!r}"
        )
    if observed.get("dirty") is not False:
        raise ValueError("Prime Agent source checkout must be clean")
    return observed


def _manifest_version(path: Path, label: str) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse Prime Agent {label}") from exc
    version = value.get("version") if isinstance(value, dict) else None
    if not isinstance(version, str):
        raise ValueError(f"Prime Agent {label} has no string version")
    return version


class PrimeAgentSourceBackend(PrimeAgentJSONBackend):
    """Execute Prime Agent from a clean, pinned, caller-built source checkout.

    Prime Agent owns the complete coding-agent loop, its IPython tools, child agents,
    prompts, compaction, and stopping behavior.  Scaffold Lab observes that whole tree
    as one outer model call.  The inherited JSON-v3 parser, process-group timeout,
    output ceilings, disposable HOME, workspace hashes, and lower-bound usage policy
    remain the protocol boundary.

    ``npm ci`` and ``npm run build`` are intentionally outside this adapter.  They are
    networked, mutable setup actions and must be completed by the caller in the pinned
    checkout.  Generated ``dist`` and ``node_modules`` paths are ignored by the tagged
    repository, so a correctly built checkout remains Git-clean.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        cwd: Path,
        node_executable: str = "node",
        npm_executable: str = "npm",
        git_executable: str = "git",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        expected_revision: str = PRIME_AGENT_SOURCE_REVISION,
        expected_version: str = PRIME_AGENT_SOURCE_VERSION,
        expected_lock_sha256: str = PRIME_AGENT_PACKAGE_LOCK_SHA256,
        expected_node_sha256: Optional[str] = None,
        expected_npm_sha256: Optional[str] = None,
        expected_git_sha256: Optional[str] = None,
        expected_bundle_sha256: Optional[str] = None,
        timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        max_output_bytes: int = 16 * 1024 * 1024,
        allow_sensitive_environment: bool = False,
        workspace_hash_max_entries: int = 250_000,
        workspace_hash_max_bytes: int = 16 * 1024 * 1024 * 1024,
        workspace_hash_timeout_seconds: float = 30.0,
    ) -> None:
        self.checkout = checkout.resolve()
        resolved_cwd = cwd.resolve()
        if not self.checkout.is_dir():
            raise ValueError(
                f"Prime Agent source checkout is not a directory: {self.checkout}"
            )
        if (
            self.checkout == resolved_cwd
            or self.checkout.is_relative_to(resolved_cwd)
            or resolved_cwd.is_relative_to(self.checkout)
        ):
            raise ValueError(
                "Prime Agent source checkout and single-trial task cwd must be disjoint"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
            raise ValueError("expected_revision must be 40 lowercase hex characters")
        if re.fullmatch(r"\d+\.\d+\.\d+", expected_version) is None:
            raise ValueError("expected_version must be a canonical semantic version")
        for value, name in (
            (expected_lock_sha256, "expected_lock_sha256"),
            (expected_node_sha256, "expected_node_sha256"),
            (expected_npm_sha256, "expected_npm_sha256"),
            (expected_git_sha256, "expected_git_sha256"),
            (expected_bundle_sha256, "expected_bundle_sha256"),
        ):
            _sha256_parameter(value, name)

        self.expected_revision = expected_revision
        self.source_version = expected_version
        self.expected_lock_sha256 = expected_lock_sha256
        self.expected_node_sha256 = expected_node_sha256
        self.expected_npm_sha256 = expected_npm_sha256
        self.expected_git_sha256 = expected_git_sha256
        self.expected_bundle_sha256 = expected_bundle_sha256
        resolved_git, git_identity = _prepare_executable(
            git_executable,
            expected_git_sha256,
            "Prime Agent Git",
        )
        if git_identity.get("available") is not True:
            raise ValueError(
                f"Prime Agent Git executable not found: {git_executable!r}"
            )
        self.git_executable = resolved_git
        self.git_identity = dict(git_identity)
        self.checkout_git = _clean_checkout(
            self.checkout,
            expected_revision,
            git_executable=self.git_executable,
        )

        self.lockfile = self.checkout / "package-lock.json"
        self.root_manifest = self.checkout / "package.json"
        self.package_manifest = self.checkout / PRIME_AGENT_PACKAGE_MANIFEST
        self.source_entrypoint = self.checkout / PRIME_AGENT_SOURCE_ENTRYPOINT
        self.bundle = self.checkout / PRIME_AGENT_BUNDLE
        for path, label in (
            (self.lockfile, "package-lock.json"),
            (self.root_manifest, "root package.json"),
            (self.package_manifest, "coding-agent package.json"),
            (self.source_entrypoint, "coding-agent source entrypoint"),
        ):
            if not path.is_file():
                raise ValueError(f"Prime Agent source checkout is missing {label}")
        if not self.bundle.is_file() or self.bundle.is_symlink():
            raise ValueError(
                "Prime Agent official source bundle is missing; install the pinned "
                "lockfile and build the checkout before constructing the backend: "
                f"{PRIME_AGENT_BUNDLE.as_posix()}"
            )

        lock_sha256 = _file_sha256(self.lockfile)
        if lock_sha256 != expected_lock_sha256:
            raise ValueError(
                "Prime Agent package-lock SHA-256 mismatch: expected "
                f"{expected_lock_sha256}, observed {lock_sha256}"
            )
        for manifest, label in (
            (self.root_manifest, "root package manifest"),
            (self.package_manifest, "coding-agent package manifest"),
        ):
            observed_version = _manifest_version(manifest, label)
            if observed_version != expected_version:
                raise ValueError(
                    f"Prime Agent {label} version mismatch: expected "
                    f"{expected_version!r}, observed {observed_version!r}"
                )

        resolved_node, node_identity = _prepare_executable(
            node_executable,
            expected_node_sha256,
            "Prime Agent Node",
        )
        resolved_npm, npm_identity = _prepare_executable(
            npm_executable,
            expected_npm_sha256,
            "Prime Agent npm",
        )
        if node_identity.get("available") is not True:
            raise ValueError(
                f"Prime Agent Node executable not found: {node_executable!r}"
            )
        if npm_identity.get("available") is not True:
            raise ValueError(
                f"Prime Agent npm executable not found: {npm_executable!r}"
            )

        self.node_executable = resolved_node
        self.npm_executable = resolved_npm
        self.node_identity = dict(node_identity)
        self.npm_identity = dict(npm_identity)
        self.lock_sha256 = lock_sha256
        self.bundle_sha256 = _file_sha256(self.bundle)
        self.source_entrypoint_sha256 = _file_sha256(self.source_entrypoint)
        self.root_manifest_sha256 = _file_sha256(self.root_manifest)
        self.package_manifest_sha256 = _file_sha256(self.package_manifest)
        if (
            expected_bundle_sha256 is not None
            and self.bundle_sha256 != expected_bundle_sha256
        ):
            raise ValueError(
                "Prime Agent built bundle SHA-256 mismatch: expected "
                f"{expected_bundle_sha256}, observed {self.bundle_sha256}"
            )

        self._runtime_directory = tempfile.TemporaryDirectory(
            prefix="scaffoldlab-prime-source-runtime-"
        )
        self.runtime_bundle = (
            Path(self._runtime_directory.name) / PRIME_AGENT_BUNDLE
        ).resolve()
        self.runtime_bundle.parent.mkdir(mode=0o700, parents=True)
        shutil.copyfile(self.bundle, self.runtime_bundle, follow_symlinks=False)
        self.runtime_bundle.chmod(0o600)
        if _file_sha256(self.runtime_bundle) != self.bundle_sha256:
            raise ValueError(
                "Prime Agent private runtime bundle does not match the caller-built "
                "bundle bytes"
            )

        # PrimeAgentJSONBackend accepts one executable path.  This tiny launcher
        # supplies its missing command-prefix seam while delegating every protocol,
        # timeout, streaming, and accounting behavior to the existing backend.
        self._launcher_directory = tempfile.TemporaryDirectory(
            prefix="scaffoldlab-prime-source-launcher-"
        )
        launcher = Path(self._launcher_directory.name) / "prime-agent-source"
        launcher.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(resolved_node)} "
            f'{shlex.quote(str(self.runtime_bundle))} "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o500)
        self.launcher = launcher
        self.launcher_sha256 = _file_sha256(launcher)

        super().__init__(
            cwd=resolved_cwd,
            executable=str(launcher),
            provider=provider,
            model=model,
            no_session=True,
            timeout_seconds=timeout_seconds,
            pass_env=pass_env,
            expected_version=expected_version,
            expected_executable_sha256=self.launcher_sha256,
            max_output_bytes=max_output_bytes,
            allow_sensitive_environment=allow_sensitive_environment,
            workspace_hash_max_entries=workspace_hash_max_entries,
            workspace_hash_max_bytes=workspace_hash_max_bytes,
            workspace_hash_timeout_seconds=workspace_hash_timeout_seconds,
        )

    def close(self) -> None:
        """Remove the private command-prefix launcher."""

        self._launcher_directory.cleanup()
        self._runtime_directory.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask an active exception or interpreter shutdown.
            pass

    def provenance(self) -> Mapping[str, Any]:
        inherited = dict(super().provenance())
        source_release_verified = (
            self.expected_revision == PRIME_AGENT_SOURCE_REVISION
            and self.source_version == PRIME_AGENT_SOURCE_VERSION
            and self.expected_lock_sha256 == PRIME_AGENT_PACKAGE_LOCK_SHA256
        )
        caller_pinned_runtime_identity_complete = all(
            value is not None
            for value in (
                self.expected_node_sha256,
                self.expected_npm_sha256,
                self.expected_bundle_sha256,
            )
        )
        return {
            **inherited,
            "provider": "prime-agent-caller-built-source-bundle",
            "checkout": str(self.checkout),
            "checkout_git": dict(self.checkout_git),
            "expected_checkout_revision": self.expected_revision,
            "source_identity_matches_audited_release": False,
            "checkout_revision_and_lock_match_audited_release": (
                source_release_verified
            ),
            "source_or_protocol_pin_verified": False,
            "bit_reproducible_runtime_verified": False,
            "source_checkout_revision_attestation_scope": (
                "resolved Git executable plus ordinary HEAD/clean-status checks; "
                "not adversarial full-tree content attestation"
            ),
            "adversarial_source_content_attestation_verified": False,
            "git_runtime": {
                **self.git_identity,
                "expected_sha256": self.expected_git_sha256,
                "caller_sha256_pin_verified": self.expected_git_sha256 is not None,
            },
            "source_entrypoint": {
                "path": PRIME_AGENT_SOURCE_ENTRYPOINT.as_posix(),
                "sha256": self.source_entrypoint_sha256,
            },
            "root_package_manifest": {
                "path": "package.json",
                "sha256": self.root_manifest_sha256,
                "version": self.source_version,
            },
            "coding_agent_package_manifest": {
                "path": PRIME_AGENT_PACKAGE_MANIFEST.as_posix(),
                "sha256": self.package_manifest_sha256,
                "version": self.source_version,
            },
            "package_lock": {
                "path": "package-lock.json",
                "sha256": self.lock_sha256,
                "expected_sha256": self.expected_lock_sha256,
                "verified": self.lock_sha256 == self.expected_lock_sha256,
            },
            "caller_built_source_bundle": {
                "path": PRIME_AGENT_BUNDLE.as_posix(),
                "sha256": self.bundle_sha256,
                "expected_sha256": self.expected_bundle_sha256,
                "caller_sha256_pin_verified": (
                    self.expected_bundle_sha256 is not None
                    and self.bundle_sha256 == self.expected_bundle_sha256
                ),
                "authoritative_release_hash_available": False,
                "pinned_unchanged_for_trial": True,
            },
            "private_runtime_bundle": {
                "path": str(self.runtime_bundle),
                "sha256": self.bundle_sha256,
                "copied_from_caller_bundle": True,
                "pinned_unchanged_for_trial": True,
            },
            "node_runtime": {
                **self.node_identity,
                "expected_sha256": self.expected_node_sha256,
                "caller_sha256_pin_verified": self.expected_node_sha256 is not None,
                "authoritative_release_hash_available": False,
                "pinned_unchanged_for_trial": True,
            },
            "npm_build_tool": {
                **self.npm_identity,
                "expected_sha256": self.expected_npm_sha256,
                "caller_sha256_pin_verified": self.expected_npm_sha256 is not None,
                "authoritative_release_hash_available": False,
                "executed_by_adapter": False,
            },
            "source_launcher": {
                "path": str(self.launcher),
                "sha256": self.launcher_sha256,
                "executes": [self.node_executable, str(self.runtime_bundle)],
            },
            "caller_built_source_bundle_entrypoint_executed": True,
            "caller_worktree_bundle_executed_directly": False,
            "source_build_performed_by_adapter": False,
            "dependency_install_performed_by_adapter": False,
            "dependency_tree_content_verified": False,
            "dependency_identity_scope": (
                "exact lockfile plus Node/npm/bundle identities; generated assets "
                "and node_modules are not recursively content-hashed"
            ),
            "pi_skip_version_check": True,
            "runtime_package_identity_matches_audited_release": False,
            "reproducible_build_identity_complete": False,
            "caller_pinned_runtime_identity_complete": (
                caller_pinned_runtime_identity_complete
            ),
            "flagship_system_card_parity_claimed": False,
            "audited_release": {
                "version": PRIME_AGENT_SOURCE_VERSION,
                "repository": "PrimeIntellect-ai/prime-agent",
                "revision": PRIME_AGENT_SOURCE_REVISION,
                "package_lock_sha256": PRIME_AGENT_PACKAGE_LOCK_SHA256,
                "bundle_entrypoint": PRIME_AGENT_BUNDLE.as_posix(),
                "node_engine": ">=22.8.0",
            },
        }

    async def prepare_for_manifest(self) -> None:
        """Version-check the built source entrypoint before run fingerprinting."""

        self._verify_source_identity()
        await self.verify_version()
        self._verify_source_identity()

    def _verify_source_identity(self, usage: Optional[Usage] = None) -> None:
        observed = _git_checkout(
            self.checkout,
            git_executable=self.git_executable,
        )
        failure: Optional[str] = None
        if observed.get("head") != self.expected_revision:
            failure = "checkout revision changed"
        elif observed.get("dirty") is not False:
            failure = "checkout became dirty"
        elif not self.lockfile.is_file() or (
            _file_sha256(self.lockfile) != self.lock_sha256
        ):
            failure = "package-lock changed"
        elif not self.source_entrypoint.is_file() or (
            _file_sha256(self.source_entrypoint) != self.source_entrypoint_sha256
        ):
            failure = "source entrypoint changed"
        elif not self.root_manifest.is_file() or (
            _file_sha256(self.root_manifest) != self.root_manifest_sha256
        ):
            failure = "root package manifest changed"
        elif not self.package_manifest.is_file() or (
            _file_sha256(self.package_manifest) != self.package_manifest_sha256
        ):
            failure = "coding-agent package manifest changed"
        elif not self.bundle.is_file() or (
            _file_sha256(self.bundle) != self.bundle_sha256
        ):
            failure = "built bundle changed"
        elif not self.runtime_bundle.is_file() or (
            _file_sha256(self.runtime_bundle) != self.bundle_sha256
        ):
            failure = "private runtime bundle changed"
        else:
            current_node = _resolved_executable_identity(self.node_executable)
            current_npm = _resolved_executable_identity(self.npm_executable)
            current_git = _resolved_executable_identity(self.git_executable)
            if current_node.get("sha256") != self.node_identity.get("sha256"):
                failure = "Node executable changed"
            elif current_npm.get("sha256") != self.npm_identity.get("sha256"):
                failure = "npm executable changed"
            elif current_git.get("sha256") != self.git_identity.get("sha256"):
                failure = "Git executable changed"
        if failure is not None:
            raise ProviderError(
                f"Prime Agent source identity violation: {failure}",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={
                    "expected_revision": self.expected_revision,
                    "observed_checkout": dict(observed),
                },
            )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.tools or request.tool_results or request.continuation:
            raise ProviderError(
                "Prime Agent source owns tools and continuation state; client tool "
                "continuations are unsupported"
            )
        self._verify_source_identity()
        try:
            response = await super().complete(request)
        except asyncio.CancelledError as exc:
            try:
                self._verify_source_identity(getattr(exc, "usage", None))
            except ProviderError as integrity_error:
                exc.source_integrity_error = str(integrity_error)  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            try:
                self._verify_source_identity(getattr(exc, "usage", None))
            except ProviderError as integrity_error:
                raise integrity_error from exc
            raise
        self._verify_source_identity(response.usage)
        return response

    async def _complete_with_environment(
        self, request: ModelRequest, environment: Mapping[str, str]
    ) -> ModelResponse:
        source_environment = dict(environment)
        source_environment["PI_SKIP_VERSION_CHECK"] = "1"
        return await super()._complete_with_environment(request, source_environment)
