"""One bash environment over any :class:`~mini_agent.runtimes.SandboxRuntime`.

Local execution, a rootless Docker container, and an Apptainer fakeroot overlay
differ only in provisioning, one command execution, and file copy. Those live in
``mini_agent.runtimes``; everything the agent and the orchestrator see — the
bash tool, the Git patch baseline, patch/archive export, transactional state
adoption, provenance, and resource identity — lives here, once.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._lifecycle import complete_in_thread, raise_lifecycle_errors
from ..runtimes.base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ProcessResult,
    ProcessRunner,
    SandboxRuntime,
    atomic_write,
    failed,
    positive_int,
    positive_number,
    require_ok,
    require_staging_name,
)
from ..runtimes.local import LocalRuntime, minimal_environment
from ..types import (
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    _require_no_symlink,
    _require_str,
)
from .base import BaseEnvironment


MINI_SWE_AGENT_REVISION = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
DEFAULT_MAX_PATCH_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
ARCHIVE_NAME = "mini-agent-workspace.tar.gz"
DEFAULT_ARCHIVE_PATH = f"/tmp/{ARCHIVE_NAME}"
LOCAL_TOOL_DESCRIPTION = (
    "Run one bash command in the repository workspace. Each call "
    "uses a new shell; filesystem changes persist."
)
_TARGET_PATCH = "mini-agent-target.patch"
_PRIOR_PATCH = "mini-agent-prior.patch"
_TARGET_ARCHIVE = "mini-agent-target.tar.gz"
_PRIOR_ARCHIVE = "mini-agent-prior.tar.gz"


@dataclass(frozen=True)
class SWEPatchState:
    base_identity: str
    patch: bytes

    def __post_init__(self) -> None:
        _require_str(self.base_identity, "SWE patch base identity")
        if not isinstance(self.patch, bytes):
            raise ValueError("SWE patch must be bytes")


@dataclass(frozen=True)
class SWEArchiveState:
    """Whole-workspace state for images without an inspectable Git baseline."""

    base_identity: str
    archive: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.base_identity, str) or not self.base_identity:
            raise ValueError("SWE archive base identity must be non-empty")
        if not isinstance(self.archive, bytes):
            raise ValueError("SWE archive must be bytes")


def patch_destination(destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise ValueError("patch destination must be a Path or None")
    return _require_no_symlink(destination.expanduser(), "patch destination")


class BashEnvironment(BaseEnvironment):
    """One stateless bash action over a persistent sandbox workspace.

    ``base_commit`` selects the export contract: with a Git baseline the
    exported state is a binary patch against it, without one the whole
    workspace tree is exported as a gzip tar archive.
    """

    def __init__(
        self,
        runtime: SandboxRuntime,
        *,
        base_commit: str | None = None,
        base_identity: str | None = None,
        tool_description: str = LOCAL_TOOL_DESCRIPTION,
        provenance_extra: Mapping[str, Any] | None = None,
        patch_export: str = "git_diff_binary",
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        force_add: bool = False,
        clean_flags: str = "-ffd",
        destroy_on_timeout: bool = False,
        archive_path: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.base_commit = base_commit
        self.base_identity = base_identity
        self.tool_description = _require_str(tool_description, "tool description")
        self.patch_export = _require_str(patch_export, "patch export contract")
        self.timeout_seconds = positive_number(timeout_seconds, "timeout_seconds")
        self.max_output_bytes = positive_int(max_output_bytes, "max_output_bytes")
        self.max_patch_bytes = positive_int(max_patch_bytes, "max_patch_bytes")
        self.max_archive_bytes = positive_int(max_archive_bytes, "max_archive_bytes")
        self.force_add = force_add
        self.clean_flags = clean_flags
        self.destroy_on_timeout = destroy_on_timeout
        staging = getattr(runtime, "archive_staging_dir", "/tmp").rstrip("/")
        self.archive_path = archive_path or f"{staging}/{ARCHIVE_NAME}"
        self._provenance_extra = dict(provenance_extra or {})
        self._closed = False

    @classmethod
    def local(
        cls,
        workspace: Path,
        *,
        home: Path | None = None,
        runner: ProcessRunner | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    ) -> "BashEnvironment":
        """Edit ``workspace`` in place; local execution is not a sandbox."""

        return cls(
            LocalRuntime(workspace, home=home, runner=runner),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            max_patch_bytes=max_patch_bytes,
            force_add=True,
            clean_flags="-ffdx",
            provenance_extra=_local_provenance(),
        )

    @classmethod
    async def isolated(
        cls,
        source: Path,
        *,
        scratch_root: Path | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
        runner: ProcessRunner | None = None,
    ) -> "BashEnvironment":
        """Copy ``source`` into a private root and commit a Git baseline."""

        runtime = await LocalRuntime.isolated(
            source, scratch_root=scratch_root, runner=runner
        )
        try:
            base_commit = await create_git_baseline(
                runtime,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
            return cls(
                runtime,
                base_commit=base_commit,
                base_identity="git-commit:" + base_commit,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                max_patch_bytes=max_patch_bytes,
                force_add=True,
                clean_flags="-ffdx",
                provenance_extra=_local_provenance(),
            )
        except BaseException as operation_error:
            cleanup_error: BaseException | None = None
            try:
                await runtime.close()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors(
                "isolated SWE setup", operation_error, cleanup_error
            )
            raise AssertionError("unreachable")

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="bash",
                description=self.tool_description,
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
        )

    async def _exec(
        self, command: str, *, max_output_bytes: int | None = None
    ) -> ProcessResult:
        if self._closed:
            raise RuntimeError("SWE environment is closed")
        return await self.runtime.exec(
            ("/bin/bash", "--noprofile", "--norc", "-c", command),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "bash":
            raise InvalidAction(f"unsupported SWE tool {action.name!r}")
        command = _require_str(
            action.arguments.get("command"), "bash command", error=InvalidAction
        )
        result = await self._exec(command)
        if result.timed_out and self.destroy_on_timeout:
            operation_error = RuntimeError(
                "SWE-bench command timed out; its container was destroyed"
            )
            cleanup_error: BaseException | None = None
            try:
                await self.close()
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors("SWE-bench command", operation_error, cleanup_error)
            raise AssertionError("unreachable")
        output = result.text()
        if result.timed_out:
            output += "\n[command timed out]"
        return ToolExecution(
            output=output,
            is_error=result.returncode != 0 or result.timed_out,
            metadata={
                "exit_code": result.returncode,
                "output_bytes": result.total_output_bytes,
                "output_truncated": result.truncated,
                "timed_out": result.timed_out,
            },
        )

    async def export_patch(self, destination: Path | None = None) -> bytes:
        if self._closed:
            raise RuntimeError("SWE environment is closed")
        if self.base_commit is None:
            raise RuntimeError(
                "this workspace has no Git baseline; export_archive() is the "
                "only workspace export"
            )
        # Official benchmark images can carry ignored build products. Force-adding
        # them would produce a patch before the agent edits anything, so only an
        # environment that created its own baseline stages ignored files.
        force = " --force" if self.force_add else ""
        staged = await self._exec(f"git add --all{force} -- .")
        require_ok(staged, "could not stage workspace changes")
        patch = await self._exec(
            "git diff --cached --binary --full-index --no-ext-diff "
            f"--no-textconv --no-renames {self.base_commit} -- .",
            max_output_bytes=self.max_patch_bytes,
        )
        if patch.timed_out:
            raise RuntimeError("git diff timed out")
        if patch.truncated:
            raise RuntimeError(
                f"SWE patch exceeded the configured {self.max_patch_bytes}-byte limit"
            )
        if patch.returncode != 0:
            raise RuntimeError("could not capture SWE patch: " + patch.text())
        if destination is not None:
            await complete_in_thread(
                atomic_write, patch_destination(destination), patch.output
            )
        return patch.output

    async def export_archive(self, destination: Path | None = None) -> bytes:
        """Export the whole workspace tree as one gzip tar archive."""

        workdir = self.runtime.workdir
        created = await self._exec(
            f"rm -f {self.archive_path} && "
            f"tar -czf {self.archive_path} -C {workdir} . && "
            f"stat -c %s {self.archive_path}"
        )
        require_ok(created, "could not archive the container workspace")
        reported = created.text().strip().rsplit("\n", 1)[-1].strip()
        if not reported.isdigit() or int(reported) > self.max_archive_bytes:
            raise RuntimeError(
                "container workspace archive exceeded the configured "
                f"{self.max_archive_bytes}-byte limit"
            )
        content = await self.runtime.read_file(self.archive_path)
        if len(content) > self.max_archive_bytes:
            raise RuntimeError(
                "container workspace archive exceeded the configured "
                f"{self.max_archive_bytes}-byte limit"
            )
        if destination is not None:
            await complete_in_thread(
                atomic_write, patch_destination(destination), content
            )
        return content

    async def export_state(self) -> SWEPatchState | SWEArchiveState:
        if self.base_identity is None:
            raise RuntimeError("SWE workspace has no adoption baseline")
        if self.base_commit is None:
            return SWEArchiveState(self.base_identity, await self.export_archive())
        return SWEPatchState(self.base_identity, await self.export_patch())

    async def adopt_state(self, state: Any) -> None:
        if self.base_identity is None:
            raise ProtocolError("SWE state adoption requires a workspace baseline")
        if self.base_commit is None:
            await self._adopt_archive_state(state)
            return
        if not isinstance(state, SWEPatchState):
            raise ProtocolError("SWE state has an incompatible type")
        if state.base_identity != self.base_identity:
            raise ProtocolError("SWE state came from a different baseline")
        if len(state.patch) > self.max_patch_bytes:
            raise ProtocolError("SWE state exceeds the configured patch limit")
        previous = await self.export_patch()
        operation_error: BaseException | None = None
        try:
            await self._reset()
            if state.patch:
                applied = await self._apply_patch(_TARGET_PATCH, state.patch)
                require_ok(
                    applied, "SWE state could not be applied", error=ProtocolError
                )
        except BaseException as exc:
            operation_error = exc
        if operation_error is None:
            return
        rollback_error: BaseException | None = None
        try:
            await self._reset()
            if previous:
                restored = await self._apply_patch(_PRIOR_PATCH, previous)
                require_ok(restored, "prior SWE state could not be restored")
        except BaseException as exc:
            rollback_error = exc
        raise_lifecycle_errors("SWE state adoption", operation_error, rollback_error)
        raise AssertionError("unreachable")

    async def _adopt_archive_state(self, state: Any) -> None:
        if not isinstance(state, SWEArchiveState) or (
            state.base_identity != self.base_identity
        ):
            raise ProtocolError("SWE state came from a different baseline")
        if len(state.archive) > self.max_archive_bytes:
            raise ProtocolError("SWE state exceeds the workspace archive limit")
        previous = await self.export_archive()
        operation_error: BaseException | None = None
        try:
            await self._replace_workspace(_TARGET_ARCHIVE, state.archive)
        except BaseException as exc:
            operation_error = exc
        if operation_error is None:
            return
        rollback_error: BaseException | None = None
        try:
            await self._replace_workspace(_PRIOR_ARCHIVE, previous)
        except BaseException as exc:
            rollback_error = exc
        raise_lifecycle_errors("SWE state adoption", operation_error, rollback_error)
        raise AssertionError("unreachable")

    async def _apply_patch(self, name: str, patch: bytes) -> ProcessResult:
        path = await self.runtime.write_file(require_staging_name(name), patch)
        try:
            return await self._exec(
                f"git apply --binary --index {shlex.quote(path)}"
            )
        finally:
            await self.runtime.remove_file(path)

    async def _replace_workspace(self, name: str, archive: bytes) -> None:
        workdir = self.runtime.workdir
        path = await self.runtime.write_file(require_staging_name(name), archive)
        try:
            applied = await self._exec(
                f"find {workdir} -mindepth 1 -delete && "
                f"tar -xzf {shlex.quote(path)} -C {workdir} && "
                f"rm -f {shlex.quote(path)}"
            )
            require_ok(applied, "SWE state could not be applied", error=ProtocolError)
        finally:
            await self.runtime.remove_file(path)

    async def _reset(self) -> None:
        # ``clean_flags`` decides whether ignored files survive: a benchmark
        # image's build products are part of its runtime and are kept, while a
        # self-created baseline owns everything in the copied workspace.
        if self.base_commit is None:
            raise RuntimeError("SWE workspace has no reset baseline")
        result = await self._exec(
            f"git reset --hard {self.base_commit} && "
            f"git clean {self.clean_flags} -q"
        )
        require_ok(result, "could not reset the SWE workspace")

    async def close(self) -> None:
        if self._closed:
            return
        await self.runtime.close()
        self._closed = True

    def provenance(self) -> dict[str, object]:
        return {
            "application": "swe",
            **self._provenance_extra,
            "tools": ["bash"],
            "base_commit": self.base_commit,
            "patch_export": self.patch_export,
            **self.runtime.provenance(),
        }

    def resource_identity(self) -> str:
        return f"swe-{self.runtime.resource_identity()}"


def _local_provenance() -> dict[str, Any]:
    return {
        "design_reference": {
            "project": "mini-swe-agent",
            "revision": MINI_SWE_AGENT_REVISION,
        },
    }


async def verify_git_baseline(
    runtime: SandboxRuntime,
    *,
    expected_base_commit: str | None = None,
    timeout_seconds: float = 60.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> str:
    """Return the image's clean HEAD, optionally proving it contains a commit."""

    async def probe(command: str) -> ProcessResult:
        return await runtime.exec(
            ("/bin/bash", "--noprofile", "--norc", "-c", command),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    baseline = await probe(
        "git rev-parse HEAD && "
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"'
    )
    base_commit = baseline.text().strip().casefold()
    if failed(baseline) or not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise RuntimeError(
            "SWE-bench image has no usable Git baseline: " + baseline.text()
        )
    if expected_base_commit is not None:
        ancestry = await probe(
            f"git merge-base --is-ancestor {expected_base_commit} {base_commit}"
        )
        if failed(ancestry):
            raise RuntimeError("SWE-bench image does not contain task base_commit")
    return base_commit


async def container_bash_environment(
    runtime: SandboxRuntime,
    *,
    require_git_baseline: bool = True,
    expected_base_commit: str | None = None,
    base_identity_prefix: str,
    tool_description: str,
    provenance_extra: Mapping[str, Any],
    destroy_on_timeout: bool = False,
    startup_label: str = "SWE-bench container startup",
    timeout_seconds: float = 60.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> BashEnvironment:
    """Bind a started sandbox to its image baseline, destroying it on failure.

    ``base_identity_prefix`` is the immutable image identity every exported
    state is bound to, so a descendant state can never be adopted by a worker
    running different image bytes.
    """

    try:
        base_commit: str | None = None
        if require_git_baseline:
            base_commit = await verify_git_baseline(
                runtime,
                expected_base_commit=expected_base_commit,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
        elif expected_base_commit is not None:
            raise ValueError(
                "a task base_commit cannot be verified without a Git baseline"
            )
    except BaseException as operation_error:
        cleanup_error: BaseException | None = None
        try:
            await runtime.close()
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors(startup_label, operation_error, cleanup_error)
        raise AssertionError("unreachable")
    suffix = runtime.workdir if base_commit is None else base_commit
    return BashEnvironment(
        runtime,
        base_commit=base_commit,
        base_identity=f"{base_identity_prefix}@{suffix}",
        tool_description=tool_description,
        provenance_extra=provenance_extra,
        patch_export=(
            "git_diff_binary" if base_commit is not None else "workspace_tar_gz"
        ),
        destroy_on_timeout=destroy_on_timeout,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        max_patch_bytes=max_patch_bytes,
        max_archive_bytes=max_archive_bytes,
    )


async def create_git_baseline(
    runtime: LocalRuntime,
    *,
    timeout_seconds: float = 60.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> str:
    """Commit a private, reproducible Git baseline for a copied workspace."""

    identity = {
        **minimal_environment(runtime.home),
        "GIT_AUTHOR_NAME": "mini-agent",
        "GIT_AUTHOR_EMAIL": "mini-agent@invalid",
        "GIT_COMMITTER_NAME": "mini-agent",
        "GIT_COMMITTER_EMAIL": "mini-agent@invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    }
    commands = (
        ("git", "init", "--quiet"),
        ("git", "add", "--all", "--force", "--", "."),
        (
            "git",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "--no-verify",
            "-m",
            "mini-agent temporary baseline",
        ),
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    for index, argv in enumerate(commands):
        result = await runtime.exec(
            argv,
            env=identity,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if result.timed_out or result.returncode != 0:
            raise RuntimeError(
                "could not create temporary Git baseline: " + result.text()
            )
        if index == len(commands) - 1 and result.output:
            raise RuntimeError("temporary Git baseline was not clean after commit")
    revision = await runtime.exec(
        ("git", "rev-parse", "HEAD"),
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    if revision.timed_out or revision.returncode != 0 or not revision.text().strip():
        raise RuntimeError("could not identify temporary Git baseline")
    return revision.text().strip()


__all__ = [
    "DEFAULT_ARCHIVE_PATH",
    "container_bash_environment",
    "verify_git_baseline",
    "DEFAULT_MAX_ARCHIVE_BYTES",
    "DEFAULT_MAX_PATCH_BYTES",
    "LOCAL_TOOL_DESCRIPTION",
    "MINI_SWE_AGENT_REVISION",
    "BashEnvironment",
    "SWEArchiveState",
    "SWEPatchState",
    "create_git_baseline",
    "patch_destination",
]
