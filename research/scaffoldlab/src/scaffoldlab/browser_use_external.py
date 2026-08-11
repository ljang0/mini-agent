"""Bounded adapter for the pinned upstream Browser-Use runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from .providers import ProviderError
from .external import _workspace_tree_sha256, _workspace_tree_sha256_async
from .source_integrity import require_standalone_git_checkout
from .environment_policy import reject_runtime_environment_overrides
from .types import ModelRequest, ModelResponse, Usage


BROWSER_USE_RELEASE_VERSION = "0.13.7"
BROWSER_USE_RELEASE_REVISION = "f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc"
BROWSER_USE_RESULT_MARKER = "__SCAFFOLDLAB_BROWSER_USE_RESULT_V1__="
_MAX_SOURCE_ARCHIVE_ENTRIES = 250_000
_MAX_SOURCE_ARCHIVE_BYTES = 512 * 1024 * 1024
_SOURCE_HASH_TIMEOUT_SECONDS = 60.0

BROWSER_USE_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure-openai",
        "browser-use",
        "google",
        "groq",
        "litellm",
        "mistral",
        "oci-raw",
        "ollama",
        "openai",
        "vercel",
    }
)


def _process_environment(pass_env: Sequence[str]) -> dict[str, str]:
    allowed = {
        "LANG",
        "LC_ALL",
        "PATH",
        "TERM",
        *pass_env,
    }
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _git_checkout(root: Path, git_executable: str = "git") -> Mapping[str, Any]:
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
    try:
        head = subprocess.run(
            [git_executable, "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            env=environment,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            [
                git_executable,
                "-C",
                str(root),
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


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a JSON object")
    return copied


def _executable_identity(executable: str) -> Mapping[str, Any]:
    candidate = shutil.which(executable)
    if candidate is None:
        explicit = Path(executable)
        candidate = str(explicit) if explicit.is_file() else None
    if candidate is None:
        return {"available": False}
    invoked = Path(candidate).absolute()
    resolved = invoked.resolve()
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {
            "available": False,
            "invoked_path": str(invoked),
            "resolved_path": str(resolved),
        }
    return {
        "available": True,
        "invoked_path": str(invoked),
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
    }


def _prepare_executable(
    executable: str,
    expected_sha256: Optional[str],
    label: str = "Browser-Use Python",
) -> tuple[str, Mapping[str, Any]]:
    identity = _executable_identity(executable)
    if expected_sha256 is not None and identity.get("sha256") != expected_sha256:
        raise ProviderError(
            f"{label} executable SHA-256 mismatch",
            raw={"executable_identity": dict(identity)},
        )
    resolved = identity.get("resolved_path")
    return (resolved if isinstance(resolved, str) else executable, identity)


def _extract_verified_git_archive(
    archive: Path,
    destination: Path,
    *,
    max_entries: int,
    max_bytes: int,
) -> None:
    """Extract a local Git archive without trusting archive paths or link order."""

    with tarfile.open(archive, mode="r:") as stream:
        members = stream.getmembers()
        if len(members) > max_entries:
            raise ProviderError("Browser-Use Git archive exceeds its entry limit")

        normalized: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        seen_paths: set[str] = set()
        casefolded_paths: set[str] = set()
        symlink_paths: set[PurePosixPath] = set()
        non_directory_paths: set[PurePosixPath] = set()
        total_bytes = 0
        for member in members:
            path = PurePosixPath(member.name)
            normalized_path = PurePosixPath(posixpath.normpath(member.name))
            if (
                path.is_absolute()
                or not path.parts
                or path == PurePosixPath(".")
                or path != normalized_path
                or ".." in path.parts
            ):
                raise ProviderError("Browser-Use Git archive contains an unsafe path")
            path_text = path.as_posix()
            if path_text in seen_paths or path_text.casefold() in casefolded_paths:
                raise ProviderError(
                    "Browser-Use Git archive contains duplicate or case-ambiguous paths"
                )
            seen_paths.add(path_text)
            casefolded_paths.add(path_text.casefold())
            if member.isdir():
                pass
            elif member.isfile():
                total_bytes += member.size
                if total_bytes > max_bytes:
                    raise ProviderError(
                        "Browser-Use Git archive exceeds its expanded-byte limit"
                    )
                non_directory_paths.add(path)
            elif member.issym():
                link = PurePosixPath(member.linkname)
                link_target = PurePosixPath(posixpath.normpath(str(path.parent / link)))
                if link.is_absolute() or (
                    link_target.parts and link_target.parts[0] == ".."
                ):
                    raise ProviderError(
                        "Browser-Use Git archive contains an escaping symbolic link"
                    )
                symlink_paths.add(path)
                non_directory_paths.add(path)
            else:
                raise ProviderError("Browser-Use Git archive contains a special file")
            normalized.append((member, path))

        for _member, path in normalized:
            for index in range(1, len(path.parts)):
                ancestor = PurePosixPath(*path.parts[:index])
                if ancestor in non_directory_paths:
                    raise ProviderError(
                        "Browser-Use Git archive places content below a non-directory"
                    )
        for member, path in normalized:
            if not member.issym():
                continue
            link_target = PurePosixPath(
                posixpath.normpath(str(path.parent / member.linkname))
            )
            if link_target in symlink_paths:
                raise ProviderError(
                    "Browser-Use Git archive contains a symbolic-link chain"
                )

        destination.mkdir(mode=0o700)
        for member, path in sorted(normalized, key=lambda item: len(item[1].parts)):
            if not member.isdir():
                continue
            extracted_path = destination.joinpath(*path.parts)
            extracted_path.mkdir(mode=member.mode & 0o777, parents=True, exist_ok=True)
        for member, path in normalized:
            if not member.isfile():
                continue
            extracted_path = destination.joinpath(*path.parts)
            extracted_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ProviderError("Browser-Use Git archive file is unreadable")
            with source, extracted_path.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted_path.chmod(member.mode & 0o777)
        for member, path in normalized:
            if not member.issym():
                continue
            extracted_path = destination.joinpath(*path.parts)
            extracted_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.symlink(member.linkname, extracted_path)


class _OutputLimitExceeded(RuntimeError):
    pass


async def _read_stream(
    stream: asyncio.StreamReader,
    destination: bytearray,
    *,
    max_bytes: int,
    label: str,
) -> None:
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return
        remaining = max_bytes - len(destination)
        if len(chunk) > remaining:
            if remaining > 0:
                destination.extend(chunk[:remaining])
            raise _OutputLimitExceeded(
                f"Browser-Use process {label} exceeded {max_bytes} bytes"
            )
        destination.extend(chunk)


async def _write_input(
    stream: Optional[asyncio.StreamWriter], input_data: bytes
) -> None:
    if stream is None:
        raise RuntimeError("Browser-Use process has no stdin pipe")
    try:
        stream.write(input_data)
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream.close()
        wait_closed = getattr(stream, "wait_closed", None)
        if callable(wait_closed):
            try:
                await wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass


async def _communicate_limited(
    process: asyncio.subprocess.Process,
    input_data: bytes,
    *,
    max_output_bytes: int,
    stdout: bytearray,
    stderr: bytearray,
) -> tuple[bytes, bytes]:
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    if not isinstance(stdout_stream, asyncio.StreamReader) or not isinstance(
        stderr_stream, asyncio.StreamReader
    ):
        raw_stdout, raw_stderr = await process.communicate(input_data)
        raw_stdout = raw_stdout or b""
        raw_stderr = raw_stderr or b""
        stdout.extend(raw_stdout[:max_output_bytes])
        stderr.extend(raw_stderr[:max_output_bytes])
        if len(raw_stdout) > max_output_bytes or len(raw_stderr) > max_output_bytes:
            raise _OutputLimitExceeded(
                f"Browser-Use process output exceeded {max_output_bytes} bytes"
            )
        return raw_stdout, raw_stderr

    tasks = [
        asyncio.create_task(
            _read_stream(
                stdout_stream,
                stdout,
                max_bytes=max_output_bytes,
                label="stdout",
            )
        ),
        asyncio.create_task(
            _read_stream(
                stderr_stream,
                stderr,
                max_bytes=max_output_bytes,
                label="stderr",
            )
        ),
        asyncio.create_task(_write_input(process.stdin, input_data)),
        asyncio.create_task(process.wait()),
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return bytes(stdout), bytes(stderr)


def _signal_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    _signal_group(process, signal.SIGTERM)
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    # The direct Python process can exit before a Chrome descendant.  Kill the
    # isolated process group even when the direct process already returned.
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


def _marker_result(data: bytes) -> Mapping[str, Any]:
    marker_payloads = [
        line[len(BROWSER_USE_RESULT_MARKER) :]
        for line in data.decode("utf-8", errors="replace").splitlines()
        if line.startswith(BROWSER_USE_RESULT_MARKER)
    ]
    if len(marker_payloads) != 1:
        raise ProviderError(
            "Browser-Use upstream runner did not emit exactly one marker result"
        )
    try:
        result = json.loads(marker_payloads[0])
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "Browser-Use upstream runner emitted invalid marker JSON"
        ) from exc
    if not isinstance(result, Mapping) or result.get("schema_version") != 1:
        raise ProviderError(
            "Browser-Use upstream runner emitted an unsupported result schema"
        )
    return result


def _browser_use_usage(summary: Any) -> tuple[Usage, int]:
    """Project upstream observed invocations as an explicitly incomplete lower bound."""

    if summary is None:
        return Usage(cost_known=False, complete=False), 0
    if not isinstance(summary, Mapping):
        raise ProviderError("Browser-Use usage summary must be an object")

    parsed: dict[str, int] = {}
    for field in (
        "total_prompt_tokens",
        "total_prompt_cached_tokens",
        "total_prompt_cache_creation_tokens",
        "total_completion_tokens",
        "entry_count",
    ):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProviderError(
                f"Browser-Use usage summary emitted invalid {field!r}",
                usage=Usage(
                    input_tokens=parsed.get("total_prompt_tokens", 0),
                    output_tokens=parsed.get("total_completion_tokens", 0),
                    cost_known=False,
                    complete=False,
                ),
            )
        parsed[field] = value

    cache_read = parsed["total_prompt_cached_tokens"]
    cache_write = parsed["total_prompt_cache_creation_tokens"]
    # The pinned Anthropic wrapper records cache creation separately and its
    # aggregate prompt field can be smaller than the two cache classes combined.
    # ``max`` is a conservative lower bound that also satisfies Scaffold Lab's
    # disjoint cache-class accounting invariant without guessing double counts.
    input_tokens = max(parsed["total_prompt_tokens"], cache_read + cache_write)
    raw_cost = summary.get("total_cost")
    if (
        not isinstance(raw_cost, (int, float))
        or isinstance(raw_cost, bool)
        or not math.isfinite(raw_cost)
        or raw_cost < 0
    ):
        raise ProviderError("Browser-Use usage summary emitted invalid total_cost")
    return (
        Usage(
            input_tokens=input_tokens,
            output_tokens=parsed["total_completion_tokens"],
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            cost_usd=float(raw_cost),
            # The pinned runtime records only calls whose response carried usage and
            # has no independent expected-call count.  Pricing was intentionally not
            # fetched, so both token completeness and total cost remain unknown.
            cost_known=False,
            complete=False,
        ),
        parsed["entry_count"],
    )


class BrowserUseUpstreamBackend:
    """Execute Browser-Use 0.13.7 Agent and Browser from a clean source checkout."""

    tool_family = "browser-use-upstream"

    def __init__(
        self,
        *,
        checkout: Path,
        provider: str,
        model: str,
        python_executable: str = "python3",
        git_executable: str = "git",
        llm_kwargs: Optional[Mapping[str, Any]] = None,
        browser_kwargs: Optional[Mapping[str, Any]] = None,
        agent_kwargs: Optional[Mapping[str, Any]] = None,
        max_steps: int = 100,
        process_timeout_seconds: float = 1800.0,
        pass_env: Sequence[str] = (),
        expected_checkout_revision: str = BROWSER_USE_RELEASE_REVISION,
        expected_python_sha256: Optional[str] = None,
        expected_git_sha256: Optional[str] = None,
        allow_sensitive_environment: bool = False,
        max_input_bytes: int = 16 * 1024 * 1024,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if os.name != "posix":
            raise ValueError(
                "Browser-Use upstream adapter currently requires POSIX process groups"
            )
        self.checkout = checkout.resolve()
        package_init = self.checkout / "browser_use" / "__init__.py"
        if not self.checkout.is_dir() or not package_init.is_file():
            raise ValueError(
                "Browser-Use checkout must contain browser_use/__init__.py"
            )
        require_standalone_git_checkout(
            self.checkout, label="Browser-Use source checkout"
        )
        if provider not in BROWSER_USE_PROVIDERS:
            raise ValueError(f"unsupported Browser-Use provider {provider!r}")
        for label, value in (
            ("model", model),
            ("python_executable", python_executable),
            ("git_executable", git_executable),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{label} must be a non-empty string")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            not isinstance(process_timeout_seconds, (int, float))
            or isinstance(process_timeout_seconds, bool)
            or not math.isfinite(process_timeout_seconds)
            or process_timeout_seconds <= 0
        ):
            raise ValueError("process_timeout_seconds must be positive and finite")
        for limit_name, limit_value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
        ):
            if (
                not isinstance(limit_value, int)
                or isinstance(limit_value, bool)
                or limit_value < 1024
            ):
                raise ValueError(f"{limit_name} must be an integer of at least 1024")
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in pass_env
        ):
            raise ValueError("pass_env entries must be non-empty environment names")
        if len(set(pass_env)) != len(pass_env):
            raise ValueError("pass_env entries must be unique")
        reject_runtime_environment_overrides(
            pass_env,
            label="Browser-Use upstream",
            reserved_prefixes=("BROWSER_USE_",),
        )
        if not isinstance(allow_sensitive_environment, bool):
            raise ValueError("allow_sensitive_environment must be a boolean")
        if pass_env and not allow_sensitive_environment:
            raise ValueError(
                "Browser-Use and visited pages can inspect passed environment values; "
                "acknowledge scoped credentials and use an outer sandbox"
            )
        if re.fullmatch(r"[0-9a-f]{40}", expected_checkout_revision) is None:
            raise ValueError(
                "expected_checkout_revision must be 40 lowercase hex chars"
            )
        for digest, label in (
            (expected_python_sha256, "expected_python_sha256"),
            (expected_git_sha256, "expected_git_sha256"),
        ):
            if digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{label} must be 64 lowercase hex chars")

        resolved_llm_kwargs = _json_mapping(llm_kwargs or {}, "llm_kwargs")
        if "model" in resolved_llm_kwargs:
            raise ValueError("set model explicitly instead of llm_kwargs.model")
        resolved_browser_kwargs = _json_mapping(browser_kwargs or {}, "browser_kwargs")
        if resolved_browser_kwargs.get("keep_alive") is True:
            raise ValueError("Browser-Use upstream sessions require keep_alive=false")
        resolved_agent_kwargs = _json_mapping(agent_kwargs or {}, "agent_kwargs")
        adapter_owned = {
            "browser",
            "browser_session",
            "calculate_cost",
            "enable_signal_handler",
            "extend_system_message",
            "llm",
            "override_system_message",
            "task",
            "task_id",
        }
        overlap = adapter_owned & resolved_agent_kwargs.keys()
        if overlap:
            raise ValueError(
                f"agent_kwargs cannot override adapter-owned fields: {sorted(overlap)}"
            )

        resolved_git, git_identity = _prepare_executable(
            git_executable,
            expected_git_sha256,
            "Browser-Use Git",
        )
        if git_identity.get("available") is not True:
            raise ValueError(
                f"Browser-Use Git executable was not found: {git_executable!r}"
            )
        git_checkout = _git_checkout(self.checkout, resolved_git)
        if not git_checkout.get("available"):
            raise ValueError("Browser-Use checkout must be a readable Git checkout")
        if git_checkout.get("head") != expected_checkout_revision:
            raise ValueError(
                "Browser-Use checkout revision mismatch: expected "
                f"{expected_checkout_revision}, observed {git_checkout.get('head')}"
            )
        if git_checkout.get("dirty"):
            raise ValueError("Browser-Use checkout must be clean and immutable")

        runner = Path(__file__).with_name("upstream_browser_use_runner.py").resolve()
        if not runner.is_file():
            raise ValueError(f"Browser-Use upstream runner is missing: {runner}")

        self.provider = provider
        self.model = model
        self.python_executable = python_executable
        self.git_executable = git_executable
        self._resolved_git = resolved_git
        self._git_identity = dict(git_identity)
        self.llm_kwargs = resolved_llm_kwargs
        self.browser_kwargs = resolved_browser_kwargs
        self.agent_kwargs = resolved_agent_kwargs
        self.max_steps = max_steps
        self.process_timeout_seconds = float(process_timeout_seconds)
        self.pass_env = tuple(pass_env)
        self.expected_checkout_revision = expected_checkout_revision
        self.expected_python_sha256 = expected_python_sha256
        self.expected_git_sha256 = expected_git_sha256
        self.allow_sensitive_environment = allow_sensitive_environment
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.git_checkout = git_checkout
        self.runner = runner
        self.runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()
        self._source_directory = tempfile.TemporaryDirectory(
            prefix="scaffoldlab-browser-use-source-"
        )
        self.runtime_checkout = (Path(self._source_directory.name) / "source").resolve()
        self._source_export_lock = asyncio.Lock()
        self._source_exported = False
        self._source_archive_sha256: Optional[str] = None
        self._source_export_tree_sha256: Optional[str] = None
        self._python_identity: Mapping[str, Any] = {"available": False}
        self._observed_python_version: Any = None

    def close(self) -> None:
        """Remove the private source export."""

        self._source_directory.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask active exceptions or interpreter shutdown.
            pass

    @staticmethod
    def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "browser-use-upstream-runtime",
            "checkout": str(self.checkout),
            "git_checkout": dict(self.git_checkout),
            "expected_checkout_revision": self.expected_checkout_revision,
            "runtime_checkout": (
                str(self.runtime_checkout) if self._source_exported else None
            ),
            "runtime_source_identity_verified": self._source_exported,
            "source_or_protocol_pin_verified": self._source_exported,
            "bit_reproducible_runtime_verified": False,
            "runtime_source_matches_audited_release": (
                self.expected_checkout_revision == BROWSER_USE_RELEASE_REVISION
            ),
            "browser_use_llm_provider": self.provider,
            "model": self.model,
            "llm_kwargs_keys": sorted(self.llm_kwargs),
            "llm_kwargs_sha256": self._mapping_fingerprint(self.llm_kwargs),
            "browser_kwargs_keys": sorted(self.browser_kwargs),
            "browser_kwargs_sha256": self._mapping_fingerprint(self.browser_kwargs),
            "agent_kwargs_keys": sorted(self.agent_kwargs),
            "agent_kwargs_sha256": self._mapping_fingerprint(self.agent_kwargs),
            "effective_browser_options": {
                "headless": self.browser_kwargs.get("headless", True),
                "keep_alive": False,
            },
            "effective_agent_overrides": {
                "calculate_cost": False,
                "enable_signal_handler": False,
            },
            "max_steps": self.max_steps,
            "process_timeout_seconds": self.process_timeout_seconds,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes_per_stream": self.max_output_bytes,
            "passed_environment_names": sorted(self.pass_env),
            "sensitive_environment_acknowledged": self.allow_sensitive_environment,
            "python_executable": self.python_executable,
            "expected_python_sha256": self.expected_python_sha256,
            "runtime_python": dict(self._python_identity),
            "git_executable": self.git_executable,
            "expected_git_sha256": self.expected_git_sha256,
            "runtime_git": dict(self._git_identity),
            "source_execution_scope": "private_git_archive_of_expected_revision",
            "source_archive_sha256": self._source_archive_sha256,
            "source_export_tree_sha256": self._source_export_tree_sha256,
            "caller_worktree_executed": False,
            "adversarial_git_executable_identity_pinned": (
                self.expected_git_sha256 is not None
            ),
            "observed_python_version": self._observed_python_version,
            "runner": str(self.runner),
            "runner_sha256": self.runner_sha256,
            "json_stdin_protocol": 1,
            "one_backend_call_is_browser_use_agent_session": True,
            "upstream_agent_and_browser_instantiated": True,
            "flat_parallel_reimplementation": False,
            "scaffoldlab_domain_tools_injected": False,
            "usage_scope": (
                "Browser-Use TokenCost entries whose model responses carried usage; "
                "lower bound"
            ),
            "whole_session_usage_verified": False,
            "flagship_system_card_parity_claimed": False,
            "cost_known": False,
            "audited_release": {
                "package": "browser-use",
                "version": BROWSER_USE_RELEASE_VERSION,
                "repository": "browser-use/browser-use",
                "revision": BROWSER_USE_RELEASE_REVISION,
                "requires_python": ">=3.11,<4.0",
            },
        }

    async def prepare_for_manifest(self) -> None:
        """Export source and resolve identities before fingerprinting the run."""

        await self._ensure_source_export()
        self._verify_checkout()
        _resolved_python, identity = _prepare_executable(
            self.python_executable, self.expected_python_sha256
        )
        if identity.get("available") is not True:
            raise ProviderError(
                "Browser-Use Python executable was not found",
                raw={"runtime_python": dict(identity)},
            )
        self._python_identity = identity

    async def _ensure_source_export(self) -> None:
        async with self._source_export_lock:
            if self._source_exported:
                return
            self._verify_checkout()
            archive = Path(self._source_directory.name) / "source.tar"
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

            def export() -> tuple[bytes, bytes, int]:
                completed = subprocess.run(
                    [
                        self._resolved_git,
                        "-C",
                        str(self.checkout),
                        "archive",
                        "--format=tar",
                        f"--output={archive}",
                        self.expected_checkout_revision,
                    ],
                    check=False,
                    capture_output=True,
                    env=environment,
                    timeout=300,
                )
                return completed.stdout, completed.stderr, completed.returncode

            try:
                stdout, stderr, returncode = await asyncio.to_thread(export)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                raise ProviderError(
                    "Browser-Use could not export the pinned source commit"
                ) from exc
            self._verify_checkout()
            if (
                returncode != 0
                or not archive.is_file()
                or archive.is_symlink()
                or archive.stat().st_size > _MAX_SOURCE_ARCHIVE_BYTES
            ):
                raise ProviderError(
                    "Browser-Use could not export the pinned source commit",
                    raw={
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                    },
                )
            self._source_archive_sha256 = hashlib.sha256(
                archive.read_bytes()
            ).hexdigest()
            try:
                await asyncio.to_thread(
                    _extract_verified_git_archive,
                    archive,
                    self.runtime_checkout,
                    max_entries=_MAX_SOURCE_ARCHIVE_ENTRIES,
                    max_bytes=_MAX_SOURCE_ARCHIVE_BYTES,
                )
            finally:
                archive.unlink(missing_ok=True)
            package_init = self.runtime_checkout / "browser_use" / "__init__.py"
            if not package_init.is_file() or package_init.is_symlink():
                raise ProviderError(
                    "Browser-Use private source export is missing browser_use/__init__.py"
                )
            self._source_export_tree_sha256 = await _workspace_tree_sha256_async(
                self.runtime_checkout,
                max_entries=_MAX_SOURCE_ARCHIVE_ENTRIES,
                max_bytes=_MAX_SOURCE_ARCHIVE_BYTES,
                timeout_seconds=_SOURCE_HASH_TIMEOUT_SECONDS,
            )
            self._source_exported = True

    def _verify_checkout(self) -> None:
        observed = _git_checkout(self.checkout, self._resolved_git)
        if observed.get("head") != self.expected_checkout_revision:
            raise ProviderError(
                "Browser-Use checkout changed revision after backend initialization"
            )
        if observed.get("dirty"):
            raise ProviderError(
                "Browser-Use checkout became dirty before or during the trial"
            )

    def _verify_runtime_identity(self, usage: Optional[Usage] = None) -> None:
        try:
            self._verify_checkout()
        except ProviderError as exc:
            raise ProviderError(
                str(exc),
                usage=usage or Usage(cost_known=False, complete=False),
            ) from exc
        try:
            runner_sha256 = hashlib.sha256(self.runner.read_bytes()).hexdigest()
        except OSError as exc:
            raise ProviderError(
                "Browser-Use adapter runner became unreadable",
                usage=usage or Usage(cost_known=False, complete=False),
            ) from exc
        if runner_sha256 != self.runner_sha256:
            raise ProviderError(
                "Browser-Use adapter runner changed during the trial",
                usage=usage or Usage(cost_known=False, complete=False),
            )
        current_git = _executable_identity(self._resolved_git)
        if current_git.get("resolved_path") != self._git_identity.get(
            "resolved_path"
        ) or current_git.get("sha256") != self._git_identity.get("sha256"):
            raise ProviderError(
                "Browser-Use Git executable changed during the trial",
                usage=usage or Usage(cost_known=False, complete=False),
                raw={"runtime_git": dict(current_git)},
            )
        if not self._source_exported or self._source_export_tree_sha256 is None:
            raise ProviderError(
                "Browser-Use private source export is unavailable",
                usage=usage or Usage(cost_known=False, complete=False),
            )
        try:
            current_source_tree = _workspace_tree_sha256(
                self.runtime_checkout,
                max_entries=_MAX_SOURCE_ARCHIVE_ENTRIES,
                max_bytes=_MAX_SOURCE_ARCHIVE_BYTES,
                timeout_seconds=_SOURCE_HASH_TIMEOUT_SECONDS,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProviderError(
                "Browser-Use private source export became unreadable",
                usage=usage or Usage(cost_known=False, complete=False),
            ) from exc
        if current_source_tree != self._source_export_tree_sha256:
            raise ProviderError(
                "Browser-Use private source export changed during the trial",
                usage=usage or Usage(cost_known=False, complete=False),
            )
        expected_python_sha256 = self._python_identity.get("sha256")
        if expected_python_sha256 is not None:
            current = _executable_identity(self.python_executable)
            if (
                current.get("resolved_path")
                != self._python_identity.get("resolved_path")
                or current.get("sha256") != expected_python_sha256
            ):
                raise ProviderError(
                    "Browser-Use Python executable changed during the trial",
                    usage=usage or Usage(cost_known=False, complete=False),
                    raw={"runtime_python": dict(current)},
                )

    def _payload(self, request: ModelRequest) -> bytes:
        if request.tools or request.tool_results or request.continuation is not None:
            raise ProviderError(
                "Browser-Use upstream owns its Browser tools and continuation; "
                "client tool continuation is unsupported"
            )
        if request.max_output_tokens is not None:
            raise ProviderError(
                "Browser-Use upstream has no request-level max_output_tokens; "
                "configure the pinned LLM through llm_kwargs"
            )
        if not isinstance(request.prompt, str) or not request.prompt:
            raise ProviderError("Browser-Use upstream task prompt must be non-empty")
        if not isinstance(request.system, str):
            raise ProviderError(
                "Browser-Use upstream system extension must be a string"
            )
        task_id = request.metadata.get("task_id")
        if task_id is not None and (
            not isinstance(task_id, str) or not task_id or "\x00" in task_id
        ):
            raise ProviderError("Browser-Use upstream task_id must be a string")
        payload = {
            "schema_version": 1,
            "checkout": str(self.runtime_checkout),
            "provider": self.provider,
            "model": self.model,
            "llm_kwargs": self.llm_kwargs,
            "browser_kwargs": self.browser_kwargs,
            "agent_kwargs": self.agent_kwargs,
            "task": request.prompt,
            "task_id": task_id,
            "system_extension": request.system or None,
            "max_steps": self.max_steps,
        }
        try:
            encoded = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "Browser-Use upstream request is not JSON serializable"
            ) from exc
        if len(encoded) > self.max_input_bytes:
            raise ProviderError(
                f"Browser-Use upstream JSON input exceeds {self.max_input_bytes} bytes"
            )
        return encoded

    async def complete(self, request: ModelRequest) -> ModelResponse:
        await self._ensure_source_export()
        self._verify_runtime_identity()
        try:
            response = await self._complete_once(request)
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

    async def _complete_once(self, request: ModelRequest) -> ModelResponse:
        self._verify_checkout()
        input_data = self._payload(request)
        resolved_python, identity = _prepare_executable(
            self.python_executable, self.expected_python_sha256
        )
        self._python_identity = identity

        with tempfile.TemporaryDirectory(
            prefix="scaffoldlab-browser-use-upstream-"
        ) as temp_dir:
            temp_root = Path(temp_dir)
            home = temp_root / "home"
            xdg_config = temp_root / "xdg-config"
            xdg_cache = temp_root / "xdg-cache"
            process_tmp = temp_root / "tmp"
            browser_config = xdg_config / "browser-use"
            for directory in (
                home,
                xdg_config,
                xdg_cache,
                process_tmp,
                browser_config,
            ):
                directory.mkdir(mode=0o700)
            environment = _process_environment(self.pass_env)
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(xdg_config),
                    "XDG_CACHE_HOME": str(xdg_cache),
                    "TMPDIR": str(process_tmp),
                    "BROWSER_USE_CONFIG_DIR": str(browser_config),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "BROWSER_USE_SETUP_LOGGING": "false",
                    "ANONYMIZED_TELEMETRY": "false",
                    "BROWSER_USE_CLOUD_SYNC": "false",
                    "BROWSER_USE_VERSION_CHECK": "false",
                    "BROWSER_USE_CALCULATE_COST": "false",
                }
            )
            command = [resolved_python, "-I", "-B", str(self.runner)]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(temp_root),
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"Browser-Use Python executable not found: "
                    f"{self.python_executable!r}"
                ) from exc

            captured_stdout = bytearray()
            captured_stderr = bytearray()
            try:
                stdout, stderr = await asyncio.wait_for(
                    _communicate_limited(
                        process,
                        input_data,
                        max_output_bytes=self.max_output_bytes,
                        stdout=captured_stdout,
                        stderr=captured_stderr,
                    ),
                    timeout=self.process_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await _terminate_process_tree(process)
                usage = Usage(cost_known=False, complete=False)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                raise ProviderError(
                    "Browser-Use upstream process timed out",
                    usage=usage,
                    raw={
                        "stdout": bytes(captured_stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured_stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except asyncio.CancelledError as exc:
                await _terminate_process_tree(process)
                usage = Usage(cost_known=False, complete=False)
                if request.usage_reporter is not None:
                    request.usage_reporter(usage)
                exc.usage = usage  # type: ignore[attr-defined]
                raise
            except _OutputLimitExceeded as exc:
                await _terminate_process_tree(process)
                raise ProviderError(
                    "Browser-Use upstream process exceeded its output limit",
                    usage=Usage(cost_known=False, complete=False),
                    raw={
                        "stdout": bytes(captured_stdout).decode(
                            "utf-8", errors="replace"
                        ),
                        "stderr": bytes(captured_stderr).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                ) from exc
            except Exception:
                await _terminate_process_tree(process)
                raise
            await _terminate_process_tree(process)

        raw: dict[str, Any] = {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        try:
            result = _marker_result(stdout)
        except ProviderError as exc:
            raise ProviderError(
                str(exc),
                usage=Usage(cost_known=False, complete=False),
                raw=raw,
            ) from exc
        raw["result"] = dict(result)
        try:
            usage, underlying_calls = _browser_use_usage(result.get("usage_summary"))
        except ProviderError as exc:
            raise ProviderError(
                str(exc),
                usage=exc.usage or Usage(cost_known=False, complete=False),
                raw=raw,
            ) from exc
        raw.update(
            {
                "underlying_model_calls": underlying_calls,
                "underlying_model_calls_observed": (
                    result.get("usage_summary") is not None
                ),
                "usage_is_lower_bound": True,
                "whole_session_usage_verified": False,
            }
        )
        python_version = result.get("python_version")
        if isinstance(python_version, list):
            self._observed_python_version = list(python_version)

        try:
            self._verify_checkout()
        except ProviderError as exc:
            raise ProviderError(str(exc), usage=usage, raw=raw) from exc
        if result.get("ok") is not True:
            error = result.get("error")
            error_type = error.get("type") if isinstance(error, Mapping) else None
            suffix = f" ({error_type})" if isinstance(error_type, str) else ""
            raise ProviderError(
                f"Browser-Use upstream runner reported failure{suffix}",
                usage=usage,
                raw=raw,
            )
        if process.returncode != 0:
            raise ProviderError(
                f"Browser-Use upstream process exited with status {process.returncode}",
                usage=usage,
                raw=raw,
            )
        answer = result.get("response")
        if not isinstance(answer, str) or not answer:
            raise ProviderError(
                "Browser-Use upstream result contained no response text",
                usage=usage,
                raw=raw,
            )
        latency = result.get("execution_time_seconds")
        provider_latency = (
            float(latency)
            if isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(latency)
            and latency >= 0
            else 0.0
        )
        return ModelResponse(
            text=answer,
            usage=usage,
            provider_latency_seconds=provider_latency,
            raw=raw,
        )
