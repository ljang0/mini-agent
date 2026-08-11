from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ..types import Task
from .base import EnvironmentFactory, EnvironmentScope, ToolEnvironment
from .browser import BrowserEnvironment, new_playwright_browser_driver
from .composite import CompositeEnvironment
from .computer import ComputerEnvironment, PlaywrightComputerDriver
from .swe import (
    SWEEnvironment,
    _DEFAULT_MAX_PATCH_BYTES,
    _TRUSTED_PROCESS_PATH,
    _minimal_git_environment,
    _terminate_process_group,
)


def _positive_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"environment {key} must be a positive integer")
    return value


def _positive_number(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"environment {key} must be positive")
    return float(value)


def _string_list(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"environment {key} must be a list of non-empty strings")
    return tuple(value)


class ConfiguredEnvironmentFactory(EnvironmentFactory):
    """JSON-configured browser, SWE, computer, or hybrid trial factory."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("environment config must be an object")
        environment_type = config.get("type")
        if environment_type not in {"browser", "swe", "computer", "swe_computer"}:
            raise ValueError(
                "environment type must be browser, swe, computer, or swe_computer"
            )
        isolation = config.get("isolation", "per_agent")
        if isolation not in {"shared", "per_agent"}:
            raise ValueError("environment isolation must be shared or per_agent")
        protocol = config.get("protocol", "auto")
        if protocol not in {"auto", "generic"}:
            raise ValueError("environment protocol must be auto or generic")
        workspace_mode = config.get("workspace_mode", "copy")
        if workspace_mode not in {"copy", "direct"}:
            raise ValueError("environment workspace_mode must be copy or direct")
        allowed_hosts = _string_list(config, "allowed_hosts")
        if (
            environment_type in {"browser", "computer", "swe_computer"}
            and not allowed_hosts
        ):
            raise ValueError("browser/computer environments require allowed_hosts")
        workspace = config.get("workspace")
        if workspace is not None and (not isinstance(workspace, str) or not workspace):
            raise ValueError("environment workspace must be a non-empty string")
        if environment_type in {"swe", "swe_computer"} and workspace is None:
            # A task may still supply metadata.environment.workspace; validate at run.
            pass
        for flag in (
            "allow_write",
            "allow_shell",
            "allow_native_shell",
            "headless",
            "export_patch",
        ):
            value = config.get(flag, True if flag == "headless" else False)
            if not isinstance(value, bool):
                raise ValueError(f"environment {flag} must be a boolean")
        self.config = dict(config)
        self.environment_type = str(environment_type)
        self.isolation = str(isolation)
        self.protocol = str(protocol)
        self.workspace_mode = str(workspace_mode)
        self.workspace = workspace
        self.allowed_hosts = allowed_hosts
        self.allow_write = bool(config.get("allow_write", False))
        self.allow_shell = bool(config.get("allow_shell", False))
        self.allow_native_shell = bool(config.get("allow_native_shell", False))
        self.export_patch = bool(config.get("export_patch", False))
        if self.export_patch and self.environment_type not in {"swe", "swe_computer"}:
            raise ValueError("environment export_patch requires an SWE environment")
        if self.export_patch and self.workspace_mode != "copy":
            raise ValueError(
                "environment export_patch requires workspace_mode='copy' so patch "
                "capture cannot mutate the caller's Git index"
            )
        self.command_allowlist = _string_list(config, "command_allowlist")
        if (
            self.environment_type in {"swe", "swe_computer"}
            and self.workspace_mode == "direct"
            and self.isolation != "shared"
        ):
            raise ValueError(
                "direct SWE workspaces require isolation='shared'; use copy mode "
                "for per-agent checkouts"
            )
        if self.allow_native_shell and not self.allow_shell:
            raise ValueError("environment allow_native_shell requires allow_shell")
        if self.allow_native_shell and self.protocol != "auto":
            raise ValueError("environment allow_native_shell requires protocol='auto'")
        if (
            self.allow_shell
            and not self.allow_native_shell
            and not self.command_allowlist
        ):
            raise ValueError(
                "environment command_allowlist is required when allow_shell is true "
                "and allow_native_shell is false"
            )
        self.headless = bool(config.get("headless", True))
        self.start_url = str(config.get("start_url", "about:blank"))
        self.viewport_width = _positive_int(config, "viewport_width", 1440)
        self.viewport_height = _positive_int(config, "viewport_height", 900)
        self.timeout_seconds = _positive_number(config, "tool_timeout_seconds", 60)
        self.max_output_bytes = _positive_int(
            config, "max_tool_output_bytes", 256 * 1024
        )
        self.max_patch_bytes = _positive_int(
            config, "max_patch_bytes", _DEFAULT_MAX_PATCH_BYTES
        )
        self._patch_export_armed = False
        self._deferred_temporary_roots: list[Path] = []
        self._validate_start_url()

    def _validate_start_url(self) -> None:
        if self.start_url == "about:blank":
            return
        parsed = urlparse(self.start_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("environment start_url must be about:blank or HTTP(S)")
        host = parsed.hostname.casefold()
        allowed = "*" in self.allowed_hosts or any(
            host == candidate.casefold() or host.endswith("." + candidate.casefold())
            for candidate in self.allowed_hosts
        )
        if not allowed:
            raise ValueError("environment start_url host is not allowlisted")

    def _task_workspace(self, task: Task) -> Path:
        raw_environment = task.metadata.get("environment")
        task_workspace: Any = None
        if raw_environment is not None:
            if not isinstance(raw_environment, Mapping):
                raise ValueError("task metadata.environment must be an object")
            task_workspace = raw_environment.get("workspace")
        raw = task_workspace or self.workspace
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                f"task {task.task_id!r} needs metadata.environment.workspace or "
                "the config needs environment.workspace"
            )
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"environment workspace is not a directory: {path}")
        return path

    async def begin(self, task: Task) -> EnvironmentScope:
        source_workspace = (
            self._task_workspace(task)
            if self.environment_type in {"swe", "swe_computer"}
            else None
        )
        return ConfiguredEnvironmentScope(self, task, source_workspace)

    def prepare_trial_patch_export(self) -> None:
        """Defer copy cleanup until the matrix runner externalizes private patches."""

        if not self.export_patch:
            return
        if self._patch_export_armed or self._deferred_temporary_roots:
            raise RuntimeError("a prior SWE patch-export trial is still active")
        self._patch_export_armed = True

    @property
    def patch_export_cleanup_deferred(self) -> bool:
        return self.export_patch and self._patch_export_armed

    def register_deferred_temporary_root(self, root: Path) -> None:
        if not self.patch_export_cleanup_deferred:
            raise RuntimeError("SWE patch-export cleanup is not armed")
        self._deferred_temporary_roots.append(root)

    async def finish_trial_patch_export(self) -> None:
        """Remove every retained copy even when artifact export or grading failed."""

        roots = tuple(self._deferred_temporary_roots)
        self._deferred_temporary_roots.clear()
        self._patch_export_armed = False
        first_error: BaseException | None = None
        for root in roots:
            try:
                await asyncio.to_thread(shutil.rmtree, root)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def validate_artifact_path(self, output_dir: Path, tasks: Sequence[Task]) -> None:
        if self.environment_type not in {"swe", "swe_computer"}:
            return
        resolved_output = output_dir.expanduser().resolve(strict=False)
        for task in tasks:
            source = self._task_workspace(task)
            if (
                resolved_output == source
                or resolved_output.is_relative_to(source)
                or source.is_relative_to(resolved_output)
            ):
                raise ValueError(
                    "run output directory must be disjoint from every SWE source "
                    f"workspace (task {task.task_id!r})"
                )

    def validate_trial_plan(self, trial_count: int) -> None:
        if (
            self.environment_type in {"swe", "swe_computer"}
            and self.workspace_mode == "direct"
            and trial_count != 1
        ):
            raise ValueError(
                "direct SWE workspace mode requires exactly one planned trial; "
                "use copy mode for matrices"
            )

    def provenance(self) -> Mapping[str, Any]:
        redacted = dict(self.config)
        redacted.setdefault("allow_native_shell", self.allow_native_shell)
        redacted.setdefault("export_patch", self.export_patch)
        redacted.setdefault("max_patch_bytes", self.max_patch_bytes)
        if "workspace" in redacted:
            workspace = str(redacted.pop("workspace"))
            redacted["workspace_path_redacted"] = True
            redacted["workspace_path_sha256"] = hashlib.sha256(
                workspace.encode("utf-8")
            ).hexdigest()
        provenance: dict[str, Any] = {
            "factory": "configured",
            "config": redacted,
            "playwright_is_execution_substrate": True,
            "benchmark_environment_claimed": False,
            "subprocess_environment": {
                "inherit_host_environment": False,
                "path": _TRUSTED_PROCESS_PATH,
            },
        }
        if self.allow_native_shell:
            provenance["warnings"] = [
                "provider-native shell bypasses command_allowlist; use an outer "
                "container or VM sandbox"
            ]
        return provenance


class ConfiguredEnvironmentScope(EnvironmentScope):
    def __init__(
        self,
        factory: ConfiguredEnvironmentFactory,
        task: Task,
        source_workspace: Path | None,
    ) -> None:
        self.factory = factory
        self.task = task
        self.source_workspace = source_workspace
        self._environments: dict[str, ToolEnvironment] = {}
        self._temporary_roots: list[Path] = []
        self._defer_temporary_cleanup = factory.patch_export_cleanup_deferred
        self._lock = asyncio.Lock()
        self._closed = False

    def _key(self, agent_id: str) -> str:
        return "/shared" if self.factory.isolation == "shared" else agent_id

    async def _workspace_for(self, key: str) -> Path:
        if self.source_workspace is None:
            raise RuntimeError("SWE environment has no source workspace")
        source_workspace: Path = self.source_workspace
        if self.factory.workspace_mode == "direct":
            if self.factory.isolation != "shared":
                raise ValueError(
                    "direct SWE workspaces require isolation='shared'; use copy mode "
                    "for per-agent checkouts"
                )
            return source_workspace
        temporary_root = Path(tempfile.mkdtemp(prefix="scaffoldlab-swe-trial-"))
        self._temporary_roots.append(temporary_root)
        if self._defer_temporary_cleanup:
            self.factory.register_deferred_temporary_root(temporary_root)
        destination = temporary_root / "workspace"

        def copy_workspace() -> None:
            def ignore_git_metadata(_directory: str, names: list[str]) -> list[str]:
                return [".git"] if ".git" in names else []

            shutil.copytree(
                source_workspace,
                destination,
                symlinks=True,
                ignore_dangling_symlinks=False,
                ignore=ignore_git_metadata,
            )

        copy_task = asyncio.create_task(asyncio.to_thread(copy_workspace))
        try:
            await asyncio.shield(copy_task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(copy_task)
            except Exception:
                pass
            raise
        await self._initialize_git_baseline(destination)
        return destination

    async def _initialize_git_baseline(self, workspace: Path) -> None:
        environment = {
            **_minimal_git_environment(),
            "GIT_AUTHOR_NAME": "Scaffold Lab",
            "GIT_AUTHOR_EMAIL": "scaffoldlab@invalid",
            "GIT_COMMITTER_NAME": "Scaffold Lab",
            "GIT_COMMITTER_EMAIL": "scaffoldlab@invalid",
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
                "Scaffold Lab temporary baseline",
            ),
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        )
        for index, command in enumerate(commands):
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.factory.timeout_seconds
                )
            except BaseException:
                await _terminate_process_group(process)
                raise
            if process.returncode != 0:
                diagnostic = stderr[: self.factory.max_output_bytes].decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(
                    f"could not create temporary Git baseline: {diagnostic}"
                )
            if index == len(commands) - 1 and stdout:
                raise RuntimeError("temporary Git baseline was not clean after commit")

    async def _new_playwright(self) -> Any:
        driver = await new_playwright_browser_driver(
            headless=self.factory.headless,
            viewport_width=self.factory.viewport_width,
            viewport_height=self.factory.viewport_height,
        )
        if self.factory.start_url != "about:blank":
            try:
                await driver.navigate(self.factory.start_url)
            except BaseException:
                try:
                    await driver.close()
                except BaseException:
                    pass
                raise
        return driver

    async def _create(self, key: str) -> ToolEnvironment:
        environment_type = self.factory.environment_type
        if environment_type == "swe":
            workspace = await self._workspace_for(key)
            return SWEEnvironment(
                workspace,
                allow_write=self.factory.allow_write,
                allow_shell=self.factory.allow_shell,
                allow_native_shell=self.factory.allow_native_shell,
                command_allowlist=self.factory.command_allowlist,
                timeout_seconds=self.factory.timeout_seconds,
                max_output_bytes=self.factory.max_output_bytes,
                export_patch=self.factory.export_patch,
                max_patch_bytes=self.factory.max_patch_bytes,
                git_baseline_owned=self.factory.workspace_mode == "copy",
                protocol=self.factory.protocol,
            )
        if environment_type == "browser":
            driver = await self._new_playwright()
            environment = BrowserEnvironment(
                driver,
                allowed_hosts=self.factory.allowed_hosts,
                start_url="about:blank",
                max_output_bytes=self.factory.max_output_bytes,
            )
            return await environment.start()
        if environment_type == "computer":
            driver = await self._new_playwright()
            return ComputerEnvironment(
                PlaywrightComputerDriver(driver), protocol=self.factory.protocol
            )
        if environment_type == "swe_computer":
            workspace = await self._workspace_for(key)
            swe = SWEEnvironment(
                workspace,
                allow_write=self.factory.allow_write,
                allow_shell=self.factory.allow_shell,
                allow_native_shell=self.factory.allow_native_shell,
                command_allowlist=self.factory.command_allowlist,
                timeout_seconds=self.factory.timeout_seconds,
                max_output_bytes=self.factory.max_output_bytes,
                export_patch=self.factory.export_patch,
                max_patch_bytes=self.factory.max_patch_bytes,
                git_baseline_owned=self.factory.workspace_mode == "copy",
                protocol=self.factory.protocol,
            )
            driver = await self._new_playwright()
            computer = ComputerEnvironment(
                PlaywrightComputerDriver(driver), protocol=self.factory.protocol
            )
            return CompositeEnvironment((swe, computer))
        raise AssertionError(f"unknown environment type {environment_type}")

    async def get(self, agent_id: str) -> ToolEnvironment:
        if self._closed:
            raise RuntimeError("environment scope is already closed")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        key = self._key(agent_id)
        async with self._lock:
            existing = self._environments.get(key)
            if existing is not None:
                return existing
            environment = await self._create(key)
            self._environments[key] = environment
            return environment

    async def summary(self) -> Mapping[str, Any]:
        summaries = []
        for key, environment in sorted(self._environments.items()):
            value = dict(await environment.summary())
            value["session_key_sha256"] = hashlib.sha256(
                key.encode("utf-8")
            ).hexdigest()
            summaries.append(value)
        return {
            "type": self.factory.environment_type,
            "isolation": self.factory.isolation,
            "workspace_mode": self.factory.workspace_mode
            if self.source_workspace is not None
            else None,
            "sessions_created": len(self._environments),
            "sessions": summaries,
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for environment in reversed(tuple(self._environments.values())):
            try:
                await environment.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._environments.clear()
        if not self._defer_temporary_cleanup:
            for root in self._temporary_roots:
                try:
                    await asyncio.to_thread(shutil.rmtree, root)
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        self._temporary_roots.clear()
        if first_error is not None:
            raise first_error


def build_environment_factory(
    raw: Any,
) -> ConfiguredEnvironmentFactory | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("config environment must be an object or null")
    # Round-trip once so mutable/non-JSON configuration cannot leak into manifests.
    try:
        copied = json.loads(json.dumps(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("config environment must contain only JSON values") from exc
    return ConfiguredEnvironmentFactory(copied)
