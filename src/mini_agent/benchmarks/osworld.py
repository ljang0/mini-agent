"""OSWorld v1/v2 lifecycle adapter; the hidden evaluator stays outside agents."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, cast

from ..environments.cua import (
    OSWORLD_V1_REVISION,
    OSWORLD_V2_COMMIT,
    OSWORLD_V2_REVISION,
    OSWorldClient,
    OSWorldEnvironment,
    validate_png,
)
from ..environments.base import complete_in_thread
from ..models import Model
from ..orchestrator import Orchestrator
from ..runtime import RunContext
from ..specs import AgentSpecV1
from ..types import BudgetExceeded, BudgetLimits, strict_json_loads
from .base import (
    task_agent_builder,
    BenchmarkTask,
    EvaluationOutcome,
    atomic_bytes,
    atomic_json,
    combine_errors,
    machine_image_identity,
    raise_after_cleanup,
    task_agent_root,
)
from .checkout import (
    git as _git,
    reject_untracked_execution_files,
    task_string,
)


ModelFactory = Callable[[str], Model | Awaitable[Model]]
DesktopFactory = Callable[[str, Path], Any | Awaitable[Any]]

_DOCKER_CLIENT_LOCK = threading.Lock()
_DOCKER_CLIENTS: dict[int, Any] = {}
_ORIGINAL_DOCKER_FROM_ENV: Callable[..., Any] | None = None
_DOCKER_MODULE: Any | None = None


@dataclass(frozen=True)
class OSWorldCheckout:
    path: Path
    version: str
    revision: str
    dirty: bool

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "version": self.version,
            "revision": self.revision,
            "dirty": self.dirty,
        }


def inspect_osworld_checkout(
    checkout: Path,
    *,
    version: str,
    allow_dirty: bool = False,
) -> OSWorldCheckout:
    expanded = checkout.expanduser()
    if expanded.is_symlink():
        raise ValueError("OSWorld checkout must not be a symlink")
    root = expanded.resolve()
    if version not in {"v1", "v2"}:
        raise ValueError("OSWorld version must be v1 or v2")
    if not isinstance(allow_dirty, bool):
        raise ValueError("allow_dirty must be boolean")
    if not (root / "desktop_env" / "desktop_env.py").is_file():
        raise ValueError(f"not an OSWorld checkout: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    expected = OSWORLD_V1_REVISION if version == "v1" else None
    if expected is not None and revision != expected:
        raise ValueError(
            f"OSWorld {version} checkout must be {expected}, found {revision}"
        )
    if version == "v2":
        tag = _git(root, "describe", "--tags", "--exact-match", "HEAD")
        if tag != OSWORLD_V2_REVISION or revision != OSWORLD_V2_COMMIT:
            raise ValueError(
                "OSWorld v2 checkout must be "
                f"{OSWORLD_V2_REVISION} at {OSWORLD_V2_COMMIT}, "
                f"found {tag} at {revision}"
            )
    if dirty and not allow_dirty:
        raise ValueError("OSWorld checkout has tracked modifications")
    reject_untracked_execution_files(
        root,
        label="OSWorld",
        exempt=_v2_untracked_exemption if version == "v2" else None,
        run_git=_git,
    )
    return OSWorldCheckout(root, version, revision, dirty)


def load_osworld(
    checkout: Path,
    *,
    version: str,
    task_list: Path | None = None,
    limit: int | None = None,
    exclude_gitlab: bool = True,
) -> tuple[BenchmarkTask, ...]:
    """Load official task IDs and agent-visible instructions from a checkout."""

    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("OSWorld limit must be positive")
    if not isinstance(exclude_gitlab, bool):
        raise ValueError("exclude_gitlab must be boolean")

    info = inspect_osworld_checkout(checkout, version=version)
    if Path.cwd().resolve() != info.path:
        raise RuntimeError(
            "OSWorld uses checkout-relative task assets; run evaluation with the "
            "checkout as the process working directory"
        )
    default = (
        info.path / "evaluation_examples" / "test_all.json"
        if version == "v1"
        else info.path / "evaluation_examples" / "test_v2.json"
    )
    selected = (task_list or default).expanduser().resolve()
    raw = strict_json_loads(selected.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("OSWorld task list must be an object of domain to task IDs")
    tasks: list[BenchmarkTask] = []
    for domain, task_ids in raw.items():
        if not isinstance(domain, str) or not isinstance(task_ids, list):
            raise ValueError("OSWorld task list has an invalid domain entry")
        _safe_component(domain, "OSWorld domain")
        if exclude_gitlab and domain.casefold() == "gitlab":
            continue
        for task_id in task_ids:
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("OSWorld task IDs must be non-empty strings")
            _safe_component(task_id, "OSWorld task ID")
            class_sha256 = _task_class_sha256(info, domain, task_id)
            config = _load_task_config(info, domain, task_id)
            if _task_class_sha256(info, domain, task_id) != class_sha256:
                raise RuntimeError(
                    f"OSWorld task class changed while loading {task_id!r}"
                )
            instruction = _task_value(config, "instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"OSWorld task {task_id!r} has no instruction")
            if exclude_gitlab and _mentions_gitlab(config):
                continue
            tasks.append(
                BenchmarkTask(
                    task_id,
                    instruction,
                    {
                        "checkout": str(info.path),
                        "version": version,
                        "revision": info.revision,
                        "domain": domain,
                        "task_config_sha256": _config_sha256(config),
                        "task_class_sha256": class_sha256,
                    },
                )
            )
            if limit is not None and len(tasks) == limit:
                return tuple(tasks)
    if not tasks:
        raise ValueError(
            "OSWorld task data is unavailable or every selected task was excluded"
        )
    return tuple(tasks)


class UpstreamDesktopFactory:
    """Construct independent official DesktopEnv instances without model code."""

    def __init__(
        self,
        checkout: Path,
        *,
        version: str,
        provider_name: str = "docker",
        path_to_vm: str | None = None,
        headless: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        enable_proxy: bool = False,
        client_password: str = "",
        apptainer_image: Path | None = None,
        apptainer_executable: str = "apptainer",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("OSWorld provider_name must be non-empty")
        if path_to_vm is not None and not isinstance(path_to_vm, str):
            raise ValueError("OSWorld path_to_vm must be a string or None")
        if not isinstance(headless, bool) or not isinstance(enable_proxy, bool):
            raise ValueError("OSWorld boolean options must be booleans")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (screen_width, screen_height)
        ):
            raise ValueError("OSWorld screen dimensions must be positive integers")
        if not isinstance(client_password, str):
            raise ValueError("OSWorld client_password must be a string")
        normalized_provider = provider_name.casefold().strip()
        if normalized_provider == "docker" and path_to_vm is None:
            raise ValueError("OSWorld Docker provider requires an explicit VM image")
        if apptainer_image is not None and normalized_provider != "docker":
            raise ValueError("OSWorld Apptainer compatibility requires Docker provider")
        if apptainer_image is not None and not isinstance(apptainer_image, Path):
            raise ValueError("OSWorld apptainer_image must be a Path or None")
        if not isinstance(apptainer_executable, str) or not (
            apptainer_executable.strip()
        ):
            raise ValueError("OSWorld apptainer_executable must be non-empty")
        if not callable(sleep):
            raise ValueError("OSWorld sleep must be callable")
        self.checkout = inspect_osworld_checkout(checkout, version=version)
        self.provider_name = provider_name
        self.vm_image = (
            machine_image_identity(Path(path_to_vm), label="OSWorld VM image")
            if normalized_provider == "docker" and path_to_vm is not None
            else None
        )
        self.path_to_vm = (
            str(self.vm_image["path"])
            if self.vm_image is not None
            else path_to_vm
        )
        self.apptainer_image = (
            machine_image_identity(
                apptainer_image, label="OSWorld Apptainer image"
            )
            if apptainer_image is not None
            else None
        )
        self.apptainer_executable = apptainer_executable
        self.headless = headless
        self.screen_size = (screen_width, screen_height)
        self.enable_proxy = enable_proxy
        self.client_password = client_password
        self.sleep = sleep
        # These are the fixed waits in the generic runner at the pinned
        # revisions.  V2 removed the final v1 settle but retained the initial
        # reset settle and fresh observation.
        self.initial_settle_seconds = 60.0
        self.evaluation_settle_seconds = 20.0 if version == "v1" else 0.0
        self.refresh_initial_observation = True

    async def __call__(self, agent_id: str, cache: Path) -> Any:
        del agent_id
        await self.verify_checkout()
        await self._verify_runtime_images()
        desktop = _desktop_env_class(self.checkout)
        keywords: dict[str, Any] = {
            "provider_name": self.provider_name,
            "path_to_vm": self.path_to_vm,
            "action_space": "pyautogui",
            "cache_dir": str(cache),
            "screen_size": self.screen_size,
            "headless": self.headless,
            "require_a11y_tree": False,
            "os_type": "Ubuntu",
            "enable_proxy": self.enable_proxy,
            "client_password": self.client_password,
        }
        signature = inspect.signature(desktop)
        keywords = {
            key: value for key, value in keywords.items() if key in signature.parameters
        }
        if self.apptainer_image is None:
            return await asyncio.to_thread(desktop, **keywords)
        if self.vm_image is None:
            raise AssertionError("OSWorld Apptainer launch has no VM image")
        from ..environments.osworld_apptainer import (
            OSWorldApptainerDockerClient,
        )

        client = OSWorldApptainerDockerClient(
            apptainer_image=Path(self.apptainer_image["path"]),
            vm_image=Path(self.vm_image["path"]),
            work_root=cache / "apptainer-launcher",
            executable=self.apptainer_executable,
        )
        return await asyncio.to_thread(
            _construct_with_docker_client, desktop, keywords, client
        )

    async def _verify_runtime_images(self) -> None:
        """Fail closed if a manifest-bound image changed before launch."""

        for expected, label in (
            (self.vm_image, "OSWorld VM image"),
            (self.apptainer_image, "OSWorld Apptainer image"),
        ):
            if expected is None:
                continue
            observed = await asyncio.to_thread(
                machine_image_identity,
                Path(str(expected["path"])),
                label=label,
            )
            if observed != expected:
                raise RuntimeError(
                    f"{label} changed after its manifest identity was recorded"
                )

    async def verify_checkout(self) -> None:
        """Revalidate the exact executable checkout at a lifecycle boundary."""

        observed = await asyncio.to_thread(
            inspect_osworld_checkout,
            self.checkout.path,
            version=self.checkout.version,
        )
        if observed != self.checkout:
            raise RuntimeError(
                "OSWorld checkout changed after its manifest identity was recorded"
            )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "checkout": self.checkout.as_dict(),
            "provider_name": self.provider_name,
            "screen_size": list(self.screen_size),
            "headless": self.headless,
            "enable_proxy": self.enable_proxy,
            "path_to_vm_configured": self.path_to_vm is not None,
            "vm_image": dict(self.vm_image) if self.vm_image is not None else None,
            "container_runtime": (
                "apptainer" if self.apptainer_image is not None else "docker"
                if self.provider_name.casefold().strip() == "docker"
                else None
            ),
            "apptainer_image": (
                dict(self.apptainer_image)
                if self.apptainer_image is not None
                else None
            ),
            "apptainer_executable": (
                self.apptainer_executable
                if self.apptainer_image is not None
                else None
            ),
            "runtime_adaptation": (
                {
                    "scope": "container-network-display-and-firmware-mode-only",
                    "network": "qemu-user-hostfwd",
                    "display": "disabled",
                    "firmware": (
                        "exact-container-bytes-materialized-0600-for-fakeroot-qemu"
                    ),
                    "official_container_entrypoint": True,
                    "official_guest_and_evaluator": True,
                }
                if self.apptainer_image is not None
                else None
            ),
            "reference_timing": {
                "initial_settle_seconds": self.initial_settle_seconds,
                "fresh_observation_after_initial_settle": (
                    self.refresh_initial_observation
                ),
                "evaluation_settle_seconds": self.evaluation_settle_seconds,
                # Passed to DesktopEnv.step as sleep_after_execution; the
                # adapter re-observes explicitly, so no extra pause is added.
                "step_pause_seconds": 0,
            },
        }


def _construct_with_docker_client(
    desktop: Any, keywords: Mapping[str, Any], client: Any
) -> Any:
    """Route one upstream DockerProvider construction to a local client."""

    global _DOCKER_MODULE, _ORIGINAL_DOCKER_FROM_ENV
    docker_module = importlib.import_module("docker")
    thread_id = threading.get_ident()
    with _DOCKER_CLIENT_LOCK:
        if thread_id in _DOCKER_CLIENTS:
            raise RuntimeError("nested OSWorld Docker client override")
        if not _DOCKER_CLIENTS:
            original = getattr(docker_module, "from_env", None)
            if not callable(original):
                raise RuntimeError("OSWorld Docker SDK has no callable from_env")
            _DOCKER_MODULE = docker_module
            _ORIGINAL_DOCKER_FROM_ENV = original
            setattr(docker_module, "from_env", _docker_from_env)
        elif docker_module is not _DOCKER_MODULE:
            raise RuntimeError("OSWorld Docker SDK module changed during construction")
        _DOCKER_CLIENTS[thread_id] = client
    try:
        return desktop(**dict(keywords))
    finally:
        with _DOCKER_CLIENT_LOCK:
            _DOCKER_CLIENTS.pop(thread_id, None)
            if not _DOCKER_CLIENTS:
                if _DOCKER_MODULE is not None and _ORIGINAL_DOCKER_FROM_ENV is not None:
                    setattr(
                        _DOCKER_MODULE,
                        "from_env",
                        _ORIGINAL_DOCKER_FROM_ENV,
                    )
                _DOCKER_MODULE = None
                _ORIGINAL_DOCKER_FROM_ENV = None


def _docker_from_env(*args: Any, **kwargs: Any) -> Any:
    with _DOCKER_CLIENT_LOCK:
        client = _DOCKER_CLIENTS.get(threading.get_ident())
        original = _ORIGINAL_DOCKER_FROM_ENV
    if client is not None:
        if args or kwargs:
            raise RuntimeError("upstream OSWorld changed its docker.from_env contract")
        return client
    if original is None:
        raise RuntimeError("OSWorld Docker client override is inactive")
    return original(*args, **kwargs)


class _DesktopPool:
    def __init__(
        self,
        *,
        task: BenchmarkTask,
        directory: Path,
        desktop_factory: DesktopFactory,
    ) -> None:
        if not callable(desktop_factory):
            raise ValueError("desktop_factory must be callable")
        self.task = task
        self.directory = directory
        self.desktop_factory = desktop_factory
        self.desktops: list[Any] = []
        self.sleep = getattr(desktop_factory, "sleep", asyncio.sleep)
        if not callable(self.sleep):
            raise ValueError("OSWorld desktop factory sleep must be callable")
        self.initial_settle_seconds = _factory_delay(
            desktop_factory, "initial_settle_seconds"
        )
        self.evaluation_settle_seconds = _factory_delay(
            desktop_factory, "evaluation_settle_seconds"
        )
        refresh = getattr(
            desktop_factory, "refresh_initial_observation", False
        )
        if not isinstance(refresh, bool):
            raise ValueError(
                "OSWorld refresh_initial_observation must be boolean"
            )
        self.refresh_initial_observation = refresh

    async def environment(self, agent_id: str) -> OSWorldEnvironment:
        config = _task_config_for_benchmark(self.task)
        _reject_unsupported_v2_lifecycle(self.task, config)
        branch = (
            self.directory
            / "branches"
            / hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
        )
        branch.mkdir(parents=True, exist_ok=True)
        created = self.desktop_factory(agent_id, branch / "cache")
        if inspect.isawaitable(created):
            creation = asyncio.ensure_future(created)
            desktop: Any = None
            try:
                desktop = await asyncio.shield(creation)
            except BaseException as operation_error:
                creation_cleanup_error: BaseException | None = None
                try:
                    desktop = await creation
                except BaseException as exc:
                    if exc is not operation_error:
                        creation_cleanup_error = exc
                if desktop is not None:
                    try:
                        await _close_desktop(desktop)
                    except BaseException as exc:
                        creation_cleanup_error = combine_errors(
                            creation_cleanup_error, exc
                        )
                raise_after_cleanup(
                    "OSWorld environment creation",
                    operation_error,
                    creation_cleanup_error,
                )
        else:
            desktop = created
        self.desktops.append(desktop)
        try:
            initial = await complete_in_thread(desktop.reset, task_config=config)
            if not isinstance(initial, Mapping):
                raise RuntimeError("OSWorld reset did not return an observation")
            if self.initial_settle_seconds:
                await self.sleep(self.initial_settle_seconds)
            if self.refresh_initial_observation:
                getter = getattr(desktop, "_get_obs", None)
                if not callable(getter):
                    raise RuntimeError(
                        "OSWorld reference timing requires callable _get_obs"
                    )
                initial = await complete_in_thread(getter)
                if not isinstance(initial, Mapping):
                    raise RuntimeError(
                        "OSWorld _get_obs did not return an observation"
                    )
            screenshot = initial.get("screenshot")
            if not isinstance(screenshot, bytes):
                raise RuntimeError("OSWorld reset returned no screenshot")
            validate_png(screenshot)
            await complete_in_thread(atomic_bytes, branch / "step_0000.png", screenshot)
        except BaseException as operation_error:
            self.desktops.remove(desktop)
            reset_cleanup_error: BaseException | None = None
            try:
                await _close_desktop(desktop)
            except BaseException as exc:
                reset_cleanup_error = exc
            raise_after_cleanup(
                "OSWorld environment reset", operation_error, reset_cleanup_error
            )

        trajectory_number = 0

        async def record(transition: Mapping[str, Any]) -> None:
            nonlocal trajectory_number
            trajectory_number += 1
            observation = transition.get("observation")
            if not isinstance(observation, Mapping):
                raise RuntimeError("OSWorld transition has no observation")
            png = observation.get("screenshot")
            upstream_step = transition.get("step")
            if not isinstance(png, bytes) or not isinstance(upstream_step, int):
                raise RuntimeError("OSWorld transition is malformed")
            validate_png(png)
            name = f"step_{trajectory_number:04d}.png"
            await complete_in_thread(atomic_bytes, branch / name, png)
            event = {
                "step": trajectory_number,
                "environment_step": upstream_step,
                "action": transition.get("action"),
                "encoded_action": transition.get("encoded_action"),
                "done": transition.get("done"),
                "screenshot": name,
            }
            line = (
                json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            await complete_in_thread(_append_bytes, branch / "trajectory.jsonl", line)

        client = OSWorldClient(
            desktop,
            initial,
            transition_sink=record,
            owns_environment=False,
            resource_identity=f"osworld:{self.task.task_id}:{hashlib.sha256(agent_id.encode()).hexdigest()}",
            pause_seconds=0,
        )
        return OSWorldEnvironment(client, version=str(self.task.data["version"]))

    async def settle_before_evaluation(self) -> None:
        if self.evaluation_settle_seconds:
            await self.sleep(self.evaluation_settle_seconds)

    async def close(self) -> None:
        errors: list[BaseException] = []
        cancelled = False
        remaining: list[Any] = []
        for desktop in reversed(self.desktops):
            try:
                await _close_desktop(desktop)
            except asyncio.CancelledError:
                cancelled = True
                remaining.append(desktop)
            except Exception as exc:
                errors.append(exc)
                remaining.append(desktop)
        self.desktops = list(reversed(remaining))
        if errors:
            raise RuntimeError(
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]
        if cancelled:
            raise asyncio.CancelledError()


async def run_osworld_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    desktop_factory: DesktopFactory,
    model_factory: ModelFactory,
    system_prompt: str,
    max_steps: int,
    multi_agent: bool = False,
    max_active_agents: int = 4,
    max_total_agents: int = 8,
    per_agent_limits: BudgetLimits | None = None,
    agent_spec: AgentSpecV1 | None = None,
) -> EvaluationOutcome:
    """Run inference, then score only the root-selected live desktop."""

    if not callable(desktop_factory) or not callable(model_factory):
        raise ValueError("desktop_factory and model_factory must be callable")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(multi_agent, bool):
        raise ValueError("multi_agent must be boolean")
    checkout = Path(_task_string(task, "checkout")).resolve()
    version = _task_string(task, "version")
    checkout_info = inspect_osworld_checkout(checkout, version=version)
    if task.data.get("revision") != checkout_info.revision:
        raise RuntimeError("OSWorld checkout changed after manifest creation")
    if Path.cwd().resolve() != checkout:
        raise RuntimeError("OSWorld task must run from its pinned checkout")
    pool = _DesktopPool(
        task=task,
        directory=directory,
        desktop_factory=desktop_factory,
    )
    root_id = task_agent_root(task.task_id)
    root_environment: OSWorldEnvironment | None = None
    answer = ""
    steps = 0
    agent_error: str | None = None
    agent_failure: BaseException | None = None
    agent_statuses: Mapping[str, str] = {}
    state_selection = "root_environment"
    state_adoption_history: tuple[str, ...] = ()

    async def environment_for(agent_id: str) -> OSWorldEnvironment:
        return await pool.environment(agent_id)

    agent_for = task_agent_builder(
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=max_steps,
        agent_spec=agent_spec,
    )

    score: float | None = None
    evaluator_result: Mapping[str, Any] | None = None
    operation_error: BaseException | None = None
    try:
        if multi_agent:
            orchestrator = Orchestrator(
                agent_builder=agent_for,
                environment_factory=environment_for,
                context=context,
                max_active_agents=max_active_agents,
                max_total_agents=max_total_agents,
                per_agent_limits=per_agent_limits,
                root_id=root_id,
            )
            try:
                result = await orchestrator.run(task.prompt)
                answer, steps = result.answer, result.steps
            except asyncio.CancelledError:
                raise
            except BudgetExceeded as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
                agent_failure = exc
            root_record = orchestrator.records.get(root_id)
            if root_record is not None and root_record.environment is not None:
                root_environment = root_record.environment.base
                state_adoption_history = tuple(root_record.adoption_history)
                if state_adoption_history:
                    state_selection = "adopted_descendant_environment"
            agent_statuses = {
                agent_id: record.status
                for agent_id, record in orchestrator.records.items()
            }
        else:
            root_environment = await pool.environment(root_id)
            agent = await agent_for(root_id, root_environment, context)
            inference_error: BaseException | None = None
            try:
                result = await agent.run(task.prompt)
                answer, steps = result.answer, result.steps
            except asyncio.CancelledError as exc:
                inference_error = exc
            except BudgetExceeded as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
                agent_failure = exc
            close_error: BaseException | None = None
            try:
                await root_environment.close()
            except BaseException as exc:
                close_error = exc
            raise_after_cleanup("OSWorld root inference", inference_error, close_error)
            agent_statuses = {root_id: "failed" if agent_error else "completed"}

        if agent_failure is not None:
            raise agent_failure
        if root_environment is None:
            raise RuntimeError("root OSWorld environment was never prepared")
        desktop = cast(OSWorldClient, root_environment.client).environment
        await pool.settle_before_evaluation()
        raw_score = await complete_in_thread(desktop.evaluate)
        score, evaluator_result = _osworld_score(raw_score)
    except BaseException as exc:
        operation_error = exc

    cleanup_error: BaseException | None = None
    try:
        await pool.close()
    except BaseException as exc:
        cleanup_error = exc
    checkout_error: BaseException | None = None
    checkout_hook = getattr(desktop_factory, "verify_checkout", None)
    if checkout_hook is not None:
        try:
            if not callable(checkout_hook):
                raise RuntimeError("OSWorld checkout verifier must be callable")
            checked = checkout_hook()
            if inspect.isawaitable(checked):
                await checked
            elif checked is not None:
                raise RuntimeError("OSWorld checkout verifier must return None")
        except BaseException as exc:
            checkout_error = exc
    if checkout_error is not None:
        cleanup_error = combine_errors(cleanup_error, checkout_error)
    raise_after_cleanup("OSWorld task", operation_error, cleanup_error)
    if score is None:
        raise AssertionError("successful OSWorld task has no score")
    provenance_hook = getattr(desktop_factory, "provenance", None)
    factory_provenance = (
        dict(provenance_hook())
        if callable(provenance_hook)
        else {"factory": type(desktop_factory).__name__}
    )
    metadata = {
        "version": task.data["version"],
        "revision": task.data["revision"],
        "domain": task.data["domain"],
        "mode": "multi" if multi_agent else "single",
        "agents": dict(agent_statuses),
        "agent_steps": steps,
        "agent_error": agent_error,
        "score_scale": "0-1",
        "environment_factory": factory_provenance,
        "verifier_exposed_to_agent": False,
        "state_selection": state_selection,
        **(
            {"state_adoption_history": list(state_adoption_history)}
            if multi_agent
            else {}
        ),
    }
    if evaluator_result is not None:
        metadata["evaluator_result"] = dict(evaluator_result)
    score_path = directory / "score.json"
    atomic_json(score_path, {"task_id": task.task_id, "score": score, **metadata})
    metadata["score_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=answer,
        score=score,
        metadata=metadata,
    )


def _task_config_for_benchmark(task: BenchmarkTask) -> Any:
    info = OSWorldCheckout(
        path=Path(_task_string(task, "checkout")).resolve(),
        version=_task_string(task, "version"),
        revision=_task_string(task, "revision"),
        dirty=False,
    )
    domain = _task_string(task, "domain")
    expected_class = task.data.get("task_class_sha256")
    observed_class = _task_class_sha256(info, domain, task.task_id)
    if expected_class != observed_class:
        raise RuntimeError("OSWorld task class changed after manifest creation")
    config = _load_task_config(info, domain, task.task_id)
    if _task_class_sha256(info, domain, task.task_id) != expected_class:
        raise RuntimeError("OSWorld task class changed while loading")
    expected = task.data.get("task_config_sha256")
    if not isinstance(expected, str) or _config_sha256(config) != expected:
        raise RuntimeError("OSWorld task config changed after manifest creation")
    return config


def _factory_delay(factory: Any, name: str) -> float:
    value = getattr(factory, name, 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"OSWorld {name} must be finite and non-negative")
    return float(value)


def _reject_unsupported_v2_lifecycle(task: BenchmarkTask, config: Any) -> None:
    if task.data.get("version") != "v2":
        return
    phases_getter = getattr(config, "get_phases", None)
    if callable(phases_getter):
        phases = phases_getter() or []
        if not isinstance(phases, list):
            raise TypeError("OSWorld v2 get_phases() must return a list")
        if phases:
            raise NotImplementedError(
                "OSWorld v2 multi-phase tasks require the upstream phase "
                "setup/evaluate/gating lifecycle and are not supported by "
                "the minimal single-loop adapter"
            )
    if _task_value(config, "user_simulator"):
        raise NotImplementedError(
            "OSWorld v2 user-simulator tasks require ASK_USER turns and are "
            "not supported by the minimal computer action protocol"
        )


def _task_class_sha256(
    info: OSWorldCheckout, domain: str, task_id: str
) -> str | None:
    if info.version != "v2":
        return None
    _safe_component(domain, "OSWorld domain")
    _safe_component(task_id, "OSWorld task ID")
    base = info.path / "evaluation_examples"
    _activate_checkout(info.path)
    loader = _import_from_checkout("task_loader", info.path)
    class_path = loader.find_task_class_path(
        task_id=task_id,
        base_dir=str(base),
        domain=domain,
        eval_version="v2",
    )
    if class_path is None:
        return None
    if not isinstance(class_path, str):
        raise ValueError("OSWorld v2 task class path is invalid")
    path = Path(class_path)
    _require_contained(path, base)
    if path.is_symlink() or not path.is_file():
        raise ValueError("OSWorld v2 task class must be a regular non-symlink file")
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("OSWorld task class changed while hashing")
    return digest


def _load_task_config(info: OSWorldCheckout, domain: str, task_id: str) -> Any:
    _safe_component(domain, "OSWorld domain")
    _safe_component(task_id, "OSWorld task ID")
    base = info.path / "evaluation_examples"
    if info.version == "v1":
        path = base / "examples" / domain / f"{task_id}.json"
        _require_contained(path, base)
        if not path.is_file():
            raise FileNotFoundError(f"missing OSWorld v1 task: {path}")
        return strict_json_loads(path.read_text(encoding="utf-8"))
    _activate_checkout(info.path)
    loader = _import_from_checkout("task_loader", info.path)
    config_path = loader.resolve_task_json_path(
        task_id=task_id,
        base_dir=str(base),
        domain=domain,
        eval_version="v2",
    )
    if not isinstance(config_path, str):
        raise FileNotFoundError("OSWorld v2 task path could not be resolved")
    _require_contained(Path(config_path), base)
    class_path = loader.find_task_class_path(
        task_id=task_id,
        base_dir=str(base),
        domain=domain,
        eval_version="v2",
    )
    if class_path is not None:
        if not isinstance(class_path, str):
            raise ValueError("OSWorld v2 task class path is invalid")
        _require_contained(Path(class_path), base)
    try:
        return loader.load_task_config(
            config_path,
            task_id=task_id,
            base_dir=str(base),
            domain=domain,
            eval_version="v2",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "OSWorld v2 gated task classes/assets are missing; install the exact "
            f"{OSWORLD_V2_REVISION} release data under {base}"
        ) from exc


def _desktop_env_class(info: OSWorldCheckout) -> Any:
    _activate_checkout(info.path)
    module = _import_from_checkout("desktop_env.desktop_env", info.path)
    return module.DesktopEnv


def _import_from_checkout(name: str, checkout: Path) -> Any:
    module = importlib.import_module(name)
    source = Path(inspect.getfile(module)).resolve()
    try:
        source.relative_to(checkout)
    except ValueError as exc:
        raise RuntimeError("a different OSWorld checkout is already imported") from exc
    return module


def _activate_checkout(path: Path) -> None:
    # Exact upstream trees are execution inputs, not writable Python caches.
    # Prevent imports after inspection from creating a later shadowing input.
    sys.dont_write_bytecode = True
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _task_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except TypeError:
            pass
    return getattr(config, name, None)


def _mentions_gitlab(config: Any) -> bool:
    if not isinstance(config, Mapping):
        dump = getattr(config, "model_dump", None)
        config = dump(mode="json") if callable(dump) else vars(config)

    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return "gitlab" in value.casefold()
        if isinstance(value, Mapping):
            return any(contains(key) or contains(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains(item) for item in value)
        return False

    return contains(config)


def _config_sha256(config: Any) -> str:
    if isinstance(config, Mapping):
        value = dict(config)
    else:
        dump = getattr(config, "model_dump", None)
        value = dump(mode="json") if callable(dump) else vars(config)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _osworld_score(value: Any) -> tuple[float, Mapping[str, Any] | None]:
    details: Mapping[str, Any] | None = None
    if isinstance(value, Mapping):
        details = dict(value)
        if "score" not in value:
            raise RuntimeError("OSWorld evaluator result has no score")
        value = value["score"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError("OSWorld evaluator returned a non-finite score")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("OSWorld evaluator score must be in [0, 1]")
    return score, details


def _safe_component(value: str, label: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one safe path component")


def _task_string(task: BenchmarkTask, name: str) -> str:
    return task_string(task, name, label="OSWorld")


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("OSWorld task path escapes evaluation_examples") from exc


async def _close_desktop(desktop: Any) -> None:
    close = getattr(desktop, "close", None)
    if close is None:
        return
    if inspect.iscoroutinefunction(close):
        await close()
    else:
        await complete_in_thread(close)


def _append_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(content)
        stream.flush()


def _v2_untracked_exemption(relative: Path, status: os.stat_result) -> bool:
    # The gated v2 release installs these outside Git. The selected class is
    # content-hashed immediately before and after its loader executes.
    return (
        _v2_gated_task_class(relative)
        and stat.S_ISREG(status.st_mode)
        and not status.st_mode & 0o111
    )


def _v2_gated_task_class(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) != 3 or parts[:2] != ("evaluation_examples", "task_class"):
        return False
    name = parts[2]
    if not name.startswith("task_") or not name.endswith(".py"):
        return False
    task_id = name[len("task_") : -len(".py")]
    return bool(task_id) and len(task_id) <= 128 and all(
        character.isascii()
        and (character.isalnum() or character in {"-", "_"})
        for character in task_id
    )




__all__ = [
    "DesktopFactory",
    "OSWorldCheckout",
    "UpstreamDesktopFactory",
    "inspect_osworld_checkout",
    "load_osworld",
    "run_osworld_task",
]
