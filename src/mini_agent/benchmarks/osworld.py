"""OSWorld v1/v2 lifecycle adapter; the hidden evaluator stays outside agents."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
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
from ..types import (
    BudgetExceeded,
    BudgetLimits,
    _require_bool,
    _require_callable,
    _require_finite_number,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
    strict_json_loads,
)
from .._hash import machine_image_identity, stat_key
from ..storage import (
    atomic_bytes,
    atomic_json,
)
from .base import (
    task_agent_builder,
    BenchmarkTask,
    EvaluationOutcome,
    combine_errors,
    raise_after_cleanup,
    shielded_create,
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
    checkout: Path, *, version: str, allow_dirty: bool = False
) -> OSWorldCheckout:
    root = _require_no_symlink(checkout.expanduser(), "OSWorld checkout").resolve()
    if version not in {"v1", "v2"}:
        raise ValueError("OSWorld version must be v1 or v2")
    _require_bool(allow_dirty, "allow_dirty")
    if not (root / "desktop_env" / "desktop_env.py").is_file():
        raise ValueError(f"not an OSWorld checkout: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    if version == "v1" and revision != OSWORLD_V1_REVISION:
        raise ValueError(
            f"OSWorld v1 checkout must be {OSWORLD_V1_REVISION}, found {revision}"
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
    exempt = _v2_untracked_exemption if version == "v2" else None
    reject_untracked_execution_files(root, label="OSWorld", exempt=exempt, run_git=_git)
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

    if limit is not None:
        _require_positive_int(limit, "OSWorld limit")
    _require_bool(exclude_gitlab, "exclude_gitlab")

    info = inspect_osworld_checkout(checkout, version=version)
    if Path.cwd().resolve() != info.path:
        raise RuntimeError(
            "OSWorld uses checkout-relative task assets; run evaluation with the "
            "checkout as the process working directory"
        )
    name = "test_all.json" if version == "v1" else "test_v2.json"
    default = info.path / "evaluation_examples" / name
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
        _require_str(provider_name, "OSWorld provider_name")
        if path_to_vm is not None and not isinstance(path_to_vm, str):
            raise ValueError("OSWorld path_to_vm must be a string or None")
        for flag, name in ((headless, "headless"), (enable_proxy, "enable_proxy")):
            _require_bool(flag, f"OSWorld {name}")
        for size in (screen_width, screen_height):
            _require_positive_int(size, "OSWorld screen dimensions")
        _require_str(client_password, "OSWorld client_password", non_empty=False)
        normalized_provider = provider_name.casefold().strip()
        if normalized_provider == "docker" and path_to_vm is None:
            raise ValueError("OSWorld Docker provider requires an explicit VM image")
        if apptainer_image is not None and normalized_provider != "docker":
            raise ValueError("OSWorld Apptainer compatibility requires Docker provider")
        if apptainer_image is not None and not isinstance(apptainer_image, Path):
            raise ValueError("OSWorld apptainer_image must be a Path or None")
        _require_str(apptainer_executable, "OSWorld apptainer_executable")
        _require_callable(sleep, "OSWorld sleep")
        self.checkout = inspect_osworld_checkout(checkout, version=version)
        self.provider_name = provider_name
        self.vm_image = (
            machine_image_identity(Path(path_to_vm), label="OSWorld VM image")
            if normalized_provider == "docker" and path_to_vm is not None
            else None
        )
        self.path_to_vm = (
            str(self.vm_image["path"]) if self.vm_image is not None else path_to_vm
        )
        self.apptainer_image = (
            machine_image_identity(apptainer_image, label="OSWorld Apptainer image")
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
        from ..runtimes.osworld_apptainer import OSWorldApptainerDockerClient

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
                machine_image_identity, Path(str(expected["path"])), label=label
            )
            if observed != expected:
                raise RuntimeError(
                    f"{label} changed after its manifest identity was recorded"
                )

    async def verify_checkout(self) -> None:
        """Revalidate the exact executable checkout at a lifecycle boundary."""

        observed = await asyncio.to_thread(
            inspect_osworld_checkout, self.checkout.path, version=self.checkout.version
        )
        if observed != self.checkout:
            raise RuntimeError(
                "OSWorld checkout changed after its manifest identity was recorded"
            )

    def provenance(self) -> Mapping[str, Any]:
        apptainer = self.apptainer_image
        return {
            "checkout": self.checkout.as_dict(),
            "provider_name": self.provider_name,
            "screen_size": list(self.screen_size),
            "headless": self.headless,
            "enable_proxy": self.enable_proxy,
            "vm_image": dict(self.vm_image) if self.vm_image is not None else None,
            "container_runtime": (
                "apptainer" if apptainer is not None else "docker"
                if self.provider_name.casefold().strip() == "docker"
                else None
            ),
            "apptainer_image": dict(apptainer) if apptainer is not None else None,
            "apptainer_executable": (
                self.apptainer_executable if apptainer is not None else None
            ),
            "runtime_adaptation": (
                {
                    "scope": "container-network-display-and-firmware-mode-only",
                    "network": "qemu-user-hostfwd",
                    "display": "disabled",
                    "firmware": (
                        "exact-container-bytes-materialized-0600-for-fakeroot-qemu"
                    ),
                }
                if apptainer is not None
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
                    setattr(_DOCKER_MODULE, "from_env", _ORIGINAL_DOCKER_FROM_ENV)
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
        self, *, task: BenchmarkTask, directory: Path, desktop_factory: DesktopFactory
    ) -> None:
        _require_callable(desktop_factory, "desktop_factory")
        self.task = task
        self.directory = directory
        self.desktop_factory = desktop_factory
        self.desktops: list[Any] = []
        self.sleep = _require_callable(
            getattr(desktop_factory, "sleep", asyncio.sleep),
            "OSWorld desktop factory sleep",
        )
        self.initial_settle_seconds = _factory_delay(
            desktop_factory, "initial_settle_seconds"
        )
        self.evaluation_settle_seconds = _factory_delay(
            desktop_factory, "evaluation_settle_seconds"
        )
        self.refresh_initial_observation = _require_bool(
            getattr(desktop_factory, "refresh_initial_observation", False),
            "OSWorld refresh_initial_observation",
        )

    async def environment(self, agent_id: str) -> OSWorldEnvironment:
        config = _task_config_for_benchmark(self.task)
        _reject_unsupported_v2_lifecycle(self.task, config)
        digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
        branch = self.directory / "branches" / digest
        branch.mkdir(parents=True, exist_ok=True)
        created = self.desktop_factory(agent_id, branch / "cache")
        if inspect.isawaitable(created):
            desktop: Any = await shielded_create(
                asyncio.ensure_future(created),
                label="OSWorld environment creation",
                close=_close_desktop,
            )
        else:
            desktop = created
        self.desktops.append(desktop)
        try:
            initial = await complete_in_thread(desktop.reset, task_config=config)
            _require_mapping(initial, "OSWorld reset observation", error=RuntimeError)
            if self.initial_settle_seconds:
                await self.sleep(self.initial_settle_seconds)
            if self.refresh_initial_observation:
                hook = getattr(desktop, "_get_obs", None)
                getter = _require_callable(hook, "OSWorld _get_obs", error=RuntimeError)
                initial = await complete_in_thread(getter)
                _require_mapping(
                    initial, "OSWorld _get_obs observation", error=RuntimeError
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
            line = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
            await complete_in_thread(_append_bytes, branch / "trajectory.jsonl", line)

        client = OSWorldClient(
            desktop,
            initial,
            transition_sink=record,
            owns_environment=False,
            resource_identity=f"osworld:{self.task.task_id}:{digest}",
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

    for factory, name in ((desktop_factory, "desktop"), (model_factory, "model")):
        _require_callable(factory, f"{name}_factory")
    _require_str(system_prompt, "system_prompt", non_empty=False)
    _require_positive_int(max_steps, "max_steps")
    _require_bool(multi_agent, "multi_agent")
    checkout = Path(task_string(task, "checkout", label="OSWorld")).resolve()
    version = task_string(task, "version", label="OSWorld")
    checkout_info = inspect_osworld_checkout(checkout, version=version)
    if task.data.get("revision") != checkout_info.revision:
        raise RuntimeError("OSWorld checkout changed after manifest creation")
    if Path.cwd().resolve() != checkout:
        raise RuntimeError("OSWorld task must run from its pinned checkout")
    pool = _DesktopPool(
        task=task, directory=directory, desktop_factory=desktop_factory
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
    hook = getattr(desktop_factory, "provenance", None)
    factory_provenance = (
        dict(hook()) if callable(hook) else {"factory": type(desktop_factory).__name__}
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
    }
    if multi_agent:
        metadata["state_adoption_history"] = list(state_adoption_history)
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
        path=Path(task_string(task, "checkout", label="OSWorld")).resolve(),
        version=task_string(task, "version", label="OSWorld"),
        revision=task_string(task, "revision", label="OSWorld"),
        dirty=False,
    )
    domain = task_string(task, "domain", label="OSWorld")
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
    return _require_finite_number(
        getattr(factory, name, 0.0), f"OSWorld {name}", minimum=0
    )


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


def _task_class_sha256(info: OSWorldCheckout, domain: str, task_id: str) -> str | None:
    if info.version != "v2":
        return None
    base = info.path / "evaluation_examples"
    loader, selection = _v2_loader(info, domain, task_id)
    class_path = loader.find_task_class_path(**selection)
    if class_path is None:
        return None
    path = Path(_require_str(class_path, "OSWorld v2 task class path"))
    _require_contained(path, base)
    if path.is_symlink() or not path.is_file():
        raise ValueError("OSWorld v2 task class must be a regular non-symlink file")
    before = stat_key(path.stat())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if stat_key(path.stat()) != before:
        raise RuntimeError("OSWorld task class changed while hashing")
    return digest


def _v2_loader(
    info: OSWorldCheckout, domain: str, task_id: str
) -> tuple[Any, dict[str, Any]]:
    """Import the pinned upstream loader plus one task's selection arguments."""

    _safe_component(domain, "OSWorld domain")
    _safe_component(task_id, "OSWorld task ID")
    loader = _import_from_checkout("task_loader", info.path)
    return loader, {
        "task_id": task_id,
        "base_dir": str(info.path / "evaluation_examples"),
        "domain": domain,
        "eval_version": "v2",
    }


def _load_task_config(info: OSWorldCheckout, domain: str, task_id: str) -> Any:
    base = info.path / "evaluation_examples"
    if info.version == "v1":
        _safe_component(domain, "OSWorld domain")
        _safe_component(task_id, "OSWorld task ID")
        path = base / "examples" / domain / f"{task_id}.json"
        _require_contained(path, base)
        if not path.is_file():
            raise FileNotFoundError(f"missing OSWorld v1 task: {path}")
        return strict_json_loads(path.read_text(encoding="utf-8"))
    loader, selection = _v2_loader(info, domain, task_id)
    config_path = loader.resolve_task_json_path(**selection)
    if not isinstance(config_path, str):
        raise FileNotFoundError("OSWorld v2 task path could not be resolved")
    _require_contained(Path(config_path), base)
    class_path = loader.find_task_class_path(**selection)
    if class_path is not None:
        _require_contained(
            Path(_require_str(class_path, "OSWorld v2 task class path")), base
        )
    try:
        return loader.load_task_config(config_path, **selection)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "OSWorld v2 gated task classes/assets are missing; install the exact "
            f"{OSWORLD_V2_REVISION} release data under {base}"
        ) from exc


def _desktop_env_class(info: OSWorldCheckout) -> Any:
    return _import_from_checkout("desktop_env.desktop_env", info.path).DesktopEnv


def _import_from_checkout(name: str, checkout: Path) -> Any:
    _activate_checkout(checkout)
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


def _config_data(config: Any) -> Any:
    """Return the plain JSON-shaped data behind a v1 mapping or a v2 model."""

    if isinstance(config, Mapping):
        return dict(config)
    dump = getattr(config, "model_dump", None)
    return dump(mode="json") if callable(dump) else vars(config)


def _mentions_gitlab(config: Any) -> bool:
    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return "gitlab" in value.casefold()
        if isinstance(value, Mapping):
            return any(contains(key) or contains(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains(item) for item in value)
        return False

    return contains(_config_data(config))


def _config_sha256(config: Any) -> str:
    encoded = json.dumps(
        _config_data(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _osworld_score(value: Any) -> tuple[float, Mapping[str, Any] | None]:
    details: Mapping[str, Any] | None = None
    if isinstance(value, Mapping):
        details = dict(value)
        if "score" not in value:
            raise RuntimeError("OSWorld evaluator result has no score")
        value = value["score"]
    score = _require_finite_number(value, "OSWorld evaluator score", error=RuntimeError)
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("OSWorld evaluator score must be in [0, 1]")
    return score, details


def _safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or set(value) & {"/", "\\", "\x00"}
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


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
        character.isascii() and (character.isalnum() or character in {"-", "_"})
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
