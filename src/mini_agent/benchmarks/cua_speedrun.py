"""Pinned cua-speed-run adapter for local QEMU/Apptainer computer canaries."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence, cast

from ..environments.cua import (
    CUA_SPEED_RUN_REVISION,
    CUAEnvironment,
    CUASpeedRunAdapterClient,
    ComputerObservation,
)
from ..environments.base import complete_in_thread
from ..models import Model
from ..orchestrator import Orchestrator
from ..runtime import RunContext
from ..specs import AgentSpecV1
from ..types import BudgetExceeded, BudgetLimits
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


GYM_ANYTHING_REVISION = "70d9e51d2517049d995cc820a319a355c3c6e979"
_ImageStat = tuple[int, int, int, int, int, int]
_MACHINE_IMAGE_IDENTITIES: dict[
    tuple[str, _ImageStat, _ImageStat | None], Mapping[str, Any]
] = {}


class _TaskClockExpired(TimeoutError):
    pass


@dataclass(frozen=True)
class _FailureVerdict:
    passed: bool
    score: float
    detail: str


async def _on_task_clock(awaitable: Awaitable[Any], timeout: float) -> Any:
    task: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError as exc:
        if task.cancelled():
            raise _TaskClockExpired from exc
        raise


@dataclass(frozen=True)
class CUASpeedRunCheckout:
    path: Path
    revision: str
    dirty: bool
    gym_anything_path: Path
    gym_anything_revision: str
    gym_anything_dirty: bool


def inspect_cua_speedrun_checkout(
    checkout: Path, *, allow_dirty: bool = False
) -> CUASpeedRunCheckout:
    expanded = checkout.expanduser()
    if expanded.is_symlink():
        raise ValueError("cua-speed-run checkout must not be a symlink")
    if not isinstance(allow_dirty, bool):
        raise ValueError("allow_dirty must be boolean")
    root = expanded.resolve()
    if not (root / "src" / "cua_speedrun" / "specs.py").is_file():
        raise ValueError(f"not a cua-speed-run checkout: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != CUA_SPEED_RUN_REVISION:
        raise ValueError(
            f"cua-speed-run must be {CUA_SPEED_RUN_REVISION}, found {revision}"
        )
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise ValueError("cua-speed-run checkout has tracked modifications")
    reject_untracked_execution_files(root, label="cua-speed-run", run_git=_git)
    gitlink = _git(root, "ls-tree", "HEAD", "--", "third_party/gym-anything")
    expected_gitlink = (
        f"160000 commit {GYM_ANYTHING_REVISION}\tthird_party/gym-anything"
    )
    if gitlink != expected_gitlink:
        raise ValueError(
            "cua-speed-run has the wrong gym-anything gitlink: "
            f"expected {GYM_ANYTHING_REVISION}"
        )
    gym_anything = root / "third_party" / "gym-anything"
    if (
        gym_anything.is_symlink()
        or not (gym_anything / "src" / "gym_anything" / "__init__.py").is_file()
    ):
        raise ValueError(
            f"cua-speed-run gym-anything submodule is not initialized at {gym_anything}"
        )
    gym_revision = _git(gym_anything, "rev-parse", "HEAD")
    if gym_revision != GYM_ANYTHING_REVISION:
        raise ValueError(
            "cua-speed-run gym-anything submodule must be "
            f"{GYM_ANYTHING_REVISION}, found {gym_revision}"
        )
    gym_dirty = bool(
        _git(gym_anything, "status", "--porcelain", "--untracked-files=all")
    )
    reject_untracked_execution_files(
        gym_anything, label="gym-anything", run_git=_git
    )
    if gym_dirty and not allow_dirty:
        raise ValueError("cua-speed-run gym-anything submodule is dirty")
    return CUASpeedRunCheckout(
        root,
        revision,
        dirty,
        gym_anything,
        gym_revision,
        gym_dirty,
    )


def load_cua_speedrun(
    checkout: Path,
    benchmark: Path,
    *,
    seed: int = 0,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[BenchmarkTask, ...]:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("cua-speed-run seed must be an integer")
    if isinstance(task_ids, (str, bytes)) or not all(
        isinstance(task_id, str) and task_id for task_id in task_ids
    ):
        raise ValueError("cua-speed-run task_ids must be non-empty strings")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("cua-speed-run task_ids must be unique")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("cua-speed-run limit must be positive")
    info = inspect_cua_speedrun_checkout(checkout)
    _activate(info.path)
    specification = _load_benchmark(benchmark)
    selected = list(specification.tasks)
    if task_ids:
        by_id = {task.task_id: task for task in selected}
        missing = sorted(set(task_ids).difference(by_id))
        if missing:
            raise ValueError(f"unknown cua-speed-run tasks: {missing}")
        selected = [by_id[task_id] for task_id in task_ids]
    if limit is not None:
        selected = selected[:limit]
    tasks: list[BenchmarkTask] = []
    for upstream in selected:
        generator = upstream.load_generator()
        environment_identity = _gym_anything_task_identity(upstream, checkout=info.path)
        prompt = (
            str(environment_identity["prompt"])
            if environment_identity is not None
            else upstream.description
        )
        if generator is not None:
            generated = generator.generate(seed)
            generated_sha256, prompt = _generated_task_sha256(generated)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"cua-speed-run task {upstream.task_id!r} has no prompt")
        raw_timeout = upstream.timeout_sec
        raw_grace = upstream.grace_sec
        if (
            isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or not math.isfinite(float(raw_timeout))
            or raw_timeout <= 0
        ):
            raise ValueError("cua-speed-run task timeout must be finite and positive")
        if (
            isinstance(raw_grace, bool)
            or not isinstance(raw_grace, (int, float))
            or not math.isfinite(float(raw_grace))
            or raw_grace < 0
        ):
            raise ValueError("cua-speed-run task grace must be finite and non-negative")
        timeout = float(raw_timeout)
        grace = float(raw_grace)
        tasks.append(
            BenchmarkTask(
                upstream.task_id,
                prompt,
                {
                    "checkout": str(info.path),
                    "revision": info.revision,
                    "gym_anything_revision": info.gym_anything_revision,
                    "benchmark": str(specification.benchmark_dir),
                    "benchmark_name": specification.name,
                    "benchmark_version": specification.version,
                    "seed": seed,
                    "timeout_seconds": timeout,
                    "grace_seconds": grace,
                    "task_source_sha256": _task_source_sha256(upstream),
                    **(
                        {"generated_task_sha256": generated_sha256}
                        if generator is not None
                        else {}
                    ),
                    **(
                        {
                            "environment_source_sha256": environment_identity["sha256"],
                            "environment_spec": environment_identity[
                                "environment_spec"
                            ],
                            "environment_task_spec": environment_identity["task_spec"],
                            "environment_source_file_count": environment_identity[
                                "source_file_count"
                            ],
                            "environment_mount_count": environment_identity[
                                "mount_count"
                            ],
                            "prompt_source": environment_identity["prompt_source"],
                        }
                        if environment_identity is not None
                        else {}
                    ),
                },
            )
        )
    if not tasks:
        raise ValueError("cua-speed-run benchmark contains no selected tasks")
    return tuple(tasks)


class _AdapterPool:
    def __init__(
        self,
        *,
        task: BenchmarkTask,
        directory: Path,
        backend_name: str,
        checkout_identity: CUASpeedRunCheckout | None = None,
    ) -> None:
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise ValueError("backend_name must be non-empty")
        self.task = task
        self.directory = directory
        self.backend_name = backend_name
        self.checkout_identity = checkout_identity
        self.backend: Any | None = None
        self.prepared: list[Any] = []
        self.by_adapter: dict[int, tuple[Any, Callable[[], Any] | None]] = {}
        self.runtime_assets: dict[int, tuple[Mapping[str, Any], Any]] = {}
        self.evidence: list[Mapping[str, Any]] = []
        self.available: list[tuple[int, Any, Callable[[], Any] | None]] = []

    async def prepare(self, slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1:
            raise ValueError("adapter slot must be a positive integer")
        await self.verify_checkout()
        prepared_directory = self.directory / "prepared" / f"{slot:04d}"
        prepared_directory.mkdir(parents=True, exist_ok=True)
        upstream_task = _upstream_task(self.task)
        if self.backend is None:
            self.backend = _backend(self.backend_name)
        backend = self.backend
        prepare = asyncio.create_task(
            asyncio.to_thread(
                backend.prepare,
                upstream_task.env,
                int(self.task.data["seed"]),
                prepared_directory,
            )
        )
        prepared: Any = None
        try:
            prepared = await asyncio.shield(prepare)
        except BaseException as operation_error:
            creation_cleanup_error: BaseException | None = None
            try:
                prepared = await prepare
            except BaseException as exc:
                if exc is not operation_error:
                    creation_cleanup_error = exc
            if prepared is not None:
                try:
                    await complete_in_thread(prepared.adapter.close)
                except BaseException as exc:
                    creation_cleanup_error = combine_errors(creation_cleanup_error, exc)
            raise_after_cleanup(
                "cua-speed-run environment creation",
                operation_error,
                creation_cleanup_error,
            )
        try:
            generator = upstream_task.load_generator()
            resolved_description = prepared.description
            checker: Callable[[], Any] | None = None
            if generator is not None:
                generated = generator.generate(int(self.task.data["seed"]))
                generated_sha256, resolved_description = _generated_task_sha256(
                    generated
                )
                if generated_sha256 != self.task.data.get("generated_task_sha256"):
                    raise RuntimeError(
                        "generated cua-speed-run task differs from the manifest"
                    )
                assert isinstance(generated, Mapping)
                checker = _checker(generator, prepared, generated["expected"])
            if not isinstance(resolved_description, str):
                raise RuntimeError("prepared task instruction is not a string")
            if resolved_description != self.task.prompt:
                raise RuntimeError(
                    "prepared task instruction differs from the manifest"
                )
            prepare_time = float(prepared.prepare_time_sec)
            if not math.isfinite(prepare_time) or prepare_time < 0:
                raise RuntimeError(
                    "environment preparation time must be finite and non-negative"
                )
            if not isinstance(prepared.info, Mapping):
                raise RuntimeError("prepared environment info must be an object")
            runtime_assets = _prepared_runtime_assets(prepared, upstream_task)
            evidence = {
                "slot": slot,
                "prepare_time_seconds": prepare_time,
                "info": dict(prepared.info),
                "runtime_assets": runtime_assets,
            }
        except BaseException as operation_error:
            validation_cleanup_error: BaseException | None = None
            try:
                await complete_in_thread(prepared.adapter.close)
            except BaseException as exc:
                validation_cleanup_error = exc
            raise_after_cleanup(
                "cua-speed-run environment preparation",
                operation_error,
                validation_cleanup_error,
            )
        self.prepared.append(prepared)
        self.runtime_assets[id(prepared.adapter)] = (
            runtime_assets,
            upstream_task,
        )
        self.evidence.append(evidence)
        self.available.append((slot, prepared, checker))

    async def prewarm(self, count: int, concurrency: int) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (count, concurrency)
        ):
            raise ValueError("prewarm count and concurrency must be positive integers")
        semaphore = asyncio.Semaphore(concurrency)

        async def one(slot: int) -> None:
            async with semaphore:
                await self.prepare(slot)

        tasks = [asyncio.create_task(one(slot)) for slot in range(1, count + 1)]
        try:
            await asyncio.gather(*tasks)
            self.available.sort(key=lambda item: item[0])
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def environment(self, agent_id: str) -> CUAEnvironment:
        await self.verify_checkout()
        branch = (
            self.directory
            / "branches"
            / hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
        )
        branch.mkdir(parents=True, exist_ok=True)
        if not self.available:
            await self.prepare(len(self.prepared) + 1)
        _slot, prepared, checker = self.available.pop(0)
        expected_assets, upstream_task = self.runtime_assets[id(prepared.adapter)]
        observed_assets = await asyncio.to_thread(
            _prepared_runtime_assets, prepared, upstream_task
        )
        if observed_assets != expected_assets:
            raise RuntimeError(
                "cua-speed-run runtime assets changed after preparation and "
                "before the model call"
            )
        self.by_adapter[id(prepared.adapter)] = (prepared, checker)
        observation_number = 0
        transition_number = 0

        async def observation_sink(observation: ComputerObservation) -> None:
            nonlocal observation_number
            observation_number += 1
            await complete_in_thread(
                atomic_bytes,
                branch / f"observation_{observation_number:04d}.png",
                observation.png,
            )

        async def transition_sink(transition: Mapping[str, Any]) -> None:
            nonlocal transition_number
            transition_number += 1
            safe = {
                "step": transition_number,
                "actions": transition.get("actions"),
                "done": (
                    transition.get("result", {}).get("done")
                    if isinstance(transition.get("result"), Mapping)
                    else None
                ),
            }
            await complete_in_thread(_append_jsonl, branch / "trajectory.jsonl", safe)

        client = CUASpeedRunAdapterClient(
            prepared.adapter,
            owns_adapter=False,
            resource_identity=(
                f"cua-speed-run:{self.task.task_id}:"
                + hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
            ),
            observation_sink=observation_sink,
            transition_sink=transition_sink,
        )
        return CUAEnvironment(client, benchmark="cua-speed-run")

    async def verify_checkout(self) -> None:
        if self.checkout_identity is None:
            return
        observed = await asyncio.to_thread(
            inspect_cua_speedrun_checkout, self.checkout_identity.path
        )
        if observed != self.checkout_identity:
            raise RuntimeError(
                "cua-speed-run checkout changed after its manifest identity "
                "was recorded"
            )

    async def evaluate(self, environment: CUAEnvironment) -> Any:
        adapter = cast(CUASpeedRunAdapterClient, environment.client).adapter
        record = self.by_adapter.get(id(adapter))
        if record is None:
            raise RuntimeError("selected computer adapter is not owned by this task")
        prepared, checker = record
        if checker is not None:
            return await complete_in_thread(checker)
        if prepared.checker is not None:
            return await complete_in_thread(prepared.checker)
        return await complete_in_thread(adapter.finalize)

    async def close(self) -> None:
        errors: list[BaseException] = []
        cancelled = False
        remaining: list[Any] = []
        for prepared in reversed(self.prepared):
            try:
                await complete_in_thread(prepared.adapter.close)
            except asyncio.CancelledError:
                cancelled = True
                remaining.append(prepared)
            except Exception as exc:
                errors.append(exc)
                remaining.append(prepared)
        self.prepared = list(reversed(remaining))
        remaining_ids = {id(prepared.adapter) for prepared in remaining}
        self.by_adapter = {
            key: value for key, value in self.by_adapter.items() if key in remaining_ids
        }
        self.runtime_assets = {
            key: value
            for key, value in self.runtime_assets.items()
            if key in remaining_ids
        }
        self.available = [
            value for value in self.available if id(value[1].adapter) in remaining_ids
        ]
        if errors:
            raise RuntimeError(
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]
        if cancelled:
            raise asyncio.CancelledError()


async def run_cua_speedrun_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    backend_name: str,
    model_factory: ModelFactory,
    system_prompt: str,
    max_steps: int,
    multi_agent: bool = False,
    max_active_agents: int = 4,
    max_total_agents: int = 8,
    per_agent_limits: BudgetLimits | None = None,
    agent_spec: AgentSpecV1 | None = None,
) -> EvaluationOutcome:
    """Run the minimal agent on a pinned cua-speed-run environment backend."""

    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("backend_name must be non-empty")
    if not callable(model_factory):
        raise ValueError("model_factory must be callable")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(multi_agent, bool):
        raise ValueError("multi_agent must be boolean")
    checkout = _task_string(task, "checkout")
    info = inspect_cua_speedrun_checkout(Path(checkout))
    if task.data.get("revision") != info.revision:
        raise RuntimeError("cua-speed-run checkout changed after manifest creation")
    expected_gym_revision = task.data.get("gym_anything_revision")
    if expected_gym_revision is not None and expected_gym_revision != getattr(
        info, "gym_anything_revision", None
    ):
        raise RuntimeError(
            "cua-speed-run gym-anything checkout changed after manifest creation"
        )
    _activate(info.path)
    pool = _AdapterPool(
        task=task,
        directory=directory,
        backend_name=backend_name,
        checkout_identity=info,
    )
    root_id = task_agent_root(task.task_id)
    root_environment: CUAEnvironment | None = None
    answer = ""
    steps = 0
    agent_error: str | None = None
    statuses: Mapping[str, str] = {}
    state_selection = "root_environment"
    state_adoption_history: tuple[str, ...] = ()

    async def environment_for(agent_id: str) -> CUAEnvironment:
        return await pool.environment(agent_id)

    agent_for = task_agent_builder(
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=max_steps,
        agent_spec=agent_spec,
    )

    timeout = _task_number(task, "timeout_seconds", positive=True)
    operation_error: BaseException | None = None
    verdict: Any = None
    task_time_seconds = 0.0
    finish_reason = "agent_error"
    try:
        if multi_agent:
            await pool.prewarm(max_total_agents, max_active_agents)
            orchestrator = Orchestrator(
                agent_builder=agent_for,
                environment_factory=environment_for,
                context=context,
                max_active_agents=max_active_agents,
                max_total_agents=max_total_agents,
                per_agent_limits=per_agent_limits,
                root_id=root_id,
            )
            started = time.monotonic()
            try:
                result = await _on_task_clock(orchestrator.run(task.prompt), timeout)
                answer, steps = result.answer, result.steps
                finish_reason = "done"
            except _TaskClockExpired:
                task_time_seconds = timeout
                finish_reason = "timeout"
            except asyncio.CancelledError:
                raise
            except BudgetExceeded as exc:
                # Step-budget exhaustion still runs the hidden checker: the
                # machine may already show the requested state.
                agent_error = f"{type(exc).__name__}: {exc}"
                finish_reason = "step_budget"
            except Exception as exc:
                root_record = orchestrator.records.get(root_id)
                if root_record is None or (
                    root_record.environment is None and not pool.by_adapter
                ):
                    # Provisioning and integrity failures belong to the
                    # evaluator.  Do not turn them into an agent score.
                    raise
                agent_error = f"{type(exc).__name__}: {exc}"
            if not task_time_seconds:
                task_time_seconds = min(time.monotonic() - started, timeout)
            root = orchestrator.records.get(root_id)
            if root is not None and root.environment is not None:
                root_environment = root.environment.base
                state_adoption_history = tuple(root.adoption_history)
                if state_adoption_history:
                    state_selection = "adopted_descendant_environment"
            statuses = {
                agent_id: record.status
                for agent_id, record in orchestrator.records.items()
            }
        else:
            root_environment = await pool.environment(root_id)
            agent = await agent_for(root_id, root_environment, context)
            started = time.monotonic()
            inference_error: BaseException | None = None
            try:
                result = await _on_task_clock(agent.run(task.prompt), timeout)
                answer, steps = result.answer, result.steps
                finish_reason = "done"
            except _TaskClockExpired:
                task_time_seconds = timeout
                finish_reason = "timeout"
            except asyncio.CancelledError as exc:
                inference_error = exc
            except BudgetExceeded as exc:
                # Step-budget exhaustion still runs the hidden checker: the
                # machine may already show the requested state.
                agent_error = f"{type(exc).__name__}: {exc}"
                finish_reason = "step_budget"
            except Exception as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
            close_error: BaseException | None = None
            try:
                await root_environment.close()
            except BaseException as exc:
                close_error = exc
            raise_after_cleanup(
                "cua-speed-run root inference", inference_error, close_error
            )
            if not task_time_seconds:
                task_time_seconds = min(time.monotonic() - started, timeout)
            statuses = {
                root_id: (
                    "timed_out"
                    if finish_reason == "timeout"
                    else "failed"
                    if agent_error
                    else "completed"
                )
            }
        if agent_error is not None and finish_reason == "agent_error":
            # cua-speed-run gives an agent-process failure zero without running
            # the checker.  The root environment may already have been closed
            # by the scheduler when model construction itself failed.
            verdict = _FailureVerdict(False, 0.0, agent_error)
        else:
            if root_environment is None:
                raise RuntimeError("root computer environment was never prepared")
            grace = _task_number(task, "grace_seconds", positive=False)
            if grace:
                await asyncio.sleep(grace)
            verdict = await pool.evaluate(root_environment)
    except BaseException as exc:
        operation_error = exc

    cleanup_error: BaseException | None = None
    try:
        await pool.close()
    except BaseException as exc:
        cleanup_error = exc
    checkout_error: BaseException | None = None
    try:
        observed_checkout = await asyncio.to_thread(
            inspect_cua_speedrun_checkout, info.path
        )
        if observed_checkout != info:
            raise RuntimeError(
                "cua-speed-run checkout changed after its manifest identity "
                "was recorded"
            )
    except BaseException as exc:
        checkout_error = exc
    if checkout_error is not None:
        cleanup_error = combine_errors(cleanup_error, checkout_error)
    raise_after_cleanup("cua-speed-run task", operation_error, cleanup_error)
    if verdict is None:
        raise AssertionError("successful cua-speed-run task has no verdict")
    passed_value = getattr(verdict, "passed", None)
    if not isinstance(passed_value, bool):
        raise RuntimeError("cua-speed-run verifier returned a non-boolean verdict")
    passed = passed_value
    raw_score = getattr(verdict, "score", None)
    score = (
        100.0
        if raw_score is None and passed
        else 0.0
        if raw_score is None
        else raw_score
    )
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise RuntimeError("cua-speed-run verifier returned a non-finite score")
    if not 0.0 <= float(score) <= 100.0:
        raise RuntimeError("cua-speed-run verifier score must be in [0, 100]")
    metadata = {
        "benchmark": task.data["benchmark_name"],
        "benchmark_version": task.data["benchmark_version"],
        "harness_revision": task.data["revision"],
        "backend": backend_name,
        "mode": "multi" if multi_agent else "single",
        "agents": dict(statuses),
        "agent_steps": steps,
        "agent_error": agent_error,
        "passed": passed,
        "verifier_exposed_to_agent": False,
        "score_scale": "0-100",
        "state_selection": state_selection,
        **(
            {"state_adoption_history": list(state_adoption_history)}
            if multi_agent
            else {}
        ),
        "timing": {
            "environment_preparation_on_agent_clock": False,
            "task_time_seconds": task_time_seconds,
            "finish_reason": finish_reason,
            "grace_seconds": _task_number(task, "grace_seconds", positive=False),
        },
        "environments": sorted(pool.evidence, key=lambda item: int(item["slot"])),
    }
    detail = getattr(verdict, "detail", "")
    if not isinstance(detail, str):
        raise RuntimeError("cua-speed-run verifier detail must be a string")
    verdict_path = directory / "verdict.json"
    atomic_json(
        verdict_path,
        {
            "task_id": task.task_id,
            "score": float(score),
            "detail": detail,
            **metadata,
        },
    )
    metadata["verdict_sha256"] = hashlib.sha256(verdict_path.read_bytes()).hexdigest()
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=answer,
        score=float(score),
        metadata=metadata,
    )


def preflight_cua_speedrun(
    checkout: Path,
    benchmark: Path,
    *,
    backend_name: str,
) -> Mapping[str, Any]:
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("backend_name must be non-empty")
    info = inspect_cua_speedrun_checkout(checkout)
    _activate(info.path)
    expanded_benchmark = benchmark.expanduser()
    if expanded_benchmark.is_symlink():
        raise ValueError("cua-speed-run benchmark path must not be a symlink")
    benchmark_path = expanded_benchmark.resolve()
    if not (benchmark_path / "manifest.yaml").is_file():
        raise ValueError(
            "computer doctor requires an already materialized cua-speed-run "
            "benchmark directory containing manifest.yaml"
        )
    specification = _load_benchmark(benchmark_path)
    backend = _backend(backend_name)
    checked_environments = _nonprovisioning_backend_checks(
        specification,
        backend_name=backend_name,
        gym_anything_root=info.gym_anything_path,
    )
    return {
        "status": "source_ready",
        "checkout": str(info.path),
        "revision": info.revision,
        "gym_anything": {
            "path": str(info.gym_anything_path),
            "revision": info.gym_anything_revision,
            "dirty": info.gym_anything_dirty,
            "module_origin": _gym_anything_module_origin(info.gym_anything_path),
        },
        "benchmark": specification.name,
        "benchmark_version": specification.version,
        "backend": backend.name,
        "requested_runner": getattr(backend, "runner", None),
        "required_runner": getattr(backend, "require_runner", None),
        "observed_runner": None,
        "tasks": len(specification.tasks),
        "environment_directories_checked": checked_environments,
        "upstream_provisioning_preflight_run": False,
        "machine_launch_canary_run": False,
    }


def prepare_cua_speedrun_backend(
    checkout: Path,
    benchmark: Path,
    *,
    backend_name: str,
) -> Mapping[str, Any]:
    """Run cua-speed-run's real provisioning preflight before evaluation."""

    source = preflight_cua_speedrun(
        checkout,
        benchmark,
        backend_name=backend_name,
    )
    specification = _load_benchmark(benchmark)
    backend = _backend(backend_name)
    preflight = getattr(backend, "preflight", None)
    if not callable(preflight):
        raise RuntimeError("cua-speed-run backend does not expose preflight")
    preflight(specification)
    return {
        **dict(source),
        "status": "backend_ready",
        "observed_runner": getattr(backend, "observed_runner_name", None),
        "upstream_provisioning_preflight_run": True,
        "machine_image_identity_scope": (
            "existing_conventional_cache_candidates_not_runner_selection"
        ),
        "machine_images": _cua_machine_images(),
        "machine_launch_canary_run": False,
    }


def _cua_machine_images() -> list[Mapping[str, Any]]:
    """Observe conventional cache files without claiming runner selection."""

    cache = Path(
        os.environ.get(
            "GYM_ANYTHING_QEMU_CACHE",
            "~/.cache/gym-anything/qemu",
        )
    ).expanduser()
    candidates = [
        cache / "base_ubuntu_gnome.qcow2",
        cache / "base_ubuntu_gnome_arm64.qcow2",
        cache / "base_windows_11.qcow2",
        cache / "base_android_14.qcow2",
    ]
    configured = os.environ.get("OSWORLD_QEMU_BASE_IMAGE", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    images: list[Mapping[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        identity = dict(_cached_machine_image_identity(candidate))
        identity.update(
            {
                "identity_scope": "unverified_cache_candidate",
                "effective_runtime_input": None,
            }
        )
        images.append(identity)
    return images


def _prepared_runtime_assets(prepared: Any, upstream_task: Any) -> Mapping[str, Any]:
    """Record the runner inputs that actually backed a prepared environment."""

    adapter = getattr(prepared, "adapter", None)
    environment = getattr(adapter, "_env", None)
    runner = getattr(environment, "_runner", None)
    if runner is None:
        return {
            "identity_scope": "backend_did_not_expose_prepared_runner",
            "runner": None,
            "files": [],
            "container": None,
        }

    files: list[Mapping[str, Any]] = []
    seen: set[tuple[str, Path]] = set()

    def add_file(role: str, value: Any) -> None:
        if not isinstance(value, (str, os.PathLike)):
            return
        path = Path(value).expanduser()
        if path.is_symlink() or not path.is_file():
            return
        resolved = path.resolve()
        key = (role, resolved)
        if key in seen:
            return
        seen.add(key)
        files.append(
            {
                "role": role,
                **dict(_cached_machine_image_identity(resolved)),
            }
        )

    add_file("base_image", getattr(runner, "base_qcow2", None))
    env = getattr(upstream_task, "env", None)
    if isinstance(env, Mapping) and bool(env.get("use_cache", True)):
        checkpoint_path = getattr(runner, "_get_checkpoint_path", None)
        if callable(checkpoint_path):
            add_file("selected_checkpoint", checkpoint_path())
        else:
            add_file("selected_checkpoint", getattr(runner, "env_checkpoint", None))

    raw_container = getattr(runner, "_container_image", None)
    container: Mapping[str, Any] | None = None
    if isinstance(raw_container, str) and raw_container:
        candidate = Path(raw_container).expanduser()
        if candidate.is_file() and not candidate.is_symlink():
            container = {
                "kind": "local_file",
                **dict(_cached_machine_image_identity(candidate)),
            }
        elif candidate.is_dir() and not candidate.is_symlink():
            tree_sha256, file_count = _source_roots_sha256(
                (("runtime-container", candidate),)
            )
            container = {
                "kind": "local_directory",
                "path": str(candidate.resolve()),
                "sha256": tree_sha256,
                "file_count": file_count,
            }
        else:
            container = {
                "kind": "remote_reference",
                "reference": raw_container,
                "content_addressed": bool(
                    re.search(r"@sha256:[0-9a-fA-F]{64}$", raw_container)
                ),
            }
    return {
        "identity_scope": "effective_prepared_runner_inputs_before_model_call",
        "runner": type(runner).__name__,
        "files": files,
        "container": container,
    }


def _cached_machine_image_identity(path: Path) -> Mapping[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(
            f"cua-speed-run runtime image must not be a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    sidecar = Path(str(resolved) + ".provenance.json")
    key = (
        str(resolved),
        _image_stat(resolved),
        (
            _image_stat(sidecar, follow_symlinks=False)
            if sidecar.exists() or sidecar.is_symlink()
            else None
        ),
    )
    cached = _MACHINE_IMAGE_IDENTITIES.get(key)
    if cached is not None:
        return cached
    identity = machine_image_identity(expanded, label="cua-speed-run runtime image")
    _MACHINE_IMAGE_IDENTITIES[key] = identity
    return identity


def _image_stat(path: Path, *, follow_symlinks: bool = True) -> _ImageStat:
    observed = path.stat(follow_symlinks=follow_symlinks)
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


_UNEXPANDED_VARIABLE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def _nonprovisioning_backend_checks(
    specification: Any, *, backend_name: str, gym_anything_root: Path
) -> int:
    """Check local CUA inputs without creating, downloading, or booting an image."""
    if not backend_name.startswith("gym-anything"):
        return 0
    try:
        gym_anything = importlib.import_module("gym_anything")
    except ImportError as exc:
        raise RuntimeError(
            "gym-anything is not importable; install the exact pinned "
            "cua-speed-run submodule before running this backend"
        ) from exc
    _assert_module_origin(
        gym_anything,
        gym_anything_root,
        label="gym-anything",
    )

    _check_local_runner_commands(backend_name)
    checked: set[Path] = set()
    for task in specification.tasks:
        env = getattr(task, "env", None)
        if not isinstance(env, Mapping) or env.get("kind") != "gym-anything":
            raise ValueError(
                "gym-anything backend requires gym-anything task env blocks"
            )
        for key in ("env_dir", "task_id"):
            if not isinstance(env.get(key), str) or not env[key].strip():
                raise ValueError(f"gym-anything task env requires a non-empty {key}")
        task_id = env["task_id"]
        if (
            task_id in {".", ".."}
            or "/" in task_id
            or "\\" in task_id
            or "\x00" in task_id
        ):
            raise ValueError("gym-anything task_id must be one safe path component")
        expanded = os.path.expanduser(os.path.expandvars(env["env_dir"]))
        if _UNEXPANDED_VARIABLE.search(expanded):
            raise ValueError(
                f"unresolved variable in gym-anything env_dir: {env['env_dir']}"
            )
        env_dir = Path(expanded).resolve()
        if not env_dir.is_dir():
            raise FileNotFoundError(f"gym-anything environment not found: {env_dir}")
        if not any((env_dir / name).is_file() for name in _SPEC_FILENAMES):
            raise FileNotFoundError(
                f"gym-anything environment has no env spec: {env_dir}"
            )
        task_dir = env_dir / "tasks" / task_id
        try:
            task_dir.resolve().relative_to(env_dir)
        except ValueError as exc:
            raise ValueError("gym-anything task_id escapes env_dir") from exc
        if not any((task_dir / name).is_file() for name in _TASK_FILENAMES):
            raise FileNotFoundError(f"gym-anything task spec not found: {task_dir}")
        checked.add(env_dir)
    return len(checked)


_SPEC_FILENAMES = ("env.yaml", "env.yml", "env.json")
_TASK_FILENAMES = ("task.yaml", "task.yml", "task.json")


def _gym_anything_task_identity(
    task: Any, *, checkout: Path | None = None
) -> Mapping[str, Any] | None:
    """Resolve the exact task text and source bytes the upstream backend loads."""

    env = getattr(task, "env", None)
    if not isinstance(env, Mapping) or env.get("kind") != "gym-anything":
        return None
    for key in ("env_dir", "task_id"):
        if not isinstance(env.get(key), str) or not env[key].strip():
            raise ValueError(f"gym-anything task env requires a non-empty {key}")
    task_id = env["task_id"]
    if task_id in {".", ".."} or "/" in task_id or "\\" in task_id or "\x00" in task_id:
        raise ValueError("gym-anything task_id must be one safe path component")
    expanded = os.path.expanduser(os.path.expandvars(env["env_dir"]))
    if _UNEXPANDED_VARIABLE.search(expanded):
        raise ValueError(
            f"unresolved variable in gym-anything env_dir: {env['env_dir']}"
        )
    untrusted_env_dir = Path(expanded)
    if untrusted_env_dir.is_symlink():
        raise ValueError("gym-anything environment directory must not be a symlink")
    env_dir = untrusted_env_dir.resolve()
    if not env_dir.is_dir():
        raise FileNotFoundError(f"gym-anything environment not found: {env_dir}")
    environment_spec = _first_file(env_dir, _SPEC_FILENAMES, "environment spec")
    task_dir = env_dir / "tasks" / task_id
    try:
        task_dir.resolve().relative_to(env_dir)
    except ValueError as exc:
        raise ValueError("gym-anything task_id escapes env_dir") from exc
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise ValueError("gym-anything task directory must be a real directory")
    task_spec = _first_file(task_dir, _TASK_FILENAMES, "task spec")

    loading = importlib.import_module("gym_anything.config.loading")
    if checkout is not None:
        _assert_module_origin(
            loading,
            checkout / "third_party" / "gym-anything",
            label="gym-anything loader",
        )
    loaded_task = loading._load_taskspec(task_spec)
    natural_language = getattr(loaded_task, "natural_language", None)
    description = getattr(loaded_task, "description", None)
    if isinstance(natural_language, Mapping):
        prompt = natural_language.get("prompt") or description
        prompt_source = "gym-anything.task.natural_language.prompt"
    elif isinstance(natural_language, str) and natural_language.strip():
        prompt = natural_language
        prompt_source = "gym-anything.task.natural_language"
    else:
        prompt = description
        prompt_source = "gym-anything.task.description"
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"gym-anything task {task_id!r} has no prompt")

    loaded_environment = loading._load_envspec(environment_spec)
    loading._resolve_mount_sources(loaded_environment, env_dir)
    source_roots: list[tuple[str, Path]] = [
        ("environment-spec", environment_spec),
        ("selected-task", task_dir),
    ]
    mounts = getattr(loaded_environment, "mounts", None)
    if not isinstance(mounts, list):
        raise RuntimeError("gym-anything environment mounts are not a list")
    for index, mount in enumerate(mounts):
        raw_source = getattr(mount, "source", None)
        target = getattr(mount, "target", None)
        mode = getattr(mount, "mode", None)
        if not isinstance(raw_source, str) or not raw_source:
            raise RuntimeError("gym-anything environment has an invalid mount")
        if not isinstance(target, str) or not target:
            raise RuntimeError("gym-anything environment has an invalid mount")
        if not isinstance(mode, str) or not mode:
            raise RuntimeError("gym-anything environment has an invalid mount")
        mount_source = Path(raw_source).expanduser()
        if not mount_source.is_absolute():
            mount_source = mount_source.resolve()
        source_roots.append((f"mount:{index}:{mode}:{target}", mount_source))
    source_sha256, source_file_count = _source_roots_sha256(source_roots)
    return {
        "prompt": prompt,
        "prompt_source": prompt_source,
        "sha256": source_sha256,
        "source_file_count": source_file_count,
        "mount_count": len(mounts),
        "environment_spec": str(environment_spec),
        "task_spec": str(task_spec),
    }


def _reject_python_bytecode(paths: Sequence[Path], *, label: str) -> None:
    bytecode = next(
        (path for path in paths if path.suffix.casefold() in {".pyc", ".pyo"}),
        None,
    )
    if bytecode is not None:
        raise ValueError(f"{label} must not contain Python bytecode: {bytecode}")


def _source_roots_sha256(roots: Sequence[tuple[str, Path]]) -> tuple[str, int]:
    """Hash the exact files copied or mounted by one gym-anything task."""

    digest = hashlib.sha256()
    identities: dict[Path, tuple[int, int, int, int, int]] = {}
    listings: dict[Path, tuple[Path, ...]] = {}
    file_count = 0
    for label, untrusted_root in roots:
        if not isinstance(label, str) or not label or "\x00" in label:
            raise ValueError("gym-anything source label must be non-empty")
        if untrusted_root.is_symlink():
            raise ValueError(
                f"gym-anything source root must not be a symlink: {untrusted_root}"
            )
        root = untrusted_root.resolve()
        source_entries: tuple[Path, ...]
        if root.is_file():
            source_entries = (root,)
        elif root.is_dir():
            discovered = tuple(root.rglob("*"))
            _reject_python_bytecode(discovered, label="gym-anything sources")
            source_entries = tuple(sorted((root,) + discovered))
            listings[root] = source_entries
        else:
            raise FileNotFoundError(f"gym-anything source is missing: {root}")
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        for path in source_entries:
            if path.is_symlink():
                raise ValueError(
                    f"gym-anything sources must not contain symlinks: {path}"
                )
            before = path.stat()
            if path.is_dir():
                kind = b"d"
                size = 0
            elif path.is_file():
                kind = b"f"
                size = before.st_size
                file_count += 1
            else:
                raise ValueError(f"gym-anything sources contain a special file: {path}")
            relative = (
                b"."
                if path == root
                else path.relative_to(root).as_posix().encode("utf-8")
            )
            mode = stat.S_IMODE(before.st_mode)
            digest.update(kind)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(mode.to_bytes(4, "big"))
            digest.update(size.to_bytes(8, "big"))
            if kind == b"f":
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
            after = path.stat()
            identity = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
            )
            if identity != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise RuntimeError(f"gym-anything source changed while hashing: {path}")
            identities[path] = identity
    for root, expected_entries in listings.items():
        discovered = tuple(root.rglob("*"))
        _reject_python_bytecode(discovered, label="gym-anything sources")
        observed_entries = tuple(sorted((root,) + discovered))
        if observed_entries != expected_entries:
            raise RuntimeError(
                f"gym-anything source tree changed while hashing: {root}"
            )
    for path, identity in identities.items():
        current = path.stat()
        if identity != (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise RuntimeError(f"gym-anything source changed while hashing: {path}")
    return digest.hexdigest(), file_count


def _first_file(root: Path, names: Sequence[str], label: str) -> Path:
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"gym-anything {label} must not be a symlink")
        if path.is_file():
            return path
    raise FileNotFoundError(f"gym-anything {label} not found under {root}")


def _check_local_runner_commands(backend_name: str) -> None:
    if backend_name in {
        "gym-anything",
        "gym-anything-local",
        "gym-anything-qemu",
    }:
        has_native = all(
            shutil.which(command) is not None
            for command in ("qemu-system-x86_64", "qemu-img")
        )
        if shutil.which("apptainer") is None and not has_native:
            raise RuntimeError(
                "gym-anything QEMU needs Apptainer or qemu-system-x86_64 and qemu-img"
            )
    elif backend_name == "gym-anything-qemu-apptainer":
        if shutil.which("apptainer") is None:
            raise RuntimeError("gym-anything-qemu-apptainer needs Apptainer")
    elif backend_name == "gym-anything-qemu-native":
        missing = [
            command
            for command in ("qemu-system-x86_64", "qemu-img")
            if shutil.which(command) is None
        ]
        if missing:
            raise RuntimeError(
                "gym-anything-qemu-native is missing: " + ", ".join(missing)
            )
    elif backend_name == "gym-anything-avd-native":
        missing = [
            command for command in ("emulator", "adb") if shutil.which(command) is None
        ]
        if missing:
            raise RuntimeError(
                "gym-anything-avd-native is missing: " + ", ".join(missing)
            )

    if (
        backend_name != "gym-anything-avd-native"
        and sys.platform == "linux"
        and not os.environ.get("CS_ALLOW_QEMU_TCG")
    ):
        if not Path("/dev/kvm").exists():
            raise RuntimeError("gym-anything local QEMU needs /dev/kvm")
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise RuntimeError("this user cannot read and write /dev/kvm")


def _checker(generator: Any, prepared: Any, expected: Any) -> Callable[[], Any]:
    def check() -> Any:
        verdict_type = importlib.import_module("cua_speedrun.envs.base").Verdict
        result = generator.check(prepared.adapter, expected)
        if not isinstance(result, Mapping):
            raise RuntimeError("cua-speed-run generator checker returned a non-object")
        passed = result.get("passed")
        if not isinstance(passed, bool):
            raise RuntimeError(
                "cua-speed-run generator checker returned a non-boolean verdict"
            )
        detail = result.get("detail", "")
        if not isinstance(detail, str):
            raise RuntimeError(
                "cua-speed-run generator checker returned a non-string detail"
            )
        return verdict_type(
            passed=passed,
            score=100.0 if passed else 0.0,
            detail=detail,
        )

    return check


def _generated_task_sha256(generated: Any) -> tuple[str, str]:
    """Bind a seeded instruction and hidden expected value without exposing it."""

    if not isinstance(generated, Mapping):
        raise ValueError("cua-speed-run generator must return an object")
    instruction = generated.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            "cua-speed-run generator instruction must be a non-empty string"
        )
    if "expected" not in generated:
        raise ValueError("cua-speed-run generator omitted expected state")
    try:
        canonical = json.dumps(
            {"expected": generated["expected"], "instruction": instruction},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(
            "cua-speed-run generator output must be canonical JSON data"
        ) from exc
    return hashlib.sha256(canonical).hexdigest(), instruction


def _task_source_sha256(task: Any) -> str:
    task_dir = getattr(task, "task_dir", None)
    if not isinstance(task_dir, Path):
        raise ValueError("cua-speed-run task has no source directory")
    if task_dir.is_symlink():
        raise ValueError("cua-speed-run task source directory must not be a symlink")
    if not task_dir.is_dir():
        raise ValueError("cua-speed-run task source directory does not exist")
    entries = sorted(task_dir.rglob("*"))
    _reject_python_bytecode(entries, label="cua-speed-run task sources")
    if any(path.is_symlink() for path in entries):
        raise ValueError("cua-speed-run task sources must not contain symlinks")
    special = [path for path in entries if not path.is_dir() and not path.is_file()]
    if special:
        raise ValueError("cua-speed-run task sources contain a special file")
    paths = [path for path in entries if path.is_file()]
    if not paths:
        raise ValueError("cua-speed-run task source directory is empty")
    digest = hashlib.sha256()
    identities: dict[Path, tuple[int, int, int, int]] = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing cua-speed-run task source: {path}")
        relative = path.relative_to(task_dir).as_posix().encode("utf-8")
        before = path.stat()
        content = path.read_bytes()
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
            raise RuntimeError("cua-speed-run task source changed while hashing")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        identities[path] = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    observed_entries = sorted(task_dir.rglob("*"))
    _reject_python_bytecode(
        observed_entries, label="cua-speed-run task sources"
    )
    if observed_entries != entries:
        raise RuntimeError("cua-speed-run task sources changed while hashing")
    for path, identity in identities.items():
        current = path.stat()
        if identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise RuntimeError("cua-speed-run task source changed while hashing")
    return digest.hexdigest()


def _upstream_task(task: BenchmarkTask) -> Any:
    benchmark = _load_benchmark(Path(_task_string(task, "benchmark")))
    if benchmark.name != task.data.get(
        "benchmark_name"
    ) or benchmark.version != task.data.get("benchmark_version"):
        raise RuntimeError("cua-speed-run benchmark changed after manifest creation")
    matches = [value for value in benchmark.tasks if value.task_id == task.task_id]
    if len(matches) != 1:
        raise RuntimeError("cua-speed-run task changed after manifest creation")
    upstream = matches[0]
    if _task_source_sha256(upstream) != task.data.get("task_source_sha256"):
        raise RuntimeError("cua-speed-run task sources changed after manifest creation")
    environment_identity = _gym_anything_task_identity(
        upstream, checkout=Path(_task_string(task, "checkout"))
    )
    expected_environment = task.data.get("environment_source_sha256")
    if environment_identity is None:
        if expected_environment is not None:
            raise RuntimeError(
                "cua-speed-run environment source changed after manifest creation"
            )
    elif (
        environment_identity["sha256"] != expected_environment
        or environment_identity["environment_spec"] != task.data.get("environment_spec")
        or environment_identity["task_spec"] != task.data.get("environment_task_spec")
        or environment_identity["source_file_count"]
        != task.data.get("environment_source_file_count")
        or environment_identity["mount_count"]
        != task.data.get("environment_mount_count")
    ):
        raise RuntimeError(
            "cua-speed-run environment sources changed after manifest creation"
        )
    elif (
        upstream.load_generator() is None
        and environment_identity["prompt"] != task.prompt
    ):
        raise RuntimeError(
            "cua-speed-run environment prompt changed after manifest creation"
        )
    for field, observed in (
        ("timeout_seconds", upstream.timeout_sec),
        ("grace_seconds", upstream.grace_sec),
    ):
        expected = task.data.get(field)
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(float(expected))
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(expected) != float(observed)
        ):
            raise RuntimeError("cua-speed-run timing changed after manifest creation")
    return upstream


def _task_string(task: BenchmarkTask, name: str) -> str:
    return task_string(task, name, label="cua-speed-run")


def _task_number(task: BenchmarkTask, name: str, *, positive: bool) -> float:
    value = task.data.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"cua-speed-run task {name} must be finite and {qualifier}")
    return float(value)


def _load_benchmark(path: Path) -> Any:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("cua-speed-run benchmark path must not be a symlink")
    benchmark_type = importlib.import_module("cua_speedrun.specs").Benchmark
    return benchmark_type.load(expanded.resolve())


def _backend(name: str) -> Any:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("cua-speed-run backend name must be non-empty")
    return importlib.import_module("cua_speedrun.envs").get_backend(name)


def _activate(checkout: Path) -> None:
    # Exact upstream trees are execution inputs, not writable Python caches.
    # Prevent imports after inspection from creating a later shadowing input.
    sys.dont_write_bytecode = True
    gym_anything = checkout / "third_party" / "gym-anything"
    for source in (checkout / "src", gym_anything / "src"):
        text = str(source)
        if text not in sys.path:
            sys.path.insert(0, text)
    loaded_cua = sys.modules.get("cua_speedrun")
    if loaded_cua is not None:
        _assert_module_origin(
            loaded_cua,
            checkout,
            label="cua-speed-run",
        )
    loaded_gym = sys.modules.get("gym_anything")
    if loaded_gym is not None:
        _assert_module_origin(
            loaded_gym,
            gym_anything,
            label="gym-anything",
        )


def _gym_anything_module_origin(root: Path) -> str:
    module = importlib.import_module("gym_anything")
    return str(_assert_module_origin(module, root, label="gym-anything"))


def _assert_module_origin(module: Any, root: Path, *, label: str) -> Path:
    try:
        module_path = Path(inspect.getfile(module)).resolve()
        module_path.relative_to(root.resolve())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"a different {label} checkout is already imported") from exc
    return module_path


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()




__all__ = [
    "CUASpeedRunCheckout",
    "inspect_cua_speedrun_checkout",
    "load_cua_speedrun",
    "preflight_cua_speedrun",
    "run_cua_speedrun_task",
]
