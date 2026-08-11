from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from ..agent import MiniAgent
from ..environments.cua import (
    OSWORLD_REVISION,
    OSWorldClient,
    OSWorldEnvironment,
    validate_png,
)
from ..types import AgentResult


Sleep = Callable[[float], Awaitable[None]]
AgentFactory = Callable[[OSWorldEnvironment], MiniAgent | Awaitable[MiniAgent]]


@dataclass(frozen=True)
class OSWorldTaskResult:
    task_id: str
    score: float
    output_directory: Path
    agent_result: AgentResult | None
    agent_error: str | None
    source_revision: str = OSWORLD_REVISION


def _prepare_output(directory: Path) -> Path:
    output = directory.expanduser().resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError(f"OSWorld output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def _call_optional(target: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, name, None)
    if method is None:
        return None
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class OSWorldTaskRunner:
    """Own one OSWorld reset/action/evaluate lifecycle outside the agent.

    The raw DesktopEnv and its evaluator never enter ``agent_factory``. The
    factory receives only an :class:`OSWorldEnvironment` exposing the computer
    tool, preserving the benchmark's hidden-verifier boundary.
    """

    def __init__(
        self,
        desktop_environment: Any,
        *,
        ready_wait_seconds: float = 60.0,
        settle_wait_seconds: float = 20.0,
        sleep: Sleep = asyncio.sleep,
        protocol: str = "generic",
    ) -> None:
        if ready_wait_seconds < 0 or settle_wait_seconds < 0:
            raise ValueError("OSWorld wait durations must be non-negative")
        self.desktop_environment = desktop_environment
        self.ready_wait_seconds = ready_wait_seconds
        self.settle_wait_seconds = settle_wait_seconds
        self.sleep = sleep
        self.protocol = protocol

    async def run(
        self,
        *,
        task_config: Mapping[str, Any],
        agent_factory: AgentFactory,
        output_directory: Path,
    ) -> OSWorldTaskResult:
        task_id = task_config.get("id")
        instruction = task_config.get("instruction")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("OSWorld task config requires a string id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("OSWorld task config requires an instruction")
        output = _prepare_output(output_directory)
        trajectory_path = output / "traj.jsonl"
        recording_path = output / "recording.mp4"
        controller = getattr(self.desktop_environment, "controller", None)
        recording_started = False
        environment: OSWorldEnvironment | None = None
        agent_result: AgentResult | None = None
        agent_error: str | None = None
        score = 0.0
        run_status = "running"

        async def record_transition(transition: Mapping[str, Any]) -> None:
            observation = transition.get("observation")
            if not isinstance(observation, Mapping):
                raise RuntimeError("OSWorld transition is missing its observation")
            screenshot = observation.get("screenshot")
            if not isinstance(screenshot, bytes):
                raise RuntimeError("OSWorld transition is missing screenshot bytes")
            validate_png(screenshot)
            step = transition.get("step")
            if not isinstance(step, int) or step < 1:
                raise RuntimeError("OSWorld transition has an invalid step")
            screenshot_name = f"step_{step:04d}.png"
            _atomic_write(output / screenshot_name, screenshot)
            payload = {
                "step_num": step,
                "action": transition.get("action"),
                "encoded_action": transition.get("encoded_action"),
                "reward": transition.get("reward"),
                "done": transition.get("done"),
                "info": transition.get("info"),
                "screenshot_file": screenshot_name,
            }
            with trajectory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

        try:
            await asyncio.to_thread(
                self.desktop_environment.reset,
                task_config=dict(task_config),
            )
            if self.ready_wait_seconds:
                await self.sleep(self.ready_wait_seconds)
            get_observation = getattr(self.desktop_environment, "_get_obs", None)
            if get_observation is None:
                raise RuntimeError("OSWorld DesktopEnv does not expose _get_obs")
            initial = await asyncio.to_thread(get_observation)
            if not isinstance(initial, Mapping):
                raise RuntimeError("OSWorld initial observation must be an object")
            initial_screenshot = initial.get("screenshot")
            if not isinstance(initial_screenshot, bytes):
                raise RuntimeError("OSWorld initial observation requires screenshot bytes")
            validate_png(initial_screenshot)
            _atomic_write(output / "step_0000.png", initial_screenshot)

            if controller is not None and hasattr(controller, "start_recording"):
                await _call_optional(controller, "start_recording")
                recording_started = True

            client = OSWorldClient(
                self.desktop_environment,
                initial,
                transition_sink=record_transition,
                owns_environment=False,
                resource_identity=f"osworld-task:{task_id}",
            )
            environment = OSWorldEnvironment(client, protocol=self.protocol)
            built = agent_factory(environment)
            agent = await built if inspect.isawaitable(built) else built
            if not isinstance(agent, MiniAgent):
                raise TypeError("OSWorld agent_factory must return MiniAgent")
            try:
                agent_result = await agent.run(instruction)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                agent_error = f"{type(exc).__name__}: {exc}"
            finally:
                # This closes only the agent-facing adapter. The raw environment
                # remains alive for the external evaluator below.
                await environment.close()

            if self.settle_wait_seconds:
                await self.sleep(self.settle_wait_seconds)
            raw_score = await asyncio.to_thread(self.desktop_environment.evaluate)
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
            ):
                raise RuntimeError("OSWorld evaluator returned a non-finite score")
            score = float(raw_score)
            _atomic_write(output / "result.txt", f"{score}\n".encode("utf-8"))
            _atomic_write(
                output / "result.json",
                (
                    json.dumps(
                    {
                        "task_id": task_id,
                        "score": score,
                        "agent_answer": agent_result.answer if agent_result else None,
                        "agent_steps": agent_result.steps if agent_result else None,
                        "agent_error": agent_error,
                        "source_revision": OSWORLD_REVISION,
                        "verifier_exposed_to_agent": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                    + "\n"
                ).encode("utf-8"),
            )
            run_status = "completed"
        except asyncio.CancelledError:
            run_status = "cancelled"
            raise
        except BaseException:
            run_status = "failed"
            raise
        finally:
            cleanup_errors: list[BaseException] = []
            try:
                if environment is not None:
                    await environment.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                if recording_started and controller is not None:
                    await _call_optional(
                        controller, "end_recording", str(recording_path)
                    )
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                await _call_optional(self.desktop_environment, "close")
            except BaseException as exc:
                cleanup_errors.append(exc)
            _atomic_write(
                output / "lifecycle.json",
                (
                    json.dumps(
                        {
                            "status": run_status,
                            "cleanup_errors": [
                                {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                }
                                for error in cleanup_errors
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            if cleanup_errors and run_status == "completed":
                raise RuntimeError(
                    "; ".join(
                        f"{type(error).__name__}: {error}" for error in cleanup_errors
                    )
                ) from cleanup_errors[0]

        return OSWorldTaskResult(
            task_id=task_id,
            score=score,
            output_directory=output,
            agent_result=agent_result,
            agent_error=agent_error,
        )


__all__ = ["AgentFactory", "OSWorldTaskResult", "OSWorldTaskRunner"]
