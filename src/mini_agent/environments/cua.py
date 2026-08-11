from __future__ import annotations

import asyncio
import base64
import inspect
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import httpx

from scaffoldlab.environments.base import ToolExecution

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import BaseEnvironment


CUA_SPEED_RUN_REVISION = "7230223cbc57df68331cad32889adf01f3601651"
OSWORLD_REVISION = "091f5ef1d5544bc74953c77875d5feb5bed30108"


@dataclass(frozen=True)
class ComputerObservation:
    png: bytes
    meta: Mapping[str, Any] = field(default_factory=dict)


class ComputerClient(Protocol):
    async def observe(self) -> ComputerObservation: ...

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]: ...

    async def done(self) -> None: ...

    async def close(self) -> None: ...


class CUASpeedRunClient:
    """Async client for the public cua-speed-run observe/step/done gateway."""

    def __init__(self, env_url: str, *, timeout_seconds: float = 120.0) -> None:
        self.env_url = env_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def observe(self) -> ComputerObservation:
        response = await self._client.get(f"{self.env_url}/observe")
        response.raise_for_status()
        payload = response.json()
        return ComputerObservation(
            png=base64.b64decode(payload["png_b64"]),
            meta=dict(payload.get("meta", {})),
        )

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        response = await self._client.post(
            f"{self.env_url}/step", json={"actions": actions}
        )
        response.raise_for_status()
        payload = response.json()
        return dict(payload) if isinstance(payload, Mapping) else {}

    async def done(self) -> None:
        response = await self._client.post(f"{self.env_url}/done")
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()

    def provenance(self) -> Mapping[str, Any]:
        return {"client": "cua_speed_run_http", "env_url": self.env_url}


class OSWorldClient:
    """Bridge a live OSWorld DesktopEnv without exposing its evaluator."""

    def __init__(
        self,
        environment: Any,
        initial_observation: Mapping[str, Any],
        *,
        action_encoder: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.environment = environment
        self._observation = dict(initial_observation)
        self.action_encoder = action_encoder or self._default_encoder

    @staticmethod
    def _default_encoder(action: Mapping[str, Any]) -> Any:
        script = action.get("script")
        if not isinstance(script, str) or not script:
            raise ProtocolError("OSWorld action requires a non-empty script")
        statements = [line.strip() for line in script.splitlines() if line.strip()]
        if not statements or any(
            not line.startswith(("pyautogui.", "time.sleep(")) for line in statements
        ):
            raise ProtocolError(
                "OSWorld script bridge permits only pyautogui actions and time.sleep"
            )
        return script

    async def observe(self) -> ComputerObservation:
        screenshot = self._observation.get("screenshot")
        if not isinstance(screenshot, bytes):
            raise ProtocolError("OSWorld observation requires screenshot bytes")
        return ComputerObservation(png=screenshot, meta={"source": "osworld"})

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        info: Mapping[str, Any] = {}
        for action in actions:
            result = await asyncio.to_thread(
                self.environment.step, self.action_encoder(action)
            )
            if not isinstance(result, tuple) or len(result) < 4:
                raise ProtocolError("OSWorld env.step returned an invalid result")
            observation, _, _, raw_info = result[:4]
            if not isinstance(observation, Mapping):
                raise ProtocolError("OSWorld env.step returned an invalid observation")
            self._observation = dict(observation)
            info = dict(raw_info) if isinstance(raw_info, Mapping) else {}
        return info

    async def done(self) -> None:
        # The outer OSWorld runner owns termination and evaluation.
        return None

    async def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def provenance(self) -> Mapping[str, Any]:
        return {"client": "osworld_live_environment", "verifier_exposed": False}


class CUAEnvironment(BaseEnvironment):
    """One batched computer tool over an observe/step/done client."""

    def __init__(
        self,
        client: ComputerClient,
        *,
        benchmark: str = "cua-speed-run",
        allow_scripts: bool = False,
    ) -> None:
        self.client = client
        self.benchmark = benchmark
        self.allow_scripts = allow_scripts
        self._last_observation: ComputerObservation | None = None
        self._finished = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                name="computer",
                description=(
                    "Execute an ordered batch of gym-anything mouse, keyboard, or "
                    "wait actions, then return a screenshot."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "object"},
                            "minItems": 1,
                        }
                    },
                    "required": ["actions"],
                    "additionalProperties": False,
                },
            ),
        )

    def _validate_action(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise ProtocolError("computer actions must be objects")
        forbidden = {"reset", "snapshot", "shell", "exec", "verify"}.intersection(action)
        if forbidden:
            raise ProtocolError(f"computer action contains forbidden keys {sorted(forbidden)}")
        allowed = {"mouse", "keyboard", "action"}
        if self.allow_scripts:
            allowed.add("script")
        if not set(action).intersection(allowed):
            raise ProtocolError("computer action must be mouse, keyboard, or wait")
        if "action" in action and action.get("action") != "wait":
            raise ProtocolError("the only direct computer action is wait")

        def validate_coordinates(value: Any) -> None:
            if isinstance(value, Mapping):
                for child in value.values():
                    validate_coordinates(child)
            elif isinstance(value, list):
                if len(value) == 2 and all(
                    isinstance(number, (int, float)) and not isinstance(number, bool)
                    for number in value
                ):
                    if not all(math.isfinite(float(number)) and number >= 0 for number in value):
                        raise ProtocolError("computer coordinates must be finite and non-negative")
                else:
                    for child in value:
                        validate_coordinates(child)

        validate_coordinates(action)
        return dict(action)

    @staticmethod
    def _execution(observation: ComputerObservation, info: Mapping[str, Any]) -> ToolExecution:
        image = "data:image/png;base64," + base64.b64encode(observation.png).decode("ascii")
        return ToolExecution(
            output=json.dumps(
                {"environment": dict(info), "observation": dict(observation.meta)},
                sort_keys=True,
                default=str,
            ),
            image_data_url=image,
            metadata={"screenshot_bytes": len(observation.png)},
        )

    async def initial_observation(self) -> ToolExecution:
        self._last_observation = await self.client.observe()
        return self._execution(self._last_observation, {})

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "computer":
            raise ProtocolError(f"unsupported CUA tool {action.name!r}")
        raw_actions = action.arguments.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ProtocolError("computer requires a non-empty actions list")
        actions = [self._validate_action(item) for item in raw_actions]
        info = await self.client.step(actions)
        self._last_observation = await self.client.observe()
        return self._execution(self._last_observation, info)

    async def finish(self) -> None:
        if not self._finished:
            self._finished = True
            await self.client.done()

    async def close(self) -> None:
        await self.client.close()

    def provenance(self) -> dict[str, object]:
        revision = (
            OSWORLD_REVISION if self.benchmark == "osworld" else CUA_SPEED_RUN_REVISION
        )
        client_provenance = getattr(self.client, "provenance", None)
        return {
            "application": "cua",
            "benchmark": self.benchmark,
            "source_revision": revision,
            "agent_can_access_verifier": False,
            "agent_can_reset_snapshot_or_shell": False,
            "script_actions": self.allow_scripts,
            "client": (
                dict(client_provenance()) if client_provenance is not None else {}
            ),
        }


class OSWorldEnvironment(CUAEnvironment):
    def __init__(self, client: OSWorldClient) -> None:
        super().__init__(client, benchmark="osworld", allow_scripts=True)
