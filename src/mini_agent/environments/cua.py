"""Canonical computer tool and narrow CUA/OSWorld gateways."""

from __future__ import annotations

import base64
import inspect
import json
import math
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    TypeGuard,
)

from ..types import (
    InfrastructureError,
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    _json_mapping,
    _require_bool,
    _require_finite_number,
    _require_int,
    _require_mapping,
    _require_str,
)
from .base import BaseEnvironment, complete_in_thread, raise_lifecycle_errors


OSWORLD_V1_REVISION = "091f5ef1d5544bc74953c77875d5feb5bed30108"
OSWORLD_V2_REVISION = "v2026.06.24"
OSWORLD_V2_COMMIT = "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6"
OSWORLD_REVISION = OSWORLD_V1_REVISION
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SCREENSHOT_BYTES = 16 * 1024 * 1024
MAX_SCREENSHOT_WIDTH = 8192
MAX_SCREENSHOT_HEIGHT = 8192
MAX_SCREENSHOT_PIXELS = 16 * 1024 * 1024
_BUTTONS = frozenset({"left", "middle", "right"})
_NO_OWNER_EXPORT = "live state export requires an external benchmark session owner"
_NO_OWNER_ADOPT = "live state adoption requires an external benchmark session owner"
_PNG_DEPTHS = {
    0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}
}
_ACTION_FIELDS = {
    "click": {"x", "y", "button", "clicks"},
    "move": {"x", "y", "duration"},
    "drag": {"x", "y", "to_x", "to_y", "button", "duration"},
    "scroll": {"dx", "dy"},
    "type": {"text"},
    "key": {"keys"},
    "key_down": {"key"},
    "key_up": {"key"},
    "wait": {"seconds"},
    "screenshot": set[str](),
    "fail": set[str](),
}
_ACTION_TYPES = frozenset(_ACTION_FIELDS)
_KEY_ALIASES = {
    "return": "enter",
    "escape": "esc",
    "control": "ctrl",
    "command": "win",
    "cmd": "win",
    "super": "win",
    "option": "alt",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
}


def _infra(prefix: str, detail: str) -> InfrastructureError:
    """Build a prefixed infrastructure error so guards stay one line each."""

    return InfrastructureError(f"{prefix} {detail}")


def _integer(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> TypeGuard[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _optional_callable(value: Any, label: str) -> None:
    if value is not None and not callable(value):
        raise ValueError(f"{label} must be callable or None")


def _optional_identity(value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("resource_identity must be non-empty or None")


@dataclass(frozen=True)
class ComputerObservation:
    png: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Derived once here: ``png`` is immutable ``bytes`` on a frozen dataclass,
    # so every later reader shares this result instead of walking the chunk
    # stream again on a multi-megabyte screenshot.
    dimensions: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", validate_png(self.png))
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "computer observation metadata"),
        )


@dataclass
class OSWorldLiveState:
    """A single-claim pointer to a benchmark-owned live desktop branch."""

    environment: Any
    observation: Mapping[str, Any]
    resource_identity: str
    steps: int = 0
    done: bool = False
    _claimed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_mapping(self.observation, "OSWorld live observation")
        screenshot = self.observation.get("screenshot")
        if not isinstance(screenshot, bytes):
            raise ValueError("OSWorld live observation requires screenshot bytes")
        validate_png(screenshot)
        _require_str(self.resource_identity, "OSWorld live resource identity")
        _require_int(self.steps, "OSWorld live step count", minimum=0)
        _require_bool(self.done, "OSWorld live done flag")

    def claim(self) -> tuple[Any, Mapping[str, Any], str, int, bool]:
        if self._claimed:
            raise ProtocolError("OSWorld live state was already adopted")
        self._claimed = True
        return (
            self.environment,
            dict(self.observation),
            self.resource_identity,
            self.steps,
            self.done,
        )


class ComputerClient(Protocol):
    async def observe(self) -> ComputerObservation: ...

    async def step(self, actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

    async def done(self) -> None: ...

    async def close(self) -> None: ...


def validate_png(png: bytes) -> tuple[int, int]:
    """Validate the full PNG chunk stream and return IHDR dimensions."""

    if not isinstance(png, bytes) or not png.startswith(_PNG_SIGNATURE):
        raise _infra("computer observation", "is not a PNG")
    if len(png) > MAX_SCREENSHOT_BYTES:
        raise _infra("computer observation", "exceeds the screenshot byte limit")
    offset = len(_PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    first = True
    ended = False
    has_image_data = False
    while offset < len(png):
        if offset + 12 > len(png):
            raise _infra("computer observation", "contains a truncated PNG chunk")
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        start = offset + 8
        finish = start + length
        crc_finish = finish + 4
        if crc_finish > len(png):
            raise _infra("computer observation", "contains a truncated PNG chunk")
        expected = struct.unpack(">I", png[finish:crc_finish])[0]
        observed = zlib.crc32(kind)
        observed = zlib.crc32(png[start:finish], observed) & 0xFFFFFFFF
        if observed != expected:
            raise _infra("computer observation", "contains an invalid PNG checksum")
        if first and kind != b"IHDR":
            raise _infra("computer observation", "PNG does not start with IHDR")
        if kind == b"IHDR":
            if not first or length != 13:
                raise _infra("computer observation", "contains an invalid IHDR")
            ihdr = struct.unpack(">IIBBBBB", png[start:finish])
            width, height, bit_depth, color_type = ihdr[:4]
            compression, filter_method, interlace = ihdr[4:]
            if width < 1 or height < 1:
                raise _infra("computer observation", "has invalid dimensions")
            if (
                width > MAX_SCREENSHOT_WIDTH
                or height > MAX_SCREENSHOT_HEIGHT
                or width * height > MAX_SCREENSHOT_PIXELS
            ):
                raise _infra("computer observation", "exceeds the dimension limit")
            if (
                bit_depth not in _PNG_DEPTHS.get(color_type, set())
                or compression != 0
                or filter_method != 0
                or interlace not in {0, 1}
            ):
                raise _infra("computer observation", "contains an invalid IHDR")
            dimensions = (width, height)
        if kind == b"IDAT":
            has_image_data = True
        if kind == b"IEND":
            if length or crc_finish != len(png):
                raise _infra("computer observation", "contains an invalid IEND")
            ended = True
        first = False
        offset = crc_finish
    if dimensions is None or not has_image_data or not ended:
        raise _infra("computer observation", "is an incomplete PNG")
    return dimensions


def computer_action_schema(
    *,
    allow_fail: bool = False,
    allow_duration: bool = False,
) -> Mapping[str, Any]:
    """Provider-neutral schema; semantic constraints are checked at execution."""

    if not isinstance(allow_fail, bool) or not isinstance(allow_duration, bool):
        raise ValueError("computer schema options must be boolean")
    action_types = _ACTION_TYPES if allow_fail else _ACTION_TYPES - {"fail"}
    fields: dict[str, Any] = {
        "type": {"type": "string", "enum": sorted(action_types)},
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "to_x": {"type": "integer", "minimum": 0},
        "to_y": {"type": "integer", "minimum": 0},
        "button": {"type": "string", "enum": sorted(_BUTTONS)},
        "clicks": {"type": "integer", "minimum": 1, "maximum": 3},
        "dx": {"type": "integer"},
        "dy": {"type": "integer"},
        "text": {"type": "string", "maxLength": 10000},
        "keys": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {"type": "string"},
        },
        "key": {"type": "string"},
        "seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 30},
        "duration": {"type": "number", "minimum": 0, "maximum": 10},
    }
    if not allow_duration:
        del fields["duration"]
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {
                    "type": "object",
                    "properties": fields,
                    "required": ["type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def validate_computer_actions(
    value: Any,
    dimensions: tuple[int, int],
    *,
    allow_fail: bool = False,
    allow_duration: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(allow_fail, bool) or not isinstance(allow_duration, bool):
        raise ValueError("computer action options must be boolean")
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise InvalidAction("computer actions must contain 1..32 actions")
    actions: list[Mapping[str, Any]] = []
    width, height = dimensions
    for raw in value:
        if not isinstance(raw, Mapping):
            raise InvalidAction("each computer action must be an object")
        kind = raw.get("type")
        if not isinstance(kind, str) or kind not in _ACTION_TYPES:
            raise InvalidAction(f"unsupported computer action {kind!r}")
        if kind == "fail" and not allow_fail:
            raise InvalidAction("computer fail is supported only by OSWorld")
        if "duration" in raw and not allow_duration:
            raise InvalidAction("computer duration is supported only by OSWorld")
        extra = set(raw).difference(_ACTION_FIELDS[str(kind)] | {"type"})
        if extra:
            raise InvalidAction(f"unexpected {kind} fields: {sorted(extra)}")
        action = dict(raw)
        if kind in {"click", "move", "drag"}:
            _coordinate(action, "x", width)
            _coordinate(action, "y", height)
        if kind == "drag":
            _coordinate(action, "to_x", width)
            _coordinate(action, "to_y", height)
        if kind in {"click", "drag"}:
            button = action.setdefault("button", "left")
            if not isinstance(button, str) or button not in _BUTTONS:
                raise InvalidAction("computer button must be left, middle, or right")
        if kind == "click":
            clicks = action.get("clicks", 1)
            if not _integer(clicks) or not 1 <= clicks <= 3:
                raise InvalidAction("computer clicks must be an integer from 1 to 3")
            action["clicks"] = clicks
        if kind == "scroll":
            for name in ("dx", "dy"):
                action[name] = _require_int(
                    action.get(name, 0), f"computer scroll {name}", error=InvalidAction
                )
            if action["dx"] == 0 and action["dy"] == 0:
                raise InvalidAction("computer scroll requires non-zero dx or dy")
        if kind == "type":
            text = action.get("text")
            if not isinstance(text, str) or len(text) > 10_000:
                raise InvalidAction(
                    "computer type text must be at most 10000 characters"
                )
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise InvalidAction(
                    "computer type text must contain valid Unicode scalar values"
                ) from exc
        if kind == "key":
            keys = action.get("keys")
            if not isinstance(keys, list) or not 1 <= len(keys) <= 32:
                raise InvalidAction("computer key requires 1..32 keys")
            action["keys"] = [_key(item) for item in keys]
        if kind in {"key_down", "key_up"}:
            action["key"] = _key(action.get("key"))
        if kind == "wait":
            seconds = action.get("seconds")
            if not _finite(seconds) or not 0 < seconds <= 30:
                raise InvalidAction("computer wait seconds must be in (0, 30]")
            action["seconds"] = float(seconds)
        if "duration" in action:
            duration = action["duration"]
            if not _finite(duration) or not 0 <= duration <= 10:
                raise InvalidAction("computer duration must be in [0, 10]")
            action["duration"] = float(duration)
        actions.append(action)
    if any(action["type"] == "fail" for action in actions) and len(actions) != 1:
        raise InvalidAction("OSWorld fail must be the only action in its batch")
    return tuple(actions)


def _coordinate(action: Mapping[str, Any], name: str, limit: int) -> None:
    value = action.get(name)
    if not _integer(value) or not 0 <= value < limit:
        raise InvalidAction(f"computer {name} must be an on-screen integer")


def _key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 32:
        raise InvalidAction("computer key names must be non-empty and at most 32 chars")
    compact = value.strip().casefold().replace(" ", "")
    compact = _KEY_ALIASES.get(compact, compact)
    if not all(item.isalnum() or item in {"_", "+", "-"} for item in compact):
        raise InvalidAction(f"invalid computer key {value!r}")
    return compact


class OSWorldClient:
    """Wrap DesktopEnv while keeping reset and evaluate outside the agent plane."""

    def __init__(
        self,
        environment: Any,
        initial_observation: Mapping[str, Any],
        *,
        transition_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        owns_environment: bool = True,
        resource_identity: str | None = None,
        pause_seconds: float = 0,
    ) -> None:
        if not callable(getattr(environment, "step", None)):
            raise ValueError("OSWorld environment must expose step")
        _require_mapping(initial_observation, "OSWorld initial observation")
        _optional_callable(transition_sink, "transition_sink")
        if not isinstance(owns_environment, bool):
            raise ValueError("owns_environment must be boolean")
        _optional_identity(resource_identity)
        self.environment = environment
        self.observation = dict(initial_observation)
        self.transition_sink = transition_sink
        self.owns_environment = owns_environment
        self._identity = resource_identity or f"osworld-desktop:{uuid.uuid4().hex}"
        self._steps = 0
        self._done = False
        self.pause_seconds = _require_finite_number(
            pause_seconds, "OSWorld pause_seconds", minimum=0
        )

    async def observe(self) -> ComputerObservation:
        png = self.observation.get("screenshot")
        if not isinstance(png, bytes):
            raise _infra("OSWorld", "observation requires screenshot bytes")
        try:
            width, height = validate_png(png)
        except InfrastructureError as exc:
            raise _infra("OSWorld", "returned an invalid PNG") from exc
        return ComputerObservation(png, {"width": width, "height": height})

    async def step(self, actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        latest: dict[str, Any] = {}
        for action in actions:
            if action["type"] == "screenshot":
                getter = getattr(self.environment, "_get_obs", None)
                if getter is not None:
                    if not callable(getter):
                        raise _infra("OSWorld", "_get_obs must be callable")
                    observed = await complete_in_thread(getter)
                    if not isinstance(observed, Mapping):
                        raise _infra(
                            "OSWorld", "_get_obs returned an invalid observation"
                        )
                    self.observation = dict(observed)
                continue
            encoded = encode_osworld_action(action)
            result = await complete_in_thread(
                self.environment.step, encoded, self.pause_seconds
            )
            if not isinstance(result, tuple) or len(result) < 4:
                raise _infra("OSWorld", "step returned an invalid result")
            observation, reward, done, info = result[:4]
            if not isinstance(observation, Mapping):
                raise _infra("OSWorld", "step returned an invalid observation")
            if not isinstance(done, bool) or not isinstance(info, Mapping):
                raise _infra("OSWorld", "step returned invalid done or info values")
            self.observation = dict(observation)
            self._steps += 1
            self._done = done
            latest = {
                "step": self._steps,
                "reward": reward,
                "done": done,
                "info": dict(info),
            }
            if self.transition_sink is not None:
                emitted = self.transition_sink(
                    {
                        **latest,
                        "action": dict(action),
                        "encoded_action": encoded,
                        "observation": dict(self.observation),
                    }
                )
                if inspect.isawaitable(emitted):
                    await emitted
            if done:
                break
        return latest

    @property
    def episode_done(self) -> bool:
        return self._done

    async def done(self) -> None:
        return None

    async def close(self) -> None:
        if self.owns_environment and callable(getattr(self.environment, "close", None)):
            close = self.environment.close
            if inspect.iscoroutinefunction(close):
                await close()
            else:
                await complete_in_thread(close)

    async def export_state(self) -> OSWorldLiveState:
        if self.owns_environment:
            raise NotImplementedError(_NO_OWNER_EXPORT)
        return OSWorldLiveState(
            environment=self.environment,
            observation=dict(self.observation),
            resource_identity=self._identity,
            steps=self._steps,
            done=self._done,
        )

    async def adopt_state(self, state: Any) -> ComputerObservation:
        if self.owns_environment:
            raise NotImplementedError(_NO_OWNER_ADOPT)
        if not isinstance(state, OSWorldLiveState):
            raise ProtocolError("OSWorld can adopt only OSWorld live state")
        environment, observation, identity, steps, done = state.claim()
        self.environment = environment
        self.observation = dict(observation)
        self._identity = identity
        self._steps = steps
        self._done = done
        return await self.observe()

    def resource_identity(self) -> str:
        return self._identity

    def provenance(self) -> Mapping[str, Any]:
        return {"client": "osworld"}


def encode_osworld_action(action: Mapping[str, Any]) -> str:
    kind = action["type"]
    if kind == "fail":
        return "FAIL"
    if kind == "click":
        return (
            f"pyautogui.click({action['x']}, {action['y']}, "
            f"clicks={action['clicks']}, button={action['button']!r})"
        )
    if kind == "move":
        return (
            f"pyautogui.moveTo({action['x']}, {action['y']}, "
            f"duration={action.get('duration', 0.2)!r})"
        )
    if kind == "drag":
        return (
            f"pyautogui.moveTo({action['x']}, {action['y']})\n"
            f"pyautogui.dragTo({action['to_x']}, {action['to_y']}, "
            f"duration={action.get('duration', 0.5)!r}, button={action['button']!r})"
        )
    if kind == "scroll":
        lines = []
        if action["dy"]:
            lines.append(f"pyautogui.scroll({-action['dy']!r})")
        if action["dx"]:
            lines.append(f"pyautogui.hscroll({action['dx']!r})")
        return "\n".join(lines)
    if kind == "type":
        text = str(action["text"])
        if text.isascii():
            return f"pyautogui.write({text!r}, interval=0.01)"
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return (
            "import base64, pyperclip\n"
            f"_text = base64.b64decode({encoded!r}).decode('utf-8')\n"
            "pyperclip.copy(_text)\n"
            "time.sleep(0.1)\n"
            "pyautogui.hotkey('ctrl', 'v')\n"
            "time.sleep(0.1)"
        )
    if kind == "key":
        keys = ", ".join(repr(key) for key in action["keys"])
        return f"pyautogui.hotkey({keys})"
    if kind in {"key_down", "key_up"}:
        method = "keyDown" if kind == "key_down" else "keyUp"
        return f"pyautogui.{method}({action['key']!r})"
    if kind == "wait":
        return f"time.sleep({action['seconds']!r})"
    raise InvalidAction(f"cannot encode OSWorld action {kind!r}")


class CUAEnvironment(BaseEnvironment):
    """Expose one generic, batched computer tool over a leased machine."""

    def __init__(
        self,
        client: ComputerClient,
        *,
        benchmark: str = "computer",
        allow_fail: bool = False,
        allow_duration: bool = False,
    ) -> None:
        for name in ("observe", "step", "done", "close"):
            if not callable(getattr(client, name, None)):
                raise ValueError(f"computer client must expose {name}")
        if not isinstance(allow_fail, bool) or not isinstance(allow_duration, bool):
            raise ValueError("computer environment options must be boolean")
        self.client = client
        self.benchmark = _require_str(benchmark, "computer benchmark")
        self.allow_fail = allow_fail
        self.allow_duration = allow_duration
        self._observation: ComputerObservation | None = None
        self._finished = False
        self._finish_attempted = False
        self._finish_error: BaseException | None = None
        self._closed = False
        self._episode_done = False

    def tools(self) -> Sequence[ToolDefinition]:
        fail_note = " Use fail alone only when the OSWorld task is infeasible."
        return (
            ToolDefinition(
                name="computer",
                description=(
                    "Execute an ordered batch of mouse/keyboard/wait actions using "
                    "native screenshot pixel coordinates."
                    + (fail_note if self.allow_fail else "")
                ),
                input_schema=computer_action_schema(
                    allow_fail=self.allow_fail, allow_duration=self.allow_duration
                ),
            ),
        )

    async def initial_observation(self) -> ToolExecution:
        if self._closed:
            raise RuntimeError("computer environment is closed")
        observation = await self.client.observe()
        if not isinstance(observation, ComputerObservation):
            raise _infra("computer client", "observe must return ComputerObservation")
        done = getattr(self.client, "episode_done", False)
        if not isinstance(done, bool):
            raise _infra("computer client", "episode_done must be a boolean")
        self._observation = observation
        self._episode_done = done
        return self._render(observation, executed=0, episode_done=done)

    async def execute(self, call: ToolCall) -> ToolExecution:
        if self._closed or self._finish_attempted:
            raise RuntimeError("computer environment is finished")
        if self._episode_done:
            raise InvalidAction(
                "computer episode is done; return a final answer without more actions"
            )
        if call.name != "computer":
            raise InvalidAction(f"unsupported computer tool {call.name!r}")
        if self._observation is None:
            raise ProtocolError("computer must be observed before acting")
        dimensions = self._observation.dimensions
        actions = validate_computer_actions(
            call.arguments.get("actions"),
            dimensions,
            allow_fail=self.allow_fail,
            allow_duration=self.allow_duration,
        )
        actionable = [action for action in actions if action["type"] != "screenshot"]
        result = await self.client.step(actions)
        if not isinstance(result, Mapping):
            raise _infra("computer client", "step must return an object")
        done = result.get("done", False)
        if not isinstance(done, bool):
            raise _infra("computer client", "step done must be a boolean")
        self._episode_done = done
        observation = await self.client.observe()
        if not isinstance(observation, ComputerObservation):
            raise _infra("computer client", "observe must return ComputerObservation")
        self._observation = observation
        return self._render(observation, executed=len(actionable), episode_done=done)

    def _render(
        self, observation: ComputerObservation, *, executed: int, episode_done: bool
    ) -> ToolExecution:
        width, height = observation.dimensions
        image = "data:image/png;base64," + base64.b64encode(observation.png).decode()
        size = {"width": width, "height": height}
        return ToolExecution(
            output=json.dumps(
                {**size, "executed": executed, "episode_done": episode_done}
            ),
            image_data_url=image,
            metadata={**size, "screenshot_bytes": len(observation.png)},
        )

    async def finish(self) -> None:
        if self._finished:
            return
        if self._finish_attempted:
            assert self._finish_error is not None
            raise self._finish_error
        self._finish_attempted = True
        try:
            await self.client.done()
        except BaseException as exc:
            self._finish_error = exc
            raise
        self._finished = True

    async def export_state(self) -> Any:
        hook = getattr(self.client, "export_state", None)
        return None if not callable(hook) else await hook()

    async def adopt_state(self, state: Any) -> None:
        if self._closed or self._finished:
            raise RuntimeError("computer environment is finished")
        hook = getattr(self.client, "adopt_state", None)
        if not callable(hook):
            raise NotImplementedError("computer client cannot adopt state")
        observation = await hook(state)
        if not isinstance(observation, ComputerObservation):
            raise InfrastructureError("computer state adoption returned no observation")
        done = getattr(self.client, "episode_done", False)
        if not isinstance(done, bool):
            raise _infra("computer client", "episode_done must be a boolean")
        self._observation = observation
        self._episode_done = done

    async def close(self) -> None:
        if self._closed:
            return
        operation_error: BaseException | None = None
        try:
            await self.finish()
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await self.client.close()
        except BaseException as exc:
            cleanup_error = exc
        else:
            self._closed = True
        raise_lifecycle_errors("computer finish", operation_error, cleanup_error)

    def resource_identity(self) -> str:
        hook = getattr(self.client, "resource_identity", None)
        if not callable(hook):
            return super().resource_identity()
        identity = hook()
        if not isinstance(identity, str) or not identity:
            raise RuntimeError("computer client resource_identity must be non-empty")
        return identity

    def provenance(self) -> Mapping[str, Any]:
        hook = getattr(self.client, "provenance", None)
        client: Mapping[str, Any] = {}
        if callable(hook):
            value = hook()
            if not isinstance(value, Mapping):
                raise RuntimeError("computer client provenance must be an object")
            client = value
        return {
            "environment": "computer",
            "benchmark": self.benchmark,
            "tool": "computer",
            "coordinates": "native_pixels",
            "client": dict(client),
        }


class OSWorldEnvironment(CUAEnvironment):
    def __init__(self, client: OSWorldClient, *, version: str = "v1") -> None:
        if version not in {"v1", "v2"}:
            raise ValueError("OSWorld version must be v1 or v2")
        super().__init__(
            client,
            benchmark=f"osworld-{version}",
            allow_fail=True,
            allow_duration=True,
        )


__all__ = [
    "CUAEnvironment",
    "ComputerClient",
    "ComputerObservation",
    "MAX_SCREENSHOT_BYTES",
    "MAX_SCREENSHOT_HEIGHT",
    "MAX_SCREENSHOT_PIXELS",
    "MAX_SCREENSHOT_WIDTH",
    "OSWORLD_REVISION",
    "OSWORLD_V1_REVISION",
    "OSWORLD_V2_REVISION",
    "OSWORLD_V2_COMMIT",
    "OSWorldClient",
    "OSWorldEnvironment",
    "OSWorldLiveState",
    "computer_action_schema",
    "complete_in_thread",
    "encode_osworld_action",
    "validate_computer_actions",
    "validate_png",
]
