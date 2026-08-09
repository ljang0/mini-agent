from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import ToolEnvironment, ToolExecution
from .browser import PlaywrightBrowserDriver
from .swe import _object_schema


class ComputerDriver(ABC):
    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @abstractmethod
    async def execute_action(self, action: Mapping[str, Any]) -> None: ...

    @abstractmethod
    async def screenshot(self) -> bytes: ...

    async def close(self) -> None: ...


def _normalize_key(key: str) -> str:
    names = {
        "ENTER": "Enter",
        "RETURN": "Enter",
        "ESC": "Escape",
        "ESCAPE": "Escape",
        "TAB": "Tab",
        "SPACE": "Space",
        "BACKSPACE": "Backspace",
        "DELETE": "Delete",
        "DEL": "Delete",
        "HOME": "Home",
        "END": "End",
        "PAGEUP": "PageUp",
        "PAGEDOWN": "PageDown",
        "UP": "ArrowUp",
        "DOWN": "ArrowDown",
        "LEFT": "ArrowLeft",
        "RIGHT": "ArrowRight",
        "ARROWUP": "ArrowUp",
        "ARROWDOWN": "ArrowDown",
        "ARROWLEFT": "ArrowLeft",
        "ARROWRIGHT": "ArrowRight",
        "CTRL": "Control",
        "CONTROL": "Control",
        "SHIFT": "Shift",
        "ALT": "Alt",
        "OPTION": "Alt",
        "CMD": "Meta",
        "COMMAND": "Meta",
        "META": "Meta",
    }
    return names.get(key.upper(), key)


def _key_list(value: Any, *, field: str, required: bool = False) -> list[str]:
    if value is None:
        if required:
            raise ProtocolError(f"computer action requires {field}")
        return []
    if not isinstance(value, list) or not all(
        isinstance(key, str) and key.strip() for key in value
    ):
        raise ProtocolError(f"computer action {field} must be a string list")
    if required and not value:
        raise ProtocolError(f"computer action requires non-empty {field}")
    return [_normalize_key(key.strip()) for key in value]


def _shortcut_keys(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"computer action {field} must be a non-empty string")
    keys = [part.strip() for part in value.split("+")]
    if any(not key for key in keys):
        raise ProtocolError(f"computer action {field} contains an empty key")
    return [_normalize_key(key) for key in keys]


@asynccontextmanager
async def _held_keys(keyboard: Any, keys: Sequence[str]) -> AsyncIterator[None]:
    pressed: list[str] = []
    try:
        for key in keys:
            await keyboard.down(key)
            pressed.append(key)
        yield
    finally:
        for key in reversed(pressed):
            await keyboard.up(key)


class PlaywrightComputerDriver(ComputerDriver):
    """Pixel-action driver for a browser viewport, not a full desktop VM."""

    def __init__(self, browser: PlaywrightBrowserDriver) -> None:
        self.browser = browser

    @property
    def width(self) -> int:
        return self.browser.viewport_width

    @property
    def height(self) -> int:
        return self.browser.viewport_height

    def _coordinate(self, action: Mapping[str, Any]) -> tuple[float, float]:
        coordinate = action.get("coordinate")
        if isinstance(coordinate, list) and len(coordinate) == 2:
            x, y = coordinate
        else:
            x, y = action.get("x"), action.get("y")
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise ProtocolError("computer action x coordinate must be numeric")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise ProtocolError("computer action y coordinate must be numeric")
        return float(x), float(y)

    def _mouse_modifiers(
        self, action: Mapping[str, Any], action_type: str
    ) -> list[str]:
        if "type" in action:
            return _key_list(action.get("keys"), field="keys[]")

        raw_key = action.get("key")
        if raw_key is None and action_type == "scroll":
            # Anthropic's scroll modifier occupies the legacy `text` field.
            raw_key = action.get("text")
        if raw_key is None:
            return []
        return _shortcut_keys(raw_key, field="key")

    async def execute_action(self, action: Mapping[str, Any]) -> None:
        page = self.browser.page
        action_type = action.get("type") or action.get("action")
        if action_type in {"screenshot"}:
            return
        if action_type in {"click", "left_click", "right_click", "middle_click"}:
            x, y = self._coordinate(action)
            button = action.get("button")
            if action_type == "right_click":
                button = "right"
            elif action_type == "middle_click":
                button = "middle"
            button = {"wheel": "middle"}.get(str(button), button or "left")
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                await page.mouse.click(x, y, button=button)
            return
        if action_type in {"double_click", "doubleClick"}:
            x, y = self._coordinate(action)
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                await page.mouse.dblclick(x, y)
            return
        if action_type == "triple_click":
            x, y = self._coordinate(action)
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                await page.mouse.click(x, y, button="left", click_count=3)
            return
        if action_type in {"move", "mouse_move"}:
            x, y = self._coordinate(action)
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                await page.mouse.move(x, y)
            return
        if action_type in {"type", "type_text"}:
            text = action.get("text")
            if not isinstance(text, str):
                raise ProtocolError("computer type action requires string text")
            await page.keyboard.type(text)
            return
        if action_type == "keypress":
            keys = _key_list(action.get("keys"), field="keys[]", required=True)
            for key in keys:
                await page.keyboard.press(key)
            return
        if action_type == "key":
            raw_key = action.get("key")
            if raw_key is None:
                raw_key = action.get("text")
            shortcut = "+".join(_shortcut_keys(raw_key, field="key"))
            await page.keyboard.press(shortcut)
            return
        if action_type == "scroll":
            delta = action.get("scroll_amount")
            if delta is not None:
                if not isinstance(delta, (int, float)) or isinstance(delta, bool):
                    raise ProtocolError("scroll_amount must be numeric")
                direction = action.get("scroll_direction", "down")
                dx = (
                    -delta
                    if direction == "left"
                    else delta
                    if direction == "right"
                    else 0
                )
                dy = (
                    -delta if direction == "up" else delta if direction == "down" else 0
                )
            else:
                dx = action.get("scroll_x", 0)
                dy = action.get("scroll_y", 0)
            if (
                not isinstance(dx, (int, float))
                or isinstance(dx, bool)
                or not isinstance(dy, (int, float))
                or isinstance(dy, bool)
            ):
                raise ProtocolError("scroll deltas must be numeric")
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                coordinate = action.get("coordinate")
                if coordinate is not None or "x" in action or "y" in action:
                    x, y = self._coordinate(action)
                    await page.mouse.move(x, y)
                await page.mouse.wheel(float(dx), float(dy))
            return
        if action_type in {"drag", "left_click_drag"}:
            raw_path = action.get("path")
            if (
                raw_path is None
                and action.get("start_coordinate")
                and action.get("coordinate")
            ):
                raw_path = [action["start_coordinate"], action["coordinate"]]
            if not isinstance(raw_path, list) or len(raw_path) < 2:
                raise ProtocolError("drag requires at least two path coordinates")
            points: list[tuple[float, float]] = []
            for point in raw_path:
                if isinstance(point, list) and len(point) == 2:
                    point_x, point_y = point
                elif isinstance(point, dict):
                    point_x, point_y = point.get("x"), point.get("y")
                else:
                    raise ProtocolError("invalid drag path coordinate")
                if not isinstance(point_x, (int, float)) or not isinstance(
                    point_y, (int, float)
                ):
                    raise ProtocolError("drag path coordinates must be numeric")
                points.append((float(point_x), float(point_y)))
            modifiers = self._mouse_modifiers(action, str(action_type))
            async with _held_keys(page.keyboard, modifiers):
                await page.mouse.move(*points[0])
                await page.mouse.down()
                try:
                    for point in points[1:]:
                        await page.mouse.move(*point)
                finally:
                    await page.mouse.up()
            return
        if action_type in {"left_mouse_down", "left_mouse_up"}:
            method = (
                page.mouse.down if action_type == "left_mouse_down" else page.mouse.up
            )
            await method(button="left")
            return
        if action_type == "hold_key":
            raw_key = action.get("key")
            if raw_key is None:
                raw_key = action.get("text")
            keys = _shortcut_keys(raw_key, field="key")
            duration = action.get("duration")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration < 0
                or duration > 100
            ):
                raise ProtocolError(
                    "hold_key duration must be between 0 and 100 seconds"
                )
            async with _held_keys(page.keyboard, keys):
                await asyncio.sleep(float(duration))
            return
        if action_type == "wait":
            duration = action.get("duration", 2.0)
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration < 0
                or duration > 100
            ):
                raise ProtocolError("wait duration must be between 0 and 100 seconds")
            await asyncio.sleep(float(duration))
            return
        raise ProtocolError(f"unsupported computer action {action_type!r}")

    async def screenshot(self) -> bytes:
        return await self.browser.screenshot()

    async def close(self) -> None:
        await self.browser.close()


class ComputerEnvironment(ToolEnvironment):
    """Executes the public OpenAI/Anthropic CUA action contracts."""

    def __init__(self, driver: ComputerDriver, *, protocol: str = "auto") -> None:
        if protocol not in {"auto", "generic"}:
            raise ValueError("computer protocol must be 'auto' or 'generic'")
        self.driver = driver
        self.protocol = protocol
        self._calls = 0
        self._actions = 0

    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        if self.protocol == "auto" and provider_family == "openai":
            return (ToolDefinition(name="computer", kind="openai_computer"),)
        if self.protocol == "auto" and provider_family == "anthropic":
            return (
                ToolDefinition(
                    name="computer",
                    kind="anthropic_computer_20251124",
                    provider_options={
                        "display_width_px": self.driver.width,
                        "display_height_px": self.driver.height,
                    },
                ),
            )
        return (
            ToolDefinition(
                name="computer",
                description=(
                    "Execute one or more ordered mouse/keyboard/screenshot actions "
                    "in the isolated computer environment."
                ),
                input_schema=_object_schema(
                    {
                        "actions": {
                            "type": "array",
                            "items": {"type": "object"},
                            "minItems": 1,
                        }
                    },
                    ("actions",),
                ),
            ),
        )

    async def execute(self, call: ToolCall) -> ToolExecution:
        self._calls += 1
        try:
            raw_actions = call.arguments.get("actions")
            if raw_actions is None:
                raw_actions = [dict(call.arguments)]
            if (
                not isinstance(raw_actions, list)
                or not raw_actions
                or not all(isinstance(action, Mapping) for action in raw_actions)
            ):
                raise ProtocolError("computer actions must be a non-empty object list")
            for action in raw_actions:
                await self.driver.execute_action(action)
                self._actions += 1
            image = await self.driver.screenshot()
            image_data_url = "data:image/png;base64," + base64.b64encode(image).decode(
                "ascii"
            )
            return ToolExecution(
                output=json.dumps(
                    {"actions_executed": len(raw_actions), "screenshot": True},
                    sort_keys=True,
                ),
                image_data_url=image_data_url,
                native_output={"type": "computer_screenshot", "detail": "original"},
            )
        except (ProtocolError, ValueError, OSError, RuntimeError) as exc:
            output = f"{type(exc).__name__}: {exc}"
            try:
                image = await self.driver.screenshot()
            except (OSError, RuntimeError):
                return ToolExecution(output=output, is_error=True)
            image_data_url = "data:image/png;base64," + base64.b64encode(image).decode(
                "ascii"
            )
            return ToolExecution(
                output=output,
                is_error=True,
                image_data_url=image_data_url,
                native_output={"type": "computer_screenshot", "detail": "original"},
                metadata={"recovery_screenshot": True},
            )

    async def summary(self) -> Mapping[str, Any]:
        return {
            "type": "computer",
            "surface": "browser_viewport"
            if isinstance(self.driver, PlaywrightComputerDriver)
            else "custom_driver",
            "protocol": self.protocol,
            "width": self.driver.width,
            "height": self.driver.height,
            "tool_calls": self._calls,
            "actions_executed": self._actions,
        }

    async def close(self) -> None:
        await self.driver.close()
