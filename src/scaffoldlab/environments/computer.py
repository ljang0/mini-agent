from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

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
        "SPACE": "Space",
        "BACKSPACE": "Backspace",
        "DELETE": "Delete",
        "DEL": "Delete",
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
        "ALT": "Alt",
        "OPTION": "Alt",
        "CMD": "Meta",
        "COMMAND": "Meta",
        "META": "Meta",
    }
    return names.get(key.upper(), key)


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
            await page.mouse.click(x, y, button=button)
            return
        if action_type in {"double_click", "doubleClick"}:
            x, y = self._coordinate(action)
            await page.mouse.dblclick(x, y)
            return
        if action_type in {"move", "mouse_move"}:
            x, y = self._coordinate(action)
            await page.mouse.move(x, y)
            return
        if action_type in {"type", "type_text"}:
            text = action.get("text")
            if not isinstance(text, str):
                raise ProtocolError("computer type action requires string text")
            await page.keyboard.type(text)
            return
        if action_type in {"keypress", "key"}:
            keys = action.get("keys", action.get("text"))
            if isinstance(keys, str):
                keys = [item for item in keys.split("+") if item]
            if not isinstance(keys, list) or not all(
                isinstance(key, str) for key in keys
            ):
                raise ProtocolError("computer key action requires string keys")
            shortcut = "+".join(_normalize_key(key) for key in keys)
            await page.keyboard.press(shortcut)
            return
        if action_type == "scroll":
            coordinate = action.get("coordinate")
            if coordinate is not None or "x" in action or "y" in action:
                x, y = self._coordinate(action)
                await page.mouse.move(x, y)
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
            if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
                raise ProtocolError("scroll deltas must be numeric")
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
            await page.mouse.move(*points[0])
            await page.mouse.down()
            try:
                for point in points[1:]:
                    await page.mouse.move(*point)
            finally:
                await page.mouse.up()
            return
        if action_type == "wait":
            await asyncio.sleep(2.0)
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
