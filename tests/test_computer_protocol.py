from __future__ import annotations

import unittest
from typing import Any

from scaffoldlab.environments.computer import (
    ComputerEnvironment,
    PlaywrightComputerDriver,
)
from scaffoldlab.types import ProtocolError


class _FakeKeyboard:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    async def down(self, key: str) -> None:
        self.events.append(("keyboard.down", key))

    async def up(self, key: str) -> None:
        self.events.append(("keyboard.up", key))

    async def press(self, key: str) -> None:
        self.events.append(("keyboard.press", key))

    async def type(self, text: str) -> None:
        self.events.append(("keyboard.type", text))


class _FakeMouse:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.fail_click = False

    async def click(
        self,
        x: float,
        y: float,
        *,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        self.events.append(("mouse.click", x, y, button, click_count))
        if self.fail_click:
            raise RuntimeError("injected click failure")

    async def dblclick(self, x: float, y: float) -> None:
        self.events.append(("mouse.dblclick", x, y))

    async def move(self, x: float, y: float) -> None:
        self.events.append(("mouse.move", x, y))

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        self.events.append(("mouse.wheel", delta_x, delta_y))

    async def down(self, *, button: str = "left") -> None:
        self.events.append(("mouse.down", button))

    async def up(self, *, button: str = "left") -> None:
        self.events.append(("mouse.up", button))


class _FakePage:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.keyboard = _FakeKeyboard(self.events)
        self.mouse = _FakeMouse(self.events)


class _FakeBrowser:
    viewport_width = 1280
    viewport_height = 720

    def __init__(self) -> None:
        self.page = _FakePage()
        self.closed = False

    async def screenshot(self) -> bytes:
        return b"fake-png"

    async def close(self) -> None:
        self.closed = True


class ComputerProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.browser = _FakeBrowser()
        self.driver = PlaywrightComputerDriver(self.browser)  # type: ignore[arg-type]

    async def test_openai_keys_and_mouse_modifiers_follow_public_order(self) -> None:
        await self.driver.execute_action(
            {"type": "keypress", "keys": ["CTRL", "A", "RETURN"]}
        )
        await self.driver.execute_action(
            {
                "type": "click",
                "x": 10,
                "y": 20,
                "button": "left",
                "keys": ["SHIFT"],
            }
        )
        await self.driver.execute_action(
            {"type": "double_click", "x": 30, "y": 40, "keys": ["ALT"]}
        )
        await self.driver.execute_action(
            {
                "type": "drag",
                "path": [[1, 2], {"x": 3, "y": 4}],
                "keys": ["CTRL"],
            }
        )
        await self.driver.execute_action(
            {"type": "move", "x": 50, "y": 60, "keys": ["META"]}
        )
        await self.driver.execute_action(
            {
                "type": "scroll",
                "x": 70,
                "y": 80,
                "scroll_x": -5,
                "scroll_y": 15,
                "keys": ["SHIFT"],
            }
        )

        self.assertEqual(
            self.browser.page.events,
            [
                ("keyboard.press", "Control"),
                ("keyboard.press", "A"),
                ("keyboard.press", "Enter"),
                ("keyboard.down", "Shift"),
                ("mouse.click", 10.0, 20.0, "left", 1),
                ("keyboard.up", "Shift"),
                ("keyboard.down", "Alt"),
                ("mouse.dblclick", 30.0, 40.0),
                ("keyboard.up", "Alt"),
                ("keyboard.down", "Control"),
                ("mouse.move", 1.0, 2.0),
                ("mouse.down", "left"),
                ("mouse.move", 3.0, 4.0),
                ("mouse.up", "left"),
                ("keyboard.up", "Control"),
                ("keyboard.down", "Meta"),
                ("mouse.move", 50.0, 60.0),
                ("keyboard.up", "Meta"),
                ("keyboard.down", "Shift"),
                ("mouse.move", 70.0, 80.0),
                ("mouse.wheel", -5.0, 15.0),
                ("keyboard.up", "Shift"),
            ],
        )

    async def test_anthropic_current_keyboard_and_enhanced_actions(self) -> None:
        await self.driver.execute_action({"action": "key", "key": "CTRL+S"})
        await self.driver.execute_action({"action": "key", "text": "ALT+F4"})
        await self.driver.execute_action(
            {
                "action": "triple_click",
                "coordinate": [10, 20],
                "key": "SHIFT+CTRL",
            }
        )
        await self.driver.execute_action({"action": "left_mouse_down"})
        await self.driver.execute_action({"action": "left_mouse_up"})
        await self.driver.execute_action(
            {"action": "hold_key", "text": "ALT", "duration": 0}
        )
        await self.driver.execute_action(
            {
                "action": "scroll",
                "coordinate": [30, 40],
                "scroll_direction": "up",
                "scroll_amount": 2,
                "text": "CTRL",
            }
        )

        self.assertEqual(
            self.browser.page.events,
            [
                ("keyboard.press", "Control+S"),
                ("keyboard.press", "Alt+F4"),
                ("keyboard.down", "Shift"),
                ("keyboard.down", "Control"),
                ("mouse.click", 10.0, 20.0, "left", 3),
                ("keyboard.up", "Control"),
                ("keyboard.up", "Shift"),
                ("mouse.down", "left"),
                ("mouse.up", "left"),
                ("keyboard.down", "Alt"),
                ("keyboard.up", "Alt"),
                ("keyboard.down", "Control"),
                ("mouse.move", 30.0, 40.0),
                ("mouse.wheel", 0.0, -2.0),
                ("keyboard.up", "Control"),
            ],
        )

    async def test_modifier_keys_are_released_when_mouse_action_fails(self) -> None:
        self.browser.page.mouse.fail_click = True

        with self.assertRaisesRegex(RuntimeError, "injected click failure"):
            await self.driver.execute_action(
                {
                    "type": "click",
                    "x": 10,
                    "y": 20,
                    "keys": ["CTRL", "SHIFT"],
                }
            )

        self.assertEqual(
            self.browser.page.events,
            [
                ("keyboard.down", "Control"),
                ("keyboard.down", "Shift"),
                ("mouse.click", 10.0, 20.0, "left", 1),
                ("keyboard.up", "Shift"),
                ("keyboard.up", "Control"),
            ],
        )

    async def test_zoom_is_not_advertised_or_executed(self) -> None:
        tool = ComputerEnvironment(self.driver).tools("anthropic")[0]
        self.assertNotIn("enable_zoom", tool.provider_options)

        with self.assertRaisesRegex(ProtocolError, "unsupported computer action"):
            await self.driver.execute_action(
                {"action": "zoom", "region": [0, 0, 100, 100]}
            )


if __name__ == "__main__":
    unittest.main()
