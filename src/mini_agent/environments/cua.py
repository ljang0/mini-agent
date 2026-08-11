from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import inspect
import json
import math
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import httpx

from ..types import ProtocolError, ToolCall, ToolDefinition, ToolExecution
from .base import BaseEnvironment


CUA_SPEED_RUN_REVISION = "7230223cbc57df68331cad32889adf01f3601651"
OSWORLD_REVISION = "091f5ef1d5544bc74953c77875d5feb5bed30108"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_POINT_MOUSE_ACTIONS = {
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "move",
}
_BUTTON_ACTIONS = {
    "left_down",
    "left_up",
    "right_down",
    "right_up",
    "middle_down",
    "middle_up",
}
_KEY_ALIASES = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "ctrl": "ctrl",
    "control": "ctrl",
    "cmd": "super",
    "command": "super",
    "win": "super",
    "meta": "super",
    "super": "super",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "tab": "Tab",
    "backspace": "BackSpace",
    "delete": "Delete",
    "space": "space",
    "spacebar": "space",
    "pagedown": "pagedown",
    "pageup": "pageup",
    "arrowleft": "Left",
    "left": "Left",
    "arrowright": "Right",
    "right": "Right",
    "arrowup": "Up",
    "up": "Up",
    "arrowdown": "Down",
    "down": "Down",
}


@dataclass(frozen=True)
class ComputerObservation:
    png: bytes
    meta: Mapping[str, Any] = field(default_factory=dict)


class ComputerClient(Protocol):
    async def observe(self) -> ComputerObservation: ...

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]: ...

    async def done(self) -> None: ...

    async def close(self) -> None: ...


def validate_png(png: bytes) -> tuple[int, int]:
    """Validate a complete PNG stream and return its IHDR dimensions."""

    if not isinstance(png, bytes) or not png.startswith(_PNG_SIGNATURE):
        raise ProtocolError("computer observation is not a PNG")
    offset = len(_PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    first = True
    saw_iend = False
    while offset < len(png):
        if offset + 12 > len(png):
            raise ProtocolError("computer observation contains a truncated PNG chunk")
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(png):
            raise ProtocolError("computer observation contains a truncated PNG chunk")
        expected_crc = struct.unpack(">I", png[data_end:crc_end])[0]
        observed_crc = zlib.crc32(chunk_type)
        observed_crc = zlib.crc32(png[data_start:data_end], observed_crc) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise ProtocolError("computer observation contains an invalid PNG checksum")
        if first and chunk_type != b"IHDR":
            raise ProtocolError("computer observation PNG does not start with IHDR")
        if chunk_type == b"IHDR":
            if not first or length != 13:
                raise ProtocolError("computer observation contains an invalid IHDR")
            width, height = struct.unpack(">II", png[data_start : data_start + 8])
            if width < 1 or height < 1:
                raise ProtocolError("computer observation has invalid PNG dimensions")
            dimensions = (width, height)
        if chunk_type == b"IEND":
            if length != 0 or crc_end != len(png):
                raise ProtocolError("computer observation contains an invalid IEND")
            saw_iend = True
        first = False
        offset = crc_end
    if dimensions is None or not saw_iend:
        raise ProtocolError("computer observation is an incomplete PNG")
    return dimensions


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def resize_png(png: bytes, size: tuple[int, int]) -> bytes:
    """Resize ordinary 8-bit non-interlaced PNG screenshots using stdlib only."""

    width, height = validate_png(png)
    target_width, target_height = size
    if target_width < 1 or target_height < 1:
        raise ProtocolError("rendered screenshot dimensions must be positive")
    if size == (width, height):
        return png
    offset = len(_PNG_SIGNATURE)
    ihdr = b""
    compressed = bytearray()
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            ihdr = data
        elif kind == b"IDAT":
            compressed.extend(data)
        offset += 12 + length
    if len(ihdr) != 13:
        raise ProtocolError("screenshot PNG is missing IHDR")
    _, _, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if (
        bit_depth != 8
        or channels is None
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise ProtocolError(
            "screenshot resizing supports non-interlaced 8-bit grayscale/RGB/RGBA PNGs"
        )
    try:
        filtered = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ProtocolError("screenshot PNG contains invalid compressed pixels") from exc
    stride = width * channels
    if len(filtered) != height * (stride + 1):
        raise ProtocolError("screenshot PNG has an unexpected pixel payload length")
    rows: list[bytearray] = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = filtered[cursor]
        cursor += 1
        encoded = filtered[cursor : cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, byte in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                value = byte
            elif filter_type == 1:
                value = byte + left
            elif filter_type == 2:
                value = byte + above
            elif filter_type == 3:
                value = byte + ((left + above) // 2)
            elif filter_type == 4:
                value = byte + _paeth(left, above, upper_left)
            else:
                raise ProtocolError(f"screenshot PNG uses invalid filter {filter_type}")
            row[index] = value & 0xFF
        rows.append(row)
        previous = row
    rendered = bytearray()
    for target_y in range(target_height):
        source_y = min(height - 1, target_y * height // target_height)
        source = rows[source_y]
        rendered.append(0)
        for target_x in range(target_width):
            source_x = min(width - 1, target_x * width // target_width)
            start = source_x * channels
            rendered.extend(source[start : start + channels])
    output_ihdr = struct.pack(
        ">IIBBBBB",
        target_width,
        target_height,
        bit_depth,
        color_type,
        compression,
        filtering,
        interlace,
    )
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", output_ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rendered)))
        + _png_chunk(b"IEND", b"")
    )


def _point(value: Any, *, width: int | None = None, height: int | None = None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProtocolError("computer coordinate must be [x, y]")
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
        for number in value
    ):
        raise ProtocolError("computer coordinates must be finite numbers")
    x, y = round(float(value[0])), round(float(value[1]))
    if (x < 0 and width is None) or (y < 0 and height is None):
        raise ProtocolError("computer coordinates must be non-negative")
    if width is not None:
        x = max(0, min(width - 1, x))
    if height is not None:
        y = max(0, min(height - 1, y))
    return [x, y]


def normalize_key(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        raise ProtocolError("computer key names must be non-empty")
    compact = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    for suffix in ("left", "right", "l", "r"):
        if compact in {"ctrl" + suffix, "control" + suffix}:
            return "ctrl"
        if compact in {"alt" + suffix, "option" + suffix}:
            return "alt"
        if compact == "shift" + suffix:
            return "shift"
        if compact in {"super" + suffix, "win" + suffix, "meta" + suffix}:
            return "super"
    return _KEY_ALIASES.get(compact, raw)


def _keys(value: Any) -> list[str]:
    values = value.split("+") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ProtocolError("computer keys must be a non-empty string or list")
    return [normalize_key(item) for item in values]


def validate_osworld_script(script: Any, *, max_chars: int = 20_000) -> str:
    """Validate a narrow desktop-action language; this is not a Python sandbox."""

    if not isinstance(script, str) or not script.strip():
        raise ProtocolError("OSWorld action requires a non-empty script")
    if len(script) > max_chars:
        raise ProtocolError(f"OSWorld script exceeds {max_chars} characters")
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError as exc:
        raise ProtocolError(f"OSWorld script is invalid Python: {exc.msg}") from exc
    safe_modules = {"pyautogui", "time", "pyperclip", "math"}
    safe_attributes = {
        "pyautogui": {
            "click",
            "doubleClick",
            "dragRel",
            "dragTo",
            "hotkey",
            "hscroll",
            "keyDown",
            "keyUp",
            "middleClick",
            "mouseDown",
            "mouseUp",
            "moveRel",
            "moveTo",
            "position",
            "press",
            "rightClick",
            "screenshot",
            "scroll",
            "size",
            "tripleClick",
            "typewrite",
            "vscroll",
            "write",
        },
        "time": {"monotonic", "sleep", "time"},
        "pyperclip": {"copy", "paste"},
    }
    safe_builtins = {
        "abs",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    forbidden_names = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "getattr",
        "help",
        "input",
        "locals",
        "open",
        "object",
        "setattr",
        "type",
        "vars",
    }
    module_aliases: dict[str, str] = {}
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".", 1)[0]
                if module not in safe_modules or alias.name != module:
                    raise ProtocolError(
                        "OSWorld scripts may import only desktop-action modules"
                    )
                module_aliases[alias.asname or module] = module
        elif isinstance(node, ast.ImportFrom):
            raise ProtocolError("OSWorld scripts must use module-qualified imports")
        elif isinstance(node, ast.Name) and (
            node.id in forbidden_names or node.id.startswith("__")
        ):
            raise ProtocolError(f"OSWorld script may not use {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or not isinstance(node.value, ast.Name):
                raise ProtocolError(
                    "OSWorld scripts may access only direct desktop-module attributes"
                )
            module = module_aliases.get(node.value.id, node.value.id)
            allowed = safe_attributes.get(module)
            if module == "math":
                if node.attr.startswith("_"):
                    raise ProtocolError("OSWorld scripts may not access private math APIs")
            elif allowed is None or node.attr not in allowed:
                raise ProtocolError(
                    f"OSWorld scripts may not access {node.value.id}.{node.attr}"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in safe_builtins | function_names:
                raise ProtocolError(
                    f"OSWorld scripts may not call {node.func.id} directly"
                )
    return script


def validate_gym_action(
    action: Any,
    *,
    allow_scripts: bool = False,
    allow_terminal: bool = False,
) -> dict[str, Any]:
    """Validate the supported gym-anything action vocabulary atomically."""

    if not isinstance(action, Mapping):
        raise ProtocolError("computer actions must be objects")
    value = dict(action)
    if set(value) == {"action", "time"}:
        if value["action"] != "wait":
            raise ProtocolError("only wait accepts a time field")
        duration = value["time"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ProtocolError("computer wait time must be finite and positive")
        return {"action": "wait", "time": float(duration)}
    if len(value) != 1:
        raise ProtocolError("computer actions must contain exactly one action family")
    family, payload = next(iter(value.items()))
    if family == "mouse":
        if not isinstance(payload, Mapping) or len(payload) != 1:
            raise ProtocolError("mouse actions require exactly one operation")
        operation, argument = next(iter(payload.items()))
        if operation in _POINT_MOUSE_ACTIONS:
            normalized: Any = _point(argument)
        elif operation == "left_click_drag":
            if not isinstance(argument, list) or len(argument) < 2:
                raise ProtocolError("left_click_drag requires at least two points")
            normalized = [_point(point) for point in argument]
        elif operation == "scroll":
            if (
                isinstance(argument, bool)
                or not isinstance(argument, (int, float))
                or not math.isfinite(float(argument))
            ):
                raise ProtocolError("mouse scroll must be a finite number")
            normalized = argument
        elif operation == "buttons":
            if not isinstance(argument, Mapping) or not argument:
                raise ProtocolError("mouse buttons must be a non-empty object")
            if set(argument) - _BUTTON_ACTIONS or any(flag is not True for flag in argument.values()):
                raise ProtocolError("mouse button transitions must be known keys set to true")
            normalized = dict(argument)
        else:
            raise ProtocolError(f"unsupported mouse operation {operation!r}")
        return {"mouse": {str(operation): normalized}}
    if family == "keyboard":
        if not isinstance(payload, Mapping) or len(payload) != 1:
            raise ProtocolError("keyboard actions require exactly one operation")
        operation, argument = next(iter(payload.items()))
        if operation == "text":
            if not isinstance(argument, str):
                raise ProtocolError("keyboard text must be a string")
            normalized = argument
        elif operation in {"keys", "keys_down", "keys_up"}:
            normalized = _keys(argument)
        elif operation in {"key_down", "key_up"}:
            normalized = normalize_key(argument)
        else:
            raise ProtocolError(f"unsupported keyboard operation {operation!r}")
        return {"keyboard": {str(operation): normalized}}
    if family == "action":
        if not isinstance(payload, str):
            raise ProtocolError("direct computer action must be a string")
        allowed = {"wait", "screenshot"}
        if allow_terminal:
            allowed.update({"done", "fail", "infeasible"})
        if payload not in allowed:
            raise ProtocolError(f"unsupported direct computer action {payload!r}")
        if payload == "wait":
            raise ProtocolError("wait requires both action and time fields")
        return {"action": "fail" if payload == "infeasible" else payload}
    if family == "wait":
        if (
            isinstance(payload, bool)
            or not isinstance(payload, (int, float))
            or not math.isfinite(float(payload))
            or float(payload) <= 0
        ):
            raise ProtocolError("computer wait time must be finite and positive")
        return {"action": "wait", "time": float(payload)}
    if family == "script" and allow_scripts:
        return {"script": validate_osworld_script(payload)}
    raise ProtocolError(f"unsupported computer action family {family!r}")


def validate_gym_actions(
    actions: Any,
    *,
    allow_scripts: bool = False,
    allow_terminal: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(actions, list) or not actions:
        raise ProtocolError("computer requires a non-empty actions list")
    return [
        validate_gym_action(
            action,
            allow_scripts=allow_scripts,
            allow_terminal=allow_terminal,
        )
        for action in actions
    ]


def _scroll_steps(pixels: Any) -> int:
    if isinstance(pixels, bool) or not isinstance(pixels, (int, float)):
        raise ProtocolError("scroll distance must be numeric")
    value = float(pixels)
    if not math.isfinite(value):
        raise ProtocolError("scroll distance must be finite")
    if value == 0:
        return 0
    return int(math.copysign(max(1, min(30, math.ceil(abs(value) / 120))), value))


def translate_openai_action(
    action: Mapping[str, Any], width: int, height: int
) -> list[dict[str, Any]]:
    """Translate one OpenAI GA computer action into gym-anything actions."""

    if width < 1 or height < 1:
        raise ProtocolError("native computer dimensions must be positive")
    kind = str(action.get("type") or "")
    keys = _keys(action.get("keys")) if action.get("keys") else []
    down = [{"keyboard": {"key_down": key}} for key in keys]
    up = [{"keyboard": {"key_up": key}} for key in reversed(keys)]

    def xy(value: Mapping[str, Any] = action) -> list[int]:
        return _point([value.get("x"), value.get("y")], width=width, height=height)

    output: list[dict[str, Any]] = []
    if kind in {"click", "double_click"}:
        button = str(action.get("button", "left")).lower()
        names = {"left": "left_click", "right": "right_click", "middle": "middle_click", "wheel": "middle_click"}
        if button not in names:
            raise ProtocolError(f"unsupported mouse button {button!r}")
        operation = "double_click" if kind == "double_click" and button == "left" else names[button]
        output = [*down, {"mouse": {operation: xy()}}, *up]
        if kind == "double_click" and operation != "double_click":
            output.insert(len(down) + 1, {"mouse": {operation: xy()}})
    elif kind == "move":
        output = [*down, {"mouse": {"move": xy()}}, *up]
    elif kind == "drag":
        path = action.get("path")
        if not isinstance(path, list) or len(path) < 2:
            raise ProtocolError("drag requires at least two path points")
        points = [
            _point(
                [point.get("x"), point.get("y")] if isinstance(point, Mapping) else point,
                width=width,
                height=height,
            )
            for point in path
        ]
        output = [*down, {"mouse": {"left_click_drag": points}}, *up]
    elif kind == "scroll":
        output.extend(down)
        if action.get("x") is not None and action.get("y") is not None:
            output.append({"mouse": {"move": xy()}})
        vertical = _scroll_steps(action.get("scroll_y", action.get("delta_y", 0)))
        horizontal = _scroll_steps(action.get("scroll_x", action.get("delta_x", 0)))
        if vertical:
            output.append({"mouse": {"scroll": vertical}})
        if horizontal:
            output.extend(
                [
                    {"keyboard": {"key_down": "shift"}},
                    {"mouse": {"scroll": horizontal}},
                    {"keyboard": {"key_up": "shift"}},
                ]
            )
        output.extend(up)
    elif kind == "type":
        output = [{"keyboard": {"text": str(action.get("text", ""))}}]
    elif kind == "keypress":
        pressed = action.get("keys") or ([action["key"]] if action.get("key") else [])
        output = [{"keyboard": {"keys": _keys(pressed)}}]
    elif kind == "wait":
        milliseconds = action.get("ms", 1000)
        if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)):
            raise ProtocolError("wait milliseconds must be numeric")
        output = [{"wait": max(0.1, min(30.0, float(milliseconds) / 1000.0))}]
    elif kind == "screenshot":
        output = [{"action": "screenshot"}]
    else:
        raise ProtocolError(f"unsupported OpenAI computer action {kind!r}")
    return validate_gym_actions(output)


def display_size(width: int, height: int, *, max_width: int = 1280, max_height: int = 720) -> tuple[int, int]:
    if min(width, height, max_width, max_height) < 1:
        raise ProtocolError("display dimensions must be positive")
    scale = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _scaled_point(value: Any, native: tuple[int, int], display: tuple[int, int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProtocolError("computer coordinate must be [x, y]")
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
        for number in value
    ):
        raise ProtocolError("computer coordinates must be finite numbers")
    return [
        max(0, min(native[0] - 1, round(float(value[0]) * native[0] / display[0]))),
        max(0, min(native[1] - 1, round(float(value[1]) * native[1] / display[1]))),
    ]


def translate_anthropic_action(
    action: Mapping[str, Any],
    native: tuple[int, int],
    display: tuple[int, int],
    cursor: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], list[int] | None]:
    """Translate one Anthropic computer action and preserve cursor state."""

    kind = str(action.get("action") or "")
    coordinate = action.get("coordinate")
    point = _scaled_point(coordinate, native, display) if coordinate is not None else None
    next_cursor = list(cursor) if cursor is not None else None
    if point is not None:
        next_cursor = point

    modifier = _keys(action.get("text")) if action.get("text") and kind not in {"key", "type"} else []

    def wrap(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not modifier:
            return items
        return [
            {"keyboard": {"keys_down": modifier}},
            *items,
            {"keyboard": {"keys_up": list(reversed(modifier))}},
        ]

    if kind in {"screenshot", "cursor_position", "zoom"}:
        if kind == "zoom":
            region = action.get("region")
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                raise ProtocolError("zoom requires region [x1, y1, x2, y2]")
            first = _scaled_point(region[:2], native, display)
            second = _scaled_point(region[2:], native, display)
            if first[0] == second[0] or first[1] == second[1]:
                raise ProtocolError("zoom region must have non-zero area")
        return [], next_cursor
    if kind == "key":
        return validate_gym_actions([{"keyboard": {"keys": _keys(action.get("text", action.get("keys")))}}]), next_cursor
    if kind == "type":
        text = action.get("text")
        if not isinstance(text, str):
            raise ProtocolError("type requires string text")
        items: list[dict[str, Any]] = []
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if line:
                items.append({"keyboard": {"text": line}})
            if index < len(lines) - 1:
                items.append({"keyboard": {"keys": ["Return"]}})
        return validate_gym_actions(items or [{"keyboard": {"text": ""}}]), next_cursor
    if kind == "hold_key":
        held = _keys(action.get("text", action.get("keys")))
        duration = max(0.1, min(10.0, float(action.get("duration", 1.0))))
        return validate_gym_actions(
            [
                {"keyboard": {"keys_down": held}},
                {"wait": duration},
                {"keyboard": {"keys_up": list(reversed(held))}},
            ]
        ), next_cursor
    if kind == "wait":
        duration = max(0.1, min(30.0, float(action.get("duration", 1.0))))
        return validate_gym_actions([{"wait": duration}]), next_cursor
    if kind == "mouse_move":
        if point is None:
            raise ProtocolError("mouse_move requires coordinate")
        return validate_gym_actions([{"mouse": {"move": point}}]), next_cursor
    if kind in {"left_mouse_down", "left_mouse_up"}:
        items = [] if point is None else [{"mouse": {"move": point}}]
        items.append({"mouse": {"buttons": {"left_down" if kind.endswith("down") else "left_up": True}}})
        return validate_gym_actions(wrap(items)), next_cursor
    if kind == "left_click_drag":
        if point is None:
            raise ProtocolError("left_click_drag requires coordinate")
        start_value = action.get("start_coordinate")
        start = _scaled_point(start_value, native, display) if start_value is not None else cursor
        if start is None:
            raise ProtocolError("left_click_drag requires a start coordinate or cursor")
        return validate_gym_actions(wrap([{"mouse": {"left_click_drag": [list(start), point]}}])), next_cursor
    if kind == "scroll":
        direction = str(action.get("scroll_direction") or "")
        amount = action.get("scroll_amount")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ProtocolError("scroll_amount must be a non-negative integer")
        items = [] if point is None else [{"mouse": {"move": point}}]
        if direction == "up":
            items.append({"mouse": {"scroll": -amount}})
        elif direction == "down":
            items.append({"mouse": {"scroll": amount}})
        elif direction in {"left", "right"}:
            horizontal = amount if direction == "right" else -amount
            items.extend(
                [
                    {"keyboard": {"keys_down": ["shift"]}},
                    {"mouse": {"scroll": horizontal}},
                    {"keyboard": {"keys_up": ["shift"]}},
                ]
            )
        else:
            raise ProtocolError(f"invalid scroll direction {direction!r}")
        return validate_gym_actions(wrap(items)), next_cursor
    clicks = {
        "left_click": "left_click",
        "right_click": "right_click",
        "middle_click": "middle_click",
        "double_click": "double_click",
        "triple_click": "triple_click",
    }
    if kind in clicks:
        if point is None:
            raise ProtocolError(f"{kind} requires coordinate")
        return validate_gym_actions(wrap([{"mouse": {clicks[kind]: point}}])), next_cursor
    raise ProtocolError(f"unsupported Anthropic computer action {kind!r}")


def gym_action_schema(*, allow_scripts: bool = False, allow_terminal: bool = False) -> Mapping[str, Any]:
    terminal = ["screenshot"]
    if allow_terminal:
        terminal.extend(["done", "fail", "infeasible"])
    variants: list[Mapping[str, Any]] = [
        {
            "type": "object",
            "properties": {"mouse": {"type": "object", "minProperties": 1, "maxProperties": 1}},
            "required": ["mouse"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"keyboard": {"type": "object", "minProperties": 1, "maxProperties": 1}},
            "required": ["keyboard"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"wait": {"type": "number", "exclusiveMinimum": 0}},
            "required": ["wait"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "wait"},
                "time": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["action", "time"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": terminal}},
            "required": ["action"],
            "additionalProperties": False,
        },
    ]
    if allow_scripts:
        variants.append(
            {
                "type": "object",
                "properties": {"script": {"type": "string", "minLength": 1, "maxLength": 20_000}},
                "required": ["script"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}


class CUASpeedRunClient:
    """Operation-safe async client for the public agent-facing gateway plane."""

    def __init__(
        self,
        env_url: str,
        *,
        timeout_seconds: float = 120.0,
        connect_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(connect_retries, int) or isinstance(connect_retries, bool) or connect_retries < 0:
            raise ValueError("connect_retries must be a non-negative integer")
        parsed = urlsplit(env_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("env_url must be an HTTP(S) URL without query or fragment")
        self.env_url = env_url.rstrip("/")
        self.connect_retries = connect_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._sleep = sleep

    async def _request(self, method: str, path: str, *, idempotent: bool) -> httpx.Response:
        retryable: tuple[type[BaseException], ...] = (
            (httpx.ConnectError, httpx.ConnectTimeout)
            if idempotent
            else (httpx.ConnectTimeout,)
        )
        for attempt in range(self.connect_retries + 1):
            try:
                response = await self._client.request(method, f"{self.env_url}/{path}")
            except retryable:
                if attempt >= self.connect_retries:
                    raise
                await self._sleep(self.retry_delay_seconds * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise AssertionError("unreachable")

    async def observe(self) -> ComputerObservation:
        response = await self._request("GET", "observe", idempotent=True)
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ProtocolError("gateway observe response must be an object")
        encoded = payload.get("png_b64")
        if not isinstance(encoded, str):
            raise ProtocolError("gateway observe response requires png_b64")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ProtocolError("gateway observe response contains invalid base64") from exc
        width, height = validate_png(png)
        raw_meta = payload.get("meta", {})
        if not isinstance(raw_meta, Mapping):
            raise ProtocolError("gateway observe metadata must be an object")
        meta = {"width": width, "height": height, **dict(raw_meta)}
        return ComputerObservation(png=png, meta=meta)

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        # A step may have reached the environment before a read-side failure, so
        # only a connect timeout is retried.
        for attempt in range(self.connect_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.env_url}/step", json={"actions": actions}
                )
            except httpx.ConnectTimeout:
                if attempt >= self.connect_retries:
                    raise
                await self._sleep(self.retry_delay_seconds * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ProtocolError("gateway step response must be an object")
            return dict(payload)
        raise AssertionError("unreachable")

    async def done(self) -> None:
        await self._request("POST", "done", idempotent=True)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def isolation_key(self) -> str:
        return "cua-speed-run:" + hashlib.sha256(self.env_url.encode()).hexdigest()

    def provenance(self) -> Mapping[str, Any]:
        parsed = urlsplit(self.env_url)
        return {
            "client": "cua_speed_run_http",
            "gateway_origin": f"{parsed.scheme}://{parsed.netloc}",
            "env_url_sha256": hashlib.sha256(self.env_url.encode()).hexdigest(),
            "connect_retries": self.connect_retries,
        }


_OBSERVE_ONLY = object()


def _pyautogui_key(key: str) -> str:
    return {
        "Return": "enter",
        "Escape": "esc",
        "BackSpace": "backspace",
        "Delete": "delete",
        "Left": "left",
        "Right": "right",
        "Up": "up",
        "Down": "down",
        "super": "win",
    }.get(key, key.lower())


def gym_action_to_pyautogui(action: Mapping[str, Any]) -> str | object:
    """Translate a validated gym action to OSWorld's pyautogui action space."""

    if "script" in action:
        return validate_osworld_script(action["script"])
    normalized = validate_gym_action(action, allow_terminal=True)
    if normalized.get("action") == "screenshot":
        return _OBSERVE_ONLY
    if normalized.get("action") in {"done", "fail"}:
        return str(normalized["action"]).upper()
    if normalized.get("action") == "wait":
        return f"time.sleep({float(normalized['time'])!r})"
    if "mouse" in normalized:
        operation, argument = next(iter(normalized["mouse"].items()))
        if operation in {"left_click", "right_click", "middle_click"}:
            button = operation.split("_", 1)[0]
            return f"pyautogui.click({argument[0]}, {argument[1]}, button={button!r})"
        if operation in {"double_click", "triple_click"}:
            clicks = 2 if operation == "double_click" else 3
            return f"pyautogui.click({argument[0]}, {argument[1]}, clicks={clicks}, interval=0.1)"
        if operation == "move":
            return f"pyautogui.moveTo({argument[0]}, {argument[1]}, duration=0.2)"
        if operation == "scroll":
            return f"pyautogui.scroll({argument!r})"
        if operation == "left_click_drag":
            lines = [f"pyautogui.moveTo({argument[0][0]}, {argument[0][1]})", "pyautogui.mouseDown(button='left')"]
            lines.extend(f"pyautogui.moveTo({point[0]}, {point[1]}, duration=0.1)" for point in argument[1:])
            lines.append("pyautogui.mouseUp(button='left')")
            return "\n".join(lines)
        if operation == "buttons":
            lines = []
            for transition in argument:
                button, state = transition.rsplit("_", 1)
                lines.append(f"pyautogui.mouse{'Down' if state == 'down' else 'Up'}(button={button!r})")
            return "\n".join(lines)
    operation, argument = next(iter(normalized["keyboard"].items()))
    if operation == "text":
        return f"pyautogui.write({argument!r}, interval=0.01)"
    if operation == "keys":
        keys = [_pyautogui_key(key) for key in argument]
        return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"
    if operation in {"keys_down", "keys_up"}:
        state = "keyDown" if operation.endswith("down") else "keyUp"
        return "\n".join(f"pyautogui.{state}({_pyautogui_key(key)!r})" for key in argument)
    state = "keyDown" if operation == "key_down" else "keyUp"
    return f"pyautogui.{state}({_pyautogui_key(argument)!r})"


class OSWorldClient:
    """Bridge a live OSWorld DesktopEnv without exposing its evaluator to an agent."""

    def __init__(
        self,
        environment: Any,
        initial_observation: Mapping[str, Any],
        *,
        action_encoder: Callable[[Mapping[str, Any]], Any] | None = None,
        transition_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        owns_environment: bool = True,
        resource_identity: str | None = None,
    ) -> None:
        step = getattr(environment, "step", None)
        if not callable(step):
            raise ValueError("OSWorld environment must expose step")
        self._environment_step = step
        self._environment_observe = getattr(environment, "_get_obs", None)
        self._environment_close = getattr(environment, "close", None)
        self._observation = dict(initial_observation)
        self.action_encoder = action_encoder or self._default_encoder
        self.transition_sink = transition_sink
        self.owns_environment = owns_environment
        if resource_identity is not None and (
            not isinstance(resource_identity, str) or not resource_identity
        ):
            raise ValueError("OSWorld resource_identity must be a non-empty string")
        if resource_identity is None:
            resource_identity = getattr(
                environment, "_mini_agent_resource_identity", None
            )
        if not isinstance(resource_identity, str) or not resource_identity:
            resource_identity = f"osworld-desktop:{uuid.uuid4().hex}"
            try:
                setattr(environment, "_mini_agent_resource_identity", resource_identity)
            except (AttributeError, TypeError):
                pass
        self._resource_identity = resource_identity
        self._steps = 0
        self._terminal = False

    @staticmethod
    def _default_encoder(action: Mapping[str, Any]) -> Any:
        if "script" in action:
            return validate_osworld_script(action.get("script"))
        return gym_action_to_pyautogui(action)

    async def observe(self) -> ComputerObservation:
        screenshot = self._observation.get("screenshot")
        if not isinstance(screenshot, bytes):
            raise ProtocolError("OSWorld observation requires screenshot bytes")
        return ComputerObservation(png=screenshot, meta={"source": "osworld"})

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        latest: dict[str, Any] = {}
        for action in actions:
            if self._terminal:
                break
            encoded = self.action_encoder(action)
            if encoded is _OBSERVE_ONLY:
                get_observation = self._environment_observe
                if get_observation is not None:
                    observation = await asyncio.to_thread(get_observation)
                    if isinstance(observation, Mapping):
                        self._observation = dict(observation)
                continue
            result = await asyncio.to_thread(self._environment_step, encoded)
            if not isinstance(result, tuple) or len(result) < 4:
                raise ProtocolError("OSWorld env.step returned an invalid result")
            observation, reward, done, raw_info = result[:4]
            if not isinstance(observation, Mapping):
                raise ProtocolError("OSWorld env.step returned an invalid observation")
            self._observation = dict(observation)
            self._steps += 1
            latest = {
                "reward": reward,
                "done": bool(done),
                "info": dict(raw_info) if isinstance(raw_info, Mapping) else {},
                "step": self._steps,
            }
            if self.transition_sink is not None:
                transition = {
                    **latest,
                    "action": dict(action),
                    "encoded_action": encoded,
                    "observation": dict(self._observation),
                }
                sink_result = self.transition_sink(transition)
                if inspect.isawaitable(sink_result):
                    await sink_result
            self._terminal = bool(done)
        return latest

    async def done(self) -> None:
        # The outer OSWorld task runner exclusively owns evaluation.
        return None

    async def close(self) -> None:
        if not self.owns_environment:
            return
        close = self._environment_close
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def isolation_key(self) -> str:
        return self._resource_identity

    def provenance(self) -> Mapping[str, Any]:
        return {
            "client": "osworld_live_environment",
            "verifier_exposed": False,
            "runner_owns_evaluation": not self.owns_environment,
        }


class CUAEnvironment(BaseEnvironment):
    """One batched computer tool over an observe/step/done client."""

    def __init__(
        self,
        client: ComputerClient,
        *,
        benchmark: str = "cua-speed-run",
        template: str | None = None,
        allow_scripts: bool = False,
        allow_terminal: bool = False,
        protocol: str = "generic",
        coordinate_mode: str = "native_pixels",
        history_mode: str = "linear",
        display_max_width: int = 1280,
        display_max_height: int = 720,
    ) -> None:
        if protocol not in {"generic", "openai", "anthropic"}:
            raise ValueError("CUA protocol must be generic, openai, or anthropic")
        if coordinate_mode not in {
            "native_pixels",
            "pixels",
            "provider_pixels",
            "normalized_1000",
            "scaled_pixels",
        }:
            raise ValueError(f"unsupported CUA coordinate mode {coordinate_mode!r}")
        if history_mode != "linear":
            raise ValueError("the minimal wrapper currently supports only linear history")
        self.client = client
        self.benchmark = benchmark
        self.template = template
        self.allow_scripts = allow_scripts
        self.allow_terminal = allow_terminal
        self.protocol = protocol
        self.coordinate_mode = coordinate_mode
        self.history_mode = history_mode
        self.display_max_width = display_max_width
        self.display_max_height = display_max_height
        self._last_observation: ComputerObservation | None = None
        self._finished = False
        self._closed = False
        self._cursor: list[int] | None = None

    @classmethod
    def from_policy(
        cls,
        client: ComputerClient,
        *,
        benchmark: Mapping[str, Any],
        observation: Mapping[str, Any],
        history: Mapping[str, Any],
        tools: Sequence[str],
        response_parser: str,
        provider: str,
    ) -> "CUAEnvironment":
        """Resolve every executable CUA profile field or fail closed.

        Generation settings remain model-owned. This hook covers all
        environment-facing policy so the CLI never silently ignores manifest
        fields.
        """

        if tuple(tools) != ("computer",):
            raise ValueError("CUA profiles must expose exactly the computer tool")
        if response_parser != "provider_tool_calls":
            raise ValueError(f"unsupported CUA response parser {response_parser!r}")
        allowed_benchmark = {"name", "template", "tool_protocol", "coordinate_mode"}
        unknown_benchmark = set(benchmark) - allowed_benchmark
        if unknown_benchmark:
            raise ValueError(f"unknown CUA benchmark policy fields {sorted(unknown_benchmark)}")
        allowed_observation = {
            "coordinate_mode",
            "image_format",
            "screenshot_detail",
            "screenshot_max_height",
            "screenshot_max_width",
        }
        unknown_observation = set(observation) - allowed_observation
        if unknown_observation:
            raise ValueError(
                f"unknown CUA observation policy fields {sorted(unknown_observation)}"
            )
        allowed_history = {"mode", "images_to_keep", "image_removal_chunk"}
        if set(history) - allowed_history:
            raise ValueError(
                "CUA history compaction must be implemented by a model adapter; "
                f"unsupported fields are {sorted(set(history) - allowed_history)}"
            )
        history_mode = history.get("mode", "linear")
        if history_mode != "linear":
            raise ValueError("CUA history.mode must be linear")
        if observation.get("screenshot_detail", "original") != "original":
            raise ValueError("CUA screenshot_detail must be original")
        if observation.get("image_format", "PNG") != "PNG":
            raise ValueError("CUA observations must use PNG")

        selected_protocol = benchmark.get("tool_protocol")
        if selected_protocol is None:
            selected_protocol = {
                "openai-responses": "openai",
                "anthropic-messages": "anthropic",
            }.get(provider, "generic")
        if selected_protocol not in {"generic", "openai", "anthropic"}:
            raise ValueError(f"unsupported CUA tool protocol {selected_protocol!r}")
        coordinate_mode = observation.get(
            "coordinate_mode", benchmark.get("coordinate_mode", "native_pixels")
        )
        if selected_protocol == "openai" and coordinate_mode not in {
            "native_pixels",
            "pixels",
        }:
            raise ValueError("OpenAI CUA requires native pixel coordinates")
        if selected_protocol == "anthropic" and coordinate_mode != "scaled_pixels":
            raise ValueError("Anthropic CUA requires scaled pixel coordinates")
        if selected_protocol != "anthropic" and any(
            key in observation for key in ("screenshot_max_width", "screenshot_max_height")
        ):
            raise ValueError("screenshot resizing is supported only by Anthropic CUA")
        max_width = observation.get("screenshot_max_width", 1280)
        max_height = observation.get("screenshot_max_height", 720)
        if (
            not isinstance(max_width, int)
            or isinstance(max_width, bool)
            or max_width < 1
            or not isinstance(max_height, int)
            or isinstance(max_height, bool)
            or max_height < 1
        ):
            raise ValueError("CUA screenshot bounds must be positive integers")
        benchmark_name = str(benchmark.get("name", "cua_speed_run"))
        if benchmark_name not in {"cua_speed_run", "osworld"}:
            raise ValueError("CUA benchmark name must be cua_speed_run or osworld")
        is_osworld = benchmark_name == "osworld"
        template = benchmark.get("template")
        if template is not None:
            if is_osworld or not isinstance(template, str):
                raise ValueError("CUA templates require the cua_speed_run benchmark")
            from ..integrations.cua_speed_run import template_mapping

            mapping = template_mapping(template)
            if mapping.mode != "mini_agent_profile":
                raise ValueError(
                    f"CUA template {template!r} is not a MiniAgent profile"
                )
        return cls(
            client,
            benchmark="osworld" if is_osworld else "cua-speed-run",
            template=template,
            allow_scripts=is_osworld,
            allow_terminal=is_osworld,
            protocol=str(selected_protocol),
            coordinate_mode=str(coordinate_mode),
            history_mode=str(history_mode),
            display_max_width=max_width,
            display_max_height=max_height,
        )

    def _dimensions(self) -> tuple[int, int]:
        if self._last_observation is None:
            raise ProtocolError("computer dimensions are unavailable before observation")
        width = self._last_observation.meta.get("width")
        height = self._last_observation.meta.get("height")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
        ):
            return width, height
        return validate_png(self._last_observation.png)

    def tools(self) -> Sequence[ToolDefinition]:
        if self.protocol == "openai":
            return (ToolDefinition(name="computer", kind="openai_computer"),)
        if self.protocol == "anthropic":
            native = self._dimensions()
            shown = display_size(
                *native,
                max_width=self.display_max_width,
                max_height=self.display_max_height,
            )
            return (
                ToolDefinition(
                    name="computer",
                    kind="anthropic_computer_20251124",
                    provider_options={
                        "display_width_px": shown[0],
                        "display_height_px": shown[1],
                        "enable_zoom": True,
                    },
                ),
            )
        return (
            ToolDefinition(
                name="computer",
                description=(
                    "Execute an ordered batch of validated mouse, keyboard, wait, "
                    "screenshot, or permitted desktop-script actions."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": gym_action_schema(
                                allow_scripts=self.allow_scripts,
                                allow_terminal=self.allow_terminal,
                            ),
                            "minItems": 1,
                        }
                    },
                    "required": ["actions"],
                    "additionalProperties": False,
                },
            ),
        )

    def _rendered_observation(self, observation: ComputerObservation) -> ComputerObservation:
        if self.protocol != "anthropic":
            return observation
        native = self._dimensions()
        shown = display_size(
            *native,
            max_width=self.display_max_width,
            max_height=self.display_max_height,
        )
        if shown == native:
            return observation
        return ComputerObservation(
            resize_png(observation.png, shown),
            {**dict(observation.meta), "rendered_width": shown[0], "rendered_height": shown[1]},
        )

    def _execution(
        self,
        observation: ComputerObservation,
        info: Mapping[str, Any],
        *,
        acknowledged_safety_checks: Sequence[Mapping[str, Any]] = (),
    ) -> ToolExecution:
        rendered = self._rendered_observation(observation)
        image = "data:image/png;base64," + base64.b64encode(rendered.png).decode("ascii")
        return ToolExecution(
            output=json.dumps(
                {"environment": dict(info), "observation": dict(rendered.meta)},
                sort_keys=True,
                default=str,
            ),
            image_data_url=image,
            native_output={
                "type": "computer_screenshot",
                "detail": "original",
                "acknowledged_safety_checks": [
                    dict(check) for check in acknowledged_safety_checks
                ],
            },
            metadata={
                "screenshot_bytes": len(rendered.png),
                "source_screenshot_bytes": len(observation.png),
            },
        )

    async def initial_observation(self) -> ToolExecution:
        self._last_observation = await self.client.observe()
        return self._execution(self._last_observation, {})

    def _actions(self, action: ToolCall) -> list[dict[str, Any]]:
        raw_actions = action.arguments.get("actions")
        if self.protocol == "openai" or action.kind == "openai_computer":
            if not isinstance(raw_actions, list) or not raw_actions:
                raise ProtocolError("OpenAI computer call requires actions")
            width, height = self._dimensions()
            translated = [
                item
                for raw in raw_actions
                if isinstance(raw, Mapping)
                for item in translate_openai_action(raw, width, height)
            ]
            if len([raw for raw in raw_actions if isinstance(raw, Mapping)]) != len(raw_actions):
                raise ProtocolError("OpenAI computer actions must be objects")
            return translated
        if self.protocol == "anthropic" or action.kind == "anthropic_computer_20251124":
            native = self._dimensions()
            shown = display_size(
                *native,
                max_width=self.display_max_width,
                max_height=self.display_max_height,
            )
            translated, cursor = translate_anthropic_action(
                action.arguments, native, shown, self._cursor
            )
            self._cursor = cursor
            return translated
        validated = validate_gym_actions(
            raw_actions,
            allow_scripts=self.allow_scripts,
            allow_terminal=self.allow_terminal,
        )
        if self.coordinate_mode != "normalized_1000":
            return validated
        width, height = self._dimensions()
        scaled: list[dict[str, Any]] = []
        for item in validated:
            if "mouse" not in item:
                scaled.append(item)
                continue
            operation, argument = next(iter(item["mouse"].items()))

            def scale(point: Sequence[int]) -> list[int]:
                if point[0] > 1000 or point[1] > 1000:
                    raise ProtocolError("normalized coordinates must be within 0..1000")
                return [
                    round(point[0] * (width - 1) / 1000),
                    round(point[1] * (height - 1) / 1000),
                ]

            if operation in _POINT_MOUSE_ACTIONS:
                argument = scale(argument)
            elif operation == "left_click_drag":
                argument = [scale(point) for point in argument]
            scaled.append({"mouse": {operation: argument}})
        return scaled

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name != "computer":
            raise ProtocolError(f"unsupported CUA tool {action.name!r}")
        acknowledged_safety_checks: list[Mapping[str, Any]] = []
        if self.protocol == "openai" or action.kind == "openai_computer":
            raw = action.raw if isinstance(action.raw, Mapping) else {}
            pending = raw.get("pending_safety_checks", [])
            if not isinstance(pending, list) or not all(
                isinstance(check, Mapping)
                and isinstance(check.get("id"), str)
                and bool(check.get("id"))
                for check in pending
            ):
                raise ProtocolError(
                    "OpenAI pending_safety_checks must be objects with string ids"
                )
            acknowledged_safety_checks = [dict(check) for check in pending]
        actions = self._actions(action)
        info: Mapping[str, Any] = {}
        if actions:
            info = await self.client.step(actions)
        self._last_observation = await self.client.observe()
        return self._execution(
            self._last_observation,
            info,
            acknowledged_safety_checks=acknowledged_safety_checks,
        )

    async def finish(self) -> None:
        if not self._finished:
            await self.client.done()
            self._finished = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # cua-speed-run submissions call done from finally. Keeping that
            # invariant here covers provider errors, cancellation, and budgets
            # without adding domain logic to MiniAgent.
            await self.finish()
        finally:
            await self.client.close()

    def isolation_key(self) -> str:
        key = getattr(self.client, "isolation_key", None)
        return str(key()) if key is not None else f"cua-client:{id(self.client)}"

    def resource_identity(self) -> str:
        """Expose remote gateway/VM identity to the multi-agent orchestrator."""

        return self.isolation_key()

    def provenance(self) -> dict[str, object]:
        revision = OSWORLD_REVISION if self.benchmark == "osworld" else CUA_SPEED_RUN_REVISION
        client_provenance = getattr(self.client, "provenance", None)
        return {
            "application": "cua",
            "benchmark": self.benchmark,
            "template": self.template,
            "source_revision": revision,
            "agent_can_access_verifier": False,
            "agent_can_reset_snapshot_or_shell": False,
            "script_actions": self.allow_scripts,
            "terminal_actions": self.allow_terminal,
            "protocol": self.protocol,
            "coordinate_mode": self.coordinate_mode,
            "history_mode": self.history_mode,
            "client": dict(client_provenance()) if client_provenance is not None else {},
        }


class OSWorldEnvironment(CUAEnvironment):
    def __init__(self, client: OSWorldClient, *, protocol: str = "generic") -> None:
        super().__init__(
            client,
            benchmark="osworld",
            allow_scripts=True,
            allow_terminal=True,
            protocol=protocol,
        )


__all__ = [
    "CUAEnvironment",
    "CUASpeedRunClient",
    "CUA_SPEED_RUN_REVISION",
    "ComputerClient",
    "ComputerObservation",
    "OSWorldClient",
    "OSWorldEnvironment",
    "OSWORLD_REVISION",
    "display_size",
    "gym_action_schema",
    "gym_action_to_pyautogui",
    "normalize_key",
    "translate_anthropic_action",
    "translate_openai_action",
    "validate_gym_action",
    "validate_gym_actions",
    "validate_osworld_script",
    "validate_png",
]
