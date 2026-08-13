"""Unit tests for the shared validators in :mod:`mini_agent.types`."""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mini_agent.types import (
    InvalidAction,
    ProtocolError,
    _require_bool,
    _require_callable,
    _require_finite_number,
    _require_image_url,
    _require_int,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
    _require_text,
    _require_tuple_of,
)


class RequireStringTests(unittest.TestCase):
    def test_returns_the_validated_value(self) -> None:
        self.assertEqual(_require_str("value", "label"), "value")
        self.assertEqual(_require_str("", "label", non_empty=False), "")

    def test_rejects_non_strings_and_blank_values(self) -> None:
        for value in (None, 5, b"bytes", "", "   "):
            with self.assertRaisesRegex(
                ValueError, "label must be a non-empty string"
            ):
                _require_str(value, "label")

    def test_optional_emptiness_still_rejects_non_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "label must be a string"):
            _require_str(5, "label", non_empty=False)

    def test_stripped_rejects_surrounding_whitespace(self) -> None:
        self.assertEqual(_require_str(" x ", "label"), " x ")
        with self.assertRaisesRegex(ValueError, "label must be a non-empty string"):
            _require_str(" x ", "label", stripped=True)

    def test_error_class_is_configurable(self) -> None:
        with self.assertRaises(ProtocolError):
            _require_str(None, "label", error=ProtocolError)

    def test_require_text_also_enforces_utf8(self) -> None:
        self.assertEqual(_require_text("ok", "label"), "ok")
        with self.assertRaisesRegex(ValueError, "label must be valid UTF-8 text"):
            _require_text("\ud800", "label")


class RequireBooleanTests(unittest.TestCase):
    def test_accepts_only_real_booleans(self) -> None:
        self.assertIs(_require_bool(True, "flag"), True)
        for value in (1, 0, None, "true"):
            with self.assertRaisesRegex(ValueError, "flag must be a boolean"):
                _require_bool(value, "flag")


class RequireIntegerTests(unittest.TestCase):
    def test_rejects_booleans_as_integers(self) -> None:
        with self.assertRaisesRegex(ValueError, "count must be a non-negative integer"):
            _require_int(True, "count", minimum=0)

    def test_bounds_shape_the_message(self) -> None:
        cases = (
            ({"minimum": 1}, "a positive integer", 0),
            ({"minimum": 0}, "a non-negative integer", -1),
            ({"minimum": 2, "maximum": 4}, "an integer between 2 and 4", 5),
            ({"minimum": 3}, "an integer of at least 3", 2),
            ({"maximum": 3}, "an integer of at most 3", 4),
            ({}, "an integer", "x"),
        )
        for bounds, expected, invalid in cases:
            with self.assertRaisesRegex(ValueError, f"count must be {expected}"):
                _require_int(invalid, "count", **bounds)  # type: ignore[arg-type]

    def test_positive_helper_accepts_and_returns(self) -> None:
        self.assertEqual(_require_positive_int(3, "steps"), 3)
        with self.assertRaisesRegex(ValueError, "steps must be a positive integer"):
            _require_positive_int(0, "steps")


class RequireFiniteNumberTests(unittest.TestCase):
    def test_returns_a_float(self) -> None:
        value = _require_finite_number(2, "cost", minimum=0)
        self.assertIsInstance(value, float)
        self.assertEqual(value, 2.0)

    def test_rejects_non_finite_and_boolean_values(self) -> None:
        for value in (math.inf, -math.inf, math.nan, True, "1"):
            with self.assertRaises(ValueError):
                _require_finite_number(value, "cost", minimum=0)

    def test_bounds_shape_the_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost must be finite and positive"):
            _require_finite_number(0, "cost", exclusive_minimum=0)
        with self.assertRaisesRegex(ValueError, "cost must be finite and non-negative"):
            _require_finite_number(-1, "cost", minimum=0)
        with self.assertRaisesRegex(ValueError, "cost must be a finite number"):
            _require_finite_number(math.nan, "cost")


class RequireContainerTests(unittest.TestCase):
    def test_mapping_returns_the_mapping(self) -> None:
        value = {"a": 1}
        self.assertIs(_require_mapping(value, "body"), value)
        with self.assertRaisesRegex(ValueError, "body must be an object"):
            _require_mapping([], "body")

    def test_callable_returns_the_callable(self) -> None:
        self.assertIs(_require_callable(len, "factory"), len)
        with self.assertRaisesRegex(ValueError, "factory must be callable"):
            _require_callable(None, "factory")

    def test_tuple_of_checks_every_item(self) -> None:
        self.assertEqual(_require_tuple_of(("a",), str, "names"), ("a",))
        with self.assertRaisesRegex(
            ValueError, r"names must be a tuple of str values"
        ):
            _require_tuple_of(["a"], str, "names")
        with self.assertRaisesRegex(ValueError, r"names must be str values"):
            _require_tuple_of(("a", 1), str, "names", brief=True)

    def test_image_url_accepts_none_and_data_urls(self) -> None:
        self.assertIsNone(_require_image_url(None, "image"))
        self.assertEqual(
            _require_image_url("data:image/png;base64,AA", "image"),
            "data:image/png;base64,AA",
        )
        with self.assertRaisesRegex(ValueError, "image must be an image data URL"):
            _require_image_url("https://example.invalid/a.png", "image")


class RequireNoSymlinkTests(unittest.TestCase):
    def test_accepts_a_real_path_and_rejects_a_symlink(self) -> None:
        with TemporaryDirectory() as name:
            root = Path(name)
            target = root / "real"
            target.mkdir()
            self.assertEqual(_require_no_symlink(target, "asset"), target)
            link = root / "link"
            os.symlink(target, link)
            with self.assertRaisesRegex(ValueError, "asset must not be a symlink"):
                _require_no_symlink(link, "asset")
            with self.assertRaises(InvalidAction):
                _require_no_symlink(link, "asset", error=InvalidAction)


if __name__ == "__main__":
    unittest.main()
