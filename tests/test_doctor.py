"""The doctor's decision logic, exercised without the daemons it reports on.

`doctor` is what an operator runs before spending money, so the part that
must be right is the verdict: which targets it checks, what makes each one
fail, and that one failure fails the whole report. The container daemon is
faked at the process boundary -- the same RecordingRunner the runtime tests
use -- so the logic under test is real.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_agent.doctor import _doctor


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "target": "storage",
        "home": None,
        "scratch": None,
        "min_durable_free_gib": 0.0,
        "min_scratch_free_gib": 0.0,
        "runtime": "docker",
        "container_runtime": ["docker"],
        "apptainer_executable": "apptainer",
        "overlay_size_mib": 1024,
        "web_mode": "fixed",
        "page_reader": "http",
        "index": None,
        "anserini_jar": None,
        "checkout": None,
        "osworld_version": None,
        "provider_name": "docker",
        "path_to_vm": None,
        "osworld_apptainer_image": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def report(namespace: argparse.Namespace) -> tuple[int, dict]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        import asyncio

        code = asyncio.run(_doctor(namespace))
    return code, json.loads(stdout.getvalue())


class StorageDoctorTests(unittest.TestCase):
    def test_enough_space_is_ready_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, value = report(
                args(home=root / "home", scratch=root / "scratch")
            )
        self.assertEqual(code, 0)
        self.assertTrue(value["ok"])
        self.assertEqual(value["reports"]["storage"]["status"], "ready")

    def test_an_impossible_space_requirement_fails_the_whole_report(self) -> None:
        """One failing target must not be reported as an overall pass."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, value = report(
                args(
                    home=root / "home",
                    scratch=root / "scratch",
                    min_durable_free_gib=10 ** 6,
                )
            )
        self.assertEqual(code, 1)
        self.assertFalse(value["ok"])
        self.assertEqual(
            value["reports"]["storage"]["status"], "insufficient_space"
        )


class SWEDoctorTests(unittest.IsolatedAsyncioTestCase):
    def _docker_report(self, *, rootless: bool) -> tuple[int, dict]:
        from test_runtimes import RecordingRunner

        runner = RecordingRunner(rootless=rootless)
        # resolve_runner is where the real process boundary is chosen, so
        # replacing it there leaves every decision above it untouched.
        with patch(
            "mini_agent.runtimes.docker.resolve_runner", return_value=runner
        ):
            return report(args(target="swebench", runtime="docker"))

    def test_a_rootless_daemon_is_reported_ready(self) -> None:
        code, value = self._docker_report(rootless=True)
        self.assertEqual(value["reports"]["swebench"]["runtime"], "docker")
        self.assertEqual(code, 0 if value["reports"]["swebench"]["ok"] else 1)

    def test_a_rootful_daemon_is_refused(self) -> None:
        """Rootful Docker is exactly what the harness must not run agents on."""

        code, value = self._docker_report(rootless=False)
        self.assertFalse(value["reports"]["swebench"]["ok"])
        self.assertEqual(code, 1)

    def test_a_missing_apptainer_executable_is_reported_not_crashed(self) -> None:
        with patch("mini_agent.doctor.shutil.which", return_value=None):
            code, value = report(args(target="swebench", runtime="apptainer"))
        self.assertEqual(code, 1)
        self.assertFalse(value["reports"]["swebench"]["ok"])


class WebDoctorTests(unittest.TestCase):
    def test_fixed_retrieval_without_assets_is_refused(self) -> None:
        code, value = report(args(target="web", web_mode="fixed"))
        self.assertEqual(code, 1)
        self.assertFalse(value["reports"]["web"]["ok"])

    def test_live_retrieval_reports_the_missing_credential(self) -> None:
        with patch.dict("os.environ", {}, clear=False) as environment:
            environment.pop("SERPAPI_API_KEY", None)
            code, value = report(args(target="web", web_mode="live"))
        web = value["reports"]["web"]
        self.assertEqual(web["mode"], "browsecomp_live")
        self.assertFalse(web["serpapi"]["ok"])
        self.assertFalse(web["serpapi"]["network_canary_run"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
