"""The isolated grader-runtime probe, executed the way the harness runs it.

The probe never gets imported -- `grading.py` reads its text and runs it with
`python -I -c <source>` inside the official grader runtime. That means no
ordinary test reaches it, and until now nothing exercised the code that
decides whether a grader runtime is the one that was recorded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib import resources
from pathlib import Path


SOURCE = resources.files("mini_agent").joinpath("_grader_probe.py").read_text("utf-8")


def probe(required: dict[str, str], *, python: str = sys.executable) -> dict:
    """Run the probe exactly as the harness does and read its report."""

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "probe.json"
        completed = subprocess.run(
            [python, "-I", "-c", SOURCE, str(output), json.dumps(required)],
            capture_output=True,
            timeout=120,
        )
        if not output.exists():
            raise AssertionError(
                f"probe wrote nothing (exit {completed.returncode}): "
                f"{completed.stderr.decode()[:400]}"
            )
        return json.loads(output.read_text())


def _clean_runtime() -> tuple[str, str]:
    """An interpreter plus a package it reaches without traversing a symlink."""

    for python in ("/usr/bin/python3.12", "/usr/bin/python3", sys.executable):
        for name in ("setuptools", "pip", "packaging"):
            probe_source = (
                "import importlib.util,os,sys\n"
                f"spec=importlib.util.find_spec({name!r})\n"
                "ok=False\n"
                "if spec and spec.origin and spec.submodule_search_locations:\n"
                "    o=os.path.abspath(spec.origin)\n"
                "    r=os.path.abspath(list(spec.submodule_search_locations)[0])\n"
                "    ok=os.path.realpath(o)==o and os.path.realpath(r)==r\n"
                "sys.exit(0 if ok else 1)\n"
            )
            done = subprocess.run([python, "-I", "-c", probe_source],
                                  capture_output=True, timeout=60)
            if done.returncode == 0:
                return python, name
    raise unittest.SkipTest("no symlink-free runtime available")


class GraderProbeTests(unittest.TestCase):
    def test_a_matching_runtime_reports_its_exact_identity(self) -> None:
        # Needs an interpreter whose site-packages is reached without a
        # symlink; this project's venv has lib64 -> lib, which the probe
        # rejects on purpose (see the symlink test below).
        python, name = _clean_runtime()
        version = json.loads(
            subprocess.run(
                [python, "-I", "-c",
                 f"import importlib.metadata as m,json;print(json.dumps("
                 f"m.version({name!r})))"],
                capture_output=True, text=True, timeout=60).stdout)
        report = probe({name: version}, python=python)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "mini-agent-isolated-grader-runtime-v1")
        self.assertEqual(report["packages"], {name: version})
        # The module identity is what binds a grade to one installed tree.
        module = report["modules"][name]
        self.assertTrue(module["origin"].endswith("__init__.py"))
        self.assertEqual(
            module["origin"], str(Path(module["package_root"]) / "__init__.py")
        )

    def test_a_package_behind_a_symlink_is_refused(self) -> None:
        """A symlinked package root means the graded tree is not pinned.

        The condition is built here rather than borrowed from whatever the
        interpreter happens to sit on: an earlier version of this test relied
        on this project's venv having lib64 -> lib, which is true locally and
        false on CI. Dropping -I is what lets PYTHONPATH reach the fixture;
        every check the probe makes is otherwise the same.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "realpkg"
            real.mkdir()
            (real / "__init__.py").write_text("")
            (root / "linkpkg").symlink_to(real, target_is_directory=True)
            info = root / "linkpkg-1.0.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: linkpkg\nVersion: 1.0\n"
            )
            output = root / "probe.json"
            subprocess.run(
                [sys.executable, "-c", SOURCE, str(output),
                 json.dumps({"linkpkg": "1.0"})],
                capture_output=True, timeout=120,
                env={"PYTHONPATH": str(root), "PATH": os.environ.get("PATH", "")},
            )
            report = json.loads(output.read_text())

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["error"], "module")
        self.assertEqual(report["name"], "linkpkg")

    def test_a_missing_package_is_named_not_guessed(self) -> None:
        report = probe({"definitely-not-installed-xyz": "1.0.0"})

        self.assertFalse(report["ok"])
        self.assertEqual(report["error"], "missing")
        self.assertEqual(report["name"], "definitely-not-installed-xyz")
        self.assertEqual(report["expected"], "1.0.0")

    def test_a_version_mismatch_reports_both_versions(self) -> None:
        report = probe({"httpx": "0.0.0-not-this"})

        self.assertFalse(report["ok"])
        self.assertEqual(report["error"], "version")
        self.assertEqual(report["expected"], "0.0.0-not-this")
        self.assertNotEqual(report["observed"], "0.0.0-not-this")

    def test_a_malformed_request_is_rejected(self) -> None:
        for bad in ({}, {"name": 1}):
            with self.subTest(bad=bad):
                report = probe(bad)  # type: ignore[arg-type]
                self.assertFalse(report["ok"])
                self.assertEqual(report["error"], "request")

    def test_the_probe_refuses_to_overwrite_an_existing_report(self) -> None:
        """O_EXCL: a stale report must never be mistaken for a fresh one."""

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe.json"
            output.write_text("stale")
            subprocess.run(
                [sys.executable, "-I", "-c", SOURCE, str(output), "{}"],
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(output.read_text(), "stale")


if __name__ == "__main__":
    unittest.main()
