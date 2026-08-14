"""The package's own entry points: the console path and the lazy re-exports.

These are the first things a user touches and the last things a test suite
usually covers -- `python -m mini_agent` had never been executed by a test,
and the lazy shims exist precisely so that importing a protocol does not drag
in Docker or Apptainer code, which is only true if it is checked.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from mini_agent import __version__


class EntryPointTests(unittest.TestCase):
    def test_python_dash_m_reports_the_version(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "mini_agent", "--version"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(__version__, done.stdout)

    def test_python_dash_m_reports_a_usage_error_without_a_command(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "mini_agent"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("usage", done.stderr.lower())


class LazyExportTests(unittest.TestCase):
    def test_a_runtime_protocol_import_pulls_in_no_provisioning_code(self) -> None:
        source = (
            "import sys\n"
            "import mini_agent.runtimes as r\n"
            "assert r.SandboxRuntime is not None\n"
            "heavy = ('.docker', '.apptainer')\n"
            "loaded = [m for m in sys.modules if m.endswith(heavy)]\n"
            "assert not loaded, loaded\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_lazy_name_resolves_on_first_use(self) -> None:
        import mini_agent.runtimes as runtimes

        self.assertTrue(callable(runtimes.DockerRuntime.start))

    def test_an_unknown_name_raises_attribute_error(self) -> None:
        import mini_agent.environments as environments
        import mini_agent.runtimes as runtimes

        for module in (runtimes, environments):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(AttributeError, "has no attribute"):
                    module.NotAThing

    def test_every_declared_export_actually_resolves(self) -> None:
        """__all__ is a promise; check the package can keep it."""

        import mini_agent.environments as environments
        import mini_agent.runtimes as runtimes

        for module in (runtimes, environments):
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIsNotNone(getattr(module, name))


if __name__ == "__main__":
    unittest.main()
