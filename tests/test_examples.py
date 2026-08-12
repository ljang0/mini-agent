from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_agent.environments.web import JsonlSearchBackend

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPANION_SRC = REPO_ROOT / "packaging" / "multi-mini-agent" / "src"


class ExamplesTests(unittest.TestCase):
    def test_library_quickstart_runs_offline(self) -> None:
        completed = subprocess.run(
            (sys.executable, str(REPO_ROOT / "examples" / "library_quickstart.py")),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "answer: The environment replied with a greeting.", completed.stdout
        )

    def test_quickstart_corpus_loads_and_answers_the_readme_query(self) -> None:
        backend = JsonlSearchBackend(REPO_ROOT / "examples" / "corpus.jsonl")
        self.assertEqual(len(backend.documents), 10)
        hits = backend.search("wind turbine furls", k=5)
        self.assertEqual(hits[0]["ref"], "wind-1")

    def test_multi_mini_agent_wrapper_injects_the_delegation_flag(self) -> None:
        sys.path.insert(0, str(COMPANION_SRC))
        try:
            import multi_mini_agent

            with patch("mini_agent.cli.main", return_value=0) as base:
                self.assertEqual(
                    multi_mini_agent.main(["profile", "--application", "web"]), 0
                )
                multi_mini_agent.main(
                    ["run", "--multi-agent", "--task", "t", "--model", "openai/x"]
                )
                multi_mini_agent.main(["doctor", "--target", "storage"])
            first, second, third = (
                call.args[0] for call in base.call_args_list
            )
            self.assertEqual(first[:2], ["profile", "--multi-agent"])
            self.assertEqual(second.count("--multi-agent"), 1)
            self.assertNotIn("--multi-agent", third)
        finally:
            sys.path.remove(str(COMPANION_SRC))


if __name__ == "__main__":
    unittest.main()
