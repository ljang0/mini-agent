from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from mini_agent.cli import main
from mini_agent.references import get_reference
from scaffoldlab.cli import main as legacy_main


class MigrationCLITests(unittest.TestCase):
    def test_catalog_and_frontier_commands_expose_complete_migration(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("catalog", "--json")), 0)
        profiles = json.loads(output.getvalue())
        self.assertEqual(len(profiles), 55)
        self.assertEqual(
            {profile["execution_mode"] for profile in profiles},
            {"reference", "study", "unavailable"},
        )

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("frontiers", "--json")), 0)
        frontiers = json.loads(output.getvalue())
        self.assertEqual(len(frontiers), 18)
        self.assertEqual(
            sum(len(source["application_statuses"]) for source in frontiers), 54
        )

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("applications", "--json")), 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            [
                {"application": "web", "profile_count": 19},
                {"application": "cua", "profile_count": 8},
                {"application": "swe", "profile_count": 28},
            ],
        )

    def test_harness_listing_is_the_preserved_listing(self) -> None:
        mini_output = io.StringIO()
        legacy_output = io.StringIO()
        with redirect_stdout(mini_output):
            self.assertEqual(main(("harnesses",)), 0)
        with redirect_stdout(legacy_output):
            self.assertEqual(legacy_main(("list-harnesses",)), 0)
        self.assertEqual(mini_output.getvalue(), legacy_output.getvalue())

    def test_every_domain_has_an_inspectable_exact_reference(self) -> None:
        selections = (
            ("web", "openai-hosted-web-search"),
            ("cua", "openai-ga-computer-single"),
            ("swe", "prime-agent-0.7.1"),
        )
        for application, name in selections:
            with self.subTest(application=application):
                output = io.StringIO()
                with redirect_stdout(output):
                    result = main(
                        (
                            "implementation",
                            "--application",
                            application,
                            "--name",
                            name,
                        )
                    )
                self.assertEqual(result, 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["execution_mode"], "reference")
                self.assertEqual(payload["application"], application)

    def test_validate_reference_reaches_preserved_evaluator_in_each_domain(
        self,
    ) -> None:
        selections = (
            ("web", "openai-hosted-web-search"),
            ("cua", "openai-ga-computer-single"),
            ("swe", "prime-agent-0.7.1"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.jsonl"
            tasks.write_text('{"id":"one","prompt":"answer"}\n', encoding="utf-8")
            for application, name in selections:
                with self.subTest(application=application):
                    reference = get_reference(application, name)
                    config = root / f"{application}.json"
                    environment = None
                    if application == "cua":
                        environment = {
                            "type": "computer",
                            "protocol": "auto",
                            "isolation": "per_agent",
                            "allowed_hosts": ["example.com"],
                            "start_url": "https://example.com",
                            "headless": True,
                            "viewport_width": 1440,
                            "viewport_height": 900,
                            "max_tool_output_bytes": 262144,
                        }
                    config.write_text(
                        json.dumps(
                            {
                                "application": {
                                    "name": application,
                                    "implementation": name,
                                },
                                "environment": environment,
                                "limits": {"max_model_calls": 1},
                            }
                        ),
                        encoding="utf-8",
                    )
                    with redirect_stdout(io.StringIO()):
                        result = main(
                            (
                                "validate-reference",
                                "--application",
                                application,
                                "--implementation",
                                name,
                                "--tasks",
                                str(tasks),
                                "--config",
                                str(config),
                                "--provider",
                                reference.providers[0],
                            )
                        )
                    self.assertEqual(result, 0)

    def test_eval_reference_forwards_only_explicit_runtime_arguments(self) -> None:
        reference = Mock()
        reference.run.return_value = 0
        with patch("mini_agent.references.get_reference", return_value=reference):
            result = main(
                (
                    "eval-reference",
                    "--application",
                    "swe",
                    "--implementation",
                    "example",
                    "--tasks",
                    "tasks.jsonl",
                    "--config",
                    "config.json",
                    "--output",
                    "runs/example",
                    "--",
                    "--model",
                    "example-model",
                )
            )
        self.assertEqual(result, 0)
        reference.run.assert_called_once_with(
            tasks=Path("tasks.jsonl"),
            config=Path("config.json"),
            output=Path("runs/example"),
            provider=None,
            arguments=("--model", "example-model"),
        )

    def test_identity_sensitive_clis_reject_abbreviated_options(self) -> None:
        mini_arguments = (
            "validate-reference",
            "--app",
            "web",
            "--implementation",
            "openai-hosted-web-search",
            "--tasks",
            "tasks.jsonl",
            "--config",
            "config.json",
        )
        legacy_arguments = (
            "validate",
            "--tas",
            "tasks.jsonl",
            "--config",
            "config.json",
            "--provider",
            "openai-responses",
        )
        for command in (
            lambda: main(mini_arguments),
            lambda: legacy_main(legacy_arguments),
        ):
            with self.subTest(command=command):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        command()
                self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
