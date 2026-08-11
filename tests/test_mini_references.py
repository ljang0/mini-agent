from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.applications import list_implementations

from mini_agent.references import get_reference, list_references


class ReferenceCatalogTests(unittest.TestCase):
    def test_catalog_is_a_lossless_application_rename(self) -> None:
        references = list_references()
        implementations = list_implementations()
        self.assertEqual(len(references), len(implementations))
        self.assertEqual(
            {reference.profile.key for reference in references},
            {profile.key for profile in implementations},
        )
        self.assertEqual(
            {reference.application for reference in references},
            {"swe", "web", "cua"},
        )

    def test_manifest_keeps_legacy_fidelity_claim(self) -> None:
        reference = get_reference("web", "browser-use-0.13.7-upstream")
        manifest = reference.manifest()
        self.assertEqual(manifest["application"], "web")
        self.assertEqual(manifest["legacy_application"], "browser")
        self.assertEqual(manifest["key"], "web/browser-use-0.13.7-upstream")
        self.assertEqual(manifest["fidelity"], "upstream_runtime_adapter")
        self.assertEqual(manifest["delegate"], "scaffoldlab.cli")

    def test_studies_cannot_masquerade_as_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an implementation"):
            get_reference("swe", "single-agent-control")

    def test_application_names_are_the_mini_agent_names(self) -> None:
        self.assertTrue(list_references("web"))
        self.assertTrue(list_references("cua"))
        with self.assertRaisesRegex(ValueError, "swe, web, or cua"):
            list_references("browser")


class ReferenceDelegationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.tasks = root / "tasks.jsonl"
        self.tasks.write_text('{"id":"one","prompt":"answer"}\n')
        self.config = root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "application": {
                        "name": "browser",
                        "implementation": "openai-hosted-web-search",
                    },
                    "limits": {"max_model_calls": 1},
                }
            )
        )
        self.output = root / "output"
        self.reference = get_reference("web", "openai-hosted-web-search")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validate_delegates_to_legacy_python_cli(self) -> None:
        with patch("scaffoldlab.cli.main", return_value=0) as main:
            result = self.reference.validate(tasks=self.tasks, config=self.config)
        self.assertEqual(result, 0)
        main.assert_called_once_with(
            (
                "validate",
                "--tasks",
                str(self.tasks),
                "--config",
                str(self.config),
                "--provider",
                "openai-responses",
            )
        )

    def test_run_passes_provider_options_as_literal_argv(self) -> None:
        suspicious = "model;$(touch should-never-exist)"
        with patch("scaffoldlab.cli.main", return_value=7) as main:
            result = self.reference.run(
                tasks=self.tasks,
                config=self.config,
                output=self.output,
                arguments=("--model", suspicious, "--overwrite"),
            )
        self.assertEqual(result, 7)
        arguments = main.call_args.args[0]
        self.assertEqual(arguments[0], "run")
        self.assertIn(suspicious, arguments)
        self.assertEqual(arguments[-2:], ("--output", str(self.output)))
        self.assertFalse((Path.cwd() / "should-never-exist").exists())

    def test_identity_arguments_cannot_be_overridden(self) -> None:
        for argument in (
            "--tasks",
            "--config=other.json",
            "--provider",
            "--output=/tmp/other",
        ):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, "owned by the adapter"):
                    self.reference.run(
                        tasks=self.tasks,
                        config=self.config,
                        output=self.output,
                        arguments=(argument,),
                    )

    def test_provider_and_config_identity_fail_closed_before_delegation(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support provider"):
            self.reference.run(
                tasks=self.tasks,
                config=self.config,
                output=self.output,
                provider="xai-responses",
            )
        other = self.config.parent / "other.json"
        other.write_text(
            json.dumps(
                {
                    "application": {
                        "name": "computer-use",
                        "implementation": "macu-upstream-generic-vm",
                    },
                    "limits": {"max_model_calls": 1},
                }
            )
        )
        other_reference = get_reference("web", "macu-upstream")
        with self.assertRaisesRegex(ValueError, "not reference"):
            other_reference.run(
                tasks=self.tasks,
                config=other,
                output=self.output,
            )

    def test_legacy_validation_still_runs_unchanged(self) -> None:
        with redirect_stdout(io.StringIO()):
            result = self.reference.validate(tasks=self.tasks, config=self.config)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
