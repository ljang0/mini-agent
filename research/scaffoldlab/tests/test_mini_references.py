from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scaffoldlab.applications import get_study, list_implementations
from scaffoldlab.cli import main as legacy_main

from mini_agent.references import ReferenceRuntime, get_reference, list_references


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
        self.assertEqual(manifest["legacy_key"], "browser/browser-use-0.13.7-upstream")
        self.assertEqual(manifest["execution_mode"], "reference")
        self.assertEqual(manifest["fidelity"], "upstream_runtime_adapter")
        self.assertEqual(manifest["delegate"], "scaffoldlab.cli")

    def test_studies_cannot_masquerade_as_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an implementation"):
            get_reference("swe", "single-agent-control")

    def test_public_constructor_rejects_wrong_domain_and_non_reference(self) -> None:
        profile = get_reference("web", "openai-hosted-web-search").profile
        with self.assertRaisesRegex(ValueError, "belongs to"):
            ReferenceRuntime("swe", profile)
        with self.assertRaisesRegex(ValueError, "not a runnable reference"):
            ReferenceRuntime("swe", get_study("swe", "single-agent-control"))

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
                        "name": "web",
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
                "--expected-application-key",
                "browser/openai-hosted-web-search",
                "--expected-config-sha256",
                hashlib.sha256(self.config.read_bytes()).hexdigest(),
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
        self.assertIn("--expected-application-key", arguments)
        self.assertIn("--expected-config-sha256", arguments)
        self.assertFalse((Path.cwd() / "should-never-exist").exists())

    def test_identity_arguments_cannot_be_overridden(self) -> None:
        for argument in (
            "--tasks",
            "--task",
            "--config=other.json",
            "--confi=other.json",
            "--provider",
            "--prov=xai-responses",
            "--output=/tmp/other",
            "--out=/tmp/other",
            "--expected-app=browser/other",
            "--expected-config=deadbeef",
            "--",
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
                        "name": "cua",
                        "implementation": "macu-upstream-generic-vm",
                    },
                    "limits": {"max_model_calls": 1},
                }
            )
        )
        other_reference = get_reference("web", "macu-upstream")
        with self.assertRaisesRegex(ValueError, "not implementation profile"):
            other_reference.run(
                tasks=self.tasks,
                config=other,
                output=self.output,
            )

    def test_legacy_validation_still_runs_unchanged(self) -> None:
        with redirect_stdout(io.StringIO()):
            result = self.reference.validate(tasks=self.tasks, config=self.config)
        self.assertEqual(result, 0)

    def test_direct_legacy_validation_needs_no_internal_binding_flags(self) -> None:
        with redirect_stdout(io.StringIO()):
            result = legacy_main(
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
        self.assertEqual(result, 0)

    def test_reference_config_remains_utf8_only(self) -> None:
        utf16 = self.config.parent / "utf16.json"
        utf16.write_bytes(self.config.read_text(encoding="utf-8").encode("utf-16"))
        with self.assertRaises(UnicodeDecodeError):
            self.reference.validate(tasks=self.tasks, config=utf16)

    def test_empty_runtime_argument_is_forwarded_without_rewriting(self) -> None:
        with patch("scaffoldlab.cli.main", return_value=0) as main:
            self.reference.run(
                tasks=self.tasks,
                config=self.config,
                output=self.output,
                arguments=("--model", ""),
            )
        self.assertIn("", main.call_args.args[0])

    def test_config_swap_after_authorization_fails_sha_check(self) -> None:
        original = self.config.read_text(encoding="utf-8")

        def swap_then_run(arguments: tuple[str, ...]) -> int:
            self.config.write_text(
                json.dumps(
                    {
                        "application": {
                            "name": "web",
                            "implementation": "xai-hosted-web-research-4",
                        },
                        "limits": {"max_model_calls": 1},
                    }
                ),
                encoding="utf-8",
            )
            return legacy_main(arguments)

        try:
            with patch("scaffoldlab.cli.main", side_effect=swap_then_run):
                errors = io.StringIO()
                with redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as stopped:
                        self.reference.validate(tasks=self.tasks, config=self.config)
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("config SHA-256 changed", errors.getvalue())
        finally:
            self.config.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
