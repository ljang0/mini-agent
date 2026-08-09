from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from scaffoldlab.applications import (
    APPLICATIONS,
    IMPLEMENTATIONS,
    get_implementation,
    list_applications,
    list_implementations,
    normalize_harness_specs,
    resolve_application_config,
)
from scaffoldlab.cli import _build_harnesses, main


class ApplicationRegistryTests(unittest.TestCase):
    def test_catalog_has_three_top_level_applications_and_unique_keys(self) -> None:
        self.assertEqual(list_applications(), ("browser", "computer-use", "swe"))
        keys = [profile.key for profile in IMPLEMENTATIONS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 3)
        for application, profiles in APPLICATIONS.items():
            self.assertTrue(profiles)
            self.assertTrue(
                all(profile.application == application for profile in profiles)
            )
            self.assertTrue(all(profile.sources for profile in profiles))

    def test_lookup_accepts_local_or_fully_qualified_id(self) -> None:
        local = get_implementation("swe", "single-agent-control")
        qualified = get_implementation("swe", "swe/single-agent-control")
        self.assertIs(local, qualified)
        with self.assertRaisesRegex(ValueError, "belongs to application"):
            get_implementation("browser", "swe/single-agent-control")

    def test_legacy_config_is_preserved(self) -> None:
        legacy = {
            "harnesses": ["single"],
            "limits": {"max_model_calls": 1},
        }
        resolved, selection, warnings = resolve_application_config(
            legacy, provider="openai-responses"
        )
        self.assertEqual(resolved, legacy)
        self.assertIsNone(selection)
        self.assertEqual(warnings, [])

    def test_profile_auto_fills_exact_harnesses(self) -> None:
        config: dict[str, Any] = {
            "application": "browser",
            "implementation": "openai-hosted-multi-agent-functions",
            "environment": {"type": "browser"},
        }
        resolved, selection, warnings = resolve_application_config(
            config, provider="openai-responses"
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(
            selection.implementation.key,
            config["application"] + "/" + config["implementation"],
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("published protocol boundary", warnings[0])
        self.assertEqual(
            resolved["harnesses"],
            [
                {
                    "name": "openai_hosted_multi_agent",
                    "options": {"max_concurrent_subagents": 3},
                }
            ],
        )
        harnesses = _build_harnesses(resolved)
        self.assertEqual(harnesses[0].name, "openai_hosted_multi_agent")
        self.assertEqual(getattr(harnesses[0], "max_concurrent_subagents"), 3)

    def test_nested_application_object_is_supported(self) -> None:
        resolved, selection, _ = resolve_application_config(
            {
                "application": {
                    "name": "swe",
                    "implementation": "single-agent-control",
                },
                "environment": {"type": "swe"},
            },
            provider="anthropic-messages",
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.name, "swe")
        self.assertEqual(resolved["harnesses"], [{"name": "single", "options": {}}])

    def test_explicit_matching_harness_is_accepted(self) -> None:
        resolved, selection, _ = resolve_application_config(
            {
                "application": "swe",
                "implementation": "single-agent-control",
                "harnesses": ["single"],
                "environment": {"type": "swe"},
            },
            provider="openai-compatible-chat",
        )
        self.assertEqual(resolved["harnesses"], ["single"])
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.implementation.profile_id, "single-agent-control")

    def test_profile_rejects_provider_environment_and_harness_drift(self) -> None:
        base: dict[str, Any] = {
            "application": "browser",
            "implementation": "openai-hosted-multi-agent-functions",
            "environment": {"type": "browser"},
        }
        with self.assertRaisesRegex(ValueError, "does not support provider"):
            resolve_application_config(base, provider="anthropic-messages")

        wrong_environment = {**base, "environment": {"type": "computer"}}
        with self.assertRaisesRegex(ValueError, "requires environment type browser"):
            resolve_application_config(wrong_environment, provider="openai-responses")

        wrong_harness = {**base, "harnesses": ["single"]}
        with self.assertRaisesRegex(ValueError, "requires exact harnesses"):
            resolve_application_config(wrong_harness, provider="openai-responses")

    def test_none_environment_and_catalog_only_are_enforced(self) -> None:
        managed = {
            "application": "browser",
            "implementation": "anthropic-managed-web-research",
        }
        resolved, _, _ = resolve_application_config(
            managed, provider="anthropic-managed-agents"
        )
        self.assertNotIn("environment", resolved)
        with self.assertRaisesRegex(ValueError, "requires environment type none"):
            resolve_application_config(
                {**managed, "environment": {"type": "browser"}},
                provider="anthropic-managed-agents",
            )
        with self.assertRaisesRegex(ValueError, "catalog sentinel"):
            resolve_application_config(
                {**managed, "environment": {"type": "none"}},
                provider="anthropic-managed-agents",
            )

        with self.assertRaisesRegex(ValueError, "catalog-only"):
            resolve_application_config(
                {
                    "application": "browser",
                    "implementation": "anthropic-opus5-cowork-safety",
                },
                provider="anthropic-messages",
            )

    def test_simulation_selection_emits_explicit_warning(self) -> None:
        _, _, warnings = resolve_application_config(
            {
                "application": "browser",
                "implementation": "anthropic-fable5-team-3",
                "environment": {"type": "browser"},
            },
            provider="anthropic-messages",
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("topology simulation", warnings[0])
        self.assertIn("not a 1:1 reproduction", warnings[0])

    def test_fable_task_specific_async_limits_are_not_cross_labeled(self) -> None:
        browser = get_implementation("browser", "anthropic-fable5-async-browsecomp")
        swe = get_implementation("swe", "anthropic-fable5-async-programbench")
        self.assertIn("no BrowseComp spawn cap", browser.unavailable_components[0])
        self.assertIn("four-concurrent/twenty-total", swe.exact_components[0])

    def test_rlm_local_and_upstream_are_distinct_profiles(self) -> None:
        local = get_implementation("swe", "rlm-0.1.3-contract")
        upstream = get_implementation("swe", "rlm-0.1.3-upstream")
        self.assertEqual(local.fidelity, "source_matched_reimplementation")
        self.assertEqual(local.providers[0], "openai-responses")
        self.assertEqual(upstream.fidelity, "upstream_runtime_adapter")
        self.assertEqual(upstream.providers, ("rlm-upstream",))
        self.assertEqual(upstream.environment_types, ("none",))
        self.assertEqual(
            upstream.sources[0].revision,
            "72d6940142ddfb84ee6be573dc999a37e633e671",
        )

    def test_canonical_application_configs_resolve_without_harness_drift(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config_count = 0
        for application in list_applications():
            config_directory = repository / application / "configs"
            self.assertTrue(config_directory.is_dir())
            for config_path in sorted(config_directory.glob("*.json")):
                config_count += 1
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertNotIn("harnesses", config, config_path)
                self.assertNotIn("implementation", config, config_path)
                selection = config.get("application")
                self.assertIsInstance(selection, dict, config_path)
                assert isinstance(selection, dict)
                self.assertEqual(selection.get("name"), application, config_path)
                profile = get_implementation(
                    application, str(selection.get("implementation"))
                )
                self.assertIn(profile.status, {"runnable", "simulation"}, config_path)
                self.assertTrue(profile.providers, config_path)
                resolved, selected, _ = resolve_application_config(
                    config, provider=profile.providers[0]
                )
                self.assertIsNotNone(selected, config_path)
                self.assertEqual(
                    normalize_harness_specs(resolved["harnesses"]),
                    profile.harnesses,
                    config_path,
                )
        self.assertEqual(config_count, 26)

    def test_list_implementations_filters_by_application(self) -> None:
        browser_profiles = list_implementations("browser")
        self.assertTrue(browser_profiles)
        self.assertTrue(
            all(profile.application == "browser" for profile in browser_profiles)
        )
        with self.assertRaisesRegex(ValueError, "unknown application"):
            list_implementations("unknown")


class ApplicationCLITests(unittest.TestCase):
    def test_list_commands_support_human_and_json_output(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-applications"]), 0)
        self.assertEqual(
            output.getvalue().splitlines(), ["browser", "computer-use", "swe"]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "list-implementations",
                        "--application",
                        "computer-use",
                        "--json",
                    ]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertTrue(payload)
        self.assertTrue(all(item["application"] == "computer-use" for item in payload))
        self.assertTrue(all(item["sources"] for item in payload))

    def test_validate_resolves_application_and_records_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.jsonl"
            config = root / "config.json"
            tasks.write_text(
                json.dumps({"task_id": "one", "prompt": "Solve this."}) + "\n",
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "application": {
                            "name": "swe",
                            "implementation": "single-agent-control",
                        },
                        "environment": {
                            "type": "swe",
                            "protocol": "generic",
                            "workspace_mode": "copy",
                            "isolation": "per_agent",
                        },
                        "limits": {"max_model_calls": 1},
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "validate",
                            "--tasks",
                            str(tasks),
                            "--config",
                            str(config),
                            "--provider",
                            "openai-responses",
                        ]
                    ),
                    0,
                )
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["harnesses"], ["single"])
            self.assertEqual(payload["application"]["name"], "swe")
            self.assertEqual(
                payload["application"]["implementation"]["key"],
                "swe/single-agent-control",
            )


if __name__ == "__main__":
    unittest.main()
