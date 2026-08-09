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
    GAPS,
    IMPLEMENTATIONS,
    PROFILES,
    STUDIES,
    get_implementation,
    get_profile,
    get_study,
    list_applications,
    list_gaps,
    list_implementations,
    list_profiles,
    list_studies,
    normalize_harness_specs,
    resolve_application_config,
)
from scaffoldlab.applications.base import HarnessSignature, SourceArtifact, profile
from scaffoldlab.cli import _build_harnesses, main


class ApplicationRegistryTests(unittest.TestCase):
    def test_catalog_has_three_top_level_applications_and_unique_keys(self) -> None:
        self.assertEqual(list_applications(), ("browser", "computer-use", "swe"))
        keys = [profile.key for profile in PROFILES]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 3)
        self.assertEqual(set(PROFILES), set(IMPLEMENTATIONS + STUDIES + GAPS))
        self.assertTrue(
            all(profile.catalog_kind == "implementation" for profile in IMPLEMENTATIONS)
        )
        self.assertTrue(all(profile.exactness_scope for profile in IMPLEMENTATIONS))
        self.assertTrue(all(profile.catalog_kind == "study" for profile in STUDIES))
        self.assertTrue(all(profile.catalog_kind == "gap" for profile in GAPS))
        for application, profiles in APPLICATIONS.items():
            self.assertTrue(profiles)
            self.assertTrue(
                all(profile.application == application for profile in profiles)
            )
            self.assertTrue(all(profile.sources for profile in profiles))

    def test_upstream_implementations_require_a_source_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned source revision"):
            profile(
                application="swe",
                profile_id="version-only-runtime",
                title="Version-only runtime",
                fidelity="upstream_runtime_adapter",
                runtime_owner="upstream",
                harnesses=(HarnessSignature.create("single"),),
                providers=("openai-responses",),
                environment_types=("none",),
                sources=(
                    SourceArtifact(
                        title="Release page",
                        url="https://example.com/release",
                        published="2026-08-10",
                        version="1.0.0",
                    ),
                ),
            )

    def test_lookup_accepts_local_or_fully_qualified_id(self) -> None:
        local = get_implementation("swe", "rlm-0.1.3-upstream")
        qualified = get_implementation("swe", "swe/rlm-0.1.3-upstream")
        self.assertIs(local, qualified)
        with self.assertRaisesRegex(ValueError, "belongs to application"):
            get_implementation("browser", "swe/rlm-0.1.3-upstream")
        with self.assertRaisesRegex(ValueError, "is a study, not an implementation"):
            get_implementation("swe", "single-agent-control")

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
                    "study": "single-agent-control",
                },
                "environment": {"type": "swe"},
            },
            provider="anthropic-messages",
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.name, "swe")
        self.assertEqual(selection.selection_kind, "study")
        with self.assertRaisesRegex(ValueError, "not an implementation"):
            _ = selection.implementation
        self.assertEqual(resolved["harnesses"], [{"name": "single", "options": {}}])

    def test_explicit_matching_harness_is_accepted(self) -> None:
        resolved, selection, _ = resolve_application_config(
            {
                "application": "swe",
                "study": "single-agent-control",
                "harnesses": ["single"],
                "environment": {"type": "swe"},
            },
            provider="openai-compatible-chat",
        )
        self.assertEqual(resolved["harnesses"], ["single"])
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.profile.profile_id, "single-agent-control")

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

        wrong_fidelity = {
            **base,
            "metadata": {"fidelity": "topology_simulation"},
        }
        with self.assertRaisesRegex(ValueError, "requires fidelity"):
            resolve_application_config(wrong_fidelity, provider="openai-responses")

        wrong_revision = {
            "application": {
                "name": "swe",
                "implementation": "rlm-0.1.3-upstream",
            },
            "metadata": {
                "fidelity": "upstream_runtime_adapter",
                "revision": "0" * 40,
            },
        }
        with self.assertRaisesRegex(ValueError, "does not cite metadata revision"):
            resolve_application_config(wrong_revision, provider="rlm-upstream")

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

        with self.assertRaisesRegex(ValueError, "is a gap, not an implementation"):
            get_implementation("browser", "anthropic-opus5-cowork-safety")

    def test_simulation_selection_emits_explicit_warning(self) -> None:
        _, _, warnings = resolve_application_config(
            {
                "application": "browser",
                "study": "anthropic-fable5-team-3",
                "environment": {"type": "browser"},
            },
            provider="anthropic-messages",
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("topology simulation", warnings[0])
        self.assertIn("not a 1:1 reproduction", warnings[0])

    def test_mythos_card_profiles_keep_legacy_ids_without_fable_model_claims(
        self,
    ) -> None:
        browser = get_study("browser", "anthropic-fable5-async-browsecomp")
        swe = get_study("swe", "anthropic-fable5-async-programbench")
        self.assertIn("no BrowseComp spawn cap", browser.unavailable_components[0])
        self.assertIn("four-concurrent/twenty-total", swe.exact_components[0])
        for candidate in (browser, swe):
            self.assertIn("Mythos 5", candidate.title)
            self.assertNotIn("Fable", candidate.title)
            self.assertEqual(candidate.model_families, ("Claude Mythos 5",))
            self.assertIn("Mythos 5", candidate.sources[0].version)

    def test_anthropic_exact_boundaries_are_narrowly_scoped(self) -> None:
        computer = get_implementation(
            "computer-use", "anthropic-computer-20251124-single"
        )
        self.assertIn("wire schema", computer.title)
        self.assertTrue(
            any("local Playwright" in gap for gap in computer.unavailable_components)
        )
        self.assertNotIn("documented Claude 4.x allowlist", computer.model_families)

        managed = get_implementation("browser", "anthropic-managed-web-research")
        self.assertTrue(
            any("snapshot digest" in item for item in managed.exact_components)
        )
        self.assertTrue(
            any(
                "environment definition" in item
                for item in managed.unavailable_components
            )
        )

        opus_source = get_study("browser", "anthropic-opus5-team-5").sources[0]
        self.assertEqual(
            opus_source.url,
            "https://www-cdn.anthropic.com/"
            "b514064af1408018e64b1ad24e7d5e75850b4ffd/"
            "Claude%20Opus%205%20System%20Card.pdf",
        )

    def test_external_cli_implementations_claim_only_the_public_protocol(self) -> None:
        for implementation in ("prime-agent-0.7.1", "grok-build-1.0.0"):
            candidate = get_implementation("swe", implementation)
            self.assertEqual(candidate.artifact_kind, "runtime_protocol")
            self.assertEqual(candidate.fidelity, "exact_public_protocol")
            self.assertEqual(
                candidate.exactness_scope, "published_runtime_protocol_boundary"
            )
            self.assertTrue(
                any("identity" in item for item in candidate.unavailable_components)
            )

    def test_rlm_local_and_upstream_are_distinct_profiles(self) -> None:
        local = get_study("swe", "rlm-0.1.3-contract")
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

    def test_rao_public_snapshot_and_local_inference_are_not_conflated(self) -> None:
        local = get_study("swe", "platoon-recursive-inference")
        training = get_profile("swe", "rao-policy-training")
        self.assertEqual(local.fidelity, "inference_only_reimplementation")
        self.assertEqual(training.status, "catalog_only")
        self.assertEqual(training.artifact_kind, "training_method")
        self.assertEqual(
            [source.version for source in training.sources],
            ["arXiv:2605.06639v1", "0.1.0"],
        )
        self.assertEqual(
            training.sources[1].revision,
            "d9c5857d3a0a056ebc9b047241a2a0c9515aafbe",
        )
        public_components = " ".join(training.exact_components)
        self.assertIn("Tinker/AReaL training pipelines", public_components)
        self.assertIn("reward code", public_components)
        self.assertIn("no paper-trained checkpoint", training.unavailable_components[1])
        self.assertIn("official IPython runtime", local.unavailable_components[0])

    def test_macu_local_and_upstream_are_distinct_profiles(self) -> None:
        local = get_study("computer-use", "macu-text-dag")
        upstream = get_implementation("computer-use", "macu-upstream-generic-vm")
        benchmark_gap = get_profile("computer-use", "macu-osworld1-benchmark-parity")
        self.assertEqual(local.fidelity, "source_matched_reimplementation")
        self.assertEqual(upstream.fidelity, "upstream_runtime_adapter")
        self.assertEqual(upstream.providers, ("macu-upstream",))
        self.assertEqual(
            upstream.sources[0].revision,
            "5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105",
        )
        self.assertIn("generic", upstream.title.lower())
        self.assertEqual(benchmark_gap.catalog_kind, "gap")
        self.assertIn("domain/UUID", benchmark_gap.exact_components[0])

    def test_canonical_application_configs_resolve_without_harness_drift(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config_count = 0
        canonical_implementation_keys: set[str] = set()
        for application in list_applications():
            for directory_name, selection_key, expected_kind in (
                ("implementations", "implementation", "implementation"),
                ("studies", "study", "study"),
            ):
                config_directory = repository / application / directory_name
                self.assertTrue(config_directory.is_dir())
                for config_path in sorted(config_directory.glob("*.json")):
                    config_count += 1
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertNotIn("harnesses", config, config_path)
                    self.assertNotIn("implementation", config, config_path)
                    self.assertNotIn("study", config, config_path)
                    selection = config.get("application")
                    self.assertIsInstance(selection, dict, config_path)
                    assert isinstance(selection, dict)
                    self.assertEqual(selection.get("name"), application, config_path)
                    self.assertIn(selection_key, selection, config_path)
                    self.assertNotIn(
                        "study"
                        if selection_key == "implementation"
                        else "implementation",
                        selection,
                        config_path,
                    )
                    profile = get_profile(
                        application, str(selection.get(selection_key))
                    )
                    if expected_kind == "implementation":
                        self.assertNotIn(profile.key, canonical_implementation_keys)
                        canonical_implementation_keys.add(profile.key)
                    self.assertEqual(profile.catalog_kind, expected_kind, config_path)
                    self.assertIn(
                        profile.status, {"runnable", "simulation"}, config_path
                    )
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
        self.assertEqual(config_count, 27)
        self.assertEqual(
            canonical_implementation_keys,
            {profile.key for profile in IMPLEMENTATIONS},
            "every exact implementation must have exactly one canonical app config",
        )

    def test_list_implementations_filters_by_application(self) -> None:
        browser_profiles = list_implementations("browser")
        self.assertTrue(browser_profiles)
        self.assertTrue(
            all(profile.application == "browser" for profile in browser_profiles)
        )
        self.assertTrue(
            all(
                profile.catalog_kind == "implementation" for profile in browser_profiles
            )
        )
        self.assertTrue(list_studies("browser"))
        self.assertTrue(list_gaps("browser"))
        self.assertEqual(
            len(list_profiles("browser")),
            len(list_implementations("browser"))
            + len(list_studies("browser"))
            + len(list_gaps("browser")),
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
        self.assertTrue(
            all(item["catalog_kind"] == "implementation" for item in payload)
        )
        self.assertTrue(all(item["exactness_scope"] for item in payload))
        self.assertTrue(all(item["sources"] for item in payload))

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(["list-studies", "--application", "computer-use", "--json"]),
                0,
            )
        studies = json.loads(output.getvalue())
        self.assertTrue(studies)
        self.assertTrue(all(item["catalog_kind"] == "study" for item in studies))

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
                            "study": "single-agent-control",
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
                payload["application"]["study"]["key"],
                "swe/single-agent-control",
            )


if __name__ == "__main__":
    unittest.main()
