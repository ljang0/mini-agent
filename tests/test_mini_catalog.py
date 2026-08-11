from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace

from mini_agent.catalog import (
    APPLICATIONS,
    FRONTIER_SOURCES,
    FRONTIER_SOURCES_BY_LAB,
    GAPS,
    IMPLEMENTATIONS,
    IMPLEMENTATIONS_BY_APPLICATION,
    PROFILES,
    STUDIES,
    get_frontier_source,
    get_gap,
    get_implementation,
    get_profile,
    get_study,
    list_gaps,
    list_implementations,
    list_profiles,
    list_studies,
)
from scaffoldlab.applications import (
    FRONTIER_LABS as LEGACY_FRONTIER_SOURCES,
    PROFILES as LEGACY_PROFILES,
)


class MiniCatalogTests(unittest.TestCase):
    def test_every_profile_is_exposed_one_to_one(self) -> None:
        self.assertEqual(len(PROFILES), 55)
        self.assertEqual(len(IMPLEMENTATIONS), 18)
        self.assertEqual(len(STUDIES), 28)
        self.assertEqual(len(GAPS), 9)
        self.assertEqual([value.legacy for value in PROFILES], list(LEGACY_PROFILES))
        self.assertEqual(
            {value.profile_id for value in PROFILES},
            {value.profile_id for value in LEGACY_PROFILES},
        )
        for value in PROFILES:
            with self.subTest(key=value.key):
                expected = value.legacy.as_dict()
                expected.update(
                    {
                        "application": value.application,
                        "key": value.key,
                        "legacy_application": value.legacy_application,
                        "legacy_key": value.legacy_key,
                        "execution_mode": value.execution_mode,
                    }
                )
                self.assertEqual(value.as_dict(), expected)
                self.assertEqual(value.catalog_kind, value.legacy.catalog_kind)
                self.assertEqual(value.status, value.legacy.status)
                self.assertEqual(value.fidelity, value.legacy.fidelity)
                self.assertEqual(value.exactness_scope, value.legacy.exactness_scope)
                self.assertEqual(value.providers, value.legacy.providers)
                self.assertEqual(value.as_dict()["id"], value.profile_id)
                expected_mode = {
                    "implementation": "reference",
                    "study": "study",
                    "gap": "unavailable",
                }[value.catalog_kind]
                self.assertEqual(value.execution_mode, expected_mode)

    def test_exact_implementation_partition_is_preserved(self) -> None:
        self.assertEqual(len(IMPLEMENTATIONS), 18)
        self.assertEqual(
            [value.legacy for value in IMPLEMENTATIONS],
            [
                value
                for value in LEGACY_PROFILES
                if value.catalog_kind == "implementation"
            ],
        )
        self.assertEqual(
            {
                application: len(values)
                for application, values in IMPLEMENTATIONS_BY_APPLICATION.items()
            },
            {"web": 7, "cua": 3, "swe": 8},
        )
        for value in IMPLEMENTATIONS:
            with self.subTest(key=value.key):
                self.assertEqual(value.catalog_kind, "implementation")
                self.assertEqual(value.status, "runnable")
                self.assertEqual(value.fidelity, value.legacy.fidelity)
                self.assertEqual(value.exactness_scope, value.legacy.exactness_scope)
                self.assertEqual(value.providers, value.legacy.providers)
                self.assertEqual(value.as_dict()["id"], value.profile_id)

    def test_profile_kind_filters_and_lookups_do_not_promote_studies_or_gaps(
        self,
    ) -> None:
        self.assertEqual(len(list_profiles()), 55)
        self.assertEqual(len(list_implementations()), 18)
        self.assertEqual(len(list_studies()), 28)
        self.assertEqual(len(list_gaps()), 9)
        self.assertEqual(len(list_profiles("web")), 19)
        self.assertEqual(len(list_profiles("cua")), 8)
        self.assertEqual(len(list_profiles("swe")), 28)

        study = get_study("swe", "single-agent-control")
        gap = get_gap("web", "meta-muse-spark-1.1-orchestration")
        self.assertIs(get_profile("swe", study.profile_id), study)
        self.assertIs(get_profile("web", gap.profile_id), gap)
        with self.assertRaisesRegex(ValueError, "is a study, not an implementation"):
            get_implementation("swe", study.profile_id)
        with self.assertRaisesRegex(ValueError, "is a gap, not an implementation"):
            get_implementation("web", gap.profile_id)

    def test_application_names_are_normalized_without_losing_legacy_keys(self) -> None:
        self.assertEqual(APPLICATIONS, ("web", "cua", "swe"))
        web = get_implementation("web", "openai-hosted-web-search")
        self.assertEqual(web.key, "web/openai-hosted-web-search")
        self.assertEqual(web.legacy_key, "browser/openai-hosted-web-search")
        self.assertIs(
            web,
            get_implementation("web", "browser/openai-hosted-web-search"),
        )
        cua = get_implementation(
            "cua", "computer-use/anthropic-computer-20251124-single"
        )
        self.assertEqual(cua.application, "cua")
        self.assertEqual(len(list_implementations("swe")), 8)
        with self.assertRaisesRegex(ValueError, "belongs to"):
            get_implementation("cua", "browser/openai-hosted-web-search")

    def test_every_frontier_source_and_status_is_exposed_one_to_one(self) -> None:
        self.assertEqual(len(FRONTIER_SOURCES), 18)
        self.assertEqual(len(FRONTIER_SOURCES_BY_LAB), 18)
        self.assertEqual(
            [value.legacy for value in FRONTIER_SOURCES],
            list(LEGACY_FRONTIER_SOURCES),
        )
        self.assertEqual(
            {value.lab for value in FRONTIER_SOURCES},
            {value.lab for value in LEGACY_FRONTIER_SOURCES},
        )
        self.assertEqual(
            sum(len(value.application_statuses) for value in FRONTIER_SOURCES), 54
        )
        for source in FRONTIER_SOURCES:
            with self.subTest(lab=source.lab):
                expected_source = deepcopy(source.legacy.as_dict())
                expected_source["applications"] = list(source.applications)
                expected_source["application_statuses"] = [
                    status.as_dict() for status in source.application_statuses
                ]
                self.assertEqual(source.as_dict(), expected_source)
                self.assertEqual(set(source.applications), set(APPLICATIONS))
                self.assertEqual(source.flagship_exact, source.legacy.flagship_exact)
                self.assertEqual(source.limitation, source.legacy.limitation)
                for status, legacy in zip(
                    source.application_statuses,
                    source.legacy.application_statuses,
                    strict=True,
                ):
                    self.assertEqual(status.status, legacy.status)
                    self.assertEqual(
                        status.implementation_ids, legacy.implementation_ids
                    )
                    self.assertEqual(status.boundary, legacy.boundary)
                    expected = legacy.as_dict()
                    expected.update(
                        {
                            "application": status.application,
                            "legacy_application": status.legacy_application,
                            "implementation_keys": list(status.implementation_keys),
                        }
                    )
                    self.assertEqual(status.as_dict(), expected)

    def test_frontier_source_cannot_be_constructed_with_conflicting_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "differs from legacy evidence"):
            replace(FRONTIER_SOURCES[0], lab="Not the audited lab")

    def test_frontier_links_resolve_to_exact_implementations(self) -> None:
        for source in FRONTIER_SOURCES:
            for status in source.application_statuses:
                for profile_id in status.implementation_ids:
                    implementation = get_implementation(status.application, profile_id)
                    self.assertIn(
                        implementation.fidelity,
                        {"exact_public_protocol", "upstream_runtime_adapter"},
                    )

        anthropic = get_frontier_source("Anthropic")
        self.assertIsNotNone(anthropic.distribution_identity)
        self.assertIn("distribution_identity", anthropic.as_dict())
        self.assertNotIn(
            "distribution_identity", get_frontier_source("OpenAI").as_dict()
        )


if __name__ == "__main__":
    unittest.main()
