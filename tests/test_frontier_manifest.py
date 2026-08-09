from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from scaffoldlab.applications import (
    APPLICATION_NAMES,
    FRONTIER_LABS,
    FRONTIER_LABS_BY_NAME,
    FRONTIER_MANIFEST_AS_OF,
    get_implementation,
)
from scaffoldlab.claude_code_source import (
    CLAUDE_CODE_DARWIN_ARM64_EXECUTABLE_SHA256,
    CLAUDE_CODE_DARWIN_ARM64_NPM_INTEGRITY,
    CLAUDE_CODE_PUBLIC_TAG_REVISION,
    CLAUDE_CODE_WRAPPER_NPM_INTEGRITY,
)
from scaffoldlab.cli import main


class FrontierManifestTests(unittest.TestCase):
    def test_manifest_has_eighteen_unique_labs_and_valid_source_boundaries(
        self,
    ) -> None:
        self.assertEqual(len(FRONTIER_LABS), 18)
        self.assertEqual(len(FRONTIER_LABS_BY_NAME), 18)
        self.assertEqual(
            {record.lab for record in FRONTIER_LABS}, set(FRONTIER_LABS_BY_NAME)
        )
        for record in FRONTIER_LABS:
            with self.subTest(lab=record.lab):
                self.assertTrue(record.runtime_url.startswith("https://"))
                self.assertTrue(record.evidence_url.startswith("https://"))
                self.assertTrue(record.applications)
                self.assertIn(record.flagship_exact, {"no", "partial"})
                self.assertEqual(record.audited_at, FRONTIER_MANIFEST_AS_OF)
                self.assertEqual(set(record.applications), set(APPLICATION_NAMES))
                if record.runtime_kind == "source_runtime":
                    self.assertRegex(record.runtime_revision, r"^[0-9a-f]{40}$")
                    self.assertIsNone(record.distribution_identity)
                elif record.runtime_kind == "public_distribution":
                    self.assertEqual(record.runtime_revision, "")
                    self.assertIsNotNone(record.distribution_identity)
                self.assertEqual(
                    set(record.applications),
                    {item.application for item in record.application_statuses},
                )
                for item in record.application_statuses:
                    for implementation_id in item.implementation_ids:
                        implementation = get_implementation(
                            item.application, implementation_id
                        )
                        self.assertEqual(implementation.status, "runnable")
                        if item.status == "protocol_executed":
                            self.assertEqual(
                                implementation.fidelity, "exact_public_protocol"
                            )
                        else:
                            self.assertEqual(
                                implementation.fidelity, "upstream_runtime_adapter"
                            )
                    if item.status == "source_executed":
                        self.assertEqual(record.runtime_kind, "source_runtime")
                    if item.status == "distribution_executed":
                        self.assertEqual(record.runtime_kind, "public_distribution")

    def test_manifest_does_not_promote_models_or_cataloged_source_to_implementation(
        self,
    ) -> None:
        for lab in ("DeepSeek", "MiniMax", "Cohere"):
            with self.subTest(lab=lab):
                self.assertEqual(
                    FRONTIER_LABS_BY_NAME[lab].scaffoldlab_status, "model_only"
                )
        self.assertEqual(
            FRONTIER_LABS_BY_NAME["Google DeepMind"].scaffoldlab_status,
            "catalog_gap",
        )
        kimi = {
            item.application: item.status
            for item in FRONTIER_LABS_BY_NAME["Moonshot AI / Kimi"].application_statuses
        }
        self.assertEqual(kimi["browser"], "catalog_gap")
        self.assertEqual(kimi["computer-use"], "catalog_gap")
        self.assertEqual(kimi["swe"], "source_executed")
        self.assertEqual(
            FRONTIER_LABS_BY_NAME["Moonshot AI / Kimi"].scaffoldlab_status,
            "mixed",
        )

    def test_mixed_lab_coverage_is_explicit_per_application(self) -> None:
        openai = {
            item.application: item
            for item in FRONTIER_LABS_BY_NAME["OpenAI"].application_statuses
        }
        self.assertEqual(openai["browser"].status, "protocol_executed")
        self.assertEqual(openai["computer-use"].status, "protocol_executed")
        self.assertEqual(openai["swe"].status, "source_executed")
        self.assertIn(
            "openai-codex-source-0.147.0",
            openai["swe"].implementation_ids,
        )
        self.assertEqual(FRONTIER_LABS_BY_NAME["OpenAI"].scaffoldlab_status, "mixed")

        anthropic = {
            item.application: item
            for item in FRONTIER_LABS_BY_NAME["Anthropic"].application_statuses
        }
        self.assertEqual(anthropic["swe"].status, "distribution_executed")
        self.assertEqual(
            anthropic["swe"].implementation_ids,
            ("claude-code-agent-teams-2.1.226",),
        )

        xai = {
            item.application: item
            for item in FRONTIER_LABS_BY_NAME["xAI"].application_statuses
        }
        self.assertEqual(xai["browser"].status, "protocol_executed")
        self.assertEqual(xai["computer-use"].status, "catalog_gap")
        self.assertEqual(xai["swe"].status, "source_executed")

    def test_claude_distribution_identity_is_machine_verifiable(self) -> None:
        anthropic = FRONTIER_LABS_BY_NAME["Anthropic"]
        identity = anthropic.distribution_identity
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.platform, "darwin-arm64")
        self.assertEqual(identity.wrapper_integrity, CLAUDE_CODE_WRAPPER_NPM_INTEGRITY)
        self.assertEqual(
            identity.native_integrity, CLAUDE_CODE_DARWIN_ARM64_NPM_INTEGRITY
        )
        self.assertEqual(
            identity.executable_sha256,
            CLAUDE_CODE_DARWIN_ARM64_EXECUTABLE_SHA256,
        )
        self.assertEqual(identity.public_tag_revision, CLAUDE_CODE_PUBLIC_TAG_REVISION)
        self.assertFalse(identity.public_repository_is_runtime_source)

    def test_cli_exposes_machine_readable_audit(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-frontier-sources", "--json"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(len(payload), 18)
        self.assertTrue(all(item["limitation"] for item in payload))
        self.assertTrue(all(item["flagship_exact"] != "yes" for item in payload))
        self.assertTrue(all(item["application_statuses"] for item in payload))
        self.assertTrue(all(len(item["application_statuses"]) == 3 for item in payload))
        self.assertTrue(all("scaffoldlab_status" not in item for item in payload))
        anthropic = next(item for item in payload if item["lab"] == "Anthropic")
        self.assertIn("distribution_identity", anthropic)
        self.assertNotIn("distribution_identity", payload[1])


if __name__ == "__main__":
    unittest.main()
