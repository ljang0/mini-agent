from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from mini_agent.cli import main
from mini_agent.references import get_reference, list_references


@dataclass(frozen=True)
class _ReferenceCase:
    application: str
    name: str
    provider: str
    environment_type: str


# This is intentionally explicit rather than generated from the catalog. It is the
# offline acceptance inventory: removing, renaming, or silently changing the
# provider/environment contract of an exact runtime must change this reviewable
# list and cannot make the test vacuously pass.
REFERENCE_CASES = (
    _ReferenceCase(
        "web", "openai-hosted-multi-agent-functions", "openai-responses", "browser"
    ),
    _ReferenceCase("web", "openai-hosted-web-search", "openai-responses", "none"),
    _ReferenceCase("web", "xai-hosted-web-research-4", "xai-responses", "none"),
    _ReferenceCase("web", "xai-hosted-web-research-16", "xai-responses", "none"),
    _ReferenceCase(
        "web", "browser-use-0.13.7-upstream", "browser-use-upstream", "none"
    ),
    _ReferenceCase("web", "macu-upstream", "macu-upstream", "none"),
    _ReferenceCase(
        "web",
        "anthropic-managed-web-research",
        "anthropic-managed-agents",
        "none",
    ),
    _ReferenceCase("cua", "openai-ga-computer-single", "openai-responses", "computer"),
    _ReferenceCase(
        "cua",
        "anthropic-computer-20251124-single",
        "anthropic-messages",
        "computer",
    ),
    _ReferenceCase("cua", "macu-upstream-generic-vm", "macu-upstream", "none"),
    _ReferenceCase("swe", "openai-codex-source-0.147.0", "codex-source", "none"),
    _ReferenceCase(
        "swe",
        "claude-code-agent-teams-2.1.226",
        "claude-code-agent-teams",
        "none",
    ),
    _ReferenceCase(
        "swe", "anthropic-managed-agents", "anthropic-managed-agents", "none"
    ),
    _ReferenceCase("swe", "prime-agent-0.7.1", "prime-agent", "none"),
    _ReferenceCase("swe", "grok-build-1.0.0", "grok-build", "none"),
    _ReferenceCase("swe", "grok-build-source-1.0.0", "grok-build-source", "none"),
    _ReferenceCase("swe", "kimi-code-0.34.0-upstream", "kimi-code-upstream", "none"),
    _ReferenceCase("swe", "rlm-0.1.3-upstream", "rlm-upstream", "none"),
)

CANONICAL_CONFIGS = {
    (
        "web",
        "openai-hosted-multi-agent-functions",
    ): "browser/implementations/openai-hosted-browser-functions.json",
    (
        "web",
        "openai-hosted-web-search",
    ): "browser/implementations/openai-hosted-web-search.json",
    (
        "web",
        "xai-hosted-web-research-4",
    ): "browser/implementations/xai-hosted-web-research-4.json",
    (
        "web",
        "xai-hosted-web-research-16",
    ): "browser/implementations/xai-hosted-web-research-16.json",
    (
        "web",
        "browser-use-0.13.7-upstream",
    ): "browser/implementations/browser-use-0.13.7-upstream.json",
    ("web", "macu-upstream"): "browser/implementations/macu-upstream.json",
    (
        "web",
        "anthropic-managed-web-research",
    ): "browser/implementations/anthropic-managed-web-research.json",
    (
        "cua",
        "openai-ga-computer-single",
    ): "computer-use/implementations/openai-ga-single.json",
    (
        "cua",
        "anthropic-computer-20251124-single",
    ): "computer-use/implementations/anthropic-20251124-single.json",
    (
        "cua",
        "macu-upstream-generic-vm",
    ): "computer-use/implementations/macu-upstream-generic-vm.json",
    (
        "swe",
        "openai-codex-source-0.147.0",
    ): "swe/implementations/openai-codex-source-0.147.0.json",
    (
        "swe",
        "claude-code-agent-teams-2.1.226",
    ): "swe/implementations/claude-code-agent-teams-2.1.226.json",
    (
        "swe",
        "anthropic-managed-agents",
    ): "swe/implementations/anthropic-managed-agents.json",
    ("swe", "prime-agent-0.7.1"): "swe/implementations/prime-agent-0.7.1.json",
    ("swe", "grok-build-1.0.0"): "swe/implementations/grok-build-1.0.0.json",
    (
        "swe",
        "grok-build-source-1.0.0",
    ): "swe/implementations/grok-build-source-1.0.0.json",
    (
        "swe",
        "kimi-code-0.34.0-upstream",
    ): "swe/implementations/kimi-code-0.34.0-upstream.json",
    ("swe", "rlm-0.1.3-upstream"): "swe/implementations/rlm-0.1.3-upstream.json",
}


def _environment(environment_type: str) -> dict[str, object] | None:
    if environment_type == "none":
        return None
    # The browser and computer factories share this deterministic validation
    # shape. Validation parses it and records provenance but never launches a
    # browser, contacts the start URL, or creates a model provider.
    return {
        "type": environment_type,
        "protocol": "auto",
        "isolation": "per_agent",
        "allowed_hosts": ["example.com"],
        "start_url": "https://example.com",
        "headless": True,
        "viewport_width": 1440,
        "viewport_height": 900,
        "max_tool_output_bytes": 262144,
    }


class AllExactReferenceAcceptanceTests(unittest.TestCase):
    def test_inventory_is_one_to_one_with_all_exact_references(self) -> None:
        expected = {
            (case.application, case.name, case.provider, case.environment_type)
            for case in REFERENCE_CASES
        }
        actual = {
            (
                reference.application,
                reference.name,
                reference.providers[0],
                reference.profile.environment_types[0],
            )
            for reference in list_references()
        }
        self.assertEqual(len(REFERENCE_CASES), 18)
        self.assertEqual(actual, expected)
        self.assertTrue(
            all(len(reference.providers) == 1 for reference in list_references())
        )
        self.assertTrue(
            all(
                len(reference.profile.environment_types) == 1
                for reference in list_references()
            )
        )
        self.assertEqual(
            set(CANONICAL_CONFIGS),
            {(case.application, case.name) for case in REFERENCE_CASES},
        )

    def test_every_checked_in_reference_config_validates_offline(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            tasks = Path(temporary) / "tasks.jsonl"
            tasks.write_text('{"id":"one","prompt":"answer"}\n', encoding="utf-8")
            for case in REFERENCE_CASES:
                with self.subTest(reference=f"{case.application}/{case.name}"):
                    reference = get_reference(case.application, case.name)
                    config = (
                        repository / CANONICAL_CONFIGS[(case.application, case.name)]
                    )
                    with redirect_stdout(io.StringIO()):
                        result = reference.validate(
                            tasks=tasks,
                            config=config,
                            provider=case.provider,
                        )
                    self.assertEqual(result, 0)

    def test_mini_agent_cli_resolves_and_validates_every_reference_offline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.jsonl"
            tasks.write_text('{"id":"one","prompt":"answer"}\n', encoding="utf-8")

            for case in REFERENCE_CASES:
                with self.subTest(reference=f"{case.application}/{case.name}"):
                    reference = get_reference(case.application, case.name)
                    config = root / f"{case.application}-{case.name}.json"
                    raw: dict[str, object] = {
                        "application": {
                            "name": case.application,
                            "implementation": case.name,
                        },
                        # The harnesses are deliberately omitted. Resolution must
                        # supply the exact pinned harness signatures from catalog.
                        "limits": {"max_model_calls": 1},
                    }
                    environment = _environment(case.environment_type)
                    if environment is not None:
                        raw["environment"] = environment
                    config.write_text(json.dumps(raw), encoding="utf-8")

                    output = io.StringIO()
                    with redirect_stdout(output):
                        result = main(
                            (
                                "validate-reference",
                                "--application",
                                case.application,
                                "--implementation",
                                case.name,
                                "--tasks",
                                str(tasks),
                                "--config",
                                str(config),
                                "--provider",
                                case.provider,
                            )
                        )

                    self.assertEqual(result, 0)
                    payload = json.loads(output.getvalue())
                    self.assertIs(payload["valid"], True)
                    self.assertEqual(payload["tasks"], 1)
                    self.assertEqual(
                        payload["application"]["implementation"]["key"],
                        reference.profile.key,
                    )
                    self.assertEqual(
                        payload["harnesses"],
                        [signature.name for signature in reference.profile.harnesses],
                    )
                    self.assertEqual(
                        payload["application"]["implementation"]["harnesses"],
                        json.loads(
                            json.dumps(
                                [
                                    signature.as_dict()
                                    for signature in reference.profile.harnesses
                                ]
                            )
                        ),
                    )
                    actual_environment = payload["environment"]
                    if case.environment_type == "none":
                        self.assertIsNone(actual_environment)
                    else:
                        self.assertEqual(
                            actual_environment["config"]["type"],
                            case.environment_type,
                        )


if __name__ == "__main__":
    unittest.main()
