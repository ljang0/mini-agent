from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scaffoldlab.applications import ApplicationSelection, resolve_application_config
from scaffoldlab.cli import (
    _enforce_exact_application_identity,
    _run,
    build_parser,
)


class ExactApplicationIdentityTests(unittest.TestCase):
    def _selection(
        self,
        application: str,
        implementation: str,
        provider: str,
        *,
        environment: str | None = None,
    ) -> ApplicationSelection:
        config: dict[str, object] = {
            "application": {
                "name": application,
                "implementation": implementation,
            }
        }
        if environment is not None:
            config["environment"] = {"type": environment}
        _, selection, _ = resolve_application_config(
            config,
            provider=provider,
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        return selection

    def test_every_provider_runtime_override_is_bound_to_the_catalog_pin(self) -> None:
        cases = (
            (
                "browser",
                "anthropic-managed-web-research",
                "anthropic-managed-agents",
                "managed_beta_version",
                "managed-agents-2026-04-01",
                "managed-agents-2099-01-01",
            ),
            (
                "swe",
                "prime-agent-0.7.1",
                "prime-agent",
                "prime_agent_expected_version",
                "0.7.1",
                "9.9.9",
            ),
            (
                "swe",
                "grok-build-1.0.0",
                "grok-build",
                "grok_expected_version",
                "1.0.0",
                "9.9.9",
            ),
            (
                "swe",
                "grok-build-source-1.0.0",
                "grok-build-source",
                "grok_source_expected_revision",
                "8a14c91d88875a831a38b3a066b1683116bcb31c",
                "0" * 40,
            ),
            (
                "browser",
                "macu-upstream",
                "macu-upstream",
                "macu_expected_checkout_revision",
                "5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105",
                "0" * 40,
            ),
            (
                "swe",
                "rlm-0.1.3-upstream",
                "rlm-upstream",
                "rlm_expected_checkout_revision",
                "72d6940142ddfb84ee6be573dc999a37e633e671",
                "0" * 40,
            ),
            (
                "swe",
                "kimi-code-0.34.0-upstream",
                "kimi-code-upstream",
                "kimi_expected_checkout_revision",
                "f0614c53e59f7e1e257412063b059b9eb82764cf",
                "0" * 40,
            ),
            (
                "browser",
                "browser-use-0.13.7-upstream",
                "browser-use-upstream",
                "browser_use_expected_checkout_revision",
                "f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc",
                "0" * 40,
            ),
            (
                "swe",
                "openai-codex-source-0.147.0",
                "codex-source",
                "codex_source_expected_revision",
                "be6e8eac029b183056b7e4402879f15d2c85f61b",
                "0" * 40,
            ),
            (
                "swe",
                "claude-code-agent-teams-2.1.226",
                "claude-code-agent-teams",
                "claude_code_expected_version",
                "2.1.226",
                "9.9.9",
            ),
        )
        for (
            application,
            implementation,
            provider,
            argument_name,
            recorded_pin,
            wrong_pin,
        ) in cases:
            with self.subTest(implementation=implementation):
                selection = self._selection(application, implementation, provider)
                matching = argparse.Namespace(
                    provider=provider,
                    macu_allow_dirty_checkout=False,
                    **{argument_name: recorded_pin},
                )
                _enforce_exact_application_identity(matching, selection)

                drifting = argparse.Namespace(
                    provider=provider,
                    macu_allow_dirty_checkout=False,
                    **{argument_name: wrong_pin},
                )
                with self.assertRaisesRegex(
                    ValueError, "would change the cataloged runtime boundary"
                ):
                    _enforce_exact_application_identity(drifting, selection)

    def test_exact_macu_rejects_dirty_checkout_for_both_application_profiles(
        self,
    ) -> None:
        revision = "5b1b8f91dfc5dc66a2f06af4b443b3009a9cd105"
        for application, implementation in (
            ("browser", "macu-upstream"),
            ("computer-use", "macu-upstream-generic-vm"),
        ):
            with self.subTest(application=application):
                selection = self._selection(
                    application, implementation, "macu-upstream"
                )
                args = argparse.Namespace(
                    provider="macu-upstream",
                    macu_expected_checkout_revision=revision,
                    macu_allow_dirty_checkout=True,
                )
                with self.assertRaisesRegex(
                    ValueError, "requires a clean MACU checkout"
                ):
                    _enforce_exact_application_identity(args, selection)

    def test_caller_built_prime_study_still_binds_recorded_source_revision(
        self,
    ) -> None:
        revision = "95afd319a78ae017a41241d50b013d656a0685ce"
        _, selection, _ = resolve_application_config(
            {
                "application": {
                    "name": "swe",
                    "study": "prime-agent-source-0.7.1",
                }
            },
            provider="prime-agent-source",
        )
        assert selection is not None
        _enforce_exact_application_identity(
            argparse.Namespace(
                provider="prime-agent-source",
                prime_source_expected_revision=revision,
            ),
            selection,
        )
        with self.assertRaisesRegex(
            ValueError, "would change the cataloged runtime boundary"
        ):
            _enforce_exact_application_identity(
                argparse.Namespace(
                    provider="prime-agent-source",
                    prime_source_expected_revision="0" * 40,
                ),
                selection,
            )

    def test_exact_computer_profiles_reject_undocumented_models(self) -> None:
        cases = (
            (
                "openai-ga-computer-single",
                "openai-responses",
                "gpt-5.6",
                "not-an-openai-computer-model",
            ),
            (
                "anthropic-computer-20251124-single",
                "anthropic-messages",
                "claude-opus-5",
                "claude-haiku-4-5",
            ),
        )
        for implementation, provider, supported, unsupported in cases:
            with self.subTest(implementation=implementation):
                _, selection, _ = resolve_application_config(
                    {
                        "application": {
                            "name": "computer-use",
                            "implementation": implementation,
                        },
                        "environment": {"type": "computer"},
                    },
                    provider=provider,
                )
                assert selection is not None
                _enforce_exact_application_identity(
                    argparse.Namespace(provider=provider, model=supported), selection
                )
                with self.assertRaisesRegex(
                    ValueError, "requires a documented compatible model prefix"
                ):
                    _enforce_exact_application_identity(
                        argparse.Namespace(provider=provider, model=unsupported),
                        selection,
                    )

    def test_exact_provider_protocols_reject_alternate_origins(self) -> None:
        cases = (
            (
                "browser",
                "openai-hosted-multi-agent-functions",
                "openai-responses",
                "https://api.openai.com/v1/",
                "browser",
            ),
            (
                "computer-use",
                "anthropic-computer-20251124-single",
                "anthropic-messages",
                "https://api.anthropic.com/v1",
                "computer",
            ),
            (
                "browser",
                "xai-hosted-web-research-4",
                "xai-responses",
                "https://api.x.ai/v1",
                None,
            ),
        )
        for (
            application,
            implementation,
            provider,
            official_origin,
            environment,
        ) in cases:
            with self.subTest(implementation=implementation):
                selection = self._selection(
                    application,
                    implementation,
                    provider,
                    environment=environment,
                )
                common = {"provider": provider, "base_url": official_origin}
                if implementation == "anthropic-computer-20251124-single":
                    common["model"] = "claude-opus-5"
                _enforce_exact_application_identity(
                    argparse.Namespace(**common), selection
                )
                common["base_url"] = "https://proxy.example.invalid/v1"
                with self.assertRaisesRegex(ValueError, "official provider origin"):
                    _enforce_exact_application_identity(
                        argparse.Namespace(**common), selection
                    )

    def test_unclassified_legacy_config_keeps_runtime_override_escape_hatches(
        self,
    ) -> None:
        args = argparse.Namespace(
            provider="macu-upstream",
            macu_expected_checkout_revision="0" * 40,
            macu_allow_dirty_checkout=True,
        )
        _enforce_exact_application_identity(args, None)


class ExactApplicationIdentityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_rejects_pin_drift_before_backend_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            tasks = root / "tasks.jsonl"
            output = root / "output"
            config.write_text(
                json.dumps(
                    {
                        "application": {
                            "name": "swe",
                            "implementation": "prime-agent-0.7.1",
                        },
                        "limits": {"max_model_calls": 1},
                    }
                ),
                encoding="utf-8",
            )
            tasks.write_text(
                json.dumps({"task_id": "one", "prompt": "Solve this."}) + "\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "run",
                    "--tasks",
                    str(tasks),
                    "--config",
                    str(config),
                    "--provider",
                    "prime-agent",
                    "--prime-agent-expected-version",
                    "9.9.9",
                    "--output",
                    str(output),
                ]
            )
            with self.assertRaisesRegex(
                ValueError, "would change the cataloged runtime boundary"
            ):
                await _run(args)


if __name__ == "__main__":
    unittest.main()
