from __future__ import annotations

import asyncio
import json
import unittest

from mini_agent.benchmarks.base import spec_bound_agent, task_agent_builder
from mini_agent.environments.base import BaseEnvironment
from mini_agent.models import ScriptedModel
from mini_agent.profiles import load_profile, prompt_for
from mini_agent.execution import RunContext
from mini_agent.specs import AgentSpecV1, TranslationLoss, TranslationReport
from mini_agent.types import BudgetLimits, ModelResponse, ToolDefinition, ToolExecution


def _spec(**changes: object) -> AgentSpecV1:
    values: dict[str, object] = {
        "environment": "web",
        "model": "downstream/model",
        "profile": "custom",
        "system_prompt": "Use evidence.",
        "max_steps": 12,
        "budget": BudgetLimits(wall_time_seconds=90),
        "tool_capabilities": ("browser",),
        "communication_capabilities": (),
        "fidelity": "minimal_baseline",
    }
    values.update(changes)
    return AgentSpecV1(**values)  # type: ignore[arg-type]


class AgentSpecTests(unittest.TestCase):
    def test_serialization_round_trip_and_fingerprint_are_canonical(self) -> None:
        first = _spec(
            tool_capabilities=("browser", "agent"),
            communication_capabilities=("wait", "send", "spawn"),
        )
        second = _spec(
            budget=BudgetLimits(wall_time_seconds=90.0),
            tool_capabilities=("agent", "browser"),
            communication_capabilities=("spawn", "wait", "send"),
        )

        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            AgentSpecV1.from_dict(json.loads(first.canonical_json())), first
        )
        self.assertNotEqual(
            first.fingerprint, _spec(system_prompt="Different.").fingerprint
        )

    def test_profile_document_round_trips_and_verifies_fingerprint(self) -> None:
        spec = _spec()
        document = {**spec.as_dict(), "fingerprint": spec.fingerprint}
        self.assertEqual(AgentSpecV1.from_json(json.dumps(document)), spec)
        document["max_steps"] = 13
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            AgentSpecV1.from_json(json.dumps(document))

    def test_spec_rejects_ambiguous_or_inconsistent_capabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _spec(tool_capabilities=("browser", "browser"))
        with self.assertRaisesRegex(ValueError, "require the agent"):
            _spec(communication_capabilities=("send",))
        with self.assertRaisesRegex(ValueError, "invalid"):
            _spec(tool_capabilities=("Browser UI",))
        value = _spec().as_dict()
        value["future_field"] = True
        with self.assertRaisesRegex(ValueError, "extra"):
            AgentSpecV1.from_dict(value)

    def test_identity_covers_every_portable_runtime_dimension(self) -> None:
        baseline = _spec()
        variants = (
            _spec(environment="computer"),
            _spec(model="another/model"),
            _spec(profile="another-profile"),
            _spec(budget=BudgetLimits(max_model_calls=3)),
            _spec(tool_capabilities=("browser", "search")),
            _spec(
                tool_capabilities=("agent", "browser"),
                communication_capabilities=("send",),
            ),
        )

        self.assertTrue(
            all(item.fingerprint != baseline.fingerprint for item in variants)
        )
        identity = baseline.identity_dict()
        self.assertNotIn("system_prompt", identity)
        self.assertNotIn("Use evidence.", json.dumps(identity))
        self.assertEqual(len(identity["system_prompt_sha256"]), 64)

    def test_profile_resolves_single_and_multi_agent_capabilities(self) -> None:
        profile = load_profile("browser", model="openai/test-model")
        single = profile.to_agent_spec()
        multi = profile.to_agent_spec(multi_agent=True)

        self.assertEqual(single.environment, "web")
        self.assertEqual(single.tool_capabilities, ("browser",))
        self.assertEqual(single.communication_capabilities, ())
        self.assertEqual(multi.tool_capabilities, ("agent", "browser"))
        self.assertEqual(
            multi.communication_capabilities,
            ("adopt", "inbox", "send", "spawn", "stop", "wait"),
        )
        self.assertEqual(multi.system_prompt, prompt_for("web", multi_agent=True))
        self.assertIn("available browser actions", single.system_prompt)
        self.assertNotIn("search and open", single.system_prompt)
        report = profile.translation_report(multi_agent=True)
        self.assertFalse(report.exact)
        self.assertEqual(
            [loss.field for loss in report.losses],
            ["tool_kind", "tool_result_images", "tool_result_is_error"],
        )
        anthropic_report = load_profile(
            "web", model="anthropic/test-model"
        ).translation_report()
        self.assertEqual(
            [loss.field for loss in anthropic_report.losses],
            ["tool_kind", "tool_result_image_history"],
        )

    def test_profile_boolean_contract_is_strict(self) -> None:
        profile = load_profile("swe")
        with self.assertRaisesRegex(ValueError, "boolean"):
            profile.to_agent_spec(multi_agent=1)  # type: ignore[arg-type]

    def test_spec_binds_the_declared_runtime_contract(self) -> None:
        class BrowserFixture(BaseEnvironment):
            def tools(self) -> tuple[ToolDefinition, ...]:
                return (ToolDefinition("browser"),)

            async def execute(self, action: object) -> ToolExecution:
                del action
                return ToolExecution("unused")

        spec = _spec()
        agent = spec.bind(
            model=ScriptedModel([ModelResponse("done")]),
            environment=BrowserFixture(),
            model_id=spec.model,
            environment_id=spec.environment,
        )
        self.assertEqual(agent.system_prompt, spec.system_prompt)
        self.assertEqual(agent.max_steps, spec.max_steps)
        self.assertEqual(agent.context.ledger.limits, spec.budget)
        self.assertEqual(asyncio.run(agent.run("research")).answer, "done")

        with self.assertRaisesRegex(ValueError, "tool capabilities"):
            _spec(tool_capabilities=("search",)).bind(
                model=ScriptedModel([ModelResponse("done")]),
                environment=BrowserFixture(),
                model_id=spec.model,
                environment_id=spec.environment,
            )
        with self.assertRaisesRegex(ValueError, "context budget"):
            spec.bind(
                model=ScriptedModel([ModelResponse("done")]),
                environment=BrowserFixture(),
                model_id=spec.model,
                environment_id=spec.environment,
                context=RunContext(BudgetLimits(max_model_calls=1)),
            )
        with self.assertRaisesRegex(ValueError, "model identifier"):
            spec.bind(
                model=ScriptedModel([ModelResponse("done")]),
                environment=BrowserFixture(),
                model_id="another/model",
                environment_id=spec.environment,
            )

    def test_spec_checks_the_agent_action_surface(self) -> None:
        class TeamFixture(BaseEnvironment):
            def tools(self) -> tuple[ToolDefinition, ...]:
                return (
                    ToolDefinition("browser"),
                    ToolDefinition(
                        "agent",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["send"]}
                            },
                        },
                    ),
                )

            async def execute(self, action: object) -> ToolExecution:
                del action
                return ToolExecution("unused")

        _spec(
            tool_capabilities=("agent", "browser"),
            communication_capabilities=("send",),
        ).bind(
            model=ScriptedModel([ModelResponse("done")]),
            environment=TeamFixture(),
            model_id="downstream/model",
            environment_id="web",
        )
        with self.assertRaisesRegex(ValueError, "communication capabilities"):
            _spec(
                tool_capabilities=("agent", "browser"),
                communication_capabilities=("wait",),
            ).bind(
                model=ScriptedModel([ModelResponse("done")]),
                environment=TeamFixture(),
                model_id="downstream/model",
                environment_id="web",
            )


class TranslationReportTests(unittest.TestCase):
    def test_exact_is_structural_and_explicitly_scoped(self) -> None:
        report = TranslationReport(source_format="source/v1", spec=_spec())

        self.assertTrue(report.exact)
        self.assertEqual(report.status, "exact")
        self.assertEqual(report.as_dict()["claim_scope"], "declared_fields_only")
        self.assertIs(report.require_exact(), report.spec)
        self.assertEqual(len(report.fingerprint), 64)

    def test_losses_are_deterministic_and_cannot_be_required_as_exact(self) -> None:
        report = TranslationReport(
            source_format="provider-harness/v2",
            spec=_spec(),
            losses=(
                TranslationLoss(
                    field="message_timing",
                    kind="unsupported",
                    reason="The neutral spec does not encode provider timing.",
                ),
                TranslationLoss(
                    field="tool_schema",
                    kind="approximated",
                    reason="Provider-only annotations were removed.",
                ),
            ),
        )

        self.assertFalse(report.exact)
        self.assertEqual(report.status, "lossy")
        self.assertEqual(
            [loss["field"] for loss in report.as_dict()["losses"]],
            ["message_timing", "tool_schema"],
        )
        with self.assertRaisesRegex(ValueError, "message_timing, tool_schema"):
            report.require_exact()

    def test_duplicate_and_unknown_losses_are_rejected(self) -> None:
        loss = TranslationLoss(field="policy", reason="Not available.")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            TranslationReport(
                source_format="source/v1", spec=_spec(), losses=(loss, loss)
            )
        with self.assertRaisesRegex(ValueError, "unsupported translation loss kind"):
            TranslationLoss(field="policy", reason="No.", kind="ignored")


class SpecBoundAgentTests(unittest.IsolatedAsyncioTestCase):
    def _identity_spec(self, **changes: object) -> AgentSpecV1:
        values: dict[str, object] = {
            "tool_capabilities": ("identity",),
            "system_prompt": "Use evidence.",
        }
        values.update(changes)
        return _spec(**values)

    async def test_task_agent_builder_binds_against_the_spec(self) -> None:
        from support import IsolatedEnvironment

        spec = self._identity_spec()
        agent_for = task_agent_builder(
            model_factory=lambda agent_id: ScriptedModel([ModelResponse("done")]),
            system_prompt=spec.system_prompt,
            max_steps=spec.max_steps,
            agent_spec=spec,
        )
        context = RunContext(limits=spec.budget)
        agent = await agent_for("/root", IsolatedEnvironment("/root"), context)
        self.assertIs(agent.context, context)
        result = await agent.run("task")
        self.assertEqual(result.answer, "done")

    async def test_bound_agent_rejects_prompt_and_step_drift(self) -> None:
        from support import IsolatedEnvironment

        spec = self._identity_spec()
        environment = IsolatedEnvironment("/root")
        context = RunContext(limits=spec.budget)
        model = ScriptedModel([])
        with self.assertRaisesRegex(ValueError, "system_prompt does not match"):
            spec_bound_agent(
                spec,
                model=model,
                environment=environment,
                context=context,
                agent_id="/root",
                system_prompt="different prompt",
                max_steps=spec.max_steps,
            )
        with self.assertRaisesRegex(ValueError, "max_steps does not match"):
            spec_bound_agent(
                spec,
                model=model,
                environment=environment,
                context=context,
                agent_id="/root",
                system_prompt=spec.system_prompt,
                max_steps=spec.max_steps + 1,
            )

    async def test_bound_agent_rejects_capability_and_budget_drift(self) -> None:
        from support import EchoEnvironment, IsolatedEnvironment

        spec = self._identity_spec()
        with self.assertRaisesRegex(ValueError, "tool capabilities"):
            spec_bound_agent(
                spec,
                model=ScriptedModel([]),
                environment=EchoEnvironment(),
                context=RunContext(limits=spec.budget),
                agent_id="/root",
                system_prompt=spec.system_prompt,
                max_steps=spec.max_steps,
            )
        with self.assertRaisesRegex(ValueError, "budget does not match"):
            spec_bound_agent(
                spec,
                model=ScriptedModel([]),
                environment=IsolatedEnvironment("/root"),
                context=RunContext(limits=BudgetLimits(wall_time_seconds=5)),
                agent_id="/root",
                system_prompt=spec.system_prompt,
                max_steps=spec.max_steps,
            )


class HarnessRoleSpecTests(unittest.TestCase):
    """Harness roles must not disturb the fingerprints already recorded.

    Every evaluation manifest carries an agent-spec fingerprint, and operators
    compare them across runs. The two pre-harness shapes -- single-agent and
    the free-form mesh behind --multi-agent -- have to keep producing exactly
    the fingerprints they produced before harnesses existed, or the recorded
    evidence silently stops matching the code that made it.
    """

    def setUp(self) -> None:
        from mini_agent.profiles import load_profile

        self.profile = load_profile("swe")

    # Golden values. The equality tests below prove the two construction
    # paths agree with each other; they cannot notice both paths moving
    # together, which is exactly what happens when a role's prompt is
    # edited -- profiles derives the legacy constants from the recursive
    # role. Pinning the values is what makes prompt drift visible.
    FINGERPRINTS = {
        "single/solo":
            "281ed957571d0f319af4456be2dbc1a705a7ac23ae2aa59e27e95205843480b1",
        "recursive/solver":
            "11b9e4a7eca172a24f1e60b42338e213b0a38dee277127aee1757b06ec66b8db",
        "fixed-team/lead":
            "deafef15f19b480cd074ad2e75154a2b9df368952fe7ab275809803cdd3e1e9c",
        "fixed-team/peer":
            "86cadf27ee18d55cce82734d0d293e9d1ea19d8e3f7bac339cc2e45db1ebefc8",
        "orchestrator/orchestrator":
            "762f375a2f0ae53a29c43ca582b18159789f9cc785d079a21bcc5200d11a9e54",
        "orchestrator/subagent":
            "f1677c42f901f65dc7592e386499ee77b0fe763ae08b109a6ea13a4af1b88599",
        "async-subagents/lead":
            "e3881c41f9a86a0a527638001a672013f893f784d6d5b875c9fa824276a88576",
        "async-subagents/subagent":
            "a5299fc85c463682c11cdb9916de1eafdc16ffd8994c553a4993b8e6281825ae",
    }

    def test_every_role_fingerprint_is_byte_stable(self) -> None:
        """Editing a role's prompt or actions must be a deliberate act."""

        from mini_agent.harnesses import HARNESSES

        observed = {
            f"{name}/{role_name}": self.profile.to_agent_spec(role=role).fingerprint
            for name, harness in HARNESSES.items()
            for role_name, role in harness.roles.items()
        }
        self.assertEqual(observed, self.FINGERPRINTS)

    def test_recursive_role_reproduces_the_multi_agent_fingerprint(self) -> None:
        from mini_agent.harnesses import load_harness

        role = load_harness("recursive").roles["solver"]
        self.assertEqual(
            self.profile.to_agent_spec(role=role).fingerprint,
            self.profile.to_agent_spec(multi_agent=True).fingerprint,
        )

    def test_single_role_reproduces_the_single_agent_fingerprint(self) -> None:
        from mini_agent.harnesses import load_harness

        role = load_harness("single").roles["solo"]
        self.assertEqual(
            self.profile.to_agent_spec(role=role).fingerprint,
            self.profile.to_agent_spec().fingerprint,
        )

    def test_a_restricted_role_records_fewer_capabilities(self) -> None:
        from mini_agent.harnesses import load_harness

        harness = load_harness("orchestrator")
        orchestrator = self.profile.to_agent_spec(role=harness.roles["orchestrator"])
        subagent = self.profile.to_agent_spec(role=harness.roles["subagent"])
        # An orchestrator holds no domain tools, so its spec must not claim one.
        self.assertEqual(orchestrator.tool_capabilities, ("agent",))
        self.assertEqual(orchestrator.communication_capabilities, ("delegate",))
        self.assertEqual(subagent.tool_capabilities, ("bash",))
        self.assertEqual(subagent.communication_capabilities, ())
        self.assertNotEqual(orchestrator.fingerprint, subagent.fingerprint)


if __name__ == "__main__":
    unittest.main()
