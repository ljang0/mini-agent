import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scaffoldlab.external import MACUUpstreamBackend, _macu_usage_from_summary
from scaffoldlab.harnesses import MACUUpstreamHarness
from scaffoldlab.providers import ProviderError
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, ModelRequest, ModelResponse, Task, Usage


FAKE_MACU = r"""import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tasks_file")
parser.add_argument("--task-id", required=True)
parser.add_argument("--result-dir", required=True)
parser.add_argument("--osworld-root", required=True)
parser.add_argument("--manager-provider", required=True)
parser.add_argument("--manager-model", required=True)
parser.add_argument("--cua-provider", required=True)
parser.add_argument("--max-parallelism", required=True)
parser.add_argument("--max-replans", required=True)
parser.add_argument("--max-task-timeout", required=True)
args, _ = parser.parse_known_args()

task = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))[0]
assert task["task_id"] == args.task_id
assert task["no_initial_setup"] is True
if task["confirmed_task"] == "mutate checkout":
    Path(__file__).with_name("runtime-mutation.txt").write_text(
        "mutated", encoding="utf-8"
    )
output = Path(args.result_dir) / args.task_id
output.mkdir(parents=True)
(output / "summary.json").write_text(json.dumps({
    "task_id": args.task_id,
    "manager_calls": [{
        "model": args.manager_model,
        "input_tokens": 10,
        "output_tokens": 4,
        "cost_usd": 0.3,
    }],
    "subtask_costs": {
        "worker": {
            "agent_type": "cua",
            "input_tokens": 3,
            "output_tokens": 2,
            "cost_usd": 0.12,
        },
        "final_aggregation": {
            "agent_type": "manager",
            "input_tokens": 10,
            "output_tokens": 4,
            "cost_usd": 0.3,
        },
    },
    "total_cost_usd": 0.42,
}), encoding="utf-8")
(output / "final_results.json").write_text(json.dumps({
    "task_id": args.task_id,
    "final_response": "upstream answer",
    "instruction": task["confirmed_task"],
    "replanning": {"enabled": True, "num_calls": 1},
}), encoding="utf-8")
"""


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class MACUUpstreamTests(unittest.IsolatedAsyncioTestCase):
    def _checkout(self, root: Path) -> tuple[Path, str]:
        checkout = root / "macu"
        checkout.mkdir()
        (checkout / "run_macu.py").write_text(FAKE_MACU, encoding="utf-8")
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "offline@example.invalid", cwd=checkout)
        _git("config", "user.name", "Offline Test", cwd=checkout)
        _git("add", "run_macu.py", cwd=checkout)
        _git("commit", "-q", "-m", "fake pinned MACU", cwd=checkout)
        return checkout, _git("rev-parse", "HEAD", cwd=checkout)

    async def test_runs_pinned_upstream_and_projects_whole_tree_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            osworld = root / "osworld"
            osworld.mkdir()
            backend = MACUUpstreamBackend(
                checkout=checkout,
                result_dir=root / "results",
                osworld_root=osworld,
                manager_provider="anthropic",
                manager_model="manager-model",
                cua_provider="openai",
                python_executable=sys.executable,
                expected_checkout_revision=revision,
                cua_args=("--model", "cua-model"),
            )

            response = await backend.complete(
                ModelRequest(
                    agent_id="/macu/root",
                    role="macu_upstream_session",
                    prompt="Use the computer",
                    metadata={"task_id": "task one"},
                )
            )

            self.assertEqual(response.text, "upstream answer")
            self.assertEqual(response.usage.input_tokens, 13)
            self.assertEqual(response.usage.output_tokens, 6)
            self.assertEqual(response.usage.cost_usd, 0.42)
            self.assertFalse(response.usage.complete)
            self.assertFalse(response.usage.cost_known)
            self.assertEqual(
                backend.provenance()["expected_checkout_revision"], revision
            )
            self.assertTrue(Path(str(response.raw["result_directory"])).is_dir())

    async def test_harness_uses_one_shared_ledger_call_and_trace(self) -> None:
        response = ModelResponse(
            text="done",
            usage=Usage(input_tokens=7, output_tokens=2, cost_usd=0.1),
            raw={
                "task_id": "upstream-id",
                "result_directory": "/isolated/results/upstream-id",
                "summary": {},
                "final_results": {"status": "success"},
            },
        )
        backend = ScriptedBackend({"/macu/root": [response]})
        result = await MACUUpstreamHarness().run(
            Task("local-id", "task"),
            backend,
            BudgetLimits(max_model_calls=1, wall_time_seconds=5),
        )

        self.assertEqual(result.answer, "done")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.usage.input_tokens, 7)
        self.assertTrue(result.metadata["shared_budget_ledger"])
        self.assertEqual(backend.requests[0].metadata["task_id"], "local-id")
        events = [event.event for event in result.trace]
        self.assertIn("model_call_started", events)
        self.assertIn("model_call_completed", events)

    def test_summary_does_not_double_count_manager_subtask_entries(self) -> None:
        usage = _macu_usage_from_summary(
            {
                "manager_calls": [{"input_tokens": 5, "output_tokens": 2}],
                "subtask_costs": {
                    "manager": {
                        "agent_type": "manager",
                        "input_tokens": 5,
                        "output_tokens": 2,
                    },
                    "cua": {
                        "agent_type": "cua",
                        "input_tokens": 3,
                        "output_tokens": 1,
                    },
                },
                "total_cost_usd": 0.25,
            }
        )
        self.assertEqual((usage.input_tokens, usage.output_tokens), (8, 3))

    def test_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, _ = self._checkout(root)
            osworld = root / "osworld"
            osworld.mkdir()
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                MACUUpstreamBackend(
                    checkout=checkout,
                    result_dir=root / "results",
                    osworld_root=osworld,
                    manager_provider="anthropic",
                    manager_model="manager-model",
                    cua_provider="openai",
                    expected_checkout_revision="0" * 40,
                )

    def test_dirty_escape_hatch_never_claims_verified_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            (checkout / "local-change.txt").write_text("dirty", encoding="utf-8")
            osworld = root / "osworld"
            osworld.mkdir()
            backend = MACUUpstreamBackend(
                checkout=checkout,
                result_dir=root / "results",
                osworld_root=osworld,
                manager_provider="anthropic",
                manager_model="manager-model",
                cua_provider="openai",
                expected_checkout_revision=revision,
                allow_dirty_checkout=True,
            )

            provenance = backend.provenance()
            self.assertTrue(provenance["dirty_checkout_allowed"])
            self.assertFalse(provenance["runtime_source_identity_verified"])

    async def test_runtime_checkout_mutation_fails_with_observed_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, revision = self._checkout(root)
            osworld = root / "osworld"
            osworld.mkdir()
            backend = MACUUpstreamBackend(
                checkout=checkout,
                result_dir=root / "results",
                osworld_root=osworld,
                manager_provider="anthropic",
                manager_model="manager-model",
                cua_provider="openai",
                python_executable=sys.executable,
                expected_checkout_revision=revision,
            )

            with self.assertRaisesRegex(
                ProviderError, "became dirty before or during"
            ) as caught:
                await backend.complete(
                    ModelRequest(
                        agent_id="/macu/root",
                        role="macu_upstream_session",
                        prompt="mutate checkout",
                        metadata={"task_id": "mutation"},
                    )
                )
            self.assertEqual(caught.exception.usage.input_tokens, 13)
            self.assertFalse(caught.exception.usage.complete)


if __name__ == "__main__":
    unittest.main()
