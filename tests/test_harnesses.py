import json
import unittest

from scaffoldlab.harnesses import (
    AsyncSubagentsHarness,
    BlockingOrchestratorHarness,
    FixedAgentTeamHarness,
    ExternalContextJSONSearchHarness,
    FlatParallelHarness,
    MACUHarness,
    ParallelBestOfNHarness,
    RecursiveDelegationHarness,
    SingleAgentHarness,
)
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, Task


LIMITS = BudgetLimits(
    max_model_calls=64,
    max_concurrency=8,
    wall_time_seconds=5,
    max_depth=2,
)


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_single(self) -> None:
        backend = ScriptedBackend({"/root": ["answer"]})
        result = await SingleAgentHarness().run(
            Task("single", "question"), backend, LIMITS
        )
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.model_calls, 1)

    async def test_parallel_best_of_n_uses_explicit_judge(self) -> None:
        backend = ScriptedBackend(
            {
                "/candidate/0": ["weak"],
                "/candidate/1": ["best"],
                "/candidate/2": ["other"],
                "/judge": ['{"winner": 1, "reason": "most complete"}'],
            }
        )
        result = await ParallelBestOfNHarness(n=3).run(
            Task("bon", "question"), backend, LIMITS
        )
        self.assertEqual(result.answer, "best")
        self.assertEqual(result.metadata["winner"], 1)
        self.assertEqual(result.model_calls, 4)

    async def test_flat_parallel_has_no_selector(self) -> None:
        backend = ScriptedBackend({"/flat/a": ["A"], "/flat/b": ["B"]})
        result = await FlatParallelHarness().run(
            Task(
                "flat",
                "batch",
                metadata={
                    "parallel_tasks": [
                        {"id": "a", "prompt": "one"},
                        {"id": "b", "prompt": "two"},
                    ]
                },
            ),
            backend,
            LIMITS,
        )
        parsed = json.loads(result.answer)
        self.assertEqual([item["result"] for item in parsed], ["A", "B"])
        self.assertFalse(result.metadata["selector"])

    async def test_blocking_orchestrator_waits_for_fresh_workers(self) -> None:
        backend = ScriptedBackend(
            {
                "/orchestrator:plan": [
                    '{"subtasks":[{"id":"a","instruction":"A"},'
                    '{"id":"b","instruction":"B"}]}'
                ],
                "/worker/a": ["report-a"],
                "/worker/b": ["report-b"],
                "/orchestrator:synthesize": ["combined"],
            }
        )
        result = await BlockingOrchestratorHarness(max_workers=2).run(
            Task("blocking", "question"), backend, LIMITS
        )
        self.assertEqual(result.answer, "combined")
        self.assertEqual(result.metadata["barrier_rounds"], 1)
        worker_requests = [
            request for request in backend.requests if request.role == "worker"
        ]
        self.assertTrue(
            all(request.metadata["fresh_context"] for request in worker_requests)
        )

    async def test_fixed_team_keeps_peers_long_lived(self) -> None:
        backend = ScriptedBackend(
            {
                "/team/lead": [
                    '{"type":"wait"}',
                    '{"type":"submit","content":"team answer"}',
                ],
                "/team/peer-1": ['{"type":"submit","content":"peer one report"}'],
                "/team/peer-2": ['{"type":"submit","content":"peer two report"}'],
            }
        )
        result = await FixedAgentTeamHarness(team_size=3).run(
            Task("team", "question", context="shared evidence"), backend, LIMITS
        )
        self.assertEqual(result.answer, "team answer")
        self.assertTrue(result.metadata["peer_messaging"])
        self.assertIn("shared evidence", backend.requests[0].prompt)

    async def test_async_subagent_returns_to_lead_and_idles(self) -> None:
        backend = ScriptedBackend(
            {
                "/async/lead": [
                    '{"type":"spawn","name":"researcher",'
                    '"instruction":"find evidence"}',
                    '{"type":"wait"}',
                    '{"type":"submit","content":"final from child"}',
                ],
                "/async/researcher": ['{"type":"submit","content":"evidence"}'],
            }
        )
        result = await AsyncSubagentsHarness(
            max_concurrent_subagents=2, max_total_subagents=4
        ).run(Task("async", "question"), backend, LIMITS)
        self.assertEqual(result.answer, "final from child")
        self.assertEqual(result.metadata["subagents_created"], 1)

    async def test_macu_executes_ready_frontier_and_replans(self) -> None:
        backend = ScriptedBackend(
            {
                "/macu/manager:initial_plan": [
                    '{"nodes":[{"id":"a","goal":"A","depends_on":[]},'
                    '{"id":"b","goal":"B","depends_on":[]}]}'
                ],
                "/macu/worker/a": ["A done"],
                "/macu/worker/b": ["B done"],
                "role:replan": [
                    '{"add_nodes":[],"rewrite_nodes":[],"cancel_nodes":[],'
                    '"follow_up":[],"stop":false}',
                    '{"add_nodes":[],"rewrite_nodes":[],"cancel_nodes":[],'
                    '"follow_up":[],"stop":false}',
                ],
                "/macu/manager:synthesize": ["MACU final"],
            }
        )
        result = await MACUHarness(max_workers=2).run(
            Task("macu", "question"), backend, LIMITS
        )
        self.assertEqual(result.answer, "MACU final")
        self.assertEqual(result.metadata["replans"], 2)
        self.assertEqual(result.metadata["peak_parallel_workers"], 2)

    async def test_recursive_delegation_uses_same_policy_tree(self) -> None:
        backend = ScriptedBackend(
            {
                "/recursive/root": [
                    '{"type":"delegate","tasks":["one","two"]}',
                    '{"type":"answer","content":"root synthesis"}',
                ],
                "/recursive/root/turn-0/child-0": [
                    '{"type":"answer","content":"child one"}'
                ],
                "/recursive/root/turn-0/child-1": [
                    '{"type":"answer","content":"child two"}'
                ],
            }
        )
        result = await RecursiveDelegationHarness(max_children=2).run(
            Task("recursive", "question"), backend, LIMITS
        )
        self.assertEqual(result.answer, "root synthesis")
        self.assertEqual(result.metadata["peak_depth"], 1)
        self.assertFalse(result.metadata["rao_training_reproduced"])

    async def test_recursive_children_keep_shared_context_and_ancestry(self) -> None:
        child_id = "/recursive/root/turn-0/child-0"
        grandchild_id = f"{child_id}/turn-0/child-0"
        backend = ScriptedBackend(
            {
                "/recursive/root": [
                    '{"type":"delegate","tasks":["child problem"]}',
                    '{"type":"answer","content":"root answer"}',
                ],
                child_id: [
                    '{"type":"delegate","tasks":["grandchild problem"]}',
                    '{"type":"answer","content":"child answer"}',
                ],
                grandchild_id: ['{"type":"answer","content":"grandchild answer"}'],
            }
        )
        result = await RecursiveDelegationHarness(max_children=1).run(
            Task(
                "recursive-context",
                "root question",
                context="grounded shared evidence",
            ),
            backend,
            LIMITS,
        )
        self.assertEqual(result.answer, "root answer")
        child_prompts = [
            request.prompt
            for request in backend.requests
            if request.agent_id in {child_id, grandchild_id}
        ]
        self.assertTrue(child_prompts)
        self.assertTrue(
            all("grounded shared evidence" in prompt for prompt in child_prompts)
        )
        grandchild_prompt = next(
            request.prompt
            for request in backend.requests
            if request.agent_id == grandchild_id
        )
        self.assertIn("root question", grandchild_prompt)
        self.assertIn("child problem", grandchild_prompt)

    async def test_external_context_ablation_keeps_context_out_of_controller(
        self,
    ) -> None:
        backend = ScriptedBackend(
            {
                "role:external_context_controller": [
                    '{"type":"subcall","query":"what is first?","slices":[[0,5]]}',
                    '{"type":"answer","content":"alpha"}',
                ],
                "/context-search/subcall-1": ["alpha"],
            }
        )
        result = await ExternalContextJSONSearchHarness().run(
            Task("rlm", "question", context="alpha beta gamma"), backend, LIMITS
        )
        self.assertEqual(result.answer, "alpha")
        self.assertEqual(result.metadata["subcalls"], 1)
        controller_requests = [
            request
            for request in backend.requests
            if request.role == "external_context_controller"
        ]
        self.assertTrue(
            all(
                "alpha beta gamma" not in request.prompt
                for request in controller_requests
            )
        )


if __name__ == "__main__":
    unittest.main()
