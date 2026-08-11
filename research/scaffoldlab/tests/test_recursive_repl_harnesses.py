import json
import unittest

from scaffoldlab.cli import HARNESS_TYPES, _build_harnesses
from scaffoldlab.harnesses import (
    PlatoonRecursiveInferenceHarness,
    RLMREPLHarness,
)
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import BudgetLimits, RunFailed, Task


LIMITS = BudgetLimits(
    max_model_calls=64,
    max_concurrency=8,
    wall_time_seconds=5,
    max_depth=2,
    max_tool_calls=32,
)


def execute(code: str) -> str:
    return json.dumps({"type": "execute", "code": code})


def answer(content: str) -> str:
    return json.dumps({"type": "answer", "content": content})


class RLMREPLHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_context_namespace_persists_and_queries_share_ledger(
        self,
    ) -> None:
        private_context = "PRIVATE alpha beta gamma"
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute(
                        "needle = context.split()[2]\n"
                        "result = await llm_query('Echo ' + needle)\n"
                        "result"
                    ),
                    execute("combined = needle + ':' + result\ncombined"),
                    answer("beta:beta evidence"),
                ],
                "/rlm/root/llm-1": ["beta evidence"],
            }
        )

        result = await RLMREPLHarness().run(
            Task("rlm-persistent", "find beta", context=private_context),
            backend,
            LIMITS,
        )

        self.assertEqual(result.answer, "beta:beta evidence")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.metadata["bare_subcalls"], 1)
        self.assertTrue(result.metadata["persistent_python_namespace"])
        controllers = [
            request for request in backend.requests if request.role == "rlm_controller"
        ]
        self.assertTrue(controllers)
        self.assertTrue(
            all(private_context not in request.prompt for request in controllers)
        )
        self.assertEqual(
            [event.event for event in result.trace].count("rlm_subcall_started"),
            1,
        )
        self.assertEqual(
            [event.event for event in result.trace].count("model_call_completed"),
            result.model_calls,
        )

    async def test_batched_queries_start_before_either_completes(self) -> None:
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute(
                        "answers = await llm_query_batched(['one', 'two'])\nanswers"
                    ),
                    answer("done"),
                ],
                "/rlm/root/llm-1": ["first"],
                "/rlm/root/llm-2": ["second"],
            },
            delays={
                "/rlm/root/llm-1": 0.02,
                "/rlm/root/llm-2": 0.02,
            },
        )

        result = await RLMREPLHarness().run(
            Task("rlm-batch", "batch", context="external"),
            backend,
            LIMITS,
        )

        self.assertEqual(result.metadata["subcalls"], 2)
        events = [
            (event.event, event.agent_id)
            for event in result.trace
            if event.agent_id in {"/rlm/root/llm-1", "/rlm/root/llm-2"}
            and event.event in {"model_call_started", "model_call_completed"}
        ]
        self.assertEqual(
            events[:2],
            [
                ("model_call_started", "/rlm/root/llm-1"),
                ("model_call_started", "/rlm/root/llm-2"),
            ],
        )

    async def test_recursive_query_falls_back_to_llm_at_max_depth(self) -> None:
        limits = BudgetLimits(
            max_model_calls=16,
            max_concurrency=4,
            wall_time_seconds=5,
            max_depth=1,
            max_tool_calls=8,
        )
        child_id = "/rlm/root/rlm-1"
        fallback_id = f"{child_id}/rlm-1"
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute("child = await rlm_query('child task')\nchild"),
                    answer("root answer"),
                ],
                child_id: [
                    execute("deep = await rlm_query('deep task')\ndeep"),
                    answer("child answer"),
                ],
                fallback_id: ["deep answer"],
            }
        )

        result = await RLMREPLHarness().run(
            Task("rlm-recursive", "root task", context="external corpus"),
            backend,
            limits,
        )

        self.assertEqual(result.answer, "root answer")
        self.assertEqual(result.model_calls, 5)
        self.assertEqual(result.metadata["recursive_agents_created"], 2)
        self.assertEqual(result.metadata["recursive_subcalls"], 2)
        self.assertEqual(result.metadata["depth_fallbacks"], 1)
        fallback_request = next(
            request for request in backend.requests if request.agent_id == fallback_id
        )
        self.assertTrue(fallback_request.metadata["depth_fallback"])

    async def test_batch_reservation_rejects_subcall_overflow_atomically(self) -> None:
        backend = ScriptedBackend(
            {"/rlm/root": [execute("await llm_query_batched(['one', 'two'])")]}
        )

        with self.assertRaises(RunFailed) as caught:
            await RLMREPLHarness(max_subcalls=1).run(
                Task("rlm-bound", "question", context="external"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(len(backend.requests), 1)

    async def test_restricted_repl_rejects_imports(self) -> None:
        backend = ScriptedBackend({"/rlm/root": [execute("import os")]})

        with self.assertRaises(RunFailed) as caught:
            await RLMREPLHarness().run(
                Task("rlm-restricted", "question", context="external"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(caught.exception.tool_calls, 1)

    async def test_restricted_repl_blocks_coroutine_frame_globals_escape(self) -> None:
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute(
                        "pending = llm_query('must not run')\n"
                        "escaped = pending.cr_frame.f_globals"
                    )
                ]
            }
        )

        with self.assertRaises(RunFailed) as caught:
            await RLMREPLHarness().run(
                Task("rlm-introspection", "question", context="external"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(len(backend.requests), 1)

    async def test_restricted_repl_blocks_format_string_introspection(self) -> None:
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute(
                        "pending = llm_query('must not run')\n"
                        "escaped = '{0.cr_frame.f_globals}'.format(pending)"
                    )
                ]
            }
        )

        with self.assertRaises(RunFailed) as caught:
            await RLMREPLHarness().run(
                Task("rlm-format-introspection", "question", context="external"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(len(backend.requests), 1)

    async def test_iteration_limit_stops_without_an_extra_model_call(self) -> None:
        backend = ScriptedBackend({"/rlm/root": [execute("value = len(context)")]})

        with self.assertRaises(RunFailed) as caught:
            await RLMREPLHarness(max_iterations=1).run(
                Task("rlm-stop", "question", context="external"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(caught.exception.tool_calls, 1)

    async def test_runtime_error_is_budgeted_observation_and_can_be_repaired(
        self,
    ) -> None:
        backend = ScriptedBackend(
            {
                "/rlm/root": [
                    execute("value = missing_name + 1"),
                    execute("value = 6 * 7\nvalue"),
                    answer("42"),
                ]
            }
        )

        result = await RLMREPLHarness().run(
            Task("rlm-repair", "calculate", context="external"),
            backend,
            LIMITS,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.tool_calls, 2)
        second_controller_request = [
            request for request in backend.requests if request.role == "rlm_controller"
        ][1]
        self.assertIn("NameError", second_controller_request.prompt)
        repl_events = [
            event
            for event in result.trace
            if event.event == "tool_call_completed"
            and event.data.get("tool") == "rlm_restricted_python"
        ]
        self.assertTrue(repl_events[0].data["is_error"])
        self.assertFalse(repl_events[1].data["is_error"])


class PlatoonRecursiveInferenceHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_children_use_same_policy_and_shared_trace(self) -> None:
        child_one = "/platoon/root/child-1"
        child_two = "/platoon/root/child-2"
        backend = ScriptedBackend(
            {
                "/platoon/root": [
                    execute(
                        "reports = await asyncio.gather("
                        "launch_subagent('one', max_steps=2), "
                        "launch_subagent('two', max_steps=2))\n"
                        "reports"
                    ),
                    answer("combined"),
                ],
                child_one: [answer("one report")],
                child_two: [answer("two report")],
            },
            delays={child_one: 0.02, child_two: 0.02},
        )

        result = await PlatoonRecursiveInferenceHarness().run(
            Task("platoon-parallel", "root task", context="shared evidence"),
            backend,
            LIMITS,
        )

        self.assertEqual(result.answer, "combined")
        self.assertEqual(result.metadata["agents_created"], 3)
        self.assertEqual(result.metadata["subagents_completed"], 2)
        self.assertTrue(result.metadata["inference_only"])
        self.assertFalse(result.metadata["rao_training_reproduced"])
        events = [
            (event.event, event.agent_id)
            for event in result.trace
            if event.agent_id in {child_one, child_two}
            and event.event in {"model_call_started", "model_call_completed"}
        ]
        self.assertEqual(
            events[:2],
            [
                ("model_call_started", child_one),
                ("model_call_started", child_two),
            ],
        )

    async def test_unawaited_launch_is_a_protocol_error_without_child_call(
        self,
    ) -> None:
        backend = ScriptedBackend(
            {"/platoon/root": [execute("pending = launch_subagent('forgotten')")]}
        )

        with self.assertRaises(RunFailed) as caught:
            await PlatoonRecursiveInferenceHarness().run(
                Task("platoon-unawaited", "root task"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(
            [request.agent_id for request in backend.requests],
            ["/platoon/root"],
        )

    async def test_launch_rejects_depth_overflow(self) -> None:
        limits = BudgetLimits(
            max_model_calls=8,
            max_concurrency=2,
            wall_time_seconds=5,
            max_depth=0,
            max_tool_calls=4,
        )
        backend = ScriptedBackend(
            {"/platoon/root": [execute("await launch_subagent('too deep')")]}
        )

        with self.assertRaises(RunFailed) as caught:
            await PlatoonRecursiveInferenceHarness().run(
                Task("platoon-depth", "root task"),
                backend,
                limits,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)

    async def test_attribute_store_cannot_fake_awaited_launch(self) -> None:
        backend = ScriptedBackend(
            {
                "/platoon/root": [
                    execute(
                        "pending = launch_subagent('must not run')\n"
                        "pending.awaited = True"
                    )
                ]
            }
        )

        with self.assertRaises(RunFailed) as caught:
            await PlatoonRecursiveInferenceHarness().run(
                Task("platoon-attribute-store", "root task"),
                backend,
                LIMITS,
            )

        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertEqual(caught.exception.model_calls, 1)
        self.assertEqual(
            [request.agent_id for request in backend.requests],
            ["/platoon/root"],
        )


class RecursiveHarnessRegistrationTests(unittest.TestCase):
    def test_exports_and_cli_registration(self) -> None:
        self.assertIs(HARNESS_TYPES["rlm_repl"], RLMREPLHarness)
        self.assertIs(
            HARNESS_TYPES["platoon_recursive_inference"],
            PlatoonRecursiveInferenceHarness,
        )
        built = _build_harnesses(
            {
                "harnesses": [
                    "rlm_repl",
                    "platoon_recursive_inference",
                ]
            }
        )
        self.assertEqual(
            [harness.name for harness in built],
            ["rlm_repl", "platoon_recursive_inference"],
        )


if __name__ == "__main__":
    unittest.main()
