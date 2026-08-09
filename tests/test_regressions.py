import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scaffoldlab.cli import _build_backend, build_parser
from scaffoldlab.evaluation import MatrixRunner, TrialRecord, summarize
from scaffoldlab.external import PrimeAgentJSONBackend, _usage_from_message
from scaffoldlab.harnesses import (
    AsyncSubagentsHarness,
    FixedAgentTeamHarness,
    MACUHarness,
    ParallelBestOfNHarness,
    SingleAgentHarness,
)
from scaffoldlab.providers import (
    AnthropicMessagesBackend,
    OpenAIResponsesBackend,
    ProviderError,
    TokenPricing,
    XAIResponsesBackend,
    _post_json,
)
from scaffoldlab.runtime import RunContext, ScriptedBackend
from scaffoldlab.types import (
    BudgetExceeded,
    BudgetLimits,
    ModelRequest,
    ModelResponse,
    RunFailed,
    Task,
    Usage,
)


NOOP_REVISION = json.dumps(
    {
        "add_nodes": [],
        "rewrite_nodes": [],
        "cancel_nodes": [],
        "follow_up": [],
        "stop": False,
    }
)


def _response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(text=text, usage=usage or Usage())


def _latest_macu_node(request: ModelRequest) -> dict[str, object]:
    marker = "LATEST OBSERVATION:\n"
    return json.loads(request.prompt.split(marker, 1)[1])


class _FailFirstCandidateBackend:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.started.append(request.agent_id)
        if request.agent_id == "/candidate/0":
            raise RuntimeError("candidate failed")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _HangingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


class _ReportingHangingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.usage_reporter is None:
            raise AssertionError("usage reporter was not installed")
        request.usage_reporter(
            Usage(
                input_tokens=11,
                output_tokens=3,
                cost_usd=0.4,
                cost_known=False,
                complete=False,
            )
        )
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return _response("answer", usage=Usage(cost_usd=0.25))


class _FollowUpMACUBackend:
    """Orders c after b's first replan, making b a succeeded descendant."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.b_replanned = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if request.role == "initial_plan":
            return _response(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "a", "goal": "A", "depends_on": []},
                            {"id": "c", "goal": "C", "depends_on": []},
                            {"id": "b", "goal": "B", "depends_on": ["a"]},
                        ]
                    }
                )
            )
        if request.agent_id == "/macu/worker/a":
            return _response(f"A{request.metadata['attempt']}")
        if request.agent_id == "/macu/worker/b":
            return _response(f"B{request.metadata['attempt']}")
        if request.agent_id == "/macu/worker/c":
            await self.b_replanned.wait()
            return _response("C1")
        if request.role == "replan":
            latest = _latest_macu_node(request)
            if latest["id"] == "b" and latest["attempts"] == 1:
                self.b_replanned.set()
            if latest["id"] == "c":
                return _response(
                    json.dumps(
                        {
                            "add_nodes": [],
                            "rewrite_nodes": [],
                            "cancel_nodes": [],
                            "follow_up": [{"id": "a", "goal": "improve A"}],
                            "stop": False,
                        }
                    )
                )
            return _response(NOOP_REVISION)
        if request.role == "synthesize":
            return _response("final")
        raise AssertionError(f"unexpected request: {request.agent_id}/{request.role}")


class _StopMACUBackend:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.stop_replanned = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if request.role == "initial_plan":
            return _response(
                '{"nodes":[{"id":"a","goal":"A","depends_on":[]},'
                '{"id":"b","goal":"B","depends_on":[]}]}'
            )
        if request.agent_id == "/macu/worker/a":
            return _response("A")
        if request.agent_id == "/macu/worker/b":
            await self.stop_replanned.wait()
            return _response("B")
        if request.role == "replan":
            self.stop_replanned.set()
            return _response(
                '{"add_nodes":[],"rewrite_nodes":[],"cancel_nodes":[],'
                '"follow_up":[],"stop":true}'
            )
        if request.role == "synthesize":
            return _response("stopped final")
        raise AssertionError(f"unexpected request: {request.agent_id}/{request.role}")


class _StaleMACUBackend:
    """Makes b's first attempt finish only after its ancestor is invalidated."""

    def __init__(self, *, budget_exceeded: bool = False) -> None:
        self.requests: list[ModelRequest] = []
        self.b_started = asyncio.Event()
        self.a_second_started = asyncio.Event()
        self.budget_exceeded = budget_exceeded

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if request.role == "initial_plan":
            return _response(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "a", "goal": "A", "depends_on": []},
                            {"id": "c", "goal": "C", "depends_on": []},
                            {"id": "b", "goal": "B", "depends_on": ["a"]},
                        ]
                    }
                )
            )
        if request.agent_id == "/macu/worker/a":
            attempt = int(request.metadata["attempt"])
            if attempt == 2:
                self.a_second_started.set()
            return _response(f"A{attempt}")
        if request.agent_id == "/macu/worker/c":
            await self.b_started.wait()
            return _response("C1")
        if request.agent_id == "/macu/worker/b":
            attempt = int(request.metadata["attempt"])
            if attempt == 1:
                self.b_started.set()
                await self.a_second_started.wait()
                if self.budget_exceeded:
                    return _response(
                        "stale expensive result",
                        usage=Usage(output_tokens=20, cost_usd=2.0),
                    )
                raise RuntimeError("stale worker failure")
            return _response("B2")
        if request.role == "replan":
            latest = _latest_macu_node(request)
            if latest["id"] == "c":
                return _response(
                    json.dumps(
                        {
                            "add_nodes": [],
                            "rewrite_nodes": [],
                            "cancel_nodes": [],
                            "follow_up": [{"id": "a", "goal": "retry A"}],
                            "stop": False,
                        }
                    )
                )
            return _response(NOOP_REVISION)
        if request.role == "synthesize":
            return _response("final after retry")
        raise AssertionError(f"unexpected request: {request.agent_id}/{request.role}")


class RuntimeRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_failure_cancels_queued_siblings_without_counting_them_started(
        self,
    ) -> None:
        backend = _FailFirstCandidateBackend()
        limits = BudgetLimits(
            max_model_calls=10,
            max_concurrency=1,
            wall_time_seconds=2,
        )

        with self.assertRaises(RunFailed) as caught:
            await ParallelBestOfNHarness(n=3).run(
                Task("siblings", "question"), backend, limits
            )

        failure = caught.exception
        events = [event.event for event in failure.trace]
        self.assertEqual(failure.cause_type, "RuntimeError")
        self.assertEqual(backend.started, ["/candidate/0", "/candidate/1"])
        self.assertEqual(failure.model_calls, len(backend.started))
        self.assertEqual(events.count("model_call_queued"), 3)
        self.assertEqual(events.count("model_call_started"), len(backend.started))
        self.assertEqual(events.count("model_call_cancelled"), 2)
        self.assertFalse(failure.usage.complete)
        self.assertFalse(failure.usage.cost_known)

    async def test_started_call_cancellation_is_traced_and_marks_usage_incomplete(
        self,
    ) -> None:
        backend = _HangingBackend()
        context = RunContext(
            backend,
            BudgetLimits(max_input_tokens=10, wall_time_seconds=2),
        )
        call = asyncio.create_task(
            context.call(ModelRequest(agent_id="/root", role="solver", prompt="x"))
        )
        await asyncio.wait_for(backend.started.wait(), timeout=1)

        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call

        self.assertTrue(backend.cancelled.is_set())
        self.assertEqual(context.ledger.calls, 1)
        self.assertFalse(context.ledger.usage.complete)
        self.assertFalse(context.ledger.usage.cost_known)
        self.assertIn(
            "model_call_cancelled", [event.event for event in context.trace.events]
        )
        with self.assertRaises(BudgetExceeded):
            await context.ledger.reserve_call()

    async def test_zero_and_exact_resource_caps_stop_before_another_call(self) -> None:
        zero_backend = _CountingBackend()
        zero_context = RunContext(
            zero_backend,
            BudgetLimits(max_cost_usd=0, wall_time_seconds=2),
        )
        with self.assertRaises(BudgetExceeded):
            await zero_context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="x")
            )
        self.assertEqual(zero_backend.calls, 0)

        exact_backend = _CountingBackend()
        exact_context = RunContext(
            exact_backend,
            BudgetLimits(max_cost_usd=0.25, wall_time_seconds=2),
        )
        await exact_context.call(
            ModelRequest(agent_id="/root", role="solver", prompt="first")
        )
        with self.assertRaises(BudgetExceeded):
            await exact_context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="second")
            )
        self.assertEqual(exact_backend.calls, 1)

    async def test_cancellation_side_channel_preserves_partial_usage(self) -> None:
        backend = _ReportingHangingBackend()
        context = RunContext(backend, BudgetLimits(wall_time_seconds=2))
        call = asyncio.create_task(
            context.call(ModelRequest(agent_id="/root", role="solver", prompt="x"))
        )
        await asyncio.wait_for(backend.started.wait(), timeout=1)
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call
        self.assertEqual(context.ledger.usage.input_tokens, 11)
        self.assertEqual(context.ledger.usage.output_tokens, 3)
        self.assertEqual(context.ledger.usage.cost_usd, 0.4)
        self.assertFalse(context.ledger.usage.complete)

    async def test_cancellation_while_recording_preserves_returned_usage(self) -> None:
        returned_usage = Usage(
            input_tokens=7,
            output_tokens=2,
            cost_usd=0.5,
        )
        context = RunContext(
            ScriptedBackend({"/root": [_response("done", usage=returned_usage)]}),
            BudgetLimits(wall_time_seconds=2),
        )
        original_record = context.ledger.record
        record_started = asyncio.Event()
        allow_record = asyncio.Event()

        async def stalled_record(usage: Usage) -> None:
            record_started.set()
            await allow_record.wait()
            await original_record(usage)

        context.ledger.record = stalled_record  # type: ignore[method-assign]
        call = asyncio.create_task(
            context.call(ModelRequest(agent_id="/root", role="solver", prompt="x"))
        )
        await asyncio.wait_for(record_started.wait(), timeout=1)
        call.cancel()
        allow_record.set()
        with self.assertRaises(asyncio.CancelledError):
            await call

        self.assertEqual(context.ledger.usage, returned_usage)
        self.assertTrue(context.ledger.usage.complete)
        self.assertTrue(context.ledger.usage.cost_known)

    async def test_incomplete_usage_latches_a_hard_resource_budget(self) -> None:
        context = RunContext(
            ScriptedBackend(
                {
                    "/root": [
                        _response(
                            "answer",
                            usage=Usage(
                                input_tokens=3,
                                output_tokens=1,
                                cost_known=False,
                                complete=False,
                            ),
                        )
                    ]
                }
            ),
            BudgetLimits(max_input_tokens=10, wall_time_seconds=2),
        )
        with self.assertRaises(BudgetExceeded):
            await context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="x")
            )
        with self.assertRaises(BudgetExceeded):
            await context.ledger.reserve_call()


class TeamRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_team_detects_all_wait_deadlock_immediately(self) -> None:
        backend = ScriptedBackend({"*": ['{"type":"wait"}'] * 2})
        with self.assertRaises(RunFailed) as caught:
            await FixedAgentTeamHarness(team_size=2).run(
                Task("fixed-deadlock", "question"),
                backend,
                BudgetLimits(wall_time_seconds=2),
            )
        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertIn("deadlocked", str(caught.exception))
        self.assertLess(caught.exception.wall_time_seconds, 1)

    async def test_fixed_team_consumes_report_queued_before_lead_waits(self) -> None:
        backend = ScriptedBackend(
            {
                "/team/lead": [
                    '{"type":"wait"}',
                    '{"type":"submit","content":"final"}',
                ],
                "/team/peer-1": ['{"type":"submit","content":"report"}'],
            },
            delays={"/team/lead": 0.02},
        )
        result = await FixedAgentTeamHarness(team_size=2).run(
            Task("queued-report", "question"),
            backend,
            BudgetLimits(wall_time_seconds=2),
        )
        self.assertEqual(result.answer, "final")

    async def test_async_team_detects_all_idle_deadlock_immediately(self) -> None:
        backend = ScriptedBackend({"/async/lead": ['{"type":"wait"}']})
        with self.assertRaises(RunFailed) as caught:
            await AsyncSubagentsHarness(
                max_concurrent_subagents=1,
                max_total_subagents=1,
            ).run(
                Task("async-deadlock", "question"),
                backend,
                BudgetLimits(wall_time_seconds=2),
            )
        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertIn("deadlocked", str(caught.exception))
        self.assertLess(caught.exception.wall_time_seconds, 1)

    async def test_async_team_rejects_non_string_subagent_name(self) -> None:
        backend = ScriptedBackend(
            {"/async/lead": ['{"type":"spawn","instruction":"work","name":17}']}
        )
        with self.assertRaises(RunFailed) as caught:
            await AsyncSubagentsHarness(
                max_concurrent_subagents=1,
                max_total_subagents=1,
            ).run(
                Task("bad-name", "question"),
                backend,
                BudgetLimits(wall_time_seconds=2),
            )
        self.assertEqual(caught.exception.cause_type, "ProtocolError")
        self.assertIn("non-empty string", str(caught.exception))

    def test_async_team_rejects_zero_turn_limit(self) -> None:
        with self.assertRaises(ValueError):
            AsyncSubagentsHarness(max_turns_per_agent=0)


class MACURegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_invalidates_and_reruns_succeeded_descendants(self) -> None:
        backend = _FollowUpMACUBackend()
        result = await MACUHarness(max_workers=2, max_replans=8).run(
            Task("follow-up", "question"),
            backend,
            BudgetLimits(max_model_calls=32, max_concurrency=4, wall_time_seconds=2),
        )

        b_requests = [
            request
            for request in backend.requests
            if request.agent_id == "/macu/worker/b"
        ]
        self.assertEqual(result.answer, "final")
        self.assertEqual(len(b_requests), 2)
        self.assertEqual(b_requests[1].metadata["attempt"], 2)
        self.assertIn("A2", b_requests[1].prompt)

    async def test_stop_suppresses_further_replans_while_running_workers_finish(
        self,
    ) -> None:
        backend = _StopMACUBackend()
        result = await MACUHarness(max_workers=2, max_replans=4).run(
            Task("stop", "question"),
            backend,
            BudgetLimits(max_model_calls=16, max_concurrency=4, wall_time_seconds=2),
        )

        replans = [request for request in backend.requests if request.role == "replan"]
        self.assertEqual(result.answer, "stopped final")
        self.assertTrue(result.metadata["stop_requested"])
        self.assertEqual(result.metadata["replans"], 1)
        self.assertEqual(len(replans), 1)

    async def test_invalidated_stale_worker_failure_is_discarded_and_retried(
        self,
    ) -> None:
        backend = _StaleMACUBackend()
        result = await MACUHarness(max_workers=2, max_replans=8).run(
            Task("stale", "question"),
            backend,
            BudgetLimits(max_model_calls=32, max_concurrency=4, wall_time_seconds=2),
        )

        b_requests = [
            request
            for request in backend.requests
            if request.agent_id == "/macu/worker/b"
        ]
        discarded = [
            event
            for event in result.trace
            if event.event == "dag_node_result_discarded"
        ]
        self.assertEqual(result.answer, "final after retry")
        self.assertEqual(result.metadata["stale_results_discarded"], 1)
        self.assertEqual(len(b_requests), 2)
        self.assertEqual(b_requests[1].metadata["attempt"], 2)
        self.assertIn("A2", b_requests[1].prompt)
        self.assertEqual(len(discarded), 1)
        self.assertIn("RuntimeError", discarded[0].data["stale_error"])

    async def test_invalidated_worker_cannot_hide_latched_budget_exhaustion(
        self,
    ) -> None:
        backend = _StaleMACUBackend(budget_exceeded=True)
        with self.assertRaises(RunFailed) as caught:
            await MACUHarness(max_workers=2, max_replans=8).run(
                Task("stale-budget", "question"),
                backend,
                BudgetLimits(
                    max_model_calls=32,
                    max_concurrency=4,
                    max_output_tokens=10,
                    wall_time_seconds=2,
                ),
            )

        b_requests = [
            request
            for request in backend.requests
            if request.agent_id == "/macu/worker/b"
        ]
        self.assertEqual(caught.exception.cause_type, "BudgetExceeded")
        self.assertIn("output-token budget exceeded", str(caught.exception))
        self.assertEqual(caught.exception.usage.output_tokens, 20)
        self.assertEqual(len(b_requests), 1)


class EvaluationRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_path_cannot_escape_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifacts"
            runner = MatrixRunner(
                backend=ScriptedBackend({"/root": ["answer"]}),
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=output_dir,
            )
            records, _ = await runner.run(
                [Task("../../outside/../escape", "question")],
                [SingleAgentHarness()],
            )

            trace_path = Path(records[0].metadata["trace_path"]).resolve()
            self.assertTrue(trace_path.is_relative_to(output_dir.resolve()))
            self.assertTrue(trace_path.exists())

    async def test_run_fingerprint_changes_when_task_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = ScriptedBackend({"/root": ["answer", "answer"]})
            common = {
                "backend": backend,
                "limits": BudgetLimits(wall_time_seconds=2),
            }
            _, first = await MatrixRunner(**common, output_dir=root / "first").run(
                [Task("same-id", "first prompt")], [SingleAgentHarness()]
            )
            _, second = await MatrixRunner(**common, output_dir=root / "second").run(
                [Task("same-id", "different prompt")], [SingleAgentHarness()]
            )

        self.assertNotEqual(first["run_fingerprint"], second["run_fingerprint"])

    async def test_zero_matrix_cost_cap_starts_no_trial(self) -> None:
        backend = _CountingBackend()
        with tempfile.TemporaryDirectory() as directory:
            records, summary = await MatrixRunner(
                backend=backend,
                limits=BudgetLimits(wall_time_seconds=2),
                output_dir=Path(directory),
                matrix_max_cost_usd=0,
            ).run([Task("zero-cap", "question")], [SingleAgentHarness()])

        self.assertEqual(backend.calls, 0)
        self.assertEqual(records, [])
        self.assertEqual(summary["trials"], 0)
        self.assertEqual(summary["planned_trials"], 1)
        self.assertFalse(summary["matrix_completed"])
        self.assertIn("zero", summary["termination_reason"])

    def test_summary_includes_errors_in_resource_means_and_separates_variants(
        self,
    ) -> None:
        records = [
            self._trial("one", "parallel_best_of_n@two", "completed", True, 1.0, 1.0),
            self._trial(
                "two",
                "parallel_best_of_n@two",
                "error",
                False,
                9.0,
                9.0,
                error="RuntimeError: boom",
            ),
            self._trial(
                "three", "parallel_best_of_n@three", "completed", True, 3.0, 3.0
            ),
        ]

        summary = summarize(records, run_fingerprint="fingerprint")
        by_variant = summary["harnesses"]
        two = by_variant["parallel_best_of_n@two"]

        self.assertEqual(
            set(by_variant),
            {
                "parallel_best_of_n@two",
                "parallel_best_of_n@three",
            },
        )
        self.assertEqual(two["attempts"], 2)
        self.assertEqual(two["completed"], 1)
        self.assertEqual(two["errors"], 1)
        self.assertEqual(two["attempt_success_rate"], 0.5)
        self.assertEqual(two["mean_cost_usd"], 5.0)
        self.assertEqual(two["mean_wall_time_seconds"], 5.0)
        self.assertEqual(two["cache_read_input_tokens_lower_bound"], 10)
        self.assertEqual(two["error_types"], {"RuntimeError": 1})

    @staticmethod
    def _trial(
        trial_id: str,
        variant_id: str,
        status: str,
        passed: bool,
        cost: float,
        wall_time: float,
        *,
        error: str | None = None,
    ) -> TrialRecord:
        return TrialRecord(
            trial_id=trial_id,
            observed_at_utc="2026-01-01T00:00:00+00:00",
            task_id="task",
            harness="parallel_best_of_n",
            variant_id=variant_id,
            repeat=0,
            status=status,
            score=1.0 if passed else 0.0,
            passed=passed,
            answer="answer" if status == "completed" else None,
            input_tokens=int(cost),
            output_tokens=int(cost),
            cache_read_input_tokens=int(cost),
            cache_write_input_tokens=0,
            cost_usd=cost,
            cost_known=True,
            usage_complete=True,
            model_calls=1,
            wall_time_seconds=wall_time,
            backend_active_union_seconds=wall_time,
            error=error,
        )


class ValidationRegressionTests(unittest.TestCase):
    def test_budget_limits_reject_invalid_numbers_and_types(self) -> None:
        invalid = [
            {"max_model_calls": 1.5},
            {"max_model_calls": True},
            {"max_concurrency": 0},
            {"max_concurrency": False},
            {"max_input_tokens": -1},
            {"max_output_tokens": 1.5},
            {"max_cost_usd": -0.01},
            {"max_cost_usd": math.nan},
            {"wall_time_seconds": math.inf},
            {"wall_time_seconds": True},
            {"max_depth": -1},
            {"max_depth": 1.5},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BudgetLimits(**kwargs)

    def test_usage_rejects_invalid_numbers_and_types(self) -> None:
        invalid = [
            {"input_tokens": -1},
            {"input_tokens": 1.5},
            {"output_tokens": True},
            {"cache_read_input_tokens": -1},
            {"cache_write_input_tokens": 1.5},
            {"input_tokens": 0, "cache_read_input_tokens": 1},
            {"cost_usd": -0.01},
            {"cost_usd": math.inf},
            {"cost_known": 1},
            {"complete": "yes"},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Usage(**kwargs)

    def test_pricing_and_matrix_cap_reject_non_finite_or_negative_values(self) -> None:
        for value in (-1.0, math.nan, math.inf, True):
            with self.subTest(pricing=value):
                with self.assertRaises(ValueError):
                    TokenPricing(value, 1.0)
            with self.subTest(matrix_cap=value):
                with self.assertRaises(ValueError):
                    MatrixRunner(
                        backend=_CountingBackend(),
                        limits=BudgetLimits(),
                        output_dir=Path("unused"),
                        matrix_max_cost_usd=value,
                    )


class _FakeProcess:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout
        self.returncode = 0
        self.communicated: bytes | None = None

    async def communicate(self, value: bytes | None = None) -> tuple[bytes, bytes]:
        self.communicated = value
        return self.stdout, b""


class ProviderRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_extra_body_cannot_override_core_or_tool_fields(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIResponsesBackend(
                model="test", api_key="key", extra_body={"tools": []}
            )
        with self.assertRaises(ValueError):
            AnthropicMessagesBackend(
                model="test", api_key="key", extra_body={"messages": []}
            )
        with self.assertRaises(ValueError):
            XAIResponsesBackend(
                model="grok-4.20-multi-agent-0309",
                api_key="key",
                extra_body={"tools": []},
            )

    def test_prime_071_usage_includes_cache_tokens_and_nested_cost(self) -> None:
        usage = _usage_from_message(
            {
                "usage": {
                    "input": 10,
                    "output": 4,
                    "cacheRead": 3,
                    "cacheWrite": 2,
                    "cost": {"total": 1.25},
                }
            }
        )
        self.assertEqual(usage.input_tokens, 15)
        self.assertEqual(usage.output_tokens, 4)
        self.assertEqual(usage.cache_read_input_tokens, 3)
        self.assertEqual(usage.cache_write_input_tokens, 2)
        self.assertEqual(usage.cost_usd, 1.25)
        self.assertTrue(usage.cost_known)
        zero_cost = _usage_from_message(
            {"usage": {"input": 1, "output": 1, "cost_usd": 0}}
        )
        self.assertTrue(zero_cost.cost_known)
        self.assertEqual(zero_cost.cost_usd, 0)

    async def test_prime_version_pin_is_enforced(self) -> None:
        process = _FakeProcess(b"prime-agent 0.8.0\n")
        with tempfile.TemporaryDirectory() as directory:
            backend = PrimeAgentJSONBackend(
                cwd=Path(directory),
                executable="prime-agent-test",
                expected_version="0.7.1",
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.verify_version()
        self.assertIn("version mismatch", str(caught.exception))

    def test_prime_rejects_cross_call_session_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "persistent Prime Agent sessions"):
                PrimeAgentJSONBackend(cwd=Path(directory), no_session=False)

    def test_prime_cli_defaults_to_the_audited_release_and_rejects_persistence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = build_parser()
            base_arguments = [
                "run",
                "--tasks",
                str(root / "tasks.jsonl"),
                "--config",
                str(root / "config.json"),
                "--provider",
                "prime-agent",
                "--output",
                str(root / "results"),
                "--prime-agent-cwd",
                str(root / "workspace"),
            ]
            (root / "workspace").mkdir()
            backend = _build_backend(parser.parse_args(base_arguments), {})
            self.assertEqual(backend.expected_version, "0.7.1")
            with self.assertRaisesRegex(ValueError, "requires version 0.7.1"):
                _build_backend(
                    parser.parse_args(
                        [
                            *base_arguments,
                            "--prime-agent-expected-version",
                            "0.8.0",
                        ]
                    ),
                    {},
                )
            with self.assertRaisesRegex(
                ValueError, "--prime-agent-persist-session is unsupported"
            ):
                _build_backend(
                    parser.parse_args(
                        [*base_arguments, "--prime-agent-persist-session"]
                    ),
                    {},
                )

    async def test_prime_rejects_request_system_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = PrimeAgentJSONBackend(
                cwd=Path(directory), executable="prime-agent-test"
            )
            create_process = AsyncMock()
            with patch(
                "scaffoldlab.external.asyncio.create_subprocess_exec",
                new=create_process,
            ):
                with self.assertRaisesRegex(
                    ProviderError, "no request-level system-message field"
                ):
                    await backend.complete(
                        ModelRequest(
                            agent_id="/prime",
                            role="prime_agent",
                            system="system",
                            prompt="prompt",
                        )
                    )
            create_process.assert_not_awaited()

    async def test_prime_enforces_documented_json_v3_stream_contract(self) -> None:
        message = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final answer"}],
                "stopReason": "stop",
                "usage": {"input": 1, "output": 1},
            },
        }
        session = {"type": "session", "version": 3, "id": "test-session"}
        agent_end = {"type": "agent_end", "messages": [message["message"]]}
        invalid_streams = (
            ("start with a version 3 session event", (message, agent_end)),
            (
                "start with a version 3 session event",
                ({**session, "version": 2}, message, agent_end),
            ),
            ("terminate with an agent_end event", (session, message)),
            ("non-object JSON event", (session, [], agent_end)),
        )
        for expected_error, events in invalid_streams:
            with self.subTest(expected_error=expected_error, events=events):
                stdout = (
                    "\n".join(json.dumps(event) for event in events) + "\n"
                ).encode()
                process = _FakeProcess(stdout)
                with tempfile.TemporaryDirectory() as directory:
                    backend = PrimeAgentJSONBackend(
                        cwd=Path(directory), executable="prime-agent-test"
                    )
                    with (
                        patch(
                            "scaffoldlab.external.asyncio.create_subprocess_exec",
                            new=AsyncMock(return_value=process),
                        ),
                        patch(
                            "scaffoldlab.external._terminate_process_tree",
                            new=AsyncMock(),
                        ),
                    ):
                        with self.assertRaisesRegex(ProviderError, expected_error):
                            await backend.complete(
                                ModelRequest(
                                    agent_id="/prime",
                                    role="prime_agent",
                                    prompt="prompt",
                                )
                            )

    async def test_malformed_hosted_output_preserves_billed_usage(self) -> None:
        backend = OpenAIResponsesBackend(model="test", api_key="key")
        response = {
            "status": "incomplete",
            "output": None,
            "usage": {"input_tokens": 9, "output_tokens": 4},
        }
        with patch(
            "scaffoldlab.providers._post_json",
            new=AsyncMock(return_value=response),
        ):
            with self.assertRaises(ProviderError) as caught:
                await backend.complete(
                    ModelRequest(agent_id="/root", role="solver", prompt="question")
                )
        self.assertEqual(caught.exception.usage.input_tokens, 9)
        self.assertEqual(caught.exception.usage.output_tokens, 4)
        self.assertFalse(caught.exception.usage.complete)

    async def test_anthropic_rejects_empty_completed_text(self) -> None:
        backend = AnthropicMessagesBackend(model="test", api_key="key")
        response = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": ""}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
        with patch(
            "scaffoldlab.providers._post_json",
            new=AsyncMock(return_value=response),
        ):
            with self.assertRaises(ProviderError) as caught:
                await backend.complete(
                    ModelRequest(agent_id="/root", role="solver", prompt="question")
                )
        self.assertEqual(caught.exception.usage.input_tokens, 3)

    async def test_provider_failure_raw_is_redacted_when_capture_is_disabled(
        self,
    ) -> None:
        secret = "SECRET-RESPONSE-BODY"

        class FailingBackend:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                raise ProviderError(
                    "provider returned HTTP 400",
                    raw={"response_body": secret},
                )

        context = RunContext(
            FailingBackend(),
            BudgetLimits(wall_time_seconds=2),
            capture_content=False,
        )
        with self.assertRaises(ProviderError) as caught:
            await context.call(
                ModelRequest(agent_id="/root", role="solver", prompt="question")
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(
            secret, json.dumps([event.data for event in context.trace.events])
        )

    async def test_post_workspace_hash_failure_preserves_prime_usage(self) -> None:
        session_event = {"type": "session", "version": 3, "id": "test-session"}
        successful_event = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final answer"}],
                "stopReason": "stop",
                "usage": {
                    "input": 5,
                    "output": 2,
                    "cacheRead": 1,
                    "cacheWrite": 1,
                    "cost": {"total": 0.75},
                },
            },
        }
        agent_end = {"type": "agent_end", "messages": [successful_event["message"]]}
        reported: list[Usage] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class MutatingProcess(_FakeProcess):
                async def communicate(
                    self, value: bytes | None = None
                ) -> tuple[bytes, bytes]:
                    (root / "oversized.bin").write_bytes(b"xx")
                    return await super().communicate(value)

            process = MutatingProcess(
                (
                    "\n".join(
                        json.dumps(event)
                        for event in (session_event, successful_event, agent_end)
                    )
                    + "\n"
                ).encode()
            )
            backend = PrimeAgentJSONBackend(
                cwd=root,
                executable="prime-agent-test",
                timeout_seconds=17,
                workspace_hash_max_entries=10,
                workspace_hash_max_bytes=1,
                workspace_hash_timeout_seconds=2,
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(
                            agent_id="/prime",
                            role="prime_agent",
                            prompt="prompt",
                            usage_reporter=reported.append,
                        )
                    )

        self.assertIn("after the session", str(caught.exception))
        self.assertEqual(caught.exception.usage.input_tokens, 7)
        self.assertEqual(caught.exception.usage.output_tokens, 2)
        self.assertEqual(caught.exception.usage.cost_usd, 0.75)
        self.assertFalse(caught.exception.usage.cost_known)
        self.assertFalse(caught.exception.usage.complete)
        self.assertEqual(reported, [caught.exception.usage])
        provenance = backend.provenance()
        self.assertEqual(provenance["timeout_seconds"], 17)
        self.assertEqual(
            provenance["workspace_hash_limits"],
            {
                "max_entries": 10,
                "max_content_bytes": 1,
                "timeout_seconds": 2.0,
            },
        )

    async def test_prime_reads_prompt_from_stdin_and_requires_stop_reason_stop(
        self,
    ) -> None:
        session_event = {"type": "session", "version": 3, "id": "test-session"}
        successful_event = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final answer"}],
                "stopReason": "stop",
                "usage": {
                    "input": 5,
                    "output": 2,
                    "cacheRead": 1,
                    "cacheWrite": 1,
                    "cost": {"total": 0.75},
                },
            },
        }
        agent_end = {"type": "agent_end", "messages": [successful_event["message"]]}
        process = _FakeProcess(
            (
                "\n".join(
                    json.dumps(event)
                    for event in (session_event, successful_event, agent_end)
                )
                + "\n"
            ).encode()
        )
        captured: dict[str, object] = {}

        async def create_process(*args: str, **kwargs: object) -> _FakeProcess:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return process

        with tempfile.TemporaryDirectory() as directory:
            backend = PrimeAgentJSONBackend(
                cwd=Path(directory), executable="prime-agent-test"
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=create_process,
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                response = await backend.complete(
                    ModelRequest(
                        agent_id="/prime",
                        role="prime_agent",
                        prompt="secret prompt",
                    )
                )

        self.assertEqual(response.text, "final answer")
        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 2)
        self.assertEqual(response.usage.cache_read_input_tokens, 1)
        self.assertEqual(response.usage.cache_write_input_tokens, 1)
        self.assertEqual(response.usage.cost_usd, 0.75)
        self.assertFalse(response.usage.cost_known)
        self.assertFalse(response.usage.complete)
        self.assertNotIn("secret prompt", captured["args"])
        self.assertEqual(
            captured["args"],
            ("prime-agent-test", "--mode", "json", "--no-session"),
        )
        self.assertEqual(process.communicated, b"secret prompt")
        self.assertIn("env", captured["kwargs"])

        failed_event = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "truncated"}],
                "stopReason": "length",
                "usage": {"input": 1, "output": 3, "cost": {"total": 0.2}},
            },
        }
        failed_agent_end = {
            "type": "agent_end",
            "messages": [failed_event["message"]],
        }
        failed_process = _FakeProcess(
            (
                "\n".join(
                    json.dumps(event)
                    for event in (session_event, failed_event, failed_agent_end)
                )
                + "\n"
            ).encode()
        )

        async def create_failed_process(*args: str, **kwargs: object) -> _FakeProcess:
            return failed_process

        with tempfile.TemporaryDirectory() as directory:
            backend = PrimeAgentJSONBackend(
                cwd=Path(directory), executable="prime-agent-test"
            )
            with (
                patch(
                    "scaffoldlab.external.asyncio.create_subprocess_exec",
                    new=create_failed_process,
                ),
                patch(
                    "scaffoldlab.external._terminate_process_tree",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    await backend.complete(
                        ModelRequest(
                            agent_id="/prime", role="prime_agent", prompt="prompt"
                        )
                    )
        self.assertIn("stopReason='length'", str(caught.exception))
        self.assertEqual(caught.exception.usage.output_tokens, 3)

    async def test_hosted_openai_requires_root_final_answer(self) -> None:
        backend = OpenAIResponsesBackend(model="gpt-5.6-sol", api_key="test-key")
        child_only = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "agent": {"agent_name": "/async/child"},
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "child answer"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        request = ModelRequest(
            agent_id="/root",
            role="hosted_manager",
            prompt="question",
            metadata={"openai_multi_agent": True},
        )
        with patch(
            "scaffoldlab.providers._post_json",
            new=AsyncMock(return_value=child_only),
        ):
            with self.assertRaises(ProviderError) as caught:
                await backend.complete(request)
        self.assertIn("no /root final_answer", str(caught.exception))
        self.assertEqual(caught.exception.usage.input_tokens, 2)

        root_final = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "agent": {"agent_name": "/root"},
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "root answer"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        with patch(
            "scaffoldlab.providers._post_json",
            new=AsyncMock(return_value=root_final),
        ):
            response = await backend.complete(request)
        self.assertEqual(response.text, "root answer")

    async def test_async_http_post_is_promptly_cancellable(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        exited = asyncio.Event()

        class SlowClient:
            async def __aenter__(self) -> "SlowClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                exited.set()

            async def post(self, *args: object, **kwargs: object) -> object:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                raise AssertionError("unreachable")

        client = SlowClient()
        with patch("scaffoldlab.providers.httpx.AsyncClient", return_value=client):
            call = asyncio.create_task(
                _post_json(
                    "https://example.invalid",
                    headers={},
                    payload={},
                    timeout_seconds=30,
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            call.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await call

        self.assertTrue(cancelled.is_set())
        self.assertTrue(exited.is_set())


if __name__ == "__main__":
    unittest.main()
