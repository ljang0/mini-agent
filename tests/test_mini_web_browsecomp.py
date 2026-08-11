from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_agent.agent import MiniAgent
from mini_agent.environments.web import (
    BROWSECOMP_PLUS_REVISION,
    JsonlSearchBackend,
    WebAccounting,
    WebEnvironment,
    directory_identity,
)
from mini_agent.evals.browsecomp_plus import (
    BrowseCompArtifactStore,
    BrowseCompBatchRunner,
    BrowseCompRunRecord,
    BrowseCompTask,
    BrowseCompTaskOutcome,
    format_browsecomp_task,
    load_browsecomp_tasks,
    official_evaluator_argv,
    preflight_official_directory,
    run_mini_agent_task,
    run_multi_agent_task,
    verify_browsecomp_checkout,
)
from mini_agent.models import ScriptedModel
from mini_agent.profiles import load_profile
from mini_agent.providers import ProviderError
from mini_agent.types import (
    BudgetLimits,
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from mini_agent.web_models import build_web_model, parse_web_response


FIXTURES = Path(__file__).parent / "fixtures" / "browsecomp_plus"


class _WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        self.add_special_tokens = add_special_tokens
        return text.split()

    def decode(
        self, tokens: list[str], *, skip_special_tokens: bool
    ) -> str:
        self.skip_special_tokens = skip_special_tokens
        return " ".join(tokens)


class _StaticBackend:
    def search(self, query: str, k: int = 5) -> list[dict[str, object]]:
        return [
            {
                "docid": str(index),
                "score": 10.0 - index,
                "text": f"word{index} one two three four",
            }
            for index in range(1, 7)
        ][:k]

    def get_document(self, docid: str) -> dict[str, str] | None:
        return {"docid": docid, "text": f"full document {docid}"}

    def provenance(self) -> dict[str, str]:
        return {"backend": "static_fixture"}


class _SearchThenAnswerModel:
    def __init__(self) -> None:
        self.requests: list[tuple[object, ...]] = []

    async def query(
        self, messages: tuple[object, ...], tools: tuple[object, ...]
    ) -> ModelResponse:
        self.requests.append(messages)
        if len(self.requests) == 1:
            return ModelResponse(
                text="",
                tool_calls=(ToolCall("search", "search", {"query": "alpha"}),),
                usage=Usage(input_tokens=8, output_tokens=2),
            )
        return ModelResponse(
            text="Explanation: fixture\nExact Answer: alpha\nConfidence: 1.0",
            usage=Usage(input_tokens=12, output_tokens=6),
        )


class _ModelBackend:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[object] = []

    async def complete(self, request: object) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("no model response left")
        return self.responses.pop(0)

    def provenance(self) -> dict[str, object]:
        return {"provider": "fixture", "deterministic": True}


class _BlockingModel:
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    async def query(self, messages: object, tools: object) -> ModelResponse:
        del messages, tools
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class BrowseCompTaskTests(unittest.TestCase):
    def test_jsonl_and_tsv_load_identical_inference_safe_tasks(self) -> None:
        jsonl = load_browsecomp_tasks(FIXTURES / "tasks.jsonl")
        tsv = load_browsecomp_tasks(FIXTURES / "queries.tsv")
        self.assertEqual(jsonl.tasks, tsv.tasks)
        self.assertEqual([task.query_id for task in jsonl.tasks], ["q1", "q2"])
        self.assertFalse(hasattr(jsonl.tasks[0], "answer"))
        self.assertEqual(len(jsonl.source_sha256), 64)
        self.assertEqual(jsonl.manifest()["task_count"], 2)

    def test_loader_fails_closed_on_duplicates_and_malformed_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                '{"query_id":"same","query":"one","answer":"secret"}\n'
                '{"query_id":"same","query":"two","answer":"secret"}\n',
                encoding="utf-8",
            )
            malformed = root / "malformed.tsv"
            malformed.write_text("id\tquery\textra\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_browsecomp_tasks(duplicate)
            with self.assertRaisesRegex(ValueError, "exactly two"):
                load_browsecomp_tasks(malformed)

    def test_published_search_only_task_template_is_applied_as_user_text(self) -> None:
        prompt = format_browsecomp_task("Where is alpha?")
        self.assertIn("Question: Where is alpha?", prompt)
        self.assertIn("using the search tool provided", prompt)
        self.assertIn("Explanation:", prompt)
        self.assertIn("Exact Answer:", prompt)
        self.assertIn("Confidence:", prompt)
        self.assertNotIn("{Question}", prompt)


class BrowseCompEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_top_five_token_truncation_and_direct_accounting(self) -> None:
        tokenizer = _WordTokenizer()
        environment = WebEnvironment(
            _StaticBackend(),
            top_k=5,
            snippet_chars=None,
            snippet_tokens=3,
            tokenizer=tokenizer,
            include_get_document=True,
        )
        first = await environment.execute(
            ToolCall("search-1", "search", {"query": "evidence"})
        )
        second = await environment.execute(
            ToolCall("search-2", "search", {"query": "evidence again"})
        )
        document = await environment.execute(
            ToolCall("document", "get_document", {"docid": "6"})
        )

        results = json.loads(first.output)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["snippet"], "word1 one two")
        self.assertEqual(first.metadata["retrieved_docids"], ["1", "2", "3", "4", "5"])
        self.assertEqual(len(first.metadata["query_sha256"]), 64)
        self.assertEqual(first.output, second.output)
        self.assertEqual(json.loads(document.output)["docid"], "6")
        self.assertEqual(
            environment.accounting().as_dict(),
            {
                "tool_call_counts": {"get_document": 1, "search": 2},
                "retrieved_docids": ["1", "2", "3", "4", "5"],
            },
        )
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertTrue(tokenizer.skip_special_tokens)

    async def test_fixture_character_policy_is_explicitly_a_baseline(self) -> None:
        environment = WebEnvironment(
            JsonlSearchBackend(FIXTURES / "corpus.jsonl"), snippet_chars=5
        )
        result = await environment.execute(
            ToolCall("search", "search", {"query": "alpha"})
        )
        self.assertEqual(json.loads(result.output)[0]["snippet"], "alpha")
        self.assertEqual(
            environment.provenance()["snippet_policy"],
            {"unit": "characters", "limit": 5},
        )

    def test_token_policy_requires_dependency_injection_and_rejects_inert_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "injected tokenizer"):
            WebEnvironment(
                _StaticBackend(), snippet_chars=None, snippet_tokens=512
            )
        with self.assertRaisesRegex(ValueError, "unsupported web observation"):
            WebEnvironment.from_policy(
                _StaticBackend(),
                benchmark={"name": "browsecomp_plus", "top_k": 5},
                observation={"unknown": True},
                tools=("search",),
                tokenizer=_WordTokenizer(),
            )
        configured = WebEnvironment.from_policy(
            _StaticBackend(),
            benchmark={
                "name": "browsecomp_plus",
                "retrieval": "bm25",
                "top_k": 5,
            },
            observation={"snippet_tokens": 512},
            tools=("search",),
            tokenizer=_WordTokenizer(),
        )
        self.assertEqual(configured.top_k, 5)
        self.assertEqual(configured.snippet_tokens, 512)


class BrowseCompArtifactTests(unittest.TestCase):
    def _record(
        self, query_id: str, *, status: str = "completed"
    ) -> BrowseCompRunRecord:
        return BrowseCompRunRecord.from_answer(
            BrowseCompTask(query_id, "question"),
            "answer" if status == "completed" else "",
            WebAccounting({"search": 2}, ("2", "1", "2")),
            status=status,
            metadata={"model": "fixture"},
        )

    def test_atomic_official_schema_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = BrowseCompArtifactStore(Path(temporary) / "run")
            first = store.write(self._record("q1"))
            self.assertEqual(first.parent.name, "official")
            self.assertEqual(
                sorted(path.suffix for path in first.parent.iterdir()), [".json"]
            )
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "metadata",
                    "query_id",
                    "result",
                    "retrieved_docids",
                    "status",
                    "tool_call_counts",
                },
            )
            self.assertEqual(payload["retrieved_docids"], ["1", "2"])
            with self.assertRaises(FileExistsError):
                store.write(self._record("q1"))

            tasks = (BrowseCompTask("q1", "one"), BrowseCompTask("q2", "two"))
            self.assertEqual([task.query_id for task in store.pending(tasks)], ["q2"])
            store.write(self._record("q2", status="budget_exhausted"))
            self.assertEqual(store.pending(tasks), ())
            self.assertEqual(
                [task.query_id for task in store.pending(tasks, rerun_failed=True)],
                ["q2"],
            )
            self.assertEqual(len(preflight_official_directory(store.official_dir)), 2)

    def test_unsafe_query_ids_use_stable_hash_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = BrowseCompArtifactStore(Path(temporary))
            path = store.write(self._record("../../not-a-path"))
            self.assertEqual(path.parent, store.official_dir)
            self.assertTrue(path.name.startswith("query-"))
            self.assertNotIn("..", path.name)


class BrowseCompBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_worker_is_an_ordinary_mini_agent(self) -> None:
        model = _SearchThenAnswerModel()
        outcome = await run_mini_agent_task(
            BrowseCompTask("q1", "Where is alpha?"),
            model_factory=lambda task: model,
            environment_factory=lambda task: WebEnvironment(
                JsonlSearchBackend(FIXTURES / "corpus.jsonl"), snippet_chars=64
            ),
            system_prompt="",
            max_steps=4,
            limits=BudgetLimits(max_model_calls=4, max_tool_calls=4),
            capture_content=True,
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.steps, 2)
        self.assertIn("Exact Answer: alpha", outcome.answer)
        self.assertEqual(outcome.accounting.tool_call_counts, {"search": 1})
        self.assertTrue(outcome.accounting.retrieved_docids)
        self.assertEqual(outcome.usage["input_tokens"], 20)
        self.assertIn("Question: Where is alpha?", model.requests[0][-1].content)
        self.assertEqual(
            [event["event"] for event in outcome.trace].count("model_call_completed"),
            2,
        )

    async def test_bounded_standard_layout_resume_fingerprint_and_retry(self) -> None:
        tasks = tuple(
            BrowseCompTask(query_id, f"question {query_id}")
            for query_id in ("q3", "q1", "q2")
        )
        active = 0
        peak = 0
        attempts: dict[str, int] = {}

        async def worker(task: BrowseCompTask) -> BrowseCompTaskOutcome:
            nonlocal active, peak
            attempts[task.query_id] = attempts.get(task.query_id, 0) + 1
            active += 1
            peak = max(peak, active)
            await asyncio.sleep({"q1": 0.03, "q2": 0.02, "q3": 0.01}[task.query_id])
            active -= 1
            if task.query_id == "q2" and attempts[task.query_id] == 1:
                return BrowseCompTaskOutcome(
                    status="agent_error",
                    accounting=WebAccounting({"search": 1}, ("2",)),
                    trace=({"event": "failed"},),
                    error_type="ProviderError",
                    error_message="offline failure",
                )
            return BrowseCompTaskOutcome(
                status="completed",
                answer=f"answer {task.query_id}",
                steps=1,
                accounting=WebAccounting({"search": 1}, (task.query_id,)),
                usage={"input_tokens": 1, "output_tokens": 1},
                trace=({"event": task.query_id},),
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            runner = BrowseCompBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=worker,
                max_workers=2,
                manifest={"dataset": "fixture", "profile": "web/openai"},
            )
            first = await runner.run(tasks)
            self.assertEqual(peak, 2)
            self.assertEqual(first.attempted, 3)
            self.assertEqual(first.completed, 2)
            self.assertEqual(first.failed, 1)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            for query_id in ("q1", "q2", "q3"):
                self.assertTrue((output / "instances" / query_id / "result.json").is_file())
                self.assertTrue((output / "instances" / query_id / "trace.jsonl").is_file())
                self.assertTrue((output / "official" / f"{query_id}.json").is_file())
            self.assertFalse(any(output.rglob("*.tmp")))
            self.assertEqual(len(preflight_official_directory(output / "official")), 3)

            resumed = await runner.run(tasks, resume=True)
            self.assertEqual(resumed.attempted, 0)
            self.assertEqual(resumed.skipped, 3)
            self.assertEqual(attempts, {"q1": 1, "q2": 1, "q3": 1})

            stale = output / "instances" / "q1" / ".running"
            stale.write_text('{"query_id":"q1"}\n', encoding="utf-8")
            stale_resumed = await runner.run(tasks, resume=True)
            self.assertEqual(stale_resumed.attempted, 0)
            self.assertFalse(stale.exists())
            self.assertEqual(attempts, {"q1": 1, "q2": 1, "q3": 1})

            retried = await runner.run(tasks, resume=True, retry_errors=True)
            self.assertEqual(retried.attempted, 1)
            self.assertEqual(retried.skipped, 2)
            self.assertEqual(retried.completed, 3)
            self.assertEqual(attempts["q2"], 2)
            q2 = json.loads(
                (output / "instances" / "q2" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(q2["status"], "completed")

            mismatched = BrowseCompBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=worker,
                max_workers=2,
                manifest={"dataset": "different"},
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                await mismatched.run(tasks, resume=True)

    async def test_direct_workers_preserve_normal_cancellation_semantics(self) -> None:
        task = BrowseCompTask("q1", "wait")
        started = asyncio.Event()
        single = asyncio.create_task(
            run_mini_agent_task(
                task,
                model_factory=lambda _: _BlockingModel(started),
                environment_factory=lambda _: WebEnvironment(
                    JsonlSearchBackend(FIXTURES / "corpus.jsonl")
                ),
                system_prompt="",
                max_steps=2,
                limits=BudgetLimits(max_model_calls=2, max_tool_calls=2),
            )
        )
        await started.wait()
        single.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await single

        started = asyncio.Event()
        multi = asyncio.create_task(
            run_multi_agent_task(
                task,
                model_factory=lambda _task, _agent, _profile: _BlockingModel(started),
                environment_factory=lambda _task, _agent, _profile: WebEnvironment(
                    JsonlSearchBackend(FIXTURES / "corpus.jsonl")
                ),
                system_prompt="",
                max_steps=2,
                limits=BudgetLimits(max_model_calls=2, max_tool_calls=2),
                max_agents=1,
            )
        )
        await started.wait()
        multi.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await multi

    async def test_multi_agent_worker_aggregates_retrieval_into_same_layout(self) -> None:
        task = BrowseCompTask("q1", "Find alpha and beta evidence")
        shared_backend = JsonlSearchBackend(FIXTURES / "corpus.jsonl")
        environments: dict[str, WebEnvironment] = {}
        selected_profiles: dict[str, str | None] = {}
        models = {
            "/root": ScriptedModel(
                (
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall(
                                "spawn",
                                "spawn_agent",
                                {"task": "find beta", "profile": "researcher"},
                            ),
                        ),
                    ),
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall(
                                "wait", "wait", {"agent_ids": ["/root/1"]}
                            ),
                        ),
                    ),
                    ModelResponse(
                        text="",
                        tool_calls=(ToolCall("read", "read_messages", {}),),
                    ),
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall("root-search", "search", {"query": "alpha"}),
                        ),
                    ),
                    ModelResponse(
                        text=(
                            "Explanation: alpha and beta [1] [2]\n"
                            "Exact Answer: alpha and beta\nConfidence: 100%"
                        )
                    ),
                )
            ),
            "/root/1": ScriptedModel(
                (
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall("child-search", "search", {"query": "beta"}),
                        ),
                    ),
                    ModelResponse(text="beta is in document 2"),
                )
            ),
        }

        def environment_factory(
            selected_task: BrowseCompTask,
            agent_id: str,
            profile: str | None,
        ) -> WebEnvironment:
            self.assertIs(selected_task, task)
            selected_profiles[f"environment:{agent_id}"] = profile
            environment = WebEnvironment(
                shared_backend, top_k=1, snippet_chars=64
            )
            environments[agent_id] = environment
            return environment

        def model_factory(
            selected_task: BrowseCompTask,
            agent_id: str,
            profile: str | None,
        ) -> ScriptedModel:
            self.assertIs(selected_task, task)
            selected_profiles[agent_id] = profile
            return models[agent_id]

        async def worker(selected_task: BrowseCompTask) -> BrowseCompTaskOutcome:
            return await run_multi_agent_task(
                selected_task,
                model_factory=model_factory,
                environment_factory=environment_factory,
                system_prompt="",
                max_steps=8,
                limits=BudgetLimits(
                    max_model_calls=10,
                    max_tool_calls=10,
                    max_concurrency=2,
                ),
                max_agents=2,
                per_agent_limits=BudgetLimits(
                    max_model_calls=6, max_tool_calls=6
                ),
                allowed_child_profiles=("researcher",),
                capture_content=True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "multi-run"
            summary = await BrowseCompBatchRunner(
                output_dir=output,
                model_name_or_path="test/multi-model",
                worker=worker,
                max_workers=1,
                manifest={"mode": "multi", "max_agents": 2},
            ).run((task,))
            self.assertEqual(summary.completed, 1)
            self.assertEqual(
                selected_profiles,
                {
                    "/root": None,
                    "/root/1": "researcher",
                    "environment:/root": None,
                    "environment:/root/1": "researcher",
                },
            )
            self.assertEqual(set(environments), {"/root", "/root/1"})
            self.assertIsNot(environments["/root"], environments["/root/1"])
            self.assertIs(environments["/root"].backend, shared_backend)
            self.assertIs(environments["/root/1"].backend, shared_backend)

            official = json.loads(
                (output / "official" / "q1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(official["tool_call_counts"], {"search": 2})
            self.assertEqual(official["retrieved_docids"], ["1", "2"])
            self.assertIn("Exact Answer: alpha and beta", official["result"][0]["output"])
            self.assertEqual(official["metadata"]["agent_count"], 2)
            self.assertEqual(
                official["metadata"]["agents"]["/root/1"]["status"], "completed"
            )
            trace = [
                json.loads(line)
                for line in (output / "instances" / "q1" / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in trace].count("agent_spawned"), 2
            )
            self.assertIn("message_sent", [event["event"] for event in trace])
            self.assertIn(
                "Question: Find alpha and beta evidence",
                models["/root"].queries[0][0][0].content,
            )


class BrowseCompProvenanceTests(unittest.TestCase):
    def test_directory_identity_is_content_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model":"judge"}\n')
            (root / "nested").mkdir()
            (root / "nested" / "tokenizer.json").write_text("tokens\n")

            first = directory_identity(root)
            second = directory_identity(root)
            self.assertEqual(first, second)
            self.assertEqual(first["files"], 2)
            self.assertEqual(first["path"], str(root.resolve()))

            (root / "nested" / "tokenizer.json").write_text("changed\n")
            self.assertNotEqual(directory_identity(root)["sha256"], first["sha256"])


class BrowseCompEvaluatorTests(unittest.TestCase):
    def test_safe_pinned_evaluator_argv_and_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "checkout"
            evaluator = checkout / "scripts_evaluation" / "evaluate_run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("# fixture\n", encoding="utf-8")
            qrels = checkout / "topics-qrels" / "qrel_evidence.txt"
            qrels.parent.mkdir()
            qrels.write_text("q1 0 1 1\n", encoding="utf-8")
            truth = root / "truth.jsonl"
            truth.write_text(
                '{"query_id":"q1","query":"question","answer":"answer"}\n',
                encoding="utf-8",
            )
            judge_model = root / "judge-model-snapshot"
            judge_model.mkdir()
            store = BrowseCompArtifactStore(root / "run")
            store.write(
                BrowseCompRunRecord.from_answer(
                    BrowseCompTask("q1", "question"),
                    "answer",
                    WebAccounting({"search": 1}, ("1",)),
                )
            )
            with patch(
                "mini_agent.evals.browsecomp_plus.verify_browsecomp_checkout",
                return_value=BROWSECOMP_PLUS_REVISION,
            ):
                argv = official_evaluator_argv(
                    checkout=checkout,
                    input_dir=store.official_dir,
                    ground_truth=truth,
                    eval_dir=root / "evals",
                    python_executable="python-fixture",
                    model=judge_model,
                    tensor_parallel_size=2,
                )
            self.assertEqual(argv[0], "python-fixture")
            self.assertEqual(argv[1], str(evaluator.resolve()))
            self.assertIn(str(store.official_dir), argv)
            self.assertIn(str(qrels.resolve()), argv)
            self.assertIn(str(judge_model.resolve()), argv)
            self.assertEqual(argv[-2:], ("--tensor_parallel_size", "2"))

    def test_checkout_verification_uses_literal_git_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            evaluator = checkout / "scripts_evaluation" / "evaluate_run.py"
            evaluator.parent.mkdir()
            evaluator.write_text("# fixture\n", encoding="utf-8")
            responses = (
                subprocess.CompletedProcess([], 0, BROWSECOMP_PLUS_REVISION + "\n", ""),
                subprocess.CompletedProcess([], 0, "", ""),
            )
            with patch(
                "mini_agent.evals.browsecomp_plus.subprocess.run",
                side_effect=responses,
            ) as run:
                self.assertEqual(
                    verify_browsecomp_checkout(checkout),
                    BROWSECOMP_PLUS_REVISION,
                )
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                argv = call.args[0]
                self.assertEqual(argv[0], "git")
                self.assertNotIn("shell", call.kwargs)
                self.assertIsInstance(argv, tuple)


class BrowseCompSourceProfileTests(unittest.TestCase):
    def test_all_eight_pinned_source_families_are_honestly_partitioned(self) -> None:
        expected_parsers = {
            "openai": "provider_tool_calls",
            "anthropic": "provider_tool_calls",
            "glm": "provider_tool_calls",
            "gemini": "gemini_function_calls",
            "gpt-oss": "oss_retrieval_tool_calls",
            "qwen3": "qwen_mcp_tool_calls",
            "search-r1": "search_r1_tags",
            "tongyi": "tongyi_react_tags",
        }
        for name, response_parser in expected_parsers.items():
            with self.subTest(profile=name):
                profile = load_profile("web", name)
                self.assertEqual(profile.fidelity, "profile")
                self.assertEqual(profile.source["revision"], BROWSECOMP_PLUS_REVISION)
                self.assertTrue(profile.source["implementation"])
                self.assertTrue(profile.fidelity_gaps)
                self.assertEqual(profile.tools, ("search",))
                self.assertEqual(profile.benchmark["top_k"], 5)
                self.assertEqual(profile.response_parser, response_parser)
                environment = WebEnvironment.from_policy(
                    _StaticBackend(),
                    benchmark=profile.benchmark,
                    observation=profile.observation,
                    tools=profile.tools,
                    tokenizer=_WordTokenizer(),
                )
                self.assertEqual(environment.snippet_tokens, 512)


class BrowseCompWebModelTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _search_tool() -> ToolDefinition:
        return ToolDefinition(
            "search",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def test_all_eight_profiles_resolve_to_runnable_models(self) -> None:
        for name in (
            "openai",
            "anthropic",
            "glm",
            "gemini",
            "gpt-oss",
            "qwen3",
            "search-r1",
            "tongyi",
        ):
            with self.subTest(profile=name):
                profile = load_profile("web", name)
                backend = _ModelBackend(ModelResponse(text="final"))
                model = build_web_model(
                    backend,
                    response_parser=profile.response_parser,
                    max_output_tokens=profile.generation.get("max_output_tokens"),
                )
                environment = WebEnvironment.from_policy(
                    _StaticBackend(),
                    benchmark=profile.benchmark,
                    observation=profile.observation,
                    tools=profile.tools,
                    tokenizer=_WordTokenizer(),
                )
                result = await MiniAgent(
                    model=model,
                    environment=environment,
                    system_prompt=profile.system_prompt,
                    max_steps=2,
                ).run("question")
                self.assertEqual(result.answer, "final")
                self.assertEqual(result.steps, 1)

    async def test_oss_and_qwen_tool_schemas_and_calls_are_normalized(self) -> None:
        cases = (
            (
                "oss_retrieval_tool_calls",
                "local_knowledge_base_retrieval",
                {"user_query": "alpha"},
            ),
            ("qwen_mcp_tool_calls", "search-server-search", {"query": "alpha"}),
        )
        for parser, upstream_name, arguments in cases:
            with self.subTest(parser=parser):
                backend = _ModelBackend(
                    ModelResponse(
                        text="",
                        tool_calls=(ToolCall("call", upstream_name, arguments),),
                        continuation={"provider": "fixture"},
                    )
                )
                model = build_web_model(backend, response_parser=parser)
                response = await model.query(
                    (Message(role="user", content="question"),),
                    (self._search_tool(),),
                )
                request = backend.requests[0]
                self.assertEqual(request.tools[0].name, upstream_name)
                self.assertEqual(response.tool_calls[0].name, "search")
                self.assertEqual(response.tool_calls[0].arguments, {"query": "alpha"})

    async def test_tag_parsers_run_search_round_trip_over_text_transcript(self) -> None:
        cases = (
            (
                "search_r1_tags",
                "<think>x</think><search>alpha evidence</search>",
                "<answer>alpha</answer>",
                "<information>search output</information>",
            ),
            (
                "tongyi_react_tags",
                '<think>x</think><tool_call>{"name":"search",'
                '"arguments":{"query":"alpha evidence"}}</tool_call>',
                "<answer>alpha</answer>",
                "<tool_response>\nsearch output\n</tool_response>",
            ),
        )
        for parser, action_text, final_text, expected_observation in cases:
            with self.subTest(parser=parser):
                backend = _ModelBackend(
                    ModelResponse(text=action_text), ModelResponse(text=final_text)
                )
                model = build_web_model(
                    backend,
                    response_parser=parser,
                    agent_id="/browsecomp/q1",
                    metadata={"fixture": True},
                )
                user = Message(role="user", content="question")
                first = await model.query((user,), (self._search_tool(),))
                self.assertEqual(first.tool_calls[0].name, "search")
                self.assertEqual(
                    first.tool_calls[0].arguments, {"query": "alpha evidence"}
                )
                second = await model.query(
                    (
                        user,
                        Message(
                            role="assistant",
                            content=first.text,
                            tool_calls=first.tool_calls,
                        ),
                        Message(
                            role="tool",
                            tool_results=(
                                ToolResult(
                                    first.tool_calls[0].call_id,
                                    "search",
                                    "search output",
                                ),
                            ),
                        ),
                    ),
                    (self._search_tool(),),
                )
                self.assertEqual(second.text, final_text)
                self.assertEqual(second.tool_calls, ())
                self.assertIn(expected_observation, backend.requests[1].prompt)
                self.assertEqual(backend.requests[1].agent_id, "/browsecomp/q1")
                self.assertEqual(backend.requests[1].metadata, {"fixture": True})

    def test_tongyi_parser_fails_closed_on_malformed_tag(self) -> None:
        with self.assertRaisesRegex(ProviderError, "malformed"):
            parse_web_response(
                ModelResponse(text="<tool_call>{bad}</tool_call>"),
                "tongyi_react_tags",
            )


if __name__ == "__main__":
    unittest.main()
