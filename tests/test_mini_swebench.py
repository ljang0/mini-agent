from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from mini_agent.environments.swe import BashEnvironment, ProcessResult
from mini_agent.environments.swebench import (
    DockerSWEEnvironment,
    swebench_doctor,
    swebench_image_name,
)
from mini_agent.evals.swebench import (
    MiniSWETextActionModel,
    OFFICIAL_PREDICTION_FIELDS,
    SWEbenchBatchRunner,
    SWEbenchInstance,
    SWEbenchInstanceOutcome,
    load_swebench_jsonl,
    official_grader_argv,
    parse_mini_swe_action,
    run_mini_agent_instance,
    run_multi_agent_instance,
    run_official_grader,
    select_swebench_instances,
)
from mini_agent.models import ScriptedModel
from mini_agent.agent import MiniAgent
from mini_agent.profiles import load_profile
from mini_agent.types import (
    BudgetLimits,
    ModelResponse,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
)


class FakeProcessRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 256 * 1024,
    ) -> ProcessResult:
        del cwd, environment, timeout_seconds, max_output_bytes
        call = tuple(argv)
        self.calls.append(call)
        if call[1:4] == ("image", "inspect", "--format"):
            output = b"sha256:resolved-image\n"
        elif call[1] == "exec" and "git diff --cached" in call[-1]:
            output = b"diff --git a/a.py b/a.py\n"
        else:
            output = b"ok\n"
        return ProcessResult(output, 0, len(output))


class FakeAgentWorkspace:
    def __init__(
        self,
        agent_id: str,
        events: list[str],
        *,
        resource_identity: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.events = events
        self._identity = resource_identity or f"workspace:{agent_id}"
        self.closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition(name="bash"),)

    async def execute(self, action: ToolCall) -> ToolExecution:
        self.events.append(f"execute:{self.agent_id}:{action.name}")
        return ToolExecution(output=f"ran in {self.agent_id}")

    async def export_patch(self) -> bytes:
        self.events.append(f"export:{self.agent_id}")
        return f"patch from {self.agent_id}\n".encode("utf-8")

    async def close(self) -> None:
        self.events.append(f"close:{self.agent_id}")
        self.closed = True

    def resource_identity(self) -> str:
        return self._identity


class LocalSWEEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_copy_has_private_baseline_home_and_durable_binary_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / ".git").write_text("gitdir: /outside/repository\n")
            (source / "tracked.txt").write_text("original\n", encoding="utf-8")
            (source / "deleted.txt").write_text("delete\n", encoding="utf-8")
            (source / "binary.bin").write_bytes(b"\x00\x01\x02")
            environment = await BashEnvironment.isolated(source)
            workspace = environment.workspace
            self.assertFalse((workspace / ".git").is_file())
            status = await environment.execute(
                ToolCall("status", "bash", {"command": "git status --porcelain"})
            )
            self.assertEqual(status.output, "")
            home = await environment.execute(
                ToolCall("home", "bash", {"command": "printf %s \"$HOME\""})
            )
            self.assertEqual(home.output, str(environment.home))
            self.assertNotEqual(environment.home, workspace)

            (workspace / "tracked.txt").write_text("changed\n", encoding="utf-8")
            (workspace / "new.txt").write_text("new\n", encoding="utf-8")
            (workspace / "deleted.txt").unlink()
            (workspace / "binary.bin").write_bytes(b"\x00\xff\xfe")
            durable = root / "artifacts" / "patch.diff"
            patch = await environment.export_patch(durable)
            patch_text = patch.decode("utf-8")
            self.assertEqual(durable.read_bytes(), patch)
            self.assertIn("diff --git a/tracked.txt b/tracked.txt", patch_text)
            self.assertIn("diff --git a/new.txt b/new.txt", patch_text)
            self.assertIn("deleted file mode", patch_text)
            self.assertIn("GIT binary patch", patch_text)
            self.assertNotIn("gitdir: /outside/repository", patch_text)
            await environment.close()
            self.assertFalse(workspace.exists())
            self.assertTrue(durable.is_file())
            self.assertEqual(
                (source / "tracked.txt").read_text(encoding="utf-8"), "original\n"
            )

    async def test_copy_rejects_absolute_and_escaping_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "escape").symlink_to(root / "outside")
            with self.assertRaisesRegex(ValueError, "absolute symlink"):
                await BashEnvironment.isolated(source)

            (source / "escape").unlink()
            (source / "escape").symlink_to("../../outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                await BashEnvironment.isolated(source)

    async def test_long_success_output_keeps_head_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            environment = await BashEnvironment.isolated(
                source, max_output_bytes=20
            )
            result = await environment.execute(
                ToolCall(
                    "long",
                    "bash",
                    {"command": "printf 0123456789abcdefghijklmnopqrstuvWXYZ"},
                )
            )
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["output_truncated"])
            self.assertTrue(result.output.startswith("0123456789"))
            self.assertTrue(result.output.endswith("qrstuvWXYZ"))
            self.assertIn("bytes omitted", result.output)
            await environment.close()

    async def test_oversized_patch_fails_closed_and_copy_can_be_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "value.txt").write_text("old\n", encoding="utf-8")
            environment = await BashEnvironment.isolated(
                source, max_patch_bytes=64
            )
            workspace = environment.workspace
            (workspace / "value.txt").write_text("new content\n" * 100)
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                await environment.export_patch()
            await environment.close()
            self.assertFalse(workspace.exists())


class DockerSWEEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_persistent_container_is_argv_only_and_exports_patch(self) -> None:
        runner = FakeProcessRunner()
        instance = {"instance_id": "Owner__Repo-123"}
        environment = await DockerSWEEnvironment.create(
            instance, runner=runner, platform="linux/amd64"
        )
        self.assertEqual(
            environment.image,
            "docker.io/swebench/sweb.eval.x86_64.owner_1776_repo-123:latest",
        )
        self.assertEqual(environment.image_id, "sha256:resolved-image")
        action = "printf safe; touch /testbed/file"
        observation = await environment.execute(
            ToolCall("bash", "bash", {"command": action})
        )
        self.assertFalse(observation.is_error)
        patch = await environment.export_patch()
        self.assertEqual(patch, b"diff --git a/a.py b/a.py\n")
        await environment.close()

        run = runner.calls[0]
        self.assertEqual(run[:2], ("docker", "run"))
        self.assertIn("--platform", run)
        exec_call = next(call for call in runner.calls if call[1] == "exec")
        self.assertEqual(exec_call[-1], action)
        self.assertNotIn("OPENAI_API_KEY", " ".join(run))
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    def test_explicit_image_precedence_and_validation(self) -> None:
        self.assertEqual(
            swebench_image_name(
                {"instance_id": "a__b-1", "image_name": "registry/image@sha256:1"}
            ),
            "registry/image@sha256:1",
        )
        with self.assertRaisesRegex(ValueError, "image"):
            swebench_image_name({"instance_id": "a__b-1", "image_name": "-bad"})

    async def test_doctor_is_non_mutating_and_reports_runtime_and_image(self) -> None:
        runner = FakeProcessRunner()
        report = await swebench_doctor(
            runner=runner, image="docker.io/swebench/example:latest"
        )
        self.assertTrue(report.ok)
        self.assertEqual(
            [check.name for check in report.checks],
            ["runtime_version", "daemon_platform", "image_available"],
        )
        self.assertFalse(any(call[1] in {"run", "exec", "rm"} for call in runner.calls))
        self.assertTrue(report.as_dict()["ok"])

    async def test_failed_image_identity_check_removes_started_container(self) -> None:
        class FailingInspectRunner(FakeProcessRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                result = await super().run(argv, **kwargs)
                if tuple(argv)[1:3] == ("image", "inspect"):
                    return ProcessResult(b"missing image\n", 1, 14)
                return result

        runner = FailingInspectRunner()
        with self.assertRaisesRegex(RuntimeError, "image identity"):
            await DockerSWEEnvironment.create(
                {"instance_id": "a__repo-1"}, runner=runner
            )
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))


class SWEbenchDatasetTests(unittest.TestCase):
    def test_jsonl_validation_and_deterministic_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "problem_statement": f"fix {instance_id}",
                        }
                    )
                    for instance_id in ("c__repo-3", "a__repo-1", "b__repo-2")
                )
                + "\n",
                encoding="utf-8",
            )
            instances = load_swebench_jsonl(path)
            first = select_swebench_instances(instances, shuffle=True, seed=7)
            second = select_swebench_instances(
                tuple(reversed(instances)), shuffle=True, seed=7
            )
            self.assertEqual(
                [item.instance_id for item in first],
                [item.instance_id for item in second],
            )
            selected = select_swebench_instances(
                instances, filter_pattern=r"[ab]__", start=1
            )
            self.assertEqual([item.instance_id for item in selected], ["b__repo-2"])

            with self.assertRaisesRegex(ValueError, "duplicate"):
                path.write_text(
                    json.dumps(
                        {"instance_id": "same", "problem_statement": "one"}
                    )
                    + "\n"
                    + json.dumps(
                        {"instance_id": "same", "problem_statement": "two"}
                    ),
                    encoding="utf-8",
                )
                load_swebench_jsonl(path)

    def test_action_parser_variants_are_explicit_about_fidelity(self) -> None:
        tool_call = load_profile("swe", "mini-swe-tool-call")
        self.assertEqual(tool_call.fidelity, "profile")
        self.assertEqual(tool_call.response_parser, "provider_tool_calls")
        references = {
            "mini-swe-text": "mini_swe_text",
            "mini-swe-backticks": "mini_swe_backticks",
            "mini-swe-xml": "mini_swe_xml",
        }
        for name, parser in references.items():
            profile = load_profile("swe", name)
            self.assertEqual(profile.fidelity, "profile")
            self.assertEqual(profile.provider, "")
            self.assertEqual(profile.response_parser, parser)
            self.assertTrue(profile.fidelity_gaps)

        swe_agent = load_profile("swe", "swe-agent-bash")
        self.assertEqual(swe_agent.fidelity, "profile")
        self.assertIn("submit command", " ".join(swe_agent.fidelity_gaps))

    def test_text_action_parsers_are_strict_and_deterministic(self) -> None:
        backtick = """reasoning
```mswea_bash_command
printf hello
```
"""
        for parser in ("mini_swe_text", "mini_swe_backticks"):
            self.assertEqual(
                parse_mini_swe_action(backtick, parser), "printf hello"
            )
        xml = "reasoning <mswea_bash_command>printf hello</mswea_bash_command>"
        self.assertEqual(parse_mini_swe_action(xml, "mini_swe_xml"), "printf hello")
        with self.assertRaisesRegex(ProtocolError, "exactly one action"):
            parse_mini_swe_action("no action", "mini_swe_backticks")
        with self.assertRaisesRegex(ProtocolError, "found 2"):
            parse_mini_swe_action(backtick + backtick, "mini_swe_text")


class SWEbenchBatchTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _instances() -> tuple[SWEbenchInstance, ...]:
        return tuple(
            SWEbenchInstance(instance_id, f"fix {instance_id}", {})
            for instance_id in ("c__repo-3", "a__repo-1", "b__repo-2")
        )

    async def test_bounded_run_official_predictions_resume_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = 0
            peak = 0
            attempted: list[str] = []

            async def worker(instance: SWEbenchInstance) -> SWEbenchInstanceOutcome:
                nonlocal active, peak
                attempted.append(instance.instance_id)
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(
                    {"a__repo-1": 0.03, "b__repo-2": 0.02, "c__repo-3": 0.01}[
                        instance.instance_id
                    ]
                )
                active -= 1
                if instance.instance_id == "b__repo-2":
                    return SWEbenchInstanceOutcome(
                        status="agent_error",
                        error_type="ProviderError",
                        error_message="offline failure",
                    )
                return SWEbenchInstanceOutcome(
                    status="completed",
                    patch=f"patch for {instance.instance_id}\n".encode(),
                    trace=({"event": instance.instance_id},),
                )

            output = Path(directory) / "run"
            runner = SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=worker,
                max_workers=2,
                manifest={"dataset": "fixture", "revision": "fixed"},
            )
            summary = await runner.run(self._instances())
            self.assertEqual(peak, 2)
            self.assertEqual(summary.attempted, 3)
            self.assertEqual(summary.completed, 2)
            predictions = [
                json.loads(line)
                for line in summary.predictions_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["instance_id"] for item in predictions],
                ["a__repo-1", "b__repo-2", "c__repo-3"],
            )
            self.assertTrue(
                all(tuple(item) == OFFICIAL_PREDICTION_FIELDS for item in predictions)
            )
            self.assertEqual(predictions[1]["model_patch"], "")
            self.assertFalse(any(output.rglob("*.tmp")))

            attempted.clear()
            resumed = await runner.run(self._instances(), resume=True)
            self.assertEqual(resumed.attempted, 0)
            self.assertEqual(resumed.skipped, 3)
            self.assertEqual(attempted, [])

            async def retry(instance: SWEbenchInstance) -> SWEbenchInstanceOutcome:
                attempted.append(instance.instance_id)
                return SWEbenchInstanceOutcome(status="completed", patch=b"fixed\n")

            retry_runner = SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=retry,
                max_workers=2,
                manifest={"dataset": "fixture", "revision": "fixed"},
            )
            retried = await retry_runner.run(
                self._instances(), resume=True, retry_errors=True
            )
            self.assertEqual(retried.attempted, 1)
            self.assertEqual(attempted, ["b__repo-2"])
            self.assertEqual(retried.completed, 3)
            mismatched = SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=retry,
                max_workers=2,
                manifest={"dataset": "different"},
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                await mismatched.run(self._instances(), resume=True)

    async def test_cancellation_is_durable_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            started = asyncio.Event()

            async def blocked(
                instance: SWEbenchInstance,
            ) -> SWEbenchInstanceOutcome:
                del instance
                started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            output = Path(directory) / "run"
            instance = SWEbenchInstance("a__repo-1", "fix", {})
            runner = SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=blocked,
            )
            task = asyncio.create_task(runner.run((instance,)))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            result_path = next((output / "instances").glob("*/result.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "cancelled")
            self.assertFalse((result_path.parent / ".running").exists())

            attempts = 0

            async def finish(
                instance: SWEbenchInstance,
            ) -> SWEbenchInstanceOutcome:
                nonlocal attempts
                del instance
                attempts += 1
                return SWEbenchInstanceOutcome(status="completed", patch=b"fixed\n")

            resumed = SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/model",
                worker=finish,
            )
            summary = await resumed.run((instance,), resume=True)
            self.assertEqual(attempts, 1)
            self.assertEqual(summary.completed, 1)

    async def test_mini_agent_worker_exports_before_environment_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "value.txt").write_text("old\n", encoding="utf-8")
            workspace: Path | None = None

            async def environment_factory(
                instance: SWEbenchInstance,
            ) -> BashEnvironment:
                nonlocal workspace
                del instance
                environment = await BashEnvironment.isolated(source)
                workspace = environment.workspace
                return environment

            model = ScriptedModel(
                [
                    ModelResponse(
                        text="",
                        tool_calls=(
                            ToolCall(
                                "edit",
                                "bash",
                                {"command": "printf 'new\\n' > value.txt"},
                            ),
                        )
                    ),
                    ModelResponse(text="done"),
                ]
            )
            outcome = await run_mini_agent_instance(
                SWEbenchInstance("a__repo-1", "fix it", {}),
                model_factory=lambda _instance: model,
                environment_factory=environment_factory,
                system_prompt="Use bash.",
                max_steps=4,
                limits=BudgetLimits(
                    max_model_calls=4,
                    max_tool_calls=4,
                    wall_time_seconds=30,
                ),
            )
            self.assertEqual(outcome.status, "completed")
            self.assertIn(b"diff --git a/value.txt b/value.txt", outcome.patch)
            self.assertTrue(outcome.trace)
            assert workspace is not None
            self.assertFalse(workspace.exists())
            self.assertEqual(source.read_text() if source.is_file() else "", "")
            self.assertEqual(
                (source / "value.txt").read_text(encoding="utf-8"), "old\n"
            )

    async def test_text_adapter_runs_actions_and_submission_through_mini_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            base = ScriptedModel(
                [
                    ModelResponse(
                        text=(
                            "inspect\n```mswea_bash_command\n"
                            "printf observed\n```"
                        )
                    ),
                    ModelResponse(
                        text=(
                            "done\n```mswea_bash_command\n"
                            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
                        )
                    )
                ]
            )
            model = MiniSWETextActionModel(
                response_parser="mini_swe_backticks", model=base
            )
            environment = await BashEnvironment.isolated(source)
            result = await MiniAgent(
                model=model,
                environment=environment,
                system_prompt="Use the required action syntax.",
                max_steps=4,
            ).run("fix it")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.answer, "Task complete.")
            self.assertEqual(result.steps, 2)
            self.assertEqual(len(base.queries), 2)
            self.assertEqual(
                result.messages[-2].tool_results[0].output,
                "observed",
            )
            await environment.close()

    async def test_backend_text_adapter_replays_observations_as_text(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.requests: list[Any] = []
                self.responses = [
                    ModelResponse(
                        text=(
                            "```mswea_bash_command\nprintf observed\n```"
                        )
                    ),
                    ModelResponse(
                        text=(
                            "```mswea_bash_command\n"
                            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
                        )
                    ),
                ]

            async def complete(self, request: Any) -> ModelResponse:
                self.requests.append(request)
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            backend = Backend()
            model = MiniSWETextActionModel(
                response_parser="mini_swe_text",
                backend=backend,
            )
            environment = await BashEnvironment.isolated(source)
            result = await MiniAgent(
                model=model,
                environment=environment,
                system_prompt="system",
                max_steps=5,
            ).run("task")
            self.assertEqual(result.steps, 2)
            self.assertEqual(len(backend.requests), 2)
            self.assertEqual(backend.requests[0].tools, ())
            self.assertIn(
                "<observation>\nobserved\n</observation>",
                backend.requests[1].prompt,
            )
            await environment.close()

    async def test_multi_agent_uses_isolated_workspaces_and_root_patch_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            environments: dict[str, FakeAgentWorkspace] = {}
            profiles: dict[str, str | None] = {}
            models = {
                "/root": ScriptedModel(
                    [
                        ModelResponse(
                            text="",
                            tool_calls=(
                                ToolCall(
                                    "spawn",
                                    "spawn_agent",
                                    {"task": "inspect", "profile": "research"},
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
                            text="", tool_calls=(ToolCall("read", "read_messages", {}),)
                        ),
                        ModelResponse(
                            text="",
                            tool_calls=(
                                ToolCall("edit", "bash", {"command": "edit root"}),
                            ),
                        ),
                        ModelResponse(text="root answer"),
                    ]
                ),
                "/root/1": ScriptedModel(
                    [
                        ModelResponse(
                            text="",
                            tool_calls=(
                                ToolCall("inspect", "bash", {"command": "inspect"}),
                            ),
                        ),
                        ModelResponse(
                            text="",
                            tool_calls=(
                                ToolCall(
                                    "message",
                                    "send_message",
                                    {
                                        "agent_id": "/root",
                                        "message": "child finding",
                                    },
                                ),
                            ),
                        ),
                        ModelResponse(text="child answer"),
                    ]
                ),
            }

            async def environment_factory(
                instance: SWEbenchInstance,
                agent_id: str,
                profile: str | None,
            ) -> FakeAgentWorkspace:
                del instance
                profiles[f"environment:{agent_id}"] = profile
                environment = FakeAgentWorkspace(agent_id, events)
                environments[agent_id] = environment
                return environment

            def builder(
                instance: SWEbenchInstance,
                agent_id: str,
                environment: Any,
                shared: Any,
                profile: str | None,
            ) -> MiniAgent:
                del instance
                profiles[agent_id] = profile
                return MiniAgent(
                    model=models[agent_id],
                    environment=environment,
                    context=shared,
                    agent_id=agent_id,
                    max_steps=8,
                )

            instance = SWEbenchInstance("a__repo-1", "fix it", {})
            outcome = await run_multi_agent_instance(
                instance,
                agent_builder=builder,
                environment_factory=environment_factory,
                limits=BudgetLimits(max_model_calls=16, max_tool_calls=16),
                per_agent_limits=BudgetLimits(
                    max_model_calls=8, max_tool_calls=8
                ),
                max_agents=2,
                allowed_child_profiles=("research",),
                root_profile="lead",
            )
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.answer, "root answer")
            self.assertEqual(outcome.patch, b"patch from /root\n")
            self.assertEqual(
                profiles,
                {
                    "/root": "lead",
                    "/root/1": "research",
                    "environment:/root": "lead",
                    "environment:/root/1": "research",
                },
            )
            self.assertEqual(
                {environment.resource_identity() for environment in environments.values()},
                {"workspace:/root", "workspace:/root/1"},
            )
            self.assertTrue(all(environment.closed for environment in environments.values()))
            self.assertNotIn("export:/root/1", events)
            self.assertLess(events.index("export:/root"), events.index("close:/root"))
            self.assertEqual(outcome.metadata["agents"]["/root/1"]["profile"], "research")
            self.assertTrue(
                any(event["event"] == "message_sent" for event in outcome.trace)
            )

            async def completed(_: SWEbenchInstance) -> SWEbenchInstanceOutcome:
                return outcome

            output = Path(directory) / "artifacts"
            summary = await SWEbenchBatchRunner(
                output_dir=output,
                model_name_or_path="test/multi",
                worker=completed,
                manifest={"mode": "multi"},
            ).run((instance,))
            result_path = next((output / "instances").glob("*/result.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], "mini-agent-swebench-result-v1")
            self.assertEqual(result["metadata"]["mode"], "multi")
            prediction = json.loads(
                summary.predictions_path.read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["model_patch"], "patch from /root\n")

    async def test_multi_agent_allowlist_and_per_agent_budget_are_enforced(self) -> None:
        events: list[str] = []
        root = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    tool_calls=(
                        ToolCall(
                            "spawn",
                            "spawn_agent",
                            {"task": "child", "profile": "not-allowed"},
                        ),
                    ),
                ),
                ModelResponse(text="recovered"),
            ]
        )

        def builder(
            instance: SWEbenchInstance,
            agent_id: str,
            environment: Any,
            shared: Any,
            profile: str | None,
        ) -> MiniAgent:
            del instance, profile
            return MiniAgent(
                model=root,
                environment=environment,
                context=shared,
                agent_id=agent_id,
            )

        outcome = await run_multi_agent_instance(
            SWEbenchInstance("a__repo-1", "fix", {}),
            agent_builder=builder,
            environment_factory=lambda _instance, agent_id, _profile: FakeAgentWorkspace(
                agent_id, events
            ),
            limits=BudgetLimits(max_model_calls=4, max_tool_calls=4),
            per_agent_limits=BudgetLimits(max_model_calls=2, max_tool_calls=2),
            max_agents=2,
            allowed_child_profiles=("research",),
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.answer, "recovered")
        self.assertIn(
            "not allowlisted",
            root.queries[1][0][-1].tool_results[0].output,
        )
        self.assertEqual(outcome.metadata["agents"].keys(), {"/root"})

        budget_root = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    tool_calls=(ToolCall("bash", "bash", {"command": "one"}),),
                ),
                ModelResponse(text="should not run"),
            ]
        )

        def budget_builder(
            instance: SWEbenchInstance,
            agent_id: str,
            environment: Any,
            shared: Any,
            profile: str | None,
        ) -> MiniAgent:
            del instance, profile
            return MiniAgent(
                model=budget_root,
                environment=environment,
                context=shared,
                agent_id=agent_id,
            )

        exhausted = await run_multi_agent_instance(
            SWEbenchInstance("a__repo-2", "fix", {}),
            agent_builder=budget_builder,
            environment_factory=lambda _instance, agent_id, _profile: FakeAgentWorkspace(
                agent_id, events
            ),
            limits=BudgetLimits(max_model_calls=8, max_tool_calls=8),
            per_agent_limits=BudgetLimits(max_model_calls=1, max_tool_calls=4),
        )
        self.assertEqual(exhausted.status, "budget_exhausted")
        self.assertEqual(exhausted.patch, b"patch from /root\n")


class OfficialGraderAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_grader_uses_explicit_argv_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            argv = official_grader_argv(
                dataset_name="SWE-bench/SWE-bench_Lite",
                predictions_path=predictions,
                run_id="safe-run_1",
                max_workers=3,
                instance_ids=("a__repo-1",),
                python_executable="/usr/bin/python3",
            )
            self.assertEqual(argv[1:3], ("-m", "swebench.harness.run_evaluation"))
            self.assertEqual(argv[argv.index("--max_workers") + 1], "3")
            self.assertIn(str(predictions.resolve()), argv)
            with self.assertRaisesRegex(ValueError, "run_id"):
                official_grader_argv(
                    dataset_name="dataset",
                    predictions_path=predictions,
                    run_id="bad/run;id",
                    max_workers=1,
                )

            fake = FakeProcessRunner()
            result = await run_official_grader(
                dataset_name="SWE-bench/SWE-bench_Lite",
                predictions_path=predictions,
                run_id="safe-run",
                max_workers=1,
                runner=fake,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(fake.calls[0][1:3], ("-m", "swebench.harness.run_evaluation"))


if __name__ == "__main__":
    unittest.main()
