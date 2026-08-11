from __future__ import annotations

import base64
import asyncio
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import AsyncMock, patch

import httpx

from mini_agent import MiniAgent, ModelResponse, ScriptedModel, ToolCall
from mini_agent.environments.cua import (
    CUAEnvironment,
    CUASpeedRunClient,
    ComputerObservation,
    OSWorldClient,
    ProtocolError,
    translate_anthropic_action,
    translate_openai_action,
    validate_gym_action,
    validate_osworld_script,
    validate_png,
)
from mini_agent.evals.osworld import OSWorldTaskRunner
from mini_agent.evals.cua import (
    ExternalProcessResult,
    _python_from_console_script,
    resolve_cua_speed_run_executable,
    run_cua_speed_run_reference,
)
from mini_agent.integrations.cua_speed_run import (
    TEMPLATE_MAPPINGS,
    build_agent_argv,
    build_cua_speed_run_argv,
    build_profile_environment,
    export_submission,
    preflight,
    validate_submission,
)
from mini_agent.profiles import load_profile


def _chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _png(width: int = 2, height: int = 2, *, value: int = 0) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([0]) + bytes([value, value, value]) * width
    pixels = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", pixels) + _chunk(b"IEND", b"")


def _mini_agent_wheel(
    root: Path,
    *,
    requires_dist: str | None = None,
    installable: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / "mini_agent-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mini_agent/__init__.py", "__version__ = '0.3.0'\n")
        archive.writestr(
            "mini_agent-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: mini-agent\nVersion: 0.3.0\n"
            + (f"Requires-Dist: {requires_dist}\n" if requires_dist else ""),
        )
        if installable:
            archive.writestr(
                "mini_agent-0.3.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nGenerator: mini-agent-tests\n"
                "Root-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr("mini_agent-0.3.0.dist-info/RECORD", "")
    return wheel


class GatewayClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_routes_payload_png_and_operation_safe_retries(self) -> None:
        calls: list[tuple[str, str, Any]] = []
        attempts = {"observe": 0, "step": 0, "done": 0}
        pauses: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            operation = request.url.path.rsplit("/", 1)[-1]
            attempts[operation] += 1
            body = json.loads(request.content) if request.content else None
            calls.append((request.method, request.url.path, body))
            if operation == "observe" and attempts[operation] == 1:
                raise httpx.ConnectError("tunnel blip", request=request)
            if operation == "step" and attempts[operation] == 1:
                raise httpx.ConnectTimeout("not connected", request=request)
            if operation == "done" and attempts[operation] == 1:
                raise httpx.ConnectError("tunnel blip", request=request)
            if operation == "observe":
                return httpx.Response(
                    200,
                    json={"png_b64": base64.b64encode(_png(3, 4)).decode(), "meta": {"screen": "main"}},
                )
            return httpx.Response(200, json={"ok": True})

        async def sleep(seconds: float) -> None:
            pauses.append(seconds)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as raw_client:
            client = CUASpeedRunClient(
                "https://gateway.invalid/run-token",
                client=raw_client,
                sleep=sleep,
            )
            observation = await client.observe()
            self.assertEqual(observation.meta["width"], 3)
            await client.step([{"mouse": {"left_click": [1, 2]}}])
            await client.done()

        self.assertEqual(attempts, {"observe": 2, "step": 2, "done": 2})
        self.assertEqual(pauses, [1.0, 1.0, 1.0])
        self.assertIn(("GET", "/run-token/observe", None), calls)
        self.assertIn(
            (
                "POST",
                "/run-token/step",
                {"actions": [{"mouse": {"left_click": [1, 2]}}]},
            ),
            calls,
        )
        self.assertIn(("POST", "/run-token/done", None), calls)

    async def test_step_read_failure_is_never_retried_and_png_is_validated(self) -> None:
        step_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal step_calls
            if request.url.path.endswith("/step"):
                step_calls += 1
                raise httpx.ReadTimeout("ambiguous delivery", request=request)
            return httpx.Response(
                200, json={"png_b64": base64.b64encode(b"not png").decode(), "meta": {}}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as raw_client:
            client = CUASpeedRunClient("https://gateway.invalid/token", client=raw_client)
            with self.assertRaises(httpx.ReadTimeout):
                await client.step([{"action": "screenshot"}])
            with self.assertRaises(ProtocolError):
                await client.observe()
        self.assertEqual(step_calls, 1)


class ActionContractTests(unittest.TestCase):
    def test_png_validation_checks_crc_and_dimensions(self) -> None:
        png = _png(11, 7)
        self.assertEqual(validate_png(png), (11, 7))
        corrupt = bytearray(png)
        corrupt[-5] ^= 1
        with self.assertRaises(ProtocolError):
            validate_png(bytes(corrupt))

    def test_gym_validation_is_strict_and_terminal_is_explicit(self) -> None:
        self.assertEqual(
            validate_gym_action({"action": "wait", "time": 1.5}),
            {"action": "wait", "time": 1.5},
        )
        self.assertEqual(
            validate_gym_action({"action": "infeasible"}, allow_terminal=True),
            {"action": "fail"},
        )
        invalid = (
            {"mouse": {"left_click": [1, 2], "move": [2, 3]}},
            {"keyboard": {"text": "x"}, "mouse": {"left_click": [1, 2]}},
            {"action": "reset"},
            {"mouse": {"buttons": {"left_down": False}}},
            {"wait": float("nan")},
        )
        for action in invalid:
            with self.assertRaises(ProtocolError):
                validate_gym_action(action)

    def test_provider_action_translation_matches_gym_schema(self) -> None:
        openai = translate_openai_action(
            {"type": "drag", "path": [{"x": -5, "y": 2}, {"x": 500, "y": 300}]},
            100,
            80,
        )
        self.assertEqual(openai[0]["mouse"]["left_click_drag"], [[0, 2], [99, 79]])
        anthropic, cursor = translate_anthropic_action(
            {"action": "left_click", "coordinate": [640, 360]},
            (1920, 1080),
            (1280, 720),
        )
        self.assertEqual(anthropic, [{"mouse": {"left_click": [960, 540]}}])
        self.assertEqual(cursor, [960, 540])

    def test_osworld_script_allows_normal_actions_and_rejects_escape_apis(self) -> None:
        script = "import pyautogui\nfor x in [1, 2]:\n    pyautogui.click(x, x)"
        self.assertEqual(validate_osworld_script(script), script)
        for unsafe in (
            "import os",
            "open('/tmp/x', 'w')",
            "eval('1+1')",
            "pyautogui.os.system('id')",
            "getattr(object, '__subclasses__')()",
        ):
            with self.assertRaises(ProtocolError):
                validate_osworld_script(unsafe)


class _LifecycleClient:
    def __init__(
        self,
        *,
        fail_done: bool = False,
        observation: ComputerObservation | None = None,
    ) -> None:
        self.done_calls = 0
        self.close_calls = 0
        self.actions: list[list[dict[str, Any]]] = []
        self.fail_done = fail_done
        self.observation = observation or ComputerObservation(
            _png(), {"width": 2, "height": 2}
        )

    async def observe(self) -> ComputerObservation:
        return self.observation

    async def step(self, actions: list[dict[str, Any]]) -> Mapping[str, Any]:
        self.actions.append(actions)
        return {"ok": True}

    async def done(self) -> None:
        self.done_calls += 1
        if self.fail_done:
            raise RuntimeError("done failed")

    async def close(self) -> None:
        self.close_calls += 1


class EnvironmentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_guarantees_done_and_still_closes_after_done_failure(self) -> None:
        client = _LifecycleClient()
        environment = CUAEnvironment(client)
        await environment.close()
        await environment.close()
        self.assertEqual((client.done_calls, client.close_calls), (1, 1))

        failing = _LifecycleClient(fail_done=True)
        environment = CUAEnvironment(failing)
        with self.assertRaisesRegex(RuntimeError, "done failed"):
            await environment.close()
        self.assertEqual((failing.done_calls, failing.close_calls), (1, 1))

    async def test_native_openai_environment_translates_before_step(self) -> None:
        client = _LifecycleClient()
        environment = CUAEnvironment(client, protocol="openai")
        await environment.initial_observation()
        self.assertEqual(environment.tools()[0].kind, "openai_computer")
        execution = await environment.execute(
            ToolCall(
                "native",
                "computer",
                {"actions": [{"type": "click", "x": 50, "y": 50, "button": "left"}]},
                kind="openai_computer",
                raw={
                    "pending_safety_checks": [
                        {"id": "check-1", "code": "external_side_effect"}
                    ]
                },
            )
        )
        self.assertEqual(client.actions, [[{"mouse": {"left_click": [1, 1]}}]])
        self.assertEqual(
            execution.native_output["acknowledged_safety_checks"],
            [{"id": "check-1", "code": "external_side_effect"}],
        )
        await environment.close()

    async def test_profile_policy_applies_coordinates_resize_and_rejects_inert_fields(self) -> None:
        qwen = load_profile("cua", "qwen3vl")
        qwen_client = _LifecycleClient(
            observation=ComputerObservation(_png(101, 51), {"width": 101, "height": 51})
        )
        qwen_environment = CUAEnvironment.from_policy(
            qwen_client,
            benchmark=qwen.benchmark,
            observation=qwen.observation,
            history=qwen.history,
            tools=qwen.tools,
            response_parser=qwen.response_parser,
            provider=qwen.provider,
        )
        await qwen_environment.initial_observation()
        await qwen_environment.execute(
            ToolCall(
                "normalized",
                "computer",
                {"actions": [{"mouse": {"left_click": [1000, 500]}}]},
            )
        )
        self.assertEqual(
            qwen_client.actions,
            [[{"mouse": {"left_click": [100, 25]}}]],
        )
        await qwen_environment.close()

        resized_client = _LifecycleClient(
            observation=ComputerObservation(_png(4, 2), {"width": 4, "height": 2})
        )
        resized = CUAEnvironment.from_policy(
            resized_client,
            benchmark={"name": "cua_speed_run", "tool_protocol": "anthropic"},
            observation={
                "coordinate_mode": "scaled_pixels",
                "screenshot_detail": "original",
                "screenshot_max_width": 2,
                "screenshot_max_height": 2,
            },
            history={"mode": "linear"},
            tools=("computer",),
            response_parser="provider_tool_calls",
            provider="anthropic-messages",
        )
        initial = await resized.initial_observation()
        encoded = initial.image_data_url.split(",", 1)[1]
        self.assertEqual(validate_png(base64.b64decode(encoded)), (2, 1))
        await resized.close()

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            CUAEnvironment.from_policy(
                _LifecycleClient(),
                benchmark={"name": "cua_speed_run"},
                observation={},
                history={"mode": "linear", "magic_policy": True},
                tools=("computer",),
                response_parser="provider_tool_calls",
                provider="openai-compatible-chat",
            )


class SubmissionAndCatalogTests(unittest.TestCase):
    def test_catalog_partition_profiles_and_doctor_preflight(self) -> None:
        modes = json.loads(
            (Path(__file__).parent / "fixtures" / "cua" / "template_modes.json").read_text()
        )
        observed = {
            mode: sorted(item.name for item in TEMPLATE_MAPPINGS if item.mode == mode)
            for mode in modes
        }
        self.assertEqual(observed, modes)
        self.assertEqual(len(TEMPLATE_MAPPINGS), 18)
        for name in modes["mini_agent_profile"]:
            profile = load_profile("cua", name)
            self.assertEqual(profile.fidelity, "profile")
            self.assertEqual(profile.source["revision"], "7230223cbc57df68331cad32889adf01f3601651")
            self.assertTrue(profile.fidelity_gaps)
        report = preflight()
        self.assertTrue(report["ok"])
        self.assertEqual(report["checks"][0]["detail"]["total"], 18)

    def test_export_is_exact_and_argv_never_uses_shell_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            exported = export_submission(
                submission,
                template="gpt54",
                model="gpt-5.4",
                provider="openai-responses",
                required_environment=["OPENAI_API_KEY"],
                mode="multi",
                max_agents=3,
                child_profiles=("gpt54",),
                wheel=_mini_agent_wheel(root),
            )
            self.assertEqual(sorted(path.name for path in submission.iterdir()), ["agent.py", "init.py"])
            init_script, agent_script = validate_submission(submission)
            self.assertEqual(set(exported.files), {"init.py", "agent.py"})
            self.assertEqual(len(exported.runtime_wheel_sha256), 64)
            argv = build_agent_argv(
                "python3",
                agent_script,
                "https://gateway.invalid/token",
                "task; touch /tmp/never",
            )
            self.assertEqual(argv[-1], "task; touch /tmp/never")
            self.assertNotIn("-c", argv)
            self.assertIn("OPENAI_API_KEY", init_script.read_text())
            self.assertIn("embedded wheel failed", init_script.read_text())
            self.assertIn("--no-index", init_script.read_text())
            self.assertEqual(exported.dependency_wheel_sha256, {})
            self.assertIn("MODE = 'multi'", agent_script.read_text())
            self.assertIn("MAX_AGENTS = 3", agent_script.read_text())
            with self.assertRaises(ValueError):
                export_submission(root / "external", template="codex_cli", model="x", provider="x")
            with self.assertRaisesRegex(ValueError, "unsupported MiniAgent provider"):
                export_submission(
                    root / "invalid-provider",
                    template="gpt54",
                    model="gpt-5.4",
                    provider="nonsense",
                    wheel=_mini_agent_wheel(root),
                )
            with self.assertRaisesRegex(ValueError, "incompatible with profile"):
                export_submission(
                    root / "wrong-provider",
                    template="gpt54",
                    model="gpt-5.4",
                    provider="anthropic-messages",
                    wheel=_mini_agent_wheel(root),
                )
            with self.assertRaisesRegex(ValueError, "complete --dependency-wheel"):
                export_submission(
                    root / "missing-wheelhouse",
                    template="gpt54",
                    model="gpt-5.4",
                    provider="openai-responses",
                    wheel=_mini_agent_wheel(
                        root / "dependent", requires_dist="httpx==0.28.1"
                    ),
                )

            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "manifest.yaml").write_text("tasks: []\n", encoding="utf-8")
            eval_argv = build_cua_speed_run_argv(
                "cua-speedrun",
                submission=submission,
                benchmark=benchmark,
                output_root=root / "runs",
                task_ids=["task-1"],
            )
            self.assertEqual(eval_argv[:2], ("cua-speedrun", "run"))
            self.assertIn("--task", eval_argv)

    def test_exported_init_bootstraps_offline_and_rejects_invalid_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "submission"
            export_submission(
                submission,
                template="gpt54",
                model="gpt-5.4",
                provider="openai-responses",
                wheel=_mini_agent_wheel(root / "valid"),
            )
            virtualenv = root / "venv"
            subprocess.run(
                (sys.executable, "-m", "venv", str(virtualenv)),
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            python = virtualenv / "bin" / "python"
            environment = {
                "HOME": str(root / "home"),
                "OPENAI_API_KEY": "offline-test",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PIP_NO_INDEX": "1",
            }
            initialized = subprocess.run(
                (str(python), str(submission / "init.py")),
                cwd=submission,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr)
            self.assertIn("mini-agent init ready", initialized.stdout)
            installed = subprocess.run(
                (str(python), "-I", "-c", "import mini_agent; print(mini_agent.__version__)"),
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(installed.stdout.strip(), "0.3.0")

            with self.assertRaisesRegex(ValueError, "offline wheel bundle is not installable"):
                export_submission(
                    root / "invalid-submission",
                    template="gpt54",
                    model="gpt-5.4",
                    provider="openai-responses",
                    wheel=_mini_agent_wheel(root / "invalid", installable=False),
                )

    def test_cua_runner_executable_must_belong_to_the_verified_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            executable = checkout / ".venv" / "bin" / "cua-speedrun"
            executable.parent.mkdir(parents=True)
            interpreter = executable.parent / "python"
            interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
            interpreter.chmod(0o755)
            executable.write_text(f"#!{interpreter}\n", encoding="utf-8")
            executable.chmod(0o755)
            self.assertEqual(
                resolve_cua_speed_run_executable(checkout, executable),
                executable.resolve(),
            )
            self.assertEqual(
                _python_from_console_script(checkout, executable),
                interpreter.resolve(),
            )
            system_python = root / "system-python"
            system_python.write_text("#!/bin/sh\n", encoding="utf-8")
            system_python.chmod(0o755)
            interpreter.unlink()
            interpreter.symlink_to(system_python)
            self.assertEqual(
                _python_from_console_script(checkout, executable),
                interpreter.parent.resolve() / interpreter.name,
            )
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkout-local interpreter"):
                _python_from_console_script(checkout, executable)
            outside = root / "outside"
            outside.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            outside.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "verified checkout"):
                resolve_cua_speed_run_executable(checkout, outside)


class ReferenceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_bypasses_console_script_and_pins_source_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            interpreter = checkout / ".venv" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
            interpreter.chmod(0o755)
            executable = interpreter.parent / "cua-speedrun"
            executable.write_text(f"#!{interpreter}\nmalicious body\n", encoding="utf-8")
            executable.chmod(0o755)
            source_cli = checkout / "src" / "cua_speedrun" / "cli.py"
            source_cli.parent.mkdir(parents=True)
            source_cli.write_text("# pinned source\n", encoding="utf-8")
            submission = root / "submission"
            submission.mkdir()
            (submission / "agent.py").write_text("# agent\n", encoding="utf-8")
            (submission / "init.py").write_text("# init\n", encoding="utf-8")
            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "manifest.yaml").write_text("tasks: []\n", encoding="utf-8")
            completed = ExternalProcessResult((), 0, "ok", "")
            with patch(
                "mini_agent.evals.cua.verify_cua_speed_run_checkout",
                new=AsyncMock(return_value="revision"),
            ), patch(
                "mini_agent.evals.cua._run_external",
                new=AsyncMock(return_value=completed),
            ) as execute:
                result = await run_cua_speed_run_reference(
                    source_root=checkout,
                    executable=executable,
                    submission=submission,
                    benchmark=benchmark,
                    output_root=root / "runs",
                    environment={"PYTHONPATH": "existing"},
                )
            self.assertIs(result, completed)
            argv = execute.await_args.args[0]
            self.assertEqual(
                argv[:4],
                (str(interpreter.parent.resolve() / interpreter.name), "-m", "cua_speedrun.cli", "run"),
            )
            self.assertNotIn(str(executable), argv)
            selected_environment = execute.await_args.kwargs["environment"]
            self.assertEqual(
                selected_environment["PYTHONPATH"].split(os.pathsep)[:2],
                [
                    str(checkout.resolve() / "src"),
                    str(checkout.resolve() / "third_party" / "gym-anything" / "src"),
                ],
            )


class ProfileExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_claimed_profile_changes_executed_prompt_and_tool_policy(self) -> None:
        modes = json.loads(
            (Path(__file__).parent / "fixtures" / "cua" / "template_modes.json").read_text()
        )
        observed_prompts: set[str] = set()
        for name in modes["mini_agent_profile"]:
            profile = load_profile("cua", name)
            self.assertEqual(profile.response_parser, "provider_tool_calls")
            client = _LifecycleClient()
            environment = build_profile_environment(profile, client)
            model = ScriptedModel([ModelResponse(text=f"{name} complete")])
            result = await MiniAgent(
                model=model,
                environment=environment,
                system_prompt=profile.system_prompt,
                max_steps=1,
            ).run("task")
            self.assertEqual(result.answer, f"{name} complete")
            self.assertEqual(model.queries[0][0][0].content, profile.system_prompt)
            expected_kind = {
                "openai": "openai_computer",
                "anthropic": "anthropic_computer_20251124",
            }.get(profile.benchmark.get("tool_protocol"), "function")
            self.assertEqual(model.queries[0][1][0].kind, expected_kind)
            observed_prompts.add(profile.system_prompt)
            await environment.close()
        self.assertEqual(len(observed_prompts), len(modes["mini_agent_profile"]))


class OSWorldRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_keeps_evaluator_outside_agent_and_writes_artifacts(self) -> None:
        events: list[str] = []

        class Controller:
            def start_recording(self) -> None:
                events.append("start_recording")

            def end_recording(self, path: str) -> None:
                events.append("end_recording")
                Path(path).write_bytes(b"video")

        class Desktop:
            def __init__(self) -> None:
                self.controller = Controller()
                self.counter = 0

            def reset(self, *, task_config: Mapping[str, Any]) -> None:
                self.task_config = dict(task_config)
                events.append("reset")

            def _get_obs(self) -> Mapping[str, Any]:
                return {"screenshot": _png(value=self.counter)}

            def step(self, action: str) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
                events.append("step")
                self.counter += 1
                return self._get_obs(), 0.0, False, {"action": action}

            def evaluate(self) -> float:
                events.append("evaluate")
                return 0.75

            def close(self) -> None:
                events.append("close")

        desktop = Desktop()
        model = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    tool_calls=(
                        ToolCall(
                            "computer-1",
                            "computer",
                            {"actions": [{"script": "pyautogui.click(1, 1)"}]},
                        ),
                    ),
                ),
                ModelResponse(text="done"),
            ]
        )

        def factory(environment: Any) -> MiniAgent:
            self.assertFalse(hasattr(environment, "evaluate"))
            self.assertFalse(hasattr(environment.client, "environment"))
            return MiniAgent(model=model, environment=environment, max_steps=2)

        async def no_sleep(_: float) -> None:
            return None

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task"
            result = await OSWorldTaskRunner(
                desktop,
                ready_wait_seconds=0,
                settle_wait_seconds=0,
                sleep=no_sleep,
            ).run(
                task_config={"id": "task-1", "instruction": "click", "evaluator": {}},
                agent_factory=factory,
                output_directory=output,
            )
            self.assertEqual(result.score, 0.75)
            self.assertTrue((output / "step_0000.png").is_file())
            self.assertTrue((output / "step_0001.png").is_file())
            self.assertTrue((output / "traj.jsonl").is_file())
            self.assertEqual((output / "result.txt").read_text(), "0.75\n")
            result_json = json.loads((output / "result.json").read_text())
            self.assertFalse(result_json["verifier_exposed_to_agent"])
        self.assertLess(events.index("step"), events.index("evaluate"))
        self.assertEqual(events[-1], "close")

    async def test_cancellation_skips_evaluation_and_still_closes(self) -> None:
        events: list[str] = []
        started = asyncio.Event()

        class Desktop:
            controller = None

            def reset(self, *, task_config: Mapping[str, Any]) -> None:
                events.append("reset")

            def _get_obs(self) -> Mapping[str, Any]:
                return {"screenshot": _png()}

            def step(self, action: str) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
                raise AssertionError(f"unexpected action: {action}")

            def evaluate(self) -> float:
                events.append("evaluate")
                return 1.0

            def close(self) -> None:
                events.append("close")

        class BlockingModel:
            async def query(self, messages: Any, tools: Any) -> ModelResponse:
                started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task"
            running = asyncio.create_task(
                OSWorldTaskRunner(
                    Desktop(), ready_wait_seconds=0, settle_wait_seconds=0
                ).run(
                    task_config={"id": "task-1", "instruction": "wait"},
                    agent_factory=lambda environment: MiniAgent(
                        model=BlockingModel(), environment=environment
                    ),
                    output_directory=output,
                )
            )
            await started.wait()
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            lifecycle = json.loads((output / "lifecycle.json").read_text())
            self.assertEqual(lifecycle["status"], "cancelled")
        self.assertNotIn("evaluate", events)
        self.assertEqual(events[-1], "close")


class OSWorldClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_result_stops_remaining_batch(self) -> None:
        class Desktop:
            def __init__(self) -> None:
                self.actions: list[str] = []

            def step(self, action: str) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
                self.actions.append(action)
                return {"screenshot": _png()}, 0.0, action == "FAIL", {}

        desktop = Desktop()
        client = OSWorldClient(desktop, {"screenshot": _png()}, owns_environment=False)
        result = await client.step(
            [
                {"action": "fail"},
                {"script": "pyautogui.click(1, 1)"},
            ]
        )
        self.assertEqual(desktop.actions, ["FAIL"])
        self.assertTrue(result["done"])


if __name__ == "__main__":
    unittest.main()
