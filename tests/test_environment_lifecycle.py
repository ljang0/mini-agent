import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from scaffoldlab.environments.base import (
    EnvironmentFactory,
    EnvironmentScope,
    ToolEnvironment,
    ToolExecution,
)
from scaffoldlab.environments.browser import PlaywrightBrowserDriver
from scaffoldlab.environments.swe import PersistentBash
from scaffoldlab.harnesses import SingleAgentHarness
from scaffoldlab.runtime import ScriptedBackend
from scaffoldlab.types import (
    BudgetLimits,
    RunFailed,
    Task,
    ToolCall,
    ToolDefinition,
)


class _NoToolsEnvironment(ToolEnvironment):
    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        return ()

    async def execute(self, call: ToolCall) -> ToolExecution:
        raise AssertionError("no tools are exposed")


class _FaultingScope(EnvironmentScope):
    def __init__(
        self,
        *,
        summary_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.environment = _NoToolsEnvironment()
        self.summary_error = summary_error
        self.close_error = close_error
        self.closed = False

    async def get(self, agent_id: str) -> ToolEnvironment:
        return self.environment

    async def summary(self) -> Mapping[str, Any]:
        if self.summary_error is not None:
            raise self.summary_error
        return {"scope": "test"}

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _StaticFactory(EnvironmentFactory):
    def __init__(
        self, scope: _FaultingScope, *, cancel_after_begin: bool = False
    ) -> None:
        self.scope = scope
        self.cancel_after_begin = cancel_after_begin

    async def begin(self, task: Task) -> EnvironmentScope:
        if self.cancel_after_begin:
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
        return self.scope

    def provenance(self) -> Mapping[str, Any]:
        return {"factory": "test"}


class _FailingFactory(EnvironmentFactory):
    async def begin(self, task: Task) -> EnvironmentScope:
        raise RuntimeError("begin exploded")

    def provenance(self) -> Mapping[str, Any]:
        return {"factory": "failing-test"}


class EnvironmentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_environment_begin_failure_is_structured(self) -> None:
        with self.assertRaises(RunFailed) as caught:
            await SingleAgentHarness().run(
                Task("task", "question"),
                ScriptedBackend({"/root": ["unused"]}),
                BudgetLimits(wall_time_seconds=2),
                environment_factory=_FailingFactory(),
            )
        self.assertEqual(caught.exception.cause_type, "RuntimeError")
        self.assertEqual(
            [event.event for event in caught.exception.trace],
            ["run_started", "run_failed"],
        )

    async def test_summary_failure_still_closes_and_becomes_run_failed(self) -> None:
        scope = _FaultingScope(summary_error=RuntimeError("summary exploded"))
        with self.assertRaises(RunFailed) as caught:
            await SingleAgentHarness().run(
                Task("task", "question"),
                ScriptedBackend({"/root": ["answer"]}),
                BudgetLimits(wall_time_seconds=2),
                environment_factory=_StaticFactory(scope),
            )
        self.assertTrue(scope.closed)
        self.assertEqual(caught.exception.cause_type, "RuntimeError")
        events = [event.event for event in caught.exception.trace]
        self.assertIn("environment_summary_failed", events)
        self.assertIn("environment_closed", events)
        self.assertEqual(events[-1], "run_failed")

    async def test_cancellation_is_preserved_when_environment_close_fails(self) -> None:
        scope = _FaultingScope(close_error=RuntimeError("close exploded"))
        task = asyncio.create_task(
            SingleAgentHarness().run(
                Task("task", "question"),
                ScriptedBackend({"/root": ["unused"]}),
                BudgetLimits(wall_time_seconds=2),
                environment_factory=_StaticFactory(scope, cancel_after_begin=True),
            )
        )
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(scope.closed)
        self.assertTrue(task.cancelled())

    async def test_playwright_close_attempts_every_resource(self) -> None:
        calls: list[str] = []

        class _Resource:
            def __init__(self, name: str, *, fail: bool = False) -> None:
                self.name = name
                self.fail = fail

            async def close(self) -> None:
                calls.append(self.name)
                if self.fail:
                    raise RuntimeError(f"{self.name} failed")

            async def stop(self) -> None:
                calls.append(self.name)

        driver = PlaywrightBrowserDriver()
        driver._context = _Resource("context", fail=True)
        driver._browser = _Resource("browser")
        driver._playwright = _Resource("playwright")
        driver._page = object()
        with self.assertRaisesRegex(RuntimeError, "context failed"):
            await driver.close()
        self.assertEqual(calls, ["context", "browser", "playwright"])
        self.assertIsNone(driver._context)
        self.assertIsNone(driver._browser)
        self.assertIsNone(driver._playwright)
        self.assertIsNone(driver._page)

    async def test_persistent_bash_enforces_exact_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bash = PersistentBash(
                Path(directory), timeout_seconds=2, max_output_bytes=2
            )
            with self.assertRaisesRegex(RuntimeError, "exceeded max_output_bytes"):
                await bash.run("printf 123")
            self.assertIsNone(bash._process)


if __name__ == "__main__":
    unittest.main()
