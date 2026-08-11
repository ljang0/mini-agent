import asyncio
import json
import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from scaffoldlab.cli import _build_backend, _validate_compatibility, build_parser
from scaffoldlab.harnesses import (
    AnthropicManagedAgentsHarness,
    SingleAgentHarness,
)
from scaffoldlab.providers import AnthropicManagedAgentsBackend, ProviderError
from scaffoldlab.types import BudgetLimits, ModelRequest, Task, Usage


def _usage(
    *,
    input_tokens: int = 10,
    output_tokens: int = 4,
    cache_read: int = 3,
    cache_5m: int = 5,
    cache_1h: int = 1,
    cents: str = "187",
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_5m,
            "ephemeral_1h_input_tokens": cache_1h,
        },
        "list_cost": {"amount": cents, "currency": "USD"},
        "active_seconds": 1.5,
        "server_tool_use": {"web_search_requests": 1},
    }


def _events(*, answer: str = "final answer") -> list[dict[str, Any]]:
    return [
        {
            "id": "event-thread",
            "type": "session.thread_created",
            "thread": {"id": "thread-1"},
        },
        {
            "id": "event-span-1",
            "type": "span.model_request_end",
            "session_thread_id": "thread-1",
        },
        {
            "id": "event-span-2",
            "type": "span.model_request_end",
        },
        {
            "id": "event-draft",
            "type": "agent.message",
            "content": [{"type": "text", "text": "earlier draft"}],
        },
        {
            "id": "event-message",
            "type": "agent.message",
            "content": [{"type": "text", "text": answer}],
        },
        {
            "id": "event-child-message",
            "type": "agent.message",
            "session_thread_id": "thread-1",
            "content": [{"type": "text", "text": "child answer"}],
        },
        {
            "id": "event-idle",
            "type": "session.status_idle",
            "stop_reason": {"type": "end_turn"},
        },
    ]


def _agent_snapshot(*, version: int = 1, coordinator: bool = True) -> dict[str, Any]:
    return {
        "id": "agent-1",
        "version": version,
        "model": {
            "id": "claude-opus-5",
            "effort": {"type": "high"},
            "speed": "standard",
        },
        "system": "Coordinate the work.",
        "tools": [{"type": "agent_toolset_20260401"}],
        "skills": [],
        "mcp_servers": [],
        "multiagent": (
            {"type": "coordinator", "agents": [{"type": "self"}]}
            if coordinator
            else None
        ),
    }


class ManagedAgentsBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_create_poll_paginate_delete_and_usage_contract(self) -> None:
        calls: list[dict[str, Any]] = []
        final_events = _events()

        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "timeout_seconds": timeout_seconds,
                    "payload": payload,
                }
            )
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-1",
                    "status": "running",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(version=7),
                    "usage": _usage(input_tokens=2, output_tokens=1, cents="20"),
                }
            if method == "GET" and url.endswith("/sessions/session-1"):
                return {
                    "id": "session-1",
                    "status": "idle",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(version=7),
                    "usage": _usage(),
                }
            if method == "GET" and url.endswith("/events"):
                return {"data": final_events[:2], "next_page": "cursor /2"}
            if method == "GET" and url.endswith("/events?page=cursor%20%2F2"):
                return {"data": final_events[2:], "next_page": None}
            if method == "DELETE" and url.endswith("/sessions/session-1"):
                return {}
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            agent_version=7,
            environment_id="environment-1",
            api_key="test-key",
            base_url="https://api.example/v1",
            poll_interval_seconds=0.01,
            budget_cents=2500,
            resources=(
                {
                    "type": "github_repository",
                    "url": "https://github.com/example/repo",
                    "authorization_token": "repo-secret",
                    "checkout": {"type": "commit", "sha": "abc123"},
                },
            ),
            vault_ids=("vault-1",),
            cleanup="delete",
        )
        reported: list[Usage] = []
        request = ModelRequest(
            agent_id="/anthropic-managed/coordinator",
            role="anthropic_managed_session",
            prompt="solve this",
            metadata={"anthropic_managed_agents": True},
            usage_reporter=reported.append,
        )

        with (
            patch("scaffoldlab.providers._request_json", new=request_json),
            patch("scaffoldlab.providers.asyncio.sleep", new=AsyncMock()),
        ):
            response = await backend.complete(request)

        self.assertEqual(response.text, "final answer")
        self.assertEqual(response.usage.input_tokens, 19)
        self.assertEqual(response.usage.output_tokens, 4)
        self.assertEqual(response.usage.cache_read_input_tokens, 3)
        self.assertEqual(response.usage.cache_write_input_tokens, 6)
        self.assertEqual(response.usage.cost_usd, 1.87)
        self.assertTrue(response.usage.complete)
        self.assertEqual(response.raw["events"]["pages"], 2)
        self.assertEqual(response.raw["coordinator_roster_size"], 1)
        self.assertRegex(response.raw["resolved_agent_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(reported), 2)
        self.assertFalse(reported[0].complete)
        self.assertTrue(reported[1].complete)

        create = calls[0]
        self.assertEqual(create["method"], "POST")
        self.assertEqual(create["url"], "https://api.example/v1/sessions")
        self.assertEqual(
            create["headers"],
            {
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "managed-agents-2026-04-01",
            },
        )
        self.assertEqual(
            create["payload"]["agent"],
            {"type": "agent", "id": "agent-1", "version": 7},
        )
        self.assertEqual(create["payload"]["environment_id"], "environment-1")
        self.assertEqual(
            create["payload"]["initial_events"],
            [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "solve this"}],
                }
            ],
        )
        self.assertEqual(
            create["payload"]["budget"],
            {
                "type": "limit",
                "max_list_cost": {"amount": "2500", "currency": "USD"},
            },
        )
        self.assertEqual(calls[-1]["method"], "DELETE")

        provenance_json = json.dumps(backend.provenance(), sort_keys=True)
        self.assertNotIn("repo-secret", provenance_json)
        self.assertIn("<configured>", provenance_json)
        self.assertEqual(backend.provenance()["transport"], "polling")

        memory_backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
            resources=({"type": "memory_store", "memory_store_id": "memory-1"},),
        )
        self.assertEqual(
            memory_backend._headers["anthropic-beta"],
            "managed-agents-2026-04-01",
        )
        self.assertNotIn("memory_beta_version", memory_backend.provenance())

    async def test_requires_action_fails_closed_and_archives_idle_session(self) -> None:
        calls: list[tuple[str, str]] = []

        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds, payload
            calls.append((method, url))
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-2",
                    "status": "idle",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(),
                    "usage": _usage(),
                }
            if method == "GET" and url.endswith("/events"):
                return {
                    "data": [
                        {"id": "tool-1", "type": "agent.custom_tool_use"},
                        {
                            "id": "idle-1",
                            "type": "session.status_idle",
                            "stop_reason": {"type": "requires_action"},
                        },
                    ]
                }
            if method == "POST" and url.endswith("/archive"):
                return {}
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
            cleanup="archive",
        )
        with patch("scaffoldlab.providers._request_json", new=request_json):
            with self.assertRaisesRegex(ProviderError, "requires_action") as caught:
                await backend.complete(
                    ModelRequest(
                        agent_id="/managed",
                        role="session",
                        prompt="question",
                        metadata={"anthropic_managed_agents": True},
                    )
                )

        failed_usage = caught.exception.usage
        assert failed_usage is not None
        self.assertEqual(failed_usage.cost_usd, 1.87)
        self.assertFalse(failed_usage.complete)
        self.assertEqual(
            calls[-1],
            (
                "POST",
                "https://api.anthropic.com/v1/sessions/session-2/archive",
            ),
        )

    async def test_dedicated_harness_records_full_tree_summary_in_shared_trace(
        self,
    ) -> None:
        captured_payloads: list[dict[str, Any]] = []

        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds
            if method == "POST" and url.endswith("/sessions"):
                captured_payloads.append(dict(payload))
                return {
                    "id": "session-3",
                    "status": "idle",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(version=9),
                    "usage": _usage(cents="125"),
                }
            if method == "GET" and url.endswith("/events"):
                return {"data": _events(answer="harness answer")}
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            agent_version=9,
            environment_id="environment-1",
            api_key="key",
        )
        with patch("scaffoldlab.providers._request_json", new=request_json):
            result = await AnthropicManagedAgentsHarness().run(
                Task("task-1", "question", context="grounding"),
                backend,
                BudgetLimits(
                    max_model_calls=1,
                    max_cost_usd=2.0,
                    wall_time_seconds=2,
                ),
            )

        self.assertEqual(result.answer, "harness answer")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.usage.cost_usd, 1.25)
        self.assertEqual(result.metadata["underlying_model_calls"], 2)
        self.assertEqual(result.metadata["thread_count_observed"], 1)
        self.assertEqual(result.metadata["resolved_agent_version"], 9)
        self.assertEqual(result.metadata["coordinator_roster_size"], 1)
        self.assertRegex(result.metadata["resolved_agent_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result.metadata["session_stop_reason"], "end_turn")
        self.assertEqual(
            [event.event for event in result.trace].count("managed_session_observed"),
            1,
        )
        prompt = captured_payloads[0]["initial_events"][0]["content"][0]["text"]
        self.assertIn("question", prompt)
        self.assertIn("grounding", prompt)

    async def test_backend_rejects_non_dedicated_and_local_loop_requests(self) -> None:
        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
        )
        with self.assertRaisesRegex(ProviderError, "dedicated harness"):
            await backend.complete(
                ModelRequest(agent_id="/root", role="solver", prompt="question")
            )
        with self.assertRaisesRegex(ProviderError, "system configuration"):
            await backend.complete(
                ModelRequest(
                    agent_id="/root",
                    role="solver",
                    prompt="question",
                    system="override",
                    metadata={"anthropic_managed_agents": True},
                )
            )

    async def test_session_requires_resolved_coordinator_snapshot(self) -> None:
        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds, payload
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-single",
                    "status": "idle",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(version=4, coordinator=False),
                    "usage": _usage(),
                }
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            agent_version=4,
            environment_id="environment-1",
            api_key="key",
        )
        with patch("scaffoldlab.providers._request_json", new=request_json):
            with self.assertRaisesRegex(ProviderError, "multiagent coordinator"):
                await backend.complete(
                    ModelRequest(
                        agent_id="/managed",
                        role="session",
                        prompt="question",
                        metadata={"anthropic_managed_agents": True},
                    )
                )

    async def test_cancellation_interrupts_running_remote_session(self) -> None:
        sleep_started = asyncio.Event()
        interrupt_calls: list[dict[str, Any]] = []

        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-running",
                    "status": "running",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(),
                    "usage": _usage(input_tokens=1, output_tokens=0, cents="1"),
                }
            if method == "POST" and url.endswith("/sessions/session-running/events"):
                interrupt_calls.append(dict(payload))
                return {"data": [{"type": "user.interrupt"}]}
            raise AssertionError(f"unexpected request: {method} {url}")

        async def blocked_sleep(delay: float) -> None:
            del delay
            sleep_started.set()
            await asyncio.Event().wait()

        reported: list[Usage] = []
        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
        )
        request = ModelRequest(
            agent_id="/managed",
            role="session",
            prompt="question",
            metadata={"anthropic_managed_agents": True},
            usage_reporter=reported.append,
        )
        with (
            patch("scaffoldlab.providers._request_json", new=request_json),
            patch("scaffoldlab.providers.asyncio.sleep", new=blocked_sleep),
        ):
            running = asyncio.create_task(backend.complete(request))
            await asyncio.wait_for(sleep_started.wait(), timeout=1)
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running

        self.assertEqual(
            interrupt_calls,
            [{"events": [{"type": "user.interrupt"}]}],
        )
        self.assertTrue(reported)
        self.assertTrue(all(not usage.complete for usage in reported))

    async def test_poll_timeout_interrupts_and_does_not_delete_running_session(
        self,
    ) -> None:
        calls: list[tuple[str, str, Any]] = []

        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds
            calls.append((method, url, payload))
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-timeout",
                    "status": "running",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(),
                    "usage": _usage(input_tokens=2, output_tokens=1, cents="25"),
                }
            if method == "POST" and url.endswith("/session-timeout/events"):
                return {"data": [{"type": "user.interrupt"}]}
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
            timeout_seconds=1,
            cleanup="delete",
        )
        with (
            patch("scaffoldlab.providers._request_json", new=request_json),
            patch("scaffoldlab.providers._monotonic", side_effect=[0.0, 2.0]),
        ):
            with self.assertRaisesRegex(ProviderError, "polling timed out") as caught:
                await backend.complete(
                    ModelRequest(
                        agent_id="/managed",
                        role="session",
                        prompt="question",
                        metadata={"anthropic_managed_agents": True},
                    )
                )

        timeout_usage = caught.exception.usage
        assert timeout_usage is not None
        self.assertFalse(timeout_usage.complete)
        self.assertEqual(calls[-1][0], "POST")
        self.assertTrue(calls[-1][1].endswith("/session-timeout/events"))
        self.assertEqual(calls[-1][2], {"events": [{"type": "user.interrupt"}]})
        self.assertFalse(any(method == "DELETE" for method, _, _ in calls))

    async def test_terminated_session_fails_closed_with_accounted_usage(self) -> None:
        async def request_json(
            method: str,
            url: str,
            *,
            headers: Any,
            timeout_seconds: float,
            payload: Any = None,
        ) -> dict[str, Any]:
            del headers, timeout_seconds, payload
            if method == "POST" and url.endswith("/sessions"):
                return {
                    "id": "session-terminated",
                    "status": "terminated",
                    "environment_id": "environment-1",
                    "agent": _agent_snapshot(),
                    "usage": _usage(cents="75"),
                }
            if method == "GET" and url.endswith("/events"):
                return {
                    "data": [
                        {
                            "id": "terminated-1",
                            "type": "session.status_terminated",
                        }
                    ]
                }
            raise AssertionError(f"unexpected request: {method} {url}")

        backend = AnthropicManagedAgentsBackend(
            agent_id="agent-1",
            environment_id="environment-1",
            api_key="key",
        )
        with patch("scaffoldlab.providers._request_json", new=request_json):
            with self.assertRaisesRegex(ProviderError, "status='terminated'") as caught:
                await backend.complete(
                    ModelRequest(
                        agent_id="/managed",
                        role="session",
                        prompt="question",
                        metadata={"anthropic_managed_agents": True},
                    )
                )

        terminated_usage = caught.exception.usage
        assert terminated_usage is not None
        self.assertEqual(terminated_usage.cost_usd, 0.75)
        self.assertFalse(terminated_usage.complete)


class ManagedAgentsCLITests(unittest.TestCase):
    def test_cli_requires_managed_agent_version(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--tasks",
                "tasks.jsonl",
                "--config",
                "config.json",
                "--output",
                "output",
                "--provider",
                "anthropic-managed-agents",
                "--managed-agent-id",
                "agent-1",
                "--managed-environment-id",
                "environment-1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "managed-agent-version"):
            _build_backend(args, {})

    def test_cli_builds_pinned_managed_backend_and_parses_resources(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--tasks",
                "tasks.jsonl",
                "--config",
                "config.json",
                "--output",
                "output",
                "--provider",
                "anthropic-managed-agents",
                "--managed-agent-id",
                "agent-1",
                "--managed-agent-version",
                "11",
                "--managed-environment-id",
                "environment-1",
                "--managed-budget-cents",
                "500",
                "--managed-resource-json",
                '{"type":"file","file_id":"file-1"}',
                "--managed-vault-id",
                "vault-1",
                "--managed-cleanup",
                "archive",
            ]
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
            backend = _build_backend(args, {})

        self.assertIsInstance(backend, AnthropicManagedAgentsBackend)
        self.assertEqual(backend.agent_version, 11)
        self.assertEqual(backend.budget_cents, 500)
        self.assertEqual(backend.resources[0]["file_id"], "file-1")
        self.assertEqual(backend.vault_ids, ("vault-1",))
        self.assertEqual(backend.cleanup, "archive")

    def test_compatibility_is_bidirectional_and_rejects_local_environment(self) -> None:
        harness = AnthropicManagedAgentsHarness()
        task = Task("task", "question")
        with self.assertRaisesRegex(ValueError, "requires"):
            _validate_compatibility(
                [harness], "anthropic-messages", Path("tasks.jsonl"), [task]
            )
        with self.assertRaisesRegex(ValueError, "owns its remote environment"):
            _validate_compatibility(
                [harness],
                "anthropic-managed-agents",
                Path("tasks.jsonl"),
                [task],
                environment_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "may only run"):
            _validate_compatibility(
                [SingleAgentHarness()],
                "anthropic-managed-agents",
                Path("tasks.jsonl"),
                [task],
            )

        warnings = _validate_compatibility(
            [harness],
            "anthropic-managed-agents",
            Path("tasks.jsonl"),
            [task],
        )
        self.assertTrue(any("remote agent topology" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
