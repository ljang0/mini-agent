from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, ProtocolError, Task
from .base import Harness, require_int_at_least


class ClaudeCodeAgentTeamsDistributionHarness(Harness):
    """Ask the pinned Claude Code distribution to form its native agent team."""

    name = "claude_code_agent_teams_distribution"

    def __init__(self, *, team_size: int = 3) -> None:
        self.team_size = require_int_at_least(team_size, "team_size")

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = (
            "Use Claude Code's native Agent Teams feature for this task. "
            f"Spawn exactly {self.team_size} named teammates with independent "
            "contexts; do not substitute ordinary subagents. Give them independent "
            "work, use the shared task list and team messaging where useful, wait "
            "for every teammate, and synthesize the final answer. Use the current "
            "implicit session-derived team: spawn named teammates directly with the "
            "Agent tool and let Claude Code perform its automatic session cleanup.\n\n"
            f"{task.prompt}"
        )
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/claude-code-team/lead",
                role="claude_code_agent_teams_session",
                prompt=prompt,
                metadata={
                    "external_session_tree": True,
                    "task_tools": False,
                    "requested_team_size": self.team_size,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        evidence = raw.get("team_evidence")
        if not isinstance(evidence, Mapping):
            raise ProtocolError("Claude Code response omitted Agent Teams evidence")
        distribution = raw.get("distribution")
        if not isinstance(distribution, Mapping) or (
            distribution.get("official_distribution_verified") is not True
        ):
            raise ProtocolError(
                "Claude Code response was not anchored to an audited official "
                "platform distribution"
            )
        workspace = raw.get("workspace")
        workspace_mapping = workspace if isinstance(workspace, Mapping) else {}
        count = evidence.get("named_teammate_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ProtocolError(
                "Claude Code response contained invalid Agent Teams evidence"
            )
        if count != self.team_size:
            raise ProtocolError(
                "Claude Code did not spawn the requested number of distinct named "
                f"teammates: expected {self.team_size}, observed {count}"
            )
        return response.text, {
            "released_runtime_adapter": True,
            "upstream_source_executed": False,
            "official_binary_distribution_executed": True,
            "external_session_tree": True,
            "native_agent_teams_enabled": True,
            "native_agent_teams_observed": bool(evidence.get("native_team_observed")),
            "requested_team_size": self.team_size,
            "observed_named_teammates": count,
            "observed_agent_tool_calls": evidence.get("agent_tool_calls"),
            "observed_send_message_calls": evidence.get("send_message_calls"),
            "removed_team_tool_calls_observed": (
                int(evidence.get("removed_team_create_calls", 0))
                + int(evidence.get("removed_team_delete_calls", 0))
            ),
            "deprecated_team_name_inputs_observed": evidence.get(
                "deprecated_team_name_inputs"
            ),
            "session_derived_team_name": evidence.get("session_derived_team_name"),
            "live_team_config_snapshot_count": evidence.get(
                "live_team_config_snapshot_count"
            ),
            "session_team_config_removed_after_exit": evidence.get(
                "session_team_config_removed_after_exit"
            ),
            "persisted_task_file_count": evidence.get("persisted_task_file_count"),
            "shared_budget_ledger": True,
            "outer_model_calls": 1,
            "underlying_model_calls_observed": False,
            "whole_tree_usage_reported_by_upstream": False,
            "whole_tree_usage_independently_verified": False,
            "usage_scope": (
                "terminal result is a lower bound; authoritative whole-team "
                "coverage is not documented"
            ),
            "usage_is_incomplete": True,
            "domain_tools_injected": False,
            "workspace": workspace,
            "swe_patch_exported": "patch" in workspace_mapping,
            "distribution": raw.get("distribution"),
            "fidelity": "audited_official_distribution_protocol",
            "exactness_scope": (
                "official executable and public CLI invocation; server-managed "
                "policy remains unobservable"
            ),
            "source_or_protocol_pin_verified": True,
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }
