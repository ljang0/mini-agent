from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class CodexSourceHarness(Harness):
    """One pinned native Codex source session behind the shared run ledger."""

    name = "codex_source"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/codex-source/root",
                role="codex_source_session",
                prompt=prompt,
                metadata={
                    "external_session_tree": True,
                    "task_tools": False,
                },
            )
        )
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        source = raw.get("_scaffoldlab_source")
        source = dict(source) if isinstance(source, Mapping) else {}
        codex = raw.get("_scaffoldlab_codex")
        codex = dict(codex) if isinstance(codex, Mapping) else {}
        raw_spawn_calls = codex.get("spawn_agent_calls_observed")
        spawn_calls = (
            raw_spawn_calls
            if isinstance(raw_spawn_calls, int)
            and not isinstance(raw_spawn_calls, bool)
            and raw_spawn_calls >= 0
            else 0
        )
        raw_child_ids = codex.get("spawned_child_thread_ids_observed")
        child_ids = (
            sorted(
                {
                    thread_id
                    for thread_id in raw_child_ids
                    if isinstance(thread_id, str) and thread_id
                }
            )
            if isinstance(raw_child_ids, list)
            else []
        )
        raw_collab_tools = codex.get("completed_collaboration_tools")
        collab_tools = (
            [tool for tool in raw_collab_tools if isinstance(tool, str)]
            if isinstance(raw_collab_tools, list)
            else []
        )
        execution_observed = spawn_calls > 0 and bool(child_ids)
        source_execution_evidence = (
            isinstance(source.get("repository"), str)
            and bool(source.get("repository"))
            and isinstance(source.get("revision"), str)
            and bool(source.get("revision"))
            and codex.get("native_multi_agent_tools_enabled") is True
        )
        return response.text, {
            "released_runtime_adapter": source_execution_evidence,
            "upstream_source_executed": source_execution_evidence,
            "source_entrypoint": (
                "fresh CARGO_TARGET_DIR/release/codex exec"
                if source_execution_evidence
                else None
            ),
            "external_session_tree": True,
            "native_multi_agent_tools_enabled": (
                codex.get("native_multi_agent_tools_enabled") is True
            ),
            # Capability and observed execution are intentionally distinct. A Codex
            # session that never calls spawn_agent is a single-agent run.
            "spawn_agent_calls_observed": spawn_calls,
            "spawned_child_thread_ids_observed": child_ids,
            "completed_collaboration_tools": collab_tools,
            "multi_agent_execution_observed": execution_observed,
            "single_agent_execution": not execution_observed,
            "requested_multi_agent_version": codex.get("requested_multi_agent_version"),
            "effective_multi_agent_version": codex.get("effective_multi_agent_version"),
            "effective_multi_agent_version_verified": (
                codex.get("effective_multi_agent_version_verified") is True
            ),
            "multi_agent_version": codex.get("effective_multi_agent_version"),
            "max_subagents": codex.get("max_subagents"),
            "max_depth": codex.get("max_depth"),
            "configured_v1_max_depth": codex.get("configured_v1_max_depth"),
            "v2_total_concurrency_including_root": codex.get(
                "v2_total_concurrency_including_root"
            ),
            "v2_min_wait_timeout_ms": codex.get("v2_min_wait_timeout_ms"),
            "v2_max_wait_timeout_ms": codex.get("v2_max_wait_timeout_ms"),
            "v2_default_wait_timeout_ms": codex.get("v2_default_wait_timeout_ms"),
            "shared_budget_ledger": True,
            "outer_model_calls": 1,
            "underlying_model_calls_observed": False,
            "whole_tree_usage_reported_by_upstream": False,
            "whole_tree_usage_independently_verified": False,
            "usage_scope": raw.get("usage_scope"),
            "usage_is_incomplete": raw.get("usage_is_incomplete") is True,
            "cost_is_unknown": raw.get("cost_is_unknown") is True,
            "scaffoldlab_domain_tools_injected": False,
            "workspace": raw.get("_scaffoldlab_workspace"),
            "source": source,
            "fidelity": (
                "exact_pinned_public_source_runtime_boundary"
                if source_execution_evidence
                and source.get("exact_runtime_boundary_verified") is True
                else (
                    "pinned_public_source_with_recorded_local_build"
                    if source_execution_evidence
                    and source.get("official_public_pin_verified") is True
                    else "custom_source_runtime_boundary"
                )
            ),
            "exactness_scope": (
                "public Codex source and exec JSONL protocol; local Git/Cargo/rustc/"
                "binary hashes are recorded per trial, but remote model catalog, "
                "model snapshot, service routing, cloud/managed policy, and complete "
                "tree billing are not independently pinned"
            ),
            "source_or_protocol_pin_verified": (
                source.get("official_public_pin_verified") is True
            ),
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }
