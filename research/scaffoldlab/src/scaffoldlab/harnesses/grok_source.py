from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..environments.swe import SWEPatchPayload
from ..runtime import RunContext
from ..types import ModelRequest, Task
from .base import Harness


class GrokBuildSourceHarness(Harness):
    """Outer lifecycle for the pinned public Grok Build Rust runtime."""

    name = "grok_build_source"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        prompt = task.prompt
        if task.context:
            prompt += f"\n\n<context>\n{task.context}\n</context>"
        response = await context.call(
            ModelRequest(
                agent_id="/grok-build-source/root",
                role="grok_build_source_session",
                prompt=prompt,
                metadata={"external_session_tree": True},
            )
        )
        result = response.raw if isinstance(response.raw, dict) else {}
        source = result.get("_scaffoldlab_source")
        source = dict(source) if isinstance(source, Mapping) else {}
        source_execution_evidence = source.get("source_archive_verified") is True and (
            isinstance(source.get("executable_sha256"), str)
            and len(source["executable_sha256"]) == 64
        )
        official_public_pin_verified = (
            source.get("official_public_pin_verified") is True
        )
        patch_payload = result.get("_scaffoldlab_swe_patch")
        model_usage = result.get("modelUsage")
        underlying_calls = 0
        if isinstance(model_usage, dict):
            for value in model_usage.values():
                if not isinstance(value, dict):
                    continue
                calls = value.get("modelCalls")
                if (
                    isinstance(calls, int)
                    and not isinstance(calls, bool)
                    and calls >= 0
                ):
                    underlying_calls += calls
        return response.text, {
            "released_runtime_adapter": source_execution_evidence,
            "upstream_source_executed": source_execution_evidence,
            "source_entrypoint": (
                "private git archive/Cargo build/xai-grok-pager headless JSON"
                if source_execution_evidence
                else None
            ),
            "external_session_tree": True,
            "native_subagents_enabled": True,
            "underlying_model_calls_observed": underlying_calls > 0,
            "underlying_model_calls": underlying_calls,
            "full_tree_usage_verified": False,
            "usage_scope": (
                "terminal JSON totals are a lower bound; compaction, side-model, "
                "unfinished, and some nested calls may be absent"
            ),
            "usage_is_incomplete": True,
            "cost_is_partial": True,
            "num_main_agent_turns": result.get("num_turns"),
            "session_id_present": isinstance(result.get("sessionId"), str),
            "workspace": result.get("_scaffoldlab_workspace"),
            "source": source,
            "swe_patch_exported": isinstance(patch_payload, SWEPatchPayload),
            "swe_patch_applied_or_scored": False,
            "patch_artifact": patch_payload,
            "fidelity": (
                "pinned_public_source_with_recorded_local_build"
                if source_execution_evidence and official_public_pin_verified
                else "custom_source_runtime_boundary"
            ),
            "exactness_scope": (
                "public Grok Build source and headless JSON protocol only; local "
                "Git/Cargo/rustc/binary identities are recorded, but the source pin "
                "does not pin the compiler/OS, hosted model snapshot or scheduler, "
                "encrypted service state, or complete whole-tree billing"
            ),
            "source_or_protocol_pin_verified": official_public_pin_verified,
            "executable_hash_pin_verified": (
                source.get("executable_hash_pin_verified") is True
            ),
            "bit_reproducible_runtime_verified": False,
            "hosted_multi_agent_parity_claimed": False,
            "flagship_system_card_parity_claimed": False,
        }
