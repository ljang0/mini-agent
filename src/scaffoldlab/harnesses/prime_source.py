from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..runtime import RunContext
from ..types import Task
from .prime_agent import PrimeAgentHarness


class PrimeAgentSourceHarness(PrimeAgentHarness):
    """One outer call into a caller-built bundle from pinned Prime Agent source."""

    name = "prime_agent_source"

    async def _execute(
        self, task: Task, context: RunContext
    ) -> Tuple[str, Mapping[str, Any]]:
        answer, inherited = await super()._execute(task, context)
        return answer, {
            **dict(inherited),
            "released_runtime_adapter": False,
            "caller_built_runtime_adapter": True,
            "source_checkout_adapter": True,
            "caller_built_source_bundle_entrypoint_executed": True,
            "source_build_performed_by_adapter": False,
            "dependency_install_performed_by_adapter": False,
            "whole_tree_usage_verified": False,
            "usage_scope": "assistant message_end events; whole tree unverified",
            "fidelity": "caller_built_runtime_study",
            "exactness_scope": (
                "private_copy_of_caller_built_bundle_with_revision_checked_source"
            ),
            "caller_worktree_bundle_executed_directly": False,
            "adversarial_source_content_attestation_verified": False,
            "source_or_protocol_pin_verified": False,
            "bit_reproducible_runtime_verified": False,
            "flagship_system_card_parity_claimed": False,
        }
