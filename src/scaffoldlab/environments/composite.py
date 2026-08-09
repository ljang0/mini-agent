from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import ToolEnvironment, ToolExecution


class CompositeEnvironment(ToolEnvironment):
    """Expose several environments in one agent session (for example SWE + CUA)."""

    def __init__(self, environments: Sequence[ToolEnvironment]) -> None:
        if not environments:
            raise ValueError("a composite environment requires at least one component")
        self.environments = tuple(environments)
        self._routing: dict[str, ToolEnvironment] = {}

    def tools(self, provider_family: str) -> Sequence[ToolDefinition]:
        routing: dict[str, ToolEnvironment] = {}
        definitions: list[ToolDefinition] = []
        for environment in self.environments:
            for definition in environment.tools(provider_family):
                if definition.name in routing:
                    raise ValueError(
                        f"duplicate composite tool name {definition.name!r}"
                    )
                routing[definition.name] = environment
                definitions.append(definition)
        self._routing = routing
        return definitions

    async def execute(self, call: ToolCall) -> ToolExecution:
        environment = self._routing.get(call.name)
        if environment is None:
            raise ProtocolError(f"unknown composite tool {call.name!r}")
        return await environment.execute(call)

    async def summary(self) -> Mapping[str, Any]:
        return {
            "type": "composite",
            "components": [dict(await item.summary()) for item in self.environments],
        }

    async def close(self) -> None:
        first_error: BaseException | None = None
        for environment in reversed(self.environments):
            try:
                await environment.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
