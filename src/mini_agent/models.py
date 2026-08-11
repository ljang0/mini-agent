from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Protocol, Sequence

from .types import Message, ModelRequest, ModelResponse, ToolDefinition, ToolResult


class Model(Protocol):
    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse: ...


class BackendModel:
    """Make existing provider codecs implement the minimal linear-history API."""

    def __init__(
        self,
        backend: Any,
        *,
        agent_id: str = "/root",
        role: str = "solver",
        max_output_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.agent_id = agent_id
        self.role = role
        self.max_output_tokens = max_output_tokens
        self.metadata = dict(metadata or {})
        self.tool_family = str(getattr(backend, "tool_family", "generic"))
        self._continuation: Any = None

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        system = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        if self._continuation is None:
            prompt = "\n\n".join(
                message.content for message in messages if message.role == "user"
            )
            images = tuple(
                message.image_data_url
                for message in messages
                if message.role == "user" and message.image_data_url is not None
            )
            tool_results: tuple[ToolResult, ...] = ()
        else:
            prompt = ""
            images = ()
            tool_results = tuple(
                result
                for message in messages
                if message.role == "tool"
                for result in message.tool_results
            )
            if not tool_results:
                raise ValueError("continued provider query requires tool results")
            # Only the results after the latest assistant action belong to this turn.
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].role == "assistant":
                    tool_results = tuple(
                        result
                        for message in messages[index + 1 :]
                        if message.role == "tool"
                        for result in message.tool_results
                    )
                    break
        response = await self.backend.complete(
            ModelRequest(
                agent_id=self.agent_id,
                role=self.role,
                prompt=prompt,
                system=system,
                max_output_tokens=self.max_output_tokens,
                metadata=self.metadata,
                input_images=images,
                tools=tuple(tools),
                tool_results=tool_results,
                continuation=self._continuation,
            )
        )
        if response.tool_calls and response.continuation is None:
            raise ValueError("provider tool calls require continuation state")
        self._continuation = response.continuation
        return response

    def provenance(self) -> Mapping[str, Any]:
        provenance = getattr(self.backend, "provenance", None)
        return dict(provenance()) if provenance is not None else {}


class ScriptedModel:
    """Small deterministic model for offline tests and examples."""

    tool_family = "generic"

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = deque(responses)
        self.queries: list[tuple[tuple[Message, ...], tuple[ToolDefinition, ...]]] = []

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        self.queries.append((tuple(messages), tuple(tools)))
        if not self.responses:
            raise AssertionError("scripted model has no response left")
        return self.responses.popleft()


class ScriptedBackend:
    """Deterministic backend adapter for CLI and evaluation integration tests."""

    tool_family = "generic"

    def __init__(self, responses: Mapping[str, Sequence[ModelResponse]]) -> None:
        self._responses = {
            agent_id: deque(values) for agent_id, values in responses.items()
        }
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        queue = self._responses.get(request.agent_id)
        if not queue:
            raise AssertionError(f"no scripted response for {request.agent_id}")
        return queue.popleft()

    def provenance(self) -> Mapping[str, Any]:
        return {"provider": "scripted", "deterministic": True}


__all__ = ["BackendModel", "Model", "ScriptedBackend", "ScriptedModel"]
