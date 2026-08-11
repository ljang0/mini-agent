"""Small BrowseComp client-family adapters layered over :class:`MiniAgent`.

The provider transport remains independent.  This module only translates the
published client families' tool names or text tags into the common ``search``
tool contract used by :class:`~mini_agent.environments.web.WebEnvironment`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .models import BackendModel, Model
from .providers import ProviderError
from .types import Message, ModelRequest, ModelResponse, ToolCall, ToolDefinition


PROVIDER_TOOL_CALLS = "provider_tool_calls"
GEMINI_FUNCTION_CALLS = "gemini_function_calls"
OSS_RETRIEVAL_TOOL_CALLS = "oss_retrieval_tool_calls"
QWEN_MCP_TOOL_CALLS = "qwen_mcp_tool_calls"
SEARCH_R1_TAGS = "search_r1_tags"
TONGYI_REACT_TAGS = "tongyi_react_tags"
SUPPORTED_WEB_RESPONSE_PARSERS = frozenset(
    {
        PROVIDER_TOOL_CALLS,
        GEMINI_FUNCTION_CALLS,
        OSS_RETRIEVAL_TOOL_CALLS,
        QWEN_MCP_TOOL_CALLS,
        SEARCH_R1_TAGS,
        TONGYI_REACT_TAGS,
    }
)
_SEARCH_R1 = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_TONGYI_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _search_definition(
    definition: ToolDefinition, parser: str
) -> ToolDefinition:
    if definition.name != "search":
        return definition
    if parser == OSS_RETRIEVAL_TOOL_CALLS:
        return ToolDefinition(
            name="local_knowledge_base_retrieval",
            description=definition.description,
            input_schema={
                "type": "object",
                "properties": {"user_query": {"type": "string"}},
                "required": ["user_query"],
                "additionalProperties": False,
            },
            provider_options=definition.provider_options,
        )
    if parser == QWEN_MCP_TOOL_CALLS:
        return ToolDefinition(
            name="search-server-search",
            description=definition.description,
            input_schema=definition.input_schema,
            provider_options=definition.provider_options,
        )
    return definition


def _translate_provider_call(call: ToolCall, parser: str) -> ToolCall:
    name = call.name
    arguments = dict(call.arguments)
    if parser == OSS_RETRIEVAL_TOOL_CALLS and name == "local_knowledge_base_retrieval":
        name = "search"
        arguments = {"query": arguments.get("user_query")}
    elif parser == QWEN_MCP_TOOL_CALLS and name.startswith("search-server-"):
        name = name.removeprefix("search-server-")
    return ToolCall(
        call_id=call.call_id,
        name=name,
        arguments=arguments,
        kind=call.kind,
        agent_id=call.agent_id,
        raw=call.raw,
    )


def parse_web_response(
    response: ModelResponse,
    parser: str,
    *,
    call_id: str = "web-call-1",
) -> ModelResponse:
    """Normalize one published response convention into a model response."""

    if parser not in SUPPORTED_WEB_RESPONSE_PARSERS:
        raise ValueError(f"unsupported web response parser {parser!r}")
    if parser in {
        PROVIDER_TOOL_CALLS,
        GEMINI_FUNCTION_CALLS,
        OSS_RETRIEVAL_TOOL_CALLS,
        QWEN_MCP_TOOL_CALLS,
    }:
        calls = tuple(_translate_provider_call(call, parser) for call in response.tool_calls)
    elif parser == SEARCH_R1_TAGS:
        matches = _SEARCH_R1.findall(response.text)
        calls = (
            (ToolCall(call_id, "search", {"query": matches[-1].strip()}),)
            if matches and matches[-1].strip()
            else ()
        )
    else:
        calls_list: list[ToolCall] = []
        for index, encoded in enumerate(_TONGYI_CALL.findall(response.text), 1):
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ProviderError("Tongyi returned malformed <tool_call> JSON") from exc
            if not isinstance(value, Mapping):
                raise ProviderError("Tongyi <tool_call> must contain a JSON object")
            name = value.get("name")
            arguments = value.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise ProviderError("Tongyi <tool_call> requires name and arguments")
            calls_list.append(
                ToolCall(f"{call_id}-{index}", name, dict(arguments))
            )
        calls = tuple(calls_list)
    return ModelResponse(
        text=response.text,
        usage=response.usage,
        provider_latency_seconds=response.provider_latency_seconds,
        raw=response.raw,
        tool_calls=calls,
        continuation=response.continuation,
    )


class WebToolModel:
    """Translate provider-native tool schemas/calls for a web client family."""

    def __init__(self, model: Model, response_parser: str) -> None:
        if response_parser not in SUPPORTED_WEB_RESPONSE_PARSERS:
            raise ValueError(f"unsupported web response parser {response_parser!r}")
        if response_parser in {SEARCH_R1_TAGS, TONGYI_REACT_TAGS}:
            raise ValueError("tag parsers require TaggedWebModel")
        self.model = model
        self.response_parser = response_parser

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        translated_tools = tuple(
            _search_definition(definition, self.response_parser)
            for definition in tools
        )
        response = await self.model.query(messages, translated_tools)
        return parse_web_response(response, self.response_parser)

    def provenance(self) -> Mapping[str, Any]:
        provenance = getattr(self.model, "provenance", None)
        return {
            **(dict(provenance()) if provenance is not None else {}),
            "web_response_parser": self.response_parser,
        }


class TaggedWebModel:
    """Run Search-R1/Tongyi text protocols over any compatible backend.

    Each request is stateless and contains the complete MiniAgent transcript.
    That keeps locally executed ``search`` calls valid without pretending the
    generic endpoint reproduces either upstream model's exact serving runtime.
    """

    def __init__(
        self,
        backend: Any,
        response_parser: str,
        *,
        max_output_tokens: int | None = None,
        agent_id: str = "/root",
        role: str = "solver",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if response_parser not in {SEARCH_R1_TAGS, TONGYI_REACT_TAGS}:
            raise ValueError("TaggedWebModel requires a tag response parser")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        self.backend = backend
        self.response_parser = response_parser
        self.max_output_tokens = max_output_tokens
        self.agent_id = agent_id
        self.role = role
        self.metadata = dict(metadata or {})
        self._call_number = 0

    def _tool_output(self, output: str) -> str:
        if self.response_parser == SEARCH_R1_TAGS:
            return f"<information>{output}</information>"
        return f"<tool_response>\n{output}\n</tool_response>"

    def _transcript(self, messages: Sequence[Message]) -> tuple[str, str]:
        system_parts: list[str] = []
        transcript: list[str] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role == "user":
                transcript.append(message.content)
            elif message.role == "assistant":
                transcript.append(message.content)
            else:
                transcript.extend(
                    self._tool_output(result.output) for result in message.tool_results
                )
        return "\n\n".join(system_parts), "\n\n".join(transcript)

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        if tuple(definition.name for definition in tools) != ("search",):
            raise ValueError("tagged web models require exactly the search tool")
        system, prompt = self._transcript(messages)
        response = await self.backend.complete(
            ModelRequest(
                agent_id=self.agent_id,
                role=self.role,
                prompt=prompt,
                system=system,
                max_output_tokens=self.max_output_tokens,
                metadata=self.metadata,
            )
        )
        self._call_number += 1
        return parse_web_response(
            response,
            self.response_parser,
            call_id=f"web-call-{self._call_number}",
        )

    def provenance(self) -> Mapping[str, Any]:
        provenance = getattr(self.backend, "provenance", None)
        return {
            **(dict(provenance()) if provenance is not None else {}),
            "web_response_parser": self.response_parser,
            "history_transport": "stateless_text_transcript",
            "agent_id": self.agent_id,
            "role": self.role,
        }


def build_web_model(
    backend: Any,
    *,
    response_parser: str,
    max_output_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    agent_id: str = "/root",
    role: str = "solver",
) -> Model:
    """Resolve every executable web profile to the shared model interface."""

    if response_parser not in SUPPORTED_WEB_RESPONSE_PARSERS:
        raise ValueError(f"unsupported web response parser {response_parser!r}")
    if response_parser in {SEARCH_R1_TAGS, TONGYI_REACT_TAGS}:
        return TaggedWebModel(
            backend,
            response_parser,
            max_output_tokens=max_output_tokens,
            agent_id=agent_id,
            role=role,
            metadata=metadata,
        )
    return WebToolModel(
        BackendModel(
            backend,
            agent_id=agent_id,
            role=role,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
        ),
        response_parser,
    )


__all__ = [
    "GEMINI_FUNCTION_CALLS",
    "OSS_RETRIEVAL_TOOL_CALLS",
    "PROVIDER_TOOL_CALLS",
    "QWEN_MCP_TOOL_CALLS",
    "SEARCH_R1_TAGS",
    "SUPPORTED_WEB_RESPONSE_PARSERS",
    "TONGYI_REACT_TAGS",
    "TaggedWebModel",
    "WebToolModel",
    "build_web_model",
    "parse_web_response",
]
