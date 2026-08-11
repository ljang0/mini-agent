from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

import httpx

from .types import (
    MiniAgentError,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)


_monotonic = time.monotonic


class ProviderError(MiniAgentError):
    def __init__(
        self,
        message: str,
        *,
        usage: Optional[Usage] = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.raw = raw


@dataclass(frozen=True)
class TokenPricing:
    input_per_million_usd: float
    output_per_million_usd: float
    cache_read_per_million_usd: Optional[float] = None
    cache_write_per_million_usd: Optional[float] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_million_usd", self.input_per_million_usd),
            ("output_per_million_usd", self.output_per_million_usd),
            ("cache_read_per_million_usd", self.cache_read_per_million_usd),
            ("cache_write_per_million_usd", self.cache_write_per_million_usd),
        ):
            if value is None:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
    ) -> float:
        uncached = input_tokens - cache_read_input_tokens - cache_write_input_tokens
        cache_read_rate = (
            self.cache_read_per_million_usd
            if self.cache_read_per_million_usd is not None
            else self.input_per_million_usd
        )
        cache_write_rate = (
            self.cache_write_per_million_usd
            if self.cache_write_per_million_usd is not None
            else self.input_per_million_usd
        )
        return (
            uncached * self.input_per_million_usd
            + cache_read_input_tokens * cache_read_rate
            + cache_write_input_tokens * cache_write_rate
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000


async def _post_json(
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json", **dict(headers)},
                json=dict(payload),
            )
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"provider request failed ({type(exc).__name__})",
            raw={"exception": str(exc)},
        ) from exc
    if response.is_error:
        raise ProviderError(
            f"provider returned HTTP {response.status_code}",
            raw={
                "status_code": response.status_code,
                "response_body": response.text,
            },
        )
    try:
        parsed = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError("provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return parsed


async def _request_json(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
    payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.request(
                method,
                url,
                headers={"Content-Type": "application/json", **dict(headers)},
                json=dict(payload) if payload is not None else None,
            )
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"provider request failed ({type(exc).__name__})",
            raw={"exception": str(exc)},
        ) from exc
    if response.is_error:
        raise ProviderError(
            f"provider returned HTTP {response.status_code}",
            raw={"status_code": response.status_code, "response_body": response.text},
        )
    if not response.content:
        return {}
    try:
        parsed = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError("provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return parsed


def _usage_int(
    raw: Mapping[str, Any],
    name: str,
    *,
    required: bool = False,
) -> int:
    if name not in raw or raw.get(name) is None:
        if required:
            raise ValueError(f"missing {name}")
        return 0
    value = raw[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _tool_kind_by_name(definitions: Sequence[ToolDefinition], name: str) -> str:
    for definition in definitions:
        if definition.name == name:
            return definition.kind
    return "function"


def _openai_strict_schema_compatible(schema: Mapping[str, Any]) -> bool:
    """Return whether every object node meets OpenAI strict-mode requirements."""

    if schema.get("type") != "object" or "$ref" in schema:
        return False

    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            if "$ref" in value:
                return False
            raw_type = value.get("type")
            is_object = raw_type == "object" or (
                isinstance(raw_type, list) and "object" in raw_type
            )
            if is_object:
                properties = value.get("properties")
                if not isinstance(properties, Mapping):
                    return False
                if not all(isinstance(name, str) for name in properties):
                    return False
                required = value.get("required", [])
                if (
                    not isinstance(required, list)
                    or not all(isinstance(name, str) for name in required)
                    or len(required) != len(set(required))
                    or set(required) != set(properties)
                    or value.get("additionalProperties") is not False
                ):
                    return False
            return all(visit(child) for child in value.values())
        if isinstance(value, list):
            return all(visit(child) for child in value)
        return True

    return visit(schema)


def _openai_tool(
    definition: ToolDefinition, *, native_openai: bool
) -> Mapping[str, Any]:
    if definition.kind == "function":
        options = dict(definition.provider_options)
        has_explicit_strict = "strict" in options
        explicit_strict = options.pop("strict", None)
        if has_explicit_strict and not isinstance(explicit_strict, bool):
            raise ProviderError("OpenAI function tool strict option must be boolean")
        encoded: dict[str, Any] = {
            "type": "function",
            "name": definition.name,
            "description": definition.description,
            "parameters": dict(definition.input_schema),
        }
        if native_openai:
            encoded["strict"] = (
                explicit_strict is not False
                and _openai_strict_schema_compatible(definition.input_schema)
            )
        elif has_explicit_strict:
            encoded["strict"] = explicit_strict
        encoded.update(options)
        return encoded
    if definition.kind == "openai_computer":
        return {"type": "computer", **dict(definition.provider_options)}
    if definition.kind == "openai_shell_local":
        return {
            "type": "shell",
            "environment": {"type": "local"},
            **dict(definition.provider_options),
        }
    raise ProviderError(
        f"OpenAI Responses does not support tool kind {definition.kind!r}"
    )


def _openai_function_output(result: ToolResult) -> Any:
    if result.image_data_url is None:
        return result.output
    return [
        {"type": "input_text", "text": result.output},
        {
            "type": "input_image",
            "image_url": result.image_data_url,
            "detail": "original",
        },
    ]


def _openai_tool_result(result: ToolResult) -> Mapping[str, Any]:
    if result.kind == "openai_computer":
        if result.image_data_url is None:
            raise ProviderError("computer tool result requires a screenshot")
        encoded: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": result.call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": result.image_data_url,
                "detail": "original",
            },
        }
        if isinstance(result.native_output, Mapping):
            checks = result.native_output.get("acknowledged_safety_checks")
            if checks is not None:
                if not isinstance(checks, list) or not all(
                    isinstance(check, Mapping) for check in checks
                ):
                    raise ProviderError(
                        "acknowledged_safety_checks must be an object list"
                    )
                encoded["acknowledged_safety_checks"] = [
                    dict(check) for check in checks
                ]
        return encoded
    if result.kind == "openai_shell_local" and isinstance(
        result.native_output, Mapping
    ):
        return {
            "type": "shell_call_output",
            "call_id": result.call_id,
            **dict(result.native_output),
        }
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": _openai_function_output(result),
    }


def _continuation_history(continuation: Any, provider: str) -> list[Mapping[str, Any]]:
    if continuation is None:
        return []
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("provider") != provider
    ):
        raise ProviderError(f"invalid {provider} continuation state")
    history = continuation.get("history")
    if not isinstance(history, list) or not all(
        isinstance(item, Mapping) for item in history
    ):
        raise ProviderError(f"invalid {provider} continuation history")
    return [dict(item) for item in history]


def _retain_anthropic_images(
    messages: list[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Drop old Anthropic image blocks while preserving surrounding tool results."""

    keep = metadata.get("images_to_keep")
    if keep is None:
        return messages
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
        raise ProviderError("images_to_keep must be a non-negative integer")
    copied: list[Mapping[str, Any]] = copy.deepcopy(messages)
    images: list[tuple[list[Any], int]] = []

    def collect(value: Any) -> None:
        if isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, Mapping) and child.get("type") == "image":
                    images.append((value, index))
                else:
                    collect(child)
        elif isinstance(value, Mapping):
            for child in value.values():
                collect(child)

    collect(copied)
    removal_count = max(0, len(images) - keep)
    chunk = metadata.get("image_removal_chunk", 1)
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ProviderError("image_removal_chunk must be a positive integer")
    if removal_count:
        removal_count = min(len(images), max(chunk, removal_count))
    for container, index in images[:removal_count]:
        container[index] = {"type": "text", "text": "[older screenshot omitted]"}
    return copied


def _hosted_tool_payloads(
    metadata: Mapping[str, Any],
    *,
    metadata_key: str,
    allowed_types: frozenset[str],
    provider: str,
) -> list[Mapping[str, str]]:
    """Validate request-scoped server tools and encode their public API shape.

    Server tools are deliberately carried in metadata rather than ``request.tools``:
    the latter represents developer/client tools whose calls the local runtime must
    execute.  Accepting only bare type strings prevents arbitrary provider payloads
    from bypassing this adapter's audited allowlist.
    """

    raw_tools = metadata.get(metadata_key, ())
    if not isinstance(raw_tools, (list, tuple)):
        raise ProviderError(f"{provider} hosted tools must be a list of tool types")
    encoded: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for tool_type in raw_tools:
        if not isinstance(tool_type, str) or tool_type not in allowed_types:
            raise ProviderError(
                f"unsupported {provider} hosted tool {tool_type!r}; "
                f"allowed types are {sorted(allowed_types)}"
            )
        if tool_type in seen:
            raise ProviderError(f"duplicate {provider} hosted tool {tool_type!r}")
        seen.add(tool_type)
        encoded.append({"type": tool_type})
    return encoded


class OpenAIResponsesBackend:
    """OpenAI-compatible Responses API backend.

    It also supports OpenAI's hosted multi-agent beta when requested by the
    OpenAIHostedMultiAgentHarness. Compatible providers may ignore or reject that beta;
    use it only with an endpoint that documents the same fields.
    """

    tool_family = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 300.0,
        pricing: Optional[TokenPricing] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        tool_family: str = "openai",
        provider_label: str = "openai-responses",
    ) -> None:
        self.model = model
        self.api_key: str = api_key or os.getenv(api_key_env, "") or ""
        if not self.api_key:
            raise ValueError(f"missing API key; set {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.pricing = pricing
        self.extra_body = dict(extra_body or {})
        if not isinstance(tool_family, str) or not tool_family:
            raise ValueError("tool_family must be a non-empty string")
        if not isinstance(provider_label, str) or not provider_label:
            raise ValueError("provider_label must be a non-empty string")
        self.tool_family = tool_family
        self.provider_label = provider_label
        reserved = {
            "model",
            "input",
            "store",
            "instructions",
            "max_output_tokens",
            "multi_agent",
            "tools",
            "tool_choice",
            "stream",
            "background",
            "previous_response_id",
            "conversation",
        } & self.extra_body.keys()
        if reserved:
            raise ValueError(
                "provider_extra_body cannot override OpenAI core fields: "
                f"{sorted(reserved)}"
            )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider_label,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "pricing": (
                {
                    "input_per_million_usd": self.pricing.input_per_million_usd,
                    "output_per_million_usd": self.pricing.output_per_million_usd,
                    "cache_read_per_million_usd": self.pricing.cache_read_per_million_usd,
                    "cache_write_per_million_usd": self.pricing.cache_write_per_million_usd,
                }
                if self.pricing
                else None
            ),
            "extra_body": self.extra_body,
            "tool_family": self.tool_family,
            "supported_hosted_tools": ["web_search"],
        }

    async def complete(self, request: ModelRequest) -> ModelResponse:
        history = _continuation_history(request.continuation, "openai-responses")
        if history:
            input_items: Any = [
                *history,
                *(_openai_tool_result(result) for result in request.tool_results),
            ]
        else:
            if request.tool_results:
                raise ProviderError("tool results require OpenAI continuation state")
            if request.input_images:
                input_items = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": request.prompt},
                            *(
                                {
                                    "type": "input_image",
                                    "image_url": image,
                                    "detail": "original",
                                }
                                for image in request.input_images
                            ),
                        ],
                    }
                ]
            else:
                input_items = request.prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
            **self.extra_body,
        }
        if request.system:
            payload["instructions"] = request.system
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        hosted_multi_agent = request.metadata.get("openai_multi_agent") is True
        if "openai_hosted_tools" in request.metadata and not hosted_multi_agent:
            raise ProviderError(
                "OpenAI hosted tools require an openai_hosted_multi_agent request"
            )
        hosted_tools = (
            _hosted_tool_payloads(
                request.metadata,
                metadata_key="openai_hosted_tools",
                allowed_types=frozenset({"web_search"}),
                provider="OpenAI",
            )
            if hosted_multi_agent
            else []
        )
        encoded_tools = [
            _openai_tool(tool, native_openai=self.tool_family == "openai")
            for tool in request.tools
        ]
        if encoded_tools or hosted_tools:
            payload["tools"] = [*encoded_tools, *hosted_tools]
        if hosted_multi_agent and not self.model.startswith("gpt-5.6"):
            raise ProviderError(
                "OpenAI hosted multi-agent beta requires a GPT-5.6 model"
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if hosted_multi_agent:
            payload["multi_agent"] = {
                "enabled": True,
                "max_concurrent_subagents": int(
                    request.metadata.get("max_concurrent_subagents", 3)
                ),
            }
            headers["OpenAI-Beta"] = "responses_multi_agent=v1"
        started = time.perf_counter()
        data = await _post_json(
            f"{self.base_url}/responses",
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        text_parts: list[str] = []
        fallback_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        parse_error: Optional[str] = None
        raw_output = data.get("output")
        output_malformed = not isinstance(raw_output, list)
        for item in raw_output if isinstance(raw_output, list) else []:
            if not isinstance(item, dict):
                output_malformed = True
                continue
            item_type = item.get("type")
            raw_agent = item.get("agent")
            item_agent = (
                raw_agent.get("agent_name", "") if isinstance(raw_agent, dict) else ""
            )
            if item_type == "function_call":
                raw_arguments = item.get("arguments")
                if not isinstance(raw_arguments, (str, bytes, bytearray)):
                    parse_error = "invalid function arguments: expected JSON text"
                    continue
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must decode to an object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    parse_error = f"invalid function arguments: {exc}"
                    continue
                name = item.get("name")
                call_id = item.get("call_id")
                if not isinstance(name, str) or not isinstance(call_id, str):
                    parse_error = "function_call requires string name and call_id"
                    continue
                tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        kind=_tool_kind_by_name(request.tools, name),
                        agent_id=item_agent,
                        raw=item,
                    )
                )
                continue
            if item_type == "computer_call":
                call_id = item.get("call_id")
                actions = item.get("actions")
                if not isinstance(call_id, str) or not isinstance(actions, list):
                    parse_error = "computer_call requires call_id and actions[]"
                    continue
                tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name="computer",
                        arguments={"actions": actions},
                        kind="openai_computer",
                        agent_id=item_agent,
                        raw=item,
                    )
                )
                continue
            if item_type == "shell_call":
                call_id = item.get("call_id")
                action = item.get("action")
                if not isinstance(call_id, str) or not isinstance(action, dict):
                    parse_error = "shell_call requires call_id and action"
                    continue
                tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name="shell",
                        arguments=action,
                        kind="openai_shell_local",
                        agent_id=item_agent,
                        raw=item,
                    )
                )
                continue
            if item_type != "message":
                continue
            agent = item.get("agent") or {}
            is_root_final = (
                isinstance(agent, dict)
                and agent.get("agent_name") == "/root"
                and item.get("phase") == "final_answer"
            )
            raw_content = item.get("content")
            if not isinstance(raw_content, list):
                output_malformed = True
                continue
            for content in raw_content:
                if (
                    not isinstance(content, dict)
                    or content.get("type") != "output_text"
                ):
                    continue
                value = content.get("text")
                if isinstance(value, str):
                    fallback_parts.append(value)
                    if is_root_final:
                        text_parts.append(value)
        answer = "".join(text_parts if hosted_multi_agent else fallback_parts)
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            try:
                input_tokens = _usage_int(raw_usage, "input_tokens", required=True)
                output_tokens = _usage_int(raw_usage, "output_tokens", required=True)
                raw_details = raw_usage.get("input_tokens_details")
                if raw_details is not None and not isinstance(raw_details, dict):
                    raise ValueError("input_tokens_details must be an object")
                input_details = raw_details or {}
                cache_read_tokens = _usage_int(input_details, "cached_tokens")
                cost_ticks = raw_usage.get("cost_in_usd_ticks")
                if cost_ticks is not None and (
                    not isinstance(cost_ticks, int)
                    or isinstance(cost_ticks, bool)
                    or cost_ticks < 0
                ):
                    raise ValueError("cost_in_usd_ticks must be a non-negative integer")
            except ValueError as exc:
                raise ProviderError(
                    f"OpenAI Responses API returned invalid usage: {exc}",
                    usage=Usage(cost_known=False, complete=False),
                    raw=data,
                ) from exc
            if cost_ticks is not None:
                cost = cost_ticks / 10_000_000_000
                cost_known = True
            elif self.pricing is not None:
                cost = self.pricing.estimate(
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                )
                cost_known = True
            else:
                cost = 0.0
                cost_known = False
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cost_usd=cost,
                cost_known=cost_known,
            )
        else:
            usage = Usage(cost_known=False, complete=False)
        if data.get("status") != "completed":
            raise ProviderError(
                "Responses API did not report status='completed': "
                f"{data.get('status')!r}",
                usage=Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_write_input_tokens=usage.cache_write_input_tokens,
                    cost_usd=usage.cost_usd,
                    cost_known=usage.cost_known,
                    complete=False,
                ),
                raw=data,
            )
        if output_malformed:
            raise ProviderError(
                "Responses API returned malformed message output",
                usage=usage,
                raw=data,
            )
        if parse_error is not None:
            raise ProviderError(
                f"OpenAI Responses API returned {parse_error}", usage=usage, raw=data
            )
        if not answer and not tool_calls:
            if hosted_multi_agent:
                raise ProviderError(
                    "hosted multi-agent response contained no /root final_answer",
                    usage=usage,
                    raw=data,
                )
            raise ProviderError(
                "Responses API returned no output_text", usage=usage, raw=data
            )
        return ModelResponse(
            text=answer,
            usage=usage,
            provider_latency_seconds=latency,
            raw=data,
            tool_calls=tuple(tool_calls),
            continuation={
                "provider": "openai-responses",
                "history": [
                    *(
                        history
                        if history
                        else [{"role": "user", "content": request.prompt}]
                    ),
                    *(_openai_tool_result(result) for result in request.tool_results),
                    *(raw_output if isinstance(raw_output, list) else []),
                ],
            },
        )


class OpenAICompatibleChatBackend:
    """Generic OpenAI-compatible Chat Completions tool-loop backend.

    This adapter is intentionally limited to public chat/function-call semantics.  It
    does not imply that a compatible endpoint implements OpenAI's hosted scheduler,
    native computer tool, shell tool, or Responses continuation behavior.
    """

    tool_family = "generic"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str,
        timeout_seconds: float = 300.0,
        pricing: Optional[TokenPricing] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        provider_label: str = "openai-compatible-chat",
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env, "") or ""
        if not self.api_key:
            raise ValueError(f"missing API key; set {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.pricing = pricing
        self.extra_body = dict(extra_body or {})
        self.provider_label = provider_label
        reserved = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "max_tokens",
        } & self.extra_body.keys()
        if reserved:
            raise ValueError(
                "provider_extra_body cannot override chat core fields: "
                f"{sorted(reserved)}"
            )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider_label,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "tool_family": self.tool_family,
            "native_computer": False,
            "native_multi_agent": False,
            "extra_body": self.extra_body,
        }

    @staticmethod
    def _tool_result_message(result: ToolResult) -> Mapping[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.output,
        }

    @classmethod
    def _tool_result_messages(
        cls, results: Sequence[ToolResult]
    ) -> list[Mapping[str, Any]]:
        messages = [cls._tool_result_message(result) for result in results]
        image_content: list[Mapping[str, Any]] = []
        for result in results:
            if result.image_data_url is None:
                continue
            image_content.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            f"Image returned by tool {result.name!r} for call "
                            f"{result.call_id}."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": result.image_data_url,
                            "detail": "high",
                        },
                    },
                )
            )
        if image_content:
            # Chat tool messages accept text parts only. Supply screenshot outputs in
            # one user image message after every pending tool call has its matching
            # tool response, preserving the required assistant/tool ordering.
            messages.append({"role": "user", "content": image_content})
        return messages

    async def complete(self, request: ModelRequest) -> ModelResponse:
        history = _continuation_history(request.continuation, "openai-chat")
        if history:
            messages: list[Mapping[str, Any]] = [
                *history,
                *self._tool_result_messages(request.tool_results),
            ]
        else:
            if request.tool_results:
                raise ProviderError("tool results require chat continuation state")
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            if request.input_images:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.prompt},
                            *(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image, "detail": "high"},
                                }
                                for image in request.input_images
                            ),
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.tools:
            encoded_tools = []
            for definition in request.tools:
                if definition.kind != "function":
                    raise ProviderError(
                        "OpenAI-compatible Chat supports generic function tools only"
                    )
                options = dict(definition.provider_options)
                has_explicit_strict = "strict" in options
                explicit_strict = options.pop("strict", None)
                if has_explicit_strict and not isinstance(explicit_strict, bool):
                    raise ProviderError(
                        "compatible Chat function tool strict option must be boolean"
                    )
                function: dict[str, Any] = {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": dict(definition.input_schema),
                    **options,
                }
                if has_explicit_strict:
                    function["strict"] = explicit_strict
                encoded_tools.append({"type": "function", "function": function})
            payload["tools"] = encoded_tools
        started = time.perf_counter()
        data = await _post_json(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            try:
                input_tokens = _usage_int(raw_usage, "prompt_tokens", required=True)
                output_tokens = _usage_int(
                    raw_usage, "completion_tokens", required=True
                )
                raw_details = raw_usage.get("prompt_tokens_details")
                if raw_details is not None and not isinstance(raw_details, dict):
                    raise ValueError("prompt_tokens_details must be an object")
                cache_read = _usage_int(raw_details or {}, "cached_tokens")
            except ValueError as exc:
                raise ProviderError(
                    f"chat endpoint returned invalid usage: {exc}",
                    usage=Usage(cost_known=False, complete=False),
                    raw=data,
                ) from exc
            cost = (
                self.pricing.estimate(
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens=cache_read,
                )
                if self.pricing is not None
                else 0.0
            )
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cost_usd=cost,
                cost_known=self.pricing is not None,
            )
        else:
            usage = Usage(cost_known=False, complete=False)
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderError(
                "chat endpoint must return exactly one choice", usage=usage, raw=data
            )
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ProviderError(
                "chat endpoint returned malformed choice", usage=usage, raw=data
            )
        message = choice["message"]
        raw_text = message.get("content")
        text = raw_text if isinstance(raw_text, str) else ""
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ProviderError(
                "chat endpoint returned malformed tool_calls", usage=usage, raw=data
            )
        calls: list[ToolCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict) or not isinstance(
                raw_call.get("function"), dict
            ):
                raise ProviderError(
                    "chat endpoint returned malformed tool call", usage=usage, raw=data
                )
            call_id = raw_call.get("id")
            function = raw_call["function"]
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, (str, bytes, bytearray)):
                raise ProviderError(
                    "chat endpoint returned invalid tool arguments",
                    usage=usage,
                    raw=data,
                )
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ProviderError(
                    "chat endpoint returned invalid tool arguments",
                    usage=usage,
                    raw=data,
                ) from exc
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                raise ProviderError(
                    "chat endpoint returned invalid tool call fields",
                    usage=usage,
                    raw=data,
                )
            calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    kind="function",
                    raw=raw_call,
                )
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "tool_calls"}:
            raise ProviderError(
                f"chat endpoint stopped with {finish_reason!r}",
                usage=Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cost_usd=usage.cost_usd,
                    cost_known=usage.cost_known,
                    complete=False,
                ),
                raw=data,
            )
        if not text and not calls:
            raise ProviderError(
                "chat endpoint returned neither text nor tool calls",
                usage=usage,
                raw=data,
            )
        return ModelResponse(
            text=text,
            usage=usage,
            provider_latency_seconds=latency,
            raw=data,
            tool_calls=tuple(calls),
            continuation={
                "provider": "openai-chat",
                "history": [
                    *messages,
                    message,
                ],
            },
        )


class XAIResponsesBackend:
    """xAI Responses backend for the documented hosted multi-agent model.

    The public API exposes the leader's final output and aggregate billed usage, but
    not the server scheduler or unencrypted intermediate subagent state. Documented
    server-owned search tools may be enabled per request; developer/client tools stay
    rejected because the multi-agent model does not support them.
    """

    tool_family = "xai_hosted"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "XAI_API_KEY",
        base_url: str = "https://api.x.ai/v1",
        timeout_seconds: float = 600.0,
        pricing: Optional[TokenPricing] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.api_key: str = api_key or os.getenv(api_key_env, "") or ""
        if not self.api_key:
            raise ValueError(f"missing API key; set {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.pricing = pricing
        self.extra_body = dict(extra_body or {})
        reserved = {
            "model",
            "input",
            "reasoning",
            "tools",
            "stream",
            "background",
            "previous_response_id",
            "conversation",
        } & self.extra_body.keys()
        if reserved:
            raise ValueError(
                "provider_extra_body cannot override xAI hosted multi-agent fields: "
                f"{sorted(reserved)}"
            )
        if self.model != "grok-4.20-multi-agent-0309":
            raise ValueError(
                "xAI hosted multi-agent adapter requires the pinned model "
                "'grok-4.20-multi-agent-0309'"
            )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "xai-responses",
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "hosted_multi_agent_beta": True,
            "supported_built_in_tools": ["web_search", "x_search"],
            "built_in_tools_request_scoped": True,
            "leader_output_only": True,
            "server_scheduler_open": False,
            "plaintext_subagent_state_available": False,
            "encrypted_continuation_implemented": False,
            "pricing": (
                {
                    "input_per_million_usd": self.pricing.input_per_million_usd,
                    "output_per_million_usd": self.pricing.output_per_million_usd,
                    "cache_read_per_million_usd": self.pricing.cache_read_per_million_usd,
                    "cache_write_per_million_usd": self.pricing.cache_write_per_million_usd,
                }
                if self.pricing
                else None
            ),
            "extra_body": self.extra_body,
        }

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.tools or request.tool_results or request.continuation is not None:
            raise ProviderError(
                "xAI hosted multi-agent adapter does not implement client tools"
            )
        if request.metadata.get("xai_multi_agent") is not True:
            raise ProviderError(
                "xAI hosted backend requires an xai_hosted_multi_agent request"
            )
        reasoning_effort = request.metadata.get("reasoning_effort")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ProviderError(
                "xAI multi-agent reasoning_effort must be low, medium, high, or xhigh"
            )
        hosted_tools = _hosted_tool_payloads(
            request.metadata,
            metadata_key="xai_hosted_tools",
            allowed_types=frozenset({"web_search", "x_search"}),
            provider="xAI",
        )
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"
        payload: dict[str, Any] = {
            **self.extra_body,
            "model": self.model,
            "reasoning": {"effort": reasoning_effort},
            "input": [{"role": "user", "content": prompt}],
        }
        if hosted_tools:
            payload["tools"] = hosted_tools
        started = time.perf_counter()
        data = await _post_json(
            f"{self.base_url}/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started

        text_parts: list[str] = []
        raw_output = data.get("output")
        output_malformed = not isinstance(raw_output, list)
        for item in raw_output if isinstance(raw_output, list) else []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            raw_content = item.get("content")
            if not isinstance(raw_content, list):
                output_malformed = True
                continue
            for content in raw_content:
                if (
                    not isinstance(content, dict)
                    or content.get("type") != "output_text"
                ):
                    continue
                value = content.get("text")
                if isinstance(value, str):
                    text_parts.append(value)

        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            try:
                input_tokens = _usage_int(raw_usage, "input_tokens", required=True)
                output_tokens = _usage_int(raw_usage, "output_tokens", required=True)
                raw_details = raw_usage.get("input_tokens_details")
                if raw_details is not None and not isinstance(raw_details, dict):
                    raise ValueError("input_tokens_details must be an object")
                input_details = raw_details or {}
                cache_read_tokens = _usage_int(input_details, "cached_tokens")
                cost_ticks = raw_usage.get("cost_in_usd_ticks")
                if cost_ticks is not None and (
                    not isinstance(cost_ticks, int)
                    or isinstance(cost_ticks, bool)
                    or cost_ticks < 0
                ):
                    raise ValueError("cost_in_usd_ticks must be a non-negative integer")
            except ValueError as exc:
                raise ProviderError(
                    f"xAI Responses API returned invalid usage: {exc}",
                    usage=Usage(cost_known=False, complete=False),
                    raw=data,
                ) from exc
            if cost_ticks is not None:
                cost = cost_ticks / 10_000_000_000
                cost_known = True
            elif self.pricing is not None:
                cost = self.pricing.estimate(
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                )
                cost_known = True
            else:
                cost = 0.0
                cost_known = False
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cost_usd=cost,
                cost_known=cost_known,
            )
        else:
            usage = Usage(cost_known=False, complete=False)
        if data.get("status") != "completed":
            raise ProviderError(
                "xAI Responses API did not report status='completed': "
                f"{data.get('status')!r}",
                usage=Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_write_input_tokens=usage.cache_write_input_tokens,
                    cost_usd=usage.cost_usd,
                    cost_known=usage.cost_known,
                    complete=False,
                ),
                raw=data,
            )
        if output_malformed:
            raise ProviderError(
                "xAI Responses API returned malformed message output",
                usage=usage,
                raw=data,
            )
        answer = "".join(text_parts)
        if not answer:
            raise ProviderError(
                "xAI Responses API returned no leader output_text",
                usage=usage,
                raw=data,
            )
        return ModelResponse(
            text=answer,
            usage=usage,
            provider_latency_seconds=latency,
            raw=data,
        )


class AnthropicMessagesBackend:
    tool_family = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        default_max_output_tokens: int = 4096,
        timeout_seconds: float = 300.0,
        pricing: Optional[TokenPricing] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.api_key: str = api_key or os.getenv(api_key_env, "") or ""
        if not self.api_key:
            raise ValueError(f"missing API key; set {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.default_max_output_tokens = default_max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.pricing = pricing
        self.extra_body = dict(extra_body or {})
        reserved = {
            "model",
            "messages",
            "system",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
        } & self.extra_body.keys()
        if reserved:
            raise ValueError(
                "provider_extra_body cannot override Anthropic core fields: "
                f"{sorted(reserved)}"
            )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "anthropic-messages",
            "model": self.model,
            "base_url": self.base_url,
            "anthropic_version": self.anthropic_version,
            "default_max_output_tokens": self.default_max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "pricing": (
                {
                    "input_per_million_usd": self.pricing.input_per_million_usd,
                    "output_per_million_usd": self.pricing.output_per_million_usd,
                    "cache_read_per_million_usd": self.pricing.cache_read_per_million_usd,
                    "cache_write_per_million_usd": self.pricing.cache_write_per_million_usd,
                }
                if self.pricing
                else None
            ),
            "extra_body": self.extra_body,
        }

    async def complete(self, request: ModelRequest) -> ModelResponse:
        history = _continuation_history(request.continuation, "anthropic-messages")
        if history:
            tool_result_blocks = [
                self._tool_result_block(result) for result in request.tool_results
            ]
            messages: list[Mapping[str, Any]] = [
                *history,
                {"role": "user", "content": tool_result_blocks},
            ]
        else:
            if request.tool_results:
                raise ProviderError("tool results require Anthropic continuation state")
            if request.input_images:
                content: list[Mapping[str, Any]] = [
                    {"type": "text", "text": request.prompt}
                ]
                for image in request.input_images:
                    prefix = "data:image/png;base64,"
                    if not image.startswith(prefix):
                        raise ProviderError(
                            "Anthropic input image must be a PNG data URL"
                        )
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image[len(prefix) :],
                            },
                        }
                    )
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": request.prompt}]
        messages = _retain_anthropic_images(messages, request.metadata)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens or self.default_max_output_tokens,
            "messages": messages,
            **self.extra_body,
        }
        if request.system:
            payload["system"] = request.system
        beta_features: list[str] = []
        if request.tools:
            payload["tools"] = []
            for definition in request.tools:
                encoded, beta = self._tool_definition(definition)
                payload["tools"].append(encoded)
                if beta is not None and beta not in beta_features:
                    beta_features.append(beta)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        if beta_features:
            headers["anthropic-beta"] = ",".join(beta_features)
        started = time.perf_counter()
        data = await _post_json(
            f"{self.base_url}/messages",
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        raw_content = data.get("content")
        content_malformed = not isinstance(raw_content, list)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_parse_error: Optional[str] = None
        for block in raw_content if isinstance(raw_content, list) else []:
            if not isinstance(block, dict):
                content_malformed = True
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
                continue
            if block.get("type") != "tool_use":
                continue
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                content_parse_error = "malformed tool_use content block"
                continue
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    kind=_tool_kind_by_name(request.tools, name),
                    raw=block,
                )
            )
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            try:
                uncached_input_tokens = _usage_int(
                    raw_usage, "input_tokens", required=True
                )
                cache_write_tokens = _usage_int(
                    raw_usage, "cache_creation_input_tokens"
                )
                cache_read_tokens = _usage_int(raw_usage, "cache_read_input_tokens")
                output_tokens = _usage_int(raw_usage, "output_tokens", required=True)
            except ValueError as exc:
                raise ProviderError(
                    f"Anthropic Messages API returned invalid usage: {exc}",
                    usage=Usage(cost_known=False, complete=False),
                    raw=data,
                ) from exc
            input_tokens = (
                uncached_input_tokens + cache_write_tokens + cache_read_tokens
            )
            cost = (
                self.pricing.estimate(
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                    cache_write_input_tokens=cache_write_tokens,
                )
                if self.pricing is not None
                else 0.0
            )
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cache_write_input_tokens=cache_write_tokens,
                cost_usd=cost,
                cost_known=self.pricing is not None,
            )
        else:
            usage = Usage(cost_known=False, complete=False)
        stop_reason = data.get("stop_reason")
        successful_stop = stop_reason in {"end_turn", "stop_sequence"} or (
            stop_reason == "tool_use" and bool(tool_calls)
        )
        if not successful_stop:
            raise ProviderError(
                "Anthropic Messages API did not report a successful stop reason: "
                f"{data.get('stop_reason')!r}",
                usage=Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_write_input_tokens=usage.cache_write_input_tokens,
                    cost_usd=usage.cost_usd,
                    cost_known=usage.cost_known,
                    complete=False,
                ),
                raw=data,
            )
        if content_malformed:
            raise ProviderError(
                "Anthropic Messages API returned malformed content",
                usage=usage,
                raw=data,
            )
        if content_parse_error is not None:
            raise ProviderError(
                f"Anthropic Messages API returned {content_parse_error}",
                usage=usage,
                raw=data,
            )
        answer = "".join(text_parts)
        if not answer and not tool_calls:
            raise ProviderError(
                "Anthropic Messages API returned no non-empty text",
                usage=usage,
                raw=data,
            )
        return ModelResponse(
            text=answer,
            usage=usage,
            provider_latency_seconds=latency,
            raw=data,
            tool_calls=tuple(tool_calls),
            continuation={
                "provider": "anthropic-messages",
                "history": [
                    *messages,
                    {"role": "assistant", "content": raw_content},
                ],
            },
        )

    @staticmethod
    def _tool_definition(
        definition: ToolDefinition,
    ) -> tuple[Mapping[str, Any], Optional[str]]:
        options = dict(definition.provider_options)
        if definition.kind == "function":
            return (
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": dict(definition.input_schema),
                    "strict": True,
                    **options,
                },
                None,
            )
        if definition.kind == "anthropic_bash_20250124":
            return ({"type": "bash_20250124", "name": "bash", **options}, None)
        if definition.kind == "anthropic_text_editor_20250728":
            return (
                {
                    "type": "text_editor_20250728",
                    "name": "str_replace_based_edit_tool",
                    **options,
                },
                None,
            )
        if definition.kind == "anthropic_computer_20251124":
            return (
                {
                    "type": "computer_20251124",
                    "name": "computer",
                    **options,
                },
                "computer-use-2025-11-24",
            )
        raise ProviderError(
            f"Anthropic Messages does not support tool kind {definition.kind!r}"
        )

    @staticmethod
    def _tool_result_block(result: ToolResult) -> Mapping[str, Any]:
        if result.image_data_url is None:
            content: Any = result.output
        else:
            prefix = "data:image/png;base64,"
            if not result.image_data_url.startswith(prefix):
                raise ProviderError("Anthropic tool image must be a PNG data URL")
            content = [
                {"type": "text", "text": result.output},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.image_data_url[len(prefix) :],
                    },
                },
            ]
        return {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": content,
            "is_error": result.is_error,
        }


class AnthropicManagedAgentsBackend:
    """Exact public HTTP boundary for Anthropic Managed Agents beta sessions.

    The referenced agent and environment must already exist.  Their immutable
    versions, toolsets, coordinator roster, sandbox, and filesystem are owned by the
    managed service; this adapter starts one fresh session and records its
    authoritative full-tree session usage.
    """

    tool_family = "anthropic_managed"

    def __init__(
        self,
        *,
        agent_id: str,
        environment_id: str,
        agent_version: Optional[int] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        beta_version: str = "managed-agents-2026-04-01",
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 1.0,
        budget_cents: Optional[int] = None,
        resources: Sequence[Mapping[str, Any]] = (),
        vault_ids: Sequence[str] = (),
        cleanup: str = "retain",
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("managed agent_id must be a non-empty string")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("managed environment_id must be a non-empty string")
        if agent_version is not None and (
            not isinstance(agent_version, int)
            or isinstance(agent_version, bool)
            or agent_version < 1
        ):
            raise ValueError("managed agent_version must be a positive integer")
        if budget_cents is not None and (
            not isinstance(budget_cents, int)
            or isinstance(budget_cents, bool)
            or budget_cents < 1
        ):
            raise ValueError("managed budget_cents must be a positive integer")
        if cleanup not in {"retain", "archive", "delete"}:
            raise ValueError("managed cleanup must be retain, archive, or delete")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("managed timeout_seconds must be finite and positive")
        if (
            not isinstance(poll_interval_seconds, (int, float))
            or isinstance(poll_interval_seconds, bool)
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
        ):
            raise ValueError("managed poll_interval_seconds must be positive")
        if not isinstance(anthropic_version, str) or not anthropic_version:
            raise ValueError("managed anthropic_version must be a non-empty string")
        if not isinstance(beta_version, str) or not beta_version:
            raise ValueError("managed beta_version must be a non-empty string")
        if not all(isinstance(item, Mapping) for item in resources):
            raise ValueError("managed resources must contain objects")
        if not all(isinstance(item, str) and item for item in vault_ids):
            raise ValueError("managed vault_ids must contain non-empty strings")
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.agent_version = agent_version
        self.api_key = api_key or os.getenv(api_key_env, "") or ""
        if not self.api_key:
            raise ValueError(f"missing API key; set {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.beta_version = beta_version
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.budget_cents = budget_cents
        self.resources = tuple(dict(item) for item in resources)
        self.vault_ids = tuple(vault_ids)
        self.cleanup = cleanup

    @property
    def _headers(self) -> Mapping[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            # Memory stores can be mounted as session resources, but session
            # endpoints still use only the Managed Agents beta.  The distinct
            # agent-memory beta replaces this header on /memory_stores endpoints;
            # this backend never calls those endpoints.
            "anthropic-beta": self.beta_version,
        }

    def provenance(self) -> Mapping[str, Any]:
        vault_digest = hashlib.sha256(
            json.dumps(self.vault_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "provider": "anthropic-managed-agents",
            "base_url": self.base_url,
            "anthropic_version": self.anthropic_version,
            "beta_version": self.beta_version,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "environment_id": self.environment_id,
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "transport": "polling",
            "budget_cents": self.budget_cents,
            "resource_count": len(self.resources),
            "resources": [
                self._redacted_resource(resource) for resource in self.resources
            ],
            "vault_count": len(self.vault_ids),
            "vault_ids_sha256": vault_digest,
            "cleanup": self.cleanup,
            "server_scheduler_open": False,
            "session_usage_scope": "authoritative_full_tree",
            "resolved_coordinator_snapshot_required": True,
        }

    def _validated_session_snapshot(
        self, session: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], str, int]:
        environment_id = session.get("environment_id")
        if environment_id != self.environment_id:
            raise ProviderError(
                "Managed Agents session returned an unexpected environment_id: "
                f"expected {self.environment_id!r}, got {environment_id!r}",
                raw=session,
            )
        raw_agent = session.get("agent")
        if not isinstance(raw_agent, Mapping):
            raise ProviderError(
                "Managed Agents session omitted its resolved agent snapshot",
                raw=session,
            )
        agent = dict(raw_agent)
        if agent.get("id") != self.agent_id:
            raise ProviderError(
                "Managed Agents resolved an unexpected agent id: "
                f"expected {self.agent_id!r}, got {agent.get('id')!r}",
                raw=session,
            )
        version = agent.get("version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
            or (self.agent_version is not None and version != self.agent_version)
        ):
            raise ProviderError(
                "Managed Agents resolved an unexpected agent version: "
                f"expected {self.agent_version!r}, got {version!r}",
                raw=session,
            )
        multiagent = agent.get("multiagent")
        if (
            not isinstance(multiagent, Mapping)
            or multiagent.get("type") != "coordinator"
        ):
            raise ProviderError(
                "Managed Agents profile requires a resolved multiagent coordinator",
                raw=session,
            )
        roster = multiagent.get("agents")
        if (
            not isinstance(roster, list)
            or not roster
            or not all(isinstance(item, Mapping) for item in roster)
        ):
            raise ProviderError(
                "Managed Agents coordinator snapshot requires a non-empty agent roster",
                raw=session,
            )
        try:
            encoded = json.dumps(
                agent,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "Managed Agents resolved agent snapshot is not valid JSON",
                raw=session,
            ) from exc
        return agent, hashlib.sha256(encoded).hexdigest(), len(roster)

    @classmethod
    def _redacted_resource(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                name = str(key)
                lowered = name.lower()
                if any(
                    marker in lowered
                    for marker in ("authorization", "token", "secret", "password")
                ):
                    redacted[name] = "<configured>"
                else:
                    redacted[name] = cls._redacted_resource(item)
            return redacted
        if isinstance(value, (list, tuple)):
            return [cls._redacted_resource(item) for item in value]
        return value

    @staticmethod
    def _usage(raw: Any) -> Usage:
        if not isinstance(raw, Mapping):
            return Usage(cost_known=False, complete=False)
        try:
            uncached = _usage_int(raw, "input_tokens", required=True)
            output = _usage_int(raw, "output_tokens", required=True)
            cache_read = _usage_int(raw, "cache_read_input_tokens")
            raw_creation = raw.get("cache_creation")
            if raw_creation is not None and not isinstance(raw_creation, Mapping):
                raise ValueError("cache_creation must be an object")
            creation = raw_creation or {}
            cache_write = _usage_int(creation, "ephemeral_5m_input_tokens") + (
                _usage_int(creation, "ephemeral_1h_input_tokens")
            )
            raw_cost = raw.get("list_cost")
            if not isinstance(raw_cost, Mapping):
                raise ValueError("list_cost must be an object")
            amount = raw_cost.get("amount")
            currency = raw_cost.get("currency")
            if not isinstance(amount, str) or not amount.isdigit() or currency != "USD":
                raise ValueError("list_cost must be whole USD cents")
        except ValueError as exc:
            raise ProviderError(
                f"Managed Agents returned invalid usage: {exc}",
                usage=Usage(cost_known=False, complete=False),
                raw=raw,
            ) from exc
        return Usage(
            input_tokens=uncached + cache_read + cache_write,
            output_tokens=output,
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            cost_usd=int(amount) / 100,
            cost_known=True,
            complete=True,
        )

    @staticmethod
    def _event_list(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        data = raw.get("data")
        if not isinstance(data, list) or not all(
            isinstance(item, Mapping) for item in data
        ):
            raise ProviderError("Managed Agents events response is malformed", raw=raw)
        return [dict(item) for item in data]

    @staticmethod
    def _incomplete(usage: Usage) -> Usage:
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            cost_usd=usage.cost_usd,
            cost_known=usage.cost_known,
            complete=False,
        )

    async def _load_events(self, session_id: str) -> Mapping[str, Any]:
        events: list[Mapping[str, Any]] = []
        page: Optional[str] = None
        seen_pages: set[str] = set()
        page_count = 0
        while True:
            suffix = ""
            if page is not None:
                suffix = f"?page={quote(page, safe='')}"
            raw = await _request_json(
                "GET",
                f"{self.base_url}/sessions/{session_id}/events{suffix}",
                headers=self._headers,
                timeout_seconds=self.timeout_seconds,
            )
            events.extend(self._event_list(raw))
            page_count += 1
            next_page = raw.get("next_page")
            if next_page is None:
                break
            if not isinstance(next_page, str) or not next_page:
                raise ProviderError(
                    "Managed Agents events response has invalid next_page", raw=raw
                )
            if next_page in seen_pages or page_count >= 10_000:
                raise ProviderError(
                    "Managed Agents events pagination did not terminate", raw=raw
                )
            seen_pages.add(next_page)
            page = next_page
        return {"data": events, "pages": page_count}

    @staticmethod
    def _last_answer(events: Sequence[Mapping[str, Any]]) -> str:
        answers: list[str] = []
        for event in events:
            if event.get("type") != "agent.message":
                continue
            # Child-thread events can be cross-posted to the primary history with
            # session_thread_id.  The session answer is the latest buffered message
            # from the primary thread, not a child preview or child-stream message.
            if event.get("session_thread_id") not in {None, ""}:
                continue
            content = event.get("content")
            if not isinstance(content, list):
                continue
            text = "".join(
                str(block["text"])
                for block in content
                if isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
            if text:
                answers.append(text)
        if not answers:
            raise ProviderError("Managed Agents session emitted no agent.message text")
        return answers[-1]

    @staticmethod
    def _idle_reason(events: Sequence[Mapping[str, Any]]) -> Optional[str]:
        for event in reversed(events):
            if event.get("type") != "session.status_idle":
                continue
            reason = event.get("stop_reason")
            if isinstance(reason, str):
                return reason
            if isinstance(reason, Mapping) and isinstance(reason.get("type"), str):
                return str(reason["type"])
        return None

    async def _cleanup_session(self, session_id: str) -> None:
        if self.cleanup == "retain":
            return
        if self.cleanup == "archive":
            await _request_json(
                "POST",
                f"{self.base_url}/sessions/{session_id}/archive",
                headers=self._headers,
                timeout_seconds=self.timeout_seconds,
                payload={},
            )
            return
        await _request_json(
            "DELETE",
            f"{self.base_url}/sessions/{session_id}",
            headers=self._headers,
            timeout_seconds=self.timeout_seconds,
        )

    async def _interrupt_session(self, session_id: str) -> None:
        await _request_json(
            "POST",
            f"{self.base_url}/sessions/{session_id}/events",
            headers=self._headers,
            timeout_seconds=min(self.timeout_seconds, 10.0),
            payload={"events": [{"type": "user.interrupt"}]},
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("anthropic_managed_agents") is not True:
            raise ProviderError("Managed Agents backend requires its dedicated harness")
        if request.tools or request.tool_results or request.continuation is not None:
            raise ProviderError(
                "Managed Agents owns its tool loop; local tools are not accepted"
            )
        if request.system:
            raise ProviderError(
                "Managed Agents system configuration must be pinned on the agent"
            )
        if request.max_output_tokens is not None:
            raise ProviderError(
                "Managed Agents output limits must be pinned on the agent"
            )
        agent: Any = self.agent_id
        if self.agent_version is not None:
            agent = {
                "type": "agent",
                "id": self.agent_id,
                "version": self.agent_version,
            }
        payload: dict[str, Any] = {
            "agent": agent,
            "environment_id": self.environment_id,
            "initial_events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": request.prompt}],
                }
            ],
            "resources": list(self.resources),
            "vault_ids": list(self.vault_ids),
            "metadata": {
                "mini_agent_id": request.agent_id,
                "mini_agent_role": request.role,
            },
        }
        if self.budget_cents is not None:
            payload["budget"] = {
                "type": "limit",
                "max_list_cost": {
                    "amount": str(self.budget_cents),
                    "currency": "USD",
                },
            }
        started = time.perf_counter()
        session = await _request_json(
            "POST",
            f"{self.base_url}/sessions",
            headers=self._headers,
            timeout_seconds=self.timeout_seconds,
            payload=payload,
        )
        session_id = session.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise ProviderError(
                "Managed Agents create response omitted session id", raw=session
            )
        cleanup_error: Optional[Exception] = None
        try:
            _, resolved_agent_sha256, coordinator_roster_size = (
                self._validated_session_snapshot(session)
            )
            deadline = _monotonic() + self.timeout_seconds
            while session.get("status") in {"running", "rescheduling"}:
                raw_usage = session.get("usage")
                if raw_usage is not None and request.usage_reporter is not None:
                    request.usage_reporter(self._incomplete(self._usage(raw_usage)))
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    raise ProviderError(
                        "Managed Agents session polling timed out",
                        usage=self._incomplete(self._usage(raw_usage)),
                        raw=session,
                    )
                await asyncio.sleep(min(self.poll_interval_seconds, remaining))
                session = await _request_json(
                    "GET",
                    f"{self.base_url}/sessions/{session_id}",
                    headers=self._headers,
                    timeout_seconds=min(self.timeout_seconds, remaining),
                )
                _, current_snapshot_sha256, current_roster_size = (
                    self._validated_session_snapshot(session)
                )
                if current_snapshot_sha256 != resolved_agent_sha256:
                    raise ProviderError(
                        "Managed Agents resolved agent snapshot changed during session",
                        raw=session,
                    )
                if current_roster_size != coordinator_roster_size:
                    raise ProviderError(
                        "Managed Agents coordinator roster changed during session",
                        raw=session,
                    )
            usage = self._usage(session.get("usage"))
            if request.usage_reporter is not None:
                request.usage_reporter(usage)
            events_raw = await self._load_events(session_id)
            events = self._event_list(events_raw)
            reason = self._idle_reason(events)
            if session.get("status") != "idle" or reason != "end_turn":
                raise ProviderError(
                    "Managed Agents session did not finish with idle/end_turn: "
                    f"status={session.get('status')!r}, reason={reason!r}",
                    usage=self._incomplete(usage),
                    raw={"session": session, "events": events_raw},
                )
            answer = self._last_answer(events)
            response = ModelResponse(
                text=answer,
                usage=usage,
                provider_latency_seconds=time.perf_counter() - started,
                raw={
                    "session": session,
                    "events": events_raw,
                    "session_id": session_id,
                    "cleanup": self.cleanup,
                    "resolved_agent_sha256": resolved_agent_sha256,
                    "coordinator_roster_size": coordinator_roster_size,
                },
            )
        except (asyncio.CancelledError, Exception):
            if session.get("status") in {"running", "rescheduling"}:
                raw_usage = session.get("usage")
                if raw_usage is not None and request.usage_reporter is not None:
                    try:
                        request.usage_reporter(self._incomplete(self._usage(raw_usage)))
                    except ProviderError:
                        pass
                try:
                    await asyncio.shield(self._interrupt_session(session_id))
                except BaseException:
                    # Preserve the original failure/cancellation.  The remote
                    # session remains discoverable because cleanup is skipped while
                    # it may still be running.
                    pass
            raise
        finally:
            if session.get("status") not in {"running", "rescheduling"}:
                try:
                    await asyncio.shield(self._cleanup_session(session_id))
                except Exception as exc:
                    cleanup_error = exc
        if cleanup_error is not None:
            raise ProviderError(
                f"Managed Agents session cleanup failed: {cleanup_error}",
                usage=response.usage if "response" in locals() else None,
                raw={"session_id": session_id, "cleanup": self.cleanup},
            ) from cleanup_error
        return response
