"""Small provider codecs for the model boundary; no provider owns the loop."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping
from urllib.parse import urlsplit

import httpx

from ._http import ResponseBodyTooLarge, read_bounded_body
from .specs import TranslationLoss
from .types import (
    InfrastructureError,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    Usage,
    _json_mapping,
    strict_json_loads,
)


MAX_PROVIDER_RESPONSE_BYTES = 16 * 1024 * 1024
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ProviderError(InfrastructureError):
    def __init__(
        self,
        message: str,
        *,
        usage: Usage | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.attempts = attempts


def _validate_endpoint(base_url: str, timeout_seconds: float) -> None:
    parsed = urlsplit(base_url)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("provider base_url has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("provider base_url must be an HTTP(S) URL without credentials")
    hostname = parsed.hostname.casefold().rstrip(".")
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost"
    if parsed.scheme == "http" and not loopback:
        raise ValueError("non-loopback provider base_url values require HTTPS")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("provider timeout_seconds must be finite and positive")


def _validate_api_key_env(value: Any) -> str:
    if not isinstance(value, str) or _ENVIRONMENT_NAME.fullmatch(value) is None:
        raise ValueError(
            "api_key_env must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return value


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.input_per_million,
            self.output_per_million,
            self.cache_read_per_million,
            self.cache_write_per_million,
        )
        if any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            )
            for value in values
        ):
            raise ValueError("token prices must be finite and non-negative")

    def cost(self, usage: Usage) -> float:
        uncached = (
            usage.input_tokens
            - usage.cache_read_input_tokens
            - usage.cache_write_input_tokens
        )
        input_cost = uncached * self.input_per_million
        cache_read_price = (
            self.input_per_million
            if self.cache_read_per_million is None
            else self.cache_read_per_million
        )
        cache_write_price = (
            self.input_per_million
            if self.cache_write_per_million is None
            else self.cache_write_per_million
        )
        cache_read = usage.cache_read_input_tokens * cache_read_price
        cache_write = usage.cache_write_input_tokens * cache_write_price
        total = (
            input_cost
            + cache_read
            + cache_write
            + usage.output_tokens * self.output_per_million
        )
        return total / 1_000_000


@dataclass(frozen=True)
class _BackendConfig:
    model: str
    api_key: str | None
    api_key_env: str
    base_url: str
    timeout_seconds: float
    pricing: TokenPricing | None
    default_body: Mapping[str, Any]
    default_headers: Mapping[str, str]
    max_retries: int


def _validated_backend_config(
    *,
    model: str,
    api_key: str | None,
    api_key_env: str,
    base_url: str,
    timeout_seconds: float,
    pricing: TokenPricing | None,
    default_body: Mapping[str, Any] | None,
    default_headers: Mapping[str, str] | None,
    allowed_headers: set[str],
    reserved_body_fields: set[str],
    protocol_label: str,
    max_retries: int,
) -> _BackendConfig:
    """Validate the constructor surface shared by every backend codec."""

    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("model must be a non-empty string")
    _validate_api_key_env(api_key_env)
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string or None")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("model and base_url must be non-empty")
    _validate_endpoint(base_url, timeout_seconds)
    if pricing is not None and not isinstance(pricing, TokenPricing):
        raise ValueError("pricing must be TokenPricing or None")
    if default_body is not None and not isinstance(default_body, Mapping):
        raise ValueError("default_body must be an object or None")
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or not 0 <= max_retries <= 10
    ):
        raise ValueError("max_retries must be an integer between 0 and 10")
    body = _json_mapping(default_body or {}, "provider default_body")
    headers = _validate_headers(default_headers, allowed_headers)
    _reject_reserved_body_fields(body, reserved_body_fields)
    _require_non_streaming(body, protocol_label)
    return _BackendConfig(
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url.rstrip("/"),
        timeout_seconds=timeout_seconds,
        pricing=pricing,
        default_body=body,
        default_headers=headers,
        max_retries=max_retries,
    )


def _credential(api_key: str | None, api_key_env: str) -> str:
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise ProviderError(f"missing credential in {api_key_env}")
    return key


class OpenAIResponsesBackend:
    """OpenAI Responses adapter using generic function tools.

    ``provider`` names the deployment for provenance and error reporting; a
    Meta deployment exposing this protocol is selected by ``build_model`` with
    ``provider="meta"`` and an explicit ``base_url``.
    """

    translation_losses: ClassVar[tuple[TranslationLoss, ...]] = (
        TranslationLoss(
            field="tool_result_is_error",
            kind="dropped",
            reason=(
                "Responses function_call_output items carry no error flag; "
                "error results are delivered as plain output text."
            ),
        ),
        TranslationLoss(
            field="tool_result_images",
            kind="approximated",
            reason=(
                "Tool-result images are re-sent as a separate synthetic user "
                "message because function_call_output cannot carry images."
            ),
        ),
        TranslationLoss(
            field="tool_kind",
            kind="unsupported",
            reason=(
                "Only generic function tools are encoded; provider-native "
                "tool kinds are rejected."
            ),
        ),
    )

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 300,
        max_retries: int = 3,
        pricing: TokenPricing | None = None,
        default_body: Mapping[str, Any] | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        self.provider = provider
        config = _validated_backend_config(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            pricing=pricing,
            default_body=default_body,
            default_headers=default_headers,
            allowed_headers={"authorization", "content-type"},
            reserved_body_fields={
                "conversation",
                "input",
                "instructions",
                "max_output_tokens",
                "model",
                "previous_response_id",
                "tools",
            },
            protocol_label="Responses",
            max_retries=max_retries,
        )
        self.model = config.model
        self.api_key = config.api_key
        self.api_key_env = config.api_key_env
        self.base_url = config.base_url
        self.timeout_seconds = config.timeout_seconds
        self.pricing = config.pricing
        self.default_body = config.default_body
        self.default_headers = config.default_headers
        self.max_retries = config.max_retries
        store = self.default_body.get("store")
        if "store" in self.default_body and not isinstance(store, bool):
            raise ValueError("provider body store must be boolean")
        if store is False:
            raise ValueError(
                "the Responses adapter uses previous_response_id continuation and "
                "does not support store=false"
            )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            **self.default_body,
            "model": self.model,
            "input": _openai_input(request),
        }
        if request.system:
            payload["instructions"] = request.system
        if request.tools:
            payload["tools"] = [_openai_tool(tool) for tool in request.tools]
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.continuation is not None:
            if not isinstance(request.continuation, str):
                raise ProviderError("OpenAI continuation must be a response id")
            payload["previous_response_id"] = request.continuation
        attempt_counter: list[int] = []
        data = await _post_json(
            f"{self.base_url}/responses",
            headers={
                **self.default_headers,
                "Authorization": (
                    f"Bearer {_credential(self.api_key, self.api_key_env)}"
                ),
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            attempt_counter=attempt_counter,
        )
        usage = _openai_usage(data.get("usage"), self.pricing)
        if data.get("status") != "completed":
            raise ProviderError(
                f"{self.provider} response status is {data.get('status')!r}",
                usage=usage,
            )
        text: list[str] = []
        calls: list[ToolCall] = []
        output = data.get("output", [])
        if not isinstance(output, list):
            raise ProviderError(
                f"{self.provider} response output must be a list", usage=usage
            )
        for item in output:
            if not isinstance(item, Mapping):
                raise ProviderError(
                    f"{self.provider} response output items must be objects",
                    usage=usage,
                )
            if item.get("type") == "message":
                if item.get("role") != "assistant":
                    raise ProviderError(
                        f"{self.provider} output message role must be assistant",
                        usage=usage,
                    )
                content = item.get("content", [])
                if not isinstance(content, list):
                    raise ProviderError(
                        f"{self.provider} message content must be a list", usage=usage
                    )
                for block in content:
                    if not isinstance(block, Mapping):
                        raise ProviderError(
                            f"{self.provider} message content items must be objects",
                            usage=usage,
                        )
                    if block.get("type") in {"output_text", "text"}:
                        value = block.get("text")
                        if not isinstance(value, str):
                            raise ProviderError(
                                f"{self.provider} text content must be a string",
                                usage=usage,
                            )
                        text.append(value)
                    if block.get("type") == "refusal":
                        refusal = block.get("refusal")
                        if not isinstance(refusal, str):
                            raise ProviderError(
                                f"{self.provider} refusal content must be a string",
                                usage=usage,
                            )
                        raise ProviderError(
                            f"{self.provider} refused the request", usage=usage
                        )
            if item.get("type") in {"function_call", "tool_call"}:
                name = item.get("name")
                call_id = item.get("call_id", item.get("id"))
                if not isinstance(name, str) or not isinstance(call_id, str):
                    raise ProviderError(
                        f"{self.provider} returned a malformed tool call", usage=usage
                    )
                try:
                    arguments = _arguments(item.get("arguments", {}), self.provider)
                except ProviderError as exc:
                    raise ProviderError(str(exc), usage=usage) from exc
                try:
                    calls.append(
                        ToolCall(call_id=call_id, name=name, arguments=arguments)
                    )
                except ValueError as exc:
                    raise ProviderError(
                        f"{self.provider} returned a malformed tool call", usage=usage
                    ) from exc
        if len({call.call_id for call in calls}) != len(calls):
            raise ProviderError(
                f"{self.provider} returned duplicate tool call ids", usage=usage
            )
        response_id = data.get("id")
        if calls and not isinstance(response_id, str):
            raise ProviderError(f"{self.provider} tool response has no id", usage=usage)
        return ModelResponse(
            text="\n".join(part for part in text if part).strip(),
            usage=usage,
            tool_calls=tuple(calls),
            continuation=response_id if calls else None,
            resolved_model=_response_model(data, self.provider, usage),
            retries=attempt_counter[0] - 1 if attempt_counter else 0,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "protocol": "responses",
            "model": self.model,
            "base_url": self.base_url,
            "continuation": "previous_response_id",
            "max_response_bytes": MAX_PROVIDER_RESPONSE_BYTES,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


class ChatCompletionsBackend:
    """OpenAI-compatible Chat Completions adapter with full-transcript replay.

    Unlike the Responses adapter there is no server-side continuation state:
    the full wire transcript is resent on every call, mirroring the Anthropic
    codec. The request body carries only harness-owned fields; sampling
    parameters are never set so the server's defaults always apply.
    """

    translation_losses: ClassVar[tuple[TranslationLoss, ...]] = (
        TranslationLoss(
            field="tool_result_is_error",
            kind="dropped",
            reason=(
                "Chat Completions tool messages carry no error flag; error "
                "results are delivered as plain content text."
            ),
        ),
        TranslationLoss(
            field="tool_result_images",
            kind="approximated",
            reason=(
                "Tool-result images are re-sent as a separate synthetic user "
                "message because the tool role cannot carry images."
            ),
        ),
        TranslationLoss(
            field="tool_kind",
            kind="unsupported",
            reason=(
                "Only generic function tools are encoded; provider-native "
                "tool kinds are rejected."
            ),
        ),
        TranslationLoss(
            field="tool_result_image_history",
            kind="approximated",
            reason=(
                "Replayed transcript images beyond the configured "
                "max_history_images window are replaced with a fixed text "
                "placeholder; the current step's images are always sent."
            ),
        ),
    )

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 300,
        max_retries: int = 3,
        max_history_images: int | None = 4,
        pricing: TokenPricing | None = None,
        default_body: Mapping[str, Any] | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        self.provider = provider
        config = _validated_backend_config(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            pricing=pricing,
            default_body=default_body,
            default_headers=default_headers,
            allowed_headers={"authorization", "content-type"},
            reserved_body_fields={
                "max_completion_tokens",
                "messages",
                "model",
                "tools",
            },
            protocol_label="Chat Completions",
            max_retries=max_retries,
        )
        self.model = config.model
        self.api_key = config.api_key
        self.api_key_env = config.api_key_env
        self.base_url = config.base_url
        self.timeout_seconds = config.timeout_seconds
        self.pricing = config.pricing
        self.default_body = config.default_body
        self.default_headers = config.default_headers
        self.max_retries = config.max_retries
        self.max_history_images = _validated_history_images(
            max_history_images
        )
        choices = self.default_body.get("n", 1)
        if not isinstance(choices, int) or isinstance(choices, bool) or choices != 1:
            raise ValueError(
                "the Chat Completions adapter consumes exactly one choice; n must be 1"
            )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        transcript = _chat_transcript(request, self.max_history_images)
        messages: list[Mapping[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(transcript)
        payload: dict[str, Any] = {
            **self.default_body,
            "model": self.model,
            "messages": messages,
        }
        if request.tools:
            payload["tools"] = [_chat_tool(tool) for tool in request.tools]
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        attempt_counter: list[int] = []
        data = await _post_json(
            f"{self.base_url}/chat/completions",
            headers={
                **self.default_headers,
                "Authorization": (
                    f"Bearer {_credential(self.api_key, self.api_key_env)}"
                ),
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            attempt_counter=attempt_counter,
        )
        usage = _chat_usage(data.get("usage"), self.pricing)
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderError(
                f"{self.provider} response choices must contain exactly one item",
                usage=usage,
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderError(
                f"{self.provider} response choices must be objects", usage=usage
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderError(
                f"{self.provider} choice message must be an object", usage=usage
            )
        if message.get("role") != "assistant":
            raise ProviderError(
                f"{self.provider} choice message role must be assistant",
                usage=usage,
            )
        refusal = message.get("refusal")
        if refusal is not None:
            if not isinstance(refusal, str):
                raise ProviderError(
                    f"{self.provider} message refusal must be a string or null",
                    usage=usage,
                )
            raise ProviderError(f"{self.provider} refused the request", usage=usage)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderError(
                f"{self.provider} message content must be a string or null",
                usage=usage,
            )
        raw_calls = message.get("tool_calls")
        if raw_calls is not None and not isinstance(raw_calls, list):
            raise ProviderError(
                f"{self.provider} message tool_calls must be a list", usage=usage
            )
        calls: list[ToolCall] = []
        for item in raw_calls or []:
            if not isinstance(item, Mapping) or item.get("type") not in {
                None,
                "function",
            }:
                raise ProviderError(
                    f"{self.provider} returned a malformed tool call", usage=usage
                )
            call_id = item.get("id")
            function = item.get("function")
            if not isinstance(call_id, str) or not isinstance(function, Mapping):
                raise ProviderError(
                    f"{self.provider} returned a malformed tool call", usage=usage
                )
            name = function.get("name")
            if not isinstance(name, str):
                raise ProviderError(
                    f"{self.provider} returned a malformed tool call", usage=usage
                )
            try:
                arguments = _arguments(function.get("arguments", {}), self.provider)
            except ProviderError as exc:
                raise ProviderError(str(exc), usage=usage) from exc
            try:
                calls.append(
                    ToolCall(call_id=call_id, name=name, arguments=arguments)
                )
            except ValueError as exc:
                raise ProviderError(
                    f"{self.provider} returned a malformed tool call", usage=usage
                ) from exc
        if len({call.call_id for call in calls}) != len(calls):
            raise ProviderError(
                f"{self.provider} returned duplicate tool call ids", usage=usage
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "function_call",
        }:
            raise ProviderError(
                f"{self.provider} returned an unsupported finish_reason",
                usage=usage,
            )
        if finish_reason == "length":
            raise ProviderError(
                f"{self.provider} exhausted max tokens before finishing",
                usage=usage,
            )
        if finish_reason == "content_filter":
            raise ProviderError(
                f"{self.provider} filtered the completion", usage=usage
            )
        if calls and finish_reason not in {"tool_calls", "function_call"}:
            raise ProviderError(
                f"{self.provider} returned tool calls with an inconsistent "
                "finish_reason",
                usage=usage,
            )
        if not calls and finish_reason in {"tool_calls", "function_call"}:
            raise ProviderError(
                f"{self.provider} returned a tool finish_reason without tool calls",
                usage=usage,
            )
        continuation = (*transcript, dict(message)) if calls else None
        return ModelResponse(
            text=(content or "").strip(),
            usage=usage,
            tool_calls=tuple(calls),
            continuation=continuation,
            resolved_model=_response_model(data, self.provider, usage),
            retries=attempt_counter[0] - 1 if attempt_counter else 0,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "protocol": "chat-completions",
            "model": self.model,
            "base_url": self.base_url,
            "continuation": "full-transcript",
            "max_history_images": self.max_history_images,
            "max_response_bytes": MAX_PROVIDER_RESPONSE_BYTES,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


class AnthropicMessagesBackend:
    """Anthropic Messages adapter using generic function tools."""

    provider = "anthropic"

    translation_losses: ClassVar[tuple[TranslationLoss, ...]] = (
        TranslationLoss(
            field="tool_kind",
            kind="unsupported",
            reason=(
                "Only generic function tools are encoded; provider-native "
                "tool kinds are rejected."
            ),
        ),
        TranslationLoss(
            field="tool_result_image_history",
            kind="approximated",
            reason=(
                "Replayed transcript images beyond the configured "
                "max_history_images window are replaced with a fixed text "
                "placeholder; the current step's images are always sent."
            ),
        ),
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        timeout_seconds: float = 300,
        max_retries: int = 3,
        max_history_images: int | None = 4,
        pricing: TokenPricing | None = None,
        default_body: Mapping[str, Any] | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(api_version, str) or not api_version.strip():
            raise ValueError("api_version must be a non-empty string")
        self.api_version = api_version
        config = _validated_backend_config(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            pricing=pricing,
            default_body=default_body,
            default_headers=default_headers,
            allowed_headers={"x-api-key", "anthropic-version", "content-type"},
            reserved_body_fields={
                "max_tokens",
                "messages",
                "model",
                "system",
                "tools",
            },
            protocol_label="Anthropic Messages",
            max_retries=max_retries,
        )
        self.model = config.model
        self.api_key = config.api_key
        self.api_key_env = config.api_key_env
        self.base_url = config.base_url
        self.timeout_seconds = config.timeout_seconds
        self.pricing = config.pricing
        self.default_body = config.default_body
        self.default_headers = config.default_headers
        self.max_retries = config.max_retries
        self.max_history_images = _validated_history_images(
            max_history_images
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        messages = _anthropic_messages(request, self.max_history_images)
        payload: dict[str, Any] = {
            **self.default_body,
            "model": self.model,
            "max_tokens": request.max_output_tokens or 8192,
            "messages": messages,
        }
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [_anthropic_tool(tool) for tool in request.tools]
        attempt_counter: list[int] = []
        data = await _post_json(
            f"{self.base_url}/messages",
            headers={
                **self.default_headers,
                "x-api-key": _credential(self.api_key, self.api_key_env),
                "anthropic-version": self.api_version,
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            attempt_counter=attempt_counter,
        )
        usage = _anthropic_usage(data.get("usage"), self.pricing)
        if data.get("type") != "message" or data.get("role") != "assistant":
            raise ProviderError(
                "Anthropic response must be an assistant message", usage=usage
            )
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError(
                "Anthropic response content must be a list", usage=usage
            )
        text: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, Mapping):
                raise ProviderError(
                    "Anthropic response content items must be objects", usage=usage
                )
            if block.get("type") == "text":
                if not isinstance(block.get("text"), str):
                    raise ProviderError(
                        "Anthropic text content must be a string", usage=usage
                    )
                text.append(block["text"])
            if block.get("type") == "tool_use":
                call_id = block.get("id")
                name = block.get("name")
                arguments = block.get("input", {})
                if (
                    not isinstance(call_id, str)
                    or not isinstance(name, str)
                    or not isinstance(arguments, Mapping)
                ):
                    raise ProviderError(
                        "Anthropic returned a malformed tool call", usage=usage
                    )
                try:
                    calls.append(ToolCall(call_id, name, dict(arguments)))
                except ValueError as exc:
                    raise ProviderError(
                        "Anthropic returned a malformed tool call", usage=usage
                    ) from exc
        if len({call.call_id for call in calls}) != len(calls):
            raise ProviderError(
                "Anthropic returned duplicate tool call ids", usage=usage
            )
        stop_reason = data.get("stop_reason")
        if stop_reason not in {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        }:
            raise ProviderError(
                "Anthropic returned an unsupported stop_reason", usage=usage
            )
        if stop_reason == "max_tokens":
            raise ProviderError(
                "Anthropic exhausted max_tokens before finishing", usage=usage
            )
        if stop_reason == "model_context_window_exceeded":
            raise ProviderError(
                "Anthropic exhausted the model context window before finishing",
                usage=usage,
            )
        if stop_reason == "refusal":
            raise ProviderError("Anthropic refused the request", usage=usage)
        if stop_reason == "pause_turn":
            raise ProviderError(
                "Anthropic returned pause_turn, which requires an unsupported "
                "server-tool continuation",
                usage=usage,
            )
        if calls and stop_reason != "tool_use":
            raise ProviderError(
                "Anthropic returned tool calls with an inconsistent stop_reason",
                usage=usage,
            )
        if not calls and stop_reason == "tool_use":
            raise ProviderError(
                "Anthropic returned tool_use without client tool calls", usage=usage
            )
        continuation = (
            (*messages, {"role": "assistant", "content": content}) if calls else None
        )
        return ModelResponse(
            text="\n".join(part for part in text if part).strip(),
            usage=usage,
            tool_calls=tuple(calls),
            continuation=continuation,
            resolved_model=_response_model(data, self.provider, usage),
            retries=attempt_counter[0] - 1 if attempt_counter else 0,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "protocol": "messages",
            "model": self.model,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "max_history_images": self.max_history_images,
            "max_response_bytes": MAX_PROVIDER_RESPONSE_BYTES,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def _openai_input(request: ModelRequest) -> list[Mapping[str, Any]]:
    if request.continuation is None:
        content: list[Mapping[str, Any]] = []
        if request.prompt:
            content.append({"type": "input_text", "text": request.prompt})
        content.extend(
            {"type": "input_image", "image_url": image}
            for image in request.input_images
        )
        return [
            {"role": "user", "content": content or [{"type": "input_text", "text": ""}]}
        ]
    items: list[Mapping[str, Any]] = []
    images: list[Mapping[str, Any]] = []
    for result in request.tool_results:
        items.append(
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": result.output,
            }
        )
        if result.image_data_url:
            images.append({"type": "input_image", "image_url": result.image_data_url})
    if images:
        items.append({"role": "user", "content": images})
    return items


def _openai_tool(tool: ToolDefinition) -> Mapping[str, Any]:
    if tool.kind != "function":
        raise ProviderError(
            f"OpenAI adapter supports function tools, not {tool.kind!r}"
        )
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.input_schema),
    }


_ELIDED_IMAGE_TEXT = "[earlier screenshot elided]"


def _validated_history_images(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            "max_history_images must be a non-negative integer or None"
        )
    return value


def _evicted_flags(
    image_message_indices: list[int], boundary: int, limit: int
) -> list[bool]:
    """Per image block (document order): True when it must be elided.

    Blocks in messages at or past ``boundary`` belong to the current call and
    are never elided; the newest of the remaining blocks fill whatever budget
    the current call's images leave.
    """

    protected = [index >= boundary for index in image_message_indices]
    budget = max(0, limit - sum(protected))
    kept: set[int] = set()
    for position in range(len(image_message_indices) - 1, -1, -1):
        if protected[position] or budget <= 0:
            continue
        kept.add(position)
        budget -= 1
    return [
        not protected[position] and position not in kept
        for position in range(len(image_message_indices))
    ]


def _bound_chat_images(
    items: list[Mapping[str, Any]], boundary: int, limit: int | None
) -> list[Mapping[str, Any]]:
    if limit is None:
        return items
    positions: list[int] = []
    for index, message in enumerate(items):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            positions.extend(
                index
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "image_url"
            )
    evict = _evicted_flags(positions, boundary, limit)
    if not any(evict):
        return items
    cursor = 0
    bounded: list[Mapping[str, Any]] = []
    for message in items:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            bounded.append(message)
            continue
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "image_url":
                blocks.append(
                    {"type": "text", "text": _ELIDED_IMAGE_TEXT}
                    if evict[cursor]
                    else block
                )
                cursor += 1
            else:
                blocks.append(block)
        bounded.append({**message, "content": blocks})
    return bounded


def _bound_anthropic_images(
    messages: list[Mapping[str, Any]], boundary: int, limit: int | None
) -> list[Mapping[str, Any]]:
    if limit is None:
        return messages
    positions: list[int] = []

    def scan(blocks: list[Any], index: int) -> None:
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "image":
                positions.append(index)
            elif block.get("type") == "tool_result" and isinstance(
                block.get("content"), list
            ):
                scan(block["content"], index)

    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            scan(content, index)
    evict = _evicted_flags(positions, boundary, limit)
    if not any(evict):
        return messages
    cursor = 0

    def rebuild(blocks: list[Any]) -> list[Any]:
        nonlocal cursor
        rebuilt: list[Any] = []
        for block in blocks:
            if isinstance(block, Mapping) and block.get("type") == "image":
                rebuilt.append(
                    {"type": "text", "text": _ELIDED_IMAGE_TEXT}
                    if evict[cursor]
                    else block
                )
                cursor += 1
            elif isinstance(block, Mapping) and block.get(
                "type"
            ) == "tool_result" and isinstance(block.get("content"), list):
                rebuilt.append({**block, "content": rebuild(block["content"])})
            else:
                rebuilt.append(block)
        return rebuilt

    bounded: list[Mapping[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            bounded.append({**message, "content": rebuild(list(content))})
        else:
            bounded.append(message)
    return bounded


def _chat_transcript(
    request: ModelRequest, max_history_images: int | None
) -> list[Mapping[str, Any]]:
    if request.continuation is None:
        if request.input_images:
            content: list[Mapping[str, Any]] = []
            if request.prompt:
                content.append({"type": "text", "text": request.prompt})
            content.extend(
                {"type": "image_url", "image_url": {"url": image}}
                for image in request.input_images
            )
            return [{"role": "user", "content": content}]
        return [{"role": "user", "content": request.prompt}]
    if not isinstance(request.continuation, (tuple, list)):
        raise ProviderError(
            "chat completions continuation must be a message sequence"
        )
    boundary = len(request.continuation)
    items: list[Mapping[str, Any]] = list(request.continuation)
    images: list[Mapping[str, Any]] = []
    for result in request.tool_results:
        items.append(
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.output,
            }
        )
        if result.image_data_url:
            images.append(
                {"type": "image_url", "image_url": {"url": result.image_data_url}}
            )
    if images:
        # The tool role cannot carry images in this protocol.
        items.append({"role": "user", "content": images})
    return _bound_chat_images(items, boundary, max_history_images)


def _chat_tool(tool: ToolDefinition) -> Mapping[str, Any]:
    if tool.kind != "function":
        raise ProviderError(
            f"the Chat Completions adapter supports function tools, "
            f"not {tool.kind!r}"
        )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _chat_usage(value: Any, pricing: TokenPricing | None) -> Usage:
    if not isinstance(value, Mapping):
        return Usage(cost_known=False, complete=False)
    details = value.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
    return _priced_usage(
        _usage_int(value.get("prompt_tokens", 0) or 0, "prompt_tokens"),
        _usage_int(value.get("completion_tokens", 0) or 0, "completion_tokens"),
        _usage_int(cached or 0, "cached_tokens"),
        0,
        pricing,
        complete="prompt_tokens" in value and "completion_tokens" in value,
    )


def _anthropic_messages(
    request: ModelRequest, max_history_images: int | None
) -> list[Mapping[str, Any]]:
    if request.continuation is None:
        content: list[Mapping[str, Any]] = []
        if request.prompt:
            content.append({"type": "text", "text": request.prompt})
        content.extend(_anthropic_image(image) for image in request.input_images)
        return [{"role": "user", "content": content or [{"type": "text", "text": ""}]}]
    if not isinstance(request.continuation, (tuple, list)):
        raise ProviderError("Anthropic continuation must be a message sequence")
    boundary = len(request.continuation)
    results: list[Mapping[str, Any]] = []
    for result in request.tool_results:
        result_content: list[Mapping[str, Any]] = [
            {"type": "text", "text": result.output}
        ]
        if result.image_data_url:
            result_content.append(_anthropic_image(result.image_data_url))
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "is_error": result.is_error,
                "content": result_content,
            }
        )
    messages = [*request.continuation, {"role": "user", "content": results}]
    return _bound_anthropic_images(messages, boundary, max_history_images)


def _anthropic_image(value: str) -> Mapping[str, Any]:
    if not value.startswith("data:image/") or ";base64," not in value:
        raise ProviderError("Anthropic image must be a base64 data URL")
    header, data = value.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ProviderError(f"Anthropic does not support {media_type!r} images")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _anthropic_tool(tool: ToolDefinition) -> Mapping[str, Any]:
    if tool.kind != "function":
        raise ProviderError(
            f"Anthropic adapter supports function tools, not {tool.kind!r}"
        )
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.input_schema),
    }


def _arguments(value: Any, provider: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = strict_json_loads(value)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"{provider} tool arguments are invalid JSON") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ProviderError(f"{provider} tool arguments must be an object")


def _response_model(
    value: Mapping[str, Any], provider: str, usage: Usage
) -> str:
    model = value.get("model")
    if (
        not isinstance(model, str)
        or not model.strip()
        or model != model.strip()
    ):
        raise ProviderError(
            f"{provider} response model must be a non-empty string", usage=usage
        )
    return model


def _validate_headers(
    headers: Mapping[str, str] | None, reserved: set[str]
) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise ValueError("provider default_headers must be a mapping or None")
    validated: dict[str, str] = {}
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError(
                "provider headers must map non-empty names to non-empty values"
            )
        if any(character in "\r\n\x00" for character in name + value):
            raise ValueError("provider headers cannot contain control characters")
        if name.lower() in reserved:
            raise ValueError(
                f"provider header {name!r} is harness-owned; credentials belong "
                "in api_key_env"
            )
        validated[name] = value
    return validated


def _reject_reserved_body_fields(body: Mapping[str, Any], reserved: set[str]) -> None:
    overlap = sorted(set(body).intersection(reserved))
    if overlap:
        raise ValueError(
            f"provider body cannot override harness-owned fields {overlap}"
        )


def _require_non_streaming(body: Mapping[str, Any], protocol: str) -> None:
    stream = body.get("stream")
    if "stream" in body and stream is not False:
        raise ValueError(
            f"the {protocol} adapter is single-shot and does not support streaming"
        )


def _openai_usage(value: Any, pricing: TokenPricing | None) -> Usage:
    if not isinstance(value, Mapping):
        return Usage(cost_known=False, complete=False)
    details = value.get("input_tokens_details", {})
    cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
    written = (
        details.get("cache_write_tokens", 0) if isinstance(details, Mapping) else 0
    )
    return _priced_usage(
        _usage_int(value.get("input_tokens", 0), "input_tokens"),
        _usage_int(value.get("output_tokens", 0), "output_tokens"),
        _usage_int(cached or 0, "cached_tokens"),
        _usage_int(written or 0, "cache_write_tokens"),
        pricing,
        complete="input_tokens" in value and "output_tokens" in value,
    )


def _anthropic_usage(value: Any, pricing: TokenPricing | None) -> Usage:
    if not isinstance(value, Mapping):
        return Usage(cost_known=False, complete=False)
    cached = _usage_int(
        value.get("cache_read_input_tokens", 0) or 0,
        "cache_read_input_tokens",
    )
    written = _usage_int(
        value.get("cache_creation_input_tokens", 0) or 0,
        "cache_creation_input_tokens",
    )
    input_tokens = (
        _usage_int(value.get("input_tokens", 0) or 0, "input_tokens") + cached + written
    )
    return _priced_usage(
        input_tokens,
        _usage_int(value.get("output_tokens", 0) or 0, "output_tokens"),
        cached,
        written,
        pricing,
        complete="input_tokens" in value and "output_tokens" in value,
    )


def _usage_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderError(f"provider usage {name} must be a non-negative integer")
    return value


def _priced_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
    pricing: TokenPricing | None,
    *,
    complete: bool,
) -> Usage:
    try:
        base = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            cost_known=pricing is not None and complete,
            complete=complete,
        )
    except ValueError as exc:
        raise ProviderError(f"provider returned inconsistent usage: {exc}") from exc
    if pricing is None:
        return base
    if not complete:
        return base
    try:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            cost_usd=pricing.cost(base),
        )
    except ValueError as exc:
        raise ProviderError(f"provider returned inconsistent usage: {exc}") from exc


_ERROR_BODY_BYTES = 2048
_SAFE_ERROR_TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


async def _bounded_error_body(response: Any) -> bytes:
    collected = bytearray()
    try:
        async for chunk in response.aiter_bytes(chunk_size=1024):
            if not isinstance(chunk, bytes):
                return b""
            collected.extend(chunk)
            if len(collected) > _ERROR_BODY_BYTES:
                return b""
    except Exception:
        return b""
    return bytes(collected)


def _error_code_detail(body: bytes) -> str:
    """Extract only allowlisted short error codes; never echo free text.

    Untrusted response bodies can echo credentials or hidden benchmark
    content, and ProviderError messages reach durable artifacts, so free-form
    fields such as ``error.message`` are deliberately excluded.
    """

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return ""
    parts = []
    for key in ("type", "code"):
        value = error.get(key)
        if isinstance(value, str) and _SAFE_ERROR_TOKEN.fullmatch(value):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


_RETRYABLE_STATUSES = frozenset({408, 429})
_RETRY_AFTER_CAP_SECONDS = 30.0
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0


async def _retry_sleep(seconds: float) -> None:
    """Patchable backoff seam so tests never sleep for real."""

    await asyncio.sleep(seconds)


def _retry_delay(attempt_index: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            requested = float(retry_after)
        except ValueError:
            requested = -1.0
        if requested >= 0:
            # The server asked for a specific delay; honor it without jitter.
            return min(requested, _RETRY_AFTER_CAP_SECONDS)
    ceiling = min(
        _BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2.0**attempt_index)
    )
    return random.uniform(0.0, ceiling)


async def _post_json(
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
    max_retries: int = 0,
    attempt_counter: list[int] | None = None,
) -> Mapping[str, Any]:
    # A fresh client per request trades connection reuse for lifecycle
    # simplicity: backends have no owner with a close hook, and model calls
    # are minutes-long, so handshake cost is negligible here. Retries cover
    # only pre-parse transport failures and retryable statuses, so a
    # successfully parsed response is never retried and usage can never be
    # charged twice. Backoff sleeps run inside the caller's wall-clock guard.
    attempts = max_retries + 1
    content: bytes | None = None
    for attempt in range(attempts):
        final = attempt == attempts - 1
        retry_after: str | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream(
                    "POST", url, headers=dict(headers), json=dict(payload)
                ) as response:
                    status = response.status_code
                    if not 200 <= status < 300:
                        retryable = (
                            status in _RETRYABLE_STATUSES or 500 <= status < 600
                        )
                        detail = _error_code_detail(
                            await _bounded_error_body(response)
                        )
                        if retryable and not final:
                            retry_after = response.headers.get("retry-after")
                        else:
                            raise ProviderError(
                                f"provider HTTP {status}"
                                + (f" ({detail})" if detail else "")
                                + (
                                    f" after {attempt + 1} attempts"
                                    if attempt
                                    else ""
                                ),
                                attempts=attempt + 1,
                            )
                    else:
                        content = await read_bounded_body(
                            response, MAX_PROVIDER_RESPONSE_BYTES
                        )
        except ResponseBodyTooLarge as exc:
            raise ProviderError(
                "provider response exceeds the "
                f"{MAX_PROVIDER_RESPONSE_BYTES}-byte limit",
                attempts=attempt + 1,
            ) from exc
        except httpx.TransportError as exc:
            if final:
                raise ProviderError(
                    "provider response transport failed"
                    + (f" after {attempt + 1} attempts" if attempt else ""),
                    attempts=attempt + 1,
                ) from exc
            retry_after = None
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise ProviderError(
                "provider response transport failed", attempts=attempt + 1
            ) from exc
        if content is not None:
            break
        await _retry_sleep(_retry_delay(attempt, retry_after))
    assert content is not None  # loop exits only via break or raise
    if attempt_counter is not None:
        attempt_counter.append(attempt + 1)
    try:
        data = strict_json_loads(content)
    except ValueError as exc:
        raise ProviderError(
            "provider returned invalid JSON", attempts=attempt + 1
        ) from exc
    if not isinstance(data, Mapping):
        raise ProviderError(
            "provider response must be an object", attempts=attempt + 1
        )
    return dict(data)


__all__ = [
    "AnthropicMessagesBackend",
    "ChatCompletionsBackend",
    "MAX_PROVIDER_RESPONSE_BYTES",
    "OpenAIResponsesBackend",
    "ProviderError",
    "TokenPricing",
]
