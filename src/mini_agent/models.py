"""Model selection: parse a provider spec, build the backend it names.

The scripted and echo models live here too, so a run needs no API key to
exercise the loop.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from .types import (
    Message,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
    ToolResult,
    _require_positive_int,
    _require_str,
)

if TYPE_CHECKING:
    from .providers import TokenPricing


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
        max_output_tokens: int | None = None,
        expected_resolved_model: str | None = None,
    ) -> None:
        if max_output_tokens is not None:
            _require_positive_int(max_output_tokens, "max_output_tokens")
        if expected_resolved_model is not None:
            _require_str(
                expected_resolved_model, "expected_resolved_model", stripped=True
            )
            if "\x00" in expected_resolved_model:
                raise ValueError("expected_resolved_model must not contain NUL")
        if not callable(getattr(backend, "complete", None)):
            raise ValueError("backend must expose an async complete method")
        self.backend = backend
        self.max_output_tokens = max_output_tokens
        self.expected_resolved_model = expected_resolved_model
        self._continuation: Any = None
        self._resolved_model: str | None = None

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        """Answer from the scripted responses, in order.
        
                The point of this model is that a run can exercise the entire loop --
                tools, budgets, tracing, artifacts -- with no API key and no cost, so
                every benchmark path in this project is testable offline.
                """
        if not any(message.role == "assistant" for message in messages):
            self._continuation = None
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
            # Only the results after the latest assistant action belong to this
            # turn; an assistant message exists here because the continuation
            # token is discarded above when the history has none.
            last_assistant = max(
                index
                for index, message in enumerate(messages)
                if message.role == "assistant"
            )
            tool_results = tuple(
                result
                for message in messages[last_assistant + 1 :]
                if message.role == "tool"
                for result in message.tool_results
            )
            if not tool_results:
                # Invariant relied on by every provider codec: a continued
                # request always carries at least one tool result.
                raise ValueError("continued provider query requires tool results")
        response = await self.backend.complete(
            ModelRequest(
                prompt=prompt,
                system=system,
                max_output_tokens=self.max_output_tokens,
                input_images=images,
                tools=tuple(tools),
                tool_results=tool_results,
                continuation=self._continuation,
            )
        )
        if response.tool_calls and response.continuation is None:
            raise ValueError("provider tool calls require continuation state")
        if response.resolved_model is not None:
            if (
                self.expected_resolved_model is not None
                and response.resolved_model != self.expected_resolved_model
            ):
                from .providers import ProviderError

                raise ProviderError(
                    "provider response model does not match the expected snapshot",
                    usage=response.usage,
                )
            if self._resolved_model is None:
                self._resolved_model = response.resolved_model
            elif self._resolved_model != response.resolved_model:
                from .providers import ProviderError

                raise ProviderError(
                    "provider changed resolved model during one agent run",
                    usage=response.usage,
                )
        self._continuation = response.continuation
        return response

class ScriptedModel:
    """Small deterministic model for offline tests and examples."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        if isinstance(responses, (str, bytes)):
            raise ValueError("scripted responses must be ModelResponse values")
        values = tuple(responses)
        if not all(isinstance(response, ModelResponse) for response in values):
            raise ValueError("scripted responses must be ModelResponse values")
        self.responses = deque(values)
        self.queries: list[tuple[tuple[Message, ...], tuple[ToolDefinition, ...]]] = []

    async def query(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        self.queries.append((tuple(messages), tuple(tools)))
        if not self.responses:
            raise AssertionError("scripted model has no response left")
        return self.responses.popleft()


SUPPORTED_PROVIDERS = frozenset({"anthropic", "meta", "openai"})


def translation_losses_for(
    provider: str, protocol: str | None = None
) -> tuple[Any, ...]:
    """Return the declared translation losses of the codec build_model selects."""

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; choose openai, anthropic, or meta"
        )
    if protocol not in {None, "responses", "chat-completions"}:
        raise ValueError(
            "protocol must be 'responses', 'chat-completions', or None"
        )
    from . import providers

    if provider == "anthropic":
        if protocol is not None:
            raise ValueError("anthropic models always use the Messages protocol")
        return providers.AnthropicMessagesBackend.translation_losses
    if protocol == "chat-completions":
        return providers.ChatCompletionsBackend.translation_losses
    return providers.OpenAIResponsesBackend.translation_losses


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``provider/model`` and validate its shape, not the provider set."""

    if not isinstance(spec, str):
        raise ValueError("model must use provider/model syntax")
    provider, separator, model_name = spec.partition("/")
    if (
        not separator
        or not provider
        or not model_name.strip()
        or model_name != model_name.strip()
        or "\x00" in spec
    ):
        raise ValueError("model must use provider/model syntax")
    return provider, model_name


_UNSET: Any = object()


_PROVIDER_DEFAULTS: Mapping[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "meta": ("", "MODEL_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
}


def build_model(
    spec: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_output_tokens: int | None = None,
    default_body: Mapping[str, Any] | None = None,
    default_headers: Mapping[str, str] | None = None,
    pricing: TokenPricing | None = None,
    protocol: str | None = None,
    expected_resolved_model: str | None = None,
    max_retries: int | None = None,
    timeout_seconds: float | None = None,
    max_history_images: Any = _UNSET,
) -> BackendModel:
    """Resolve the three maintained provider adapters from ``provider/model``.

    ``max_retries``/``timeout_seconds`` default to the backend constructors'
    values when None. ``max_history_images`` applies only to transcript-replay
    protocols (chat-completions and Anthropic Messages); pass an int, or None
    for unlimited replay. The Responses protocol keeps continuation server-side
    and rejects the option.
    """

    provider, model_name = parse_model_spec(spec)
    if protocol not in {None, "responses", "chat-completions"}:
        raise ValueError(
            "protocol must be 'responses', 'chat-completions', or None"
        )
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"unsupported provider {provider!r}; choose openai, anthropic, or meta"
        )
    transport: dict[str, Any] = {}
    if max_retries is not None:
        transport["max_retries"] = max_retries
    if timeout_seconds is not None:
        transport["timeout_seconds"] = timeout_seconds
    replay_transport = dict(transport)
    if max_history_images is not _UNSET:
        replay_transport["max_history_images"] = max_history_images

    def require_replay_protocol() -> None:
        if max_history_images is not _UNSET:
            raise ValueError(
                "max_history_images requires a transcript-replay protocol; "
                "the Responses adapter's continuation is server-side"
            )
    if provider == "meta" and base_url is None:
        raise ValueError(
            "meta models require an explicit --base-url naming the deployment"
        )
    if provider == "anthropic" and protocol is not None:
        raise ValueError("anthropic models always use the Messages protocol")
    default_url, default_key_env = _PROVIDER_DEFAULTS[provider]
    common: dict[str, Any] = {
        "model": model_name,
        "base_url": default_url if base_url is None else base_url,
        "api_key_env": default_key_env if api_key_env is None else api_key_env,
        "default_body": default_body,
        "default_headers": default_headers,
        "pricing": pricing,
    }
    if provider == "anthropic":
        from .providers import AnthropicMessagesBackend

        resolved_backend: Any = AnthropicMessagesBackend(**common, **replay_transport)
    elif protocol == "chat-completions":
        from .providers import ChatCompletionsBackend

        resolved_backend = ChatCompletionsBackend(
            provider=provider, **common, **replay_transport
        )
    else:
        require_replay_protocol()
        from .providers import OpenAIResponsesBackend

        resolved_backend = OpenAIResponsesBackend(
            provider=provider, **common, **transport
        )
    return BackendModel(
        resolved_backend,
        max_output_tokens=max_output_tokens,
        expected_resolved_model=expected_resolved_model,
    )


__all__ = [
    "BackendModel",
    "Model",
    "SUPPORTED_PROVIDERS",
    "ScriptedModel",
    "build_model",
    "parse_model_spec",
    "translation_losses_for",
]
