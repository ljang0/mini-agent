"""Small shared guard for streaming remote response bodies."""

from __future__ import annotations

from typing import Any


class ResponseBodyTooLarge(RuntimeError):
    """Raised before a streamed response can exceed its configured byte cap."""


async def read_bounded_body(response: Any, max_bytes: int) -> bytes:
    """Read a response incrementally and reject it before crossing ``max_bytes``."""

    headers = getattr(response, "headers", {})
    raw_length = headers.get("content-length") if hasattr(headers, "get") else None
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("response Content-Length is invalid") from exc
        if content_length < 0:
            raise ValueError("response Content-Length is invalid")
        if content_length > max_bytes:
            raise ResponseBodyTooLarge

    body = bytearray()
    chunk_size = min(64 * 1024, max_bytes + 1)
    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
        if not isinstance(chunk, bytes):
            raise TypeError("response stream yielded a non-bytes chunk")
        if len(body) + len(chunk) > max_bytes:
            raise ResponseBodyTooLarge
        body.extend(chunk)
    return bytes(body)


__all__ = ["ResponseBodyTooLarge", "read_bounded_body"]
