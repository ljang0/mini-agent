"""Shared deterministic fixtures for the mini-agent test suite."""

from __future__ import annotations

import asyncio
import os
import struct
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from mini_agent.environments.base import BaseEnvironment
from mini_agent.types import ModelResponse, ToolCall, ToolDefinition, ToolExecution


def png(width: int = 8, height: int = 6) -> bytes:
    def chunk(kind: bytes, content: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(content, checksum) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", checksum)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\0" + b"\0\0\0" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class WordTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[str]:
        self.add_special_tokens = add_special_tokens
        return text.split()

    def decode(self, tokens: Sequence[Any], *, skip_special_tokens: bool) -> str:
        self.skip_special_tokens = skip_special_tokens
        return " ".join(str(token) for token in tokens)


class BlockingModel:
    """Model that never answers; ``started`` fires once a query arrives."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def query(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class EmptyEnvironment(BaseEnvironment):
    def tools(self) -> Sequence[ToolDefinition]:
        return ()


class EchoEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.finished = False

    def tools(self) -> Sequence[ToolDefinition]:
        return (
            ToolDefinition(
                "echo",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            ),
        )

    async def execute(self, action: ToolCall) -> ToolExecution:
        self.calls.append(action)
        return ToolExecution(str(action.arguments["value"]))

    async def finish(self) -> None:
        self.finished = True


class IsolatedEnvironment(BaseEnvironment):
    def __init__(
        self,
        agent_id: str,
        *,
        identity: str | None = None,
        fail_close: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.identity = identity or f"resource:{agent_id}"
        self.fail_close = fail_close
        self.closed = False
        self.state = agent_id
        self.adoptions: list[Any] = []

    def tools(self) -> Sequence[ToolDefinition]:
        return (ToolDefinition("identity"),)

    async def execute(self, action: ToolCall) -> ToolExecution:
        return ToolExecution(self.agent_id)

    async def export_state(self) -> Any:
        return self.state

    async def adopt_state(self, state: Any) -> None:
        self.adoptions.append(state)
        self.state = state

    async def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")

    def resource_identity(self) -> str:
        return self.identity


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
