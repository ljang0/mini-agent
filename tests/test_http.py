from __future__ import annotations

import unittest
from typing import Any, AsyncIterator, Sequence

from mini_agent._http import ResponseBodyTooLarge, read_bounded_body


class FakeStreamResponse:
    def __init__(
        self,
        chunks: Sequence[Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.headers = headers or {}
        self.requested_chunk_sizes: list[int] = []

    async def aiter_bytes(self, *, chunk_size: int) -> AsyncIterator[Any]:
        self.requested_chunk_sizes.append(chunk_size)
        for chunk in self.chunks:
            yield chunk


class ReadBoundedBodyTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_body_within_limit(self) -> None:
        response = FakeStreamResponse([b"hello ", b"world"])
        body = await read_bounded_body(response, 11)
        self.assertEqual(body, b"hello world")

    async def test_rejects_oversized_content_length_before_reading(self) -> None:
        response = FakeStreamResponse([b"x"], headers={"content-length": "100"})
        with self.assertRaises(ResponseBodyTooLarge):
            await read_bounded_body(response, 99)
        self.assertEqual(response.requested_chunk_sizes, [])

    async def test_accepts_exact_content_length_and_body(self) -> None:
        response = FakeStreamResponse([b"abcd"], headers={"content-length": "4"})
        self.assertEqual(await read_bounded_body(response, 4), b"abcd")

    async def test_rejects_invalid_content_length(self) -> None:
        for value in ("not-a-number", "-1"):
            response = FakeStreamResponse([b"x"], headers={"content-length": value})
            with self.assertRaisesRegex(ValueError, "Content-Length is invalid"):
                await read_bounded_body(response, 10)

    async def test_rejects_stream_that_crosses_the_limit(self) -> None:
        response = FakeStreamResponse([b"abc", b"def", b"ghi"])
        with self.assertRaises(ResponseBodyTooLarge):
            await read_bounded_body(response, 8)

    async def test_rejects_non_bytes_chunks(self) -> None:
        response = FakeStreamResponse(["text"])
        with self.assertRaisesRegex(TypeError, "non-bytes chunk"):
            await read_bounded_body(response, 10)

    async def test_chunk_size_never_exceeds_limit_plus_one(self) -> None:
        response = FakeStreamResponse([b"ab"])
        await read_bounded_body(response, 2)
        self.assertEqual(response.requested_chunk_sizes, [3])

    async def test_missing_headers_attribute_is_tolerated(self) -> None:
        class Bare:
            async def aiter_bytes(self, *, chunk_size: int) -> AsyncIterator[bytes]:
                yield b"ok"

        self.assertEqual(await read_bounded_body(Bare(), 2), b"ok")


if __name__ == "__main__":
    unittest.main()
