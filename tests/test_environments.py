from __future__ import annotations

import asyncio
import json
import struct
import tempfile
import threading
import time
import unittest
import httpx
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, patch

from mini_agent.environments.cua import (
    AdapterLiveState,
    CUAEnvironment,
    CUASpeedRunClient,
    ComputerObservation,
    OSWorldClient,
    OSWorldEnvironment,
    complete_in_thread,
    encode_osworld_action,
    to_cua_speedrun_actions,
    validate_computer_actions,
    validate_png,
)
from mini_agent.environments.swe import (
    BashEnvironment,
    LocalProcessRunner,
    ProcessResult,
    SWEPatchState,
)
from mini_agent.environments.swebench import (
    ApptainerSWEEnvironment,
    DockerSWEEnvironment,
    SWEArchiveState,
    SWEbenchImageBinding,
    _materialize_apptainer_image,
    resolve_swebench_image_binding,
    swebench_doctor,
)
from mini_agent.environments.web import (
    ANSERINI_JAR_SHA256,
    BrowserEnvironment,
    BrowserSessionState,
    BrowseCompPlusBackend,
    HttpPageReader,
    JsonlSearchBackend,
    PlaywrightPageReader,
    SerpAPIBackend,
    directory_sha256,
)
from mini_agent.environments import web as web_environment_module
from mini_agent.runtime import RunContext
from mini_agent.types import InfrastructureError, ProtocolError, ToolCall


from support import WordTokenizer, png

TEST_DOCKER_ID = "sha256:" + "a" * 64


class StaticSearch:
    def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
        del query
        return [
            {
                "ref": str(index),
                "docid": str(index),
                "score": float(10 - index),
                "snippet": f"document {index} has several words",
            }
            for index in range(k)
        ]

    def open(self, reference: str) -> Mapping[str, Any] | None:
        return {"ref": reference, "docid": reference, "text": f"full {reference}"}

    def provenance(self) -> Mapping[str, Any]:
        return {"backend": "static"}


class WebEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_only_browser_exposes_and_records_one_action(self) -> None:
        environment = BrowserEnvironment(StaticSearch(), allow_open=False)
        tool = environment.tools()[0]
        self.assertEqual(tool.input_schema["properties"]["action"]["enum"], ["search"])
        self.assertEqual(tool.input_schema["required"], ["action", "query"])
        self.assertNotIn("ref", tool.input_schema["properties"])
        self.assertIn("top-5 hits with docid, score, and snippet", tool.description)
        self.assertEqual(environment.provenance()["actions"], ["search"])
        result = await environment.execute(
            ToolCall("search", "browser", {"action": "search", "query": "q"})
        )
        self.assertEqual(
            set(json.loads(result.output)[0]),
            {"docid", "score", "snippet"},
        )
        with self.assertRaisesRegex(ProtocolError, "not enabled"):
            await environment.execute(
                ToolCall("open", "browser", {"action": "open", "ref": "1"})
            )

    async def test_unbounded_whole_observation_preserves_bounded_snippets(self) -> None:
        environment = BrowserEnvironment(
            StaticSearch(),
            top_k=5,
            max_observation_chars=None,
            snippet_tokens=3,
            tokenizer=WordTokenizer(),
            allow_open=False,
        )
        result = await environment.execute(
            ToolCall("search", "browser", {"action": "search", "query": "q"})
        )
        values = json.loads(result.output)
        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 5)
        self.assertEqual(values[0]["snippet"], "document 0 has")
        self.assertIsNone(environment.provenance()["max_observation_chars"])

    async def test_malformed_search_result_does_not_publish_its_reference(self) -> None:
        class MalformedBackend:
            async def search(
                self, query: str, k: int = 5
            ) -> Sequence[Mapping[str, Any]]:
                del query, k
                return ({"ref": "hidden", "bad": object()},)

            async def open(self, reference: str) -> Mapping[str, Any]:
                raise AssertionError(f"unpublished reference opened: {reference}")

        environment = BrowserEnvironment(MalformedBackend())
        with self.assertRaisesRegex(InfrastructureError, "non-JSON"):
            await environment.execute(
                ToolCall("search", "browser", {"action": "search", "query": "q"})
            )
        self.assertEqual(environment.accounting()["tool_calls"], {"search": 1})
        with self.assertRaisesRegex(ProtocolError, "not returned"):
            await environment.execute(
                ToolCall("open", "browser", {"action": "open", "ref": "hidden"})
            )

    async def test_browser_rejects_text_that_cannot_be_written_as_utf8(self) -> None:
        class SurrogateBackend(StaticSearch):
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return [{"ref": "1", "title": "\ud800"}]

        environment = BrowserEnvironment(SurrogateBackend())
        with self.assertRaisesRegex(InfrastructureError, "non-JSON"):
            await environment.execute(
                ToolCall("search", "browser", {"action": "search", "query": "q"})
            )

    async def test_search_open_accounting_and_token_truncation(self) -> None:
        tokenizer = WordTokenizer()
        environment = BrowserEnvironment(
            StaticSearch(),
            top_k=3,
            snippet_tokens=3,
            tokenizer=tokenizer,
        )
        result = await environment.execute(
            ToolCall("search", "browser", {"action": "search", "query": "alpha"})
        )
        values = json.loads(result.output)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0]["snippet"], "document 0 has")
        opened = await environment.execute(
            ToolCall("open", "browser", {"action": "open", "ref": "1"})
        )
        self.assertEqual(json.loads(opened.output)["text"], "full 1")
        self.assertEqual(
            environment.accounting(),
            {
                "tool_calls": {"open": 1, "search": 1},
                "references": ["0", "1", "2"],
            },
        )
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertTrue(tokenizer.skip_special_tokens)

    async def test_browser_state_adoption_transfers_open_rights(self) -> None:
        child = BrowserEnvironment(StaticSearch(), top_k=2)
        await child.execute(
            ToolCall("search", "browser", {"action": "search", "query": "alpha"})
        )
        state = await child.export_state()
        self.assertIsInstance(state, BrowserSessionState)
        assert state is not None
        self.assertEqual(state.references, ("0", "1"))

        parent = BrowserEnvironment(StaticSearch(), top_k=2)
        with self.assertRaisesRegex(ProtocolError, "not returned"):
            await parent.execute(
                ToolCall("open", "browser", {"action": "open", "ref": "1"})
            )
        await parent.adopt_state(state)
        opened = await parent.execute(
            ToolCall("open", "browser", {"action": "open", "ref": "1"})
        )
        self.assertEqual(json.loads(opened.output)["text"], "full 1")

    async def test_browser_state_adoption_transfers_serpapi_open_rights(self) -> None:
        class Reader:
            async def open(self, url: str) -> Mapping[str, Any]:
                return {"ref": url, "text": f"page {url}"}

            async def close(self) -> None:
                return None

        class DeterministicSerpAPI(SerpAPIBackend):
            async def search(
                self, query: str, k: int = 5
            ) -> Sequence[Mapping[str, Any]]:
                del query, k
                url = "https://example.com/result"
                current = set(self.export_reference_state())
                self.replace_reference_state(tuple(current | {url}))
                return ({"ref": url, "title": "result", "snippet": "text"},)

        child_backend = DeterministicSerpAPI(api_key="test", page_reader=Reader())
        child = BrowserEnvironment(child_backend)
        await child.execute(
            ToolCall("search", "browser", {"action": "search", "query": "q"})
        )
        state = await child.export_state()
        assert state is not None

        parent_backend = DeterministicSerpAPI(api_key="test", page_reader=Reader())
        parent = BrowserEnvironment(parent_backend)
        self.assertEqual(parent_backend.export_reference_state(), ())
        await parent.adopt_state(state)

        opened = await parent.execute(
            ToolCall(
                "open",
                "browser",
                {"action": "open", "ref": "https://example.com/result"},
            )
        )
        self.assertEqual(
            json.loads(opened.output)["text"],
            "page https://example.com/result",
        )
        self.assertEqual(
            parent_backend.export_reference_state(),
            ("https://example.com/result",),
        )

    async def test_serpapi_stream_rejects_oversized_json(self) -> None:
        http = FakeHTTPClient([FakeResponse(content=b'{"too":"large"}')])
        backend = SerpAPIBackend(api_key="test", max_response_bytes=8)
        with patch(
            "mini_agent.environments.web.httpx.AsyncClient", return_value=http
        ):
            with self.assertRaisesRegex(InfrastructureError, "byte limit"):
                await backend.search("bounded")

    async def test_serpapi_rejects_non_success_search_status(self) -> None:
        http = FakeHTTPClient(
            [FakeResponse({"search_metadata": {"status": "Processing"}})]
        )
        backend = SerpAPIBackend(api_key="test")
        with patch(
            "mini_agent.environments.web.httpx.AsyncClient", return_value=http
        ):
            with self.assertRaisesRegex(InfrastructureError, "did not complete"):
                await backend.search("unfinished")

    async def test_serpapi_requires_official_success_metadata(self) -> None:
        http = FakeHTTPClient([FakeResponse({"organic_results": []})])
        backend = SerpAPIBackend(api_key="test")
        with patch(
            "mini_agent.environments.web.httpx.AsyncClient", return_value=http
        ):
            with self.assertRaisesRegex(InfrastructureError, "search_metadata"):
                await backend.search("missing status")

        http = FakeHTTPClient(
            [
                FakeResponse(
                    {
                        "search_metadata": {"status": "Success"},
                        "organic_results": [
                            {
                                "link": "https://example.com/result",
                                "title": "result",
                                "snippet": "text",
                            }
                        ],
                    }
                )
            ]
        )
        backend = SerpAPIBackend(api_key="test")
        with (
            patch(
                "mini_agent.environments.web.httpx.AsyncClient",
                return_value=http,
            ),
            patch("mini_agent.environments.web._validate_public_url", AsyncMock()),
        ):
            results = await backend.search("successful")
        self.assertEqual(results[0]["ref"], "https://example.com/result")

    async def test_browser_state_adoption_rolls_back_both_reference_sets(self) -> None:
        class FailingStateBackend:
            def __init__(self) -> None:
                self.references: set[str] = set()
                self.fail_next_replace = False

            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                references = [f"{query}-{index}" for index in range(k)]
                self.references.update(references)
                return [{"ref": reference} for reference in references]

            def open(self, reference: str) -> Mapping[str, Any]:
                if reference not in self.references:
                    raise ProtocolError("backend reference was not authorized")
                return {"ref": reference, "text": reference}

            def provenance(self) -> Mapping[str, Any]:
                return {"backend": "failing-state-fixture"}

            def export_reference_state(self) -> tuple[str, ...]:
                return tuple(sorted(self.references))

            def replace_reference_state(self, references: Sequence[str]) -> None:
                replacement = set(references)
                if self.fail_next_replace:
                    self.fail_next_replace = False
                    self.references = {min(replacement)}
                    raise ProtocolError("injected state replacement failure")
                self.references = replacement

        child_backend = FailingStateBackend()
        child = BrowserEnvironment(child_backend, top_k=2)
        await child.execute(
            ToolCall("search", "browser", {"action": "search", "query": "child"})
        )
        state = await child.export_state()
        assert state is not None

        parent_backend = FailingStateBackend()
        parent = BrowserEnvironment(parent_backend, top_k=2)
        await parent.execute(
            ToolCall("search", "browser", {"action": "search", "query": "parent"})
        )
        backend_before = parent_backend.export_reference_state()
        accounting_before = parent.accounting()
        parent_backend.fail_next_replace = True

        with self.assertRaisesRegex(ProtocolError, "injected state replacement"):
            await parent.adopt_state(state)

        self.assertEqual(parent_backend.export_reference_state(), backend_before)
        self.assertEqual(parent.accounting(), accounting_before)
        with self.assertRaisesRegex(ProtocolError, "not returned"):
            await parent.execute(
                ToolCall(
                    "open", "browser", {"action": "open", "ref": "child-0"}
                )
            )

    async def test_browser_state_adoption_fails_closed(self) -> None:
        class OtherBackend(StaticSearch):
            def provenance(self) -> Mapping[str, Any]:
                return {"backend": "other-corpus"}

        class NoProvenanceBackend:
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return []

            def open(self, reference: str) -> None:
                del reference
                return None

        child = BrowserEnvironment(StaticSearch())
        await child.execute(
            ToolCall("search", "browser", {"action": "search", "query": "alpha"})
        )
        state = await child.export_state()
        assert state is not None

        mismatched = BrowserEnvironment(OtherBackend())
        with self.assertRaisesRegex(ProtocolError, "different search backend"):
            await mismatched.adopt_state(state)

        parent = BrowserEnvironment(StaticSearch())
        with self.assertRaisesRegex(ProtocolError, "incompatible type"):
            await parent.adopt_state(SWEPatchState("base", b""))

        anonymous = BrowserEnvironment(NoProvenanceBackend())
        self.assertIsNone(await anonymous.export_state())
        with self.assertRaisesRegex(ProtocolError, "different search backend"):
            await anonymous.adopt_state(state)

        await parent.close()
        with self.assertRaises(RuntimeError):
            await parent.export_state()
        with self.assertRaises(RuntimeError):
            await parent.adopt_state(state)

    async def test_short_snippet_is_not_normalized_by_tokenizer(self) -> None:
        class ShortSearch(StaticSearch):
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return [{"ref": "1", "snippet": "two  spaces"}]

        tokenizer = WordTokenizer()
        environment = BrowserEnvironment(
            ShortSearch(), snippet_tokens=3, tokenizer=tokenizer
        )
        result = await environment.execute(
            ToolCall("search", "browser", {"action": "search", "query": "q"})
        )
        self.assertEqual(json.loads(result.output)[0]["snippet"], "two  spaces")
        self.assertFalse(hasattr(tokenizer, "skip_special_tokens"))

    async def test_open_requires_a_session_reference(self) -> None:
        environment = BrowserEnvironment(StaticSearch())
        with self.assertRaisesRegex(ProtocolError, "not returned"):
            await environment.execute(
                ToolCall("open", "browser", {"action": "open", "ref": "1"})
            )

        await environment.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await environment.execute(
                ToolCall(
                    "search",
                    "browser",
                    {"action": "search", "query": "after close"},
                )
            )

    async def test_bounded_observation_is_always_valid_json(self) -> None:
        environment = BrowserEnvironment(StaticSearch(), max_observation_chars=128)
        result = await environment.execute(
            ToolCall("search", "browser", {"action": "search", "query": "x"})
        )
        value = json.loads(result.output)
        self.assertTrue(value["truncated"])
        self.assertLessEqual(len(result.output), 128)

    async def test_malformed_search_reference_is_not_stringified(self) -> None:
        class MalformedSearch(StaticSearch):
            def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
                del query, k
                return [{"ref": None, "snippet": "text"}]

        environment = BrowserEnvironment(MalformedSearch())
        with self.assertRaisesRegex(InfrastructureError, "without ref"):
            await environment.execute(
                ToolCall("search", "browser", {"action": "search", "query": "alpha"})
            )

    async def test_playwright_cleanup_attempts_every_owned_layer(self) -> None:
        calls: list[str] = []

        class Resource:
            def __init__(self, name: str, *, fail: bool = False) -> None:
                self.name = name
                self.fail = fail

            async def close(self) -> None:
                calls.append(self.name)
                if self.fail:
                    raise RuntimeError(f"{self.name} failed")

            async def stop(self) -> None:
                await self.close()

        reader = PlaywrightPageReader()
        reader._context = Resource("context", fail=True)
        reader._browser = Resource("browser")
        reader._playwright = Resource("playwright")
        with self.assertRaisesRegex(RuntimeError, "context failed"):
            await reader.close()
        self.assertEqual(calls, ["context", "browser", "playwright"])

    async def test_playwright_extracts_page_text_with_a_dom_side_cap(self) -> None:
        class Page:
            url = "https://example.com/final"

            def __init__(self) -> None:
                self.limit: int | None = None

            async def goto(self, url: str, *, wait_until: str) -> None:
                self.url = url
                self.wait_until = wait_until

            async def evaluate(self, expression: str, limit: int) -> Mapping[str, str]:
                self.expression = expression
                self.limit = limit
                return {"title": "title"[:limit], "text": "x" * limit}

        page = Page()
        reader = PlaywrightPageReader(max_page_chars=7)
        reader._page = page
        with patch(
            "mini_agent.environments.web._validate_public_url", AsyncMock()
        ):
            result = await reader.open("https://example.com/start")
        self.assertEqual(page.limit, 7)
        self.assertIn("innerText.slice(0, limit)", page.expression)
        self.assertEqual(result["text"], "x" * 7)

    async def test_playwright_rejects_http_error_responses(self) -> None:
        class Response:
            status = 404

        class Page:
            url = "https://example.com/missing"

            async def goto(self, url: str, *, wait_until: str) -> Response:
                del url, wait_until
                return Response()

        reader = PlaywrightPageReader()
        reader._page = Page()
        with patch(
            "mini_agent.environments.web._validate_public_url", AsyncMock()
        ):
            with self.assertRaisesRegex(InfrastructureError, "HTTP status"):
                await reader.open("https://example.com/missing")

    async def test_playwright_rejects_a_final_redirect_response(self) -> None:
        class Response:
            status = 302

        class Page:
            url = "https://example.com/redirect"

            async def goto(self, url: str, *, wait_until: str) -> Response:
                del url, wait_until
                return Response()

        reader = PlaywrightPageReader()
        reader._page = Page()
        with patch(
            "mini_agent.environments.web._validate_public_url", AsyncMock()
        ):
            with self.assertRaisesRegex(InfrastructureError, "HTTP status"):
                await reader.open("https://example.com/redirect")

    async def test_backend_corruption_is_terminal_but_bad_action_is_repairable(
        self,
    ) -> None:
        class MalformedBackend(StaticSearch):
            def search(self, query: str, k: int = 5) -> object:
                del query, k
                return object()

        malformed = BrowserEnvironment(MalformedBackend())
        malformed_tools = tuple(malformed.tools())
        context = RunContext()
        with self.assertRaisesRegex(InfrastructureError, "result sequence"):
            await context.execute(
                malformed,
                ToolCall(
                    "remote", "browser", {"action": "search", "query": "valid"}
                ),
                malformed_tools,
                agent_id="/root",
                role="solver",
            )

        healthy = BrowserEnvironment(StaticSearch())
        repairable = await context.execute(
            healthy,
            ToolCall("bad", "browser", {"action": "search", "query": ""}),
            tuple(healthy.tools()),
            agent_id="/root",
            role="solver",
        )
        self.assertTrue(repairable.is_error)
        self.assertIn("must be a non-empty string", repairable.output)

    async def test_http_reader_rejects_private_networks(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "non-public"):
            await HttpPageReader().open("http://127.0.0.1/private")
        with self.assertRaisesRegex(ProtocolError, "invalid port"):
            await HttpPageReader().open("http://example.com:0/")

    async def test_http_reader_matches_html_content_type_case_insensitively(
        self,
    ) -> None:
        class Response:
            is_redirect = False
            is_error = False
            status_code = 200
            headers = {"content-type": "TEXT/HTML; charset=UTF-8"}
            encoding = "utf-8"
            url = "https://example.com/final"

            async def __aenter__(self) -> "Response":
                return self

            async def __aexit__(self, *args: object) -> None:
                del args

            async def aiter_bytes(self, *, chunk_size: int) -> Any:
                del chunk_size
                yield b"<p>Hello &amp; &amp;lt; goodbye</p>"

        class Client:
            async def __aenter__(self) -> "Client":
                return self

            async def __aexit__(self, *args: object) -> None:
                del args

            def stream(self, method: str, url: str) -> Response:
                self.request = (method, url)
                return Response()

        client = Client()
        with (
            patch(
                "mini_agent.environments.web.httpx.AsyncClient",
                return_value=client,
            ),
            patch("mini_agent.environments.web._validate_public_url", AsyncMock()),
        ):
            result = await HttpPageReader().open("https://example.com/start")
        self.assertEqual(client.request, ("GET", "https://example.com/start"))
        self.assertEqual(result["url"], "https://example.com/final")
        self.assertEqual(result["text"], "Hello & &lt; goodbye")

    async def test_http_reader_rejects_non_success_nonredirect_status(self) -> None:
        class Response:
            is_redirect = False
            is_error = False
            status_code = 304
            headers: dict[str, str] = {}
            encoding = "utf-8"
            url = "https://example.com/not-modified"

            async def __aenter__(self) -> "Response":
                return self

            async def __aexit__(self, *args: object) -> None:
                del args

        class Client:
            async def __aenter__(self) -> "Client":
                return self

            async def __aexit__(self, *args: object) -> None:
                del args

            def stream(self, method: str, url: str) -> Response:
                del method, url
                return Response()

        with (
            patch(
                "mini_agent.environments.web.httpx.AsyncClient",
                return_value=Client(),
            ),
            patch("mini_agent.environments.web._validate_public_url", AsyncMock()),
        ):
            with self.assertRaisesRegex(InfrastructureError, "HTTP 304"):
                await HttpPageReader().open("https://example.com/not-modified")

    def test_jsonl_bm25_and_tree_identity_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"docid":"b","text":"beta only"}\n'
                '{"docid":"a","text":"alpha alpha evidence"}\n',
                encoding="utf-8",
            )
            backend = JsonlSearchBackend(corpus)
            self.assertEqual(backend.search("alpha")[0]["ref"], "a")
            self.assertEqual(len(backend.provenance()["sha256"]), 64)
            first = directory_sha256(root)
            self.assertEqual(first, directory_sha256(root))
            link = root / "link"
            link.symlink_to(corpus)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                directory_sha256(root)

    def test_fixed_backend_uses_pinned_lucene_shape_without_dense_imports(
        self,
    ) -> None:
        class Searcher:
            closed = False

            def search(self, query: str, k: int) -> list[Any]:
                self.search_args = (query, k)
                document = SimpleNamespace(
                    get=lambda field: '{"contents":"fixture evidence"}'
                )
                return [
                    SimpleNamespace(docid="doc-1", score=2.5, lucene_document=document)
                ]

            def doc(self, reference: str) -> Any:
                self.opened = reference
                return SimpleNamespace(
                    get=lambda field: '{"contents":"fixture evidence"}'
                )

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary)
            (index / "segments_1").write_bytes(b"fixture")
            jar = index / "anserini-1.1.1-fatjar.jar"
            jar.write_bytes(b"fixture jar")
            searcher = Searcher()
            with (
                patch(
                    "mini_agent.environments.web._file_sha256",
                    return_value=ANSERINI_JAR_SHA256,
                ),
                patch(
                    "mini_agent.environments.web._lucene_searcher",
                    return_value=searcher,
                ),
            ):
                backend = BrowseCompPlusBackend(index, jar)
            self.assertEqual(backend.search("query", 5)[0]["ref"], "doc-1")
            self.assertEqual(searcher.search_args, ("query", 5))
            self.assertEqual(backend.open("doc-1")["text"], "fixture evidence")
            self.assertEqual(searcher.opened, "doc-1")
            backend.close()
            self.assertTrue(searcher.closed)
            self.assertEqual(
                backend.provenance()["anserini_jar_sha256"],
                ANSERINI_JAR_SHA256,
            )

    def test_fixed_backend_rejects_unpinned_anserini_jar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index"
            index.mkdir()
            (index / "segments_1").write_bytes(b"fixture")
            jar = root / "anserini.jar"
            jar.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "JAR hash"):
                BrowseCompPlusBackend(index, jar)

    def test_lucene_searcher_rejects_pyjnius_dependency_drift(self) -> None:
        with patch(
            "mini_agent.environments.web.importlib.metadata.version",
            return_value="unexpected",
        ):
            with self.assertRaisesRegex(RuntimeError, "pyjnius==1.6.1"):
                web_environment_module._lucene_searcher(Path("index"), Path("jar"))


class SWEEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_swe_uses_the_non_login_upstream_shell_contract(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.calls: list[tuple[str, ...]] = []

            async def run(
                self, argv: Sequence[str], **kwargs: Any
            ) -> ProcessResult:
                del kwargs
                self.calls.append(tuple(argv))
                return ProcessResult(b"ok\n", 0, 3)

        with tempfile.TemporaryDirectory() as temporary:
            runner = Runner()
            environment = BashEnvironment(Path(temporary), runner=runner)
            try:
                await environment.execute(
                    ToolCall("call", "bash", {"command": "pwd"})
                )
                self.assertEqual(
                    runner.calls,
                    [("/bin/bash", "--noprofile", "--norc", "-c", "pwd")],
                )
            finally:
                await environment.close()

    async def test_isolated_patch_export_and_transactional_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            scratch = Path(temporary) / "scratch"
            source.mkdir()
            (source / "value.txt").write_text("original\n", encoding="utf-8")
            first = await BashEnvironment.isolated(source, scratch_root=scratch)
            second = await BashEnvironment.isolated(source, scratch_root=scratch)
            try:
                self.assertEqual([tool.name for tool in first.tools()], ["bash"])
                await first.execute(
                    ToolCall(
                        "edit",
                        "bash",
                        {"command": "printf 'changed\\n' > value.txt"},
                    )
                )
                state = await first.export_state()
                self.assertIn(b"value.txt", state.patch)
                await second.adopt_state(state)
                self.assertEqual(
                    (second.workspace / "value.txt").read_text(), "changed\n"
                )
                self.assertEqual((source / "value.txt").read_text(), "original\n")
            finally:
                await first.close()
                await second.close()

    async def test_isolated_patch_survives_an_agent_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "value.txt").write_text("original\n", encoding="utf-8")
            environment = await BashEnvironment.isolated(source)
            try:
                await environment.execute(
                    ToolCall(
                        "edit",
                        "bash",
                        {
                            "command": (
                                "printf 'changed\\n' > value.txt && "
                                "git add value.txt && "
                                "git -c user.name=agent -c user.email=agent@invalid "
                                "commit -m agent-change"
                            )
                        },
                    )
                )
                patch_bytes = await environment.export_patch()
                self.assertIn(b"-original", patch_bytes)
                self.assertIn(b"+changed", patch_bytes)
            finally:
                await environment.close()

    async def test_isolation_rejects_escaping_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "escape").symlink_to("../outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                await BashEnvironment.isolated(source)

    async def test_process_timeout_and_head_tail_are_bounded(self) -> None:
        runner = LocalProcessRunner()
        result = await runner.run(
            (
                "/bin/sh",
                "-c",
                "python3 -c 'print(\"x\" * 10000)'",
            ),
            max_output_bytes=100,
        )
        self.assertTrue(result.truncated)
        self.assertGreater(result.total_output_bytes, 100)
        self.assertLessEqual(len(result.output), 100)

        timed = await runner.run(
            ("/bin/sh", "-c", "sleep 10"),
            timeout_seconds=0.01,
        )
        self.assertTrue(timed.timed_out)

        background = await runner.run(
            ("/bin/sh", "-c", "sleep 10 &"),
            timeout_seconds=0.05,
        )
        self.assertTrue(background.timed_out)

    async def test_adoption_rolls_back_after_reset_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "value.txt").write_text("original\n", encoding="utf-8")
            environment = await BashEnvironment.isolated(source)
            try:
                (environment.workspace / "value.txt").write_text(
                    "prior\n", encoding="utf-8"
                )
                original_reset = environment._reset_to_baseline
                calls = 0

                async def failing_reset_once() -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        (environment.workspace / "value.txt").write_text(
                            "partial\n", encoding="utf-8"
                        )
                        raise RuntimeError("reset interrupted")
                    await original_reset()

                with patch.object(
                    environment, "_reset_to_baseline", failing_reset_once
                ):
                    with self.assertRaisesRegex(RuntimeError, "reset interrupted"):
                        identity = (await environment.export_state()).base_identity
                        await environment.adopt_state(SWEPatchState(identity, b""))
                self.assertEqual(
                    (environment.workspace / "value.txt").read_text(), "prior\n"
                )
            finally:
                await environment.close()

    def test_external_callers_cannot_claim_recursive_cleanup_ownership(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mini-agent-swe-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            home = root / "home"
            workspace.mkdir()
            home.mkdir()
            with self.assertRaisesRegex(ValueError, "created only by isolated"):
                BashEnvironment(workspace, owned_root=root, home=home)


class RecordingRunner:
    def __init__(self, *, rootless: bool = True) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.rootless = rootless
        self.pull_calls = 0

    async def run(
        self,
        argv: Sequence[str],
        **kwargs: Any,
    ) -> ProcessResult:
        del kwargs
        call = tuple(argv)
        self.calls.append(call)
        output = b""
        if any("SecurityOptions" in item for item in call):
            output = b'["name=rootless"]\n' if self.rootless else b"[]\n"
        elif call[1:2] == ("version",):
            output = b"26.1\n"
        elif call[1:2] == ("info",):
            output = b"linux/amd64\n"
        elif call[1:3] == ("image", "inspect"):
            output = (TEST_DOCKER_ID + "\n").encode()
        elif call[1:2] == ("inspect",):
            output = (TEST_DOCKER_ID + "\n").encode()
        elif call[1:3] == ("overlay", "create"):
            Path(call[-1]).write_bytes(b"overlay")
        elif call[1:2] == ("pull",) and "--force" in call:
            self.pull_calls += 1
            Path(call[-2]).write_bytes(b"sif-content")
        elif "exec" in call[1:3]:
            output = ("c" * 40 + "\n").encode()
        return ProcessResult(output, 0, len(output))


class ArchiveRecordingRunner(RecordingRunner):
    """Recording runner that also serves workspace archive export/adoption."""

    def __init__(self, archive: bytes = b"workspace-archive") -> None:
        super().__init__()
        self.archive = archive

    async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
        call = tuple(argv)
        if call[1:2] == ("cp",):
            self.calls.append(call)
            if ":" in call[2]:
                Path(call[3]).write_bytes(self.archive)
            return ProcessResult(b"", 0, 0)
        if "exec" in call[1:3] and "tar -czf" in call[-1]:
            self.calls.append(call)
            output = f"{len(self.archive)}\n".encode()
            return ProcessResult(output, 0, len(output))
        return await super().run(argv, **kwargs)


class SWEbenchEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_container_disables_network_and_exports_an_archive(
        self,
    ) -> None:
        runner = ArchiveRecordingRunner()
        environment = await DockerSWEEnvironment.create(
            {
                "instance_id": "org__tool.abc1234",
                "image_name": "programbench/org_1776_tool.abc1234:task_cleanroom_v6",
            },
            runner=runner,
            workdir="/workspace",
            network_disabled=True,
            require_git_baseline=False,
            benchmark_identity={"benchmark": "programbench"},
        )
        try:
            start = next(call for call in runner.calls if call[1:2] == ("run",))
            self.assertEqual(start[start.index("--network") + 1], "none")
            self.assertEqual(start[start.index("--workdir") + 1], "/workspace")
            self.assertFalse(
                any("git rev-parse" in call[-1] for call in runner.calls)
            )
            self.assertIsNone(environment.base_commit)
            provenance = environment.provenance()
            self.assertTrue(provenance["network_disabled"])
            self.assertEqual(provenance["benchmark"], "programbench")
            self.assertEqual(provenance["workdir"], "/workspace")
            self.assertEqual(provenance["patch_export"], "workspace_tar_gz")
            with self.assertRaisesRegex(RuntimeError, "no Git baseline"):
                await environment.export_patch()
            self.assertEqual(await environment.export_archive(), b"workspace-archive")
            archived = next(
                call for call in reversed(runner.calls) if "tar -czf" in call[-1]
            )
            self.assertIn("-C /workspace .", archived[-1])
            state = await environment.export_state()
            self.assertIsInstance(state, SWEArchiveState)
            await environment.adopt_state(state)
            replaced = next(
                call for call in reversed(runner.calls) if "tar -xzf" in call[-1]
            )
            self.assertIn("find /workspace -mindepth 1 -delete", replaced[-1])
            with self.assertRaisesRegex(ProtocolError, "different container image"):
                await environment.adopt_state(SWEArchiveState("elsewhere", b""))
            environment.max_archive_bytes = 4
            with self.assertRaisesRegex(RuntimeError, "byte limit"):
                await environment.export_archive()
        finally:
            await environment.close()
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_docker_image_preflight_materializes_and_binds_an_exact_id(
        self,
    ) -> None:
        class MissingImageRunner(RecordingRunner):
            def __init__(self) -> None:
                super().__init__()
                self.inspections = 0

            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:3] == ("image", "inspect"):
                    self.inspections += 1
                    if self.inspections == 1:
                        self.calls.append(call)
                        return ProcessResult(b"not found", 1, 9)
                return await super().run(argv, **kwargs)

        runner = MissingImageRunner()
        binding = await resolve_swebench_image_binding(
            {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:latest",
            },
            runtime="docker",
            runner=runner,
        )
        self.assertEqual(binding.identity, TEST_DOCKER_ID)
        self.assertEqual(binding.execution_ref, TEST_DOCKER_ID)
        self.assertEqual(
            binding.manifest_identity(),
            {
                "runtime": "docker",
                "requested": "example/image:latest",
                "identity": TEST_DOCKER_ID,
            },
        )
        self.assertEqual(runner.inspections, 2)
        self.assertTrue(any(call[1:2] == ("pull",) for call in runner.calls))

    async def test_docker_doctor_and_environment_use_rootless_unmounted_contract(
        self,
    ) -> None:
        runner = RecordingRunner()
        report = await swebench_doctor(
            runner=runner,
            image="example/image:tag",
        )
        self.assertTrue(report.ok)
        self.assertEqual(
            [check.name for check in report.checks],
            [
                "runtime_version",
                "daemon_platform",
                "rootless_security",
                "image_available",
            ],
        )

        environment = await DockerSWEEnvironment.create(
            {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:tag",
            },
            runner=runner,
        )
        try:
            start = next(call for call in runner.calls if call[1:2] == ("run",))
            self.assertIn("--workdir", start)
            self.assertNotIn("--volume", start)
            self.assertNotIn("--mount", start)
            self.assertEqual(environment.image_id, TEST_DOCKER_ID)
            self.assertEqual(start[-3], TEST_DOCKER_ID)
            self.assertNotIn("example/image:tag", start)
            self.assertFalse(environment.provenance()["host_credentials_mounted"])
            self.assertEqual(
                environment.provenance()["benchmark_revision"],
                "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            )
            self.assertEqual(environment.provenance()["benchmark_tag"], "v4.1.0")
            await environment.execute(ToolCall("call", "bash", {"command": "pwd"}))
            execution = runner.calls[-1]
            self.assertIn("/bin/bash", execution)
            self.assertIn("BASH_ENV=/root/.bashrc", execution)
            self.assertEqual(execution[-1], "pwd")
            await environment.export_patch()
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "target.diff"
                target.write_text("keep")
                link = Path(temporary) / "link.diff"
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    await environment.export_patch(link)
                self.assertEqual(target.read_text(), "keep")
            stage_call = next(
                call
                for call in reversed(runner.calls)
                if call[-1].startswith("git add")
            )
            self.assertNotIn("--force", stage_call[-1])
            await environment.adopt_state(await environment.export_state())
            reset_call = next(
                call
                for call in reversed(runner.calls)
                if call[-1].startswith("git reset")
            )
            self.assertIn("git clean -ffd -q", reset_call[-1])
            self.assertNotIn("-ffdx", reset_call[-1])
            self.assertIn(environment.base_commit, reset_call[-1])
        finally:
            await environment.close()
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

        unprivileged = await swebench_doctor(runner=RecordingRunner(rootless=False))
        self.assertFalse(unprivileged.ok)
        with self.assertRaisesRegex(RuntimeError, "rootless daemon"):
            await DockerSWEEnvironment.create(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=RecordingRunner(rootless=False),
            )

        class DeceptiveSecurityRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if any("SecurityOptions" in item for item in argv):
                    value = b'["name=notrootless"]\n'
                    self.calls.append(tuple(argv))
                    return ProcessResult(value, 0, len(value))
                return await super().run(argv, **kwargs)

        deceptive = await swebench_doctor(runner=DeceptiveSecurityRunner())
        self.assertFalse(deceptive.ok)

    async def test_docker_container_must_match_its_preflight_binding(self) -> None:
        other_id = "sha256:" + "b" * 64

        class DriftRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("inspect",):
                    self.calls.append(call)
                    output = (other_id + "\n").encode()
                    return ProcessResult(output, 0, len(output))
                return await super().run(argv, **kwargs)

        runner = DriftRunner()
        binding = SWEbenchImageBinding(
            runtime="docker",
            requested="example/image:tag",
            identity=TEST_DOCKER_ID,
            execution_ref=TEST_DOCKER_ID,
        )
        with self.assertRaisesRegex(RuntimeError, "does not match its binding"):
            await DockerSWEEnvironment.create(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                image_binding=binding,
                runner=runner,
            )
        start = next(call for call in runner.calls if call[1:2] == ("run",))
        self.assertEqual(start[-3], TEST_DOCKER_ID)
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_task_base_commit_must_match_the_selected_image(self) -> None:
        class WrongCommitRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if "merge-base --is-ancestor" in tuple(argv)[-1]:
                    self.calls.append(tuple(argv))
                    return ProcessResult(b"", 1, 0)
                return await super().run(argv, **kwargs)

        runner = WrongCommitRunner()
        with self.assertRaisesRegex(RuntimeError, "task base_commit"):
            await DockerSWEEnvironment.create(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                    "base_commit": "d" * 40,
                },
                runner=runner,
            )
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_docker_startup_surfaces_cleanup_failure(self) -> None:
        class FailingRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("inspect",):
                    self.calls.append(call)
                    return ProcessResult(b"inspect failed", 1, 14)
                if call[1:3] == ("rm", "--force"):
                    self.calls.append(call)
                    return ProcessResult(b"remove failed", 1, 13)
                return await super().run(argv, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "cleanup also failed"):
            await DockerSWEEnvironment.create(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=FailingRunner(),
            )

    async def test_docker_run_failure_still_removes_named_container(self) -> None:
        class StartFailureRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                call = tuple(argv)
                if call[1:2] == ("run",):
                    self.calls.append(call)
                    return ProcessResult(b"start failed", 1, 12)
                return await super().run(argv, **kwargs)

        runner = StartFailureRunner()
        with self.assertRaisesRegex(RuntimeError, "could not start"):
            await DockerSWEEnvironment.create(
                {
                    "instance_id": "repo__issue-1",
                    "image_name": "example/image:tag",
                },
                runner=runner,
            )
        self.assertEqual(runner.calls[-1][1:3], ("rm", "--force"))

    async def test_apptainer_overlay_exception_removes_owned_scratch(self) -> None:
        class OverlayFailureRunner(RecordingRunner):
            async def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
                if tuple(argv)[1:3] == ("overlay", "create"):
                    raise RuntimeError("overlay failed")
                return await super().run(argv, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            with self.assertRaisesRegex(RuntimeError, "overlay failed"):
                await ApptainerSWEEnvironment.create(
                    {
                        "instance_id": "repo__issue-1",
                        "image_name": "example/image:tag",
                    },
                    scratch_root=scratch,
                    overlay_size_mib=1024,
                    runner=OverlayFailureRunner(),
                )
            self.assertEqual(list(scratch.iterdir()), [])

    async def test_apptainer_materializes_once_and_uses_private_fakeroot_overlay(
        self,
    ) -> None:
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = {
                "instance_id": "repo__issue-1",
                "image_name": "example/image:tag",
            }
            first = await ApptainerSWEEnvironment.create(
                arguments,
                image="docker://example/image:tag",
                scratch_root=root / "scratch",
                image_cache=root / "cache",
                overlay_size_mib=1024,
                runner=runner,
            )
            second = await ApptainerSWEEnvironment.create(
                arguments,
                image="docker://example/image:tag",
                scratch_root=root / "scratch",
                image_cache=root / "cache",
                overlay_size_mib=1024,
                runner=runner,
            )
            try:
                self.assertEqual(runner.pull_calls, 1)
                self.assertNotEqual(first.overlay, second.overlay)
                self.assertTrue(first.image.startswith(str(root / "cache")))
                self.assertTrue(first.image_identity.startswith("sha256:"))
                exec_call = next(
                    call
                    for call in runner.calls
                    if "exec" in call[1:3] and "--overlay" in call
                )
                self.assertEqual(exec_call[1:3], ("--silent", "exec"))
                self.assertIn("--cleanenv", exec_call)
                self.assertIn("--containall", exec_call)
                self.assertIn("--fakeroot", exec_call)
                self.assertNotIn("HOME=/root", exec_call)
                self.assertTrue(
                    any(
                        "BASH_ENV=/root/.bashrc" in item for item in exec_call
                    )
                )
                await first.export_patch()
                stage_call = next(
                    call
                    for call in reversed(runner.calls)
                    if call[-1].startswith("git add")
                )
                self.assertNotIn("--force", stage_call[-1])
                await first.adopt_state(await first.export_state())
                reset_call = next(
                    call
                    for call in reversed(runner.calls)
                    if call[-1].startswith("git reset")
                )
                self.assertIn("git clean -ffd -q", reset_call[-1])
                self.assertNotIn("-ffdx", reset_call[-1])
                self.assertIn(first.base_commit, reset_call[-1])
            finally:
                await first.close()
                await second.close()

    async def test_apptainer_preflight_binding_rejects_changed_bytes(self) -> None:
        runner = RecordingRunner()
        instance = {
            "instance_id": "repo__issue-1",
            "image_name": "example/image:tag",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = await resolve_swebench_image_binding(
                instance,
                runtime="apptainer",
                apptainer_image_cache=root / "cache",
                runner=runner,
            )
            Path(binding.execution_ref).write_bytes(b"different-sif-content")
            with self.assertRaisesRegex(RuntimeError, "preflight binding"):
                await ApptainerSWEEnvironment.create(
                    instance,
                    image_binding=binding,
                    scratch_root=root / "scratch",
                    image_cache=root / "cache",
                    overlay_size_mib=1024,
                    runner=runner,
                )
            self.assertEqual(list((root / "scratch").iterdir()), [])

    async def test_apptainer_cache_lock_wait_is_bounded(self) -> None:
        runner = RecordingRunner()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("fcntl.flock", side_effect=BlockingIOError),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache lock"):
                await _materialize_apptainer_image(
                    "docker://example/image:tag",
                    executable="apptainer",
                    runner=runner,
                    cache=Path(temporary),
                    timeout_seconds=0.001,
                    max_output_bytes=1024,
                )
        self.assertEqual(runner.pull_calls, 0)


class FakeComputerClient:
    def __init__(self) -> None:
        self.actions: list[Sequence[Mapping[str, Any]]] = []
        self.done_calls = 0
        self.close_calls = 0

    async def observe(self) -> ComputerObservation:
        return ComputerObservation(png())

    async def step(self, actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        self.actions.append(actions)
        return {"done": False}

    async def done(self) -> None:
        self.done_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


class FakeResponse:
    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        content: bytes | None = None,
        status_code: int = 200,
    ) -> None:
        self.payload = payload or {}
        self.content = (
            content if content is not None else json.dumps(self.payload).encode()
        )
        self.is_error = status_code >= 400
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def aiter_bytes(self, *, chunk_size: int) -> Any:
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Mapping[str, Any]:
        return self.payload


class FakeHTTPClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str, Mapping[str, str], Any]] = []

    async def __aenter__(self) -> "FakeHTTPClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs
        self.requests.append((method, url, headers or {}, json))
        return self.responses.pop(0)


class ComputerEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def test_client_identity_is_not_stringified(self) -> None:
        client = FakeComputerClient()
        client.resource_identity = lambda: object()  # type: ignore[attr-defined]
        environment = CUAEnvironment(client)
        with self.assertRaisesRegex(RuntimeError, "resource_identity"):
            environment.resource_identity()

    def test_png_and_action_validation(self) -> None:
        self.assertEqual(validate_png(png(11, 7)), (11, 7))
        corrupt = bytearray(png())
        corrupt[-5] ^= 1
        with self.assertRaises(InfrastructureError):
            validate_png(bytes(corrupt))
        without_data = png()
        idat = without_data.index(b"IDAT")
        start = idat - 4
        length = struct.unpack(">I", without_data[start:idat])[0]
        finish = idat + 4 + length + 4
        with self.assertRaisesRegex(InfrastructureError, "incomplete PNG"):
            validate_png(without_data[:start] + without_data[finish:])

        actions = validate_computer_actions(
            [
                {"type": "click", "x": 1, "y": 2},
                {"type": "scroll", "dy": 4},
                {"type": "key", "keys": ["CTRL", "return"]},
            ],
            (8, 6),
        )
        self.assertEqual(actions[0]["button"], "left")
        self.assertEqual(actions[0]["clicks"], 1)
        self.assertEqual(actions[2]["keys"], ["ctrl", "enter"])
        with self.assertRaisesRegex(ProtocolError, "on-screen"):
            validate_computer_actions([{"type": "click", "x": 8, "y": 0}], (8, 6))
        with self.assertRaisesRegex(ProtocolError, "unsupported"):
            validate_computer_actions([{"type": []}], (8, 6))
        with self.assertRaisesRegex(ProtocolError, "1..32 keys"):
            validate_computer_actions(
                [{"type": "key", "keys": ["a"] * 33}], (8, 6)
            )

    def test_translation_contracts_are_explicit(self) -> None:
        actions = validate_computer_actions(
            [
                {"type": "click", "x": 1, "y": 2, "clicks": 2},
                {"type": "scroll", "dx": 3, "dy": 4},
            ],
            (8, 6),
        )
        translated = to_cua_speedrun_actions(actions)
        self.assertEqual(translated[0], {"mouse": {"double_click": [1, 2]}})
        self.assertIn({"mouse": {"scroll": 4}}, translated)
        self.assertEqual(
            encode_osworld_action(actions[1]),
            "pyautogui.scroll(-4)\npyautogui.hscroll(3)",
        )
        encoded = encode_osworld_action(
            validate_computer_actions(
                [{"type": "type", "text": "x'); os.system('bad"}], (8, 6)
            )[0]
        )
        self.assertTrue(encoded.startswith("pyautogui.write("))
        self.assertNotIn("\nos.system", encoded)
        unicode_action = validate_computer_actions(
            [{"type": "type", "text": "café 你好"}], (8, 6)
        )[0]
        unicode_encoded = encode_osworld_action(unicode_action)
        self.assertIn("import base64, pyperclip", unicode_encoded)
        self.assertNotIn("café", unicode_encoded)
        with self.assertRaisesRegex(ProtocolError, "Unicode scalar"):
            validate_computer_actions(
                [{"type": "type", "text": "bad\ud800"}], (8, 6)
            )

    async def test_one_computer_tool_observes_steps_and_finishes_once(self) -> None:
        client = FakeComputerClient()
        environment = CUAEnvironment(client)
        initial = await environment.initial_observation()
        self.assertTrue(initial.image_data_url.startswith("data:image/png;base64,"))
        self.assertFalse(json.loads(initial.output)["episode_done"])
        result = await environment.execute(
            ToolCall(
                "click",
                "computer",
                {"actions": [{"type": "click", "x": 1, "y": 2}]},
            )
        )
        self.assertEqual(json.loads(result.output)["executed"], 1)
        await environment.close()
        await environment.close()
        self.assertEqual((client.done_calls, client.close_calls), (1, 1))
        with self.assertRaisesRegex(RuntimeError, "finished"):
            await environment.execute(
                ToolCall(
                    "late",
                    "computer",
                    {"actions": [{"type": "screenshot"}]},
                )
            )

    async def test_episode_done_is_visible_and_blocks_more_actions(self) -> None:
        class DoneClient(FakeComputerClient):
            async def step(
                self, actions: Sequence[Mapping[str, Any]]
            ) -> Mapping[str, Any]:
                self.actions.append(actions)
                return {"done": True}

        environment = CUAEnvironment(DoneClient())
        await environment.initial_observation()
        result = await environment.execute(
            ToolCall(
                "click",
                "computer",
                {"actions": [{"type": "click", "x": 1, "y": 2}]},
            )
        )
        self.assertTrue(json.loads(result.output)["episode_done"])
        with self.assertRaisesRegex(ProtocolError, "episode is done"):
            await environment.execute(
                ToolCall(
                    "again",
                    "computer",
                    {"actions": [{"type": "click", "x": 1, "y": 2}]},
                )
            )
        await environment.close()

    async def test_generic_client_rejects_truthy_nonboolean_done(self) -> None:
        class InvalidDoneClient(FakeComputerClient):
            async def step(
                self, actions: Sequence[Mapping[str, Any]]
            ) -> Mapping[str, Any]:
                self.actions.append(actions)
                return {"done": 1}

        environment = CUAEnvironment(InvalidDoneClient())
        await environment.initial_observation()
        with self.assertRaisesRegex(InfrastructureError, "done must be a boolean"):
            await environment.execute(
                ToolCall(
                    "click",
                    "computer",
                    {"actions": [{"type": "click", "x": 1, "y": 2}]},
                )
            )

    async def test_computer_cleanup_can_be_retried_after_close_failure(self) -> None:
        class FlakyClient(FakeComputerClient):
            async def close(self) -> None:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("transient cleanup failure")

        client = FlakyClient()
        environment = CUAEnvironment(client)
        await environment.initial_observation()
        with self.assertRaisesRegex(RuntimeError, "transient"):
            await environment.close()
        await environment.close()
        self.assertEqual(client.done_calls, 1)
        self.assertEqual(client.close_calls, 2)

    async def test_computer_done_failure_does_not_skip_cleanup(self) -> None:
        class BrokenClient(FakeComputerClient):
            async def done(self) -> None:
                self.done_calls += 1
                raise RuntimeError("done failed")

            async def close(self) -> None:
                self.close_calls += 1
                raise RuntimeError("close failed")

        client = BrokenClient()
        environment = CUAEnvironment(client)
        with self.assertRaisesRegex(RuntimeError, "done failed.*close failed"):
            await environment.close()
        self.assertEqual((client.done_calls, client.close_calls), (1, 1))

    async def test_failed_done_is_not_reissued_during_cleanup(self) -> None:
        class BrokenDoneClient(FakeComputerClient):
            async def done(self) -> None:
                self.done_calls += 1
                raise RuntimeError("done failed")

        client = BrokenDoneClient()
        environment = CUAEnvironment(client)
        with self.assertRaisesRegex(RuntimeError, "done failed"):
            await environment.finish()
        with self.assertRaisesRegex(RuntimeError, "done failed"):
            await environment.close()
        self.assertEqual(client.done_calls, 1)
        self.assertEqual(client.close_calls, 1)

    async def test_gateway_client_uses_nested_upstream_action_schema(self) -> None:
        import base64

        http = FakeHTTPClient(
            [
                FakeResponse(
                    {
                        "png_b64": base64.b64encode(png()).decode("ascii"),
                        "meta": {"runner": "fixture"},
                    }
                ),
                FakeResponse({"ok": True, "info": {"done": True}}),
                FakeResponse({"ok": True}),
            ]
        )
        client = CUASpeedRunClient(
            "https://example.test/task-token",
            client=http,  # type: ignore[arg-type]
        )
        observed = await client.observe()
        self.assertEqual(validate_png(observed.png), (8, 6))
        self.assertEqual(observed.metadata["runner"], "fixture")
        actions = validate_computer_actions([{"type": "click", "x": 1, "y": 2}], (8, 6))
        result = await client.step(actions)
        self.assertTrue(result["done"])
        self.assertTrue(client.episode_done)
        await client.done()
        self.assertEqual(
            http.requests[1][3],
            {"actions": [{"mouse": {"left_click": [1, 2]}}]},
        )
        self.assertEqual(
            http.requests[0][2], {}
        )
        self.assertEqual(client.provenance()["authentication"], "url_path_token")
        self.assertTrue(http.requests[2][1].endswith("/done"))
        with self.assertRaisesRegex(ValueError, "origin"):
            CUASpeedRunClient("https://user:password@example.test")
        with self.assertRaisesRegex(ValueError, "invalid port"):
            CUASpeedRunClient("https://example.test:bad", bearer_token="secret")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            CUASpeedRunClient("http://example.test", bearer_token="secret")
        with self.assertRaisesRegex(ValueError, "run-path or bearer"):
            CUASpeedRunClient("https://example.test")
        with self.assertRaisesRegex(ValueError, "URL-safe"):
            CUASpeedRunClient("https://example.test/two/segments")
        bearer = CUASpeedRunClient(
            "https://example.test",
            bearer_token="task-secret",
            client=http,  # type: ignore[arg-type]
        )
        self.assertEqual(bearer.provenance()["authentication"], "bearer")
        self.assertFalse(
            CUASpeedRunClient(
                "http://127.0.0.1:8000", client=http  # type: ignore[arg-type]
            ).provenance()["authenticated"]
        )

    async def test_gateway_done_is_never_replayed_after_ambiguous_failure(self) -> None:
        class AmbiguousFailure:
            async def __aenter__(self) -> None:
                raise httpx.RemoteProtocolError("response connection failed")

            async def __aexit__(self, *args: object) -> None:
                del args

        class Client:
            def __init__(self) -> None:
                self.attempts = 0

            def stream(self, *args: object, **kwargs: object) -> AmbiguousFailure:
                del args, kwargs
                self.attempts += 1
                return AmbiguousFailure()

        http = Client()
        gateway = CUASpeedRunClient(
            "https://example.test/task-token",
            connect_retries=2,
            client=http,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(InfrastructureError, "done request failed"):
            await gateway.done()
        self.assertEqual(http.attempts, 1)

    async def test_gateway_stream_and_decoded_screenshot_limits_are_terminal(
        self,
    ) -> None:
        oversized_body = FakeHTTPClient([FakeResponse(content=b"123456789")])
        gateway = CUASpeedRunClient(
            "https://example.test",
            bearer_token="secret",
            client=oversized_body,  # type: ignore[arg-type]
        )
        with patch("mini_agent.environments.cua.MAX_CUA_OBSERVE_RESPONSE_BYTES", 8):
            with self.assertRaisesRegex(InfrastructureError, "byte limit"):
                await gateway.observe()

        redirect = FakeHTTPClient(
            [FakeResponse(content=b"{}", status_code=302)]
        )
        gateway = CUASpeedRunClient(
            "https://example.test",
            bearer_token="secret",
            client=redirect,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(InfrastructureError, "HTTP 302"):
            await gateway.done()

        screenshot = png()
        screenshot_body = FakeHTTPClient(
            [
                FakeResponse(
                    {
                        "png_b64": __import__("base64").b64encode(screenshot).decode(),
                        "meta": {},
                    }
                )
            ]
        )
        gateway = CUASpeedRunClient(
            "https://example.test",
            bearer_token="secret",
            client=screenshot_body,  # type: ignore[arg-type]
        )
        with patch(
            "mini_agent.environments.cua.MAX_SCREENSHOT_BYTES", len(screenshot) - 1
        ):
            with self.assertRaisesRegex(InfrastructureError, "decoded byte limit"):
                await gateway.observe()

    async def test_gateway_rejects_excessive_png_dimensions_as_infrastructure(
        self,
    ) -> None:
        import base64

        screenshot = png(8193, 1)
        http = FakeHTTPClient(
            [
                FakeResponse(
                    {
                        "png_b64": base64.b64encode(screenshot).decode(),
                        "meta": {},
                    }
                )
            ]
        )
        gateway = CUASpeedRunClient(
            "https://example.test",
            bearer_token="secret",
            client=http,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(InfrastructureError, "invalid PNG"):
            await gateway.observe()

    async def test_live_adapter_state_is_single_claim(self) -> None:
        state = AdapterLiveState(
            adapter=object(),
            observation=ComputerObservation(png()),
            resource_identity="resource",
        )
        state.claim()
        with self.assertRaisesRegex(ProtocolError, "already adopted"):
            state.claim()

    async def test_blocking_cancellation_waits_before_cleanup(self) -> None:
        started = threading.Event()
        finished = threading.Event()

        def blocking() -> None:
            started.set()
            time.sleep(0.05)
            finished.set()

        running = asyncio.create_task(complete_in_thread(blocking))
        await asyncio.to_thread(started.wait)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertTrue(finished.is_set())

    async def test_osworld_client_keeps_evaluator_fields_off_model_plane(self) -> None:
        class Desktop:
            def __init__(self) -> None:
                self.actions: list[str] = []

            def step(self, action: str, pause: float) -> tuple[Any, ...]:
                self.actions.append(action)
                return (
                    {"screenshot": png()},
                    0.75,
                    False,
                    {"hidden": "verifier-only"},
                )

        desktop = Desktop()
        client = OSWorldClient(
            desktop,
            {"screenshot": png()},
            owns_environment=False,
        )
        result = await client.step(
            validate_computer_actions([{"type": "click", "x": 1, "y": 2}], (8, 6))
        )
        self.assertIn("reward", result)
        observation = await client.observe()
        self.assertEqual(validate_png(observation.png), (8, 6))
        self.assertTrue(desktop.actions[0].startswith("pyautogui.click"))

    async def test_osworld_fail_is_explicit_terminal_action_only(self) -> None:
        class Desktop:
            def __init__(self) -> None:
                self.actions: list[str] = []

            def step(self, action: str, pause: float) -> tuple[Any, ...]:
                del pause
                self.actions.append(action)
                return ({"screenshot": png()}, 0.0, action == "FAIL", {"fail": True})

        with self.assertRaisesRegex(ProtocolError, "only by OSWorld"):
            validate_computer_actions([{"type": "fail"}], (8, 6))
        with self.assertRaisesRegex(ProtocolError, "only action"):
            validate_computer_actions(
                [
                    {"type": "fail"},
                    {"type": "click", "x": 1, "y": 2},
                ],
                (8, 6),
                allow_fail=True,
            )
        with self.assertRaisesRegex(ProtocolError, "duration.*only by OSWorld"):
            validate_computer_actions(
                [{"type": "move", "x": 1, "y": 2, "duration": 0.5}],
                (8, 6),
            )
        self.assertEqual(
            validate_computer_actions(
                [{"type": "move", "x": 1, "y": 2, "duration": 0.5}],
                (8, 6),
                allow_duration=True,
            )[0]["duration"],
            0.5,
        )

        desktop = Desktop()
        environment = OSWorldEnvironment(
            OSWorldClient(desktop, {"screenshot": png()}, owns_environment=False)
        )
        action_enum = environment.tools()[0].input_schema["properties"]["actions"][
            "items"
        ]["properties"]["type"]["enum"]
        self.assertIn("fail", action_enum)
        self.assertIn(
            "duration",
            environment.tools()[0].input_schema["properties"]["actions"]["items"][
                "properties"
            ],
        )
        self.assertNotIn(
            "fail",
            CUAEnvironment(FakeComputerClient()).tools()[0].input_schema["properties"][
                "actions"
            ]["items"]["properties"]["type"]["enum"],
        )
        self.assertNotIn(
            "duration",
            CUAEnvironment(FakeComputerClient()).tools()[0].input_schema["properties"]
            ["actions"]["items"]["properties"],
        )
        await environment.initial_observation()
        result = await environment.execute(
            ToolCall("fail", "computer", {"actions": [{"type": "fail"}]})
        )
        self.assertEqual(desktop.actions, ["FAIL"])
        self.assertTrue(json.loads(result.output)["episode_done"])

    async def test_osworld_screenshot_only_action_refreshes_the_desktop(self) -> None:
        class Desktop:
            def __init__(self) -> None:
                self.refreshes = 0

            def step(self, action: str, pause: float) -> tuple[Any, ...]:
                raise AssertionError(
                    f"screenshot action reached step: {action}, {pause}"
                )

            def _get_obs(self) -> Mapping[str, Any]:
                self.refreshes += 1
                return {"screenshot": png(9, 7)}

        desktop = Desktop()
        environment = CUAEnvironment(
            OSWorldClient(desktop, {"screenshot": png()}), benchmark="osworld"
        )
        await environment.initial_observation()
        result = await environment.execute(
            ToolCall(
                "shot",
                "computer",
                {"actions": [{"type": "screenshot"}]},
            )
        )
        self.assertEqual(desktop.refreshes, 1)
        self.assertEqual(json.loads(result.output)["width"], 9)

    async def test_osworld_rejects_truthy_nonboolean_done(self) -> None:
        class Desktop:
            def step(self, action: str, pause: float) -> tuple[Any, ...]:
                del action, pause
                return ({"screenshot": png()}, 0.0, "false", {})

        client = OSWorldClient(Desktop(), {"screenshot": png()}, owns_environment=False)
        with self.assertRaisesRegex(InfrastructureError, "done or info"):
            await client.step(
                validate_computer_actions([{"type": "click", "x": 1, "y": 2}], (8, 6))
            )


if __name__ == "__main__":
    unittest.main()
