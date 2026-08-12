"""One-tool browser environment for live and fixed-corpus research."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import inspect
import ipaddress
import json
import math
import os
import re
import socket
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Mapping, Protocol, Sequence, cast
from urllib.parse import urljoin, urlsplit

import httpx

from .._hash import stable_file_sha256
from .._http import ResponseBodyTooLarge, read_bounded_body
from ..types import (
    InfrastructureError,
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    _json_value,
    strict_json_loads,
)
from .base import (
    BaseEnvironment,
    combine_lifecycle_errors,
    raise_lifecycle_errors,
)


BROWSECOMP_PLUS_REVISION = "046949032b0328319cc9a02663a759ec601d9402"
BROWSECOMP_PLUS_INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
ANSERINI_VERSION = "1.1.1"
ANSERINI_JAR_SHA256 = "69270ba4d160826953347411ce5d7e205ce363766a9cc72ac9da3b945341af83"
PYSERINI_REFERENCE_VERSION = "1.2.0"
PYJNIUS_VERSION = "1.6.1"
HUGGINGFACE_HUB_VERSION = "0.33.4"
TOKENIZERS_VERSION = "0.21.2"
MAX_SERPAPI_RESPONSE_BYTES = 4 * 1024 * 1024
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


class SearchBackend(Protocol):
    def search(
        self, query: str, k: int = 5
    ) -> Sequence[Mapping[str, Any]] | Awaitable[Sequence[Mapping[str, Any]]]: ...

    def open(
        self, reference: str
    ) -> Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]: ...


class SearchBackendStateTransfer(Protocol):
    """Optional reference-authorization state used by session-bound backends."""

    def export_reference_state(
        self,
    ) -> Sequence[str] | Awaitable[Sequence[str]]: ...

    def replace_reference_state(
        self, references: Sequence[str]
    ) -> None | Awaitable[None]: ...


class SnippetTokenizer(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[Any]: ...

    def decode(self, tokens: Sequence[Any], *, skip_special_tokens: bool) -> str: ...


class PageReader(Protocol):
    async def open(self, url: str) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class JsonlSearchBackend:
    """Deterministic BM25 for tests and small fixed corpora."""

    def __init__(self, corpus: Path) -> None:
        expanded = corpus.expanduser()
        if expanded.is_symlink():
            raise ValueError("corpus must be a regular non-symlink file")
        self.corpus = expanded.resolve()
        if not self.corpus.is_file():
            raise ValueError(f"corpus does not exist: {self.corpus}")
        raw = self.corpus.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("corpus must be UTF-8 JSONL") from exc
        self.documents: dict[str, str] = {}
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"corpus line {number} is invalid JSON") from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"corpus line {number} must be an object")
            docid = item.get("docid", item.get("id"))
            text = item.get("text", item.get("contents"))
            if (
                not isinstance(docid, (str, int))
                or isinstance(docid, bool)
                or not str(docid)
                or not isinstance(text, str)
            ):
                raise ValueError(f"corpus line {number} requires docid and text")
            key = str(docid)
            if key in self.documents:
                raise ValueError(f"duplicate corpus docid {key!r}")
            self.documents[key] = text
        if not self.documents:
            raise ValueError("corpus must contain at least one document")
        self._terms = {
            key: Counter(self._tokenize(text)) for key, text in self.documents.items()
        }
        self._lengths = {key: sum(terms.values()) for key, terms in self._terms.items()}
        self._average = sum(self._lengths.values()) / len(self._lengths)
        self._frequency: Counter[str] = Counter()
        for terms in self._terms.values():
            self._frequency.update(terms.keys())

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return [token.casefold() for token in _TOKEN.findall(value)]

    def search(self, query: str, k: int = 5) -> Sequence[Mapping[str, Any]]:
        terms = self._tokenize(query)
        scores: list[tuple[float, str]] = []
        for docid, frequencies in self._terms.items():
            score = 0.0
            for term in terms:
                count = frequencies.get(term, 0)
                if not count:
                    continue
                frequency = self._frequency[term]
                inverse = math.log(
                    1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5)
                )
                normalizer = count + 1.2 * (
                    0.25 + 0.75 * self._lengths[docid] / max(self._average, 1)
                )
                score += inverse * count * 2.2 / normalizer
            if score:
                scores.append((score, docid))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "ref": docid,
                "title": docid,
                "score": score,
                "snippet": self.documents[docid],
            }
            for score, docid in scores[:k]
        ]

    def open(self, reference: str) -> Mapping[str, Any] | None:
        text = self.documents.get(reference)
        return None if text is None else {"ref": reference, "text": text}

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": "jsonl_bm25_fixture",
            "corpus": str(self.corpus),
            "sha256": self.sha256,
            "documents": len(self.documents),
        }


class BrowseCompPlusBackend:
    """Thin adapter around BrowseComp-Plus's pinned Lucene index."""

    def __init__(
        self,
        index_path: Path,
        anserini_jar: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        expanded_index = index_path.expanduser()
        if expanded_index.is_symlink():
            raise ValueError("Lucene index root must not be a symlink")
        self.index_path = expanded_index.resolve()
        if not self.index_path.is_dir():
            raise ValueError(f"Lucene index does not exist: {self.index_path}")
        if expected_sha256 is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_sha256
        ):
            raise ValueError("expected BrowseComp-Plus index SHA-256 must be hex")
        self.sha256 = directory_sha256(self.index_path)
        if expected_sha256 is not None and self.sha256 != expected_sha256.casefold():
            raise ValueError(
                "BrowseComp-Plus index content hash differs from --index-sha256"
            )
        self.anserini_jar, self.anserini_jar_sha256 = validate_anserini_jar(
            anserini_jar
        )
        self.searcher = _lucene_searcher(self.index_path, self.anserini_jar)

    def search(self, query: str, k: int = 5) -> Sequence[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        for hit in self.searcher.search(query, k):
            docid = getattr(hit, "docid", None)
            score = getattr(hit, "score", None)
            document = getattr(hit, "lucene_document", None)
            if (
                not isinstance(docid, (str, int))
                or isinstance(docid, bool)
                or not str(docid)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or document is None
                or not callable(getattr(document, "get", None))
            ):
                raise InfrastructureError(
                    "BrowseComp-Plus search returned a malformed hit"
                )
            raw = _lucene_document(document.get("raw"))
            results.append(
                {
                    "ref": str(docid),
                    "docid": str(docid),
                    "score": float(score),
                    "snippet": str(raw["contents"]),
                }
            )
        return results

    def open(self, reference: str) -> Mapping[str, Any] | None:
        document = self.searcher.doc(reference)
        if document is None:
            return None
        return {
            "ref": reference,
            "docid": reference,
            "text": _lucene_document(document.get("raw"))["contents"],
        }

    def close(self) -> None:
        close = getattr(self.searcher, "close", None)
        if not callable(close):
            raise RuntimeError("BrowseComp-Plus searcher does not expose close")
        close()

    def provenance(self) -> Mapping[str, Any]:
        try:
            pyjnius_version = importlib.metadata.version("pyjnius")
        except importlib.metadata.PackageNotFoundError:
            pyjnius_version = "unknown"
        return {
            "backend": "browsecomp_plus_lucene",
            "index": str(self.index_path),
            "index_sha256": self.sha256,
            "anserini_jar": str(self.anserini_jar),
            "anserini_jar_sha256": self.anserini_jar_sha256,
            "anserini_version": ANSERINI_VERSION,
            "pyjnius": pyjnius_version,
            "pyserini_reference_version": PYSERINI_REFERENCE_VERSION,
            "source_revision": BROWSECOMP_PLUS_REVISION,
        }


def validate_anserini_jar(path: Path) -> tuple[Path, str]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("Anserini fat JAR must not be a symlink")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"Anserini fat JAR does not exist: {resolved}")
    digest = _file_sha256(resolved)
    if digest != ANSERINI_JAR_SHA256:
        raise ValueError(
            "Anserini fat JAR hash does not match the pinned 1.1.1 artifact"
        )
    return resolved, digest


def _lucene_searcher(index_path: Path, anserini_jar: Path) -> Any:
    try:
        pyjnius_version = importlib.metadata.version("pyjnius")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"install pyjnius=={PYJNIUS_VERSION} for Lucene retrieval"
        ) from exc
    if pyjnius_version != PYJNIUS_VERSION:
        raise RuntimeError(
            f"Lucene retrieval requires pyjnius=={PYJNIUS_VERSION}, "
            f"found {pyjnius_version}"
        )
    try:
        import jnius_config  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "install mini-agent[web-fixed] for Lucene retrieval"
        ) from exc
    if jnius_config.vm_running:
        configured = {
            Path(item).expanduser().resolve()
            for item in jnius_config.get_classpath()
            if "*" not in item
        }
        if anserini_jar not in configured:
            raise RuntimeError(
                "the running JVM was started without the pinned Anserini JAR"
            )
    else:
        jnius_config.set_classpath(str(anserini_jar))
    try:
        from jnius import autoclass  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "install mini-agent[web-fixed] for Lucene retrieval"
        ) from exc
    searcher = autoclass("io.anserini.search.SimpleSearcher")(str(index_path))
    searcher.set_bm25(0.9, 0.4)
    return searcher


def _lucene_document(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise InfrastructureError("BrowseComp-Plus index document has no raw field")
    try:
        document = strict_json_loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InfrastructureError(
            "BrowseComp-Plus index document has invalid JSON"
        ) from exc
    if not isinstance(document, Mapping) or not isinstance(
        document.get("contents"), str
    ):
        raise InfrastructureError("BrowseComp-Plus index document has no contents")
    return document


class SerpAPIBackend:
    """Live Google results via SerpAPI, with bounded page reads."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30,
        max_page_bytes: int = 2 * 1024 * 1024,
        max_response_bytes: int = MAX_SERPAPI_RESPONSE_BYTES,
        page_reader: PageReader | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
            or not isinstance(max_page_bytes, int)
            or isinstance(max_page_bytes, bool)
            or max_page_bytes < 1
            or not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes < 1
        ):
            raise ValueError("SerpAPI timeout and page-byte limits must be positive")
        resolved_key = (
            api_key if api_key is not None else os.environ.get("SERPAPI_API_KEY", "")
        )
        if not isinstance(resolved_key, str) or not resolved_key:
            raise ValueError("SerpAPI requires SERPAPI_API_KEY")
        if page_reader is not None and any(
            not callable(getattr(page_reader, name, None)) for name in ("open", "close")
        ):
            raise ValueError("page_reader must expose open and close")
        self.api_key = resolved_key
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes
        self.max_response_bytes = max_response_bytes
        self.page_reader = page_reader or HttpPageReader(
            timeout_seconds=timeout_seconds,
            max_page_bytes=max_page_bytes,
        )
        self._urls: set[str] = set()

    async def search(self, query: str, k: int = 5) -> Sequence[Mapping[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ProtocolError("SerpAPI query must be non-empty")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ProtocolError("SerpAPI result count must be positive")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "GET",
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google",
                        "q": query,
                        "api_key": self.api_key,
                    },
                ) as response:
                    if not 200 <= response.status_code < 300:
                        raise InfrastructureError(
                            f"SerpAPI returned HTTP {response.status_code}"
                        )
                    content = await read_bounded_body(
                        response, self.max_response_bytes
                    )
        except InfrastructureError:
            raise
        except ResponseBodyTooLarge as exc:
            raise InfrastructureError(
                "SerpAPI response exceeds the configured byte limit"
            ) from exc
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise InfrastructureError("SerpAPI request failed") from exc
        try:
            payload = strict_json_loads(content)
        except ValueError as exc:
            raise InfrastructureError("SerpAPI returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise InfrastructureError("SerpAPI response must be an object")
        error = payload.get("error")
        if error is not None:
            if not isinstance(error, str):
                raise InfrastructureError("SerpAPI returned a malformed API error")
            if error:
                raise InfrastructureError("SerpAPI reported an API error")
        search_metadata = payload.get("search_metadata")
        if not isinstance(search_metadata, Mapping):
            raise InfrastructureError("SerpAPI returned malformed search_metadata")
        if search_metadata.get("status") != "Success":
            raise InfrastructureError("SerpAPI search did not complete successfully")
        organic = payload.get("organic_results", [])
        if not isinstance(organic, list):
            raise InfrastructureError("SerpAPI returned malformed organic_results")
        results: list[Mapping[str, Any]] = []
        for item in organic:
            if not isinstance(item, Mapping) or not isinstance(item.get("link"), str):
                raise InfrastructureError("SerpAPI returned a malformed organic result")
            url = item["link"]
            try:
                await _validate_public_url(url)
            except ProtocolError:
                continue
            self._urls.add(url)
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if not isinstance(title, str) or not isinstance(snippet, str):
                raise InfrastructureError("SerpAPI result text fields must be strings")
            results.append(
                {
                    "ref": url,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                }
            )
            if len(results) == k:
                break
        return results

    async def open(self, reference: str) -> Mapping[str, Any] | None:
        if reference not in self._urls:
            raise ProtocolError(
                "open accepts only a URL returned by this browser session"
            )
        return await self.page_reader.open(reference)

    def export_reference_state(self) -> tuple[str, ...]:
        return tuple(sorted(self._urls))

    def replace_reference_state(self, references: Sequence[str]) -> None:
        if not isinstance(references, Sequence) or isinstance(
            references, (str, bytes)
        ):
            raise ProtocolError("browser backend reference state must be a sequence")
        replacement: set[str] = set()
        for reference in references:
            if not isinstance(reference, str) or not reference:
                raise ProtocolError(
                    "browser backend reference state must contain non-empty strings"
                )
            replacement.add(reference)
        self._urls = replacement

    async def close(self) -> None:
        await self.page_reader.close()

    def provenance(self) -> Mapping[str, Any]:
        reader = getattr(self.page_reader, "provenance", None)
        return {
            "backend": "serpapi",
            "engine": "google",
            "max_response_bytes": self.max_response_bytes,
            "page_reader": dict(reader()) if callable(reader) else {},
        }


class HttpPageReader:
    """Bounded static HTTP page reader used when a browser is unnecessary."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30,
        max_page_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
            or not isinstance(max_page_bytes, int)
            or isinstance(max_page_bytes, bool)
            or max_page_bytes < 1
        ):
            raise ValueError("HTTP timeout and page-byte limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes

    async def open(self, url: str) -> Mapping[str, Any]:
        current = url
        try:
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=self.timeout_seconds
            ) as client:
                for redirect in range(6):
                    try:
                        await _validate_public_url(current)
                    except ProtocolError as exc:
                        if redirect == 0:
                            raise
                        raise InfrastructureError(
                            "page redirected to a non-public URL"
                        ) from exc
                    async with client.stream("GET", current) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise InfrastructureError(
                                    "page redirect has no location"
                                )
                            current = urljoin(str(response.url), location)
                            if redirect == 5:
                                raise InfrastructureError(
                                    "page exceeded the redirect limit"
                                )
                            continue
                        status = response.status_code
                        if (
                            not isinstance(status, int)
                            or isinstance(status, bool)
                            or not 200 <= status < 300
                        ):
                            raise InfrastructureError(
                                f"page returned HTTP {status}"
                            )
                        body = await read_bounded_body(
                            response, self.max_page_bytes
                        )
                        encoding = response.encoding or "utf-8"
                        content_type = response.headers.get("content-type", "")
                        final_url = str(response.url)
                        break
                else:  # pragma: no cover - loop always exits or raises
                    raise AssertionError("unreachable redirect state")
        except (InfrastructureError, ProtocolError):
            raise
        except ResponseBodyTooLarge as exc:
            raise InfrastructureError(
                "page exceeds the configured byte limit"
            ) from exc
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise InfrastructureError("page request failed") from exc
        try:
            content = body.decode(encoding, errors="replace")
        except (LookupError, TypeError) as exc:
            raise InfrastructureError("page returned an invalid text encoding") from exc
        text = _html_to_text(content) if "html" in content_type.casefold() else content
        return {"ref": url, "url": final_url, "text": text}

    async def close(self) -> None:
        return None

    def provenance(self) -> Mapping[str, Any]:
        return {
            "reader": "httpx",
            "max_page_bytes": self.max_page_bytes,
        }


class PlaywrightPageReader:
    """One isolated Chromium context, created lazily per research agent."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30,
        max_page_chars: int = 2 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            not isinstance(max_page_chars, int)
            or isinstance(max_page_chars, bool)
            or max_page_chars < 1
        ):
            raise ValueError("max_page_chars must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_page_chars = max_page_chars
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "install mini-agent[web-live] and run playwright install chromium"
            ) from exc
        operation_error: BaseException | None = None
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.timeout_seconds * 1000)
            return self._page
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await self.close()
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors(
            "Playwright browser startup", operation_error, cleanup_error
        )
        raise AssertionError("unreachable")

    async def open(self, url: str) -> Mapping[str, Any]:
        await _validate_public_url(url)
        page = await self._ensure_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:
            raise InfrastructureError("Playwright page request failed") from exc
        if response is not None:
            status = getattr(response, "status", None)
            if (
                not isinstance(status, int)
                or isinstance(status, bool)
                or not 200 <= status < 300
            ):
                raise InfrastructureError(
                    "Playwright page returned an unsuccessful HTTP status"
                )
        try:
            await _validate_public_url(page.url)
        except ProtocolError as exc:
            raise InfrastructureError(
                "Playwright page redirected to a non-public URL"
            ) from exc
        try:
            extracted = await page.evaluate(
                """limit => ({
                    title: document.title.slice(0, Math.min(limit, 4096)),
                    text: document.body === null
                        ? ""
                        : document.body.innerText.slice(0, limit),
                })""",
                self.max_page_chars,
            )
        except Exception as exc:
            raise InfrastructureError("Playwright page extraction failed") from exc
        if (
            not isinstance(extracted, Mapping)
            or not isinstance(extracted.get("title"), str)
            or not isinstance(extracted.get("text"), str)
            or len(extracted["title"]) > min(self.max_page_chars, 4096)
            or len(extracted["text"]) > self.max_page_chars
        ):
            raise InfrastructureError("Playwright returned malformed page text")
        return {
            "ref": url,
            "url": page.url,
            "title": extracted["title"],
            "text": extracted["text"],
        }

    async def close(self) -> None:
        error: BaseException | None = None
        for attribute, method in (
            ("_context", "close"),
            ("_browser", "close"),
            ("_playwright", "stop"),
        ):
            resource = getattr(self, attribute)
            if resource is None:
                continue
            try:
                await getattr(resource, method)()
            except BaseException as exc:
                error = combine_lifecycle_errors(error, exc)
            else:
                setattr(self, attribute, None)
        if self._context is None and self._browser is None:
            self._page = None
        if error is not None:
            raise error

    def provenance(self) -> Mapping[str, Any]:
        return {
            "reader": "playwright",
            "browser": "chromium",
            "isolated_context": True,
            "untrusted_content_security_boundary": False,
            "max_page_chars": self.max_page_chars,
        }


@dataclass(frozen=True)
class BrowserSessionState:
    """References a completed browser session proved it was shown.

    Adopting this state lets the recipient ``open`` the descendant's
    discovered references; answers still travel through the mailbox.
    """

    backend_identity: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_identity, str) or not self.backend_identity:
            raise ValueError("browser state backend identity must be non-empty")
        if not isinstance(self.references, tuple) or not all(
            isinstance(reference, str) and reference
            for reference in self.references
        ):
            raise ValueError("browser state references must be non-empty strings")


class BrowserEnvironment(BaseEnvironment):
    """Expose search and open as actions of one model-facing tool."""

    def __init__(
        self,
        backend: SearchBackend,
        *,
        top_k: int = 5,
        max_observation_chars: int | None = 16_384,
        snippet_tokens: int | None = None,
        tokenizer: SnippetTokenizer | None = None,
        allow_open: bool = True,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if max_observation_chars is not None and (
            not isinstance(max_observation_chars, int)
            or isinstance(max_observation_chars, bool)
            or max_observation_chars < 128
        ):
            raise ValueError("max_observation_chars must be None or at least 128")
        if (snippet_tokens is None) != (tokenizer is None):
            raise ValueError("snippet_tokens and tokenizer must be provided together")
        if snippet_tokens is not None and (
            not isinstance(snippet_tokens, int)
            or isinstance(snippet_tokens, bool)
            or snippet_tokens < 1
        ):
            raise ValueError("snippet_tokens must be a positive integer")
        if not isinstance(allow_open, bool):
            raise ValueError("allow_open must be boolean")
        for name in (("search", "open") if allow_open else ("search",)):
            if not callable(getattr(backend, name, None)):
                raise ValueError(f"search backend must expose {name}")
        if tokenizer is not None and any(
            not callable(getattr(tokenizer, name, None))
            for name in ("encode", "decode")
        ):
            raise ValueError("tokenizer must expose encode and decode")
        self.backend = backend
        self.top_k = top_k
        self.max_observation_chars = max_observation_chars
        self.snippet_tokens = snippet_tokens
        self.tokenizer = tokenizer
        self.allow_open = allow_open
        self._references: set[str] = set()
        self._counts: Counter[str] = Counter()
        self._closed = False

    def tools(self) -> Sequence[ToolDefinition]:
        actions = ["search", "open"] if self.allow_open else ["search"]
        properties: dict[str, Any] = {
            "action": {"type": "string", "enum": actions},
            "query": {"type": "string", "description": "Search query string"},
        }
        if self.allow_open:
            properties["ref"] = {"type": "string"}
        return (
            ToolDefinition(
                name="browser",
                description=(
                    "Search the web/corpus or open a result from this session."
                    if self.allow_open
                    else (
                        "Perform a search on a knowledge source. Returns "
                        f"top-{self.top_k} hits with docid, score, and snippet. "
                        "The snippet contains the document's contents (may be "
                        "truncated based on token limits)."
                    )
                ),
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": (
                        ["action"] if self.allow_open else ["action", "query"]
                    ),
                    "additionalProperties": False,
                },
            ),
        )

    async def execute(self, call: ToolCall) -> ToolExecution:
        if self._closed:
            raise RuntimeError("browser environment is closed")
        if call.name != "browser":
            raise InvalidAction(f"unsupported web tool {call.name!r}")
        action = call.arguments.get("action")
        if action == "search":
            query = _nonempty(call.arguments.get("query"), "browser search query")
            if "ref" in call.arguments:
                raise InvalidAction("browser search does not accept ref")
            self._counts["search"] += 1
            try:
                results = await _await(self.backend.search(query, self.top_k))
            except ProtocolError as exc:
                raise InfrastructureError("search backend failed") from exc
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
                raise InfrastructureError(
                    "search backend must return a result sequence"
                )
            rendered: list[dict[str, Any]] = []
            references: set[str] = set()
            for result in results[: self.top_k]:
                if not isinstance(result, Mapping):
                    raise InfrastructureError("search backend results must be objects")
                raw_reference = result.get("ref", result.get("url"))
                if (
                    not isinstance(raw_reference, (str, int))
                    or isinstance(raw_reference, bool)
                    or not str(raw_reference)
                ):
                    raise InfrastructureError(
                        "search backend returned a result without ref"
                    )
                reference = str(raw_reference)
                references.add(reference)
                item = dict(result)
                item["ref"] = reference
                if not self.allow_open and "docid" in item:
                    # The fixed BrowseComp-Plus search tool returns docid, score,
                    # and snippet. Its internal session reference is not another
                    # model-facing field when document opening is disabled.
                    item.pop("ref")
                if "snippet" in item:
                    if not isinstance(item["snippet"], str):
                        raise InfrastructureError(
                            "search result snippet must be a string"
                        )
                    item["snippet"] = self._truncate(item["snippet"])
                rendered.append(item)
            output = self._render(rendered)
            self._references.update(references)
            return ToolExecution(
                output=output,
                metadata={
                    "action": "search",
                    "result_count": len(rendered),
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                },
            )
        if action == "open":
            if not self.allow_open:
                raise InvalidAction("browser open is not enabled")
            reference = _nonempty(call.arguments.get("ref"), "browser open ref")
            if "query" in call.arguments:
                raise InvalidAction("browser open does not accept query")
            if reference not in self._references:
                raise InvalidAction(
                    "browser open ref was not returned by this session"
                )
            self._counts["open"] += 1
            try:
                document = await _await(self.backend.open(reference))
            except ProtocolError as exc:
                raise InfrastructureError("open backend failed") from exc
            if document is None:
                return ToolExecution(
                    output=json.dumps({"error": "result is unavailable"}), is_error=True
                )
            if not isinstance(document, Mapping):
                raise InfrastructureError("open backend result must be an object")
            return ToolExecution(
                output=self._render(dict(document)),
                metadata={"action": "open", "ref": reference},
            )
        raise InvalidAction(f"unsupported browser action {action!r}")

    def accounting(self) -> Mapping[str, Any]:
        return {
            "tool_calls": dict(sorted(self._counts.items())),
            "references": sorted(self._references),
        }

    def provenance(self) -> Mapping[str, Any]:
        hook = getattr(self.backend, "provenance", None)
        tokenizer: Mapping[str, Any] | None = None
        if self.tokenizer is not None:
            init = getattr(self.tokenizer, "init_kwargs", {})
            tokenizer = {
                "name": getattr(self.tokenizer, "name_or_path", None),
                "class": type(self.tokenizer).__name__,
                "revision": (
                    init.get("_commit_hash") if isinstance(init, Mapping) else None
                ),
                "tokenizer_json_sha256": getattr(
                    self.tokenizer, "tokenizer_json_sha256", None
                ),
            }
        return {
            "environment": "web",
            "tool": "browser",
            "actions": ["search", "open"] if self.allow_open else ["search"],
            "top_k": self.top_k,
            "max_observation_chars": self.max_observation_chars,
            "snippet_tokens": self.snippet_tokens,
            "tokenizer": tokenizer,
            "backend": dict(hook()) if callable(hook) else {},
        }

    def _backend_identity(self) -> str | None:
        hook = getattr(self.backend, "provenance", None)
        if not callable(hook):
            return None
        canonical = json.dumps(
            _json_value(dict(hook()), "browser backend provenance"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"browser-backend:{digest}"

    def _backend_state_transfer(self) -> SearchBackendStateTransfer | None:
        export = getattr(self.backend, "export_reference_state", None)
        replace = getattr(self.backend, "replace_reference_state", None)
        if export is None and replace is None:
            return None
        if not callable(export) or not callable(replace):
            raise RuntimeError(
                "stateful search backends must expose callable "
                "export_reference_state and replace_reference_state"
            )
        return cast(SearchBackendStateTransfer, self.backend)

    @staticmethod
    def _validated_reference_state(value: Any) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ProtocolError("browser backend reference state must be a sequence")
        references = tuple(value)
        if not all(
            isinstance(reference, str) and reference for reference in references
        ):
            raise ProtocolError(
                "browser backend reference state must contain non-empty strings"
            )
        return tuple(sorted(set(references)))

    async def export_state(self) -> BrowserSessionState | None:
        if self._closed:
            raise RuntimeError("browser environment is closed")
        identity = self._backend_identity()
        if identity is None:
            return None
        return BrowserSessionState(identity, tuple(sorted(self._references)))

    async def adopt_state(self, state: Any) -> None:
        if self._closed:
            raise RuntimeError("browser environment is closed")
        if not isinstance(state, BrowserSessionState):
            raise ProtocolError("browser state has an incompatible type")
        identity = self._backend_identity()
        if identity is None or identity != state.backend_identity:
            raise ProtocolError(
                "browser state came from a different search backend"
            )
        transfer = self._backend_state_transfer()
        if transfer is None:
            self._references.update(state.references)
            return

        backend_before = self._validated_reference_state(
            await _await(transfer.export_reference_state())
        )
        wrapper_before = set(self._references)
        combined = tuple(sorted(set(backend_before).union(state.references)))
        try:
            await _await(transfer.replace_reference_state(combined))
            self._references.update(state.references)
        except BaseException as operation_error:
            self._references = wrapper_before
            rollback_error: BaseException | None = None
            try:
                await _await(transfer.replace_reference_state(backend_before))
            except BaseException as exc:
                rollback_error = exc
            raise_lifecycle_errors(
                "browser state adoption", operation_error, rollback_error
            )

    async def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.backend, "close", None)
        if close is not None:
            if not callable(close):
                raise RuntimeError("search backend close must be callable")
            await _await(close())
        self._closed = True

    def _truncate(self, value: str) -> str:
        if self.snippet_tokens is not None:
            assert self.tokenizer is not None
            tokens = list(
                self.tokenizer.encode(
                    value,
                    add_special_tokens=False,
                )
            )
            if len(tokens) > self.snippet_tokens:
                value = self.tokenizer.decode(
                    tokens[: self.snippet_tokens], skip_special_tokens=True
                )
                if not isinstance(value, str):
                    raise InfrastructureError("tokenizer decode must return a string")
        return (
            value
            if self.max_observation_chars is None
            else value[: self.max_observation_chars]
        )

    def _render(self, value: Any) -> str:
        try:
            stable = _json_value(value, "browser backend result")
            rendered = json.dumps(
                stable, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise InfrastructureError("browser backend returned non-JSON data") from exc
        if (
            self.max_observation_chars is None
            or len(rendered) <= self.max_observation_chars
        ):
            return rendered
        max_observation_chars = self.max_observation_chars
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        low = 0
        high = max_observation_chars
        while low < high:
            middle = (low + high + 1) // 2
            candidate = json.dumps(
                {
                    "preview": rendered[:middle],
                    "sha256": digest,
                    "truncated": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if len(candidate) <= max_observation_chars:
                low = middle
            else:
                high = middle - 1
        return json.dumps(
            {
                "preview": rendered[:low],
                "sha256": digest,
                "truncated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )




async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidAction(f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > 16 * 1024:
        raise InvalidAction(f"{name} exceeds 16384 bytes")
    return value


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    # HTMLParser already resolves character references by default. A second
    # unescape would corrupt literal entity text such as ``&amp;lt;``.
    text = " ".join(parser.parts)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


async def _validate_public_url(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 16 * 1024
        or any(ord(character) < 32 for character in value)
    ):
        raise ProtocolError("browser URL is invalid or too long")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProtocolError("browser URLs must be public HTTP(S) URLs")
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ProtocolError("browser URLs cannot target localhost")
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise ProtocolError("browser URL has an invalid port") from exc
    if explicit_port is not None and explicit_port < 1:
        raise ProtocolError("browser URL has an invalid port")
    port = explicit_port if explicit_port is not None else (
        443 if parsed.scheme == "https" else 80
    )
    try:
        records = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ProtocolError("browser URL hostname could not be resolved") from exc
    addresses = {str(record[4][0]).split("%", 1)[0] for record in records}
    if not addresses or any(
        not ipaddress.ip_address(address).is_global for address in addresses
    ):
        raise ProtocolError("browser URL resolves to a non-public address")


def directory_sha256(path: Path) -> str:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("Lucene index root must not be a symlink")
    path = expanded.resolve()
    if not path.is_dir():
        raise ValueError(f"Lucene index is not a directory: {path}")
    digest = hashlib.sha256()
    entries = sorted(path.rglob("*"))
    links = [item for item in entries if item.is_symlink()]
    if links:
        raise ValueError(f"Lucene index must not contain symlinks: {links[0]}")
    special = [item for item in entries if not item.is_dir() and not item.is_file()]
    if special:
        raise ValueError(f"Lucene index contains a special file: {special[0]}")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise ValueError(f"Lucene index contains no files: {path}")
    identities: dict[Path, tuple[int, int, int, int]] = {}
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest.update(before.st_size.to_bytes(8, "big"))
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"Lucene index changed while hashing: {item}")
        identities[item] = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    if sorted(path.rglob("*")) != entries:
        raise RuntimeError("Lucene index changed while hashing")
    for item, identity in identities.items():
        current = item.stat()
        if identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise RuntimeError(f"Lucene index changed while hashing: {item}")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return stable_file_sha256(path, label="hash input")


__all__ = [
    "ANSERINI_JAR_SHA256",
    "ANSERINI_VERSION",
    "BROWSECOMP_PLUS_REVISION",
    "BROWSECOMP_PLUS_INDEX_REVISION",
    "HUGGINGFACE_HUB_VERSION",
    "PYJNIUS_VERSION",
    "PYSERINI_REFERENCE_VERSION",
    "TOKENIZERS_VERSION",
    "BrowseCompPlusBackend",
    "BrowserEnvironment",
    "BrowserSessionState",
    "JsonlSearchBackend",
    "MAX_SERPAPI_RESPONSE_BYTES",
    "HttpPageReader",
    "PageReader",
    "PlaywrightPageReader",
    "SearchBackend",
    "SearchBackendStateTransfer",
    "SerpAPIBackend",
    "SnippetTokenizer",
    "directory_sha256",
    "validate_anserini_jar",
]
