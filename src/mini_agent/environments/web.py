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

from .._hash import stable_file_sha256, stat_key
from .._http import ResponseBodyTooLarge, read_bounded_body
from ..types import (
    InfrastructureError,
    InvalidAction,
    ProtocolError,
    ToolCall,
    ToolDefinition,
    ToolExecution,
    _json_value,
    _require_bool,
    _require_callable,
    _require_finite_number,
    _require_int,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
    _require_tuple_of,
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
_WEB_FIXED_HINT = "install mini-agent[web-fixed] for Lucene retrieval"
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


class SnippetTokenizerAdapter:
    """Expose the two upstream snippet operations without a model framework.

    ``name_or_path`` and ``init_kwargs`` deliberately mirror the Transformers
    tokenizer attributes upstream would expose, so :meth:`WebEnvironment.
    provenance` records the same identity for either object.
    """

    def __init__(
        self,
        backend: Any,
        *,
        name: str,
        revision: str | None,
        tokenizer_json_sha256: str,
    ) -> None:
        self._backend = backend
        self.name_or_path = name
        self.init_kwargs = {"_commit_hash": revision}
        self.tokenizer_json_sha256 = tokenizer_json_sha256

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]:
        return list(
            self._backend.encode(text, add_special_tokens=add_special_tokens).ids
        )

    def decode(self, tokens: Sequence[Any], *, skip_special_tokens: bool) -> str:
        return self._backend.decode(
            list(tokens), skip_special_tokens=skip_special_tokens
        )


def commit_hash(value: Any) -> str | None:
    """Return the casefolded value when it names a full 40-character commit."""

    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value):
        return value.casefold()
    return None


def load_snippet_tokenizer(
    repo_id: str, revision: Any, *, require_revision: bool = False
) -> tuple[SnippetTokenizerAdapter, str | None, str]:
    """Download, version-pin, and hash the tokenizer the fixed adapter needs.

    Returns the adapter, the resolved 40-character revision, and the SHA-256 of
    the exact ``tokenizer.json`` bytes that were parsed.
    """

    if require_revision and commit_hash(revision) is None:
        raise ValueError(
            "BrowseComp-Plus evaluation requires "
            "--snippet-tokenizer-revision as a full 40-character commit"
        )
    if require_revision:
        for name, expected in (
            ("huggingface-hub", HUGGINGFACE_HUB_VERSION),
            ("tokenizers", TOKENIZERS_VERSION),
        ):
            try:
                observed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"BrowseComp-Plus evaluation requires {name}=={expected}"
                ) from exc
            if observed != expected:
                raise RuntimeError(
                    f"BrowseComp-Plus evaluation requires {name}=={expected}, "
                    f"found {observed}"
                )
    try:
        from huggingface_hub import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            hf_hub_download,
        )
        from tokenizers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            Tokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "token-bounded fixed retrieval requires mini-agent[web-fixed]"
        ) from exc
    tokenizer_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            revision=revision,
        )
    )
    snapshot = tokenizer_path.parent
    snapshot_revision = (
        commit_hash(snapshot.name) if snapshot.parent.name == "snapshots" else None
    )
    exact_requested = commit_hash(revision)
    if (
        exact_requested is not None
        and snapshot_revision is not None
        and snapshot_revision != exact_requested
    ):
        raise RuntimeError("downloaded tokenizer revision does not match the request")
    resolved = snapshot_revision or exact_requested
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    try:
        tokenizer_json = tokenizer_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("downloaded tokenizer.json is not UTF-8") from exc
    tokenizer = SnippetTokenizerAdapter(
        # Parse the same immutable bytes that were hashed. Reloading this path
        # would leave a hash/load race in a shared Hugging Face cache.
        Tokenizer.from_str(tokenizer_json),
        name=repo_id,
        revision=resolved,
        tokenizer_json_sha256=tokenizer_sha256,
    )
    return tokenizer, resolved, tokenizer_sha256


class PageReader(Protocol):
    async def open(self, url: str) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class JsonlSearchBackend:
    """Deterministic BM25 for tests and small fixed corpora."""

    def __init__(self, corpus: Path) -> None:
        expanded = _require_no_symlink(corpus.expanduser(), "corpus")
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
            _require_mapping(item, f"corpus line {number}")
            key = _document_key(item.get("docid", item.get("id")))
            text = item.get("text", item.get("contents"))
            if key is None or not isinstance(text, str):
                raise ValueError(f"corpus line {number} requires docid and text")
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
            {"ref": key, "title": key, "score": score, "snippet": self.documents[key]}
            for score, key in scores[:k]
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
        expanded_index = _require_no_symlink(
            index_path.expanduser(), "Lucene index root"
        )
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
        jar, jar_sha256 = validate_anserini_jar(anserini_jar)
        self.anserini_jar, self.anserini_jar_sha256 = jar, jar_sha256
        self.searcher = _lucene_searcher(self.index_path, self.anserini_jar)

    def search(self, query: str, k: int = 5) -> Sequence[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        for hit in self.searcher.search(query, k):
            docid = _document_key(getattr(hit, "docid", None))
            score = getattr(hit, "score", None)
            document = getattr(hit, "lucene_document", None)
            if (
                docid is None or document is None or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not callable(getattr(document, "get", None))
            ):
                raise InfrastructureError(
                    "BrowseComp-Plus search returned a malformed hit"
                )
            raw = _lucene_document(document.get("raw"))
            results.append(
                {"ref": docid, "docid": docid, "score": float(score),
                 "snippet": str(raw["contents"])}
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
    expanded = _require_no_symlink(path.expanduser(), "Anserini fat JAR")
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
        raise RuntimeError(_WEB_FIXED_HINT) from exc
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
        raise RuntimeError(_WEB_FIXED_HINT) from exc
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
    contents = document.get("contents") if isinstance(document, Mapping) else None
    if not isinstance(contents, str):
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
        _require_finite_number(
            timeout_seconds, "SerpAPI timeout_seconds", exclusive_minimum=0
        )
        _require_positive_int(max_page_bytes, "SerpAPI max_page_bytes")
        _require_positive_int(max_response_bytes, "SerpAPI max_response_bytes")
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
            timeout_seconds=timeout_seconds, max_page_bytes=max_page_bytes
        )
        self._urls: set[str] = set()

    async def search(self, query: str, k: int = 5) -> Sequence[Mapping[str, Any]]:
        _require_str(query, "SerpAPI query", error=ProtocolError)
        _require_positive_int(k, "SerpAPI result count", error=ProtocolError)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                params = {"engine": "google", "q": query, "api_key": self.api_key}
                async with client.stream(
                    "GET", "https://serpapi.com/search.json", params=params
                ) as response:
                    _success_status(response.status_code, "SerpAPI")
                    content = await read_bounded_body(response, self.max_response_bytes)
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
        _require_mapping(payload, "SerpAPI response", error=InfrastructureError)
        error = payload.get("error")
        if error is not None and _require_str(
            error, "SerpAPI API error", non_empty=False, error=InfrastructureError
        ):
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
            results.append({"ref": url, "url": url, "title": title, "snippet": snippet})
            if len(results) == k:
                break
        return results

    async def open(self, reference: str) -> Mapping[str, Any] | None:
        if reference not in self._urls:
            raise ProtocolError("open accepts only a URL from this browser session")
        return await self.page_reader.open(reference)

    def export_reference_state(self) -> tuple[str, ...]:
        return tuple(sorted(self._urls))

    def replace_reference_state(self, references: Sequence[str]) -> None:
        self._urls = set(_validated_reference_state(references))

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
        _require_finite_number(timeout_seconds, "timeout_seconds", exclusive_minimum=0)
        _require_positive_int(max_page_bytes, "max_page_bytes")
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
                        _success_status(response.status_code, "page")
                        body = await read_bounded_body(response, self.max_page_bytes)
                        encoding = response.encoding or "utf-8"
                        content_type = response.headers.get("content-type", "")
                        final_url = str(response.url)
                        break
                else:  # pragma: no cover - loop always exits or raises
                    raise AssertionError("unreachable redirect state")
        except (InfrastructureError, ProtocolError):
            raise
        except ResponseBodyTooLarge as exc:
            raise InfrastructureError("page exceeds the configured byte limit") from exc
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
        return {"reader": "httpx", "max_page_bytes": self.max_page_bytes}


class PlaywrightPageReader:
    """One isolated Chromium context, created lazily per research agent."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30,
        max_page_chars: int = 2 * 1024 * 1024,
    ) -> None:
        _require_finite_number(timeout_seconds, "timeout_seconds", exclusive_minimum=0)
        _require_positive_int(max_page_chars, "max_page_chars")
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
            _success_status(getattr(response, "status", None), "Playwright page")
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
        return {"ref": url, "url": page.url, "title": extracted["title"],
                "text": extracted["text"]}

    async def close(self) -> None:
        error: BaseException | None = None
        layers = (("_context", "close"), ("_browser", "close"), ("_playwright", "stop"))
        for attribute, method in layers:
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
        _require_str(self.backend_identity, "browser state backend identity")
        _require_tuple_of(self.references, str, "browser state references")
        for reference in self.references:
            _require_str(reference, "browser state references")


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
        _require_positive_int(top_k, "top_k")
        if max_observation_chars is not None:
            _require_int(max_observation_chars, "max_observation_chars", minimum=128)
        if (snippet_tokens is None) != (tokenizer is None):
            raise ValueError("snippet_tokens and tokenizer must be provided together")
        if snippet_tokens is not None:
            _require_positive_int(snippet_tokens, "snippet_tokens")
        _require_bool(allow_open, "allow_open")
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
                    "required": ["action"] if self.allow_open else ["action", "query"],
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
        if action not in ("search", "open"):
            raise InvalidAction(f"unsupported browser action {action!r}")
        if action == "open" and not self.allow_open:
            raise InvalidAction("browser open is not enabled")
        wanted, refused = ("query", "ref") if action == "search" else ("ref", "query")
        argument = _nonempty(call.arguments.get(wanted), f"browser {action} {wanted}")
        if refused in call.arguments:
            raise InvalidAction(f"browser {action} does not accept {refused}")
        if action == "search":
            self._counts["search"] += 1
            try:
                results = await _await(self.backend.search(argument, self.top_k))
            except ProtocolError as exc:
                raise InfrastructureError("search backend failed") from exc
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
                raise InfrastructureError(
                    "search backend must return a result sequence"
                )
            rendered: list[dict[str, Any]] = []
            references: set[str] = set()
            for result in results[: self.top_k]:
                _require_mapping(result, "search result", error=InfrastructureError)
                reference = _document_key(result.get("ref", result.get("url")))
                if reference is None:
                    raise InfrastructureError(
                        "search backend returned a result without ref"
                    )
                references.add(reference)
                item = dict(result)
                item["ref"] = reference
                if not self.allow_open and "docid" in item:
                    # The fixed BrowseComp-Plus search tool returns docid, score,
                    # and snippet. Its internal session reference is not another
                    # model-facing field when document opening is disabled.
                    item.pop("ref")
                if "snippet" in item:
                    snippet = _require_str(
                        item["snippet"], "search result snippet",
                        non_empty=False, error=InfrastructureError,
                    )
                    item["snippet"] = self._truncate(snippet)
                rendered.append(item)
            output = self._render(rendered)
            self._references.update(references)
            return ToolExecution(
                output=output,
                metadata={
                    "action": "search",
                    "result_count": len(rendered),
                    "query_sha256": hashlib.sha256(argument.encode()).hexdigest(),
                },
            )
        if argument not in self._references:
            raise InvalidAction("browser open ref was not returned by this session")
        self._counts["open"] += 1
        try:
            document = await _await(self.backend.open(argument))
        except ProtocolError as exc:
            raise InfrastructureError("open backend failed") from exc
        if document is None:
            return ToolExecution(
                output=json.dumps({"error": "result is unavailable"}), is_error=True
            )
        _require_mapping(document, "open backend result", error=InfrastructureError)
        return ToolExecution(
            output=self._render(dict(document)),
            metadata={"action": "open", "ref": argument},
        )

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
            sort_keys=True, separators=(",", ":"), allow_nan=False,
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
            raise ProtocolError("browser state came from a different search backend")
        transfer = self._backend_state_transfer()
        if transfer is None:
            self._references.update(state.references)
            return

        backend_before = _validated_reference_state(
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
            _require_callable(close, "search backend close", error=RuntimeError)
            await _await(close())
        self._closed = True

    def _truncate(self, value: str) -> str:
        if self.snippet_tokens is not None:
            assert self.tokenizer is not None
            tokens = list(self.tokenizer.encode(value, add_special_tokens=False))
            if len(tokens) > self.snippet_tokens:
                value = _require_str(
                    self.tokenizer.decode(
                        tokens[: self.snippet_tokens], skip_special_tokens=True
                    ),
                    "tokenizer decode result",
                    non_empty=False,
                    error=InfrastructureError,
                )
        limit = self.max_observation_chars
        return value if limit is None else value[:limit]

    def _render(self, value: Any) -> str:
        try:
            stable = _json_value(value, "browser backend result")
            rendered = json.dumps(
                stable, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise InfrastructureError("browser backend returned non-JSON data") from exc
        limit = self.max_observation_chars
        if limit is None or len(rendered) <= limit:
            return rendered
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

        def truncated(size: int) -> str:
            payload = {"preview": rendered[:size], "sha256": digest, "truncated": True}
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)

        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if len(truncated(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        return truncated(low)


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _nonempty(value: Any, name: str) -> str:
    text = _require_str(value, name, error=InvalidAction)
    if len(text.encode("utf-8")) > 16 * 1024:
        raise InvalidAction(f"{name} exceeds 16384 bytes")
    return text


def _document_key(value: Any) -> str | None:
    """Return a docid-like value as a non-empty string, or ``None`` if unusable."""

    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    return str(value) or None


def _success_status(value: Any, label: str) -> int:
    """Return a 2xx HTTP status, rejecting anything else as infrastructure loss."""

    if not isinstance(value, int) or isinstance(value, bool) or not 200 <= value < 300:
        raise InfrastructureError(f"{label} returned unsuccessful HTTP status {value}")
    return value


def _validated_reference_state(value: Any) -> tuple[str, ...]:
    """Return sorted unique references from an untrusted backend state sequence."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError("browser backend reference state must be a sequence")
    references = tuple(value)
    if not all(isinstance(reference, str) and reference for reference in references):
        raise ProtocolError(
            "browser backend reference state must contain non-empty strings"
        )
    return tuple(sorted(set(references)))


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
        not isinstance(value, str) or not value
        or len(value.encode("utf-8")) > 16 * 1024
        or any(ord(character) < 32 for character in value)
    ):
        raise ProtocolError("browser URL is invalid or too long")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
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
    port = explicit_port or (443 if parsed.scheme == "https" else 80)
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
    expanded = _require_no_symlink(path.expanduser(), "Lucene index root")
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
    identities: dict[Path, tuple[int, ...]] = {}
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            before = stat_key(opened)
            # Recorded operators hold --index-sha256 values over exactly these
            # bytes: name length, name, file size, then contents.
            digest.update(opened.st_size.to_bytes(8, "big"))
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            after = stat_key(os.fstat(stream.fileno()))
        if before != after:
            raise RuntimeError(f"Lucene index changed while hashing: {item}")
        identities[item] = after
    if sorted(path.rglob("*")) != entries:
        raise RuntimeError("Lucene index changed while hashing")
    for item, identity in identities.items():
        if identity != stat_key(item.stat()):
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
    "SnippetTokenizerAdapter",
    "commit_hash",
    "load_snippet_tokenizer",
    "directory_sha256",
    "validate_anserini_jar",
]
