from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..types import ProtocolError, ToolCall, ToolDefinition, ToolExecution
from .base import BaseEnvironment


BROWSECOMP_PLUS_REVISION = "046949032b0328319cc9a02663a759ec601d9402"


def directory_identity(path: Path) -> Mapping[str, Any]:
    """Hash a local data snapshot by relative paths and file contents."""

    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"snapshot directory does not exist: {root}")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for entry in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not entry.is_file():
            continue
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = 0
        with entry.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files += 1
        total_bytes += size
    if files == 0:
        raise ValueError(f"snapshot directory contains no files: {root}")
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
    }


_TOKEN = re.compile(r"[\w]+", re.UNICODE)


class SearchBackend(Protocol):
    def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]: ...

    def get_document(self, docid: str) -> Mapping[str, Any] | None: ...


class SnippetTokenizer(Protocol):
    """Small boundary implemented by Hugging Face and deterministic test doubles."""

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[Any]: ...

    def decode(
        self, tokens: Sequence[Any], *, skip_special_tokens: bool
    ) -> str: ...


@dataclass(frozen=True)
class WebAccounting:
    """Evaluator-facing accounting for one environment instance/query."""

    tool_call_counts: Mapping[str, int]
    retrieved_docids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "tool_call_counts": dict(self.tool_call_counts),
            "retrieved_docids": list(self.retrieved_docids),
        }


class JsonlSearchBackend:
    """Deterministic pure-Python BM25 for fixtures and small offline corpora."""

    def __init__(self, corpus: Path) -> None:
        self.corpus = corpus.expanduser().resolve()
        self.documents: dict[str, str] = {}
        for line_number, line in enumerate(
            self.corpus.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, Mapping):
                raise ValueError(f"corpus line {line_number} is not an object")
            docid = item.get("docid", item.get("id"))
            text = item.get("text", item.get("contents"))
            if not isinstance(docid, (str, int)) or not isinstance(text, str):
                raise ValueError(
                    f"corpus line {line_number} requires docid/id and text/contents"
                )
            key = str(docid)
            if key in self.documents:
                raise ValueError(f"duplicate corpus docid {key!r}")
            self.documents[key] = text
        if not self.documents:
            raise ValueError("corpus must contain at least one document")
        self.corpus_sha256 = hashlib.sha256(self.corpus.read_bytes()).hexdigest()
        self._terms = {
            docid: Counter(self._tokenize(text))
            for docid, text in self.documents.items()
        }
        self._lengths = {docid: sum(terms.values()) for docid, terms in self._terms.items()}
        self._average_length = sum(self._lengths.values()) / len(self._lengths)
        self._document_frequency: Counter[str] = Counter()
        for terms in self._terms.values():
            self._document_frequency.update(terms.keys())

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return [token.casefold() for token in _TOKEN.findall(value)]

    def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
        terms = self._tokenize(query)
        if not terms:
            return []
        count = len(self.documents)
        scores: list[tuple[float, str]] = []
        for docid, frequencies in self._terms.items():
            score = 0.0
            length = self._lengths[docid]
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[term]
                inverse = math.log(
                    1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * length / max(self._average_length, 1)
                )
                score += inverse * frequency * 2.2 / denominator
            if score > 0:
                scores.append((score, docid))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"docid": docid, "score": score, "text": self.documents[docid]}
            for score, docid in scores[:k]
        ]

    def get_document(self, docid: str) -> Mapping[str, Any] | None:
        text = self.documents.get(str(docid))
        return None if text is None else {"docid": str(docid), "text": text}

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": "jsonl_bm25_test",
            "corpus": str(self.corpus),
            "corpus_sha256": self.corpus_sha256,
            "documents": len(self.documents),
        }


class BrowseCompPlusBackend:
    """Thin adapter around BrowseComp-Plus's canonical local Lucene index."""

    def __init__(self, index_path: Path) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "BrowseComp-Plus Lucene retrieval requires the optional pyserini package"
            ) from exc
        self.index_path = index_path.expanduser().resolve()
        self.index_identity = directory_identity(self.index_path)
        self.searcher = LuceneSearcher(str(self.index_path))
        set_bm25 = getattr(self.searcher, "set_bm25", None)
        if set_bm25 is not None:
            set_bm25()

    def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        for hit in self.searcher.search(query, k):
            raw = json.loads(hit.lucene_document.get("raw"))
            results.append(
                {"docid": hit.docid, "score": hit.score, "text": raw["contents"]}
            )
        return results

    def get_document(self, docid: str) -> Mapping[str, Any] | None:
        document = self.searcher.doc(str(docid))
        if document is None:
            return None
        return {"docid": str(docid), "text": json.loads(document.raw())["contents"]}

    def provenance(self) -> Mapping[str, Any]:
        try:
            java = subprocess.run(
                ("java", "-version"),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            java_version: list[str] = []
        else:
            java_version = (java.stderr or java.stdout).splitlines()
        try:
            pyserini_version = importlib.metadata.version("pyserini")
        except importlib.metadata.PackageNotFoundError:
            pyserini_version = "unknown"
        return {
            "backend": "browsecomp_plus_lucene",
            "index": dict(self.index_identity),
            "source_revision": BROWSECOMP_PLUS_REVISION,
            "retrieval": "bm25",
            "pyserini_version": pyserini_version,
            "java_version": java_version[0] if java_version else "unknown",
        }


class WebEnvironment(BaseEnvironment):
    def __init__(
        self,
        backend: SearchBackend,
        *,
        top_k: int = 5,
        snippet_chars: int | None = 4096,
        snippet_tokens: int | None = None,
        tokenizer: SnippetTokenizer | None = None,
        tokenizer_identity: Mapping[str, Any] | None = None,
        include_get_document: bool = False,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if snippet_chars is not None and (
            not isinstance(snippet_chars, int)
            or isinstance(snippet_chars, bool)
            or snippet_chars < 1
        ):
            raise ValueError("snippet_chars must be a positive integer or null")
        if snippet_tokens is not None and (
            not isinstance(snippet_tokens, int)
            or isinstance(snippet_tokens, bool)
            or snippet_tokens < 1
        ):
            raise ValueError("snippet_tokens must be a positive integer or null")
        if snippet_chars is not None and snippet_tokens is not None:
            raise ValueError("select either character or token snippet truncation")
        if snippet_tokens is not None and tokenizer is None:
            raise ValueError("token snippet truncation requires an injected tokenizer")
        if snippet_tokens is None and tokenizer is not None:
            raise ValueError("an injected tokenizer requires snippet_tokens")
        self.backend = backend
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.snippet_tokens = snippet_tokens
        self.tokenizer = tokenizer
        self.tokenizer_identity = dict(tokenizer_identity or {})
        self.include_get_document = include_get_document
        self._tool_call_counts: Counter[str] = Counter()
        self._retrieved_docids: set[str] = set()

    @classmethod
    def from_policy(
        cls,
        backend: SearchBackend,
        *,
        benchmark: Mapping[str, Any],
        observation: Mapping[str, Any],
        tools: Sequence[str],
        tokenizer: SnippetTokenizer | None = None,
        tokenizer_identity: Mapping[str, Any] | None = None,
    ) -> "WebEnvironment":
        """Build from the only benchmark/profile fields implemented here.

        Unknown policy keys fail closed so a profile cannot silently advertise an
        observation setting that the environment ignores.
        """

        unknown_benchmark = set(benchmark) - {"name", "retrieval", "top_k"}
        unknown_observation = set(observation) - {"snippet_chars", "snippet_tokens"}
        if unknown_benchmark:
            raise ValueError(
                f"unsupported web benchmark policy fields: {sorted(unknown_benchmark)}"
            )
        if unknown_observation:
            raise ValueError(
                "unsupported web observation policy fields: "
                f"{sorted(unknown_observation)}"
            )
        if benchmark.get("name", "browsecomp_plus") != "browsecomp_plus":
            raise ValueError("web benchmark name must be browsecomp_plus")
        if benchmark.get("retrieval", "bm25") != "bm25":
            raise ValueError("web benchmark retrieval must be bm25")
        requested_tools = tuple(tools)
        if requested_tools not in {("search",), ("search", "get_document")}:
            raise ValueError("web tools must be search with optional get_document")
        snippet_tokens = observation.get("snippet_tokens")
        snippet_chars = observation.get(
            "snippet_chars", None if snippet_tokens is not None else 4096
        )
        return cls(
            backend,
            top_k=benchmark.get("top_k", 5),
            snippet_chars=snippet_chars,
            snippet_tokens=snippet_tokens,
            tokenizer=tokenizer,
            tokenizer_identity=tokenizer_identity,
            include_get_document="get_document" in requested_tools,
        )

    def accounting(self) -> WebAccounting:
        return WebAccounting(
            tool_call_counts=dict(sorted(self._tool_call_counts.items())),
            retrieved_docids=tuple(sorted(self._retrieved_docids)),
        )

    def reset_accounting(self) -> None:
        self._tool_call_counts.clear()
        self._retrieved_docids.clear()

    def _truncate(self, text: str) -> str:
        if self.snippet_tokens is not None:
            assert self.tokenizer is not None
            tokens = list(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
            if len(tokens) > self.snippet_tokens:
                return self.tokenizer.decode(
                    tokens[: self.snippet_tokens], skip_special_tokens=True
                )
            return text
        if self.snippet_chars is not None:
            return text[: self.snippet_chars]
        return text

    def tools(self) -> Sequence[ToolDefinition]:
        tools = [
            ToolDefinition(
                name="search",
                description=(
                    f"Search the fixed offline corpus. Returns the top {self.top_k} "
                    "documents with docid, score, and snippet."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]
        if self.include_get_document:
            tools.append(
                ToolDefinition(
                    name="get_document",
                    description="Retrieve a full document by docid.",
                    input_schema={
                        "type": "object",
                        "properties": {"docid": {"type": "string"}},
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                )
            )
        return tuple(tools)

    async def execute(self, action: ToolCall) -> ToolExecution:
        if action.name == "search":
            self._tool_call_counts["search"] += 1
            query = action.arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ProtocolError("search query must be a non-empty string")
            candidates = self.backend.search(query, self.top_k)
            results: list[dict[str, Any]] = []
            for candidate in candidates:
                item: dict[str, Any] = {
                    "docid": str(candidate["docid"]),
                    "snippet": self._truncate(
                        str(candidate.get("text", candidate.get("snippet", "")))
                    ),
                }
                if candidate.get("score") is not None:
                    item["score"] = float(candidate["score"])
                results.append(item)
            self._retrieved_docids.update(item["docid"] for item in results)
            return ToolExecution(
                output=json.dumps(results, ensure_ascii=False, sort_keys=True),
                metadata={
                    "retrieved_docids": [item["docid"] for item in results],
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                },
            )
        if action.name == "get_document" and self.include_get_document:
            self._tool_call_counts["get_document"] += 1
            docid = action.arguments.get("docid")
            if not isinstance(docid, str) or not docid:
                raise ProtocolError("get_document docid must be a non-empty string")
            document = self.backend.get_document(docid)
            if document is None:
                return ToolExecution(
                    output=json.dumps({"error": f"document {docid!r} not found"}),
                    is_error=True,
                )
            return ToolExecution(
                output=json.dumps(dict(document), ensure_ascii=False, sort_keys=True)
            )
        raise ProtocolError(f"unsupported web tool {action.name!r}")

    def provenance(self) -> dict[str, object]:
        backend_provenance = getattr(self.backend, "provenance", None)
        return {
            "application": "web",
            "benchmark": "browsecomp_plus",
            "source_revision": BROWSECOMP_PLUS_REVISION,
            "tools": [tool.name for tool in self.tools()],
            "top_k": self.top_k,
            "snippet_policy": (
                {"unit": "tokens", "limit": self.snippet_tokens}
                if self.snippet_tokens is not None
                else {"unit": "characters", "limit": self.snippet_chars}
            ),
            "retrieval": (
                dict(backend_provenance()) if backend_provenance is not None else {}
            ),
            "tokenizer": dict(self.tokenizer_identity),
        }
