from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from scaffoldlab.environments.base import ToolExecution

from ..types import ProtocolError, ToolCall, ToolDefinition
from .base import BaseEnvironment


BROWSECOMP_PLUS_REVISION = "046949032b0328319cc9a02663a759ec601d9402"
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


class SearchBackend(Protocol):
    def search(self, query: str, k: int = 5) -> list[Mapping[str, Any]]: ...

    def get_document(self, docid: str) -> Mapping[str, Any] | None: ...


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
        self.searcher = LuceneSearcher(str(self.index_path))

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
        return {
            "backend": "browsecomp_plus_lucene",
            "index_path": str(self.index_path),
            "source_revision": BROWSECOMP_PLUS_REVISION,
        }


class WebEnvironment(BaseEnvironment):
    def __init__(
        self,
        backend: SearchBackend,
        *,
        top_k: int = 5,
        snippet_chars: int = 4096,
        include_get_document: bool = False,
    ) -> None:
        if top_k < 1 or snippet_chars < 1:
            raise ValueError("top_k and snippet_chars must be positive")
        self.backend = backend
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.include_get_document = include_get_document

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
            query = action.arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ProtocolError("search query must be a non-empty string")
            candidates = self.backend.search(query, self.top_k)
            results: list[dict[str, Any]] = []
            for candidate in candidates:
                item: dict[str, Any] = {
                    "docid": str(candidate["docid"]),
                    "snippet": str(
                        candidate.get("text", candidate.get("snippet", ""))
                    )[: self.snippet_chars],
                }
                if candidate.get("score") is not None:
                    item["score"] = float(candidate["score"])
                results.append(item)
            return ToolExecution(
                output=json.dumps(results, ensure_ascii=False, sort_keys=True),
                metadata={"retrieved_docids": [item["docid"] for item in results]},
            )
        if action.name == "get_document" and self.include_get_document:
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
            "retrieval": (
                dict(backend_provenance()) if backend_provenance is not None else {}
            ),
        }
