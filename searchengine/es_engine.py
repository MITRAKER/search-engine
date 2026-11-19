"""The same search engine, on Elasticsearch.

Everything the from-scratch engine does by hand, Elasticsearch does inside
Lucene. ``ElasticsearchEngine`` creates an index that behaves like
``InvertedIndex`` (see ``es_dsl.py`` for how), fills it and searches it, with
the same arguments, so the two engines can be compared fairly.

On top of that it uses what a real product needs: a category filter with
facet counts, highlighted snippets, autocomplete and "did you mean".

It needs the optional ``elasticsearch`` package: ``pip install -e ".[es]"``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from elasticsearch import Elasticsearch, NotFoundError

from .analyzer import is_searchable, tokenize
from .es_dsl import (
    ANALYZER,
    SPELLING_FIELD,
    QueryType,
    apply_suggestions,
    autocomplete_query,
    build_query,
    build_rank_body,
    build_search_body,
    build_suggesters,
    check_fields,
    index_mappings,
    index_settings,
)
from .index import (
    DEFAULT_FIELDS,
    MAX_QUERY_TERMS,
    SearchOptions,
    check_options,
    normalize_doc_id,
    normalize_document,
    resolve_field_weights,
)
from .scoring import DEFAULT_B, DEFAULT_K1

DEFAULT_INDEX: Final = "searchengine"
MAX_AUTOCOMPLETE: Final = 20
MAX_PREFIX_LENGTH: Final = 256  # characters, for autocomplete
MAX_ANALYZE_LENGTH: Final = 10_000  # characters; Elasticsearch stops _analyze at 10,000 tokens
BULK_CHUNK_SIZE: Final = 500  # documents per bulk request
_INDEX_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}")


def default_url() -> str:
    """Where Elasticsearch runs: the ES_URL environment variable, or localhost."""
    return os.environ.get("ES_URL", "http://localhost:9200")


class IndexingError(RuntimeError):
    """Elasticsearch rejected documents in a bulk request."""


@dataclass(frozen=True, slots=True)
class EsHit:
    """One result. Snippets are HTML-escaped, with the matches in <mark> tags."""

    doc_id: str
    score: float
    title: str
    highlights: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class EsResult:
    """One page of results, how many documents matched, and the facet counts."""

    hits: list[EsHit]
    total: int
    facets: dict[str, int]
    took_ms: int


@dataclass(frozen=True, slots=True)
class EsRanking:
    """Only the ranking: document ids and scores, best first (for benchmarks)."""

    doc_ids: list[str]
    scores: list[float]
    took_ms: int  # Elasticsearch's own time, without the trip over the network


class ElasticsearchEngine:
    """Create, fill and search one Elasticsearch index."""

    def __init__(
        self,
        client: Elasticsearch | None = None,
        *,
        index: str = DEFAULT_INDEX,
        fields: Iterable[str] = DEFAULT_FIELDS,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if not isinstance(index, str) or not _INDEX_NAME.fullmatch(index):
            raise ValueError(f"invalid index name {index!r}: use a-z, 0-9, '-' and '_'")
        self.index = index
        self.fields = check_fields(fields)
        self.settings = index_settings(k1, b)
        self.k1, self.b = k1, b
        self.client = client if client is not None else Elasticsearch(default_url())

    # ---------- managing the index ----------

    def create_index(self, *, recreate: bool = False) -> bool:
        """Create the index. Returns False if it already existed and was kept."""
        exists = bool(self.client.indices.exists(index=self.index))
        if exists and not recreate:
            self._check_similarity()
            return False
        if exists:
            self.client.indices.delete(index=self.index)
        self.client.indices.create(
            index=self.index, settings=self.settings, mappings=index_mappings(self.fields)
        )
        return True

    def _check_similarity(self) -> None:
        """An index made with other BM25 settings would score differently."""
        response = self.client.indices.get_settings(index=self.index, flat_settings=True)
        settings = response.body[self.index]["settings"]
        prefix = "index.similarity.default."
        found = (settings.get(prefix + "k1"), settings.get(prefix + "b"))
        if None in found or (float(found[0]), float(found[1])) != (self.k1, self.b):
            raise ValueError(
                f"index {self.index!r} has k1, b = {found}, not {(self.k1, self.b)}; recreate it"
            )

    def delete_index(self) -> None:
        """Delete the index, if it exists."""
        self.client.options(ignore_status=404).indices.delete(index=self.index)

    def index_documents(self, docs: Iterable[Mapping[str, Any]]) -> int:
        """Add documents with the bulk API, then refresh so they are searchable.

        Each document is checked like ``InvertedIndex.add``. A bulk request
        holds an action line and a source line per document. We use the
        ``create`` action, not ``index``, so a duplicate id is an error instead
        of a silent overwrite. Unlike our engine, Elasticsearch has no
        transactions: documents sent before a failure stay indexed.
        """
        added = 0
        operations: list[dict[str, Any]] = []
        for doc in docs:
            source = normalize_document(doc)
            doc_id = source.pop("id")
            if SPELLING_FIELD in source:
                raise ValueError(f"document {doc_id}: field {SPELLING_FIELD!r} is reserved")
            operations += [{"create": {"_index": self.index, "_id": doc_id}}, source]
            if len(operations) == 2 * BULK_CHUNK_SIZE:
                added += self._bulk(operations)
                operations = []
        if operations:
            added += self._bulk(operations)
        self.client.indices.refresh(index=self.index)
        return added

    def _bulk(self, operations: list[dict[str, Any]]) -> int:
        response = self.client.bulk(operations=operations).body
        items = [item["create"] for item in response["items"]]
        failed = [item for item in items if "error" in item]
        if failed:
            first = failed[0]
            raise IndexingError(
                f"{len(failed)} of {len(items)} documents failed; "
                f"first: id {first['_id']}: {first['error']['reason']}"
            )
        return len(items)

    def count(self) -> int:
        """How many documents the index holds."""
        return int(self.client.count(index=self.index).body["count"])

    # ---------- searching ----------

    def _query(
        self,
        query: str,
        options: SearchOptions | None,
        field_weights: Mapping[str, float] | None,
        query_type: QueryType = "combined_fields",
    ) -> tuple[dict[str, Any], list[str]]:
        options = SearchOptions(k1=self.k1, b=self.b) if options is None else options
        check_options(options)
        if (options.k1, options.b) != (self.k1, self.b):
            raise ValueError("k1 and b are fixed when the Elasticsearch index is created")
        weights = resolve_field_weights(self.fields, field_weights)
        return build_query(query, options, weights, query_type), list(weights)

    def rank(
        self,
        query: str,
        *,
        k: int = 10,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
        query_type: QueryType = "combined_fields",
    ) -> EsRanking:
        """Rank only: the ids and scores of the k best documents, for benchmarks.

        ``query_type="best_fields"`` scores each field on its own instead of
        as one (BM25F), like BEIR's Elasticsearch baseline.
        """
        es_query, _ = self._query(query, options, field_weights, query_type)
        response = self.client.search(index=self.index, **build_rank_body(es_query, k)).body
        hits = response["hits"]["hits"]
        return EsRanking(
            doc_ids=[str(hit["_id"]) for hit in hits],
            scores=[float(hit["_score"]) for hit in hits],
            took_ms=int(response["took"]),
        )

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
        category: str | None = None,
    ) -> EsResult:
        """Search, optionally in one category. Same arguments as ``InvertedIndex.search``."""
        es_query, fields = self._query(query, options, field_weights)
        body = build_search_body(es_query, k, fields, category)
        response = self.client.search(index=self.index, **body).body
        hits = [
            EsHit(
                doc_id=str(hit["_id"]),
                score=float(hit["_score"]),
                title=str(hit["_source"].get("title", "")),
                highlights=dict(hit.get("highlight", {})),
            )
            for hit in response["hits"]["hits"]
        ]
        buckets = response["aggregations"]["facets"]["buckets"]
        facets = {str(bucket["key"]): int(bucket["doc_count"]) for bucket in buckets}
        total = int(response["hits"]["total"]["value"])
        return EsResult(hits, total, facets, int(response["took"]))

    def explain(
        self,
        query: str,
        doc_id: str | int,
        *,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Elasticsearch's own breakdown of one document's score.

        ``matched`` is False when the document does not match (or does not exist).
        """
        es_query, _ = self._query(query, options, field_weights)
        doc_id = normalize_doc_id(doc_id)
        try:
            response = self.client.explain(index=self.index, id=doc_id, query=es_query)
        except NotFoundError as error:
            if isinstance(error.body, dict) and error.body.get("matched") is False:
                return dict(error.body)  # no such document (a missing index still raises)
            raise
        return dict(response.body)

    def suggest(self, query: str) -> str | None:
        """'Did you mean ...?' from Elasticsearch's term suggester."""
        words = tokenize(query)
        if sum(1 for word in words if is_searchable(word)) > MAX_QUERY_TERMS:
            raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
        suggesters = build_suggesters(words)
        if not suggesters:
            return None
        response = self.client.search(index=self.index, size=0, suggest=suggesters)
        return apply_suggestions(words, response.body["suggest"])

    def autocomplete(self, prefix: str, *, k: int = 5) -> list[str]:
        """Titles that complete what the user has typed so far."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")
        if len(prefix) > MAX_PREFIX_LENGTH:
            raise ValueError(f"prefix is longer than {MAX_PREFIX_LENGTH} characters")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_AUTOCOMPLETE:
            raise ValueError(f"k must be an int between 1 and {MAX_AUTOCOMPLETE}")
        if "title" not in self.fields or not prefix.strip():
            return []
        response = self.client.search(
            index=self.index, size=k, query=autocomplete_query(prefix), source_includes=["title"]
        )
        titles = [str(hit["_source"].get("title", "")) for hit in response.body["hits"]["hits"]]
        return list(dict.fromkeys(title for title in titles if title))

    def analyze(self, text: str) -> list[tuple[str, int]]:
        """Run Elasticsearch's copy of our analyzer: (term, position) pairs."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        if len(text) > MAX_ANALYZE_LENGTH:
            raise ValueError(f"text is longer than {MAX_ANALYZE_LENGTH} characters")
        response = self.client.indices.analyze(index=self.index, analyzer=ANALYZER, text=text)
        return [(str(t["token"]), int(t["position"])) for t in response.body["tokens"]]
