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

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from elasticsearch import Elasticsearch, NotFoundError

from .analyzer import tokenize
from .es_dsl import (
    ANALYZER,
    SPELLING_FIELD,
    apply_suggestions,
    autocomplete_query,
    build_query,
    build_search_body,
    build_suggest_request,
    check_fields,
    index_fingerprint,
    index_mappings,
    index_settings,
)
from .index import (
    DEFAULT_FIELDS,
    SearchOptions,
    check_options,
    normalize_doc_id,
    normalize_document,
    resolve_field_weights,
    split_phrase,
)
from .scoring import DEFAULT_B, DEFAULT_K1

DEFAULT_INDEX: Final = "searchengine"
MAX_AUTOCOMPLETE: Final = 20
MAX_ANALYZE_LENGTH: Final = 10_000  # characters; Elasticsearch stops _analyze at 10,000 tokens
BULK_CHUNK_SIZE: Final = 500  # documents per bulk request
# Bytes of documents per bulk request. Elasticsearch refuses a request over
# http.max_content_length (100 MB), and a document may be 10 MB.
BULK_CHUNK_BYTES: Final = 10 * 2**20
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
        self.mappings = index_mappings(self.fields)
        self.fingerprint = index_fingerprint(self.settings, self.mappings)
        self.k1, self.b = k1, b
        self.client = client if client is not None else Elasticsearch(default_url())

    # ---------- managing the index ----------

    def create_index(self, *, recreate: bool = False) -> bool:
        """Create the index. Returns False if it already existed and was kept.

        An existing index is kept only if this engine made it, with the same
        settings and fields (its fingerprint says so), it still has one shard
        and the same k1 and b, and the last load into it finished.
        """
        if not isinstance(recreate, bool):
            raise TypeError("recreate must be True or False")
        exists = bool(self.client.indices.exists(index=self.index))
        if exists and not recreate:
            self._check_index()
            return False
        if exists:
            self.client.indices.delete(index=self.index)
        # Empty until a load finishes: if this run stops now, the next one
        # sees an unfinished index, and refuses it instead of searching it.
        mappings = self.mappings | {"_meta": self._meta(complete=False)}
        self.client.indices.create(index=self.index, settings=self.settings, mappings=mappings)
        return True

    def _meta(self, *, complete: bool) -> dict[str, Any]:
        """What the index records about itself, in its mapping's ``_meta``."""
        return {"fingerprint": self.fingerprint, "complete": complete}

    def _mark(self, *, complete: bool) -> None:
        self.client.indices.put_mapping(index=self.index, meta=self._meta(complete=complete))

    def _only(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        """The entry of a per-index answer. The name may be an alias of several indices."""
        entries = list(body.values())
        if len(entries) != 1:
            raise ValueError(f"{self.index!r} names {len(entries)} indices, not one")
        return dict(entries[0])

    def _check_index(self) -> None:
        """An index made in any other way would score differently, or miss documents."""
        response = self.client.indices.get_settings(index=self.index, flat_settings=True)
        settings = self._only(response.body)["settings"]
        # k1 and b can be changed on a closed index, so check what they are now.
        prefix = "index.similarity.default."
        found = (settings.get(prefix + "k1"), settings.get(prefix + "b"))
        if None in found or (float(found[0]), float(found[1])) != (self.k1, self.b):
            raise ValueError(
                f"index {self.index!r} has k1, b = {found}, not {(self.k1, self.b)}; recreate it"
            )
        shards = settings.get("index.number_of_shards")
        if shards != "1":  # a split or shrunk copy keeps the _meta, not the shard count
            raise ValueError(f"index {self.index!r} has {shards} shards, not 1; recreate it")
        mappings = self._only(self.client.indices.get_mapping(index=self.index).body)
        meta = mappings["mappings"].get("_meta", {})
        if meta.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"index {self.index!r} was not made by this engine with these settings "
                "and fields; recreate it"
            )
        if meta.get("complete") is not True:
            raise ValueError(
                f"index {self.index!r} was not filled completely (a load failed or was "
                "stopped); recreate it"
            )

    def delete_index(self) -> None:
        """Delete the index, if it exists."""
        self.client.options(ignore_status=404).indices.delete(index=self.index)

    def index_documents(self, docs: Iterable[Mapping[str, Any]]) -> int:
        """Add documents with the bulk API, then refresh so they are searchable.

        The index must exist (``create_index``): otherwise Elasticsearch would
        make one with its default analyzer. Each document is checked like
        ``InvertedIndex.add``. A bulk request holds an action line and a source
        line per document. A new request starts every BULK_CHUNK_SIZE
        documents, or sooner, once one holds BULK_CHUNK_BYTES of documents. We
        use the ``create`` action, not ``index``, so a duplicate id is an error
        instead of a silent overwrite.

        A load is not atomic. When one fails, Elasticsearch keeps the requests
        sent before, and every good document of the failing request, even
        those after the bad one; a document our own check refuses stops the
        load before its request is sent. So the index is marked unfinished
        while a load runs, and ``create_index`` will not reuse it until a load
        finishes.
        """
        if not self.client.indices.exists(index=self.index):
            raise ValueError(f"index {self.index!r} does not exist: create it first")
        self._mark(complete=False)
        added, size = 0, 0
        operations: list[dict[str, Any]] = []
        for doc in docs:
            source = self._source(doc)
            operations += [{"create": {"_index": self.index, "_id": source["id"]}}, source]
            size += len(json.dumps(source, ensure_ascii=False).encode("utf-8"))
            if len(operations) == 2 * BULK_CHUNK_SIZE or size >= BULK_CHUNK_BYTES:
                added += self._bulk(operations)
                operations, size = [], 0
        if operations:
            added += self._bulk(operations)
        self.client.indices.refresh(index=self.index)
        self._mark(complete=True)
        return added

    def check_documents(self, docs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Check documents as ``index_documents`` would, and that no id repeats.

        Nothing is sent. Returns the documents as they would be indexed, so a
        caller can check a whole file before it deletes an index.
        """
        checked: list[dict[str, Any]] = []
        ids: set[str] = set()
        for doc in docs:
            source = self._source(doc)
            if source["id"] in ids:
                raise ValueError(f"duplicate document id: {source['id']}")
            ids.add(source["id"])
            checked.append(source)
        return checked

    def _source(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """A document as it is sent, checked like ``InvertedIndex.add``."""
        source = normalize_document(doc)  # keeps "id": the index sorts ties by it
        if SPELLING_FIELD in source:
            raise ValueError(f"document {source['id']}: field {SPELLING_FIELD!r} is reserved")
        return source

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
        self, query: str, options: SearchOptions | None, field_weights: Mapping[str, float] | None
    ) -> tuple[dict[str, Any], list[str]]:
        options = SearchOptions(k1=self.k1, b=self.b) if options is None else options
        check_options(options)
        if (options.k1, options.b) != (self.k1, self.b):
            raise ValueError(
                "k1 and b are settings of the Elasticsearch index: "
                "use the index's values, or recreate it with new ones"
            )
        weights = resolve_field_weights(self.fields, field_weights)
        return build_query(query, options, weights), list(weights)

    def search_body(
        self,
        query: str,
        *,
        k: int = 10,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """The request ``search`` sends, after checking every argument."""
        es_query, fields = self._query(query, options, field_weights)
        return build_search_body(es_query, k, fields, category)

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
        body = self.search_body(
            query, k=k, options=options, field_weights=field_weights, category=category
        )
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
        """'Did you mean ...?' from Elasticsearch's term suggester.

        The query is checked like ``InvertedIndex.suggest`` checks it.
        """
        split_phrase(query)  # a str, within the length limit
        words = tokenize(query)
        request = build_suggest_request(words, self.fields)
        if not request:
            return None
        response = self.client.search(index=self.index, size=0, **request)
        return apply_suggestions(words, response.body)

    def autocomplete(self, prefix: str, *, k: int = 5) -> list[str]:
        """Titles that contain every word typed so far, the last one maybe in part."""
        query = autocomplete_query(prefix)  # checks the prefix
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_AUTOCOMPLETE:
            raise ValueError(f"k must be an int between 1 and {MAX_AUTOCOMPLETE}")
        if "title" not in self.fields or not prefix.strip():
            return []
        response = self.client.search(
            index=self.index, size=k, query=query, source_includes=["title"]
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
