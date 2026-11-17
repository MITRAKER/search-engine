"""An inverted index with BM25F and TF-IDF ranking, built from scratch.

An inverted index maps each term to the documents that contain it, and where:

    postings["invert"] = {"1": {"title": [0], "body": [1, 12]},
                          "15": {"body": [9]}}
              term        doc id   field     positions in that field

Searching means: look up each query term, collect its documents and score
them. We never scan every document, which is what keeps search fast.

Fields are weighted at query time (title counts 3x by default). Scoring
treats the weighted fields as one "virtual" field; this is BM25F, the same
model as Elasticsearch's ``combined_fields`` query.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from .analyzer import ENGLISH_STOPWORDS, MAX_TOKEN_LENGTH, Token, analyze_tokens, tokenize
from .porter import porter_stem
from .scoring import DEFAULT_B, DEFAULT_K1, bm25_idf, bm25_tf, tfidf_idf, tfidf_tf
from .spelling import SpellingIndex

Ranker = Literal["bm25", "tfidf"]
Mode = Literal["or", "and"]

DEFAULT_FIELDS: Final = ("title", "body")
DEFAULT_FIELD_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType({"title": 3.0, "body": 1.0})
MAX_RESULTS: Final = 1000
MAX_QUERY_TERMS: Final = 64
MAX_DOC_ID_LENGTH: Final = 512  # Elasticsearch's limit for _id, in bytes
MAX_CACHED_WEIGHTINGS: Final = 32  # bounds the statistics cache (JPL rule 3)
FORMAT_VERSION: Final = 2

# term -> doc id -> field -> positions
Postings = dict[str, dict[str, dict[str, list[int]]]]


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result."""

    doc_id: str
    score: float
    title: str


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """A query after analysis. ``phrase`` is True for "quoted" queries."""

    tokens: tuple[Token, ...]
    phrase: bool


@dataclass(frozen=True, slots=True)
class Clause:
    """One scoring unit: a single term, or a whole phrase.

    ``tfs`` maps each matching document to its field-weighted frequency, and
    ``dfs`` holds the document frequency of each term.
    """

    terms: tuple[str, ...]
    tfs: dict[str, float]
    dfs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClauseScore:
    """How one clause contributed to a document's score (see ``explain``)."""

    terms: tuple[str, ...]
    idf: float
    tf: float
    score: float


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Knobs for one search. The defaults match Elasticsearch's."""

    ranker: Ranker = "bm25"
    mode: Mode = "or"
    fuzzy: bool = False
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B


def _doc_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"document id must be a str or int, not {type(value).__name__}")
    doc_id = str(value)
    if not doc_id or len(doc_id.encode()) > MAX_DOC_ID_LENGTH:
        raise ValueError(f"document id must be 1 to {MAX_DOC_ID_LENGTH} bytes")
    return doc_id


def _check_options(options: SearchOptions) -> None:
    if options.ranker not in ("bm25", "tfidf"):
        raise ValueError(f"unknown ranker {options.ranker!r}")
    if options.mode not in ("or", "and"):
        raise ValueError(f"unknown mode {options.mode!r}")


def _check_index_data(data: object, path: str | Path) -> dict[str, Any]:
    """Check the top-level shape of a saved index before trusting it."""
    expected = {
        "fields": list,
        "docs": dict,
        "field_lengths": dict,
        "postings": dict,
        "spelling": dict,
    }
    if not isinstance(data, dict) or data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} index file")
    for key, kind in expected.items():
        if not isinstance(data.get(key), kind):
            raise ValueError(f"{path} is corrupt: {key!r} should be a {kind.__name__}")
    if set(data["docs"]) != set(data["field_lengths"]):
        raise ValueError(f"{path} is corrupt: documents and lengths disagree")
    return data


class InvertedIndex:
    """Index documents, then search them."""

    def __init__(self, fields: Iterable[str] = DEFAULT_FIELDS) -> None:
        self.fields: tuple[str, ...] = tuple(fields)
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be a non-empty list of distinct names")
        self.postings: Postings = {}
        self.field_lengths: dict[str, dict[str, int]] = {}  # doc -> field -> terms
        self.docs: dict[str, dict[str, str]] = {}
        self.spelling = SpellingIndex()
        self._stats_cache: dict[tuple[tuple[str, float], ...], tuple[int, float]] = {}

    def __len__(self) -> int:
        return len(self.docs)

    # ---------- indexing ----------

    def add(self, doc: Mapping[str, Any]) -> None:
        """Index one document, e.g. {"id": "1", "title": "...", "body": "..."}.

        Every value must be a string (the id may be an int). Fields that are
        not indexed, like "url", are stored and returned with results. The
        document is analyzed completely before anything changes, so a bad
        document never leaves the index half-updated.
        """
        doc_id = _doc_id(doc.get("id"))
        if doc_id in self.docs:
            raise ValueError(f"duplicate document id: {doc_id}")
        stored = self._stored_fields(doc, doc_id)
        analyzed = {field: analyze_tokens(stored.get(field, "")) for field in self.fields}

        for field, tokens in analyzed.items():
            self._index_tokens(doc_id, field, tokens)
        self.docs[doc_id] = stored
        self.field_lengths[doc_id] = {field: len(tokens) for field, tokens in analyzed.items()}
        self._stats_cache.clear()

    def add_many(self, docs: Iterable[Mapping[str, Any]]) -> int:
        """Index several documents. Returns how many were added."""
        count = 0
        for doc in docs:
            self.add(doc)
            count += 1
        return count

    @staticmethod
    def _stored_fields(doc: Mapping[str, Any], doc_id: str) -> dict[str, str]:
        stored = {"id": doc_id}
        for name, value in doc.items():
            if name == "id":
                continue
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError(f"document {doc_id}: field {name!r} must be a string")
            stored[name] = value
        return stored

    def _index_tokens(self, doc_id: str, field: str, tokens: list[Token]) -> None:
        for token in tokens:
            fields = self.postings.setdefault(token.term, {}).setdefault(doc_id, {})
            fields.setdefault(field, []).append(token.position)
            self.spelling.add(token.surface)

    # ---------- statistics ----------

    def _weights(self, field_weights: Mapping[str, float] | None) -> dict[str, float]:
        """Validate field weights. Only fields that get a weight are searched."""
        if field_weights is None:
            return {field: DEFAULT_FIELD_WEIGHTS.get(field, 1.0) for field in self.fields}
        unknown = set(field_weights) - set(self.fields)
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}; fields are {list(self.fields)}")
        weights = {
            field: float(field_weights[field]) for field in self.fields if field in field_weights
        }
        if not weights or not all(math.isfinite(w) and w > 0 for w in weights.values()):
            raise ValueError("field weights must be positive numbers, for at least one field")
        return weights

    def _collection_stats(self, weights: dict[str, float]) -> tuple[int, float]:
        """N and the weighted average length, over documents that have at
        least one term in the searched fields (Lucene counts the same way)."""
        key = tuple(sorted(weights.items()))
        if key not in self._stats_cache:
            if len(self._stats_cache) >= MAX_CACHED_WEIGHTINGS:
                self._stats_cache.clear()
            n, total = 0, 0.0
            for lengths in self.field_lengths.values():
                if any(lengths[field] for field in weights):
                    n += 1
                    total += self._doc_length(lengths, weights)
            self._stats_cache[key] = (n, total / n if n else 0.0)
        return self._stats_cache[key]

    @staticmethod
    def _doc_length(lengths: Mapping[str, int], weights: Mapping[str, float]) -> float:
        return sum(lengths[field] * weight for field, weight in weights.items())

    def _term_tfs(self, term: str, weights: Mapping[str, float]) -> dict[str, float]:
        """doc id -> field-weighted term frequency, in the searched fields."""
        tfs: dict[str, float] = {}
        for doc_id, fields in self.postings.get(term, {}).items():
            tf = sum(len(fields.get(field, ())) * weight for field, weight in weights.items())
            if tf > 0:
                tfs[doc_id] = tf
        return tfs

    # ---------- query parsing ----------

    def parse(self, query: str, *, fuzzy: bool = False) -> ParsedQuery:
        """Analyze a query. Wrap it in double quotes for an exact phrase."""
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        text = query.strip()
        phrase = len(text) >= 2 and text[0] == '"' and text[-1] == '"'
        tokens = analyze_tokens(text[1:-1] if phrase else text)
        if len(tokens) > MAX_QUERY_TERMS:
            raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
        if fuzzy:
            tokens = [self._correct_token(token) for token in tokens]
        return ParsedQuery(tuple(tokens), phrase)

    def _correct_token(self, token: Token) -> Token:
        if token.term in self.postings:
            return token
        fixed = self.spelling.correct(token.surface)
        return token if fixed == token.surface else Token(porter_stem(fixed), fixed, token.position)

    def suggest(self, query: str) -> str | None:
        """'Did you mean ...?' The corrected query, or None if nothing changed."""
        words = tokenize(query)
        fixed = [self._suggest_word(word) for word in words]
        return " ".join(fixed) if fixed != words else None

    def _suggest_word(self, word: str) -> str:
        if word in ENGLISH_STOPWORDS or len(word) > MAX_TOKEN_LENGTH:
            return word
        if porter_stem(word) in self.postings:
            return word
        return self.spelling.correct(word)

    # ---------- matching ----------

    def _clauses(self, parsed: ParsedQuery, weights: Mapping[str, float]) -> list[Clause]:
        """Turn a query into scoring clauses: one per term, or one phrase."""
        if not parsed.phrase or len(parsed.tokens) < 2:
            clauses = []
            for token in parsed.tokens:
                term_tfs = self._term_tfs(token.term, weights)
                clauses.append(Clause((token.term,), term_tfs, (len(term_tfs),)))
            return clauses
        first = parsed.tokens[0].position
        offsets = [(t.term, t.position - first) for t in parsed.tokens]
        per_term = [self._term_tfs(term, weights) for term, _ in offsets]
        candidates = set.intersection(*(set(tfs) for tfs in per_term))
        tfs: dict[str, float] = {}
        for doc_id in sorted(candidates):
            tf = sum(self._phrase_count(doc_id, field, offsets) * w for field, w in weights.items())
            if tf > 0:
                tfs[doc_id] = tf
        terms = tuple(term for term, _ in offsets)
        return [Clause(terms, tfs, tuple(len(t) for t in per_term))]

    def _phrase_count(self, doc_id: str, field: str, offsets: list[tuple[str, int]]) -> int:
        """How many times the phrase occurs in one field of one document.

        ``offsets`` pairs each term with its distance from the first term.
        The phrase occurs at position p when every term sits at p + distance.
        """
        positions = [set(self.postings[term][doc_id].get(field, ())) for term, _ in offsets]
        return sum(
            all(start + offset in positions[i] for i, (_, offset) in enumerate(offsets))
            for start in positions[0]
        )

    # ---------- scoring ----------

    @staticmethod
    def _clause_idf(clause: Clause, n: int, ranker: Ranker) -> float:
        """A phrase's idf is the sum of its terms' idfs, as in Lucene."""
        if ranker == "bm25":
            return sum(bm25_idf(n, df) for df in clause.dfs)
        return sum(tfidf_idf(n, df) for df in clause.dfs)

    def _score_docs(
        self, parsed: ParsedQuery, weights: dict[str, float], options: SearchOptions
    ) -> dict[str, float]:
        n, avgdl = self._collection_stats(weights)
        clauses = self._clauses(parsed, weights)
        if n == 0 or not clauses:
            return {}
        allowed: set[str] | None = None
        if options.mode == "and":
            allowed = set.intersection(*(set(clause.tfs) for clause in clauses))
        scores: dict[str, float] = {}
        for clause in clauses:
            if not clause.tfs:
                continue
            idf = self._clause_idf(clause, n, options.ranker)
            for doc_id, tf in clause.tfs.items():
                if allowed is None or doc_id in allowed:
                    part = self._tf_part(doc_id, tf, weights, avgdl, options)
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf * part
        return scores

    def _tf_part(
        self,
        doc_id: str,
        tf: float,
        weights: Mapping[str, float],
        avgdl: float,
        options: SearchOptions,
    ) -> float:
        if options.ranker == "tfidf":
            return tfidf_tf(tf)
        dl = self._doc_length(self.field_lengths[doc_id], weights)
        return bm25_tf(tf, dl, avgdl, options.k1, options.b)

    # ---------- public search API ----------

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> list[Hit]:
        """Return the k best documents for ``query``, best first.

        Ties are broken by document id, so results are always reproducible.
        """
        if not 1 <= k <= MAX_RESULTS:
            raise ValueError(f"k must be between 1 and {MAX_RESULTS}")
        options = options or SearchOptions()
        _check_options(options)
        weights = self._weights(field_weights)
        scores = self._score_docs(self.parse(query, fuzzy=options.fuzzy), weights, options)
        best = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [Hit(doc_id, score, self.docs[doc_id].get("title", "")) for doc_id, score in best]

    def explain(
        self,
        query: str,
        doc_id: str,
        *,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> list[ClauseScore]:
        """Break one document's score down by query term. Great for learning."""
        options = options or SearchOptions()
        _check_options(options)
        weights = self._weights(field_weights)
        n, avgdl = self._collection_stats(weights)
        breakdown: list[ClauseScore] = []
        for clause in self._clauses(self.parse(query, fuzzy=options.fuzzy), weights):
            tf = clause.tfs.get(doc_id, 0.0)
            if tf > 0:
                idf = self._clause_idf(clause, n, options.ranker)
                part = self._tf_part(doc_id, tf, weights, avgdl, options)
                breakdown.append(ClauseScore(clause.terms, idf, tf, idf * part))
        return breakdown

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        """Write the whole index to a JSON file."""
        data = {
            "format": FORMAT_VERSION,
            "fields": list(self.fields),
            "docs": self.docs,
            "field_lengths": self.field_lengths,
            "postings": self.postings,
            "spelling": self.spelling.counts,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        """Read an index written by ``save``, checking its shape first."""
        data = _check_index_data(json.loads(Path(path).read_text(encoding="utf-8")), path)
        index = cls(data["fields"])
        index.docs = data["docs"]
        index.field_lengths = data["field_lengths"]
        index.postings = data["postings"]
        index.spelling = SpellingIndex.from_counts(data["spelling"])
        return index

    def stats(self) -> dict[str, Any]:
        """A few numbers that describe the index."""
        top = heapq.nlargest(5, self.postings, key=lambda term: (len(self.postings[term]), term))
        docs = max(len(self.docs), 1)
        return {
            "documents": len(self.docs),
            "unique_terms": len(self.postings),
            "avg_length": {
                field: round(
                    sum(lengths[field] for lengths in self.field_lengths.values()) / docs, 1
                )
                for field in self.fields
            },
            "most_common_terms": [(term, len(self.postings[term])) for term in top],
        }
