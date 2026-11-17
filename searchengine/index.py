"""An inverted index with BM25F and TF-IDF ranking, built from scratch.

An inverted index maps each term to the documents that contain it, and where:

    postings["invert"] = {"1": {"title": [0], "body": [1, 12]},
                          "15": {"body": [9]}}
              term        doc id   field     positions in that field

Searching means: look up each query term, collect its documents and score
them. We never scan every document, which is what keeps search fast.

Fields are weighted at query time (title counts 3x by default). Scoring
treats the weighted fields as one "virtual" field. This is BM25F, computed
with the same statistics as Lucene's ``CombinedFieldQuery``, which is what
Elasticsearch's ``combined_fields`` query runs, so both give the same scores.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from .analyzer import Token, analyze_tokens, is_searchable, tokenize
from .porter import porter_stem
from .scoring import (
    DEFAULT_B,
    DEFAULT_K1,
    bm25_idf,
    bm25_tf,
    check_bm25_params,
    lucene_weighted_length,
    lucene_weighted_total,
    tfidf_idf,
    tfidf_tf,
)
from .spelling import SpellingIndex

Ranker = Literal["bm25", "tfidf"]
Mode = Literal["or", "and"]

DEFAULT_FIELDS: Final = ("title", "body")
DEFAULT_FIELD_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType({"title": 3.0, "body": 1.0})
MAX_FIELDS: Final = 32
# Field weights must be at least 1, as in Elasticsearch's combined_fields: a
# weight multiplies how often a word counts, and a word counts at least once.
# The upper bound keeps every weighted length well inside a 32-bit float.
MIN_FIELD_WEIGHT: Final = 1.0
MAX_FIELD_WEIGHT: Final = 1000.0
MAX_RESULTS: Final = 1000
MAX_QUERY_TERMS: Final = 64
MAX_DOC_ID_LENGTH: Final = 512  # Elasticsearch's limit for _id, in bytes
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
    """Knobs for one search. The defaults match Elasticsearch's.

    ``lucene_norms`` scores with document lengths rounded the way Lucene
    stores them (see ``scoring.lucene_length``), so scores match
    Elasticsearch's exactly. Turn it off to use the exact lengths.
    """

    ranker: Ranker = "bm25"
    mode: Mode = "or"
    fuzzy: bool = False
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    lucene_norms: bool = True


# ---------- checks shared with the Elasticsearch engine ----------


def normalize_doc_id(value: object) -> str:
    """Check a document id and return it as a str (an int is allowed too)."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"document id must be a str or int, not {type(value).__name__}")
    doc_id = str(value)
    if not doc_id or len(doc_id.encode()) > MAX_DOC_ID_LENGTH:
        raise ValueError(f"document id must be 1 to {MAX_DOC_ID_LENGTH} bytes")
    return doc_id


def normalize_document(doc: Mapping[str, Any]) -> dict[str, str]:
    """Check a document and return the copy of it that gets stored.

    The id may be a str or an int and is stored as a str. Every other value
    must be a string.
    """
    if not isinstance(doc, Mapping):
        raise TypeError(f"a document must be a mapping, not {type(doc).__name__}")
    doc_id = normalize_doc_id(doc.get("id"))
    stored = {"id": doc_id}
    for name, value in doc.items():
        if name == "id":
            continue
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError(f"document {doc_id}: field {name!r} must be a string")
        stored[name] = value
    return stored


def resolve_field_weights(
    fields: Sequence[str], field_weights: Mapping[str, float] | None
) -> dict[str, float]:
    """Check field weights and return them in field order.

    Only fields that get a weight are searched. ``None`` means the defaults:
    ``DEFAULT_FIELD_WEIGHTS``, and 1.0 for any other field. A weight must be
    between ``MIN_FIELD_WEIGHT`` and ``MAX_FIELD_WEIGHT``.
    """
    if field_weights is None:
        return {field: DEFAULT_FIELD_WEIGHTS.get(field, 1.0) for field in fields}
    if not isinstance(field_weights, Mapping):
        raise TypeError("field_weights must be a mapping of field name to weight")
    unknown = set(field_weights) - set(fields)
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown, key=str)}; fields are {list(fields)}")
    weights: dict[str, float] = {}
    for field in fields:
        if field in field_weights:
            weight = field_weights[field]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"the weight of {field!r} must be a number")
            weights[field] = float(weight)
    if not weights or not all(MIN_FIELD_WEIGHT <= w <= MAX_FIELD_WEIGHT for w in weights.values()):
        raise ValueError(
            f"field weights must be numbers from {MIN_FIELD_WEIGHT:g} to {MAX_FIELD_WEIGHT:g},"
            " for at least one field"
        )
    return weights


def split_phrase(query: str) -> tuple[str, bool]:
    """Strip the quotes off a "quoted phrase": (text, whether it was quoted)."""
    if not isinstance(query, str):
        raise TypeError("query must be a str")
    text = query.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1], True
    return text, False


def _check_options(options: SearchOptions) -> None:
    if not isinstance(options, SearchOptions):
        raise TypeError("options must be a SearchOptions")
    if options.ranker not in ("bm25", "tfidf"):
        raise ValueError(f"unknown ranker {options.ranker!r}")
    if options.mode not in ("or", "and"):
        raise ValueError(f"unknown mode {options.mode!r}")
    check_bm25_params(options.k1, options.b)


# ---------- checking a saved index before trusting it ----------


def _corrupt(path: str | Path, what: str) -> ValueError:
    return ValueError(f"{path} is corrupt: {what}")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_index_data(data: object, path: str | Path) -> dict[str, Any]:
    """Check a loaded index file completely, so a bad file fails here and
    not later, deep inside a search (JPL rule 7)."""
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
            raise _corrupt(path, f"{key!r} should be a {kind.__name__}")
    fields = data["fields"]
    if not fields or len(set(fields)) != len(fields) or not all(isinstance(f, str) for f in fields):
        raise _corrupt(path, "'fields' should hold distinct names")
    _check_documents(data["docs"], data["field_lengths"], fields, path)
    _check_postings(data["postings"], data["field_lengths"], set(fields), path)
    return data


def _check_documents(
    docs: dict[str, Any], lengths: dict[str, Any], fields: list[str], path: str | Path
) -> None:
    if set(docs) != set(lengths):
        raise _corrupt(path, "documents and lengths disagree")
    for doc_id, doc in docs.items():
        if not isinstance(doc, dict) or doc.get("id") != doc_id:
            raise _corrupt(path, f"document {doc_id!r} is not stored under its id")
        if not all(isinstance(value, str) for value in doc.values()):
            raise _corrupt(path, f"document {doc_id!r} has a value that is not a string")
        doc_lengths = lengths[doc_id]
        if not isinstance(doc_lengths, dict) or set(doc_lengths) != set(fields):
            raise _corrupt(path, f"document {doc_id!r} should have a length for every field")
        if not all(_is_count(length) for length in doc_lengths.values()):
            raise _corrupt(path, f"document {doc_id!r} has a length that is not a count")


def _check_postings(
    postings: dict[str, Any], lengths: dict[str, Any], fields: set[str], path: str | Path
) -> None:
    for term, docs in postings.items():
        if not isinstance(docs, dict) or not docs:
            raise _corrupt(path, f"term {term!r} has no documents")
        for doc_id, by_field in docs.items():
            if doc_id not in lengths or not isinstance(by_field, dict) or not by_field:
                raise _corrupt(path, f"term {term!r} points to unknown document {doc_id!r}")
            for field, positions in by_field.items():
                if field not in fields or not isinstance(positions, list) or not positions:
                    raise _corrupt(path, f"term {term!r} has bad positions in {doc_id!r}")
                if not all(_is_count(position) for position in positions):
                    raise _corrupt(path, f"term {term!r} has a position that is not a count")


class InvertedIndex:
    """Index documents, then search them."""

    def __init__(self, fields: Iterable[str] = DEFAULT_FIELDS) -> None:
        if isinstance(fields, str):
            raise TypeError("fields must be a list of field names, not one string")
        self.fields: tuple[str, ...] = tuple(fields)
        if not all(isinstance(field, str) and field for field in self.fields):
            raise TypeError("every field name must be a non-empty str")
        if not 0 < len(set(self.fields)) == len(self.fields) <= MAX_FIELDS:
            raise ValueError(f"fields must be 1 to {MAX_FIELDS} distinct names")
        self.postings: Postings = {}
        self.field_lengths: dict[str, dict[str, int]] = {}  # doc -> field -> terms
        self.docs: dict[str, dict[str, str]] = {}
        self.spelling = SpellingIndex()
        # Per field: documents with at least one term in it, and its total length.
        # add() keeps them up to date, so statistics never need a full scan.
        self._doc_counts = dict.fromkeys(self.fields, 0)
        self._total_lengths = dict.fromkeys(self.fields, 0)

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
        stored = normalize_document(doc)
        doc_id = stored["id"]
        if doc_id in self.docs:
            raise ValueError(f"duplicate document id: {doc_id}")
        analyzed = {field: analyze_tokens(stored.get(field, "")) for field in self.fields}

        for field, tokens in analyzed.items():
            self._index_tokens(doc_id, field, tokens)
        self.docs[doc_id] = stored
        self.field_lengths[doc_id] = {field: len(tokens) for field, tokens in analyzed.items()}
        self._count_lengths(self.field_lengths[doc_id])

    def add_many(self, docs: Iterable[Mapping[str, Any]]) -> int:
        """Index several documents. Returns how many were added."""
        count = 0
        for doc in docs:
            self.add(doc)
            count += 1
        return count

    def _index_tokens(self, doc_id: str, field: str, tokens: list[Token]) -> None:
        for token in tokens:
            fields = self.postings.setdefault(token.term, {}).setdefault(doc_id, {})
            fields.setdefault(field, []).append(token.position)
            self.spelling.add(token.surface)

    def _count_lengths(self, lengths: Mapping[str, int]) -> None:
        for field, length in lengths.items():
            self._doc_counts[field] += 1 if length else 0
            self._total_lengths[field] += length

    # ---------- statistics ----------

    def _collection_stats(
        self, weights: Mapping[str, float], lucene_norms: bool
    ) -> tuple[int, float]:
        """N and avgdl, the way Lucene's combined_fields computes them.

        N is the largest per-field document count (documents with at least
        one term in that field). avgdl is the weighted total length of the
        searched fields, divided by N. For one field, this is plain BM25.
        """
        n = max(self._doc_counts[field] for field in weights)
        if lucene_norms:
            total = float(lucene_weighted_total(self._total_lengths, weights))
        else:
            total = sum(self._total_lengths[field] * weight for field, weight in weights.items())
        return n, (total / n if n else 0.0)

    def _doc_length(self, doc_id: str, weights: Mapping[str, float], lucene_norms: bool) -> float:
        """The document's weighted length, as Lucene stores it or exact."""
        lengths = self.field_lengths[doc_id]
        if lucene_norms:
            return float(lucene_weighted_length(lengths, weights))
        return sum(lengths[field] * weight for field, weight in weights.items())

    def _term_stats(self, term: str, weights: Mapping[str, float]) -> tuple[dict[str, float], int]:
        """Field-weighted term frequency per document, and the document frequency.

        Like Lucene's combined_fields, the document frequency is the largest
        per-field one, not the number of documents with the term in any field.
        """
        tfs: dict[str, float] = {}
        dfs = dict.fromkeys(weights, 0)
        for doc_id, fields in self.postings.get(term, {}).items():
            tf = 0.0
            for field, weight in weights.items():
                count = len(fields.get(field, ()))
                if count:
                    dfs[field] += 1
                    tf += count * weight
            if tf > 0:
                tfs[doc_id] = tf
        return tfs, max(dfs.values())

    def _in_fields(self, term: str, fields: Iterable[str]) -> bool:
        """Whether the term occurs in at least one of these fields."""
        wanted = set(fields)
        return any(
            not wanted.isdisjoint(by_field) for by_field in self.postings.get(term, {}).values()
        )

    # ---------- query parsing ----------

    def parse(
        self, query: str, *, fuzzy: bool = False, fields: Iterable[str] | None = None
    ) -> ParsedQuery:
        """Analyze a query. Wrap it in double quotes for an exact phrase.

        With ``fuzzy``, a word that is not in ``fields`` (default: all fields)
        is replaced by the closest word that is.
        """
        text, phrase = split_phrase(query)
        tokens = analyze_tokens(text)
        if len(tokens) > MAX_QUERY_TERMS:
            raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
        if fuzzy:
            searched = self.fields if fields is None else tuple(fields)
            tokens = [self._correct_token(token, searched) for token in tokens]
        return ParsedQuery(tuple(tokens), phrase)

    def _correct_token(self, token: Token, fields: tuple[str, ...]) -> Token:
        if self._in_fields(token.term, fields):
            return token
        for candidate in self.spelling.candidates(token.surface):
            term = porter_stem(candidate)
            if self._in_fields(term, fields):
                return Token(term, candidate, token.position)
        return token

    def suggest(self, query: str) -> str | None:
        """'Did you mean ...?' The corrected query, or None if nothing changed."""
        words = tokenize(query)
        if sum(1 for word in words if is_searchable(word)) > MAX_QUERY_TERMS:
            raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
        fixed = [self._suggest_word(word) for word in words]
        return " ".join(fixed) if fixed != words else None

    def _suggest_word(self, word: str) -> str:
        if not is_searchable(word) or porter_stem(word) in self.postings:
            return word
        return self.spelling.correct(word)

    # ---------- matching ----------

    def _clauses(self, parsed: ParsedQuery, weights: Mapping[str, float]) -> list[Clause]:
        """Turn a query into scoring clauses: one per term, or one phrase."""
        if not parsed.phrase or len(parsed.tokens) < 2:
            clauses = []
            for token in parsed.tokens:
                term_tfs, df = self._term_stats(token.term, weights)
                clauses.append(Clause((token.term,), term_tfs, (df,)))
            return clauses
        first = parsed.tokens[0].position
        offsets = [(t.term, t.position - first) for t in parsed.tokens]
        stats = [self._term_stats(term, weights) for term, _ in offsets]
        candidates = set.intersection(*(set(term_tfs) for term_tfs, _ in stats))
        tfs: dict[str, float] = {}
        for doc_id in sorted(candidates):
            tf = sum(self._phrase_count(doc_id, field, offsets) * w for field, w in weights.items())
            if tf > 0:
                tfs[doc_id] = tf
        terms = tuple(term for term, _ in offsets)
        return [Clause(terms, tfs, tuple(df for _, df in stats))]

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

    @staticmethod
    def _allowed(clauses: list[Clause], mode: Mode) -> set[str] | None:
        """In AND mode, only documents that match every clause may score."""
        if mode == "or" or not clauses:
            return None
        return set.intersection(*(set(clause.tfs) for clause in clauses))

    # ---------- scoring ----------

    @staticmethod
    def _clause_idf(clause: Clause, n: int, ranker: Ranker) -> float:
        """A phrase's idf is the sum of its terms' idfs, as in Lucene."""
        if ranker == "bm25":
            return sum(bm25_idf(n, df) for df in clause.dfs)
        return sum(tfidf_idf(n, df) for df in clause.dfs)

    def _score_docs(
        self, clauses: list[Clause], weights: Mapping[str, float], options: SearchOptions
    ) -> dict[str, float]:
        n, avgdl = self._collection_stats(weights, options.lucene_norms)
        if n == 0:
            return {}
        allowed = self._allowed(clauses, options.mode)
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
        dl = self._doc_length(doc_id, weights, options.lucene_norms)
        return bm25_tf(tf, dl, avgdl, options.k1, options.b)

    def _prepare(
        self, query: str, options: SearchOptions | None, field_weights: Mapping[str, float] | None
    ) -> tuple[list[Clause], dict[str, float], SearchOptions]:
        """Check the arguments and match the query: what search and explain share."""
        options = SearchOptions() if options is None else options
        _check_options(options)
        weights = resolve_field_weights(self.fields, field_weights)
        parsed = self.parse(query, fuzzy=options.fuzzy, fields=weights)
        return self._clauses(parsed, weights), weights, options

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
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if not 1 <= k <= MAX_RESULTS:
            raise ValueError(f"k must be between 1 and {MAX_RESULTS}")
        clauses, weights, options = self._prepare(query, options, field_weights)
        scores = self._score_docs(clauses, weights, options)
        best = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [Hit(doc_id, score, self.docs[doc_id].get("title", "")) for doc_id, score in best]

    def explain(
        self,
        query: str,
        doc_id: str | int,
        *,
        options: SearchOptions | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> list[ClauseScore]:
        """Break one document's score down by query clause. Great for learning.

        The parts add up to the document's score in ``search``. A document
        that ``search`` would not return gets an empty list.
        """
        doc_id = normalize_doc_id(doc_id)
        clauses, weights, options = self._prepare(query, options, field_weights)
        n, avgdl = self._collection_stats(weights, options.lucene_norms)
        allowed = self._allowed(clauses, options.mode)
        if n == 0 or (allowed is not None and doc_id not in allowed):
            return []
        breakdown: list[ClauseScore] = []
        for clause in clauses:
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
        """Read an index written by ``save``, checking all of it first."""
        data = _check_index_data(json.loads(Path(path).read_text(encoding="utf-8")), path)
        index = cls(data["fields"])
        index.docs = data["docs"]
        index.field_lengths = data["field_lengths"]
        index.postings = data["postings"]
        index.spelling = SpellingIndex.from_counts(data["spelling"])
        for lengths in index.field_lengths.values():
            index._count_lengths(lengths)
        return index

    def stats(self) -> dict[str, Any]:
        """A few numbers that describe the index."""
        top = heapq.nlargest(5, self.postings, key=lambda term: (len(self.postings[term]), term))
        docs = max(len(self.docs), 1)
        return {
            "documents": len(self.docs),
            "unique_terms": len(self.postings),
            "avg_length": {
                field: round(self._total_lengths[field] / docs, 1) for field in self.fields
            },
            "most_common_terms": [(term, len(self.postings[term])) for term in top],
        }
