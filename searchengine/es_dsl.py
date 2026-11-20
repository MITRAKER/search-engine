"""Elasticsearch requests as plain data: index settings, mappings and queries.

Elasticsearch is driven by JSON. These functions only build that JSON, as
Python dicts: they need no server and no client library. So they are easy to
test, and ``scripts/make_parity_golden.py`` records Elasticsearch's scores
with exactly the analyzer and the queries the engine uses. ``es_engine.py``
sends them to a real server.

The index is set up so Elasticsearch scores like ``InvertedIndex``:

* a custom analyzer built from the same steps as ``analyzer.py``: split on
  anything that is not a letter, combining mark or digit, lowercase, drop
  tokens longer than 255, drop Lucene's English stopwords, Porter stemmer;
* BM25 with the same k1 and b, and one shard, so every document is scored
  with the same statistics;
* ``combined_fields`` queries, which are BM25F, like ``InvertedIndex.search``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final, Literal, TypeAlias

from .analyzer import ENGLISH_STOPWORDS, MAX_TOKEN_LENGTH, Token, analyze_tokens, is_searchable
from .index import (
    DEFAULT_FIELDS,
    MAX_QUERY_TERMS,
    MAX_RESULTS,
    SearchOptions,
    check_options,
    split_phrase,
)
from .scoring import DEFAULT_B, DEFAULT_K1, check_bm25_params
from .spelling import max_edits_auto

ANALYZER: Final = "scratch_english"  # the same steps as analyzer.py
WORDS_ANALYZER: Final = "scratch_words"  # the same, without stemming: real words, for spelling
PLAIN_ANALYZER: Final = "plain"  # keeps every word: for autocomplete
SPELLING_FIELD: Final = "spelling"
FACET_FIELD: Final = "category"
RESERVED_FIELDS: Final = frozenset({"id", SPELLING_FIELD, FACET_FIELD})
# The tokenizer splits on this Java regex: runs of anything that is not a
# letter (\p{L}), a combining mark (\p{M}) or a digit (\p{N}), in any script.
WORD_SEPARATORS: Final = r"[^\p{L}\p{M}\p{N}]+"
MAX_FACETS: Final = 50
MAX_CATEGORY_LENGTH: Final = 256  # characters
MAX_EXPLAIN_LINES: Final = 200
HIGHLIGHT_TAGS: Final = ("<mark>", "</mark>")
# BEIR's Elasticsearch baseline: best_fields, with half the score of the other fields.
BEST_FIELDS_TIE_BREAKER: Final = 0.5
_FIELD_NAME: Final = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


# ---------- the index ----------


def check_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Field names that are safe to map: a letter first, no dots, not reserved."""
    if isinstance(fields, str):
        raise TypeError("fields must be a list of names, not one string")
    names = tuple(fields)
    if not names or len(set(names)) != len(names):
        raise ValueError("fields must be a non-empty list of distinct names")
    for name in names:
        if not isinstance(name, str) or not _FIELD_NAME.fullmatch(name):
            raise ValueError(f"invalid field name {name!r}: use letters, digits and '_'")
        if name in RESERVED_FIELDS:
            raise ValueError(f"field name {name!r} is reserved")
    return names


def analysis_settings() -> dict[str, Any]:
    """Elasticsearch analyzers built from the same steps as ``analyzer.py``."""
    base = ["lowercase", "max_token_length"]
    return {
        "tokenizer": {"words": {"type": "pattern", "pattern": WORD_SEPARATORS}},
        "filter": {
            # "length" drops longer tokens; the standard tokenizer would cut them up instead.
            "max_token_length": {"type": "length", "max": MAX_TOKEN_LENGTH},
            "lucene_english_stop": {"type": "stop", "stopwords": sorted(ENGLISH_STOPWORDS)},
        },
        "analyzer": {
            ANALYZER: {
                "type": "custom",
                "tokenizer": "words",
                "filter": [*base, "lucene_english_stop", "porter_stem"],
            },
            WORDS_ANALYZER: {
                "type": "custom",
                "tokenizer": "words",
                "filter": [*base, "lucene_english_stop"],
            },
            PLAIN_ANALYZER: {"type": "custom", "tokenizer": "words", "filter": base},
        },
    }


def index_mappings(fields: Iterable[str] = DEFAULT_FIELDS) -> dict[str, Any]:
    """Field types. Searchable fields use our analyzer and feed the spelling field."""
    properties: dict[str, Any] = {
        name: {"type": "text", "analyzer": ANALYZER, "copy_to": SPELLING_FIELD}
        for name in check_fields(fields)
    }
    if "title" in properties:
        # search_as_you_type adds shingle and prefix sub-fields, for autocomplete.
        properties["title"]["fields"] = {
            "complete": {"type": "search_as_you_type", "analyzer": PLAIN_ANALYZER}
        }
    properties[SPELLING_FIELD] = {"type": "text", "analyzer": WORDS_ANALYZER}
    properties[FACET_FIELD] = {"type": "keyword"}  # exact values, for filters and facets
    # Other fields, like "url", are kept in _source and returned, but not indexed.
    return {"dynamic": "false", "properties": properties}


def index_settings(k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> dict[str, Any]:
    """One shard, so scores use the same statistics as the from-scratch engine."""
    check_bm25_params(k1, b)
    return {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}},
        "analysis": analysis_settings(),
    }


# ---------- queries ----------


QueryType: TypeAlias = Literal["combined_fields", "best_fields"]


def build_query(
    query: str,
    options: SearchOptions,
    weights: Mapping[str, float],
    query_type: QueryType = "combined_fields",
) -> dict[str, Any]:
    """Translate a query and its options into Elasticsearch's query DSL.

    * words      -> ``combined_fields``: BM25F, the same scores as our engine.
      With ``query_type="best_fields"``, a field-centric ``multi_match``
      instead: each field scored on its own (for comparison, in benchmarks)
    * "a phrase" -> ``multi_match`` of type ``phrase``: on one field, the same
      score as ours; on several, the best field's score
    * fuzzy      -> one fuzzy ``multi_match`` per word, joined by ``bool``

    ``weights`` must already be checked (``index.resolve_field_weights``).
    """
    check_options(options)
    if query_type not in ("combined_fields", "best_fields"):
        raise ValueError("query_type must be combined_fields or best_fields")
    if options.ranker != "bm25":
        raise ValueError("the Elasticsearch engine ranks with BM25 only")
    if not options.lucene_norms:
        raise ValueError("Elasticsearch always stores lengths in one byte: lucene_norms=False")
    text, phrase = split_phrase(query)
    tokens = analyze_tokens(text)
    if len(tokens) > MAX_QUERY_TERMS:
        raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
    if not tokens:
        return {"match_none": {}}  # like our engine: no terms, no results
    fields = [f"{name}^{weight}" for name, weight in weights.items()]
    if phrase:
        if options.fuzzy:
            raise ValueError("Elasticsearch has no fuzzy phrase query: drop the quotes or fuzzy")
        return {"multi_match": {"query": text, "type": "phrase", "fields": fields}}
    if options.fuzzy:
        return _fuzzy_query(tokens, fields, options.mode)
    if query_type == "best_fields":
        return {
            "multi_match": {
                "query": text,
                "type": "best_fields",
                "fields": fields,
                "operator": options.mode,  # "and": every word in the SAME field
                "tie_breaker": BEST_FIELDS_TIE_BREAKER,
            }
        }
    return {"combined_fields": {"query": text, "fields": fields, "operator": options.mode}}


def _fuzzy_query(tokens: list[Token], fields: list[str], mode: str) -> dict[str, Any]:
    """One fuzzy clause per word.

    multi_match's ``operator`` applies inside each field, so "and" would need
    every word in the SAME field. A bool query of one clause per word means
    "every word, in any field", like our engine.
    """
    words = [
        {"multi_match": {"query": token.surface, "fields": fields, "fuzziness": "AUTO"}}
        for token in tokens
    ]
    return {"bool": {"must" if mode == "and" else "should": words}}


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be an int between 1 and {MAX_RESULTS}")


def build_search_body(
    query: Mapping[str, Any], k: int, fields: Iterable[str], category: str | None
) -> dict[str, Any]:
    """The whole request: query, category filter, facets and highlighting.

    The category filter goes in ``post_filter``, which runs *after* the facets
    are counted. So the facets still count every category the query matches,
    and a user can switch to another category without searching again.
    """
    _check_k(k)
    body: dict[str, Any] = {
        "query": dict(query),
        "size": k,
        "aggs": {"facets": {"terms": {"field": FACET_FIELD, "size": MAX_FACETS}}},
        "highlight": {
            "encoder": "html",  # escape the document text, so snippets are safe in a page
            "pre_tags": [HIGHLIGHT_TAGS[0]],
            "post_tags": [HIGHLIGHT_TAGS[1]],
            "fields": {name: {} for name in fields},
        },
        "track_total_hits": True,  # an exact count, not "10,000 or more"
    }
    if category is not None:
        if not isinstance(category, str) or not 0 < len(category) <= MAX_CATEGORY_LENGTH:
            raise ValueError(f"category must be a str of 1 to {MAX_CATEGORY_LENGTH} characters")
        body["post_filter"] = {"term": {FACET_FIELD: category}}
    return body


def build_rank_body(query: Mapping[str, Any], k: int) -> dict[str, Any]:
    """A lean request that only ranks: ids and scores, for benchmarks.

    No document text, snippets, facets or exact count: each costs time, and
    a benchmark of ranking should time the ranking alone.
    """
    _check_k(k)
    return {"query": dict(query), "size": k, "source": False, "track_total_hits": False}


def autocomplete_query(prefix: str) -> dict[str, Any]:
    """Match titles that start with what the user typed, word by word.

    ``bool_prefix`` treats every word as a whole term except the last, which
    may be the start of a word ("inv" for "inverted").
    """
    complete = "title.complete"
    fields = [complete, f"{complete}._2gram", f"{complete}._3gram"]
    return {"multi_match": {"query": prefix, "type": "bool_prefix", "fields": fields}}


# ---------- "did you mean" ----------


def build_suggesters(words: list[str]) -> dict[str, Any]:
    """One term suggester per word that may be misspelled.

    Each word gets the same number of allowed typos as fuzziness AUTO (and as
    our ``SpellingIndex``): none up to 2 letters, 1 up to 5, then 2.
    """
    suggesters: dict[str, Any] = {}
    for i, word in enumerate(words):
        edits = max_edits_auto(word)
        if edits and is_searchable(word):
            suggesters[f"word{i}"] = {
                "text": word,
                "term": {
                    "field": SPELLING_FIELD,
                    "suggest_mode": "missing",  # only fix words the index does not know
                    "max_edits": edits,
                    "prefix_length": 0,  # allow a typo in the first letter too
                    "min_word_length": 1,
                },
            }
    return suggesters


def apply_suggestions(words: list[str], suggest: Mapping[str, Any]) -> str | None:
    """Rebuild the query with each misspelled word replaced by its best fix."""
    fixed = []
    for i, word in enumerate(words):
        entries = suggest.get(f"word{i}") or [{"options": []}]
        options = entries[0]["options"]
        fixed.append(str(options[0]["text"]) if options else word)
    return " ".join(fixed) if fixed != words else None


# ---------- explanations ----------


def explanation_lines(explanation: Mapping[str, Any], max_depth: int = 3) -> list[str]:
    """Flatten Elasticsearch's explanation tree into indented lines.

    A stack replaces recursion (JPL rule 1), and the output is capped.
    """
    lines: list[str] = []
    stack: list[tuple[Mapping[str, Any], int]] = [(explanation, 0)]
    for _ in range(MAX_EXPLAIN_LINES):
        if not stack:
            break
        node, depth = stack.pop()
        value = float(node["value"])
        lines.append(f"{'  ' * depth}{value:.3f}  {node['description']}")
        if depth < max_depth:
            stack.extend((child, depth + 1) for child in reversed(node.get("details", [])))
    return lines
