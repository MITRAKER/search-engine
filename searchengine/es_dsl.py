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

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from .analyzer import ENGLISH_STOPWORDS, MAX_TOKEN_LENGTH, Token, analyze_tokens, is_searchable
from .index import (
    DEFAULT_FIELDS,
    MAX_QUERY_LENGTH,
    MAX_QUERY_TERMS,
    MAX_RESULTS,
    SearchOptions,
    check_field_names,
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
MAX_PREFIX_LENGTH: Final = 256  # characters, for autocomplete
HIGHLIGHT_TAGS: Final = ("<mark>", "</mark>")
# Elasticsearch refuses to highlight past index.highlight.max_analyzed_offset
# (1,000,000 characters) and fails the whole search, unless the request sets a
# lower limit: then it highlights only the start of a longer field.
HIGHLIGHT_MAX_OFFSET: Final = 999_999
_FIELD_NAME: Final = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


# ---------- the index ----------


def check_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Field names that are safe to map: the same limits as ``InvertedIndex``
    (1 to MAX_FIELDS distinct names), and each a letter first, then letters,
    digits or '_', and not reserved."""
    names = check_field_names(fields)
    for name in names:
        if not _FIELD_NAME.fullmatch(name):
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
    """Field types. Searchable fields use our analyzer and feed the spelling field.

    The document id is also kept as a keyword, so results with the same score
    can be sorted by id, as ``InvertedIndex`` sorts them.
    """
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
    properties["id"] = {"type": "keyword"}  # to break ties between equal scores
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


def index_fingerprint(settings: Mapping[str, Any], mappings: Mapping[str, Any]) -> str:
    """A hash of the settings and mappings an index is made with.

    The engine stores it in the index (in the mapping's ``_meta``) and reuses
    an index only if it finds its own fingerprint there. So an index made
    with another analyzer, other fields, other BM25 parameters or more
    shards, by hand or by another version of this code, is never reused.
    """
    data = json.dumps([settings, mappings], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


# ---------- queries ----------


def build_query(query: str, options: SearchOptions, weights: Mapping[str, float]) -> dict[str, Any]:
    """Translate a query and its options into Elasticsearch's query DSL.

    * words      -> ``combined_fields``: BM25F, the same scores as our engine
    * "a phrase" -> ``multi_match`` of type ``phrase``: on one field, the same
      score as ours; on several, the best field's score
    * fuzzy      -> one fuzzy ``multi_match`` per word, joined by ``bool``

    ``weights`` must already be checked (``index.resolve_field_weights``).
    """
    check_options(options)
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
            # Elasticsearch's phrase queries have no fuzziness. Intervals and span
            # queries can match a fuzzy phrase, but they score differently.
            raise ValueError("no fuzzy phrase search on Elasticsearch: drop the quotes or fuzzy")
        return {"multi_match": {"query": text, "type": "phrase", "fields": fields}}
    if options.fuzzy:
        return _fuzzy_query(tokens, fields, options.mode)
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


def build_search_body(
    query: Mapping[str, Any], k: int, fields: Iterable[str], category: str | None
) -> dict[str, Any]:
    """The whole request: query, category filter, facets and highlighting.

    The category filter goes in ``post_filter``, which runs *after* the facets
    are counted. So the facets still count every category the query matches,
    and a user can switch to another category without searching again.
    Results with the same score are sorted by id, as ``InvertedIndex`` does.
    """
    if not isinstance(query, Mapping):
        raise TypeError("query must be a query DSL mapping")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be an int between 1 and {MAX_RESULTS}")
    body: dict[str, Any] = {
        "query": dict(query),
        "size": k,
        "sort": [{"_score": "desc"}, {"id": "asc"}],
        "aggs": {"facets": {"terms": {"field": FACET_FIELD, "size": MAX_FACETS}}},
        "highlight": {
            "encoder": "html",  # escape the document text, so snippets are safe in a page
            "pre_tags": [HIGHLIGHT_TAGS[0]],
            "post_tags": [HIGHLIGHT_TAGS[1]],
            "max_analyzed_offset": HIGHLIGHT_MAX_OFFSET,
            "fields": {name: {} for name in check_field_names(fields)},
        },
        "track_total_hits": True,  # an exact count, not "10,000 or more"
    }
    if category is not None:
        if not isinstance(category, str) or not 0 < len(category) <= MAX_CATEGORY_LENGTH:
            raise ValueError(f"category must be a str of 1 to {MAX_CATEGORY_LENGTH} characters")
        body["post_filter"] = {"term": {FACET_FIELD: category}}
    return body


def autocomplete_query(prefix: str) -> dict[str, Any]:
    """Match titles that contain every word the user typed.

    ``bool_prefix`` treats every word as a whole term except the last, which
    may be the start of a word ("inv" for "inverted"). The operator "and"
    makes every word count: by default one matching word would be enough.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")
    if len(prefix) > MAX_PREFIX_LENGTH:
        raise ValueError(f"prefix is longer than {MAX_PREFIX_LENGTH} characters")
    complete = "title.complete"
    fields = [complete, f"{complete}._2gram", f"{complete}._3gram"]
    return {
        "multi_match": {"query": prefix, "type": "bool_prefix", "fields": fields, "operator": "and"}
    }


# ---------- "did you mean" ----------


def _check_words(words: Sequence[str]) -> None:
    """The words of one query: a list of str, no more than a query can hold."""
    if isinstance(words, str) or not isinstance(words, Sequence):
        raise TypeError("words must be a list of words")
    if len(words) > MAX_QUERY_LENGTH:
        raise ValueError(f"a query has at most {MAX_QUERY_LENGTH} words")
    if not all(isinstance(word, str) for word in words):
        raise TypeError("every word must be a str")


def build_suggest_request(words: Sequence[str], fields: Iterable[str]) -> dict[str, Any]:
    """The "did you mean" request for a query's words, or {} if none may be misspelled.

    Each word that may be misspelled gets a term suggester, with the same
    number of allowed typos as fuzziness AUTO (and as our ``SpellingIndex``):
    none up to 2 letters, 1 up to 5, then 2. The suggester only knows the
    words of the spelling field, which are not stemmed. So the request also
    counts the documents each word matches in the searched fields, where it
    is stemmed: a word that matches a document is known, and stays as it is,
    as ``InvertedIndex.suggest`` keeps a word whose stem is in the index.
    """
    _check_words(words)
    names = check_field_names(fields)
    suggesters: dict[str, Any] = {}
    known: dict[str, Any] = {}
    for i, word in enumerate(words):
        edits = max_edits_auto(word)
        if edits and is_searchable(word):
            suggesters[f"word{i}"] = {
                "text": word,
                "term": {
                    "field": SPELLING_FIELD,
                    "suggest_mode": "missing",  # only fix words the spelling field lacks
                    "max_edits": edits,
                    "prefix_length": 0,  # allow a typo in the first letter too
                    "min_word_length": 1,
                },
            }
            known[f"word{i}"] = {"multi_match": {"query": word, "fields": list(names)}}
    if len(suggesters) > MAX_QUERY_TERMS:
        raise ValueError(f"queries are limited to {MAX_QUERY_TERMS} terms")
    if not suggesters:
        return {}
    return {"suggest": suggesters, "aggs": {"known": {"filters": {"filters": known}}}}


def apply_suggestions(words: Sequence[str], response: Mapping[str, Any]) -> str | None:
    """Rebuild the query with each unknown, misspelled word replaced by its best fix."""
    _check_words(words)
    suggest = response.get("suggest", {})
    known = response.get("aggregations", {}).get("known", {}).get("buckets", {})
    fixed = []
    for i, word in enumerate(words):
        entries = suggest.get(f"word{i}") or [{"options": []}]
        options = entries[0]["options"]
        matches = known.get(f"word{i}", {}).get("doc_count", 0)
        fixed.append(str(options[0]["text"]) if options and not matches else word)
    return " ".join(fixed) if fixed != list(words) else None


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
