"""The Elasticsearch requests, checked as plain data (no server needed).

Whether Elasticsearch really scores and analyzes like our engine is checked
against its recorded answers in test_es_parity.py, and live in
test_es_integration.py.
"""

import itertools
from typing import Any

import pytest

from searchengine import ENGLISH_STOPWORDS, SearchOptions
from searchengine.analyzer import MAX_TOKEN_LENGTH
from searchengine.es_dsl import (
    ANALYZER,
    FACET_FIELD,
    HIGHLIGHT_MAX_OFFSET,
    HIGHLIGHT_TAGS,
    MAX_CATEGORY_LENGTH,
    MAX_EXPLAIN_LINES,
    MAX_PREFIX_LENGTH,
    PLAIN_ANALYZER,
    SPELLING_FIELD,
    WORD_SEPARATORS,
    WORDS_ANALYZER,
    analysis_settings,
    apply_suggestions,
    autocomplete_query,
    build_query,
    build_search_body,
    build_suggest_request,
    check_fields,
    explanation_lines,
    index_fingerprint,
    index_mappings,
    index_settings,
)
from searchengine.index import MAX_FIELDS, MAX_QUERY_LENGTH, MAX_QUERY_TERMS, MAX_RESULTS

WEIGHTS = {"title": 3.0, "body": 1.0}

# ---------- the index ----------


def test_the_analyzer_has_the_same_steps_as_ours():
    analysis = analysis_settings()
    assert analysis["tokenizer"]["words"] == {"type": "pattern", "pattern": WORD_SEPARATORS}
    assert WORD_SEPARATORS == r"[^\p{L}\p{M}\p{N}]+"  # letters, combining marks, digits
    steps = analysis["analyzer"][ANALYZER]
    assert steps["tokenizer"] == "words"
    assert steps["filter"] == [
        "lowercase",
        "max_token_length",
        "lucene_english_stop",
        "porter_stem",
    ]
    filters = analysis["filter"]
    assert filters["max_token_length"] == {"type": "length", "max": MAX_TOKEN_LENGTH}
    assert filters["lucene_english_stop"]["stopwords"] == sorted(ENGLISH_STOPWORDS)


def test_spelling_and_autocomplete_analyzers_skip_steps():
    analyzers = analysis_settings()["analyzer"]
    assert "porter_stem" not in analyzers[WORDS_ANALYZER]["filter"]  # real words, not stems
    assert analyzers[PLAIN_ANALYZER]["filter"] == ["lowercase", "max_token_length"]


def test_mappings_index_each_field_with_our_analyzer():
    mappings = index_mappings()
    properties = mappings["properties"]
    assert mappings["dynamic"] == "false"  # other fields are stored, not indexed
    for name in ("title", "body"):
        assert properties[name]["analyzer"] == ANALYZER
        assert properties[name]["copy_to"] == SPELLING_FIELD
    assert properties["title"]["fields"]["complete"]["type"] == "search_as_you_type"
    assert properties[SPELLING_FIELD]["analyzer"] == WORDS_ANALYZER
    assert properties[FACET_FIELD] == {"type": "keyword"}
    assert properties["id"] == {"type": "keyword"}  # to sort ties by id


def test_mappings_without_a_title_have_no_autocomplete():
    properties = index_mappings(["body", "abstract"])["properties"]
    assert set(properties) == {"body", "abstract", SPELLING_FIELD, FACET_FIELD, "id"}
    assert "fields" not in properties["body"]


@pytest.mark.parametrize(
    "fields",
    [
        [],
        ["title", "title"],
        ["1st"],
        ["a.b"],
        ["has space"],
        ["x" * 65],
        [f"f{n}" for n in range(MAX_FIELDS + 1)],
        itertools.repeat("body"),  # endless: rejected after MAX_FIELDS + 1 names
    ],
)
def test_field_names_must_be_safe(fields):
    with pytest.raises(ValueError, match="field"):
        check_fields(fields)


@pytest.mark.parametrize("fields", [[None], [""], [b"title"]])
def test_field_names_must_be_text(fields):
    with pytest.raises(TypeError, match="field name"):
        check_fields(fields)


@pytest.mark.parametrize("name", sorted({"id", SPELLING_FIELD, FACET_FIELD}))
def test_reserved_field_names_are_refused(name):
    with pytest.raises(ValueError, match="reserved"):
        check_fields(["title", name])


def test_fields_must_be_a_list_not_a_string():
    with pytest.raises(TypeError, match="list"):
        check_fields("title")


def test_settings_use_one_shard_and_our_bm25():
    settings = index_settings(k1=1.5, b=0.5)
    assert settings["number_of_shards"] == 1
    assert settings["similarity"]["default"] == {"type": "BM25", "k1": 1.5, "b": 0.5}
    assert settings["analysis"] == analysis_settings()


@pytest.mark.parametrize(("k1", "b"), [(-1, 0.75), (1.2, 2.0), (float("nan"), 0.75)])
def test_settings_reject_bad_bm25_parameters(k1, b):
    with pytest.raises(ValueError, match="k1"):
        index_settings(k1, b)


# ---------- queries ----------


def test_words_become_combined_fields():
    query = build_query("inverted index", SearchOptions(mode="and"), WEIGHTS)
    assert query == {
        "combined_fields": {
            "query": "inverted index",
            "fields": ["title^3.0", "body^1.0"],
            "operator": "and",
        }
    }


def test_a_phrase_becomes_a_phrase_query():
    query = build_query('"search engine"', SearchOptions(), {"body": 1.0})
    assert query == {
        "multi_match": {"query": "search engine", "type": "phrase", "fields": ["body^1.0"]}
    }


@pytest.mark.parametrize(("mode", "occur"), [("or", "should"), ("and", "must")])
def test_fuzzy_becomes_one_clause_per_word(mode, occur):
    query = build_query("The Rankng elastisearch", SearchOptions(fuzzy=True, mode=mode), WEIGHTS)
    clauses = query["bool"][occur]
    assert [clause["multi_match"]["query"] for clause in clauses] == ["rankng", "elastisearch"]
    assert all(clause["multi_match"]["fuzziness"] == "AUTO" for clause in clauses)


def test_a_query_without_terms_matches_nothing():
    assert build_query("the and of", SearchOptions(), WEIGHTS) == {"match_none": {}}
    assert build_query('""', SearchOptions(), WEIGHTS) == {"match_none": {}}


@pytest.mark.parametrize(
    ("query", "options", "message"),
    [
        ("x", SearchOptions(ranker="tfidf"), "BM25 only"),
        ("x", SearchOptions(lucene_norms=False), "one byte"),
        ('"a phrase"', SearchOptions(fuzzy=True), "no fuzzy phrase search"),
        (" ".join(["word"] * (MAX_QUERY_TERMS + 1)), SearchOptions(), "limited"),
        ("x", SearchOptions(mode="xor"), "mode"),  # type: ignore[arg-type]
        ("x", SearchOptions(k1=-1), "k1"),
    ],
)
def test_queries_the_engine_cannot_run_are_refused(query, options, message):
    with pytest.raises(ValueError, match=message):
        build_query(query, options, WEIGHTS)


def test_options_must_be_search_options():
    with pytest.raises(TypeError, match="SearchOptions"):
        build_query("x", {"mode": "and"}, WEIGHTS)  # type: ignore[arg-type]


def test_search_body_counts_facets_before_the_category_filter():
    body = build_search_body({"match_all": {}}, 10, ["title", "body"], "ranking")
    assert body["size"] == 10
    assert body["aggs"]["facets"]["terms"]["field"] == FACET_FIELD
    assert body["post_filter"] == {"term": {FACET_FIELD: "ranking"}}  # after the facets
    assert body["highlight"]["encoder"] == "html"  # snippets are escaped
    assert body["highlight"]["pre_tags"] == [HIGHLIGHT_TAGS[0]]
    assert set(body["highlight"]["fields"]) == {"title", "body"}
    assert body["track_total_hits"] is True
    assert "post_filter" not in build_search_body({"match_all": {}}, 10, ["body"], None)


def test_search_body_sorts_ties_by_id_like_our_engine():
    body = build_search_body({"match_all": {}}, 10, ["body"], None)
    assert body["sort"] == [{"_score": "desc"}, {"id": "asc"}]


def test_search_body_highlights_only_the_start_of_very_long_fields():
    # Below index.highlight.max_analyzed_offset (1,000,000), so a longer field
    # is highlighted in part instead of failing the whole search.
    body = build_search_body({"match_all": {}}, 10, ["body"], None)
    assert body["highlight"]["max_analyzed_offset"] == HIGHLIGHT_MAX_OFFSET < 1_000_000


@pytest.mark.parametrize(
    ("query", "fields", "error"),
    [
        ("match_all", ["body"], TypeError),
        ({"match_all": {}}, "body", TypeError),
        ({"match_all": {}}, [], ValueError),
    ],
)
def test_search_body_checks_the_query_and_fields(query, fields, error):
    with pytest.raises(error):
        build_search_body(query, 10, fields, None)


@pytest.mark.parametrize("k", [0, MAX_RESULTS + 1, True, 2.5])
def test_search_body_checks_k(k):
    with pytest.raises(ValueError, match="k must be"):
        build_search_body({"match_all": {}}, k, ["body"], None)


@pytest.mark.parametrize("category", ["", "x" * (MAX_CATEGORY_LENGTH + 1), 3])
def test_search_body_checks_the_category(category):
    with pytest.raises(ValueError, match="category"):
        build_search_body({"match_all": {}}, 10, ["body"], category)


def test_autocomplete_matches_the_last_word_as_a_prefix():
    query = autocomplete_query("inverted ind")["multi_match"]
    assert query["type"] == "bool_prefix"
    assert query["fields"] == ["title.complete", "title.complete._2gram", "title.complete._3gram"]
    assert query["operator"] == "and"  # every typed word must match, not just one


@pytest.mark.parametrize(
    ("prefix", "error"), [(42, TypeError), ("x" * (MAX_PREFIX_LENGTH + 1), ValueError)]
)
def test_autocomplete_checks_the_prefix(prefix, error):
    with pytest.raises(error):
        autocomplete_query(prefix)


# ---------- "did you mean" ----------


def test_suggesters_skip_words_too_short_to_fix_and_stopwords():
    request = build_suggest_request(["the", "of", "rankng", "elastisearch", "bm"], ["body"])
    suggesters = request["suggest"]
    assert set(suggesters) == {"word2", "word3"}  # numbered by position in the query
    assert suggesters["word2"]["term"]["max_edits"] == 2  # 6 letters: 2 typos, like AUTO
    assert suggesters["word2"]["term"]["field"] == SPELLING_FIELD
    rnk = build_suggest_request(["rnk"], ["body"])
    assert rnk["suggest"]["word0"]["term"]["max_edits"] == 1
    assert build_suggest_request(["the", "bm"], ["body"]) == {}  # nothing to ask


def test_suggest_request_counts_the_documents_each_word_matches():
    # The spelling field is not stemmed, so "analyzer" can be missing from it
    # while its stem matches documents: those counts say which words are known.
    request = build_suggest_request(["the", "analyzer"], ["title", "body"])
    known = request["aggs"]["known"]["filters"]["filters"]
    assert known == {"word1": {"multi_match": {"query": "analyzer", "fields": ["title", "body"]}}}


def test_suggestions_replace_only_unknown_misspelled_words():
    words = ["the", "rankng", "elastisearch", "analyzer"]
    response = {
        "suggest": {
            "word1": [{"options": [{"text": "ranking"}]}],
            "word2": [{"options": []}],
            "word3": [{"options": [{"text": "analyzers"}]}],
        },
        "aggregations": {
            "known": {"buckets": {"word1": {"doc_count": 0}, "word3": {"doc_count": 3}}}
        },
    }
    assert apply_suggestions(words, response) == "the ranking elastisearch analyzer"
    assert apply_suggestions(words, {}) is None  # nothing to fix


@pytest.mark.parametrize(
    ("words", "error"),
    [
        ("rankng", TypeError),
        ([1], TypeError),
        (["x"] * (MAX_QUERY_LENGTH + 1), ValueError),
        (["rankng"] * (MAX_QUERY_TERMS + 1), ValueError),
    ],
)
def test_suggestions_check_the_words(words, error):
    with pytest.raises(error):
        build_suggest_request(words, ["body"])


# ---------- explanations ----------


def node(value: float, description: str, *details: dict[str, Any]) -> dict[str, Any]:
    return {"value": value, "description": description, "details": list(details)}


def test_explanations_are_flattened_with_indentation():
    tree = node(2.5, "sum of:", node(2.0, "weight(title:bm25)"), node(0.5, "weight(body:bm25)"))
    assert explanation_lines(tree) == [
        "2.500  sum of:",
        "  2.000  weight(title:bm25)",
        "  0.500  weight(body:bm25)",
    ]


def test_explanations_stop_at_the_depth_limit():
    deep = node(1, "a", node(1, "b", node(1, "c")))
    assert len(explanation_lines(deep, max_depth=1)) == 2


def test_explanations_are_capped():
    wide = node(1, "root", *[node(0, f"child {i}") for i in range(2 * MAX_EXPLAIN_LINES)])
    assert len(explanation_lines(wide)) == MAX_EXPLAIN_LINES


# ---------- the fingerprint ----------


def test_the_fingerprint_is_a_sha256_of_settings_and_mappings_in_any_key_order():
    fingerprint = index_fingerprint(index_settings(), index_mappings())
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0  # hexadecimal
    reordered = dict(reversed(list(index_settings().items())))
    assert index_fingerprint(reordered, index_mappings()) == fingerprint


def test_the_fingerprint_changes_with_any_setting_or_mapping():
    fingerprint = index_fingerprint(index_settings(), index_mappings())
    other_analyzer = index_settings()
    other_analyzer["analysis"]["analyzer"][ANALYZER]["tokenizer"] = "standard"
    others = [
        (index_settings(k1=2.0), index_mappings()),
        (index_settings(b=0.5), index_mappings()),
        (other_analyzer, index_mappings()),
        (index_settings() | {"number_of_shards": 3}, index_mappings()),
        (index_settings(), index_mappings(["title"])),
    ]
    assert all(index_fingerprint(*other) != fingerprint for other in others)
