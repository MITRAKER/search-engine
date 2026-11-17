import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from searchengine import InvertedIndex, SearchOptions
from searchengine.index import MAX_QUERY_TERMS, MAX_RESULTS
from tests.helpers import make_index

# ---------- indexing ----------


def test_postings_store_positions_per_field():
    index = make_index("inverted index of terms", titles=("Inverted",))
    assert index.postings["invert"] == {"1": {"title": [0], "body": [0]}}
    assert index.postings["term"] == {"1": {"body": [3]}}
    assert index.field_lengths["1"] == {"title": 1, "body": 3}


def test_extra_fields_are_stored_not_indexed():
    index = InvertedIndex()
    index.add({"id": 7, "title": "T", "body": "B", "url": "https://example.org"})
    assert index.docs["7"]["url"] == "https://example.org"
    assert "exampl" not in index.postings


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        ({"title": "no id"}, TypeError),
        ({"id": True, "title": "bool id"}, TypeError),
        ({"id": "", "title": "empty id"}, ValueError),
        ({"id": "x" * 513}, ValueError),
        ({"id": "1", "title": 5}, TypeError),
    ],
)
def test_invalid_documents_are_rejected(doc, error):
    with pytest.raises(error):
        InvertedIndex().add(doc)


def test_duplicate_id_rejected():
    index = make_index("first")
    with pytest.raises(ValueError, match="duplicate"):
        index.add({"id": "1", "body": "second"})


def test_failed_add_leaves_index_unchanged():
    index = make_index("hello world")
    before = (json.dumps(index.postings), dict(index.field_lengths))
    with pytest.raises(TypeError):
        index.add({"id": "2", "title": "fine", "body": 123})
    assert (json.dumps(index.postings), dict(index.field_lengths)) == before
    assert len(index) == 1


def test_fields_must_be_distinct_and_non_empty():
    with pytest.raises(ValueError, match="fields"):
        InvertedIndex(fields=())
    with pytest.raises(ValueError, match="fields"):
        InvertedIndex(fields=("a", "a"))


# ---------- BM25F, checked against the formula by hand ----------


def test_bm25f_score_matches_hand_calculation():
    index = make_index("cat cat dog", "dog bird", "fish", titles=("cat",))
    # Document 1: tf = 3 (title weight) * 1 + 1 * 2 = 5, length = 3 * 1 + 3 = 6
    n, df = 3, 1
    lengths = [6, 2, 1]  # weighted lengths of the three documents
    avgdl = sum(lengths) / n
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    tf_part = 5 * 2.2 / (5 + 1.2 * (1 - 0.75 + 0.75 * 6 / avgdl))
    [hit] = index.search("cat")
    assert hit.doc_id == "1"
    assert hit.score == pytest.approx(idf * tf_part)


def test_field_weights_change_the_ranking():
    index = make_index("python python python", "snakes", titles=("snakes", "python"))
    assert index.search("python")[0].doc_id == "2"  # title match wins at weight 3
    body_only = index.search("python", field_weights={"body": 1.0})
    assert [hit.doc_id for hit in body_only] == ["1"]


def test_bm25_saturates_term_frequency():
    index = make_index("cat dog bird fish", "cat cat cat cat", "tree")
    once = index.search("cat", field_weights={"body": 1.0})
    scores = {hit.doc_id: hit.score for hit in once}
    assert scores["2"] > scores["1"]
    assert scores["2"] < 4 * scores["1"]  # 4x the occurrences, far less than 4x the score


def test_shorter_documents_win_ties_on_frequency():
    index = make_index("search", "search and many other words in a long text")
    assert [hit.doc_id for hit in index.search("search")] == ["1", "2"]


@pytest.mark.parametrize(
    "weights", [{"unknown": 1.0}, {"title": 0.0}, {"title": -1.0}, {"title": math.inf}, {}]
)
def test_invalid_field_weights(weights):
    with pytest.raises(ValueError, match="field"):
        make_index("x").search("x", field_weights=weights)


# ---------- ranking on the sample dataset ----------


@pytest.mark.parametrize("ranker", ["bm25", "tfidf"])
def test_best_match_ranks_first(sample_index, ranker):
    hits = sample_index.search("inverted index", options=SearchOptions(ranker=ranker))
    assert hits[0].title == "Inverted index"


def test_title_boost(sample_index):
    assert sample_index.search("elasticsearch")[0].title == "Elasticsearch"


def test_results_are_sorted_and_limited(sample_index):
    hits = sample_index.search("search", k=3)
    assert len(hits) == 3
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_ties_are_broken_by_id():
    index = make_index("same words", "same words", "same words")
    assert [hit.doc_id for hit in index.search("same")] == ["1", "2", "3"]


def test_and_mode_requires_all_terms(sample_index):
    hits = sample_index.search("ranking lucene", options=SearchOptions(mode="and"))
    assert hits
    for hit in hits:
        assert {"rank", "lucen"} <= {
            term for term, docs in sample_index.postings.items() if hit.doc_id in docs
        }


def test_and_mode_with_an_unknown_term_finds_nothing(sample_index):
    assert sample_index.search("ranking xylophone", options=SearchOptions(mode="and")) == []


def test_no_results(sample_index):
    assert sample_index.search("xylophone") == []
    assert sample_index.search("the and of") == []
    assert sample_index.search("") == []


def test_empty_index_returns_nothing():
    assert InvertedIndex().search("anything") == []


@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1])
def test_k_is_bounded(sample_index, k):
    with pytest.raises(ValueError, match="k must be"):
        sample_index.search("search", k=k)


def test_query_length_is_bounded(sample_index):
    with pytest.raises(ValueError, match="limited"):
        sample_index.search(" ".join(["word"] * (MAX_QUERY_TERMS + 1)))


@pytest.mark.parametrize(
    "options",
    [SearchOptions(ranker="nope"), SearchOptions(mode="xor")],  # type: ignore[arg-type]
)
def test_unknown_options_are_rejected(sample_index, options):
    with pytest.raises(ValueError, match="unknown"):
        sample_index.search("search", options=options)


def test_query_must_be_text(sample_index):
    with pytest.raises(TypeError):
        sample_index.search(42)


# ---------- phrases ----------


def test_phrase_is_stricter_than_loose_search(sample_index):
    loose = {hit.doc_id for hit in sample_index.search("search engine", k=100)}
    exact = {hit.doc_id for hit in sample_index.search('"search engine"', k=100)}
    assert exact
    assert exact < loose


def test_phrase_respects_word_order():
    index = make_index("new york city", "york new city")
    assert [hit.doc_id for hit in index.search('"new york"')] == ["1"]


def test_phrase_does_not_cross_fields():
    index = make_index("beta gamma", titles=("alpha",))
    assert index.search('"alpha beta"') == []


def test_phrase_respects_stopword_gaps():
    index = make_index("inverted the index", "inverted index")
    assert [hit.doc_id for hit in index.search('"inverted index"')] == ["2"]
    assert [hit.doc_id for hit in index.search('"inverted a index"')] == ["1"]


def test_phrase_frequency_counts_every_occurrence():
    index = make_index("new york and new york", "new york")
    assert index.explain('"new york"', "1")[0].tf == 2
    assert index.explain('"new york"', "2")[0].tf == 1


def test_single_word_phrase_is_a_plain_term(sample_index):
    assert sample_index.search('"lucene"') == sample_index.search("lucene")


# ---------- typo tolerance ----------


def test_fuzzy_search_fixes_typos(sample_index):
    hits = sample_index.search("elastisearch", options=SearchOptions(fuzzy=True))
    assert hits[0].title == "Elasticsearch"
    assert sample_index.search("elastisearch") == []


def test_suggest(sample_index):
    assert sample_index.suggest("the elastisearch rankng") == "the elasticsearch ranking"
    assert sample_index.suggest("lucene ranking") is None
    assert sample_index.suggest("") is None


def test_fuzzy_keeps_known_and_hopeless_words(sample_index):
    parsed = sample_index.parse("lucene zzzzzzzzzz", fuzzy=True)
    assert [token.term for token in parsed.tokens] == ["lucen", "zzzzzzzzzz"]


# ---------- explain ----------


@pytest.mark.parametrize("ranker", ["bm25", "tfidf"])
def test_explain_adds_up_to_the_score(sample_index, ranker):
    options = SearchOptions(ranker=ranker)
    for hit in sample_index.search("bm25 ranking lucene", options=options):
        parts = sample_index.explain("bm25 ranking lucene", hit.doc_id, options=options)
        assert sum(part.score for part in parts) == pytest.approx(hit.score)


def test_explain_unknown_document_is_empty(sample_index):
    assert sample_index.explain("ranking", "no-such-doc") == []


# ---------- persistence ----------


def test_save_and_load_round_trip(sample_index, tmp_path):
    path = tmp_path / "index.json"
    sample_index.save(path)
    loaded = InvertedIndex.load(path)
    for query in ["ranking", '"search engine"', "elastisearch"]:
        options = SearchOptions(fuzzy=True)
        assert loaded.search(query, options=options) == sample_index.search(query, options=options)
    assert loaded.stats() == sample_index.stats()


EMPTY_INDEX_FILE = {
    "format": 2,
    "fields": ["body"],
    "docs": {},
    "field_lengths": {},
    "postings": {},
    "spelling": {},
}


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"format": 1}',
        json.dumps({**EMPTY_INDEX_FILE, "docs": []}),
        json.dumps({**EMPTY_INDEX_FILE, "docs": {"1": {}}}),
    ],
)
def test_load_rejects_bad_files(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"index file|corrupt"):
        InvertedIndex.load(path)


def test_stats(sample_index):
    stats = sample_index.stats()
    assert stats["documents"] == 25
    assert stats["most_common_terms"][0] == ("search", 12)


# ---------- properties that must hold for any input ----------


@settings(max_examples=50, deadline=None)
@given(st.text(max_size=60))
def test_search_never_crashes(sample_index, query):
    for options in (SearchOptions(), SearchOptions(fuzzy=True, mode="and")):
        hits = sample_index.search(query, options=options)
        assert all(hit.score > 0 for hit in hits)


def test_empty_documents_do_not_count_towards_statistics():
    with_empty = make_index("cat dog", "", "dog")
    without = make_index("cat dog", "dog")
    assert with_empty.search("cat")[0].score == pytest.approx(without.search("cat")[0].score)
