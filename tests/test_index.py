import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from searchengine import InvertedIndex, SearchOptions
from searchengine.index import (
    MAX_FIELD_WEIGHT,
    MAX_FIELDS,
    MAX_QUERY_TERMS,
    MAX_RESULTS,
    MIN_FIELD_WEIGHT,
)
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
        (["not", "a", "mapping"], TypeError),
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
    with pytest.raises(ValueError, match="fields"):
        InvertedIndex(fields=[f"f{i}" for i in range(MAX_FIELDS + 1)])


@pytest.mark.parametrize("fields", ["body", ["title", 3], ["title", ""]])
def test_fields_must_be_a_list_of_names(fields):
    """A bare string would otherwise become one field per letter."""
    with pytest.raises(TypeError):
        InvertedIndex(fields=fields)


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
    "weights",
    [
        {"unknown": 1.0},
        {},
        {"title": 0.0},
        {"title": -1.0},
        {"title": 0.5},  # Elasticsearch's combined_fields refuses weights under 1
        {"title": MAX_FIELD_WEIGHT * 2},
        {"title": math.inf},
        {"title": math.nan},
    ],
)
def test_invalid_field_weights(weights):
    with pytest.raises(ValueError, match="field"):
        make_index("x").search("x", field_weights=weights)


def test_field_weights_at_the_limits_work():
    index = make_index("x", titles=("x",))
    weights = {"title": MIN_FIELD_WEIGHT, "body": MAX_FIELD_WEIGHT}
    assert [hit.doc_id for hit in index.search("x", field_weights=weights)] == ["1"]


@pytest.mark.parametrize("weights", [{"title": True}, {"title": "3"}, [("title", 3.0)]])
def test_field_weights_must_be_numbers_in_a_mapping(weights):
    with pytest.raises(TypeError):
        make_index("x").search("x", field_weights=weights)


# ---------- the statistics Lucene's combined_fields uses ----------


def test_document_frequency_is_the_largest_per_field_one():
    """The word "apache" is in one title and in another document's body.
    Counting documents with it anywhere would give df = 2; Lucene takes the
    larger per-field count, 1, so the term is rarer and scores higher."""
    index = make_index("apache", "web", "cats", titles=("server", "apache", "dogs"))
    weights = {"title": 1.0, "body": 1.0}
    [part] = index.explain("apache", "1", field_weights=weights)
    assert part.idf == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))


def test_n_is_the_largest_per_field_document_count():
    """Two of three titles are empty: N is max(1, 3) = 3, not 1 or 4."""
    index = make_index("cat", "dog", "bird", titles=("cat",))
    [part] = index.explain("bird", "3")
    assert part.idf == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))


def test_long_documents_are_scored_with_lucene_lengths():
    """A 41-term body is stored by Lucene as 40 terms long."""
    long_body = " ".join(["cat"] + ["dog"] * 40)
    index = make_index(long_body, "cat dog", titles=("", ""))
    avgdl = (41 + 2) / 2  # the average uses the exact total length

    def expected(dl: float) -> float:
        idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
        return idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * dl / avgdl))

    lucene = {hit.doc_id: hit.score for hit in index.search("cat")}
    exact = {
        hit.doc_id: hit.score
        for hit in index.search("cat", options=SearchOptions(lucene_norms=False))
    }
    assert lucene["1"] == pytest.approx(expected(40))
    assert exact["1"] == pytest.approx(expected(41))
    assert lucene["2"] == exact["2"] == pytest.approx(expected(2))


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


@pytest.mark.parametrize("k", [True, "5", 2.5])
def test_k_must_be_an_int(sample_index, k):
    with pytest.raises(TypeError, match="k must be"):
        sample_index.search("search", k=k)


@pytest.mark.parametrize(
    "options",
    [SearchOptions(k1=math.inf), SearchOptions(k1=-1), SearchOptions(b=1.5), SearchOptions(b=-0.1)],
)
def test_bm25_parameters_are_checked_before_scoring(options):
    """Checked up front, even when nothing matches (k1 = inf would give NaN)."""
    with pytest.raises(ValueError, match="k1"):
        make_index("cat").search("dog", options=options)


def test_options_must_be_search_options(sample_index):
    with pytest.raises(TypeError, match="SearchOptions"):
        sample_index.search("search", options={"mode": "and"})


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


def test_fuzzy_corrects_to_words_in_the_searched_fields():
    """The word "lucene" is only in a title. Searching the body alone, the
    closest body word, "lucent", is the useful correction."""
    index = make_index("apples", "lucent", titles=("lucene", ""))
    options = SearchOptions(fuzzy=True)
    assert [hit.doc_id for hit in index.search("lucene", options=options)] == ["1"]
    body_only = index.search("lucene", options=options, field_weights={"body": 1.0})
    assert [hit.doc_id for hit in body_only] == ["2"]


def test_suggest_is_bounded_like_search(sample_index):
    with pytest.raises(ValueError, match="limited"):
        sample_index.suggest(" ".join(["elastisearch"] * (MAX_QUERY_TERMS + 1)))
    assert sample_index.suggest(" ".join(["the"] * (MAX_QUERY_TERMS + 1))) is None


# ---------- explain ----------


@pytest.mark.parametrize("ranker", ["bm25", "tfidf"])
def test_explain_adds_up_to_the_score(sample_index, ranker):
    options = SearchOptions(ranker=ranker)
    for hit in sample_index.search("bm25 ranking lucene", options=options):
        parts = sample_index.explain("bm25 ranking lucene", hit.doc_id, options=options)
        assert sum(part.score for part in parts) == pytest.approx(hit.score)


def test_explain_unknown_document_is_empty(sample_index):
    assert sample_index.explain("ranking", "no-such-doc") == []


def test_explain_follows_and_mode():
    index = make_index("cat dog", "cat")
    options = SearchOptions(mode="and")
    [hit] = index.search("cat dog", options=options)
    parts = index.explain("cat dog", hit.doc_id, options=options)
    assert sum(part.score for part in parts) == pytest.approx(hit.score)
    assert index.explain("cat dog", "2", options=options) == []  # search skips it too


def test_explain_accepts_int_ids_like_add():
    index = InvertedIndex()
    index.add({"id": 7, "body": "cat"})
    assert index.explain("cat", 7) == index.explain("cat", "7") != []


# ---------- persistence ----------


def test_save_and_load_round_trip(sample_index, tmp_path):
    path = tmp_path / "index.json"
    sample_index.save(path)
    loaded = InvertedIndex.load(path)
    for query in ["ranking", '"search engine"', "elastisearch"]:
        options = SearchOptions(fuzzy=True)
        assert loaded.search(query, options=options) == sample_index.search(query, options=options)
    assert loaded.stats() == sample_index.stats()
    assert loaded.explain("ranking", "6") == sample_index.explain("ranking", "6")


EMPTY_INDEX_FILE = {
    "format": 2,
    "fields": ["body"],
    "docs": {},
    "field_lengths": {},
    "postings": {},
    "spelling": {},
}


ONE_DOC = {"1": {"id": "1", "body": "cat"}}
ONE_LENGTH = {"1": {"body": 1}}
ONE_POSTING = {"cat": {"1": {"body": [0]}}}


def index_file(**changes: object) -> str:
    good = {
        **EMPTY_INDEX_FILE,
        "docs": ONE_DOC,
        "field_lengths": ONE_LENGTH,
        "postings": ONE_POSTING,
    }
    return json.dumps({**good, **changes})


def test_index_file_helper_makes_a_valid_file(tmp_path):
    path = tmp_path / "good.json"
    path.write_text(index_file())
    assert [hit.doc_id for hit in InvertedIndex.load(path).search("cat")] == ["1"]


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"format": 1}',
        json.dumps({**EMPTY_INDEX_FILE, "docs": []}),
        json.dumps({**EMPTY_INDEX_FILE, "docs": {"1": {}}}),
        index_file(fields=["body", "body"]),
        index_file(fields=[]),
        index_file(docs={"1": "cat"}),
        index_file(docs={"1": {"id": "2", "body": "cat"}}),
        index_file(docs={"1": {"id": "1", "body": 5}}),
        index_file(field_lengths={"1": {}}),
        index_file(field_lengths={"1": {"body": -1}}),
        index_file(field_lengths={"1": {"body": "1"}}),
        index_file(postings={"cat": {}}),
        index_file(postings={"cat": {"2": {"body": [0]}}}),
        index_file(postings={"cat": {"1": {"title": [0]}}}),
        index_file(postings={"cat": {"1": {"body": 0}}}),
        index_file(postings={"cat": {"1": {"body": [-1]}}}),
        index_file(postings={"cat": {"1": {"body": [True]}}}),
        # Numbers no text of MAX_TEXT_LENGTH characters can have would
        # overflow later, in scoring.
        index_file(field_lengths={"1": {"body": 10**39}}),
        index_file(postings={"cat": {"1": {"body": [10**12]}}}),
        # A file must agree with itself: a field's length is its number of
        # positions, and positions only go up.
        index_file(field_lengths={"1": {"body": 0}}),
        index_file(field_lengths={"1": {"body": 2}}),
        index_file(field_lengths={"1": {"body": 2}}, postings={"cat": {"1": {"body": [3, 1]}}}),
        "[" * 100_000 + "]" * 100_000,  # too deep for the json module
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
