import pytest

from searchengine import InvertedIndex, analyze, edit_distance, stem
from searchengine.datasets import load_jsonl


@pytest.fixture(scope="module")
def index():
    idx = InvertedIndex()
    idx.add_many(load_jsonl())
    return idx


# ---------- analyzer ----------

def test_analyze_lowercases_and_drops_stopwords():
    assert analyze("The Search and THE Index") == ["search", "index"]


@pytest.mark.parametrize("word,root", [
    ("running", "run"), ("runs", "run"), ("queries", "query"),
    ("indexed", "index"), ("ranking", "rank"), ("cat", "cat"),
])
def test_stem(word, root):
    assert stem(word) == root


# ---------- index statistics ----------

def test_df_and_tf(index):
    assert index.df("bm25") >= 2
    assert index.df("doesnotexist") == 0
    assert index.tf("bm25", "2") > index.tf("bm25", "3")


def test_rare_terms_have_higher_idf(index):
    assert index.idf("pagerank") > index.idf("search")


def test_duplicate_id_rejected():
    idx = InvertedIndex()
    idx.add({"id": "1", "title": "a", "body": "b"})
    with pytest.raises(ValueError):
        idx.add({"id": "1", "title": "c", "body": "d"})


# ---------- ranking ----------

@pytest.mark.parametrize("ranker", ["bm25", "tfidf"])
def test_best_match_ranks_first(index, ranker):
    assert index.search("inverted index", ranker=ranker)[0].title == "Inverted index"


def test_title_boost(index):
    # "Elasticsearch" appears in many bodies but only one title.
    assert index.search("elasticsearch")[0].title == "Elasticsearch"


def test_bm25_saturates_term_frequency():
    idx = InvertedIndex(field_boosts={"title": 1, "body": 1})
    idx.add({"id": "once", "title": "", "body": "cat dog bird fish"})
    idx.add({"id": "many", "title": "", "body": "cat cat cat cat"})
    idx.add({"id": "none", "title": "", "body": "tree"})
    once = idx.score_bm25(["cat"], "once")
    many = idx.score_bm25(["cat"], "many")
    assert many > once
    assert many < 4 * once  # 4x the occurrences, far less than 4x the score


def test_and_mode_requires_all_terms(index):
    hits = index.search("ranking lucene", mode="and")
    terms = analyze("ranking lucene")
    assert hits and all(all(index.tf(t, h.doc_id) for t in terms) for h in hits)


def test_no_results(index):
    assert index.search("xylophone") == []
    assert index.search("the and of") == []  # only stopwords


# ---------- phrase search ----------

def test_phrase_query(index):
    loose = {h.doc_id for h in index.search("search engine", k=100)}
    exact = {h.doc_id for h in index.search('"search engine"', k=100)}
    assert exact and exact < loose  # phrase is stricter


def test_phrase_does_not_span_field_copies():
    idx = InvertedIndex()
    idx.add({"id": "1", "title": "alpha beta", "body": "gamma"})
    assert idx.search('"beta alpha"') == []  # end of one title copy + start of next


# ---------- typo tolerance ----------

def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("same", "same") == 0


def test_fuzzy_search_fixes_typos(index):
    assert index.search("elastisearch", fuzzy=True)[0].title == "Elasticsearch"
    assert index.suggest("levenstein") is not None
    assert index.suggest("lucene") is None


# ---------- persistence ----------

def test_save_and_load(index, tmp_path):
    path = tmp_path / "index.json"
    index.save(path)
    loaded = InvertedIndex.load(path)
    assert [h.doc_id for h in loaded.search("ranking")] == [h.doc_id for h in index.search("ranking")]
