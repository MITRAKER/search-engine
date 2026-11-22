"""ElasticsearchEngine against a fake client: what it sends, and how it reads the answers."""

import pytest
from elasticsearch import Elasticsearch, NotFoundError

from searchengine import SearchOptions
from searchengine.es_dsl import (
    ANALYZER,
    MAX_PREFIX_LENGTH,
    index_fingerprint,
    index_mappings,
    index_settings,
)
from searchengine.es_engine import (
    BULK_CHUNK_SIZE,
    MAX_ANALYZE_LENGTH,
    MAX_AUTOCOMPLETE,
    ElasticsearchEngine,
    EsRanking,
    IndexingError,
    default_url,
)
from searchengine.index import MAX_QUERY_LENGTH, MAX_QUERY_TERMS
from tests.fake_es import FakeElasticsearch, fake_client, not_found, search_response


def make_engine(*, existing: bool = False) -> tuple[ElasticsearchEngine, FakeElasticsearch]:
    client, fake = fake_client(existing=existing)
    return ElasticsearchEngine(client, index="test"), fake


# ---------- setting up ----------


def test_default_url_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("ES_URL", raising=False)
    assert default_url() == "http://localhost:9200"
    monkeypatch.setenv("ES_URL", "http://search.internal:9200")
    assert default_url() == "http://search.internal:9200"


def test_without_a_client_one_is_made_for_the_default_url():
    engine = ElasticsearchEngine()  # nothing is sent until the first request
    assert isinstance(engine.client, Elasticsearch)


@pytest.mark.parametrize("index", ["", "Upper", "_starts_with_underscore", "a/b", "x" * 101, 3])
def test_index_names_are_checked(index):
    with pytest.raises(ValueError, match="index name"):
        ElasticsearchEngine(fake_client()[0], index=index)


def test_fields_and_bm25_parameters_are_checked():
    with pytest.raises(ValueError, match="reserved"):
        ElasticsearchEngine(fake_client()[0], fields=["title", "category"])
    with pytest.raises(ValueError, match="k1"):
        ElasticsearchEngine(fake_client()[0], k1=-1)


def test_create_index_sends_our_settings_and_mappings():
    engine, fake = make_engine()
    assert engine.create_index() is True
    [create] = fake.called("indices.create")
    # The _meta says who made the index, and that no load has finished yet.
    meta = {"fingerprint": index_fingerprint(index_settings(), index_mappings()), "complete": False}
    mappings = index_mappings() | {"_meta": meta}
    assert create == {"index": "test", "settings": index_settings(), "mappings": mappings}
    assert engine.fingerprint == meta["fingerprint"]


def test_an_existing_index_is_kept_if_it_was_made_like_ours():
    engine, fake = make_engine(existing=True)
    assert engine.create_index() is False
    assert fake.called("indices.create") == []
    assert fake.called("indices.get_mapping") == [{"index": "test"}]


def test_an_alias_of_one_index_can_be_used():
    engine, fake = make_engine(existing=True)
    fake.concrete = ["test-2026"]  # the alias "test" names this index
    assert engine.create_index() is False


def test_an_alias_of_several_indices_is_refused():
    engine, fake = make_engine(existing=True)
    fake.concrete = ["test-a", "test-b"]
    with pytest.raises(ValueError, match="names 2 indices"):
        engine.create_index()


def made_for(fields: list[str], k1: float = 1.2) -> dict[str, object]:
    """The mappings of an index that an engine with these fields and k1 made."""
    client, _ = fake_client()
    engine = ElasticsearchEngine(client, fields=fields, k1=k1)
    return index_mappings(fields) | {"_meta": {"fingerprint": engine.fingerprint, "complete": True}}


@pytest.mark.parametrize(
    "mappings",
    [
        made_for(["title"]),  # other fields: "body" is not indexed
        made_for(["title", "body"], k1=2.0),  # another k1 when it was made
        index_mappings(),  # made by hand, or by an older version: no fingerprint
        index_mappings() | {"_meta": {"fingerprint": "0" * 64, "complete": True}},
        {},
    ],
)
def test_an_existing_index_made_another_way_is_refused(mappings):
    engine, fake = make_engine(existing=True)
    fake.mappings = mappings
    with pytest.raises(ValueError, match="not made by this engine with these settings and fields"):
        engine.create_index()


@pytest.mark.parametrize("meta", [{"complete": False}, {"complete": "yes"}, {}])
def test_an_existing_index_whose_last_load_did_not_finish_is_refused(meta):
    engine, fake = make_engine(existing=True)
    fake.mappings["_meta"] = {"fingerprint": engine.fingerprint} | meta
    with pytest.raises(ValueError, match="was not filled completely"):
        engine.create_index()


def test_an_existing_index_with_more_shards_is_refused():
    # A split or shrunk copy of our index keeps its mappings, _meta included.
    engine, fake = make_engine(existing=True)
    fake.settings["index.number_of_shards"] = "3"
    with pytest.raises(ValueError, match="has 3 shards, not 1"):
        engine.create_index()


def test_recreate_must_be_true_or_false():
    engine, fake = make_engine(existing=True)
    with pytest.raises(TypeError, match="True or False"):
        engine.create_index(recreate="no")  # type: ignore[arg-type]
    assert fake.calls == []


def test_an_existing_index_with_other_bm25_settings_is_refused():
    engine, fake = make_engine(existing=True)
    fake.settings["index.similarity.default.k1"] = "2.0"
    with pytest.raises(ValueError, match="recreate"):
        engine.create_index()
    fake.settings = {}  # made without our similarity at all
    with pytest.raises(ValueError, match="recreate"):
        engine.create_index()


def test_recreate_deletes_the_old_index_first():
    engine, fake = make_engine(existing=True)
    assert engine.create_index(recreate=True) is True
    assert [name for name, _ in fake.calls] == [
        "indices.exists",
        "indices.delete",
        "indices.create",
    ]


def test_delete_index_ignores_a_missing_index():
    engine, fake = make_engine()
    engine.delete_index()
    assert fake.called("options") == [{"ignore_status": 404}]
    assert fake.called("indices.delete") == [{"index": "test"}]


# ---------- indexing ----------


def docs(count: int) -> list[dict[str, object]]:
    return [{"id": n, "title": f"title {n}", "body": "text", "url": "u"} for n in range(count)]


def test_a_load_marks_the_index_unfinished_until_it_ends():
    engine, fake = make_engine(existing=True)
    engine.index_documents(docs(3))
    assert [name for name, _ in fake.calls] == [
        "indices.exists",
        "indices.put_mapping",
        "bulk",
        "indices.refresh",
        "indices.put_mapping",
    ]
    marks = [call["meta"] for call in fake.called("indices.put_mapping")]
    assert marks == [
        {"fingerprint": engine.fingerprint, "complete": False},
        {"fingerprint": engine.fingerprint, "complete": True},
    ]


def test_a_failed_load_leaves_the_index_marked_unfinished():
    engine, fake = make_engine(existing=True)
    fake.bulk_errors = {"1": "mapper_parsing_exception"}
    with pytest.raises(IndexingError):
        engine.index_documents(docs(3))
    assert fake.mappings["_meta"]["complete"] is False
    with pytest.raises(ValueError, match="was not filled completely"):
        engine.create_index()  # so it is not reused


def test_documents_are_sent_in_chunks_with_create_actions():
    engine, fake = make_engine(existing=True)
    assert engine.index_documents(docs(BULK_CHUNK_SIZE * 2 + 1)) == BULK_CHUNK_SIZE * 2 + 1
    sizes = [len(call["operations"]) // 2 for call in fake.called("bulk")]
    assert sizes == [BULK_CHUNK_SIZE, BULK_CHUNK_SIZE, 1]
    action, source = fake.called("bulk")[0]["operations"][:2]
    assert action == {"create": {"_index": "test", "_id": "0"}}  # ids become strings
    # The id stays in the document too: the index sorts equal scores by it.
    assert source == {"id": "0", "title": "title 0", "body": "text", "url": "u"}
    assert fake.called("indices.refresh") == [{"index": "test"}]


def test_large_documents_are_sent_in_smaller_chunks(monkeypatch):
    # Elasticsearch refuses requests over 100 MB, whatever the document count.
    monkeypatch.setattr("searchengine.es_engine.BULK_CHUNK_BYTES", 1000)
    engine, fake = make_engine(existing=True)
    big = [{"id": n, "body": "x" * 400} for n in range(5)]
    assert engine.index_documents(big) == 5
    sizes = [len(call["operations"]) // 2 for call in fake.called("bulk")]
    assert sizes == [3, 2]  # flushed once 1000 bytes are reached


def test_documents_need_an_existing_index():
    # Otherwise Elasticsearch would create one, with its own analyzer.
    engine, fake = make_engine()
    with pytest.raises(ValueError, match="does not exist"):
        engine.index_documents(docs(1))
    assert fake.called("bulk") == []


def test_documents_are_checked_like_the_from_scratch_engine():
    engine, fake = make_engine(existing=True)
    with pytest.raises(TypeError, match="must be a string"):
        engine.index_documents([{"id": "1", "title": 42}])
    with pytest.raises(ValueError, match="reserved"):
        engine.index_documents([{"id": "1", "spelling": "x"}])
    assert fake.called("bulk") == []


def test_check_documents_returns_them_as_they_would_be_sent():
    engine, fake = make_engine()
    checked = engine.check_documents([{"id": 7, "title": "a", "url": "u"}, {"id": "8"}])
    assert checked == [{"id": "7", "title": "a", "url": "u"}, {"id": "8"}]
    assert fake.calls == []  # nothing is sent


@pytest.mark.parametrize(
    ("documents", "error", "message"),
    [
        ([{"id": "1"}, {"id": 1}], ValueError, "duplicate document id: 1"),
        ([{"id": "1", "spelling": "x"}], ValueError, "reserved"),
        ([{"id": "1", "title": 42}], TypeError, "must be a string"),
        ([{"title": "no id"}], TypeError, "document id"),
    ],
)
def test_check_documents_refuses_what_a_load_would_refuse(documents, error, message):
    engine, fake = make_engine()
    with pytest.raises(error, match=message):
        engine.check_documents(documents)
    assert fake.calls == []


def test_rejected_documents_raise_with_the_first_reason():
    engine, fake = make_engine(existing=True)
    fake.bulk_errors = {"1": "version conflict, document already exists"}
    with pytest.raises(IndexingError, match="1 of 3 documents failed; first: id 1: version"):
        engine.index_documents(docs(3))


def test_count():
    engine, _ = make_engine()
    assert engine.count() == 25


# ---------- searching ----------

HIT = {
    "_id": "7",
    "_score": 2.5,
    "_source": {"title": "Fuzzy search"},
    "highlight": {"body": ["a <mark>fuzzy</mark> query"]},
}


def test_search_reads_hits_facets_and_totals():
    engine, fake = make_engine()
    fake.search_responses = [search_response([HIT], {"ranking": 4, "web": 1}, total=5)]
    result = engine.search("fuzzy", k=1, category="ranking")
    [hit] = result.hits
    assert (hit.doc_id, hit.score, hit.title) == ("7", 2.5, "Fuzzy search")
    assert hit.highlights == {"body": ["a <mark>fuzzy</mark> query"]}
    assert (result.total, result.facets, result.took_ms) == (5, {"ranking": 4, "web": 1}, 3)
    [request] = fake.called("search")
    assert request["index"] == "test"
    assert request["query"]["combined_fields"]["fields"] == ["title^3.0", "body^1.0"]
    assert request["post_filter"] == {"term": {"category": "ranking"}}


def test_rank_returns_only_ids_scores_and_elasticsearchs_own_time():
    engine, fake = make_engine()
    hits: list[dict[str, object]] = [{"_id": "7", "_score": 2.5}, {"_id": 12, "_score": 1.25}]
    fake.search_responses = [search_response(hits)]
    assert engine.rank("fuzzy", k=2) == EsRanking(["7", "12"], [2.5, 1.25], took_ms=3)
    [request] = fake.called("search")
    assert (request["index"], request["size"], request["source"]) == ("test", 2, False)
    assert request["sort"] == [{"_score": "desc"}, {"id": "asc"}]
    assert "combined_fields" in request["query"]  # BM25F, like search()
    assert "highlight" not in request
    assert "aggs" not in request


def test_rank_can_score_each_field_on_its_own():
    engine, fake = make_engine()
    fake.search_responses = [search_response()]
    engine.rank("fuzzy", field_weights={"body": 2.0}, query_type="best_fields")
    query = fake.called("search")[0]["query"]["multi_match"]
    assert (query["type"], query["fields"]) == ("best_fields", ["body^2.0"])


def test_rank_checks_its_arguments_like_search():
    engine, fake = make_engine()
    with pytest.raises(ValueError, match="k1 and b are settings of the Elasticsearch index"):
        engine.rank("x", options=SearchOptions(k1=2.0))
    with pytest.raises(ValueError, match="k must be"):
        engine.rank("x", k=0)
    assert fake.calls == []


def test_search_only_searches_weighted_fields():
    engine, fake = make_engine()
    fake.search_responses = [search_response()]
    assert engine.search("x", field_weights={"body": 2.0}).hits == []
    [request] = fake.called("search")
    assert request["query"]["combined_fields"]["fields"] == ["body^2.0"]
    assert set(request["highlight"]["fields"]) == {"body"}


def test_k1_and_b_are_settings_of_the_index():
    engine, _ = make_engine()
    with pytest.raises(ValueError, match="settings of the Elasticsearch index"):
        engine.search("x", options=SearchOptions(k1=2.0))


def test_search_body_checks_without_sending_anything():
    engine, fake = make_engine()
    body = engine.search_body("inverted index", k=3, category="indexing")
    assert body["size"] == 3
    assert body["post_filter"] == {"term": {"category": "indexing"}}
    with pytest.raises(ValueError, match="BM25 only"):
        engine.search_body("x", options=SearchOptions(ranker="tfidf"))
    assert fake.calls == []


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"options": {"mode": "and"}}, TypeError),
        ({"options": SearchOptions(ranker="tfidf")}, ValueError),
        ({"field_weights": {"title": 0.5}}, ValueError),
        ({"field_weights": {"category": 1.0}}, ValueError),
        ({"k": 0}, ValueError),
    ],
)
def test_bad_search_arguments_fail_before_any_request(kwargs, error):
    engine, fake = make_engine()
    with pytest.raises(error):
        engine.search("x", **kwargs)
    assert fake.called("search") == []


def test_explain_returns_elasticsearchs_breakdown():
    engine, fake = make_engine()
    fake.explain_result = {"matched": True, "explanation": {"value": 1.0, "description": "sum"}}
    assert engine.explain("x", 7)["matched"] is True
    [request] = fake.called("explain")
    assert request["id"] == "7"


def test_explain_of_a_missing_document_says_not_matched():
    engine, fake = make_engine()
    fake.explain_result = not_found({"_index": "test", "_id": "99", "matched": False})
    assert engine.explain("x", "99")["matched"] is False


def test_explain_of_a_missing_index_raises():
    engine, fake = make_engine()
    fake.explain_result = not_found({"error": {"type": "index_not_found_exception"}})
    with pytest.raises(NotFoundError):
        engine.explain("x", "1")


# ---------- did you mean, autocomplete, analyze ----------


def test_suggest_asks_only_about_words_that_may_be_misspelled():
    engine, fake = make_engine()
    fake.search_responses = [
        {
            "suggest": {"word1": [{"options": [{"text": "elasticsearch", "score": 0.8}]}]},
            "aggregations": {"known": {"buckets": {"word1": {"doc_count": 0}}}},
        }
    ]
    assert engine.suggest("the elastisearch") == "the elasticsearch"
    [request] = fake.called("search")
    assert request["size"] == 0
    assert set(request["suggest"]) == {"word1"}
    assert set(request["aggs"]["known"]["filters"]["filters"]) == {"word1"}
    assert engine.suggest("an of") is None  # nothing to fix: no request at all
    assert len(fake.called("search")) == 1


def test_suggest_keeps_words_that_match_documents():
    # "analyzer" is not in the (unstemmed) spelling field, but its stem
    # matches documents, so it is not a typo: our engine keeps it too.
    engine, fake = make_engine()
    fake.search_responses = [
        {
            "suggest": {"word0": [{"options": [{"text": "analyzers"}]}]},
            "aggregations": {"known": {"buckets": {"word0": {"doc_count": 3}}}},
        }
    ]
    assert engine.suggest("analyzer") is None


@pytest.mark.parametrize(
    ("query", "error"),
    [
        (" ".join(["elastisearch"] * (MAX_QUERY_TERMS + 1)), ValueError),
        ("a" * (MAX_QUERY_LENGTH + 1), ValueError),
        (None, TypeError),
    ],
)
def test_suggest_is_bounded_like_search(query, error):
    engine, fake = make_engine()
    with pytest.raises(error):
        engine.suggest(query)
    assert fake.called("search") == []


def test_autocomplete_returns_distinct_titles():
    engine, fake = make_engine()
    titles = ["Inverted index", "Inverted index", "Index sharding", ""]
    hits = [{"_source": {"title": title}} for title in titles]
    fake.search_responses = [search_response(hits)]
    assert engine.autocomplete("ind", k=4) == ["Inverted index", "Index sharding"]
    [request] = fake.called("search")
    assert request["size"] == 4
    assert request["source_includes"] == ["title"]


@pytest.mark.parametrize(
    ("prefix", "k", "error"),
    [
        (42, 5, TypeError),
        ("x" * (MAX_PREFIX_LENGTH + 1), 5, ValueError),
        ("x", 0, ValueError),
        ("x", MAX_AUTOCOMPLETE + 1, ValueError),
        ("x", True, ValueError),
    ],
)
def test_autocomplete_checks_its_arguments(prefix, k, error):
    engine, _ = make_engine()
    with pytest.raises(error):
        engine.autocomplete(prefix, k=k)


def test_autocomplete_needs_a_title_and_a_prefix():
    engine, fake = make_engine()
    assert engine.autocomplete("   ") == []
    no_title = ElasticsearchEngine(fake_client()[0], fields=["body"])
    assert no_title.autocomplete("ind") == []
    assert fake.called("search") == []


def test_analyze_uses_our_analyzer():
    engine, fake = make_engine()
    fake.tokens = [{"token": "run", "position": 3}]
    assert engine.analyze("The Runners are RUNNING") == [("run", 3)]
    [request] = fake.called("indices.analyze")
    assert request["analyzer"] == ANALYZER


def test_analyze_checks_its_input():
    engine, _ = make_engine()
    with pytest.raises(TypeError):
        engine.analyze(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="longer than"):
        engine.analyze("x" * (MAX_ANALYZE_LENGTH + 1))


def test_no_documents_means_no_bulk_request():
    engine, fake = make_engine(existing=True)
    assert engine.index_documents([]) == 0
    assert fake.called("bulk") == []
    assert fake.called("indices.refresh") == [{"index": "test"}]
