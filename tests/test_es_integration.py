"""Both engines side by side, on a real Elasticsearch.

These tests need a running Elasticsearch (``docker compose up -d --wait``) at
$ES_URL, default http://localhost:9200. Without one they are skipped, unless
REQUIRE_ES=1 is set, as in CI, where a missing server is a failure.

Run only these:  pytest -m integration
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch

from searchengine import Hit, InvertedIndex, SearchOptions, analyze_tokens
from searchengine.datasets import load_jsonl
from searchengine.es_engine import ElasticsearchEngine, EsHit, IndexingError, default_url

pytestmark = pytest.mark.integration

GOLDEN = json.loads((Path(__file__).parent / "data" / "es_parity.json").read_text("utf-8"))
QUERIES = [
    "inverted index",
    "bm25 ranking function",
    "search engine relevance",
    "elasticsearch lucene shard",
    "the of and",
]
WEIGHTINGS = [None, {"body": 1.0}, {"title": 1.0}, {"title": 1.7, "body": 1.3}]
PHRASES = ['"inverted index"', '"search engine"', '"ranking function"']
WORDS = ["ranking", "search", "index", "query", "score", "document", "elasticsearch", "lucene"]
# Correctly spelled words whose stems are in the index, a typo, and a fine query.
SUGGEST = ["analyzer", "ranked", "indexed", "tokenizes", "lucen", "elastisearch rankng", "bm25"]
TOLERANCE = 1e-6  # Lucene computes in 32-bit floats


def _reachable(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    try:
        with urllib.request.urlopen(url, timeout=5):  # the scheme is checked above
            return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(scope="module")
def client() -> Iterator[Elasticsearch]:
    url = default_url()
    if not _reachable(url):
        message = f"no Elasticsearch at {url} (start it: docker compose up -d --wait)"
        if os.environ.get("REQUIRE_ES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    es = Elasticsearch(url)
    yield es
    es.close()


def _engine(client: Elasticsearch, name: str) -> ElasticsearchEngine:
    engine = ElasticsearchEngine(client, index=f"test-{name}-{os.getpid()}")
    engine.create_index(recreate=True)
    return engine


@pytest.fixture(scope="module")
def engines(client: Elasticsearch) -> Iterator[tuple[InvertedIndex, ElasticsearchEngine]]:
    scratch = InvertedIndex()
    scratch.add_many(load_jsonl())
    es = _engine(client, "sample")
    try:
        es.index_documents(load_jsonl())
        yield scratch, es
    finally:
        es.delete_index()


@pytest.fixture
def scratch_index(client: Elasticsearch) -> Iterator[ElasticsearchEngine]:
    """An empty index for one test, deleted afterwards."""
    engine = _engine(client, "scratch")
    try:
        yield engine
    finally:
        engine.delete_index()


def compare(scratch_hits: Sequence[Hit], es_hits: Sequence[EsHit]) -> None:
    """The same documents, with the same scores."""
    ours = {hit.doc_id: hit.score for hit in scratch_hits}
    theirs = {hit.doc_id: hit.score for hit in es_hits}
    assert ours.keys() == theirs.keys()
    for doc_id, score in theirs.items():
        assert ours[doc_id] == pytest.approx(score, rel=TOLERANCE), doc_id


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("weights", WEIGHTINGS)
@pytest.mark.parametrize("mode", ["or", "and"])
def test_same_scores_for_words(engines, query, weights, mode):
    scratch, es = engines
    options = SearchOptions(mode=mode)
    ours = scratch.search(query, k=25, options=options, field_weights=weights)
    theirs = es.search(query, k=25, options=options, field_weights=weights).hits
    compare(ours, theirs)


@pytest.mark.parametrize("query", PHRASES)
@pytest.mark.parametrize("field", ["title", "body"])
def test_same_scores_for_phrases_in_one_field(engines, query, field):
    scratch, es = engines
    weights = {field: 1.0}
    compare(
        scratch.search(query, k=25, field_weights=weights),
        es.search(query, k=25, field_weights=weights).hits,
    )


@pytest.mark.parametrize("query", PHRASES)
@pytest.mark.parametrize("weights", [None, {"body": 2.0}])
def test_weighted_phrases_find_the_same_documents(engines, query, weights):
    """Scores differ here, by design: Elasticsearch has no BM25F for phrases."""
    scratch, es = engines
    ours = {hit.doc_id for hit in scratch.search(query, k=25, field_weights=weights)}
    theirs = {hit.doc_id for hit in es.search(query, k=25, field_weights=weights).hits}
    assert ours == theirs


def test_elasticsearch_multiplies_a_phrase_score_by_the_weight(engines):
    """Our engine treats a weight as "each occurrence counts w times" (BM25F),
    for phrases too. Elasticsearch's phrase query multiplies the score instead,
    and over several fields it keeps the best field's score."""
    _, es = engines
    [once] = es.search('"ranking function"', k=1, field_weights={"body": 1.0}).hits
    [twice] = es.search('"ranking function"', k=1, field_weights={"body": 2.0}).hits
    assert twice.score == pytest.approx(2 * once.score, rel=TOLERANCE)


@pytest.mark.parametrize(("text", "terms"), GOLDEN["analyzer"])
def test_same_terms(engines, text, terms):
    _, es = engines
    expected = [(token.term, token.position) for token in analyze_tokens(text)]
    assert es.analyze(text) == expected == [tuple(term) for term in terms]


def test_explain_adds_up_to_the_score(engines):
    _, es = engines
    [hit] = es.search("inverted index", k=1).hits
    explanation = es.explain("inverted index", hit.doc_id)
    assert explanation["matched"] is True
    assert explanation["explanation"]["value"] == pytest.approx(hit.score)
    assert es.explain("inverted index", "no-such-id")["matched"] is False


def test_facets_count_every_category_and_the_filter_applies_after(engines):
    _, es = engines
    everything = es.search("search", k=25)
    ranking = es.search("search", k=25, category="ranking")
    assert ranking.facets == everything.facets  # post_filter: counted before filtering
    assert ranking.total == everything.facets["ranking"]
    assert {hit.doc_id for hit in ranking.hits} < {hit.doc_id for hit in everything.hits}


@pytest.mark.parametrize("query", WORDS)
def test_ties_are_broken_by_id_in_both_engines(engines, query):
    """Where k cuts through equal scores, both engines keep the same documents."""
    scratch, es = engines
    for k in range(1, 11):
        ours = [hit.doc_id for hit in scratch.search(query, k=k)]
        assert [hit.doc_id for hit in es.search(query, k=k).hits] == ours, k


def test_did_you_mean_and_autocomplete(engines):
    _, es = engines
    assert es.suggest("elastisearch rankng") == "elasticsearch ranking"
    assert es.suggest("inverted index") is None
    assert "Inverted index" in es.autocomplete("inverted ind")


@pytest.mark.parametrize("query", SUGGEST)
def test_did_you_mean_agrees_with_our_engine(engines, query):
    scratch, es = engines
    assert es.suggest(query) == scratch.suggest(query)


def test_autocomplete_needs_every_typed_word(engines):
    _, es = engines
    assert es.autocomplete("inverted xyz") == []
    assert es.autocomplete("zebra ind") == []


def test_fuzzy_search_finds_the_misspelled_words(engines):
    _, es = engines
    hits = es.search("elastisearch", k=3, options=SearchOptions(fuzzy=True)).hits
    assert hits
    assert any("Elasticsearch" in hit.title for hit in hits)


def test_count_and_existing_index(engines, client):
    _, es = engines
    assert es.count() == 25
    assert ElasticsearchEngine(client, index=es.index).create_index() is False
    with pytest.raises(ValueError, match="recreate"):
        ElasticsearchEngine(client, index=es.index, k1=2.0).create_index()


def test_an_index_made_for_other_fields_is_refused(engines, client):
    _, es = engines
    other = ElasticsearchEngine(client, index=es.index, fields=["title", "body", "abstract"])
    with pytest.raises(ValueError, match="maps 'abstract' differently"):
        other.create_index()


def test_an_alias_of_the_index_can_be_used(engines, client):
    _, es = engines
    alias = f"{es.index}-alias"
    client.indices.put_alias(index=es.index, name=alias)
    try:
        engine = ElasticsearchEngine(client, index=alias)
        assert engine.create_index() is False
        assert engine.search("inverted index", k=3).hits
    finally:
        client.indices.delete_alias(index=es.index, name=alias)


def test_documents_need_an_existing_index(client):
    engine = ElasticsearchEngine(client, index=f"test-missing-{os.getpid()}")
    engine.delete_index()
    with pytest.raises(ValueError, match="does not exist"):
        engine.index_documents([{"id": "1", "body": "x"}])
    assert not client.indices.exists(index=engine.index)


def test_a_field_longer_than_the_highlight_limit_is_still_searched(scratch_index):
    """Elasticsearch highlights at most 1,000,000 characters of a field."""
    docs = [
        {"id": "1", "title": "long", "body": "search " + "lorem ipsum " * 90_000},
        {"id": "2", "title": "short", "body": "search me"},
    ]
    scratch_index.index_documents(docs)
    ours = InvertedIndex()
    ours.add_many(docs)
    result = scratch_index.search("search")
    compare(ours.search("search"), result.hits)
    assert all(hit.highlights for hit in result.hits)


def test_snippets_are_escaped_html(scratch_index):
    scratch_index.index_documents(
        [{"id": "1", "title": "XSS", "body": "a <script>alert(1)</script> search"}]
    )
    [hit] = scratch_index.search("search").hits
    [snippet] = hit.highlights["body"]
    assert "<script>" not in snippet
    assert "&lt;script&gt;" in snippet
    assert "<mark>search</mark>" in snippet


def test_duplicate_ids_are_errors_not_overwrites(scratch_index):
    scratch_index.index_documents([{"id": "1", "title": "first", "body": "x"}])
    with pytest.raises(IndexingError, match="1 of 1 documents failed"):
        scratch_index.index_documents([{"id": "1", "title": "second", "body": "x"}])
    assert scratch_index.search("first").total == 1
