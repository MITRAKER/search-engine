"""The command line with --engine es, against a fake Elasticsearch client."""

import sys

import pytest
from elastic_transport import ConnectionError as TransportConnectionError

from searchengine.cli import main
from tests.fake_es import fake_client, not_found, search_response

HIT = {
    "_id": "1",
    "_score": 1.234,
    "_source": {"title": "Inverted index"},
    "highlight": {"body": ["An <mark>inverted</mark> &lt;index&gt;"]},
}
EXPLANATION = {
    "matched": True,
    "explanation": {"value": 1.234, "description": "sum of:", "details": []},
}


def test_es_engine_fills_a_new_index_and_prints_results(capsys):
    client, fake = fake_client()
    fake.search_responses = [search_response([HIT], {"indexing": 1}), {"suggest": {}}]
    assert main(["--engine", "es", "inverted index"], es_client=client) == 0
    assert len(fake.called("bulk")[0]["operations"]) == 2 * 25  # the sample data
    out = capsys.readouterr().out
    assert "1. [ 1.234] Inverted index  (id=1)" in out
    assert "body: An [inverted] <index>" in out  # markup for the terminal, HTML unescaped
    assert "1 matches in 3 ms. Categories: indexing (1)" in out


def test_es_engine_reuses_an_existing_index_and_can_explain(capsys):
    client, fake = fake_client(existing=True)
    fake.search_responses = [search_response([HIT]), {"suggest": {}}]
    fake.explain_result = EXPLANATION
    assert main(["--engine", "es", "--explain", "inverted index"], es_client=client) == 0
    assert fake.called("bulk") == []
    out = capsys.readouterr().out
    assert "1.234  sum of:" in out
    assert "Categories: none" in out


def test_es_engine_suggests_and_reports_no_results(capsys):
    client, fake = fake_client(existing=True)
    suggestion = {"suggest": {"word0": [{"options": [{"text": "elasticsearch"}]}]}}
    fake.search_responses = [search_response(), suggestion]
    assert main(["--engine", "es", "elastisearch"], es_client=client) == 0
    out = capsys.readouterr().out
    assert "Did you mean: elasticsearch?" in out
    assert "No results." in out


def test_unreachable_elasticsearch_gets_a_hint(capsys, monkeypatch):
    monkeypatch.setenv("ES_URL", "http://localhost:1")
    client, fake = fake_client()
    fake.fail_with = TransportConnectionError("refused")
    assert main(["--engine", "es", "x"], es_client=client) == 1
    error = capsys.readouterr().err
    assert "cannot reach Elasticsearch at http://localhost:1" in error
    assert "docker compose up -d" in error


def test_elasticsearch_errors_are_one_line(capsys):
    client, fake = fake_client(existing=True)
    fake.fail_with = not_found({"error": "no such index"})
    assert main(["--engine", "es", "x"], es_client=client) == 1
    assert capsys.readouterr().err.startswith("searchengine: error: Elasticsearch: ")


def test_a_failed_fill_deletes_the_half_built_index(capsys):
    client, fake = fake_client()
    fake.bulk_errors = {"3": "mapper_parsing_exception"}
    assert main(["--engine", "es", "x"], es_client=client) == 1
    assert fake.called("indices.delete") == [{"index": "searchengine"}]
    assert "documents failed" in capsys.readouterr().err


def test_bad_queries_for_elasticsearch_are_one_line_errors(capsys):
    client, fake = fake_client(existing=True)
    assert main(["--engine", "es", "--ranker", "tfidf", "x"], es_client=client) == 1
    assert "BM25 only" in capsys.readouterr().err
    assert fake.called("search") == []


def test_elasticsearch_options_need_the_es_engine(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--category", "web", "x"])
    assert exit_info.value.code == 2
    assert "need --engine es" in capsys.readouterr().err


def test_a_missing_elasticsearch_package_is_explained(capsys, monkeypatch):
    for module in ("elastic_transport", "elasticsearch", "searchengine.es_engine"):
        monkeypatch.setitem(sys.modules, module, None)  # makes importing it fail
    assert main(["--engine", "es", "x"]) == 1
    assert "pip install 'searchengine-from-scratch[es]'" in capsys.readouterr().err
