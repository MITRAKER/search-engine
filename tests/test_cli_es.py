"""The command line with --engine es, against a fake Elasticsearch client."""

import sys

import pytest
from elastic_transport import ConnectionError as TransportConnectionError
from elastic_transport import ConnectionTimeout

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
    assert "docker compose up -d --wait" in error  # --wait: until it answers


def test_a_timeout_is_a_one_line_error(capsys):
    client, fake = fake_client(existing=True)
    fake.fail_with = ConnectionTimeout("timed out")
    assert main(["--engine", "es", "x"], es_client=client) == 1
    error = capsys.readouterr().err
    assert error.startswith("searchengine: error: Elasticsearch: ")
    assert error.count("\n") == 1  # one line, no traceback


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


@pytest.mark.parametrize("reindex", [[], ["--reindex"]])
def test_bad_queries_fail_before_the_index_is_touched(capsys, reindex):
    client, fake = fake_client(existing=True)
    assert main(["--engine", "es", *reindex, "--ranker", "tfidf", "x"], es_client=client) == 1
    assert "BM25 only" in capsys.readouterr().err
    assert fake.calls == []  # with --reindex too: the index is still there


def test_reindex_reads_the_data_before_deleting_the_index(capsys, tmp_path):
    client, fake = fake_client(existing=True)
    missing = str(tmp_path / "no-such-file.jsonl")
    assert main(["--engine", "es", "--reindex", "--data", missing, "x"], es_client=client) == 1
    assert "No such file" in capsys.readouterr().err
    assert fake.called("indices.delete") == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "1", "title": "ok"}\nnot json\n', encoding="utf-8")
    assert main(["--engine", "es", "--reindex", "--data", str(bad), "x"], es_client=client) == 1
    assert fake.called("indices.delete") == []


def test_ctrl_c_during_the_fill_deletes_the_half_built_index(monkeypatch):
    client, fake = fake_client()

    def interrupted(**kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(fake, "bulk", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(["--engine", "es", "x"], es_client=client)
    assert fake.called("indices.delete") == [{"index": "searchengine"}]


def test_an_empty_index_name_is_refused_not_replaced(capsys):
    client, fake = fake_client(existing=True)
    assert main(["--engine", "es", "--index", "", "--reindex", "x"], es_client=client) == 1
    assert "invalid index name ''" in capsys.readouterr().err
    assert fake.calls == []  # the default index was not touched


@pytest.mark.parametrize(
    "arguments", [["--category", "web"], ["--index", "notes"], ["--index", ""], ["--category", ""]]
)
def test_elasticsearch_options_need_the_es_engine(capsys, arguments):
    with pytest.raises(SystemExit) as exit_info:
        main([*arguments, "x"])
    assert exit_info.value.code == 2
    assert "need --engine es" in capsys.readouterr().err


def test_a_missing_elasticsearch_package_is_explained(capsys, monkeypatch):
    for module in ("elastic_transport", "elasticsearch", "searchengine.es_engine"):
        monkeypatch.setitem(sys.modules, module, None)  # makes importing it fail
    assert main(["--engine", "es", "x"]) == 1
    assert "pip install 'searchengine-from-scratch[es]'" in capsys.readouterr().err
