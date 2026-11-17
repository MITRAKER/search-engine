import json

import pytest

from searchengine.cli import main
from searchengine.datasets import load_jsonl, load_scifact_corpus


def test_cli_prints_ranked_results(capsys):
    assert main(["inverted index", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1. [")
    assert "Inverted index" in out


def test_cli_suggests_and_explains(capsys):
    main(["elastisearch"])
    assert "Did you mean: elasticsearch?" in capsys.readouterr().out
    main(["--fuzzy", "--explain", "elastisearch", "-k", "1"])
    out = capsys.readouterr().out
    assert "Elasticsearch" in out
    assert "idf=" in out


def test_cli_no_results(capsys):
    main(["xylophone"])
    assert "No results." in capsys.readouterr().out


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "1"}\n\n{"id": "2"}\n')
    assert [doc["id"] for doc in load_jsonl(path)] == ["1", "2"]


def test_load_jsonl_rejects_non_objects(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match=":1: expected a JSON object"):
        list(load_jsonl(path))


def test_load_scifact_corpus_joins_sentences(tmp_path):
    path = tmp_path / "corpus.jsonl"
    record = {"doc_id": 4983, "title": "Title", "abstract": ["One.", "Two."], "structured": False}
    path.write_text(json.dumps(record) + "\n")
    assert list(load_scifact_corpus(path)) == [
        {"id": "4983", "title": "Title", "body": "One. Two."}
    ]


def test_load_jsonl_rejects_huge_lines(tmp_path, monkeypatch):
    monkeypatch.setattr("searchengine.datasets.MAX_LINE_BYTES", 10)
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "a very long line"}\n')
    with pytest.raises(ValueError, match="longer than"):
        list(load_jsonl(path))
