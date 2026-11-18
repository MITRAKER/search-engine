import json

import pytest

from searchengine.cli import main
from searchengine.datasets import load_jsonl, load_scifact_corpus
from searchengine.index import MAX_QUERY_TERMS, MAX_RESULTS


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


# ---------- bad input: a clear one-line error, never a traceback ----------


def test_cli_rejects_a_bad_result_count(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["search", "-k", "0"])
    assert exit_info.value.code == 2  # argparse's usage error
    assert f"-k must be between 1 and {MAX_RESULTS}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "No such file"),
        ('{"id": "1"}\n{"id": \n', ":2: invalid JSON"),
        (b'{"id": "\xff"}\n', ":1: not valid UTF-8"),
        ("[" * 100_000 + "]" * 100_000 + "\n", ":1: the JSON is nested too deeply"),
    ],
)
def test_cli_reports_bad_data_in_one_line(tmp_path, capsys, contents, message):
    path = tmp_path / "docs.jsonl"
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    elif contents is not None:
        path.write_text(contents)
    assert main(["search", "--data", str(path)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("searchengine: error: ")
    assert message in error
    assert error.count("\n") == 1


def test_cli_reports_an_overlong_query(capsys):
    assert main([" ".join(["word"] * (MAX_QUERY_TERMS + 1))]) == 1
    assert "limited" in capsys.readouterr().err


def test_load_jsonl_says_where_the_json_is_broken(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "1"}\n\n{"id": \n')
    with pytest.raises(ValueError, match=r"docs.jsonl:3: invalid JSON: Expecting value"):
        list(load_jsonl(path))


def test_load_jsonl_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_bytes(b'{"id": "1"}\n{"id": "caf\xe9"}\n')  # Latin-1, not UTF-8
    with pytest.raises(ValueError, match=":2: not valid UTF-8"):
        list(load_jsonl(path))


def test_line_limit_counts_bytes_not_characters(tmp_path, monkeypatch):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"a":"éé"}\n', encoding="utf-8")  # 11 characters, 13 bytes
    monkeypatch.setattr("searchengine.datasets.MAX_LINE_BYTES", 12)
    with pytest.raises(ValueError, match="longer than 12 bytes"):
        list(load_jsonl(path))
    monkeypatch.setattr("searchengine.datasets.MAX_LINE_BYTES", 13)
    assert list(load_jsonl(path)) == [{"a": "éé"}]


@pytest.mark.parametrize(
    "record",
    [
        {"title": "No id", "abstract": []},
        {"doc_id": True, "title": "Bool id", "abstract": []},
        {"doc_id": 1.5, "title": "Float id", "abstract": []},
        {"doc_id": 2, "title": None, "abstract": []},
        {"doc_id": 2, "title": "Text abstract", "abstract": "not a list"},
        {"doc_id": 2, "title": "Number in abstract", "abstract": ["One.", 2]},
    ],
)
def test_load_scifact_corpus_rejects_malformed_records(tmp_path, record):
    path = tmp_path / "corpus.jsonl"
    good = {"doc_id": 1, "title": "Fine", "abstract": ["One."]}
    path.write_text(json.dumps(good) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="record 2 is not a SciFact document"):
        list(load_scifact_corpus(path))
