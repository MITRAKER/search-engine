"""The benchmark: ranking every judged query, scoring, timing, and the report.

Elasticsearch is the fake client from ``fake_es``, so these tests need no
server. ``test_es_integration.py`` runs the benchmark on a real one.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import pytest
from elastic_transport import ConnectionError as TransportConnectionError
from elastic_transport import ConnectionTimeout

from searchengine import benchmark
from searchengine.benchmark import (
    CHECK_TOLERANCE,
    DEFAULT_ES_INDEX,
    ES_SYSTEMS,
    REPORT_VERSION,
    SCRATCH_SYSTEMS,
    WARMUP_QUERIES,
    Latency,
    Report,
    System,
    SystemResult,
    check_against,
    format_markdown,
    latency,
    load_collection,
    load_report,
    main,
    report_to_json,
    run_benchmark,
    run_system,
)
from searchengine.datasets import SAMPLE, load_jsonl
from searchengine.es_engine import ElasticsearchEngine
from searchengine.evaluation import (
    MAX_JUDGED_QUERIES,
    MAX_TRIALS,
    Evaluation,
    JudgedQuery,
    QueryScores,
    judged_query,
)
from searchengine.index import InvertedIndex
from tests.fake_es import FakeElasticsearch, fake_client, not_found, search_response

SAMPLE_QUERIES = 12
# Each Elasticsearch system searches every sample query twice: warm-up, then timed.
SEARCHES_PER_SYSTEM = min(WARMUP_QUERIES, SAMPLE_QUERIES) + SAMPLE_QUERIES


def queries(count: int) -> tuple[JudgedQuery, ...]:
    return tuple(judged_query(f"q{n}", f"query number {n}", {"1": 1}) for n in range(count))


def es_engine(*, existing: bool = False) -> tuple[ElasticsearchEngine, FakeElasticsearch]:
    client, fake = fake_client(existing=existing)
    return ElasticsearchEngine(client, index=DEFAULT_ES_INDEX), fake


def ranking(*doc_ids: str) -> dict[str, Any]:
    """Elasticsearch's answer to a ranking request: these ids, best first."""
    hits = [{"_id": doc_id, "_score": float(len(doc_ids) - n)} for n, doc_id in enumerate(doc_ids)]
    return search_response(hits)  # "took": 3 ms


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), "utf-8")


def write_scifact(directory: Path) -> Path:
    """A tiny SciFact, in SciFact's own formats: three abstracts, three claims."""
    directory.mkdir(parents=True, exist_ok=True)
    corpus = [
        {"doc_id": 11, "title": "Aspirin and the heart", "abstract": ["Aspirin.", "Hearts."]},
        {"doc_id": 12, "title": "Vitamin D", "abstract": ["Vitamin D makes bones strong."]},
        {"doc_id": 13, "title": "Sleep", "abstract": ["Sleep helps memory."]},
    ]
    write_jsonl(directory / "corpus.jsonl", corpus)
    write_jsonl(
        directory / "claims_dev.jsonl",
        [
            {"id": 1, "claim": "Aspirin protects the heart.", "cited_doc_ids": [11]},
            {"id": 2, "claim": "Sleep improves memory.", "cited_doc_ids": [13]},
        ],
    )
    write_jsonl(
        directory / "claims_train.jsonl",
        [{"id": 3, "claim": "Vitamin D strengthens bones.", "cited_doc_ids": [12]}],
    )
    return directory


# ---------- latency ----------


def test_latency_summarizes_milliseconds_with_mean_median_and_tail():
    seconds = [n / 1000 for n in range(1, 101)]  # 1 to 100 ms
    for order in (seconds, seconds[::-1]):
        times = latency(order)
        assert (times.mean, times.p50, times.p95, times.p99) == pytest.approx(
            (50.5, 50.5, 95.05, 99.01)
        )


def test_one_timing_is_its_own_mean_median_and_tail():
    times = latency([0.004])
    assert (times.mean, times.p50, times.p95, times.p99) == pytest.approx((4.0, 4.0, 4.0, 4.0))


def test_latency_needs_timings():
    with pytest.raises(ValueError, match="no timings to summarize"):
        latency([])


# ---------- collections ----------


def test_the_default_collection_is_the_sample_data_with_its_judged_queries():
    collection = load_collection(None)
    assert collection.name == "sample"
    assert collection.documents == tuple(load_jsonl(SAMPLE))
    assert [query.id for query in collection.queries] == [f"q{n}" for n in range(1, 13)]


def test_load_collection_can_keep_only_the_first_queries():
    assert [query.id for query in load_collection(None, limit=2).queries] == ["q1", "q2"]


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, MAX_JUDGED_QUERIES + 1])
def test_the_number_of_queries_is_checked(limit):
    with pytest.raises(ValueError, match="the number of queries must be an int from 1 to"):
        load_collection(None, limit=limit)


def test_the_split_is_checked():
    with pytest.raises(ValueError, match="split must be dev or train"):
        load_collection(None, "test")


@pytest.mark.parametrize(("split", "ids"), [("dev", ["1", "2"]), ("train", ["3"])])
def test_load_collection_reads_scifact_and_the_claims_of_one_split(tmp_path, split, ids):
    collection = load_collection(write_scifact(tmp_path / "scifact"), split)
    assert collection.name == f"SciFact ({split} claims)"
    assert [document["id"] for document in collection.documents] == ["11", "12", "13"]
    assert [query.id for query in collection.queries] == ids


# ---------- running one system ----------


def test_run_system_ranks_every_query_with_the_scratch_engine():
    scratch = InvertedIndex()
    scratch.add_many(load_jsonl(SAMPLE))
    sample = load_collection(None)
    run, seconds, server = run_system(SCRATCH_SYSTEMS[0], sample.queries, 5, scratch=scratch)
    for query in sample.queries:
        hits = scratch.search(query.text, k=5)
        assert run[query.id] == [hit.doc_id for hit in hits]
    assert len(seconds) == SAMPLE_QUERIES
    assert all(value >= 0 for value in seconds)
    assert server == []  # only Elasticsearch reports its own time


@pytest.mark.parametrize(("count", "warmup"), [(5, 5), (WARMUP_QUERIES + 5, WARMUP_QUERIES)])
def test_run_system_warms_up_untimed_then_times_every_query(count, warmup):
    es, fake = es_engine()
    fake.search_responses = [ranking("2", "1")] * (warmup + count)
    run, seconds, server = run_system(ES_SYSTEMS[0], queries(count), 10, es=es)
    assert fake.search_responses == []  # every answer was used: warm-up and timed
    assert run == {f"q{n}": ["2", "1"] for n in range(count)}
    assert len(seconds) == count
    assert server == [0.003] * count  # Elasticsearch's own "took", in seconds


def test_each_system_ranks_with_its_own_query_type_weights_and_depth():
    es, fake = es_engine()
    fake.search_responses = [ranking()] * 2
    run_system(ES_SYSTEMS[1], queries(1), 7, es=es)
    request = fake.called("search")[-1]
    assert (request["size"], request["source"]) == (7, False)
    assert request["sort"] == [{"_score": "desc"}, {"id": "asc"}]  # ties by id, like ours
    query = request["query"]["multi_match"]
    assert (query["type"], query["fields"]) == ("best_fields", ["title^1.0", "body^1.0"])


# ---------- running the benchmark ----------


def test_run_benchmark_needs_systems():
    with pytest.raises(ValueError, match="no systems to benchmark"):
        run_benchmark(load_collection(None), [])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"k": 0}, "k must be an int from 1 to 1000"),
        ({"k": True}, "k must be an int from 1 to 1000"),
        ({"k": 20, "depth": 10}, "depth must be an int from 20 to 1000"),
        ({"depth": 1001}, "depth must be an int from 10 to 1000"),
        ({"trials": 0}, f"trials must be an int from 1 to {MAX_TRIALS}"),
    ],
)
def test_run_benchmark_checks_k_depth_and_trials(arguments, message):
    with pytest.raises(ValueError, match=message):
        run_benchmark(load_collection(None), SCRATCH_SYSTEMS, **arguments)


def test_a_scratch_benchmark_reports_every_system_in_order():
    report = run_benchmark(load_collection(None), SCRATCH_SYSTEMS, trials=100)
    assert (report.collection, report.documents, report.queries) == ("sample", 25, 12)
    assert (report.k, report.depth, report.trials) == (10, 100, 100)
    assert list(report.index_seconds) == ["scratch"]
    assert [result.system for result in report.results] == list(SCRATCH_SYSTEMS)
    assert all(result.evaluation.k == 10 for result in report.results)
    assert report.results[0].p_value is None  # the first system is the baseline
    for result in report.results[1:]:
        assert result.p_value is not None
        assert 0 < result.p_value <= 1
    assert all(result.server_latency is None for result in report.results)
    assert set(report.environment) == {"python", "machine"}


def test_elasticsearch_systems_need_an_engine():
    with pytest.raises(ValueError, match="need an ElasticsearchEngine"):
        run_benchmark(load_collection(None), ES_SYSTEMS)


def test_an_elasticsearch_benchmark_makes_its_own_index_and_deletes_it():
    es, fake = es_engine()
    fake.search_responses = [ranking("1", "2")] * (2 * SEARCHES_PER_SYSTEM)
    report = run_benchmark(load_collection(None), ES_SYSTEMS, es=es, trials=10)
    assert fake.search_responses == []
    assert [call["index"] for call in fake.called("indices.create")] == [DEFAULT_ES_INDEX]
    assert len(fake.called("bulk")[0]["operations"]) == 2 * 25  # an action and a document each
    assert fake.called("indices.delete") == [{"index": DEFAULT_ES_INDEX}]
    assert list(report.index_seconds) == ["es"]
    assert report.environment["elasticsearch"] == "9.5.5"
    assert all(result.server_latency is not None for result in report.results)


def test_an_existing_index_is_refused_not_replaced():
    es, fake = es_engine(existing=True)
    with pytest.raises(ValueError, match="'searchengine-bench' already exists"):
        run_benchmark(load_collection(None), ES_SYSTEMS, es=es)
    assert fake.called("indices.create") == []
    assert fake.called("indices.delete") == []


@pytest.mark.parametrize("error", [not_found({"error": "gone"}), KeyboardInterrupt()])
def test_the_index_is_deleted_when_the_run_fails(monkeypatch, error):
    es, fake = es_engine()

    def broken(**kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(fake, "search", broken)
    with pytest.raises(type(error)):
        run_benchmark(load_collection(None), ES_SYSTEMS, es=es)
    assert fake.called("indices.delete") == [{"index": DEFAULT_ES_INDEX}]


def test_both_engines_run_side_by_side_in_the_order_given():
    es, fake = es_engine()
    fake.search_responses = [ranking("1")] * (2 * SEARCHES_PER_SYSTEM)
    systems = SCRATCH_SYSTEMS + ES_SYSTEMS
    report = run_benchmark(load_collection(None), systems, es=es, trials=10)
    assert list(report.index_seconds) == ["scratch", "es"]
    assert [result.system for result in report.results] == list(systems)


# ---------- the report ----------


def scores(value: float) -> QueryScores:
    return QueryScores(value, value, value, value, value)


def result(
    system: System, value: float, p_value: float | None, server: Latency | None = None
) -> SystemResult:
    evaluation = Evaluation(10, 100, MappingProxyType({"q1": scores(value)}), scores(value))
    return SystemResult(system, evaluation, Latency(1.23456, 1.0, 2.0, 3.0), server, p_value)


REPORT = Report(
    collection="SciFact (dev claims)",
    documents=5183,
    queries=300,
    k=10,
    depth=100,
    trials=1000,
    index_seconds=MappingProxyType({"scratch": 0.1234, "es": 2.5}),
    results=(
        result(SCRATCH_SYSTEMS[0], 0.123456789, None),
        result(ES_SYSTEMS[0], 0.5, 0.04567, Latency(4.0, 4.0, 5.0, 6.0)),
    ),
    environment=MappingProxyType({"python": "3.13.0", "machine": "x86_64"}),
)
RECORDED = report_to_json(REPORT)


def test_the_json_report_rounds_quality_latency_and_p_values():
    assert {key: value for key, value in RECORDED.items() if key != "systems"} == {
        "version": REPORT_VERSION,
        "collection": "SciFact (dev claims)",
        "documents": 5183,
        "queries": 300,
        "k": 10,
        "depth": 100,
        "trials": 1000,
        "index_seconds": {"scratch": 0.123, "es": 2.5},
        "environment": {"python": "3.13.0", "machine": "x86_64"},
    }
    first, second = RECORDED["systems"]
    assert first == {
        "name": "BM25F, title x3",
        "engine": "scratch",
        "quality": dict.fromkeys(["nDCG@10", "MRR@10", "P@10", "Recall@100", "MAP@100"], 0.123457),
        "p_value": None,
        "latency_ms": {"mean": 1.235, "p50": 1.0, "p95": 2.0, "p99": 3.0},
        "server_latency_ms": None,
    }
    assert (second["name"], second["engine"], second["p_value"]) == (
        "combined_fields, title x3",
        "es",
        0.0457,
    )
    assert second["server_latency_ms"] == {"mean": 4.0, "p50": 4.0, "p95": 5.0, "p99": 6.0}


def test_the_markdown_report_has_a_quality_table_and_a_speed_table():
    assert format_markdown(REPORT).splitlines() == [
        "**SciFact (dev claims)**: 5,183 documents, 300 judged queries.",
        "",
        "| System | Engine | nDCG@10 | MRR@10 | P@10 | Recall@100 | MAP@100 | p (nDCG vs row 1) |",
        "|---|---|---|---|---|---|---|---|",
        "| BM25F, title x3 | scratch | 0.1235 | 0.1235 | 0.1235 | 0.1235 | 0.1235 |  |",
        "| combined_fields, title x3 | es | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0457 |",
        "",
        "| System | Engine | Mean ms | Median ms | p95 ms | p99 ms "
        "| Inside Elasticsearch, median ms |",
        "|---|---|---|---|---|---|---|",
        "| BM25F, title x3 | scratch | 1.23 | 1.00 | 2.00 | 3.00 |  |",
        "| combined_fields, title x3 | es | 1.23 | 1.00 | 2.00 | 3.00 | 4.00 |",
        "",
        "Time to build the index: scratch 0.12 s, es 2.50 s.",
    ]


# ---------- comparing with a recorded report ----------


def test_the_same_quality_is_no_change():
    assert check_against(RECORDED, RECORDED) == []


def test_speed_is_not_compared():
    slower = copy.deepcopy(RECORDED)
    slower["systems"][0]["latency_ms"]["p50"] = 99.0
    slower["index_seconds"]["scratch"] = 99.0
    assert check_against(slower, RECORDED) == []


def test_a_quality_score_that_moves_is_a_change():
    changed = copy.deepcopy(RECORDED)
    changed["systems"][1]["quality"]["nDCG@10"] = 0.51
    assert check_against(changed, RECORDED) == ["combined_fields, title x3: nDCG@10 0.5 -> 0.51"]


def test_rounding_noise_is_not_a_change():
    changed = copy.deepcopy(RECORDED)
    changed["systems"][1]["quality"]["nDCG@10"] = 0.5 + CHECK_TOLERANCE / 2
    assert check_against(changed, RECORDED) == []


@pytest.mark.parametrize(
    ("key", "value"), [("collection", "sample"), ("documents", 25), ("queries", 12), ("k", 5)]
)
def test_other_data_or_cutoffs_are_a_change(key, value):
    changed = copy.deepcopy(RECORDED)
    changed[key] = value
    assert check_against(changed, RECORDED) == [f"{key}: {RECORDED[key]} -> {value}"]


def test_a_system_the_recording_does_not_have_is_a_change():
    changed = copy.deepcopy(RECORDED)
    changed["systems"][1]["name"] = "renamed"
    assert check_against(changed, RECORDED) == ["renamed: not in the recorded report"]


def test_a_metric_the_recording_does_not_have_is_a_change():
    recorded = copy.deepcopy(RECORDED)
    del recorded["systems"][0]["quality"]["MAP@100"]
    assert check_against(RECORDED, recorded) == ["BM25F, title x3: MAP@100 None -> 0.123457"]


def test_systems_left_out_of_a_run_are_not_a_change():
    partial = copy.deepcopy(RECORDED)
    partial["systems"] = partial["systems"][:1]
    assert check_against(partial, RECORDED) == []


def test_a_written_report_loads_back(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(RECORDED), "utf-8")
    assert load_report(path) == RECORDED


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"version": REPORT_VERSION + 1, "systems": []},
        {"version": REPORT_VERSION},
        {"version": REPORT_VERSION, "systems": {}},
        {"version": REPORT_VERSION, "systems": ["BM25F"]},
        {"version": REPORT_VERSION, "systems": [{"quality": {}}]},
        {"version": REPORT_VERSION, "systems": [{"name": "BM25F"}]},
        {"version": REPORT_VERSION, "systems": [{"name": "BM25F", "quality": {"P@10": "0.5"}}]},
        {"version": REPORT_VERSION, "systems": [{"name": "BM25F", "quality": {"P@10": True}}]},
    ],
)
def test_load_report_refuses_what_is_not_a_report(tmp_path, data):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ValueError, match="is not a benchmark report"):
        load_report(path)


def test_load_report_refuses_a_file_that_is_not_json(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{", "utf-8")
    with pytest.raises(ValueError, match="Expecting property name"):
        load_report(path)


# ---------- the command line ----------


def test_the_default_run_benchmarks_our_engine_on_the_sample(capsys):
    assert main(["--trials", "10"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("**sample**: 25 documents, 12 judged queries.\n")
    for system in SCRATCH_SYSTEMS:
        assert f"| {system.name} | scratch |" in out
    assert "Time to build the index: scratch " in out


def test_the_report_is_also_written_as_json_and_markdown(tmp_path, capsys):
    report, table = tmp_path / "report.json", tmp_path / "table.md"
    assert main(["--trials", "10", "--json", str(report), "--markdown", str(table)]) == 0
    assert capsys.readouterr().out == table.read_text("utf-8") + "\n"
    names = [system["name"] for system in load_report(report)["systems"]]
    assert names == [system.name for system in SCRATCH_SYSTEMS]


def test_check_passes_when_quality_is_unchanged(tmp_path):
    report = tmp_path / "report.json"
    assert main(["--trials", "10", "--json", str(report)]) == 0
    assert main(["--trials", "10", "--check", str(report)]) == 0


def test_check_fails_and_says_what_changed(tmp_path, capsys):
    report = tmp_path / "report.json"
    assert main(["--trials", "10", "--json", str(report)]) == 0
    data = json.loads(report.read_text("utf-8"))
    data["systems"][0]["quality"]["nDCG@10"] += 0.01
    report.write_text(json.dumps(data), "utf-8")
    capsys.readouterr()
    assert main(["--trials", "10", "--check", str(report)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("searchengine-bench: quality changed: BM25F, title x3: nDCG@10 ")
    assert error.count("\n") == 1


def test_a_bad_check_file_fails_before_the_benchmark_runs(tmp_path, monkeypatch, capsys):
    def must_not_run(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("the benchmark ran")

    monkeypatch.setattr(benchmark, "run_benchmark", must_not_run)
    assert main(["--check", str(tmp_path / "missing.json")]) == 1
    assert "No such file" in capsys.readouterr().err


def test_download_needs_a_directory(capsys):
    assert main(["--download"]) == 1
    assert "--download needs --scifact DIRECTORY" in capsys.readouterr().err


def test_download_fetches_scifact_into_the_directory_first(tmp_path, monkeypatch, capsys):
    fetched: list[Path] = []

    def fake_fetch(dest: Path) -> Path:
        fetched.append(dest)
        return write_scifact(dest)

    monkeypatch.setattr(benchmark, "fetch_scifact", fake_fetch)
    directory = tmp_path / "scifact"
    assert main(["--scifact", str(directory), "--download", "--trials", "10"]) == 0
    assert fetched == [directory]
    out = capsys.readouterr().out
    assert out.startswith("**SciFact (dev claims)**: 3 documents, 2 judged queries.\n")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--queries", "0"], "the number of queries must be an int from 1 to"),
        (["-k", "0"], "k must be an int from 1 to 1000"),
        (["-k", "20", "--depth", "10"], "depth must be an int from 20 to 1000"),
        (["--trials", "0"], "trials must be an int from 1 to"),
        (["--scifact", "no-such-directory"], "No such file"),
    ],
)
def test_bad_arguments_are_one_line_errors(capsys, arguments, message):
    assert main(arguments) == 1
    error = capsys.readouterr().err
    assert error.startswith("searchengine-bench: error: ")
    assert message in error
    assert error.count("\n") == 1


def test_engine_es_benchmarks_elasticsearch_on_an_index_of_its_own(capsys):
    client, fake = fake_client()
    fake.search_responses = [ranking("1", "2")] * (2 * SEARCHES_PER_SYSTEM)
    assert main(["--engine", "es", "--index", "bench-2", "--trials", "10"], es_client=client) == 0
    out = capsys.readouterr().out
    for system in ES_SYSTEMS:
        assert f"| {system.name} | es |" in out
    assert "scratch" not in out
    assert [call["index"] for call in fake.called("indices.create")] == ["bench-2"]
    assert fake.called("indices.delete") == [{"index": "bench-2"}]


def test_engine_both_runs_all_five_systems(capsys):
    client, fake = fake_client()
    fake.search_responses = [ranking("1")] * (2 * SEARCHES_PER_SYSTEM)
    assert main(["--engine", "both", "--trials", "10"], es_client=client) == 0
    out = capsys.readouterr().out
    for system in SCRATCH_SYSTEMS + ES_SYSTEMS:
        assert f"| {system.name} | {system.engine} |" in out


def test_an_existing_index_is_a_one_line_error_and_is_kept(capsys):
    client, fake = fake_client(existing=True)
    assert main(["--engine", "es", "--index", "mine"], es_client=client) == 1
    assert "index 'mine' already exists" in capsys.readouterr().err
    assert fake.called("indices.delete") == []


def test_an_invalid_index_name_is_a_one_line_error(capsys):
    client, fake = fake_client()
    assert main(["--engine", "es", "--index", ""], es_client=client) == 1
    assert "invalid index name ''" in capsys.readouterr().err
    assert fake.calls == []


def test_unreachable_elasticsearch_gets_a_hint(capsys, monkeypatch):
    monkeypatch.setenv("ES_URL", "http://localhost:1")
    client, fake = fake_client()
    fake.fail_with = TransportConnectionError("refused")
    assert main(["--engine", "es"], es_client=client) == 1
    error = capsys.readouterr().err
    assert "cannot reach Elasticsearch at http://localhost:1" in error
    assert "docker compose up -d --wait" in error  # --wait: until it answers


@pytest.mark.parametrize(
    "error", [ConnectionTimeout("timed out"), not_found({"error": "no such index"})]
)
def test_other_elasticsearch_errors_are_one_line(capsys, error):
    client, fake = fake_client()
    fake.fail_with = error
    assert main(["--engine", "es"], es_client=client) == 1
    message = capsys.readouterr().err
    assert message.startswith("searchengine-bench: error: Elasticsearch: ")
    assert message.count("\n") == 1  # no traceback


def test_rejected_documents_are_an_error_and_the_index_is_deleted(capsys):
    client, fake = fake_client()
    fake.bulk_errors = {"3": "mapper_parsing_exception"}
    assert main(["--engine", "es"], es_client=client) == 1
    assert "1 of 25 documents failed" in capsys.readouterr().err
    assert fake.called("indices.delete") == [{"index": DEFAULT_ES_INDEX}]


def test_a_missing_elasticsearch_package_is_explained(capsys, monkeypatch):
    for module in ("elastic_transport", "elasticsearch", "searchengine.es_engine"):
        monkeypatch.setitem(sys.modules, module, None)  # makes importing it fail
    assert main(["--engine", "es"]) == 1
    assert "pip install 'searchengine-from-scratch[es]'" in capsys.readouterr().err
