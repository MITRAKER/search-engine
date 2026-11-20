"""Benchmark the engines on judged queries: how good, and how fast.

    searchengine-bench                                  # the sample data, our engine
    searchengine-bench --engine both                    # and Elasticsearch, side by side
    searchengine-bench --scifact datasets/scifact --download --engine both

Each configuration, a *system*, ranks the documents for every judged query.
``evaluation`` scores the rankings, and a randomization test checks whether
each system's nDCG differs from the first system's by more than luck would
explain. Every query is timed too, after a warm-up, so the report shows the
price of each system as well as its quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias

from .datasets import (
    SAMPLE,
    SAMPLE_QUERIES,
    fetch_scifact,
    load_jsonl,
    load_judged_queries,
    load_scifact_claims,
    load_scifact_corpus,
)
from .evaluation import (
    DEFAULT_TRIALS,
    MAX_JUDGED_QUERIES,
    MAX_TRIALS,
    Evaluation,
    JudgedQuery,
    QueryScores,
    evaluate,
    randomization_test,
)
from .index import MAX_RESULTS, InvertedIndex, SearchOptions

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from .es_dsl import QueryType
    from .es_engine import ElasticsearchEngine

Engine: TypeAlias = Literal["scratch", "es"]
WARMUP_QUERIES: Final = 20  # run untimed first: caches, connections, lazy setup
DEFAULT_ES_INDEX: Final = "searchengine-bench"
# How far a recorded quality score may move before --check calls it a change.
CHECK_TOLERANCE: Final = 1e-4
REPORT_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class System:
    """One configuration to benchmark: an engine, and how it searches."""

    name: str
    engine: Engine
    field_weights: Mapping[str, float] | None = None  # None: the engine's default
    options: SearchOptions = field(default_factory=SearchOptions)
    query_type: QueryType = "combined_fields"  # Elasticsearch only


SCRATCH_SYSTEMS: Final = (
    System("BM25F, title x3", "scratch"),
    System("BM25F, equal weights", "scratch", MappingProxyType({"title": 1.0, "body": 1.0})),
    System("TF-IDF, title x3", "scratch", options=SearchOptions(ranker="tfidf")),
)
ES_SYSTEMS: Final = (
    System("combined_fields, title x3", "es"),
    System(
        "best_fields, tie_breaker 0.5",  # BEIR's Elasticsearch baseline
        "es",
        MappingProxyType({"title": 1.0, "body": 1.0}),
        query_type="best_fields",
    ),
)


@dataclass(frozen=True, slots=True)
class Collection:
    """Documents and judged queries to benchmark on."""

    name: str
    documents: tuple[Mapping[str, Any], ...]
    queries: tuple[JudgedQuery, ...]


@dataclass(frozen=True, slots=True)
class Latency:
    """Time per query, in milliseconds."""

    mean: float
    p50: float
    p95: float
    p99: float


@dataclass(frozen=True, slots=True)
class SystemResult:
    """How one system did: quality, speed, and its difference from the first."""

    system: System
    evaluation: Evaluation
    latency: Latency
    server_latency: Latency | None  # Elasticsearch's own time, without the network
    p_value: float | None  # nDCG versus the first system; None for the first one


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one benchmark run measured."""

    collection: str
    documents: int
    queries: int
    k: int
    depth: int
    trials: int  # of the randomization test
    index_seconds: Mapping[str, float]  # engine -> time to build its index
    results: tuple[SystemResult, ...]
    environment: Mapping[str, str]


def _check_int(name: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an int from {low} to {high}")


def latency(seconds: Sequence[float]) -> Latency:
    """Summarize per-query times: the mean, the median and the slow tail."""
    if not seconds:
        raise ValueError("no timings to summarize")
    ms = sorted(value * 1000 for value in seconds)
    if len(ms) == 1:
        return Latency(ms[0], ms[0], ms[0], ms[0])
    cuts = statistics.quantiles(ms, n=100, method="inclusive")
    return Latency(statistics.fmean(ms), cuts[49], cuts[94], cuts[98])


def load_collection(
    scifact: Path | None, split: str = "dev", limit: int | None = None
) -> Collection:
    """The sample documents and queries, or SciFact from a directory."""
    if split not in ("dev", "train"):
        raise ValueError("split must be dev or train")
    if limit is not None:
        _check_int("the number of queries", limit, 1, MAX_JUDGED_QUERIES)
    if scifact is None:
        name = "sample"
        documents = tuple(load_jsonl(SAMPLE))
        queries = tuple(load_judged_queries(SAMPLE_QUERIES))
    else:
        name = f"SciFact ({split} claims)"
        documents = tuple(load_scifact_corpus(scifact / "corpus.jsonl"))
        queries = tuple(load_scifact_claims(scifact / f"claims_{split}.jsonl"))
    return Collection(name, documents, queries[:limit])


def _rank(
    system: System,
    query: str,
    depth: int,
    scratch: InvertedIndex | None,
    es: ElasticsearchEngine | None,
) -> tuple[list[str], float | None]:
    """One query on one system: the ranked ids, and Elasticsearch's own seconds."""
    if system.engine == "scratch":
        assert scratch is not None, "the scratch index is built before any search"
        hits = scratch.search(
            query, k=depth, options=system.options, field_weights=system.field_weights
        )
        return [hit.doc_id for hit in hits], None
    assert es is not None, "the Elasticsearch index is built before any search"
    ranking = es.rank(
        query,
        k=depth,
        options=system.options,
        field_weights=system.field_weights,
        query_type=system.query_type,
    )
    return ranking.doc_ids, ranking.took_ms / 1000


def run_system(
    system: System,
    queries: Sequence[JudgedQuery],
    depth: int,
    scratch: InvertedIndex | None = None,
    es: ElasticsearchEngine | None = None,
) -> tuple[dict[str, list[str]], list[float], list[float]]:
    """Rank every query: the run, the time of each call, and Elasticsearch's own times."""
    for query in queries[:WARMUP_QUERIES]:
        _rank(system, query.text, depth, scratch, es)
    run: dict[str, list[str]] = {}
    seconds: list[float] = []
    server: list[float] = []
    for query in queries:
        start = time.perf_counter()
        run[query.id], took = _rank(system, query.text, depth, scratch, es)
        seconds.append(time.perf_counter() - start)
        if took is not None:
            server.append(took)
    return run, seconds, server


def _p_value(first: Evaluation, other: Evaluation, trials: int) -> float:
    ids = list(first.per_query)
    return randomization_test(
        [first.per_query[query_id].ndcg for query_id in ids],
        [other.per_query[query_id].ndcg for query_id in ids],
        trials=trials,
    )


def _environment(es: ElasticsearchEngine | None) -> dict[str, str]:
    environment = {"python": platform.python_version(), "machine": platform.machine()}
    if es is not None:
        environment["elasticsearch"] = str(es.client.info().body["version"]["number"])
    return environment


def run_benchmark(
    collection: Collection,
    systems: Sequence[System],
    *,
    es: ElasticsearchEngine | None = None,
    k: int = 10,
    depth: int = 100,
    trials: int = DEFAULT_TRIALS,
) -> Report:
    """Index the collection in each engine the systems need, then run them all."""
    if not systems:
        raise ValueError("no systems to benchmark")
    _check_int("k", k, 1, MAX_RESULTS)
    _check_int("depth", depth, k, MAX_RESULTS)
    _check_int("trials", trials, 1, MAX_TRIALS)
    engines = {system.engine for system in systems}
    if "es" in engines and es is None:
        raise ValueError("Elasticsearch systems need an ElasticsearchEngine")
    index_seconds: dict[str, float] = {}
    scratch = None
    if "scratch" in engines:
        start = time.perf_counter()
        scratch = InvertedIndex()
        scratch.add_many(collection.documents)
        index_seconds["scratch"] = time.perf_counter() - start
    if es is not None and "es" in engines:
        es.create_index(recreate=True)
        start = time.perf_counter()
        es.index_documents(collection.documents)
        index_seconds["es"] = time.perf_counter() - start
    results: list[SystemResult] = []
    for system in systems:
        run, seconds, server = run_system(system, collection.queries, depth, scratch, es)
        evaluation = evaluate(run, collection.queries, k=k, depth=depth)
        first = results[0].evaluation if results else None
        results.append(
            SystemResult(
                system,
                evaluation,
                latency(seconds),
                latency(server) if server else None,
                None if first is None else _p_value(first, evaluation, trials),
            )
        )
    return Report(
        collection.name,
        len(collection.documents),
        len(collection.queries),
        k,
        depth,
        trials,
        MappingProxyType(index_seconds),
        tuple(results),
        MappingProxyType(_environment(es if "es" in engines else None)),
    )


# ---------- the report, as Markdown and JSON ----------


def _scores_dict(report: Report, scores: QueryScores) -> dict[str, float]:
    return {
        f"nDCG@{report.k}": round(scores.ndcg, 6),
        f"MRR@{report.k}": round(scores.mrr, 6),
        f"P@{report.k}": round(scores.precision, 6),
        f"Recall@{report.depth}": round(scores.recall, 6),
        f"MAP@{report.depth}": round(scores.map, 6),
    }


def _latency_dict(times: Latency | None) -> dict[str, float] | None:
    if times is None:
        return None
    return {
        "mean": round(times.mean, 3),
        "p50": round(times.p50, 3),
        "p95": round(times.p95, 3),
        "p99": round(times.p99, 3),
    }


def report_to_json(report: Report) -> dict[str, Any]:
    """The report as plain data: quality means, latency, and the environment."""
    return {
        "version": REPORT_VERSION,
        "collection": report.collection,
        "documents": report.documents,
        "queries": report.queries,
        "k": report.k,
        "depth": report.depth,
        "trials": report.trials,
        "index_seconds": {name: round(value, 3) for name, value in report.index_seconds.items()},
        "environment": dict(report.environment),
        "systems": [
            {
                "name": result.system.name,
                "engine": result.system.engine,
                "quality": _scores_dict(report, result.evaluation.mean),
                "p_value": None if result.p_value is None else round(result.p_value, 4),
                "latency_ms": _latency_dict(result.latency),
                "server_latency_ms": _latency_dict(result.server_latency),
            }
            for result in report.results
        ],
    }


def _quality_table(report: Report) -> list[str]:
    k, depth = report.k, report.depth
    lines = [
        f"| System | Engine | nDCG@{k} | MRR@{k} | P@{k} | Recall@{depth} | MAP@{depth} "
        "| p (nDCG vs row 1) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in report.results:
        mean = result.evaluation.mean
        p_value = "" if result.p_value is None else f"{result.p_value:.4f}"
        lines.append(
            f"| {result.system.name} | {result.system.engine} | {mean.ndcg:.4f} "
            f"| {mean.mrr:.4f} | {mean.precision:.4f} | {mean.recall:.4f} | {mean.map:.4f} "
            f"| {p_value} |"
        )
    return lines


def _speed_table(report: Report) -> list[str]:
    lines = [
        "| System | Engine | Mean ms | Median ms | p95 ms | p99 ms "
        "| Inside Elasticsearch, median ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report.results:
        times = result.latency
        server = "" if result.server_latency is None else f"{result.server_latency.p50:.2f}"
        lines.append(
            f"| {result.system.name} | {result.system.engine} | {times.mean:.2f} "
            f"| {times.p50:.2f} | {times.p95:.2f} | {times.p99:.2f} | {server} |"
        )
    return lines


def format_markdown(report: Report) -> str:
    """Two tables for people, one row per system: quality, then speed."""
    built = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in report.index_seconds.items())
    lines = [
        f"**{report.collection}**: {report.documents:,} documents, "
        f"{report.queries:,} judged queries.",
        "",
        *_quality_table(report),
        "",
        *_speed_table(report),
        "",
        f"Time to build the index: {built}.",
    ]
    return "\n".join(lines) + "\n"


def check_against(report: Mapping[str, Any], recorded: Mapping[str, Any]) -> list[str]:
    """What changed compared with a recorded report: the data, or a quality score.

    Speed is not compared: it depends on the machine. Every system in the
    report must be in the recorded one, so a new or renamed system means
    recording the report again.
    """
    changes = [
        f"{key}: {recorded.get(key)} -> {report[key]}"
        for key in ("collection", "documents", "queries", "k", "depth")
        if recorded.get(key) != report[key]
    ]
    expected = {system["name"]: system["quality"] for system in recorded["systems"]}
    for system in report["systems"]:
        before = expected.get(system["name"])
        if before is None:
            changes.append(f"{system['name']}: not in the recorded report")
            continue
        for metric, value in system["quality"].items():
            old = before.get(metric)
            if old is None or abs(value - old) > CHECK_TOLERANCE:
                changes.append(f"{system['name']}: {metric} {old} -> {value}")
    return changes


def _is_recorded_system(system: object) -> bool:
    if not isinstance(system, dict) or not isinstance(system.get("name"), str):
        return False
    quality = system.get("quality")
    return isinstance(quality, dict) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in quality.values()
    )


def load_report(path: Path) -> dict[str, Any]:
    """Read a report written with --json, checking its shape."""
    data: object = json.loads(path.read_text("utf-8"))
    systems = data.get("systems") if isinstance(data, dict) else None
    if not (
        isinstance(data, dict)
        and data.get("version") == REPORT_VERSION
        and isinstance(systems, list)
        and all(_is_recorded_system(system) for system in systems)
    ):
        raise ValueError(f"{path} is not a benchmark report (version {REPORT_VERSION})")
    return data


# ---------- command line ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchengine-bench",
        description="Measure search quality and speed on judged queries.",
    )
    parser.add_argument("--engine", choices=["scratch", "es", "both"], default="scratch")
    parser.add_argument("--scifact", type=Path, help="SciFact's directory (default: sample data)")
    parser.add_argument("--download", action="store_true", help="download SciFact if missing")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--queries", type=int, help="use only the first N queries")
    parser.add_argument("-k", type=int, default=10, help="cutoff for nDCG, MRR and P")
    parser.add_argument("--depth", type=int, default=100, help="cutoff for recall and MAP")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--index", default=DEFAULT_ES_INDEX, help="Elasticsearch index to use")
    parser.add_argument("--json", type=Path, help="write the report as JSON")
    parser.add_argument("--markdown", type=Path, help="write the table as Markdown")
    parser.add_argument("--check", type=Path, help="fail if quality differs from this report")
    return parser


def _systems(engine: str) -> tuple[System, ...]:
    if engine == "scratch":
        return SCRATCH_SYSTEMS
    if engine == "es":
        return ES_SYSTEMS
    return SCRATCH_SYSTEMS + ES_SYSTEMS


def _write_outputs(args: argparse.Namespace, report: Report) -> int:
    data = report_to_json(report)
    table = format_markdown(report)
    print(table)
    if args.json:
        args.json.write_text(json.dumps(data, indent=2) + "\n", "utf-8")
    if args.markdown:
        args.markdown.write_text(table, "utf-8")
    if args.check:
        changes = check_against(data, load_report(args.check))
        for change in changes:
            print(f"searchengine-bench: quality changed: {change}", file=sys.stderr)
        return 1 if changes else 0
    return 0


def main(argv: Sequence[str] | None = None, *, es_client: Elasticsearch | None = None) -> int:
    """Run the benchmark from the command line. Returns the exit status."""
    args = build_parser().parse_args(argv)
    try:
        if args.download:
            if args.scifact is None:
                raise ValueError("--download needs --scifact DIRECTORY")
            fetch_scifact(args.scifact)
        collection = load_collection(args.scifact, args.split, args.queries)
        systems = _systems(args.engine)
        if args.engine == "scratch":
            report = run_benchmark(
                collection, systems, k=args.k, depth=args.depth, trials=args.trials
            )
        else:
            report = _run_with_elasticsearch(args, collection, systems, es_client)
        return _write_outputs(args, report)
    except (OSError, ValueError, TypeError) as error:
        print(f"searchengine-bench: error: {error}", file=sys.stderr)
        return 1


def _run_with_elasticsearch(
    args: argparse.Namespace,
    collection: Collection,
    systems: Sequence[System],
    client: Elasticsearch | None,
) -> Report:
    """Run with an Elasticsearch index made for the run, and deleted after it."""
    # Imported here because the elasticsearch package is optional.
    try:
        from elastic_transport import ConnectionError as TransportConnectionError  # noqa: PLC0415
        from elasticsearch import ApiError  # noqa: PLC0415

        from .es_engine import ElasticsearchEngine, IndexingError, default_url  # noqa: PLC0415
    except ImportError as error:
        raise ValueError(
            "--engine es needs the elasticsearch package: "
            "pip install 'searchengine-from-scratch[es]'"
        ) from error
    es = ElasticsearchEngine(client, index=args.index)
    try:
        report = run_benchmark(
            collection, systems, es=es, k=args.k, depth=args.depth, trials=args.trials
        )
    except TransportConnectionError as error:
        raise OSError(
            f"cannot reach Elasticsearch at {default_url()}. Start it with: docker compose up -d"
        ) from error
    except (ApiError, IndexingError) as error:
        raise ValueError(f"Elasticsearch: {error}") from error
    es.delete_index()
    return report
