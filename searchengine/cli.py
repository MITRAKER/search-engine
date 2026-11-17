"""Try the engines from the terminal.

searchengine "inverted index"
searchengine --fuzzy "elastisearch rankng"
searchengine --ranker tfidf '"search engine"'
searchengine --explain "bm25 ranking"
searchengine --engine es "inverted index"          # Elasticsearch: docker compose up -d
searchengine --engine es --category indexing "index"
"""

from __future__ import annotations

import argparse
import html
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .datasets import SAMPLE, load_jsonl
from .es_dsl import explanation_lines
from .index import MAX_RESULTS, InvertedIndex, SearchOptions

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from .es_engine import ElasticsearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchengine",
        description="Search a JSON Lines file with the from-scratch engine or Elasticsearch.",
    )
    parser.add_argument("query", help='words to search for; wrap in "double quotes" for a phrase')
    parser.add_argument(
        "--engine",
        choices=["scratch", "es"],
        default="scratch",
        help="scratch (default), or es: Elasticsearch at $ES_URL (default http://localhost:9200)",
    )
    parser.add_argument(
        "--data",
        default=str(SAMPLE),
        help="JSON Lines file to index (with --engine es: when the index is created)",
    )
    parser.add_argument("--ranker", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--mode", choices=["or", "and"], default="or")
    parser.add_argument("--fuzzy", action="store_true", help="fix typos in the query")
    parser.add_argument("--explain", action="store_true", help="show why each result scored")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    es = parser.add_argument_group("Elasticsearch only")
    es.add_argument("--index", help="index name (default: searchengine)")
    es.add_argument("--reindex", action="store_true", help="rebuild the index from --data")
    es.add_argument("--category", help="only show results in this category")
    return parser


def main(argv: Sequence[str] | None = None, *, es_client: Elasticsearch | None = None) -> int:
    """Run the command line. Returns the exit status: 0 for success.

    ``es_client`` lets tests pass a stand-in for the Elasticsearch client.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.k <= MAX_RESULTS:
        parser.error(f"-k must be between 1 and {MAX_RESULTS}")
    if args.engine == "scratch" and (args.index or args.reindex or args.category):
        parser.error("--index, --reindex and --category need --engine es")
    options = SearchOptions(ranker=args.ranker, mode=args.mode, fuzzy=args.fuzzy)
    try:
        if args.engine == "es":
            return _run_elasticsearch(args, options, es_client)
        _run_scratch(args, options)
    except (OSError, ValueError, TypeError) as error:
        # Bad input (a missing file, a malformed line, an overlong query) gets
        # a one-line message, not a traceback.
        print(f"searchengine: error: {error}", file=sys.stderr)
        return 1
    return 0


def _print_suggestion(suggestion: str | None, fuzzy: bool) -> None:
    if suggestion and not fuzzy:
        print(f"Did you mean: {suggestion}?  (add --fuzzy to search for it)\n")


def _run_scratch(args: argparse.Namespace, options: SearchOptions) -> None:
    index = InvertedIndex()
    index.add_many(load_jsonl(args.data))
    # Search first: it checks the query, so a bad one fails before any other work.
    hits = index.search(args.query, k=args.k, options=options)
    _print_suggestion(index.suggest(args.query), args.fuzzy)
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:6.3f}] {hit.title}  (id={hit.doc_id})")
        if args.explain:
            for part in index.explain(args.query, hit.doc_id, options=options):
                terms = " ".join(part.terms)
                print(f"      {terms:<24} idf={part.idf:.3f}  tf={part.tf:g}  -> {part.score:.3f}")


def _run_elasticsearch(
    args: argparse.Namespace, options: SearchOptions, client: Elasticsearch | None
) -> int:
    # Imported here because the elasticsearch package is optional: the
    # from-scratch engine must work without it.
    try:
        from elastic_transport import ConnectionError as TransportConnectionError  # noqa: PLC0415
        from elasticsearch import ApiError  # noqa: PLC0415

        from . import es_engine  # noqa: PLC0415
    except ImportError:
        print(
            "searchengine: error: --engine es needs the elasticsearch package: "
            "pip install 'searchengine-from-scratch[es]'",
            file=sys.stderr,
        )
        return 1
    engine = es_engine.ElasticsearchEngine(client, index=args.index or es_engine.DEFAULT_INDEX)
    try:
        if engine.create_index(recreate=args.reindex):
            _fill(engine, args.data)
        _print_es_results(engine, args, options)
    except TransportConnectionError:
        url = es_engine.default_url()
        print(
            f"searchengine: error: cannot reach Elasticsearch at {url}. "
            "Start it with: docker compose up -d",
            file=sys.stderr,
        )
        return 1
    except (ApiError, es_engine.IndexingError) as error:
        print(f"searchengine: error: Elasticsearch: {error}", file=sys.stderr)
        return 1
    return 0


def _fill(engine: ElasticsearchEngine, data: str) -> None:
    try:
        engine.index_documents(load_jsonl(data))
    except Exception:
        engine.delete_index()  # never leave a half-built index behind
        raise


def _plain(fragment: str) -> str:
    """A highlighted snippet for the terminal: [match] instead of <mark>match</mark>."""
    return html.unescape(fragment.replace("<mark>", "[").replace("</mark>", "]"))


def _print_es_results(
    engine: ElasticsearchEngine, args: argparse.Namespace, options: SearchOptions
) -> None:
    # Search first, as for the from-scratch engine: a bad query fails right away.
    result = engine.search(args.query, k=args.k, options=options, category=args.category)
    _print_suggestion(engine.suggest(args.query), args.fuzzy)
    if not result.hits:
        print("No results.")
    for rank, hit in enumerate(result.hits, start=1):
        print(f"{rank}. [{hit.score:6.3f}] {hit.title}  (id={hit.doc_id})")
        for field, fragments in hit.highlights.items():
            print(f"      {field}: {_plain(fragments[0])}")
        if args.explain:
            explanation = engine.explain(args.query, hit.doc_id, options=options)
            for line in explanation_lines(explanation["explanation"]):
                print(f"      {line}")
    facets = ", ".join(f"{name} ({count})" for name, count in result.facets.items())
    print(f"\n{result.total} matches in {result.took_ms} ms. Categories: {facets or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
