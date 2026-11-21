"""Try the engine from the terminal.

searchengine "inverted index"
searchengine --fuzzy "elastisearch rankng"
searchengine --ranker tfidf '"search engine"'
searchengine --explain "bm25 ranking"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .datasets import SAMPLE, load_jsonl
from .index import MAX_RESULTS, InvertedIndex, SearchOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="searchengine", description="Search a JSON Lines file.")
    parser.add_argument("query", help='words to search for; wrap in "double quotes" for a phrase')
    parser.add_argument("--data", default=str(SAMPLE), help="JSON Lines file to index")
    parser.add_argument("--ranker", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--mode", choices=["or", "and"], default="or")
    parser.add_argument("--fuzzy", action="store_true", help="fix typos in the query")
    parser.add_argument("--explain", action="store_true", help="show why each result scored")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit status: 0 for success."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.k <= MAX_RESULTS:
        parser.error(f"-k must be between 1 and {MAX_RESULTS}")
    options = SearchOptions(ranker=args.ranker, mode=args.mode, fuzzy=args.fuzzy)
    try:
        _run(args, options)
    except (OSError, ValueError, TypeError) as error:
        # Bad input (a missing file, a malformed line, an overlong query) gets
        # a one-line message, not a traceback.
        print(f"searchengine: error: {error}", file=sys.stderr)
        return 1
    return 0


def _run(args: argparse.Namespace, options: SearchOptions) -> None:
    index = InvertedIndex()
    index.add_many(load_jsonl(args.data))
    # Search first: it checks the query, so a bad one fails before any other work.
    hits = index.search(args.query, k=args.k, options=options)
    suggestion = index.suggest(args.query)
    if suggestion and not args.fuzzy:
        print(f"Did you mean: {suggestion}?  (add --fuzzy to search for it)\n")
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:6.3f}] {hit.title}  (id={hit.doc_id})")
        if args.explain:
            for part in index.explain(args.query, hit.doc_id, options=options):
                terms = " ".join(part.terms)
                print(f"      {terms:<24} idf={part.idf:.3f}  tf={part.tf:g}  -> {part.score:.3f}")


if __name__ == "__main__":
    sys.exit(main())
