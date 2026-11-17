"""Try the engine from the terminal.

python -m searchengine.cli "inverted index"
python -m searchengine.cli --fuzzy "elastisearch rankng"
python -m searchengine.cli --ranker tfidf '"search engine"'
python -m searchengine.cli --explain "bm25 ranking"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .datasets import SAMPLE, load_jsonl
from .index import InvertedIndex, SearchOptions


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
    args = build_parser().parse_args(argv)
    index = InvertedIndex()
    index.add_many(load_jsonl(args.data))
    options = SearchOptions(ranker=args.ranker, mode=args.mode, fuzzy=args.fuzzy)

    suggestion = index.suggest(args.query)
    if suggestion and not args.fuzzy:
        print(f"Did you mean: {suggestion}?  (add --fuzzy to search for it)\n")

    hits = index.search(args.query, k=args.k, options=options)
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:6.3f}] {hit.title}  (id={hit.doc_id})")
        if args.explain:
            for part in index.explain(args.query, hit.doc_id, options=options):
                terms = " ".join(part.terms)
                print(f"      {terms:<24} idf={part.idf:.3f}  tf={part.tf:g}  -> {part.score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
