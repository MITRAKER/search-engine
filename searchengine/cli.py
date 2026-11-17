"""Try the engine from the terminal:

    python -m searchengine.cli "inverted index"
    python -m searchengine.cli --fuzzy "elastisearch rankng"
    python -m searchengine.cli --ranker tfidf '"search engine"'
"""
import argparse

from .datasets import SAMPLE, load_jsonl
from .index import InvertedIndex


def main() -> None:
    p = argparse.ArgumentParser(description="Search the sample dataset")
    p.add_argument("query")
    p.add_argument("--data", default=SAMPLE)
    p.add_argument("--ranker", choices=["bm25", "tfidf"], default="bm25")
    p.add_argument("--mode", choices=["or", "and"], default="or")
    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    index = InvertedIndex()
    index.add_many(load_jsonl(args.data))

    suggestion = index.suggest(args.query)
    if suggestion and not args.fuzzy:
        print(f"Did you mean: {suggestion}?  (add --fuzzy to use it)\n")

    hits = index.search(args.query, k=args.k, ranker=args.ranker, mode=args.mode, fuzzy=args.fuzzy)
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. [{hit.score:6.3f}] {hit.title}  (id={hit.doc_id})")


if __name__ == "__main__":
    main()
