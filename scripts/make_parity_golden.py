"""Record Elasticsearch's own scores in a golden file, for tests/test_es_parity.py.

    docker compose up -d                     # Elasticsearch on localhost:9200
    python scripts/make_parity_golden.py     # writes tests/data/es_parity.json

It builds a small corpus with short, long and empty fields, indexes it in
Elasticsearch with the settings, mappings and queries of the Elasticsearch
engine (``searchengine/es_dsl.py``), runs a set of queries and saves the
corpus, the queries and Elasticsearch's scores.
It also saves the terms Elasticsearch's copy of the analyzer makes from some
tricky text. The test then checks our engine against all of it, without
needing Elasticsearch. Besides the standard library it only needs this
project, installed for development (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from searchengine.es_dsl import ANALYZER, build_query, index_mappings, index_settings
from searchengine.index import SearchOptions

ROOT: Final = Path(__file__).resolve().parent.parent
OUTPUT: Final = ROOT / "tests" / "data" / "es_parity.json"
INDEX: Final = "parity-golden"
SEED: Final = 2026
VOCABULARY: Final = (
    *("search", "engine", "index", "rank", "query", "score", "token", "field", "lucene"),
    *("shard", "segment", "posting", "phrase", "fuzzy", "stem", "filter", "facet"),
    *("cluster", "node", "replica", "vector", "embed", "cache", "merge", "commit"),
    *("refresh", "analyzer", "boost", "match", "relevance", "recall", "precision"),
    *("latency", "throughput", "document", "corpus", "spell", "suggest", "highlight"),
    "snippet",
)
# Body lengths around Lucene's rounding edges (exact up to 40), then long ones.
BODY_LENGTHS: Final = (
    *(0, 0, 1, 2, 5, 10, 23, 24, 25, 39, 40, 41, 42, 43, 56, 57, 60, 75, 100, 101),
    *(130, 200, 255, 256, 300, 375, 376, 500, 640, 800, 1000, 1200),
)
# Text where Python and Java could disagree: casing rules, combining marks,
# characters outside the Basic Multilingual Plane and the token length limit.
BOLD_A: Final = "\N{MATHEMATICAL BOLD CAPITAL A}"  # 2 UTF-16 units in Java
ANALYZER_TEXTS: Final = (
    "The Runners are RUNNING! It's 2026.",
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}STANBUL and \N{LATIN SMALL LETTER DOTLESS I}",
    "\N{GREEK CAPITAL LETTER DELTA}\N{GREEK CAPITAL LETTER SIGMA} \N{GREEK CAPITAL LETTER SIGMA}",
    "\N{LATIN CAPITAL LETTER DZ WITH CARON} \N{LATIN CAPITAL LETTER SHARP S} Stra\xdfe",
    "\N{LATIN SMALL LIGATURE FI}nance \N{KELVIN SIGN} \N{OHM SIGN} \N{ROMAN NUMERAL TWELVE}",
    "hindi \u0939\u093f\u0928\u094d\u0926\u0940 \u092d\u093e\u0937\u093e",
    "arabic \u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627",
    "cafe\N{COMBINING ACUTE ACCENT} caf\xe9 \N{COMBINING ACUTE ACCENT}x",
    "snake_case kebab-case e-mail 3.14 \u0661\u0662\u0663 \u09e8\u09e6\u09e8\u09ec",
    "zero\N{ZERO WIDTH JOINER}width no\N{ZERO WIDTH NON-JOINER}join soft\N{SOFT HYPHEN}hyphen",
    "emoji \N{THUMBS UP SIGN}\N{EMOJI MODIFIER FITZPATRICK TYPE-4} word",
    "\N{BRAHMI SIGN CANDRABINDU}x \N{BRAHMI LETTER KA}\N{BRAHMI VOWEL SIGN AA}",
    "\u4e2d\u6587\u641c\u7d22 \u3053\u3093\u306b\u3061\u306f \ud55c\uad6d\uc5b4",
    "THE AND Not with",
    BOLD_A * 127 + " kept",  # 254 UTF-16 units
    BOLD_A * 128 + " dropped",  # 256 UTF-16 units
    "x" * 255 + " " + "y" * 256 + " end",
    # Letters beyond U+FFFF are two UTF-16 units, and Lucene's stemmer counts units.
    "\N{CJK UNIFIED IDEOGRAPH-20BB7}s " + BOLD_A + "s bo\N{MATHEMATICAL BOLD SMALL B}ing",
    "ha" + "\N{MATHEMATICAL BOLD SMALL B}" * 2 + "ing",
)
WEIGHTINGS: Final = (
    MappingProxyType({"title": 3.0, "body": 1.0}),
    MappingProxyType({"body": 1.0}),
    MappingProxyType({"title": 1.0}),
    MappingProxyType({"title": 1.7, "body": 1.3}),
    MappingProxyType({"title": 2.5, "body": 1.0}),
)


def request(url: str, method: str, body: object = None) -> Any:
    """One JSON request to Elasticsearch."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url}")
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310  (checked above)
        return json.loads(response.read())


def make_corpus(rng: random.Random) -> list[dict[str, str]]:
    """Documents whose words follow a Zipf-like distribution, with stopwords
    mixed in so that phrases have position gaps."""
    weights = [1 / (rank + 1) for rank in range(len(VOCABULARY))]
    lengths = list(BODY_LENGTHS) + [rng.randint(1, 1200) for _ in range(10)]
    corpus = []
    for number, body_length in enumerate(lengths):
        title_length = 0 if number % 6 == 0 else rng.randint(1, 12)
        title = rng.choices(VOCABULARY, weights=weights, k=title_length)
        body = rng.choices(VOCABULARY, weights=weights, k=body_length)
        body = [word if rng.random() > 0.15 else f"the {word}" for word in body]
        corpus.append({"id": f"d{number:02d}", "title": " ".join(title), "body": " ".join(body)})
    return corpus


def make_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for word in ("search", "lucene", "replica", "snippet", "throughput"):
        cases += [{"query": word, "weights": dict(w), "mode": "or"} for w in WEIGHTINGS]
    for query in ("search engine", "rank score latency", "fuzzy stem suggest highlight"):
        cases.append({"query": query, "weights": dict(WEIGHTINGS[0]), "mode": "or"})
        cases.append({"query": query, "weights": dict(WEIGHTINGS[3]), "mode": "and"})
    phrases = ('"search engine"', '"index rank"', '"search the engine"', '"search engine index"')
    for query in phrases:
        cases.append({"query": query, "weights": {"body": 1.0}, "mode": "or"})
    cases.append({"query": '"search engine"', "weights": {"title": 1.0}, "mode": "or"})
    return cases


def es_query(case: dict[str, Any]) -> dict[str, Any]:
    """The query the Elasticsearch engine sends for this case."""
    return build_query(case["query"], SearchOptions(mode=case["mode"]), case["weights"])


def analyzer_cases(url: str) -> list[list[Any]]:
    """Each tricky text with the terms and positions Elasticsearch makes of it."""
    cases: list[list[Any]] = []
    for text in ANALYZER_TEXTS:
        body = {"analyzer": ANALYZER, "text": text}
        tokens = request(f"{url}/{INDEX}/_analyze", "POST", body)["tokens"]
        cases.append([text, [[token["token"], token["position"]] for token in tokens]])
    return cases


def create_index(url: str, corpus: list[dict[str, str]]) -> None:
    settings, mappings = index_settings(), index_mappings(("title", "body"))
    request(f"{url}/{INDEX}?ignore_unavailable=true", "DELETE")
    request(f"{url}/{INDEX}", "PUT", {"settings": settings, "mappings": mappings})
    lines = []
    for doc in corpus:
        lines.append(json.dumps({"create": {"_index": INDEX, "_id": doc["id"]}}))
        lines.append(json.dumps({"title": doc["title"], "body": doc["body"]}))
    req = urllib.request.Request(  # noqa: S310  (url checked in request())
        f"{url}/_bulk?refresh=true",
        data=("\n".join(lines) + "\n").encode(),
        method="POST",
        headers={"Content-Type": "application/x-ndjson"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
        if json.loads(response.read())["errors"]:
            raise RuntimeError("bulk indexing failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:9200")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    info = request(args.url, "GET")
    corpus = make_corpus(random.Random(SEED))  # noqa: S311  (test data, not secrets)
    create_index(args.url, corpus)
    cases = make_cases()
    for case in cases:
        body = {"query": es_query(case), "size": len(corpus)}
        hits = request(f"{args.url}/{INDEX}/_search", "POST", body)["hits"]["hits"]
        case["hits"] = [[hit["_id"], hit["_score"]] for hit in hits]
    analyzer = analyzer_cases(args.url)
    request(f"{args.url}/{INDEX}", "DELETE")
    golden = {
        "elasticsearch": info["version"]["number"],
        "lucene": info["version"]["lucene_version"],
        "made_by": "scripts/make_parity_golden.py",
        "corpus": corpus,
        "cases": cases,
        "analyzer": analyzer,
    }
    Path(args.output).write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases over {len(corpus)} documents to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
