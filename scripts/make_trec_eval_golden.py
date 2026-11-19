"""Record trec_eval's own scores in a golden file, for tests/test_evaluation.py.

    pip install pytrec_eval-terrier==0.5.11      # trec_eval, as a Python module
    python scripts/make_trec_eval_golden.py      # writes tests/data/trec_eval.json

trec_eval is the program TREC, BEIR and most IR papers score runs with. This
script makes random judged queries and random rankings (seeded, so the file
comes out the same every time), asks trec_eval to score each ranking and
saves the queries, the rankings and trec_eval's scores. The test then checks
``searchengine.evaluation`` against all of it, without needing trec_eval.
pytrec_eval is only needed here, so it is not a dependency of the project.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent.parent
OUTPUT: Final = ROOT / "tests" / "data" / "trec_eval.json"
SEED: Final = 2026
QUERIES: Final = 200
DOCUMENTS: Final = 40
MAX_GRADE: Final = 3
# Small cutoffs, so that they cut most rankings.
K: Final = 5
DEPTH: Final = 20


def make_cases(rng: random.Random) -> list[dict[str, Any]]:
    """Random graded judgments, and a random ranking of 1 to 40 documents."""
    docs = [f"d{number:02d}" for number in range(DOCUMENTS)]
    cases = []
    for number in range(QUERIES):
        relevant = rng.sample(docs, rng.randint(1, 12))
        grades = {doc: rng.randint(1, MAX_GRADE) for doc in relevant}
        ranking = rng.sample(docs, rng.randint(1, DOCUMENTS))
        cases.append({"id": f"q{number:03d}", "grades": grades, "ranking": ranking})
    return cases


def as_run(rankings: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """trec_eval sorts each ranking by score, so give each rank a lower one."""
    return {
        query_id: {doc: float(len(ranking) - rank) for rank, doc in enumerate(ranking)}
        for query_id, ranking in rankings.items()
    }


def trec_eval_scores(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """trec_eval's nDCG@K, P@K, Recall@DEPTH, MAP@DEPTH and MRR@K, per query."""
    pytrec_eval = importlib.import_module("pytrec_eval")
    qrels = {case["id"]: case["grades"] for case in cases}
    measures = {f"ndcg_cut_{K}", f"P_{K}", f"recall_{DEPTH}", f"map_cut_{DEPTH}"}
    full = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(
        as_run({case["id"]: case["ranking"] for case in cases})
    )
    # trec_eval's reciprocal rank has no cutoff, so give it the top K only.
    top_k = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(
        as_run({case["id"]: case["ranking"][:K] for case in cases})
    )
    return {
        case["id"]: {
            "ndcg": full[case["id"]][f"ndcg_cut_{K}"],
            "precision": full[case["id"]][f"P_{K}"],
            "recall": full[case["id"]][f"recall_{DEPTH}"],
            "map": full[case["id"]][f"map_cut_{DEPTH}"],
            "mrr": top_k[case["id"]]["recip_rank"],
        }
        for case in cases
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    cases = make_cases(random.Random(SEED))  # noqa: S311  (test data, not secrets)
    scores = trec_eval_scores(cases)
    for case in cases:
        case["trec_eval"] = scores[case["id"]]
    golden = {
        "trec_eval": f"pytrec_eval-terrier {version('pytrec_eval-terrier')}",
        "made_by": "scripts/make_trec_eval_golden.py",
        "k": K,
        "depth": DEPTH,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} scored rankings to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
