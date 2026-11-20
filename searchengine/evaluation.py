"""Measuring search quality: did the engine put the right documents first?

A test collection has documents, queries, and relevance judgments: for each
query, the documents a person judged relevant, with a grade (1 = relevant,
2 = highly relevant, and so on). A *run* is the ranked list an engine
returned for each query. The metrics here compare the two.

The definitions follow trec_eval, the tool TREC and the BEIR benchmark use,
so our numbers can be compared with published ones:

- nDCG@k uses the grade itself as the gain, and a log2 discount.
- Recall@k and AP@k divide by all of the query's relevant documents.
- A judged query that the run has no answer for scores 0, as with
  ``trec_eval -c``: an engine must not look better by failing.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .index import MAX_RESULTS, normalize_doc_id

MAX_GRADE: Final = 100  # relevance grades are small integers, like 1 to 3
MAX_JUDGED_QUERIES: Final = 100_000
MAX_TRIALS: Final = 1_000_000  # for the randomization test
DEFAULT_TRIALS: Final = 10_000


@dataclass(frozen=True, slots=True)
class JudgedQuery:
    """A query, with the documents judged relevant to it and their grades."""

    id: str
    text: str
    grades: Mapping[str, int]  # document id -> grade, 1 or more


def judged_query(query_id: object, text: object, grades: object) -> JudgedQuery:
    """Check the parts of a judged query and build it."""
    if isinstance(query_id, bool) or not isinstance(query_id, (str, int)) or str(query_id) == "":
        raise ValueError("a query id must be a non-empty str or an int")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a judged query needs non-empty text")
    if not isinstance(grades, Mapping) or not grades:
        raise ValueError("a judged query needs at least one relevant document")
    checked: dict[str, int] = {}
    for doc_id, grade in grades.items():
        if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= MAX_GRADE:
            raise ValueError(f"relevance grades must be integers from 1 to {MAX_GRADE}")
        checked[normalize_doc_id(doc_id)] = grade
    return JudgedQuery(str(query_id), text, MappingProxyType(checked))


@dataclass(frozen=True, slots=True)
class QueryScores:
    """How well one ranking did. Each score is between 0 and 1."""

    ndcg: float  # nDCG@k: graded, and position-aware
    mrr: float  # reciprocal rank of the first relevant document within k
    precision: float  # share of the top k that is relevant
    recall: float  # share of the relevant documents in the top `depth`
    map: float  # average precision over the top `depth`


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The scores of a whole run: per query, and their means."""

    k: int
    depth: int
    per_query: Mapping[str, QueryScores]
    mean: QueryScores


def _check_ranking(ranked: Sequence[str], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"the cutoff must be an int from 1 to {MAX_RESULTS}")
    if len(set(ranked)) != len(ranked):
        raise ValueError("a ranking must not list a document twice")


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain: rank r counts 1 / log2(r + 1)."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """nDCG@k: the DCG of the top k, divided by the best DCG possible."""
    _check_ranking(ranked, k)
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal <= 0:
        raise ValueError("nDCG needs at least one relevant document")
    return dcg([grades.get(doc_id, 0) for doc_id in ranked[:k]]) / ideal


def reciprocal_rank(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """1 / the rank of the first relevant document in the top k, or 0."""
    _check_ranking(ranked, k)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if grades.get(doc_id, 0) > 0:
            return 1 / rank
    return 0.0


def precision(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """The share of the top k that is relevant. Missing results count as wrong."""
    _check_ranking(ranked, k)
    return sum(1 for doc_id in ranked[:k] if grades.get(doc_id, 0) > 0) / k


def recall(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """The share of all relevant documents that made the top k."""
    _check_ranking(ranked, k)
    relevant = sum(1 for grade in grades.values() if grade > 0)
    if relevant == 0:
        raise ValueError("recall needs at least one relevant document")
    return sum(1 for doc_id in ranked[:k] if grades.get(doc_id, 0) > 0) / relevant


def average_precision(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """The precision at each relevant document in the top k, summed, divided
    by the number of relevant documents (so a missed one counts as 0)."""
    _check_ranking(ranked, k)
    relevant = sum(1 for grade in grades.values() if grade > 0)
    if relevant == 0:
        raise ValueError("average precision needs at least one relevant document")
    found = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if grades.get(doc_id, 0) > 0:
            found += 1
            total += found / rank
    return total / relevant


def score_query(
    ranked: Sequence[str], grades: Mapping[str, int], k: int, depth: int
) -> QueryScores:
    """All the metrics for one query's ranking."""
    return QueryScores(
        ndcg=ndcg(ranked, grades, k),
        mrr=reciprocal_rank(ranked, grades, k),
        precision=precision(ranked, grades, k),
        recall=recall(ranked, grades, depth),
        map=average_precision(ranked, grades, depth),
    )


def mean_scores(scores: Sequence[QueryScores]) -> QueryScores:
    """The average of each metric over the queries."""
    if not scores:
        raise ValueError("nothing to average: no queries")
    return QueryScores(
        ndcg=statistics.fmean(score.ndcg for score in scores),
        mrr=statistics.fmean(score.mrr for score in scores),
        precision=statistics.fmean(score.precision for score in scores),
        recall=statistics.fmean(score.recall for score in scores),
        map=statistics.fmean(score.map for score in scores),
    )


def evaluate(
    run: Mapping[str, Sequence[str]],
    queries: Sequence[JudgedQuery],
    *,
    k: int = 10,
    depth: int = 100,
) -> Evaluation:
    """Score a run (query id -> ranked document ids) against judged queries.

    Every judged query counts, including those the run has no ranking for.
    """
    if not queries or len(queries) > MAX_JUDGED_QUERIES:
        raise ValueError(f"evaluate needs 1 to {MAX_JUDGED_QUERIES} judged queries")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("judged query ids must be unique")
    if k > depth:
        raise ValueError("k must not be larger than depth")
    per_query = {
        query.id: score_query(run.get(query.id, ()), query.grades, k, depth) for query in queries
    }
    return Evaluation(k, depth, MappingProxyType(per_query), mean_scores(list(per_query.values())))


def randomization_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization test: the p-value for "no difference".

    If the two systems were equally good, each query's pair of scores could
    just as well be swapped. Swap at random many times and count how often
    the mean difference is at least as large as the one observed. A small
    p-value (below 0.05, say) means the difference is unlikely to be luck.
    The seed makes the answer reproducible.
    """
    if len(first) != len(second) or not first or len(first) > MAX_JUDGED_QUERIES:
        raise ValueError("the test needs two equally long, non-empty lists of scores")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= MAX_TRIALS:
        raise ValueError(f"trials must be an int from 1 to {MAX_TRIALS}")
    differences = [a - b for a, b in zip(first, second, strict=True)]
    if not all(math.isfinite(diff) for diff in differences):
        raise ValueError("the scores must be finite numbers")
    observed = abs(math.fsum(differences))
    tolerance = 1e-9 * max(1.0, observed)  # do not let rounding decide a tie
    rng = random.Random(seed)  # noqa: S311 - a statistics test, not cryptography
    extreme = 0
    for _ in range(trials):
        signs = format(rng.getrandbits(len(differences)), f"0{len(differences)}b")
        pairs = zip(signs, differences, strict=True)  # one random sign per query
        flipped = math.fsum(diff if sign == "1" else -diff for sign, diff in pairs)
        if abs(flipped) >= observed - tolerance:
            extreme += 1
    # Counting the observed order as one of the trials keeps p above 0.
    return (extreme + 1) / (trials + 1)
