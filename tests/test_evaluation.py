"""Search-quality metrics and the significance test (searchengine.evaluation).

The metrics are checked three ways: on small rankings worked out by hand,
against trec_eval's own scores for 200 random rankings, and as properties
that must hold for any ranking. tests/data/trec_eval.json holds the rankings
and trec_eval's scores (recorded by scripts/make_trec_eval_golden.py), so no
trec_eval is needed here. The randomization test is checked against p-values
counted exactly over every way to swap the pairs of scores.
"""

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, astuple
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from searchengine.evaluation import (
    DEFAULT_TRIALS,
    MAX_GRADE,
    MAX_TRIALS,
    JudgedQuery,
    QueryScores,
    average_precision,
    dcg,
    evaluate,
    judged_query,
    mean_scores,
    ndcg,
    precision,
    randomization_test,
    recall,
    reciprocal_rank,
    score_query,
)
from searchengine.index import MAX_RESULTS

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "tests" / "data" / "trec_eval.json").read_text("utf-8"))
CASES = GOLDEN["cases"]
# trec_eval computes in doubles too: every score here agrees to about 1e-16.
TOLERANCE = 1e-12

METRICS = [ndcg, reciprocal_rank, precision, recall, average_precision]

# The worked example: b (grade 2), d and e (grade 1) are relevant; e was not found.
RANKING = ["a", "b", "c", "d"]
GRADES = {"b": 2, "d": 1, "e": 1}

# Ten queries' nDCG for two systems. A is ahead on seven of them.
SYSTEM_A = ["0.81", "0.62", "0.45", "0.90", "0.33", "0.71", "0.58", "0.66", "0.49", "0.77"]
SYSTEM_B = ["0.74", "0.65", "0.31", "0.86", "0.38", "0.52", "0.60", "0.49", "0.47", "0.70"]


def judged(query_id: str, grades: dict[str, int]) -> JudgedQuery:
    return judged_query(query_id, f"text of {query_id}", grades)


def floats(scores: Sequence[str]) -> list[float]:
    return [float(score) for score in scores]


# ---------- judged queries ----------


def test_judged_query_keeps_the_query_and_its_grades():
    query = judged_query("q1", "what is BM25?", {"d1": 2, "d2": 1})
    assert query == JudgedQuery("q1", "what is BM25?", {"d1": 2, "d2": 1})


@pytest.mark.parametrize(("query_id", "stored"), [("q1", "q1"), (7, "7"), (0, "0"), (-2, "-2")])
def test_query_ids_may_be_strs_or_ints_and_are_kept_as_strs(query_id, stored):
    assert judged_query(query_id, "text", {"d1": 1}).id == stored


@pytest.mark.parametrize("query_id", ["", True, False, None, 1.5, b"q1", ["q1"]])
def test_query_ids_must_be_a_non_empty_str_or_an_int(query_id):
    with pytest.raises(ValueError, match="query id"):
        judged_query(query_id, "text", {"d1": 1})


@pytest.mark.parametrize("text", ["", " ", "\t\n", None, 42, b"what is BM25?", ["BM25"]])
def test_judged_queries_need_text_that_is_not_blank(text):
    with pytest.raises(ValueError, match="non-empty text"):
        judged_query("q1", text, {"d1": 1})


@pytest.mark.parametrize("grades", [{}, MappingProxyType({}), None, [("d1", 1)], "d1", {"d1"}])
def test_judged_queries_need_a_non_empty_mapping_of_grades(grades):
    with pytest.raises(ValueError, match="at least one relevant document"):
        judged_query("q1", "text", grades)


def test_any_mapping_of_grades_will_do():
    assert judged_query("q1", "text", MappingProxyType({"d1": 3})).grades == {"d1": 3}


@pytest.mark.parametrize("grade", [1, 2, MAX_GRADE])
def test_grades_from_1_to_max_grade_are_kept(grade):
    assert judged_query("q1", "text", {"d1": grade}).grades == {"d1": grade}


# Grade 0 would mean "not relevant": such documents are simply left out.
@pytest.mark.parametrize("grade", [0, -1, MAX_GRADE + 1, True, 1.5, "1", None])
def test_other_grades_are_rejected(grade):
    with pytest.raises(ValueError, match=f"integers from 1 to {MAX_GRADE}"):
        judged_query("q1", "text", {"d1": 1, "d2": grade})  # every grade is checked


# An index stores the int id 4983 as "4983", so the judgments must name it so too.
@pytest.mark.parametrize(
    ("doc_id", "stored"),
    [("d1", "d1"), (4983, "4983"), (0, "0"), (" d1 ", " d1 "), ("é" * 256, "é" * 256)],
)
def test_document_ids_are_normalized_like_index_ids(doc_id, stored):
    assert judged_query("q1", "text", {doc_id: 1}).grades == {stored: 1}


@pytest.mark.parametrize(
    ("doc_id", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (None, TypeError),
        ("", ValueError),
        ("x" * 513, ValueError),
        ("é" * 257, ValueError),  # 257 characters, but 514 bytes: Elasticsearch allows 512
    ],
)
def test_document_ids_are_checked_like_index_ids(doc_id, error):
    with pytest.raises(error, match="document id"):
        judged_query("q1", "text", {doc_id: 1})


def test_grades_are_a_read_only_copy():
    grades = {"d1": 1}
    query = judged_query("q1", "text", grades)
    grades["d2"] = 2  # changing the caller's dict afterwards...
    assert query.grades == {"d1": 1}  # ...does not change the judged query
    with pytest.raises(TypeError, match="does not support item assignment"):
        query.grades["d3"] = 3  # type: ignore[index]


# ---------- the metrics, worked out by hand ----------


def test_dcg_discounts_rank_r_by_log2_of_r_plus_1():
    assert dcg([]) == 0
    assert dcg([3]) == 3  # rank 1 divides by log2(2) = 1: no discount
    assert dcg([0, 0, 4]) == 2  # rank 3 divides by log2(4) = 2
    # Ranks 1 to 3: 3 / log2(2) + 2 / log2(3) + 1 / log2(4) = 3 + 1.26186 + 0.5
    assert dcg([3, 2, 1]) == pytest.approx(3 + 2 / math.log2(3) + 1 / 2)


def test_ndcg_by_hand():
    # The gains of a, b, c are 0, 2, 0: DCG@3 is 2 / log2(3) = 1.26186. The best
    # ranking would show the grades 2, 1, 1, for a DCG@3 of
    # 2 / log2(2) + 1 / log2(3) + 1 / log2(4), which is 2 + 0.63093 + 0.5 = 3.13093.
    best = 2 + 1 / math.log2(3) + 1 / 2
    assert ndcg(RANKING, GRADES, 3) == pytest.approx((2 / math.log2(3)) / best)
    assert round(ndcg(RANKING, GRADES, 3), 5) == 0.40303
    # At 4, d adds 1 / log2(5). The best ranking has only three gains to show,
    # so its DCG@4 stays 3.13093: (1.26186 + 0.43068) / 3.13093 = 0.54059.
    found = 2 / math.log2(3) + 1 / math.log2(5)
    assert ndcg(RANKING, GRADES, 4) == pytest.approx(found / best)
    assert round(ndcg(RANKING, GRADES, 4), 5) == 0.54059
    assert ndcg(RANKING, GRADES, 1) == 0  # a is not relevant


def test_reciprocal_rank_precision_recall_and_average_precision_by_hand():
    assert reciprocal_rank(RANKING, GRADES, 3) == pytest.approx(1 / 2)  # b, at rank 2, is first
    assert reciprocal_rank(RANKING, GRADES, 1) == 0  # and no relevant document is in the top 1
    assert precision(RANKING, GRADES, 3) == pytest.approx(1 / 3)  # b, of a, b and c
    assert recall(RANKING, GRADES, 4) == pytest.approx(2 / 3)  # b and d, of b, d and e
    # Precision 1/2 at b and 2/4 at d, and e, never found, adds 0; over 3 relevant.
    assert average_precision(RANKING, GRADES, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_average_precision_textbook_example():
    ranking = ["r1", "x1", "r2", "x2", "x3", "r3"]
    grades = {"r1": 1, "r2": 1, "r3": 1}
    # Relevant at ranks 1, 3 and 6, of 3 relevant: 1/1, 2/3 and 3/6, summed and over 3.
    assert average_precision(ranking, grades, 6) == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)
    assert round(average_precision(ranking, grades, 6), 5) == 0.72222
    # Cut at 5, r3 is missed but still counted: 1/1 and 2/3, over 3, is 0.55556.
    assert average_precision(ranking, grades, 5) == pytest.approx((1 + 2 / 3) / 3)


def test_precision_divides_by_k_even_when_the_ranking_is_shorter():
    assert precision(["b"], GRADES, 5) == pytest.approx(1 / 5)  # 4 missing results count as wrong


def test_recall_and_average_precision_divide_by_every_relevant_document():
    grades = {"b": 1, "x": 1, "y": 1, "z": 1}
    assert recall(["b"], grades, 10) == pytest.approx(1 / 4)
    assert average_precision(["b"], grades, 10) == pytest.approx(1 / 4)  # x, y and z add 0


def test_only_ndcg_uses_the_grade_itself():
    flat = {"b": 1, "d": 1, "e": 1}  # the same documents, all with grade 1
    for metric in (reciprocal_rank, precision, recall, average_precision):
        assert metric(RANKING, flat, 4) == metric(RANKING, GRADES, 4)
    # Gains 1 at ranks 2 and 4, against the best 1, 1, 1: 1.06161 / 2.13093.
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 1 / 2)
    assert ndcg(RANKING, flat, 4) == pytest.approx(expected)


def test_a_grade_of_0_means_not_relevant():
    # trec_eval's judgment files list judged, non-relevant documents with grade 0.
    ranking = ["z", "b"]
    grades = {"z": 0, "b": 1}
    assert reciprocal_rank(ranking, grades, 2) == pytest.approx(1 / 2)
    assert precision(ranking, grades, 2) == pytest.approx(1 / 2)
    assert recall(ranking, grades, 2) == 1  # one relevant document, not two
    assert average_precision(ranking, grades, 2) == pytest.approx(1 / 2)
    assert ndcg(ranking, grades, 2) == pytest.approx(1 / math.log2(3))  # the best is 1 at rank 1


def test_precision_and_reciprocal_rank_need_no_relevant_documents():
    assert precision(["a", "b"], {}, 2) == 0
    assert reciprocal_rank(["a", "b"], {"a": 0}, 2) == 0


def test_score_query_cuts_the_top_k_metrics_at_k_and_the_others_at_depth():
    # The one relevant document is at rank 4: past k = 3, but within depth = 5.
    scores = score_query(["x1", "x2", "x3", "r", "x4"], {"r": 1}, 3, 5)
    assert scores == QueryScores(ndcg=0, mrr=0, precision=0, recall=1, map=1 / 4)


# ---------- what the metrics accept ----------


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1, True, 2.0, None, "3"])
def test_every_metric_checks_the_cutoff(metric, k):
    with pytest.raises(ValueError, match=f"cutoff must be an int from 1 to {MAX_RESULTS}"):
        metric(["a"], {"a": 1}, k)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("k", [1, MAX_RESULTS])
def test_every_metric_takes_cutoffs_from_1_to_max_results(metric, k):
    assert 0 < metric(["a"], {"a": 1}, k) <= 1


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("ranking", [["a", "a"], ["a", "b", "c", "b"]])
def test_every_metric_rejects_a_ranking_that_lists_a_document_twice(metric, ranking):
    with pytest.raises(ValueError, match="twice"):
        metric(ranking, {"a": 1}, 2)  # anywhere, even past the cutoff: the engine is broken


@pytest.mark.parametrize("metric", [ndcg, recall, average_precision])
@pytest.mark.parametrize("grades", [{}, {"a": 0}])
def test_metrics_that_divide_by_the_relevant_documents_need_one(metric, grades):
    with pytest.raises(ValueError, match="needs at least one relevant document"):
        metric(["a"], grades, 10)


# ---------- trec_eval's own scores ----------


def test_the_golden_file_is_from_trec_eval_and_not_trivial():
    assert GOLDEN["trec_eval"].startswith("pytrec_eval")
    assert (ROOT / GOLDEN["made_by"]).is_file()
    assert (GOLDEN["k"], GOLDEN["depth"]) == (5, 20)
    assert len(CASES) == 200
    for metric in ("ndcg", "precision", "recall", "map", "mrr"):
        scores = [case["trec_eval"][metric] for case in CASES]
        assert sum(score > 0 for score in scores) >= 100, metric  # a scorer of 0s fails
        assert any(0 < score < 1 for score in scores), metric  # so does one of 0s and 1s
    # The cutoffs and the grades matter: rankings shorter than k, relevant
    # documents only found past depth, grades from 1 to 3.
    assert any(len(case["ranking"]) < GOLDEN["k"] for case in CASES)
    assert any(set(case["ranking"][GOLDEN["depth"] :]) & case["grades"].keys() for case in CASES)
    assert {grade for case in CASES for grade in case["grades"].values()} == {1, 2, 3}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_scores_match_trec_eval(case):
    scores = score_query(case["ranking"], case["grades"], GOLDEN["k"], GOLDEN["depth"])
    # The same names: ndcg, precision, recall, map and mrr.
    assert asdict(scores) == pytest.approx(case["trec_eval"], abs=TOLERANCE)


# ---------- evaluating a whole run ----------


def test_a_query_missing_from_the_run_scores_0_and_still_counts():
    queries = [judged("q1", {"a": 1}), judged("q2", {"b": 1})]
    result = evaluate({"q1": ["a"]}, queries, k=1, depth=1)
    assert result.per_query["q1"] == QueryScores(1, 1, 1, 1, 1)
    assert result.per_query["q2"] == QueryScores(0, 0, 0, 0, 0)  # as with trec_eval -c
    assert result.mean == QueryScores(0.5, 0.5, 0.5, 0.5, 0.5)  # an engine gains nothing by failing


def test_the_mean_is_the_mean_of_the_per_query_scores():
    queries = [judged("q1", {"b": 1}), judged("q2", {"c": 1})]
    result = evaluate({"q1": ["a", "b"], "q2": ["c"]}, queries, k=2, depth=2)
    # q1 finds b at rank 2: nDCG 1/log2(3), RR 1/2, P@2 1/2, recall 1 and AP 1/2.
    # q2 finds c at rank 1: 1 for everything but P@2, which is 1/2 (one result, two slots).
    expected = ((1 / math.log2(3) + 1) / 2, (1 / 2 + 1) / 2, 1 / 2, 1, (1 / 2 + 1) / 2)
    assert astuple(result.mean) == pytest.approx(expected)
    assert list(result.per_query) == ["q1", "q2"]


def test_rankings_for_queries_nobody_judged_are_ignored():
    result = evaluate({"q1": ["a"], "q9": ["a", "b"]}, [judged("q1", {"a": 1})], k=1, depth=1)
    assert list(result.per_query) == ["q1"]
    assert result.mean == QueryScores(1, 1, 1, 1, 1)


def test_evaluate_keeps_its_cutoffs_which_default_to_10_and_100():
    queries = [judged("q1", {"a": 1})]
    default = evaluate({}, queries)
    assert (default.k, default.depth) == (10, 100)
    result = evaluate({}, queries, k=3, depth=7)
    assert (result.k, result.depth) == (3, 7)


def test_evaluate_checks_every_ranking_and_cutoff():
    queries = [judged("q1", {"a": 1})]
    with pytest.raises(ValueError, match="twice"):
        evaluate({"q1": ["a", "b", "a"]}, queries)
    with pytest.raises(ValueError, match="cutoff"):
        evaluate({"q1": ["a"]}, queries, k=0)
    with pytest.raises(ValueError, match="cutoff"):
        evaluate({"q1": ["a"]}, queries, depth=MAX_RESULTS + 1)


def test_judged_query_ids_must_be_unique():
    queries = [judged("q1", {"a": 1}), judged("q1", {"b": 1})]
    with pytest.raises(ValueError, match="unique"):
        evaluate({}, queries)


def test_k_must_not_be_larger_than_depth():
    queries = [judged("q1", {"a": 1})]
    with pytest.raises(ValueError, match="larger than depth"):
        evaluate({"q1": ["a"]}, queries, k=11, depth=10)
    assert evaluate({"q1": ["a"]}, queries, k=10, depth=10).mean.ndcg == 1  # equal is fine


def test_evaluate_needs_1_to_max_judged_queries(monkeypatch):
    with pytest.raises(ValueError, match="judged queries"):
        evaluate({}, [])
    monkeypatch.setattr("searchengine.evaluation.MAX_JUDGED_QUERIES", 2)
    queries = [judged(f"q{number}", {"a": 1}) for number in range(3)]
    assert len(evaluate({}, queries[:2]).per_query) == 2  # exactly the limit is fine
    with pytest.raises(ValueError, match="1 to 2 judged queries"):
        evaluate({}, queries)


def test_per_query_scores_are_read_only():
    result = evaluate({}, [judged("q1", {"a": 1})])
    with pytest.raises(TypeError, match="does not support item assignment"):
        result.per_query["q2"] = result.mean  # type: ignore[index]


def test_mean_scores_averages_each_metric_on_its_own():
    first = QueryScores(ndcg=0.2, mrr=0.4, precision=0.6, recall=0.8, map=1.0)
    second = QueryScores(ndcg=0.0, mrr=0.2, precision=0.4, recall=0.6, map=0.8)
    assert astuple(mean_scores([first, second])) == pytest.approx((0.1, 0.3, 0.5, 0.7, 0.9))


def test_mean_scores_needs_at_least_one_query():
    with pytest.raises(ValueError, match="no queries"):
        mean_scores([])


# ---------- is the difference real? The paired randomization test ----------


def exact_p_value(first: Sequence[str], second: Sequence[str]) -> Fraction:
    """The p-value over all 2**n ways to swap the pairs, in exact decimal arithmetic."""
    differences = [Fraction(a) - Fraction(b) for a, b in zip(first, second, strict=True)]
    observed = abs(sum(differences, Fraction(0)))
    patterns = list(itertools.product((1, -1), repeat=len(differences)))
    swapped = [
        sum((sign * diff for sign, diff in zip(signs, differences, strict=True)), Fraction(0))
        for signs in patterns
    ]
    return Fraction(sum(abs(total) >= observed for total in swapped), len(patterns))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Mathematically 0.1 + 0.2 = 0.3, so 6 of the 16 ways to swap the pairs
        # tie with the observed difference (exact p = 12/16). In floats 0.1 + 0.2
        # is not 0.3, and two of the ties come out 1e-16 short: only a tolerance
        # keeps rounding from deciding them.
        (["0.3", "0.1", "0.2", "0"], ["0", "0", "0", "0.3"]),
        (SYSTEM_A, SYSTEM_B),  # exact p = 31/512, about 0.06
        # Binary fractions, exact in floats, and a query where both systems agree.
        (
            ["1", "0.5", "0", "0.25", "1", "0.75", "0.5", "0"],
            ["0.5", "0.5", "0.25", "0", "0.5", "1", "0", "0.25"],
        ),
    ],
)
def test_the_p_value_estimates_the_exact_one(first, second):
    exact = float(exact_p_value(first, second))
    trials = 4000
    estimate = randomization_test(floats(first), floats(second), trials=trials)
    # An estimate from n trials has a standard error of sqrt(p (1 - p) / n). Allow 4.
    assert estimate == pytest.approx(exact, abs=4 * math.sqrt(exact * (1 - exact) / trials))


def test_identical_scores_give_p_1():
    scores = [0.2, 0.9, 0.5, 0.0, 1.0]
    assert randomization_test(scores, list(scores)) == 1.0  # every swap is as extreme as 0


def test_a_system_better_on_every_one_of_30_queries_gets_the_smallest_p_value():
    better = [0.3 + number / 50 for number in range(30)]
    worse = [score * 0.8 for score in better]
    # Only 2 of the 2**30 ways to swap the pairs (none, or all of them) make a
    # difference as large as the observed one: the exact p-value is 2**-29.
    # Random swaps almost never hit one of those two while the trials stay far
    # below 2**30 (10,000 trials: 10,000 * 2**-29 = 2e-5), so the estimate is
    # its floor, 1 / (trials + 1): only the observed order itself counts.
    assert randomization_test(better, worse) == 1 / (DEFAULT_TRIALS + 1)
    assert randomization_test(worse, better) == 1 / (DEFAULT_TRIALS + 1)
    assert randomization_test(better, worse, trials=500, seed=7) == 1 / 501


def test_a_difference_on_a_single_query_cannot_be_told_from_chance():
    # Swapping that one pair can only flip the sign of the difference, and the
    # test is two-sided: every swap is as extreme as the observed difference,
    # however large it is.
    first = [0.9, 0.5, 0.5, 0.5, 0.5]
    second = [0.1, 0.5, 0.5, 0.5, 0.5]
    assert randomization_test(first, second) == 1.0
    assert randomization_test([1.0], [0.0]) == 1.0


def test_swapping_the_two_systems_gives_the_same_p_value():
    first, second = floats(SYSTEM_A), floats(SYSTEM_B)
    p_value = randomization_test(first, second)
    assert randomization_test(second, first) == p_value
    assert 1 / (DEFAULT_TRIALS + 1) < p_value < 1  # neither extreme, so the check means something


def test_the_same_seed_gives_the_same_p_value():
    first, second = floats(SYSTEM_A), floats(SYSTEM_B)
    seeds = range(5)
    p_values = [randomization_test(first, second, trials=500, seed=seed) for seed in seeds]
    assert [randomization_test(first, second, trials=500, seed=seed) for seed in seeds] == p_values
    assert len(set(p_values)) > 1  # other seeds swap other pairs, so the estimates differ
    # The default seed is fixed too, so a benchmark report can be reproduced.
    assert len({randomization_test(first, second, trials=500) for _ in range(3)}) == 1


@pytest.mark.parametrize(("first", "second"), [([0.5, 0.4], [0.5]), ([0.5], [0.5, 0.4]), ([], [])])
def test_the_randomization_test_needs_two_equally_long_lists(first, second):
    with pytest.raises(ValueError, match="equally long, non-empty"):
        randomization_test(first, second)


def test_the_randomization_test_takes_scores_for_at_most_max_judged_queries(monkeypatch):
    monkeypatch.setattr("searchengine.evaluation.MAX_JUDGED_QUERIES", 3)
    assert randomization_test([0.5] * 3, [0.4] * 3, trials=10) <= 1  # exactly the limit is fine
    with pytest.raises(ValueError, match="equally long, non-empty"):
        randomization_test([0.5] * 4, [0.4] * 4, trials=10)


@pytest.mark.parametrize("trials", [0, -1, MAX_TRIALS + 1, True, 2.0, None])
def test_trials_must_be_an_int_from_1_to_max_trials(trials):
    with pytest.raises(ValueError, match=f"trials must be an int from 1 to {MAX_TRIALS}"):
        randomization_test([0.5], [0.4], trials=trials)


def test_any_number_of_trials_up_to_the_limit_will_do(monkeypatch):
    monkeypatch.setattr("searchengine.evaluation.MAX_TRIALS", 5)  # not a million trials
    assert randomization_test([0.5], [0.4], trials=5) == 1.0
    assert randomization_test([0.5], [0.4], trials=1) == 1.0
    with pytest.raises(ValueError, match="from 1 to 5"):
        randomization_test([0.5], [0.4], trials=6)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([math.nan, 0.5], [0.1, 0.2]),
        ([0.5, 0.5], [0.1, math.inf]),
        ([math.inf, 0.0], [0.0, math.inf]),
    ],
)
def test_scores_must_be_finite(first, second):
    # A NaN makes every comparison false: no random swap would count as
    # extreme, and the test would report its smallest p-value, as if the
    # difference were certain.
    with pytest.raises(ValueError, match="finite"):
        randomization_test(first, second)


# ---------- properties that must hold for any input ----------

DOCS = [f"d{number}" for number in range(12)]
NOT_JUDGED = [f"x{number}" for number in range(5)]  # in no grade map: never relevant
rankings = st.lists(st.sampled_from(DOCS), unique=True)
grade_maps = st.dictionaries(st.sampled_from(DOCS), st.integers(1, 3), min_size=1)
cutoffs = st.integers(1, 15)
score_pairs = st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), min_size=1, max_size=12)


@settings(max_examples=200, deadline=None)
@given(rankings, grade_maps, cutoffs, cutoffs)
def test_every_metric_is_between_0_and_1(ranked, grades, k, depth):
    assert all(0 <= score <= 1 for score in astuple(score_query(ranked, grades, k, depth)))


@settings(max_examples=100, deadline=None)
@given(grade_maps, rankings, cutoffs, st.integers(0, 3))
def test_relevant_documents_first_best_grade_first_is_a_perfect_ranking(grades, rest, k, slack):
    best_first = sorted(grades, key=grades.__getitem__, reverse=True)
    ranked = best_first + [doc for doc in rest if doc not in grades]
    scores = score_query(ranked, grades, k, len(grades) + slack)  # depth: all relevant, or more
    assert (scores.ndcg, scores.mrr, scores.recall, scores.map) == (1, 1, 1, 1)
    assert scores.precision == pytest.approx(min(k, len(grades)) / k)


@settings(max_examples=100, deadline=None)
@given(st.permutations(DOCS), grade_maps, cutoffs)
def test_the_top_k_metrics_ignore_everything_after_position_k(order, grades, k):
    for metric in (ndcg, precision, reciprocal_rank):
        assert metric(order, grades, k) == metric(order[:k], grades, k)


@settings(max_examples=100, deadline=None)
@given(rankings, grade_maps, cutoffs, cutoffs, st.integers(1, len(NOT_JUDGED)))
def test_appending_documents_that_are_not_relevant_changes_no_score(ranked, grades, k, depth, n):
    padded = ranked + NOT_JUDGED[:n]  # past position k, or filling the top k up
    assert score_query(padded, grades, k, depth) == score_query(ranked, grades, k, depth)


@settings(max_examples=100, deadline=None)
@given(rankings, grade_maps, cutoffs)
def test_precision_and_recall_count_the_relevant_documents_in_the_top_k(ranked, grades, k):
    found = len(set(ranked[:k]) & grades.keys())
    assert precision(ranked, grades, k) * k == pytest.approx(found)
    assert recall(ranked, grades, k) * len(grades) == pytest.approx(found)


@settings(max_examples=50, deadline=None)
@given(score_pairs, st.integers(0, 2**32))
def test_the_p_value_is_between_its_floor_and_1_whichever_system_comes_first(pairs, seed):
    first = [a for a, _ in pairs]
    second = [b for _, b in pairs]
    p_value = randomization_test(first, second, trials=50, seed=seed)
    assert 1 / 51 <= p_value <= 1
    assert randomization_test(second, first, trials=50, seed=seed) == p_value
