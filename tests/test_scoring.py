import math

import pytest

from searchengine import bm25_idf, bm25_tf, tfidf_idf, tfidf_tf


def test_bm25_idf_formula():
    assert bm25_idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))


def test_bm25_idf_is_positive_even_for_a_term_in_every_document():
    assert bm25_idf(10, 10) > 0


def test_rare_terms_weigh_more():
    assert bm25_idf(1000, 1) > bm25_idf(1000, 100) > bm25_idf(1000, 900)


def test_bm25_tf_saturates_at_k1_plus_1():
    assert bm25_tf(1, 10, 10) == pytest.approx(1.0)  # tf=1 at average length
    assert bm25_tf(1000, 10, 10, k1=1.2) < 2.2
    assert bm25_tf(1000, 10, 10, k1=1.2) == pytest.approx(2.2, rel=1e-2)


def test_bm25_tf_prefers_shorter_documents():
    assert bm25_tf(2, 5, 10) > bm25_tf(2, 10, 10) > bm25_tf(2, 50, 10)


def test_b_zero_turns_off_length_normalization():
    assert bm25_tf(2, 5, 10, b=0) == bm25_tf(2, 500, 10, b=0)


def test_zero_tf_scores_zero():
    assert bm25_tf(0, 10, 10) == 0.0
    assert tfidf_tf(0) == 0.0


def test_tfidf_formulas():
    assert tfidf_idf(100, 10) == pytest.approx(math.log(10))
    assert tfidf_idf(100, 100) == 0.0
    assert tfidf_tf(1) == pytest.approx(math.log(2))
    assert tfidf_tf(100) < 10 * tfidf_tf(1)


@pytest.mark.parametrize(("n", "df"), [(0, 0), (10, 0), (10, 11), (-1, 1)])
def test_idf_rejects_impossible_counts(n, df):
    with pytest.raises(ValueError, match="df"):
        bm25_idf(n, df)
    with pytest.raises(ValueError, match="df"):
        tfidf_idf(n, df)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tf": -1, "dl": 1, "avgdl": 1},
        {"tf": 1, "dl": -1, "avgdl": 1},
        {"tf": 1, "dl": 1, "avgdl": 0},
        {"tf": 1, "dl": 1, "avgdl": 1, "k1": -0.1},
        {"tf": 1, "dl": 1, "avgdl": 1, "b": 1.5},
    ],
)
def test_bm25_tf_rejects_bad_input(kwargs):
    with pytest.raises(ValueError, match=r"need|invalid"):
        bm25_tf(**kwargs)


def test_tfidf_tf_rejects_negative():
    with pytest.raises(ValueError, match="tf"):
        tfidf_tf(-1)
