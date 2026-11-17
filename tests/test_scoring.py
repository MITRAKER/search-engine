import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from searchengine import bm25_idf, bm25_tf, tfidf_idf, tfidf_tf
from searchengine.scoring import (
    check_bm25_params,
    float32,
    lucene_length,
    lucene_weighted_length,
    lucene_weighted_total,
)


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


# ---------- Lucene's one-byte document lengths ----------


def _long_to_int4(value: int) -> int:
    """Lucene's SmallFloat.longToInt4, line by line: 4 significant bits."""
    bits = value.bit_length()
    if bits < 4:
        return value
    shift = bits - 4
    return (value >> shift) & 0x07 | (shift + 1) << 3


def _int4_to_long(encoded: int) -> int:
    """Lucene's SmallFloat.int4ToLong."""
    shift = (encoded >> 3) - 1
    return encoded & 0x07 if shift == -1 else (encoded & 0x07 | 0x08) << shift


# 255 - longToInt4(Integer.MAX_VALUE): the byte values left for exact lengths.
_FREE_VALUES = 255 - _long_to_int4(2**31 - 1)


def lucene_reference(length: int) -> int:
    """SmallFloat.byte4ToInt(SmallFloat.intToByte4(length)), ported literally."""
    if length < _FREE_VALUES:
        return length
    encoded = _FREE_VALUES + _long_to_int4(length - _FREE_VALUES)
    assert encoded <= 255  # it fits in one byte
    return _FREE_VALUES + _int4_to_long(encoded - _FREE_VALUES)


def test_the_reference_port_keeps_24_exact_values():
    assert _FREE_VALUES == 24


@pytest.mark.parametrize(
    ("length", "stored"),
    [(0, 0), (23, 23), (24, 24), (40, 40), (41, 40), (42, 42), (43, 42), (375, 344), (1110, 1048)],
)
def test_lucene_length_examples(length, stored):
    assert lucene_length(length) == stored


def test_lucene_length_matches_lucene_for_every_small_length():
    assert [lucene_length(n) for n in range(5000)] == [lucene_reference(n) for n in range(5000)]


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_lucene_length_matches_lucene_for_any_int(length):
    stored = lucene_length(length)
    assert stored == lucene_reference(length)
    assert lucene_length(stored) == stored  # storing again changes nothing
    assert length * 7 / 8 < stored <= length or stored == length == 0  # loses under 1/8


def test_lucene_length_never_reorders_lengths():
    stored = [lucene_length(n) for n in range(3000)]
    assert stored == sorted(stored)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3", None])
def test_lucene_length_rejects_non_lengths(bad):
    with pytest.raises(ValueError, match="length"):
        lucene_length(bad)


# ---------- 32-bit floats, as Java computes them ----------


def test_float32_rounds_to_the_nearest_32_bit_float():
    assert float32(1.7) == 1.7000000476837158
    assert float32(0.5) == 0.5
    assert float32(float32(1.3)) == float32(1.3)


def test_float32_breaks_ties_to_even_like_java():
    assert float32(1 + 2**-24) == 1.0  # halfway: round to the even neighbour
    assert float32(1 + 3 * 2**-24) == 1 + 2**-22


def test_float32_overflows_to_infinity_like_java():
    assert float32(1e39) == math.inf
    assert float32(-1e39) == -math.inf
    assert float32(math.inf) == math.inf


# ---------- combined_fields: weighted lengths and totals ----------


def test_one_field_of_weight_one_is_just_the_stored_length():
    weights = {"body": 1.0}
    assert all(lucene_weighted_length({"body": n}, weights) == lucene_length(n) for n in range(600))


def test_weighted_length_is_summed_in_32_bit_floats():
    """Title 2 words at 1.7, body 17 words at 1.3: exactly 3.4 + 22.1 = 25.5,
    which rounds up to 26. In 32-bit floats the sum is a hair under 25.5, so
    Lucene stores 25, and so must we."""
    weights = {"title": 1.7, "body": 1.3}
    assert math.isclose(1.7 * 2 + 1.3 * 17, 25.5)
    assert lucene_weighted_length({"title": 2, "body": 17}, weights) == 25
    # The rounded sum is then stored in one byte, like any length. Title 19
    # and body 277 (stored as 264) give a hair under 375.5: 375, stored as 344.
    assert lucene_weighted_length({"title": 19, "body": 277}, weights) == 344


def test_weighted_length_ignores_fields_without_a_weight():
    lengths = {"title": 5, "body": 7}
    assert lucene_weighted_length(lengths, {"title": 2.0}) == 10


def test_weighted_total_drops_the_fraction_after_each_field():
    weights = {"title": 1.7, "body": 1.3}
    assert math.isclose(29452 * 1.3 + 357 * 1.7, 38894.5)
    assert lucene_weighted_total({"body": 29452, "title": 357}, weights) == 38893


def test_weighted_total_with_whole_weights_is_exact():
    weights = {"title": 3.0, "body": 1.0}
    assert lucene_weighted_total({"title": 120, "body": 4567}, weights) == 4927


# ---------- BM25 parameters ----------


@pytest.mark.parametrize(("k1", "b"), [(0, 0), (1.2, 0.75), (100, 1)])
def test_bm25_params_accept_sensible_values(k1, b):
    check_bm25_params(k1, b)


@pytest.mark.parametrize(
    ("k1", "b"),
    [(math.inf, 0.75), (math.nan, 0.75), (-1, 0.75), (1.2, -0.1), (1.2, 1.1), (1.2, math.nan)],
)
def test_bm25_params_reject_the_rest(k1, b):
    with pytest.raises(ValueError, match="k1"):
        check_bm25_params(k1, b)
