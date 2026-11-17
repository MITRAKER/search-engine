"""Ranking formulas as small, pure functions.

Keeping the math separate from the index makes it easy to read, test and
compare with Elasticsearch (see ``tests/test_es_parity.py``).

Notation used below:
    N      number of documents
    df     document frequency: documents that contain the term
    tf     term frequency: how often the term occurs in one document
    dl     document length, in terms
    avgdl  average document length
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import Final

# Elasticsearch's defaults for BM25.
DEFAULT_K1: Final = 1.2
DEFAULT_B: Final = 0.75

# Lucene's one-byte length encoding (SmallFloat.intToByte4) keeps 0 to 23 as
# they are, then stores (length - 24) with 4 significant bits.
_EXACT_NORMS: Final = 24


def _check_counts(n: float, df: float) -> None:
    if n <= 0 or df <= 0 or df > n:
        raise ValueError(f"need 0 < df <= N, got df={df}, N={n}")


def bm25_idf(n: float, df: float) -> float:
    """log(1 + (N - df + 0.5) / (df + 0.5)).

    Rare terms get a high weight, common terms a low one. The "1 +" keeps the
    value positive even for a term that is in every document. This is the
    exact formula Lucene and Elasticsearch use.
    """
    _check_counts(n, df)
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def check_bm25_params(k1: float, b: float) -> None:
    """k1 must be a finite number >= 0, and b between 0 and 1."""
    if not (math.isfinite(k1) and k1 >= 0 and 0 <= b <= 1):
        raise ValueError(f"need a finite k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")


def bm25_tf(
    tf: float, dl: float, avgdl: float, k1: float = DEFAULT_K1, b: float = DEFAULT_B
) -> float:
    """The term-frequency part of BM25:

                tf * (k1 + 1)
        ----------------------------------
        tf + k1 * (1 - b + b * dl / avgdl)

    * k1 controls saturation: the 10th occurrence of a word adds far less
      than the 1st, and the value can never exceed k1 + 1.
    * b controls length normalization: a match in a short document counts
      more than the same match in a long one.
    """
    if tf < 0 or dl < 0 or avgdl <= 0:
        raise ValueError(f"invalid statistics tf={tf}, dl={dl}, avgdl={avgdl}")
    check_bm25_params(k1, b)
    if tf == 0:
        return 0.0
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def tfidf_idf(n: float, df: float) -> float:
    """Classic inverse document frequency: log(N / df).

    It is zero for a term found in every document: such a term cannot help
    tell documents apart.
    """
    _check_counts(n, df)
    return math.log(n / df)


def tfidf_tf(tf: float) -> float:
    """Log-scaled term frequency: log(1 + tf).

    Without the log, a document that repeats a word 100 times would score
    100 times higher than one that uses it once. log(1 + tf) grows slowly and
    is never negative, even for fractional field-weighted frequencies.
    """
    if tf < 0:
        raise ValueError(f"tf must be >= 0, got {tf}")
    return math.log1p(tf)


def lucene_length(length: int) -> int:
    """The field length Lucene remembers for a document, in one byte.

    Lucene stores every document's field length (its "norm") in a single
    byte. Lengths up to 40 survive exactly. Above that only 4 significant
    bits are kept, so 41 becomes 40 and 375 becomes 344. BM25 then scores with
    the stored value. This is SmallFloat.byte4ToInt(SmallFloat.intToByte4(length)).
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError(f"length must be an int >= 0, got {length!r}")
    if length < _EXACT_NORMS:
        return length
    rest = length - _EXACT_NORMS
    shift = max(rest.bit_length() - 4, 0)
    return _EXACT_NORMS + (rest >> shift << shift)


def float32(value: float) -> float:
    """Round to the nearest 32-bit float, as Java's ``(float)`` cast does.

    Lucene combines document lengths in 32-bit floats. As in Java, a value
    too large for one becomes infinity.
    """
    try:
        result: float = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)
    return result


def lucene_weighted_length(lengths: Mapping[str, int], weights: Mapping[str, float]) -> int:
    """The document length Elasticsearch's ``combined_fields`` scores with.

    Each field's stored length is multiplied by the field's weight and summed,
    in 32-bit floats and in field-name order as Lucene does, then rounded and
    stored in one byte again. With one field of weight 1 it is just
    ``lucene_length``.
    """
    total = 0.0
    for field in sorted(weights):
        weighted = float32(float32(weights[field]) * lucene_length(lengths[field]))
        total = float32(total + weighted)
    return lucene_length(math.floor(total + 0.5))


def lucene_weighted_total(totals: Mapping[str, int], weights: Mapping[str, float]) -> int:
    """The total length combined_fields divides by N to get avgdl.

    Lucene adds each field's total length times its weight into a whole
    number (a Java long), so the fraction is dropped after every field: with
    weights 1.3 and 1.7 and totals 29452 and 357, that is 38893, not 38894.5.
    """
    total = 0
    for field in sorted(weights):
        total = math.floor(total + float32(weights[field]) * totals[field])
    return total
