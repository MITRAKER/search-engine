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
from typing import Final

# Elasticsearch's defaults for BM25.
DEFAULT_K1: Final = 1.2
DEFAULT_B: Final = 0.75


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
    if not (k1 >= 0 and 0 <= b <= 1):
        raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
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
