"""Text analysis: turning raw text into searchable terms.

Pipeline (the same order Elasticsearch runs its analyzers in):

    raw text -> tokenize -> lowercase -> drop long tokens -> drop stopwords -> stem

The SAME analyzer must run on documents (at index time) and on queries (at
search time). If they differ, "Running" in a query never matches "run" in the
index. ``searchengine/es_engine.py`` builds an Elasticsearch analyzer from the
same pieces, so both engines see identical terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .porter import porter_stem

# Lucene's English stopword set. Elasticsearch's "_english_" stopwords and its
# built-in "english" analyzer use exactly these 33 words.
ENGLISH_STOPWORDS: Final = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
        "the", "their", "then", "there", "these", "they", "this", "to", "was",
        "will", "with",
    )
)  # fmt: skip

# Longest token we index, in characters. Elasticsearch's standard tokenizer uses
# the same default. Bounding every input is JPL rule 2/3 in practice.
MAX_TOKEN_LENGTH: Final = 255

# Longest text we analyze in one call (characters).
MAX_TEXT_LENGTH: Final = 10_000_000

# A token is a run of letters or digits in any script ("café", "2024", "جستجو").
# [^\W_] means "word characters except underscore".
_TOKEN_RE: Final = re.compile(r"[^\W_]+")


@dataclass(frozen=True, slots=True)
class Token:
    """One analyzed word."""

    term: str  # stemmed term, the key in the inverted index
    surface: str  # the lowercased word as written, used for spelling fixes
    position: int  # word position; removed words leave gaps, like in Lucene


def _check_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text is longer than {MAX_TEXT_LENGTH} characters")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens. Nothing is removed yet."""
    _check_text(text)
    return [match.group().lower() for match in _TOKEN_RE.finditer(text)]


def analyze_tokens(text: str) -> list[Token]:
    """Run the full pipeline and keep each term's position and surface form.

    Positions count every token, including the ones we drop. So in
    "inverted the index", "index" is at position 2, not 1. That is how Lucene
    does it, and it stops the phrase "inverted index" from matching there.
    """
    tokens: list[Token] = []
    for position, word in enumerate(tokenize(text)):
        if len(word) > MAX_TOKEN_LENGTH or word in ENGLISH_STOPWORDS:
            continue
        tokens.append(Token(porter_stem(word), word, position))
    return tokens


def analyze(text: str) -> list[str]:
    """Just the terms, in order. Handy for queries and tests."""
    return [token.term for token in analyze_tokens(text)]
