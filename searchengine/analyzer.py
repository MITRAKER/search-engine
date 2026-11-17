"""Text analysis: turning raw text into searchable terms.

Pipeline:

    raw text -> tokenize -> lowercase -> drop long tokens -> drop stopwords -> stem

The SAME analyzer must run on documents (at index time) and on queries (at
search time). If they differ, "Running" in a query never matches "run" in the
index. Elasticsearch lets you build a custom analyzer from the same steps, in
the same order, so both engines see identical terms (milestone 3 does that).

Every step copies what Lucene does, down to the details that matter for
non-English text: how characters are lowercased, and how token length is
counted. ``tests/test_analyzer.py`` explains each one.
"""

from __future__ import annotations

import re
import unicodedata
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

# Longest token we index; longer ones are dropped, as Elasticsearch's "length"
# token filter drops them. (Its standard tokenizer would instead cut them into
# pieces of 255.) Length is counted the way Java counts it, in UTF-16 code
# units: a character outside the Basic Multilingual Plane, such as an emoji,
# counts twice. Bounding every input is JPL rule 2/3 in practice.
MAX_TOKEN_LENGTH: Final = 255

# Longest text we analyze in one call (characters).
MAX_TEXT_LENGTH: Final = 10_000_000


def _mark_class(start: int, stop: int) -> str:
    """A regex character class body for the combining marks in [start, stop).

    Python's ``re`` has no ``\\p{M}``, so we build the class once, at import,
    from the Unicode database of the running Python.
    """
    ranges: list[tuple[int, int]] = []
    for code in range(start, stop):
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1] = (ranges[-1][0], code)
            else:
                ranges.append((code, code))
    return "".join(f"\\U{low:08x}-\\U{high:08x}" for low, high in ranges)


# A token is a run of letters, digits and combining marks, in any script:
# "café", "2024", "जस्टिस", "مَرْحَبًا". [^\W_] means "word characters except the
# underscore", that is letters and digits. Combining marks (accents written as
# separate characters, Indic vowel signs, Arabic vowel marks) are not word
# characters for Python, but they belong inside words. They are only found in
# Unicode planes 0, 1 and 14; the lookahead skips the slower check of the
# rare planes 1 and 14 for ordinary characters.
_BMP_MARKS: Final = _mark_class(0, 0x10000)
_ASTRAL_MARKS: Final = _mark_class(0x10000, 0x20000) + _mark_class(0xE0000, 0xF0000)
_TOKEN_RE: Final = re.compile(
    rf"(?:[^\W_]|[{_BMP_MARKS}]|(?=[\U00010000-\U0010ffff])[{_ASTRAL_MARKS}])+"
)


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


def _lowercase(word: str) -> str:
    """Lowercase one character at a time, as Lucene does.

    Python's ``str.lower()`` follows Unicode's special rules: a capital sigma
    at the end of a word becomes the final-form sigma, and the dotted capital
    I (U+0130) becomes "i" plus a combining dot. Java lowercases each
    character on its own, with neither rule, and our terms must match
    Elasticsearch's exactly.
    """
    if word.isascii():
        return word.lower()
    return "".join("i" if char == "İ" else char.lower() for char in word)


def java_length(word: str) -> int:
    """Length in UTF-16 code units, the way Java (and so Lucene) counts."""
    if word.isascii():
        return len(word)
    return len(word.encode("utf-16-le")) // 2


def is_searchable(word: str) -> bool:
    """Whether a lowercase token survives analysis: not a stopword, not too long."""
    return word not in ENGLISH_STOPWORDS and java_length(word) <= MAX_TOKEN_LENGTH


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens. Nothing is removed yet."""
    _check_text(text)
    return [_lowercase(match.group()) for match in _TOKEN_RE.finditer(text)]


def analyze_tokens(text: str) -> list[Token]:
    """Run the full pipeline and keep each term's position and surface form.

    Positions count every token, including the ones we drop. So in
    "inverted the index", "index" is at position 2, not 1. That is how Lucene
    does it, and it stops the phrase "inverted index" from matching there.
    """
    tokens: list[Token] = []
    for position, word in enumerate(tokenize(text)):
        if is_searchable(word):
            tokens.append(Token(porter_stem(word), word, position))
    return tokens


def analyze(text: str) -> list[str]:
    """Just the terms, in order. Handy for queries and tests."""
    return [token.term for token in analyze_tokens(text)]
