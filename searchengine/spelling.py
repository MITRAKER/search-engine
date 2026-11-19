"""Typo tolerance: Levenshtein distance and "did you mean" corrections.

We correct the words people type (their *surface* form, like "rankng"), not
the stemmed terms, because typos are made on real words.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from typing import Final

from .analyzer import MAX_TOKEN_LENGTH

# Elasticsearch's "fuzziness": "AUTO" allows 0 edits for words of 1-2
# letters, 1 edit for 3-5 letters and 2 edits for longer words.
MAX_EDITS: Final = 2
MAX_CANDIDATES: Final = 50
# No indexed word is longer than the analyzer's longest token, so neither is
# any word worth comparing. This also bounds edit_distance's work (JPL rule 2).
MAX_WORD_LENGTH: Final = MAX_TOKEN_LENGTH


def _check_word(word: object) -> None:
    if not isinstance(word, str):
        raise TypeError(f"a word must be a str, not {type(word).__name__}")


def max_edits_auto(word: str) -> int:
    """How many typos to tolerate in ``word``, like Elasticsearch's AUTO."""
    _check_word(word)
    if len(word) <= 2:
        return 0
    return 1 if len(word) <= 5 else MAX_EDITS


def edit_distance(a: str, b: str, limit: int = MAX_EDITS) -> int:
    """Levenshtein distance: the fewest one-letter edits that turn a into b.

    Classic dynamic programming that keeps only one row in memory. It stops
    early, returning ``limit + 1``, as soon as the distance must exceed
    ``limit``. Both loops are bounded by the word lengths, which are at most
    ``MAX_WORD_LENGTH`` (JPL rule 2).
    """
    _check_word(a)
    _check_word(b)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be an int >= 0")
    if len(a) > MAX_WORD_LENGTH or len(b) > MAX_WORD_LENGTH:
        raise ValueError(f"words are limited to {MAX_WORD_LENGTH} characters")
    if abs(len(a) - len(b)) > limit:
        return limit + 1  # needs at least that many inserts or deletes
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # delete char_a
                    current[j - 1] + 1,  # insert char_b
                    previous[j - 1] + (char_a != char_b),  # substitute
                )
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return min(previous[-1], limit + 1)


def _is_count(count: object) -> bool:
    return isinstance(count, int) and not isinstance(count, bool) and count >= 1


class SpellingIndex:
    """Remembers every indexed word and how often it appears."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        # Words grouped by length, so a lookup only scans plausible lengths.
        self._by_length: dict[int, list[str]] = {}

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> SpellingIndex:
        """Rebuild from saved word counts. A bad entry raises ValueError."""
        if not isinstance(counts, Mapping):
            raise TypeError("counts must be a mapping of word to count")
        index = cls()
        for word, count in counts.items():
            if not isinstance(word, str):
                raise ValueError("invalid spelling entry: every word must be a str")
            if not _is_count(count) or len(word) > MAX_WORD_LENGTH:
                raise ValueError(f"invalid spelling entry {word[:50]!r}")
            index.add(word, count)
        return index

    def add(self, word: str, count: int = 1) -> None:
        """Count ``word`` once more (or ``count`` times more)."""
        _check_word(word)
        if len(word) > MAX_WORD_LENGTH:
            raise ValueError(f"words are limited to {MAX_WORD_LENGTH} characters")
        if not _is_count(count):
            raise ValueError("count must be an int >= 1")
        if word not in self.counts:
            self.counts[word] = 0
            self._by_length.setdefault(len(word), []).append(word)
        self.counts[word] += count

    def candidates(self, word: str) -> list[str]:
        """Known words within the allowed number of typos, best first.

        Best means fewest edits, then the more common word, then
        alphabetical order, so the answer never depends on insertion order.
        At most ``MAX_CANDIDATES`` are returned (JPL rule 3). A word longer
        than any indexed word can have has none.
        """
        _check_word(word)
        if len(word) > MAX_WORD_LENGTH:
            return []
        limit = max_edits_auto(word)
        found = []
        for length in range(len(word) - limit, len(word) + limit + 1):
            for candidate in self._by_length.get(length, ()):
                distance = edit_distance(word, candidate, limit)
                if distance <= limit:
                    found.append((distance, -self.counts[candidate], candidate))
        return [candidate for _, _, candidate in heapq.nsmallest(MAX_CANDIDATES, found)]

    def correct(self, word: str) -> str:
        """Return the closest known word, or ``word`` itself if it is known
        or nothing is close enough."""
        _check_word(word)
        if word in self.counts:
            return word
        found = self.candidates(word)
        return found[0] if found else word
