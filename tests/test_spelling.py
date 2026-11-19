from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from searchengine import SpellingIndex, edit_distance, max_edits_auto
from searchengine.spelling import MAX_WORD_LENGTH


def reference_distance(a: str, b: str) -> int:
    """Textbook full-table Levenshtein, used to check the optimized version."""
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        table[i][0] = i
    for j in range(len(b) + 1):
        table[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
    return table[-1][-1]


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("kitten", "sitting", 3), ("same", "same", 0), ("", "abc", 3), ("flaw", "lawn", 2)],
)
def test_edit_distance(a, b, distance):
    assert edit_distance(a, b, limit=10) == distance


def test_edit_distance_stops_at_the_limit():
    assert edit_distance("kitten", "sitting", limit=1) == 2  # "more than 1"
    assert edit_distance("a", "abcdef", limit=2) == 3


@pytest.mark.parametrize("limit", [-1, 1.5, True, None])
def test_edit_distance_rejects_bad_limits(limit):
    with pytest.raises(ValueError, match="limit"):
        edit_distance("a", "b", limit=limit)


@pytest.mark.parametrize("bad", [None, 3, b"word"])
def test_words_must_be_text(bad):
    spelling = SpellingIndex()
    calls: list[Callable[[], object]] = [
        lambda: edit_distance(bad, "word"),
        lambda: edit_distance("word", bad),
        lambda: max_edits_auto(bad),
        lambda: spelling.add(bad),
        lambda: spelling.candidates(bad),
        lambda: spelling.correct(bad),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="must be a str"):
            call()


def test_word_length_is_bounded_like_tokens():
    """No indexed word is longer than the longest token, which also bounds
    the work edit_distance does."""
    longest = "a" * MAX_WORD_LENGTH
    too_long = longest + "a"
    assert edit_distance(longest, longest) == 0
    for a, b in ((too_long, "a"), ("a", too_long)):
        with pytest.raises(ValueError, match="limited"):
            edit_distance(a, b)
    spelling = make_spelling("search")
    with pytest.raises(ValueError, match="limited"):
        spelling.add(too_long)
    assert spelling.counts == {"search": 1}
    assert spelling.candidates(too_long) == []
    assert spelling.correct(too_long) == too_long


@given(st.text(max_size=8), st.text(max_size=8), st.integers(min_value=0, max_value=4))
def test_edit_distance_matches_reference(a, b, limit):
    assert edit_distance(a, b, limit) == min(reference_distance(a, b), limit + 1)


@pytest.mark.parametrize(("word", "edits"), [("ab", 0), ("abc", 1), ("abcde", 1), ("abcdef", 2)])
def test_auto_fuzziness_matches_elasticsearch(word, edits):
    assert max_edits_auto(word) == edits


def make_spelling(*words: str) -> SpellingIndex:
    spelling = SpellingIndex()
    for word in words:
        spelling.add(word)
    return spelling


def test_correct_finds_the_closest_word():
    spelling = make_spelling("ranking", "rank", "elasticsearch")
    assert spelling.correct("rankng") == "ranking"
    assert spelling.correct("elastisearch") == "elasticsearch"


def test_correct_leaves_known_short_and_hopeless_words_alone():
    spelling = make_spelling("ranking")
    assert spelling.correct("ranking") == "ranking"
    assert spelling.correct("rk") == "rk"  # two letters: AUTO allows no edits
    assert spelling.correct("zzzzzzzz") == "zzzzzzzz"


def test_ties_go_to_the_more_common_word_then_alphabetical():
    spelling = make_spelling("cart", "card", "card")
    assert spelling.correct("carx") == "card"
    assert make_spelling("cart", "card").correct("carx") == "card"


def test_from_counts_round_trip():
    spelling = SpellingIndex.from_counts({"search": 3, "engine": 1})
    assert spelling.counts == {"search": 3, "engine": 1}
    assert spelling.correct("serch") == "search"


@pytest.mark.parametrize(
    "counts", [{"word": 0}, {"word": "3"}, {"word": True}, {"a" * (MAX_WORD_LENGTH + 1): 1}, {3: 1}]
)
def test_from_counts_rejects_bad_entries(counts):
    with pytest.raises(ValueError, match="invalid spelling entry"):
        SpellingIndex.from_counts(counts)


@pytest.mark.parametrize("counts", [None, [("word", 1)], "word"])
def test_from_counts_needs_a_mapping(counts):
    with pytest.raises(TypeError, match="mapping"):
        SpellingIndex.from_counts(counts)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "2"])
def test_add_rejects_bad_counts_without_changing_anything(count):
    spelling = make_spelling("word")
    with pytest.raises(ValueError, match="count"):
        spelling.add("word", count)
    with pytest.raises(ValueError, match="count"):
        spelling.add("other", count)
    assert spelling.counts == {"word": 1}
    assert spelling.candidates("other") == []


def test_candidates_come_best_first():
    """Fewest edits, then the more common word, then alphabetical."""
    spelling = make_spelling("card", "card", "cart", "care", "cord", "carts")
    assert spelling.candidates("carx") == ["card", "care", "cart"]
    assert spelling.candidates("card") == ["card", "care", "cart", "cord"]


def test_candidates_are_capped(monkeypatch):
    monkeypatch.setattr("searchengine.spelling.MAX_CANDIDATES", 2)
    spelling = make_spelling("cart", "care", "card", "card")
    assert spelling.candidates("carx") == ["card", "care"]


def test_short_words_get_no_corrections():
    spelling = make_spelling("ab", "ac")
    assert spelling.candidates("ax") == []
    assert spelling.candidates("ab") == ["ab"]
