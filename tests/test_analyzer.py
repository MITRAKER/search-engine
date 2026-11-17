import pytest
from hypothesis import given
from hypothesis import strategies as st

from searchengine import ENGLISH_STOPWORDS, Token, analyze, analyze_tokens, tokenize
from searchengine.analyzer import MAX_TEXT_LENGTH, MAX_TOKEN_LENGTH


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it", "s", "2026"]


def test_tokenize_keeps_letters_from_any_script():
    assert tokenize("Café جستجو naïve") == ["café", "جستجو", "naïve"]


def test_underscore_is_a_separator():
    assert tokenize("snake_case") == ["snake", "case"]


def test_analyze_drops_stopwords_and_stems():
    assert analyze("The Runners are RUNNING!") == ["runner", "run"]


def test_stopword_list_matches_lucene():
    assert len(ENGLISH_STOPWORDS) == 33
    assert {"the", "and", "not", "with"} <= ENGLISH_STOPWORDS
    assert "search" not in ENGLISH_STOPWORDS


def test_positions_leave_gaps_for_removed_words():
    assert analyze_tokens("inverted the index") == [
        Token("invert", "inverted", 0),
        Token("index", "index", 2),
    ]


def test_overlong_tokens_are_dropped_but_keep_their_position():
    text = "short " + "x" * (MAX_TOKEN_LENGTH + 1) + " word"
    assert [(t.term, t.position) for t in analyze_tokens(text)] == [("short", 0), ("word", 2)]


def test_only_stopwords_gives_nothing():
    assert analyze("the and of") == []


def test_rejects_non_text():
    with pytest.raises(TypeError):
        analyze(42)  # type: ignore[arg-type]


def test_rejects_huge_text():
    with pytest.raises(ValueError, match="longer than"):
        tokenize("a" * (MAX_TEXT_LENGTH + 1))


@given(st.text(max_size=200))
def test_analysis_is_total_and_positions_increase(text):
    tokens = analyze_tokens(text)
    positions = [token.position for token in tokens]
    assert positions == sorted(set(positions))
    assert all(token.term and token.surface for token in tokens)
