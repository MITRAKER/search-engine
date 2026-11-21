import pytest
from hypothesis import given
from hypothesis import strategies as st

from searchengine import ENGLISH_STOPWORDS, Token, analyze, analyze_tokens, tokenize
from searchengine.analyzer import MAX_TEXT_LENGTH, MAX_TOKEN_LENGTH, is_searchable, java_length


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


# ---------- details that must match Lucene exactly ----------


def test_lowercasing_is_one_character_at_a_time_like_java():
    """Python's str.lower() turns a dotted capital I into "i" plus a combining
    dot, and a final capital sigma into the final-form sigma. Java lowercases
    each character on its own, with neither rule, and so must we."""
    dotted_i = "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}"
    assert tokenize(f"{dotted_i}STANBUL") == ["istanbul"]
    capitals = "\N{GREEK CAPITAL LETTER DELTA}\N{GREEK CAPITAL LETTER SIGMA}"
    smalls = "\N{GREEK SMALL LETTER DELTA}\N{GREEK SMALL LETTER SIGMA}"
    assert capitals.lower() != smalls  # Python's own rule differs...
    assert tokenize(capitals) == [smalls]  # ...ours matches Java


@pytest.mark.parametrize(
    "word",
    [
        "हिन्दी",  # Devanagari vowel signs and virama are combining marks
        "cafe\N{COMBINING ACUTE ACCENT}",  # an accent typed as its own character
        "مَرْحَبًا",  # Arabic vowel marks
        "\N{BRAHMI SIGN CANDRABINDU}\N{BRAHMI VOWEL SIGN AA}x",  # marks outside the BMP
    ],
)
def test_combining_marks_stay_inside_words(word):
    assert tokenize(f"{word} next") == [word, "next"]


def test_token_length_counts_utf16_units_like_java():
    bold_a = "\N{MATHEMATICAL BOLD CAPITAL A}"  # outside the BMP: 2 units in Java
    assert java_length(bold_a) == 2
    assert java_length("café") == 4
    assert len(analyze(bold_a * (MAX_TOKEN_LENGTH // 2))) == 1  # 254 units: kept
    assert analyze(bold_a * (MAX_TOKEN_LENGTH // 2 + 1)) == []  # 256 units: dropped


@pytest.mark.parametrize("bad", [None, 3, b"word"])
def test_word_helpers_need_text(bad):
    for helper in (java_length, is_searchable):
        with pytest.raises(TypeError, match="must be a str"):
            helper(bad)


def test_is_searchable_drops_stopwords_and_long_tokens():
    assert is_searchable("search")
    assert is_searchable("x" * MAX_TOKEN_LENGTH)
    assert not is_searchable("the")
    assert not is_searchable("x" * (MAX_TOKEN_LENGTH + 1))


@given(st.text(max_size=200))
def test_a_token_tokenizes_to_itself(text):
    """Lowercasing never splits or joins tokens, so analysis is stable."""
    for token in tokenize(text):
        assert tokenize(token) == [token]
