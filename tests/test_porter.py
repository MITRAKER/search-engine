import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from searchengine.porter import measure, porter_stem

GOLDEN = json.loads((Path(__file__).parent / "data" / "porter_golden.json").read_text("utf-8"))


def test_matches_elasticsearch_on_golden_words():
    """3,107 words stemmed by Elasticsearch's own porter_stem filter."""
    stems = GOLDEN["stems"]
    mismatches = {
        word: (porter_stem(word), es) for word, es in stems.items() if porter_stem(word) != es
    }
    assert not mismatches
    assert len(stems) > 3000


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        # Examples from Porter's 1980 paper, one per rule group.
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("feed", "feed"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("conflated", "conflat"),
        ("hopping", "hop"),
        ("falling", "fall"),
        ("filing", "file"),
        ("happy", "happi"),
        ("sky", "sky"),
        ("relational", "relat"),
        ("generalizations", "gener"),
        ("oscillators", "oscil"),
        ("adoption", "adopt"),
        ("controll", "control"),
        ("roll", "roll"),
        # Words this project relies on.
        ("running", "run"),
        ("runs", "run"),
        ("engines", "engin"),
        ("engine", "engin"),
        ("indexing", "index"),
        ("indexes", "index"),
    ],
)
def test_classic_examples(word, stem):
    assert porter_stem(word) == stem


@pytest.mark.parametrize(
    ("stem", "m"),
    [
        ("tr", 0),
        ("ee", 0),
        ("tree", 0),
        ("y", 0),
        ("by", 0),
        ("trouble", 1),
        ("oats", 1),
        ("ivy", 1),
        ("troubles", 2),
        ("private", 2),
    ],
)
def test_measure(stem, m):
    """The examples Porter gives for m in his paper."""
    assert measure(stem) == m


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("\U00020bb7s", "\U00020bb7"),
        ("\U0001d41as", "\U0001d41a"),
        ("bo\U0001d41bing", "bo\U0001d41b"),
        ("ha\U0001d41b\U0001d41bing", "ha\U0001d41b\U0001d41b"),
    ],
)
def test_a_letter_beyond_u_ffff_counts_as_two_units_like_in_java(word, stem):
    """Elasticsearch's own stems for these words. A Java string holds UTF-16
    units, so a bold math letter plus "s" is three units long, and two bold
    b's are four units with no double consonant at the end."""
    assert porter_stem(word) == stem


@pytest.mark.parametrize("bad", [None, 3, b"running"])
def test_rejects_non_text(bad):
    with pytest.raises(TypeError, match="porter_stem needs a str"):
        porter_stem(bad)


@pytest.mark.parametrize("word", ["", "a", "is", "xy"])
def test_short_words_unchanged(word):
    assert porter_stem(word) == word


@given(st.text(max_size=30))
def test_never_crashes_and_never_grows_much(word):
    stem = porter_stem(word.lower())
    assert isinstance(stem, str)
    assert len(stem) <= len(word.lower()) + 1  # at most one "e" is added back
