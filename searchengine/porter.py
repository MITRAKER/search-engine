"""The Porter stemmer, implemented from scratch.

Martin Porter, "An algorithm for suffix stripping", Program 14(3), 1980.

This follows Porter's own reference implementation, which is also what
Lucene's ``PorterStemmer`` (and so Elasticsearch's ``porter_stem`` filter)
uses. ``tests/test_porter.py`` checks it against stems produced by
Elasticsearch 9.5.5 itself. Matching it exactly means our engine and
Elasticsearch see the same terms, so when we compare them later, the
differences come from ranking and not from text processing.

Key idea: any word can be written as consonant groups (C) and vowel groups (V):

        [C](VC)^m[V]

``m`` is the *measure* of the word. Most rules only fire when what remains
after removing a suffix is long enough (m > 0 or m > 1). This protects short
words: "rate" keeps its "-ate", but "activate" becomes "activ".
"""

from __future__ import annotations

from typing import Final

_VOWELS: Final = frozenset("aeiou")


def _consonant_mask(word: str) -> list[bool]:
    """For each letter, True if it acts as a consonant.

    "y" is a consonant at the start ("yes") or after a vowel ("toy"), and a
    vowel after a consonant ("gym", "happy"). Any non-English character counts
    as a consonant, like in the reference implementation.
    """
    mask: list[bool] = []
    for i, ch in enumerate(word):
        if ch in _VOWELS:
            mask.append(False)
        elif ch == "y":
            mask.append(i == 0 or not mask[i - 1])
        else:
            mask.append(True)
    return mask


def measure(stem: str) -> int:
    """Count the VC sequences in ``stem``: the m in [C](VC)^m[V]."""
    mask = _consonant_mask(stem)
    m = 0
    for i in range(1, len(mask)):
        if mask[i] and not mask[i - 1]:  # a vowel followed by a consonant
            m += 1
    return m


def _has_vowel(stem: str) -> bool:
    return not all(_consonant_mask(stem))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _consonant_mask(word)[-1]


def _ends_cvc(word: str) -> bool:
    """consonant-vowel-consonant, where the last consonant is not w, x or y.

    Used to restore an "e" on short words: hop(e), lov(e), but not snow, box.
    """
    if len(word) < 3 or word[-1] in "wxy":
        return False
    mask = _consonant_mask(word)
    return mask[-1] and not mask[-2] and mask[-3]


def _step1ab(w: str) -> str:
    """Plurals, then -eed, -ed and -ing.

    caresses -> caress, ponies -> poni, cats -> cat, agreed -> agree,
    plastered -> plaster, motoring -> motor, hopping -> hop, filing -> file.
    """
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and w[-2] != "s":
        w = w[:-1]

    if w.endswith("eed"):
        return w[:-1] if measure(w[:-3]) > 0 else w
    for suffix in ("ed", "ing"):
        stem = w[: -len(suffix)]
        if w.endswith(suffix) and _has_vowel(stem):
            return _tidy_after_ed_ing(stem)
    return w


def _tidy_after_ed_ing(w: str) -> str:
    if w.endswith(("at", "bl", "iz")):
        return w + "e"  # conflat(ed) -> conflate
    if _ends_double_consonant(w) and w[-1] not in "lsz":
        return w[:-1]  # hopp(ing) -> hop, but fall(ing) -> fall
    if measure(w) == 1 and _ends_cvc(w):
        return w + "e"  # fil(ing) -> file
    return w


def _step1c(w: str) -> str:
    """happy -> happi, but sky -> sky (no vowel before the y)."""
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


# Step 2 maps double suffixes to single ones (-ization = -ize + -ation).
# Rules are grouped by the second-to-last letter, like the reference
# implementation, and only the first suffix that matches in a group is tried.
_STEP2: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "a": (("ational", "ate"), ("tional", "tion")),
    "c": (("enci", "ence"), ("anci", "ance")),
    "e": (("izer", "ize"),),
    "l": (("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous")),
    "o": (("ization", "ize"), ("ation", "ate"), ("ator", "ate")),
    "s": (("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous")),
    "t": (("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")),
    "g": (("logi", "log"),),
}

# Step 3 handles -ic-, -full, -ness and friends, grouped by the last letter.
_STEP3: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "e": (("icate", "ic"), ("ative", ""), ("alize", "al")),
    "i": (("iciti", "ic"),),
    "l": (("ical", "ic"), ("ful", "")),
    "s": (("ness", ""),),
}

# Step 4 removes -ant, -ence etc. when the stem is long enough (m > 1).
_STEP4: Final[dict[str, tuple[str, ...]]] = {
    "a": ("al",),
    "c": ("ance", "ence"),
    "e": ("er",),
    "i": ("ic",),
    "l": ("able", "ible"),
    "n": ("ant", "ement", "ment", "ent"),
    "o": ("ion", "ou"),
    "s": ("ism",),
    "t": ("ate", "iti"),
    "u": ("ous",),
    "v": ("ive",),
    "z": ("ize",),
}


def _apply_first_match(w: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Replace the first matching suffix if the remaining stem has m > 0.

    When a suffix matches but its stem is too short, the word is returned
    unchanged and no later rule in the group is tried.
    """
    for suffix, replacement in rules:
        if w.endswith(suffix):
            stem = w[: len(w) - len(suffix)]
            return stem + replacement if measure(stem) > 0 else w
    return w


def _step4(w: str) -> str:
    for suffix in _STEP4.get(w[-2], ()):
        if not w.endswith(suffix):
            continue
        stem = w[: len(w) - len(suffix)]
        if suffix == "ion" and not stem.endswith(("s", "t")):
            continue  # adoption -> adopt, but "-ion" alone is not a suffix
        return stem if measure(stem) > 1 else w
    return w


def _step5(w: str) -> str:
    """Remove a final -e (probate -> probat, rate -> rate) and -ll -> -l."""
    if w.endswith("e"):
        m = measure(w[:-1])
        if m > 1 or (m == 1 and not _ends_cvc(w[:-1])):
            w = w[:-1]
    if w.endswith("ll") and measure(w) > 1:
        w = w[:-1]  # controll -> control, but roll -> roll
    return w


def porter_stem(word: str) -> str:
    """Stem one lowercase word. Words of one or two letters are left alone."""
    if len(word) <= 2:
        return word
    w = _step1c(_step1ab(word))
    if len(w) >= 2:
        w = _apply_first_match(w, _STEP2.get(w[-2], ()))
        w = _apply_first_match(w, _STEP3.get(w[-1], ()))
        w = _step4(w)
    return _step5(w)
