"""Text analysis: turning raw text into searchable terms.

Pipeline:  raw text -> tokenize -> lowercase -> drop stopwords -> stem

The SAME analyzer must run on documents (at index time) and on queries
(at search time). If they differ, "Running" in a query would never match
"run" in the index.
"""
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words carry almost no meaning for ranking.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its
    of on or our she so that the their them they this to was we were what when
    where which who will with you your""".split()
)

# Ordered longest-first so "ies" is tried before "s".
_SUFFIXES = [
    ("ational", "ate"), ("ization", "ize"), ("fulness", "ful"),
    ("iveness", "ive"), ("ousness", "ous"), ("ingly", ""), ("edly", ""),
    ("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", ""),
]


def stem(word: str) -> str:
    """A tiny suffix-stripping stemmer (a simplified cousin of Porter).

    It maps word variants to one root: "running", "runs" -> "run".
    The root doesn't have to be a real word, it only has to be consistent.
    """
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    # "runn" -> "run": collapse a doubled final consonant left by "-ing"/"-ed".
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


def analyze(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Full pipeline. Returns terms in order (order matters for phrase search)."""
    return [
        stem(tok)
        for tok in tokenize(text)
        if keep_stopwords or tok not in STOPWORDS
    ]
