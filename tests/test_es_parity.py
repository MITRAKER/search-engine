"""Our scores against Elasticsearch's own, recorded in a golden file.

tests/data/es_parity.json holds 42 documents with short, long and empty
fields, 36 queries (single words, several words, AND mode, phrases, five
field weightings) and the scores Elasticsearch 9.5.5 gave them. It also holds
the terms Elasticsearch's copy of our analyzer made from 17 tricky texts.
scripts/make_parity_golden.py recorded it from a real Elasticsearch, so these
tests need no server.

Long fields are the hard part: Lucene stores each length in one byte, so a
document of 41 or more terms only scores the same when we round its length
the same way (``SearchOptions.lucene_norms``).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from searchengine import InvertedIndex, SearchOptions, analyze_tokens
from searchengine.index import MAX_RESULTS

GOLDEN = json.loads((Path(__file__).parent / "data" / "es_parity.json").read_text("utf-8"))
CASES = GOLDEN["cases"]

# Lucene computes scores in 32-bit floats, which carry about 7 significant
# digits. Every score here agrees to better than 2e-7; a wrong length or
# statistic shows up as a difference of 1e-3 or more. (The documents here are
# at most about 4 times the average length. Far longer ones lose digits in
# Lucene's own float arithmetic: about 1e-6 at 20 times the average.)
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def golden_index() -> InvertedIndex:
    index = InvertedIndex()
    index.add_many(GOLDEN["corpus"])
    return index


def case_id(case: dict[str, object]) -> str:
    return f"{case['query']}|{case['weights']}|{case['mode']}"


def our_scores(
    index: InvertedIndex, case: dict[str, Any], *, lucene_norms: bool = True
) -> dict[str, float]:
    options = SearchOptions(mode=case["mode"], lucene_norms=lucene_norms)
    weights = case["weights"]
    hits = index.search(case["query"], k=MAX_RESULTS, options=options, field_weights=weights)
    return {hit.doc_id: hit.score for hit in hits}


def test_golden_file_covers_long_fields_and_every_kind_of_query():
    assert GOLDEN["elasticsearch"] == "9.5.5"
    longest = max(len(doc["body"].split()) for doc in GOLDEN["corpus"])
    assert longest > 1000  # far beyond the 40 terms Lucene stores exactly
    assert {case["mode"] for case in CASES} == {"or", "and"}
    assert any(case["query"].startswith('"') for case in CASES)
    assert sum(len(case["hits"]) for case in CASES) > 800
    assert len(GOLDEN["analyzer"]) >= 15


@pytest.mark.parametrize(
    ("text", "terms"), GOLDEN["analyzer"], ids=[f"text{n}" for n in range(len(GOLDEN["analyzer"]))]
)
def test_terms_match_elasticsearch(text, terms):
    """Casing, combining marks, emoji and the length limit: same terms, same positions."""
    assert [[token.term, token.position] for token in analyze_tokens(text)] == terms


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_scores_match_elasticsearch(golden_index, case):
    ours = our_scores(golden_index, case)
    theirs = dict(case["hits"])
    assert ours.keys() == theirs.keys()
    for doc_id, score in theirs.items():
        assert ours[doc_id] == pytest.approx(score, rel=TOLERANCE), doc_id


def test_exact_lengths_do_not_match_elasticsearch(golden_index):
    """The golden file can tell: without Lucene's rounding, long documents
    score differently. (This keeps the test above honest.)"""
    differences = []
    for case in CASES:
        ours = our_scores(golden_index, case, lucene_norms=False)
        differences += [abs(ours[doc_id] - score) / score for doc_id, score in case["hits"]]
    assert max(differences) > 0.01
