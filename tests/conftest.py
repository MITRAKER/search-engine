from __future__ import annotations

import pytest

from searchengine import InvertedIndex
from searchengine.datasets import load_jsonl


@pytest.fixture(scope="session")
def sample_index() -> InvertedIndex:
    """The 25-document sample dataset, indexed once for the whole test run."""
    index = InvertedIndex()
    index.add_many(load_jsonl())
    return index
