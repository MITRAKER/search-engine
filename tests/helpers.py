"""Small helpers shared by the tests."""

from __future__ import annotations

from searchengine import InvertedIndex


def make_index(*bodies: str, titles: tuple[str, ...] = ()) -> InvertedIndex:
    """Build a tiny index whose documents are numbered from 1."""
    index = InvertedIndex()
    for number, body in enumerate(bodies, start=1):
        title = titles[number - 1] if number <= len(titles) else ""
        index.add({"id": str(number), "title": title, "body": body})
    return index
