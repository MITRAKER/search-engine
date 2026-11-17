"""Loading documents.

Every document is a flat dict of strings: {"id": ..., "title": ..., "body": ...}.
JSON Lines (one JSON object per line) is used because it streams: you can
index millions of documents without loading the whole file into memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

SAMPLE: Final = Path(__file__).resolve().parent / "data" / "sample.jsonl"

# A single line longer than this is rejected rather than read into memory.
MAX_LINE_BYTES: Final = 10_000_000


def load_jsonl(path: str | Path = SAMPLE) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-empty line, with clear errors."""
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{number}: line is longer than {MAX_LINE_BYTES} bytes")
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def load_scifact_corpus(path: str | Path) -> Iterator[dict[str, str]]:
    """Read SciFact's corpus.jsonl and turn each paper into our format.

    SciFact stores an abstract as a list of sentences; we join them.
    """
    for record in load_jsonl(path):
        yield {
            "id": str(record["doc_id"]),
            "title": str(record["title"]),
            "body": " ".join(str(sentence) for sentence in record["abstract"]),
        }
