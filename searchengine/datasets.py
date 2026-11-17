"""Loading documents.

Every document is a flat dict of strings: {"id": ..., "title": ..., "body": ...}.
JSON Lines (one JSON object per line) is used because it streams: you can
index millions of documents without loading the whole file into memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Final

SAMPLE: Final = Path(__file__).resolve().parent / "data" / "sample.jsonl"

# A line longer than this (counting its newline) is rejected, and it is never
# read into memory whole: lines are read with this limit (JPL rule 3).
MAX_LINE_BYTES: Final = 10_000_000


def _limited_lines(file: IO[bytes]) -> Iterator[bytes]:
    """Each line, cut at MAX_LINE_BYTES + 1 bytes so an overlong one shows."""
    line = file.readline(MAX_LINE_BYTES + 1)
    while line:  # ends at the end of the file
        yield line
        line = file.readline(MAX_LINE_BYTES + 1)


def load_jsonl(path: str | Path = SAMPLE) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-empty line, with errors that say where."""
    with Path(path).open("rb") as file:
        for number, raw in enumerate(_limited_lines(file), start=1):
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{number}: line is longer than {MAX_LINE_BYTES} bytes")
            try:
                text = raw.decode("utf-8")
                record = json.loads(text) if text.strip() else None
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{number}: not valid UTF-8") from error
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if record is None:
                continue
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def load_scifact_corpus(path: str | Path) -> Iterator[dict[str, str]]:
    """Read SciFact's corpus.jsonl and turn each paper into our format.

    SciFact stores an abstract as a list of sentences; we join them.
    """
    for number, record in enumerate(load_jsonl(path), start=1):
        doc_id, abstract = record.get("doc_id"), record.get("abstract")
        if (
            isinstance(doc_id, bool)
            or not isinstance(doc_id, (int, str))
            or not isinstance(record.get("title"), str)
            or not isinstance(abstract, list)
            or not all(isinstance(sentence, str) for sentence in abstract)
        ):
            raise ValueError(f"{path}: record {number} is not a SciFact document")
        yield {
            "id": str(doc_id),
            "title": record["title"],
            "body": " ".join(abstract),
        }
