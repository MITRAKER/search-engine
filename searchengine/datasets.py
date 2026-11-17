"""Loading documents.

Every document is a dict: {"id": str, "title": str, "body": str}.
JSON Lines (one JSON object per line) is used because it streams well:
you can index millions of docs without loading the whole file at once.
"""
import json
from pathlib import Path
from typing import Iterator

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.jsonl"


def load_jsonl(path: str | Path = SAMPLE) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
