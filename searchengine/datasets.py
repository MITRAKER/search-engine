"""Loading documents.

Every document is a flat dict of strings: {"id": ..., "title": ..., "body": ...}.
JSON Lines (one JSON object per line) is used because it streams: you can
index millions of documents without loading the whole file into memory.

Judged queries, for measuring quality (see ``evaluation``), come from files
of the same kind.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Final

from .evaluation import JudgedQuery, judged_query

SAMPLE: Final = Path(__file__).resolve().parent / "data" / "sample.jsonl"
# Judged queries for the sample documents: which ones answer each query.
SAMPLE_QUERIES: Final = SAMPLE.with_name("sample_queries.jsonl")

# A line longer than this (counting its newline) is rejected, and it is never
# read into memory whole: lines are read with this limit (JPL rule 3).
MAX_LINE_BYTES: Final = 10_000_000

# SciFact (Wadden et al., EMNLP 2020), shared under CC BY-NC 2.0. It is
# downloaded rather than copied into this repository. The checksum pins the
# exact release, so every benchmark runs on the same data.
SCIFACT_URL: Final = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256: Final = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SCIFACT_FILES: Final = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")
MAX_DOWNLOAD_BYTES: Final = 50_000_000  # the archive is about 3 MB
DOWNLOAD_TIMEOUT: Final = 60  # seconds


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
            except RecursionError as error:  # json gives up on very deep nesting this way
                raise ValueError(f"{path}:{number}: the JSON is nested too deeply") from error
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


def load_judged_queries(path: str | Path = SAMPLE_QUERIES) -> Iterator[JudgedQuery]:
    """Read judged queries, one JSON object per line, in this format:

    {"id": "q1", "query": "words to search for", "relevant": {"7": 2, "12": 1}}

    ``relevant`` maps each relevant document's id to its grade: 1 for
    relevant, 2 for highly relevant. Documents not listed are not relevant.
    """
    for number, record in enumerate(load_jsonl(path), start=1):
        try:
            query = judged_query(record.get("id"), record.get("query"), record.get("relevant"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: query {number}: {error}") from error
        yield query


def load_scifact_claims(path: str | Path) -> Iterator[JudgedQuery]:
    """Read SciFact's claims (claims_train.jsonl or claims_dev.jsonl) as judged queries.

    Each claim is a query, and the abstracts it cites are its relevant
    documents, with grade 1. This is how the BEIR benchmark scores SciFact.
    BEIR's test set is the 300 dev claims, because SciFact's own test claims
    come without judgments.
    """
    for number, record in enumerate(load_jsonl(path), start=1):
        cited = record.get("cited_doc_ids")
        try:
            if not isinstance(cited, list):
                raise TypeError("cited_doc_ids must be a list")
            grades = dict.fromkeys(cited, 1)
            query = judged_query(record.get("id"), record.get("claim"), grades)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: claim {number}: {error}") from error
        yield query


def unpack_scifact(archive: bytes, dest: Path, sha256: str = SCIFACT_SHA256) -> None:
    """Check the archive's checksum, then write SciFact's files into ``dest``.

    Only the files we need are read, and they are written under names we
    choose, so an archive cannot write anywhere else (no "../" tricks).
    """
    digest = hashlib.sha256(archive).hexdigest()
    if digest != sha256:
        raise ValueError(f"the SciFact archive is not the expected one (sha256 {digest})")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for name in SCIFACT_FILES:
            try:
                member = tar.getmember(f"data/{name}")
            except KeyError as error:
                raise ValueError(f"the SciFact archive has no data/{name}") from error
            source = tar.extractfile(member) if member.isfile() else None
            if source is None or member.size > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"the SciFact archive's data/{name} is not a usable file")
            (dest / name).write_bytes(source.read(MAX_DOWNLOAD_BYTES))


def fetch_scifact(
    dest: str | Path, *, url: str = SCIFACT_URL, sha256: str = SCIFACT_SHA256
) -> Path:
    """Download SciFact into ``dest``, unless its files are already there."""
    dest = Path(dest)
    if all((dest / name).is_file() for name in SCIFACT_FILES):
        return dest
    if not url.startswith(("https://", "file://")):
        raise ValueError("SciFact must be downloaded over https")
    # The scheme is checked above, so this cannot open, say, an ftp:// URL.
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310
        archive = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(archive) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"the SciFact download is larger than {MAX_DOWNLOAD_BYTES} bytes")
    unpack_scifact(archive, dest, sha256)
    return dest
