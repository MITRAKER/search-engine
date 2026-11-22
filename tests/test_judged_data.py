"""Judged queries for the benchmark: the sample queries, files of judged queries
and SciFact claims, and downloading and unpacking SciFact safely, offline."""

import gzip
import hashlib
import http.client
import io
import json
import re
import socket
import tarfile
import urllib.request
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NoReturn

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from searchengine.datasets import (
    DOWNLOAD_TIMEOUT,
    SCIFACT_FILES,
    SCIFACT_SHA256,
    SCIFACT_URL,
    fetch_scifact,
    load_jsonl,
    load_judged_queries,
    load_scifact_claims,
    unpack_scifact,
)
from searchengine.evaluation import MAX_GRADE, JudgedQuery


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a test here fail, before anything is sent, if it tries to reach the
    network: its downloads must be file:// URLs or fakes."""

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("these tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# ---------- the sample queries ----------


def test_load_judged_queries_reads_the_twelve_sample_queries_by_default():
    queries = list(load_judged_queries())
    assert [query.id for query in queries] == [f"q{number}" for number in range(1, 13)]
    # Copied by hand from searchengine/data/sample_queries.jsonl.
    first = "how does a search engine avoid scanning every document"
    assert queries[0] == JudgedQuery("q1", first, {"1": 2, "15": 1, "20": 1})
    assert queries[11] == JudgedQuery("q12", "shoes for running", {"22": 2})


def test_the_sample_queries_judge_sample_documents_as_relevant_or_highly_relevant():
    documents = {document["id"] for document in load_jsonl()}
    judgments = [item for query in load_judged_queries() for item in query.grades.items()]
    assert {doc_id for doc_id, _ in judgments} <= documents
    assert {grade for _, grade in judgments} == {1, 2}


# ---------- files of judged queries ----------

GOOD_QUERY = {"id": "q1", "query": "inverted index", "relevant": {"1": 2}}


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"id": "q2", "query": "bm25"}, "at least one relevant document"),
        ({"id": "q2", "query": "bm25", "relevant": {}}, "at least one relevant document"),
        ({"id": "q2", "query": "bm25", "relevant": ["2", "3"]}, "at least one relevant document"),
        ({"id": "q2", "query": "bm25", "relevant": {"2": 0}}, "grades must be integers from 1"),
        ({"id": "q2", "query": "bm25", "relevant": {"2": True}}, "grades must be integers from 1"),
        ({"id": "q2", "query": "bm25", "relevant": {"2": "1"}}, "grades must be integers from 1"),
        ({"id": "q2", "query": " \t", "relevant": {"2": 1}}, "needs non-empty text"),
        ({"query": "bm25", "relevant": {"2": 1}}, "query id must be"),
    ],
)
def test_load_judged_queries_says_which_query_is_wrong(tmp_path, record, reason):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(GOOD_QUERY) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=reason) as raised:
        list(load_judged_queries(path))
    assert str(raised.value).startswith(f"{path}: query 2: ")


@settings(max_examples=50, deadline=None)
@given(
    st.one_of(st.integers(min_value=0, max_value=10**9), st.text(min_size=1, max_size=8)),
    st.text(min_size=1, max_size=40).filter(str.strip),
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.integers(min_value=1, max_value=MAX_GRADE),
        min_size=1,
        max_size=5,
    ),
)
def test_a_judged_query_reads_back_as_written(tmp_path_factory, query_id, text, grades):
    path = tmp_path_factory.mktemp("queries") / "queries.jsonl"
    path.write_text(json.dumps({"id": query_id, "query": text, "relevant": grades}) + "\n")
    assert list(load_judged_queries(path)) == [JudgedQuery(str(query_id), text, grades)]


# ---------- SciFact's claims ----------

GOOD_CLAIM: dict[str, object] = {
    "id": 1,
    "claim": "Aspirin lowers the risk of a first stroke.",
    "evidence": {"31715818": [{"sentences": [4, 5], "label": "SUPPORT"}]},
    "cited_doc_ids": [31715818],
}


def test_load_scifact_claims_makes_each_cited_abstract_relevant_with_grade_1(tmp_path):
    path = tmp_path / "claims_dev.jsonl"
    # Cited without evidence: still relevant, because only cited_doc_ids counts.
    no_evidence = {"id": 3, "claim": "Coffee causes cancer.", "cited_doc_ids": [14717500, 4983]}
    path.write_text(json.dumps(GOOD_CLAIM) + "\n" + json.dumps(no_evidence | {"evidence": {}}))
    assert list(load_scifact_claims(path)) == [
        JudgedQuery("1", "Aspirin lowers the risk of a first stroke.", {"31715818": 1}),
        JudgedQuery("3", "Coffee causes cancer.", {"14717500": 1, "4983": 1}),
    ]


def test_an_abstract_cited_twice_counts_once(tmp_path):
    path = tmp_path / "claims_dev.jsonl"
    path.write_text(json.dumps(GOOD_CLAIM | {"cited_doc_ids": [4983, 31715818, 4983]}) + "\n")
    [claim] = load_scifact_claims(path)
    assert dict(claim.grades) == {"4983": 1, "31715818": 1}


def test_load_scifact_claims_skips_blank_lines(tmp_path):
    path = tmp_path / "claims_dev.jsonl"
    second = GOOD_CLAIM | {"id": 2}
    path.write_text(f"\n{json.dumps(GOOD_CLAIM)}\n\n  \t\n{json.dumps(second)}\n\n")
    assert [claim.id for claim in load_scifact_claims(path)] == ["1", "2"]


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"id": 2, "claim": "C.", "evidence": {}}, "cited_doc_ids must be a list"),
        # A string or a dict could be iterated, into ids that were never cited.
        ({"id": 2, "claim": "C.", "cited_doc_ids": "4983"}, "cited_doc_ids must be a list"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": {"4983": 1}}, "cited_doc_ids must be a list"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": 4983}, "cited_doc_ids must be a list"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": []}, "at least one relevant document"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": [[4983]]}, "unhashable"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": [True]}, "document id must be a str or int"),
        ({"id": 2, "claim": "C.", "cited_doc_ids": [4983.0]}, "document id must be a str or int"),
        ({"id": 2, "claim": "  ", "cited_doc_ids": [4983]}, "needs non-empty text"),
        ({"claim": "C.", "cited_doc_ids": [4983]}, "query id must be"),
    ],
)
def test_load_scifact_claims_says_which_claim_is_wrong(tmp_path, record, reason):
    path = tmp_path / "claims_dev.jsonl"
    # Claim 2 is on line 3: a blank line is not a claim.
    path.write_text(json.dumps(GOOD_CLAIM) + "\n\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=reason) as raised:
        list(load_scifact_claims(path))
    assert str(raised.value).startswith(f"{path}: claim 2: ")


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=8))
def test_every_cited_id_becomes_a_str_id_of_grade_1(tmp_path_factory, cited):
    path = tmp_path_factory.mktemp("claims") / "claims_dev.jsonl"
    path.write_text(json.dumps(GOOD_CLAIM | {"cited_doc_ids": cited}) + "\n")
    [claim] = load_scifact_claims(path)
    assert dict(claim.grades) == {str(doc_id): 1 for doc_id in cited}


# ---------- SciFact archives, built in memory ----------

# One record per file, in SciFact's formats, by the name it is saved under.
SCIFACT_DATA = {
    "corpus.jsonl": b'{"doc_id": 4983, "title": "A title", "abstract": ["One.", "Two."]}\n',
    "claims_train.jsonl": b'{"id": 1, "claim": "A claim.", "cited_doc_ids": [4983]}\n',
    "claims_dev.jsonl": b'{"id": 3, "claim": "Another claim.", "cited_doc_ids": [4983]}\n',
}

Member = tuple[tarfile.TarInfo, bytes]


def regular_file(name: str, data: bytes) -> Member:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def special_file(name: str, kind: bytes, linkname: str = "") -> Member:
    """A directory, a link, or another member without data of its own."""
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info, b""


def scifact_files(data: Mapping[str, bytes] = SCIFACT_DATA) -> list[Member]:
    return [regular_file(f"data/{name}", contents) for name, contents in data.items()]


def tar_bytes(members: Iterable[Member]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def tar_gz(members: Iterable[Member]) -> bytes:
    """A .tar.gz archive, the same bytes on every run (its gzip mtime is 0)."""
    return gzip.compress(tar_bytes(members), mtime=0)


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_all(directory: Path) -> dict[str, bytes]:
    """What unpacking wrote: every file in the directory, by name."""
    return {path.name: path.read_bytes() for path in directory.iterdir()}


# Laid out like SciFact's data.tar.gz: a data/ directory with four files, of
# which three are needed.
ARCHIVE = tar_gz(
    [
        special_file("data", tarfile.DIRTYPE),
        *scifact_files(),
        regular_file("data/claims_test.jsonl", b'{"id": 2, "claim": "Not judged."}\n'),
    ]
)


def without_end_blocks(members: Iterable[Member]) -> bytes:
    """A tar archive's members, without the zero blocks that mark its end."""
    blocks = (info.tobuf() + data + bytes(-len(data) % tarfile.BLOCKSIZE) for info, data in members)
    return b"".join(blocks)


def with_wrong_crc(archive: bytes) -> bytes:
    """A gzip file with one bit of its CRC flipped: its last 8 bytes are CRC and size."""
    return archive[:-8] + bytes([archive[-8] ^ 1]) + archive[-7:]


def damaged_after(data: bytes) -> bytes:
    """``data`` gzipped, then a deflate block that no decompressor accepts."""
    stream = zlib.compressobj(wbits=31)  # 31 means deflate data in a gzip header
    # The flush ends the data on a byte boundary, so the next byte starts a new
    # block: 0b111 is the last block, of type 3, which deflate reserves.
    return stream.compress(data) + stream.flush(zlib.Z_SYNC_FLUSH) + b"\x07"


# ---------- unpacking ----------


def test_unpack_scifact_writes_the_three_files_it_needs_under_its_own_names(tmp_path):
    dest = tmp_path / "datasets" / "scifact"  # made, with its parent
    unpack_scifact(ARCHIVE, dest, checksum(ARCHIVE))
    assert read_all(dest) == SCIFACT_DATA  # no claims_test.jsonl, no data/ directory


def test_unpack_scifact_checks_the_checksum_before_writing_anything(tmp_path):
    dest = tmp_path / "scifact"
    message = re.escape(f"not the expected one (sha256 {checksum(ARCHIVE)})")
    with pytest.raises(ValueError, match=message):
        unpack_scifact(ARCHIVE, dest, "0" * 64)
    with pytest.raises(ValueError, match=message):
        unpack_scifact(ARCHIVE, dest)  # by default, against the pinned release
    assert not dest.exists()


@pytest.mark.parametrize("missing", SCIFACT_FILES)
def test_unpack_scifact_needs_each_of_the_three_files(tmp_path, missing):
    archive = tar_gz(member for member in scifact_files() if member[0].name != f"data/{missing}")
    with pytest.raises(ValueError, match=re.escape(f"has no data/{missing}")):
        unpack_scifact(archive, tmp_path / "scifact", checksum(archive))


@pytest.mark.parametrize(
    "corpus",
    [
        pytest.param(special_file("data/corpus.jsonl", tarfile.DIRTYPE), id="a directory"),
        # Links that tarfile would follow to a file in the archive: never written.
        pytest.param(
            special_file("data/corpus.jsonl", tarfile.SYMTYPE, "claims_dev.jsonl"),
            id="a symlink to a file in the archive",
        ),
        pytest.param(
            special_file("data/corpus.jsonl", tarfile.LNKTYPE, "data/claims_dev.jsonl"),
            id="a hard link",
        ),
        pytest.param(
            special_file("data/corpus.jsonl", tarfile.SYMTYPE, "/etc/passwd"),
            id="a symlink out of the archive",
        ),
    ],
)
def test_unpack_scifact_refuses_a_member_that_is_not_a_regular_file(tmp_path, corpus):
    claims = [member for member in scifact_files() if member[0].name != "data/corpus.jsonl"]
    archive = tar_gz([*claims, corpus])  # after the files that the links point to
    with pytest.raises(ValueError, match=r"data/corpus\.jsonl is not a usable file"):
        unpack_scifact(archive, tmp_path / "scifact", checksum(archive))
    assert not (tmp_path / "scifact" / "corpus.jsonl").exists()


def test_unpack_scifact_takes_files_of_exactly_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("searchengine.datasets.MAX_DOWNLOAD_BYTES", 10)
    archive = tar_gz(scifact_files(dict.fromkeys(SCIFACT_FILES, b"0123456789")))
    unpack_scifact(archive, tmp_path / "scifact", checksum(archive))
    assert read_all(tmp_path / "scifact") == dict.fromkeys(SCIFACT_FILES, b"0123456789")


def test_unpack_scifact_refuses_a_file_over_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("searchengine.datasets.MAX_DOWNLOAD_BYTES", 10)
    archive = tar_gz(scifact_files(dict.fromkeys(SCIFACT_FILES, b"0123456789+")))
    with pytest.raises(ValueError, match=r"data/corpus\.jsonl is not a usable file"):
        unpack_scifact(archive, tmp_path / "scifact", checksum(archive))
    assert not (tmp_path / "scifact" / "corpus.jsonl").exists()


def test_unpack_scifact_never_writes_outside_dest(tmp_path):
    dest = tmp_path / "parent" / "scifact"
    hostile = [
        regular_file("../escape.txt", b"up one"),
        regular_file("data/../../up.txt", b"up two, then one"),
        regular_file(str(dest.parent / "absolute.txt"), b"an absolute path"),
        special_file("data/link", tarfile.SYMTYPE, "../.."),
    ]
    archive = tar_gz([*scifact_files(), *hostile])
    unpack_scifact(archive, dest, checksum(archive))
    assert list(dest.parent.iterdir()) == [dest]
    assert read_all(dest) == SCIFACT_DATA


@pytest.mark.parametrize(
    "archive",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"<html><body>403 Forbidden</body></html>", id="an error page"),
        pytest.param(tar_bytes(scifact_files()), id="an uncompressed tar"),
        pytest.param(gzip.compress(b"plain text\n" * 100, mtime=0), id="gzip but not tar"),
        # Each case below fails in another library, with its own kind of error.
        pytest.param(ARCHIVE[:20], id="cut short (gzip: EOFError)"),
        # gzip checks the CRC at the very end, which tarfile reads only when
        # the archive lacks the zero blocks that end it.
        pytest.param(
            with_wrong_crc(gzip.compress(without_end_blocks(scifact_files()), mtime=0)),
            id="wrong CRC (gzip: BadGzipFile)",
        ),
        # zlib.error gets out of tarfile only past what was read when the archive
        # was opened (gzip reads ahead, up to 128 kB): hence 300 kB.
        pytest.param(
            damaged_after(tar_bytes([regular_file("data/corpus.jsonl", bytes(300_000))])),
            id="damaged deflate data (zlib.error)",
        ),
    ],
)
def test_unpack_scifact_says_when_the_bytes_are_not_a_tar_gz_file(tmp_path, archive):
    with pytest.raises(ValueError, match=r"not a valid \.tar\.gz file"):
        unpack_scifact(archive, tmp_path / "scifact", checksum(archive))


@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=len(ARCHIVE) - 1),
    st.integers(min_value=0, max_value=255),
    st.booleans(),
)
def test_a_damaged_archive_gives_a_value_error_or_all_three_files(
    tmp_path_factory, position, value, cut
):
    """However the archive is damaged (its checksum made to match), the error
    is a ValueError, never one from tarfile, gzip or zlib."""
    damaged = ARCHIVE[:position]
    if not cut:
        damaged += bytes([value]) + ARCHIVE[position + 1 :]
    dest = tmp_path_factory.mktemp("scifact")
    try:
        unpack_scifact(damaged, dest, checksum(damaged))
    except ValueError:
        return
    # Damage that goes unnoticed (in the gzip trailer, say) writes all three.
    assert sorted(path.name for path in dest.iterdir()) == sorted(SCIFACT_FILES)


# ---------- downloading ----------


def refuse_to_open(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("fetch_scifact opened a URL")


def served(directory: Path, archive: bytes = ARCHIVE) -> str:
    """A file:// URL that downloads the archive, as a server would."""
    path = directory / "data.tar.gz"
    path.write_bytes(archive)
    return path.as_uri()


def test_the_pinned_release_is_downloaded_over_https_and_checked_by_sha256():
    assert SCIFACT_URL.startswith("https://")
    # Lowercase hex, because it is compared with hexdigest() as a string.
    assert re.fullmatch("[0-9a-f]{64}", SCIFACT_SHA256)


def test_fetch_scifact_downloads_and_unpacks_the_archive(tmp_path):
    dest = tmp_path / "datasets" / "scifact"
    assert fetch_scifact(dest, url=served(tmp_path), sha256=checksum(ARCHIVE)) == dest
    assert read_all(dest) == SCIFACT_DATA


def test_fetch_scifact_by_default_downloads_the_pinned_release_with_a_timeout(
    tmp_path, monkeypatch
):
    opened: list[tuple[str, float]] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        opened.append((url, timeout))
        return io.BytesIO(ARCHIVE)  # not the pinned release, so its checksum is wrong

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="not the expected one"):
        fetch_scifact(tmp_path / "scifact")
    assert opened == [(SCIFACT_URL, DOWNLOAD_TIMEOUT)]
    assert not (tmp_path / "scifact").exists()


def test_fetch_scifact_does_not_download_files_it_already_has(tmp_path, monkeypatch):
    dest = tmp_path / "scifact"
    dest.mkdir()
    for name in SCIFACT_FILES:
        (dest / name).write_bytes(b"downloaded before\n")
    monkeypatch.setattr(urllib.request, "urlopen", refuse_to_open)
    assert fetch_scifact(str(dest), url=served(tmp_path), sha256=checksum(ARCHIVE)) == dest
    assert read_all(dest) == dict.fromkeys(SCIFACT_FILES, b"downloaded before\n")


def test_fetch_scifact_downloads_again_if_a_file_is_missing(tmp_path):
    dest = tmp_path / "scifact"
    dest.mkdir()
    (dest / "corpus.jsonl").write_bytes(b"downloaded before\n")
    (dest / "claims_train.jsonl").write_bytes(b"downloaded before\n")
    fetch_scifact(dest, url=served(tmp_path), sha256=checksum(ARCHIVE))
    assert read_all(dest) == SCIFACT_DATA


@pytest.mark.parametrize(
    "url",
    ["http://scifact.invalid/data.tar.gz", "ftp://scifact.invalid/data.tar.gz", "data.tar.gz"],
)
def test_fetch_scifact_refuses_urls_that_are_not_https_before_opening_them(
    tmp_path, monkeypatch, url
):
    monkeypatch.setattr(urllib.request, "urlopen", refuse_to_open)
    with pytest.raises(ValueError, match="must be downloaded over https"):
        fetch_scifact(tmp_path / "scifact", url=url)
    assert not (tmp_path / "scifact").exists()


class CutShort(io.BytesIO):
    """A download whose connection closes before the end."""

    def read(self, size: int | None = -1) -> bytes:
        raise http.client.IncompleteRead(b"partial", 100)


def test_fetch_scifact_reports_a_broken_download_as_an_os_error(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: CutShort())
    with pytest.raises(OSError, match=r"the SciFact download failed: IncompleteRead\("):
        fetch_scifact(tmp_path / "scifact")
    assert not (tmp_path / "scifact").exists()


def test_fetch_scifact_takes_a_download_of_exactly_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("searchengine.datasets.MAX_DOWNLOAD_BYTES", len(ARCHIVE))
    fetch_scifact(tmp_path / "scifact", url=served(tmp_path), sha256=checksum(ARCHIVE))
    assert read_all(tmp_path / "scifact") == SCIFACT_DATA


def test_fetch_scifact_refuses_a_download_over_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("searchengine.datasets.MAX_DOWNLOAD_BYTES", len(ARCHIVE) - 1)
    with pytest.raises(ValueError, match=f"larger than {len(ARCHIVE) - 1} bytes"):
        fetch_scifact(tmp_path / "scifact", url=served(tmp_path), sha256=checksum(ARCHIVE))
    assert not (tmp_path / "scifact").exists()
