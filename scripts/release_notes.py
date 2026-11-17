"""Print the CHANGELOG.md section for a release, or check a tag is ready.

    python scripts/release_notes.py v0.1.0                  # print the notes
    python scripts/release_notes.py --check-version v0.1.0  # verify before releasing

Used by .github/workflows/release.yml.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
CHANGELOG: Final = ROOT / "CHANGELOG.md"
INIT: Final = ROOT / "searchengine" / "__init__.py"
TAG_RE: Final = re.compile(r"^v(\d+\.\d+\.\d+)$")
VERSION_RE: Final = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)


def version_from_tag(tag: str) -> str:
    match = TAG_RE.match(tag)
    if match is None:
        raise ValueError(f"tag {tag!r} is not of the form vX.Y.Z")
    return match.group(1)


def package_version(init_source: str) -> str:
    match = VERSION_RE.search(init_source)
    if match is None:
        raise ValueError("__version__ not found in searchengine/__init__.py")
    return match.group(1)


def notes_for(version: str, changelog: str) -> str:
    """The text between '## [version]' and the next '## [' heading."""
    lines = changelog.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"## [{version}]")), None)
    if start is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    body = "\n".join(lines[start + 1 : end]).strip()
    if not body:
        raise ValueError(f"the CHANGELOG.md section for {version} is empty")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag, e.g. v0.1.0")
    parser.add_argument("--check-version", action="store_true", help="only verify, print nothing")
    args = parser.parse_args(argv)
    try:
        version = version_from_tag(args.tag)
        notes = notes_for(version, CHANGELOG.read_text(encoding="utf-8"))
        found = package_version(INIT.read_text(encoding="utf-8"))
        if found != version:
            raise ValueError(f"tag says {version} but searchengine.__version__ is {found}")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not args.check_version:
        print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
