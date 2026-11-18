"""Tests for scripts/release_notes.py, which the release workflow relies on."""

import importlib.util
import sys
from pathlib import Path

import pytest

import searchengine

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert spec is not None
assert spec.loader is not None
release_notes = importlib.util.module_from_spec(spec)
sys.modules["release_notes"] = release_notes
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0] - 2026-12-01

### Added
- Elasticsearch.

## [0.1.0] - 2026-11-17

### Added
- The engine.
"""


def test_notes_for_a_version():
    assert release_notes.notes_for("0.2.0", CHANGELOG) == "### Added\n- Elasticsearch."
    assert release_notes.notes_for("0.1.0", CHANGELOG) == "### Added\n- The engine."


def test_missing_or_empty_sections_are_errors():
    with pytest.raises(ValueError, match="no section"):
        release_notes.notes_for("9.9.9", CHANGELOG)
    with pytest.raises(ValueError, match="empty"):
        release_notes.notes_for("0.0.1", "## [0.0.1] - 2026-01-01\n\n## [0.0.0]\n")


@pytest.mark.parametrize("tag", ["0.1.0", "v1.2", "v1.2.3-rc1", "release"])
def test_bad_tags(tag):
    with pytest.raises(ValueError, match="not of the form"):
        release_notes.version_from_tag(tag)


def test_package_version():
    assert release_notes.package_version('x = 1\n__version__ = "1.2.3"\n') == "1.2.3"
    with pytest.raises(ValueError, match="not found"):
        release_notes.package_version("nothing here")


def test_main_with_the_real_changelog(capsys):
    tag = f"v{searchengine.__version__}"
    assert release_notes.main([tag, "--check-version"]) == 0
    assert release_notes.main([tag]) == 0
    assert "### Added" in capsys.readouterr().out
    assert release_notes.main(["v99.0.0"]) == 1
