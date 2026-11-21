"""Tests for scripts/release_notes.py, which the release workflow relies on."""

import pytest

import searchengine
from scripts import release_notes

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


def test_main_refuses_a_tag_that_is_not_the_package_version(tmp_path, monkeypatch, capsys):
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "0.0.9"\n')
    monkeypatch.setattr(release_notes, "INIT", init)
    assert release_notes.main([f"v{searchengine.__version__}", "--check-version"]) == 1
    assert "but searchengine.__version__ is 0.0.9" in capsys.readouterr().err
