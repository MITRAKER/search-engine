"""Settings that live in several files must agree."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compose_image() -> str:
    compose = (ROOT / "docker-compose.yml").read_text("utf-8")
    match = re.search(r"^\s+image: (elasticsearch:(\S+)@sha256:[0-9a-f]{64})$", compose, re.M)
    assert match, "docker-compose.yml must pin Elasticsearch by version and digest"
    return match.group(1)


def test_ci_tests_against_the_same_elasticsearch_as_local_development():
    """Dependabot updates docker-compose.yml, not the CI service: keep them in step."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert f"image: {compose_image()}" in ci


def test_the_golden_file_was_recorded_with_that_elasticsearch():
    golden = json.loads((ROOT / "tests" / "data" / "es_parity.json").read_text("utf-8"))
    version = compose_image().split(":")[1].split("@")[0]
    assert golden["elasticsearch"] == version
