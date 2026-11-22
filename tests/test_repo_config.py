"""Settings that live in several files must agree."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def compose_image() -> str:
    compose = (ROOT / "docker-compose.yml").read_text("utf-8")
    match = re.search(r"^\s+image: (elasticsearch:(\S+)@sha256:[0-9a-f]{64})$", compose, re.M)
    assert match, "docker-compose.yml must pin Elasticsearch by version and digest"
    return match.group(1)


@pytest.mark.parametrize(
    ("workflow", "services"),
    [("ci.yml", 2), ("release.yml", 1)],  # ci.yml: the integration tests and the benchmark
)
def test_ci_tests_against_the_same_elasticsearch_as_local_development(workflow, services):
    """Dependabot updates docker-compose.yml, not the workflows: keep them in step."""
    text = (ROOT / ".github" / "workflows" / workflow).read_text("utf-8")
    images = re.findall(r"^\s+image: (\S+)$", text, re.M)
    assert images == [compose_image()] * services


def test_the_release_runs_the_integration_tests_too():
    """A tag on a commit whose integration tests fail must not be released."""
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
    assert 'REQUIRE_ES: "1"' in release


def test_the_golden_file_was_recorded_with_that_elasticsearch():
    golden = json.loads((ROOT / "tests" / "data" / "es_parity.json").read_text("utf-8"))
    version = compose_image().split(":")[1].split("@")[0]
    assert golden["elasticsearch"] == version


def test_the_recorded_benchmark_was_run_on_that_elasticsearch():
    report = json.loads((ROOT / "benchmarks" / "scifact.json").read_text("utf-8"))
    version = compose_image().split(":")[1].split("@")[0]
    assert report["environment"]["elasticsearch"] == version
