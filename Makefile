# Shortcuts for the commands in CONTRIBUTING.md. Run `make help` to list them.
.PHONY: help install lint format types jpl test integration bench scifact audit hygiene check

help:  ## list the available commands
	@awk -F ':.*## ' '/^[a-z]+:.*## / {printf "%-12s %s\n", $$1, $$2}' Makefile

install:  ## install the project and the dev tools, and the git hooks
	python -m pip install -c requirements-dev.txt -e ".[dev]"
	pre-commit install

lint:  ## find likely bugs with ruff
	ruff check .

format:  ## format the code with ruff
	ruff format .

types:  ## strict type checks with mypy
	mypy

jpl:  ## check the NASA/JPL Power of Ten rules
	python scripts/check_jpl.py searchengine scripts tests

test:  ## run the tests with coverage (not the integration tests)
	pytest -m "not integration" --cov --cov-fail-under=95

integration:  ## both engines on a live Elasticsearch (first: docker compose up -d --wait)
	REQUIRE_ES=1 pytest -m integration

bench:  ## measure our engine's quality and speed on the sample queries
	searchengine-bench

scifact:  ## both engines on SciFact, checked against benchmarks/scifact.json (needs Elasticsearch)
	searchengine-bench --scifact datasets/scifact --download --engine both --check benchmarks/scifact.json

audit:  ## look for dependencies with known vulnerabilities
	pip-audit --strict -r requirements-dev.txt --no-deps --disable-pip

hygiene:  ## the other pre-commit checks (whitespace, YAML, large files, keys)
	SKIP=ruff-check,ruff-format,mypy,jpl-rules pre-commit run --all-files

check: lint types jpl test hygiene  ## the main CI checks, on one Python version (not the audit or integration)
	ruff format --check .
