# Shortcuts for the commands in CONTRIBUTING.md. Run `make help` to list them.
.PHONY: help install lint format types jpl test audit hygiene check

help:  ## list the available commands
	@awk -F ':.*## ' '/^[a-z]+:.*## / {printf "%-8s %s\n", $$1, $$2}' Makefile

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

test:  ## run the tests with coverage
	pytest --cov --cov-fail-under=95

audit:  ## look for dependencies with known vulnerabilities
	pip-audit --strict -r requirements-dev.txt --no-deps --disable-pip

hygiene:  ## the other pre-commit checks (whitespace, YAML, large files, keys)
	SKIP=ruff-check,ruff-format,mypy,jpl-rules pre-commit run --all-files

check: lint types jpl test hygiene  ## everything CI checks (except the audit)
	ruff format --check .
