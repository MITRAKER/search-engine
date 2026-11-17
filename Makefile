# Shortcuts for the commands in CONTRIBUTING.md. Run `make help` to list them.
.PHONY: help install lint format types jpl test audit check

help:  ## list the available commands
	@grep -E "^[a-z]+:.*##" Makefile | sed "s/:.*## /\t/"

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

check: lint types jpl test  ## everything CI checks (except the audit)
	ruff format --check .
