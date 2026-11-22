# Contributing

Thanks for helping! This guide covers setup, the checks your change must
pass, and how changes get merged and released.

## Setup

You need Python 3.11 or newer, and Docker if you work on the Elasticsearch
parts.

```bash
git clone https://github.com/MITRAKER/search-engine.git
cd search-engine
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -c requirements-dev.txt -e ".[dev]"
pre-commit install                 # run the checks before every commit
```

## The checks

Run the main checks locally before you push:

```bash
make check   # lint, format, types, JPL rules, tests and hygiene
make audit   # dependencies with known vulnerabilities
```

`make check` uses one Python version and leaves out the integration tests
and the benchmark. CI also runs the tests on Python 3.11 to 3.14, the
integration tests and the SciFact benchmark on a real Elasticsearch, builds
and installs the package, and runs CodeQL and GitHub's dependency review, so
a pull request can still fail on something `make check` passed. To run the
checks one at a time:

| Check | Command | What it catches |
|---|---|---|
| Lint | `ruff check .` | likely bugs, unused code, security smells |
| Format | `ruff format .` | inconsistent style (it fixes it for you) |
| Types | `mypy` | type errors, unhandled `None`, wrong arguments |
| JPL rules | `python scripts/check_jpl.py searchengine scripts tests` | recursion, unbounded loops, long functions, shared state, dynamic code |
| Tests | `pytest -m "not integration" --cov --cov-fail-under=95` | broken behavior; coverage must stay at 95% or higher |
| Integration | `make integration` | the two engines disagreeing on a real Elasticsearch (see below) |
| Benchmark | `make scifact` | search quality on SciFact moving, in either engine (see below) |
| Hygiene | `make hygiene` | trailing whitespace, broken YAML or JSON, conflict markers, large files, private keys |
| Audit | `make audit` | dependencies with known vulnerabilities |

Code must follow the [coding standard](docs/CODING_STANDARD.md), our
adaptation of NASA/JPL's "Power of Ten" rules.

### Golden files

Some tests compare our output with answers recorded from Elasticsearch
(`tests/data/es_parity.json`), so they run without it. The file holds
**Elasticsearch's** answers, not ours, so recording it again never makes a
failing parity test pass. Record it again only when what goes into it
changes: the corpus or the cases in `scripts/make_parity_golden.py`, the
Elasticsearch version, or the index settings and queries in
`searchengine/es_dsl.py`, which the script uses. If you change our analyzer,
first change its Elasticsearch copy, `analysis_settings()` in that file, to
match. If you change the scoring on purpose so that it departs from
Elasticsearch, the parity tests should fail, and the difference needs a
documented reason.

```bash
docker compose up -d --wait                # Elasticsearch on localhost:9200
python scripts/make_parity_golden.py       # rewrites tests/data/es_parity.json
git diff --stat tests/data                 # review what changed
```

`tests/data/trec_eval.json` holds trec_eval's own scores for random
rankings, so `tests/test_evaluation.py` can check our metrics against the
program research papers use. Record it again only when the cases in
`scripts/make_trec_eval_golden.py` change:

```bash
pip install pytrec_eval-terrier==0.5.11    # trec_eval, as a Python module
python scripts/make_trec_eval_golden.py    # rewrites tests/data/trec_eval.json
```

### Integration tests

`tests/test_es_integration.py` runs both engines side by side on a real
Elasticsearch, and checks that they find the same documents with the same
scores. Without a server these tests are skipped. CI runs them in their own
job, with `REQUIRE_ES=1`, which turns a missing server into a failure.

```bash
docker compose up -d --wait   # Elasticsearch on localhost:9200, or set ES_URL
make integration              # REQUIRE_ES=1 pytest -m integration
```

To move to a new Elasticsearch version, change the image in
`docker-compose.yml`, `.github/workflows/ci.yml` (both jobs that use it) and
`.github/workflows/release.yml` (a test checks that they match). Then record
the golden file again with `python scripts/make_parity_golden.py`, and the
benchmark report as described below. Tests check that both name the new
version.

### The benchmark

CI runs both engines on SciFact and compares every quality score with the
recorded report, `benchmarks/scifact.json`. A change that moves any of them
by more than 0.0001 fails. Speed is reported but not checked, because it
depends on the machine.

```bash
docker compose up -d --wait   # Elasticsearch on localhost:9200, or set ES_URL
make scifact                  # downloads SciFact into datasets/scifact the first time
make bench                    # our engine alone, on the sample data
```

If your change is meant to move quality, a better analyzer for example,
record the report again in the same pull request, and say in its description
what moved and why. A new or renamed system needs a new recording too.

```bash
searchengine-bench --scifact datasets/scifact --download --engine both \
    --json benchmarks/scifact.json --markdown benchmarks/scifact.md
git diff benchmarks/                       # review what changed
```

Then update the results in `README.md` and in `LEARNING.md` chapter 5.

## Making a change

1. Open an issue first for anything bigger than a small fix, so we can agree
   on the approach.
2. Create a branch from `main`: `git switch -c fix/phrase-scoring`.
3. Write a test that fails without your change, then make it pass.
4. Update `CHANGELOG.md` under **Unreleased**, and `LEARNING.md` if you
   changed how something works.
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat: add synonym expansion`, `fix: phrase gap after stopwords`,
   `docs: explain BM25F`, `test: ...`, `refactor: ...`, `ci: ...`.
6. Push and open a pull request. Fill in the template.
7. A pull request is merged when CI is green and it has been reviewed.

## Releasing (maintainers)

1. Move the **Unreleased** notes in `CHANGELOG.md` under a new version heading.
2. Bump `__version__` in `searchengine/__init__.py` following
   [Semantic Versioning](https://semver.org/): breaking change = major,
   new feature = minor, fix = patch.
3. Merge that change to `main`, then tag it:

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

The release workflow checks the tag matches the version, runs all the tests
(the integration tests too, on its own Elasticsearch), builds the package and
publishes a GitHub Release with the changelog notes.
