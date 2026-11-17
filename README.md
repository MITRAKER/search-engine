# Search Engine: from scratch, and on Elasticsearch

[![CI](https://github.com/MITRAKER/search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/MITRAKER/search-engine/actions/workflows/ci.yml)
[![CodeQL](https://github.com/MITRAKER/search-engine/actions/workflows/codeql.yml/badge.svg)](https://github.com/MITRAKER/search-engine/actions/workflows/codeql.yml)
[![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20to%203.14-blue)](pyproject.toml)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](.github/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![mypy strict](https://img.shields.io/badge/types-mypy%20strict-2a6db2)](https://mypy-lang.org/)
[![NASA/JPL Power of Ten](https://img.shields.io/badge/coding%20standard-NASA%2FJPL%20Power%20of%20Ten-0b3d91)](docs/CODING_STANDARD.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A search engine built twice: once from first principles in pure Python
(inverted index, BM25F ranking, phrase search, typo tolerance), and once on
Elasticsearch. Then the two are benchmarked against each other on a real
information-retrieval dataset.

**Highlight:** the from-scratch engine reproduces Elasticsearch 9.5's BM25
scores to 7 significant digits, for short and long documents (it rounds
lengths into one byte exactly as Lucene does), phrase queries and multi-field
`combined_fields` (BM25F) queries. A golden-file test checks this on every
run. Its tokenizer gives Elasticsearch's terms down to Unicode casing and
combining marks, and its Porter stemmer matches Elasticsearch's on all
36,007 words tested.

> 📘 **Learning as you go?** [LEARNING.md](LEARNING.md) explains every step:
> what was built, why, the concepts and math behind it, interview questions
> and exercises.

## Features

| | From scratch (`searchengine`) | Elasticsearch (milestone 3) |
|---|---|---|
| Text analysis | Unicode tokenizer, Lucene stopwords, Porter stemmer | same pipeline as a custom analyzer |
| Index | positional inverted index, per field | Lucene segments |
| Ranking | BM25F with query-time field weights, TF-IDF | BM25, `combined_fields` |
| Queries | OR, AND, exact phrases (with stopword gaps) | `match`, `match_phrase` |
| Typos | Levenshtein distance, `AUTO` fuzziness, "did you mean" | `fuzziness: AUTO`, suggesters |
| Insight | `explain()` breaks every score down | `_explain` API |

## Quick start

```bash
pip install -e .                                   # no runtime dependencies
searchengine "inverted index"                      # BM25F search on the sample data
searchengine --fuzzy "elastisearch rankng"         # typo tolerance
searchengine '"search engine"'                     # exact phrase
searchengine --explain "bm25 ranking"              # why each result scored
searchengine --ranker tfidf --mode and "ranking lucene"
```

```python
from searchengine import InvertedIndex, SearchOptions

index = InvertedIndex()
index.add({"id": "1", "title": "Inverted index", "body": "Maps every term to its documents."})
hits = index.search("inverted", options=SearchOptions(fuzzy=True))
```

## Project status

| Milestone | What | Status |
|---|---|---|
| 1 | Project setup, dataset loading | ✅ |
| 2 | From-scratch engine: analyzer, inverted index, BM25F, phrases, typos | ✅ |
| 2b | Professional engineering: CI/CD, quality gates, NASA/JPL rules | ✅ |
| 3 | Elasticsearch: analyzers, boosting, autocomplete, facets, parity tests | ⏳ |
| 4 | Evaluation on SciFact: nDCG@10, recall, latency, from scratch vs Elasticsearch | ⏳ |
| 5 | Web UI and API | ⏳ |

## Engineering

Every pull request must pass:

- **Lint and format**: ruff (20 rule sets, including bugbear and bandit security rules)
- **Types**: `mypy --strict`
- **Coding standard**: [NASA/JPL "Power of Ten" rules adapted to Python](docs/CODING_STANDARD.md),
  checked by a custom static analyzer ([`scripts/check_jpl.py`](scripts/check_jpl.py))
- **Tests**: pytest and Hypothesis property tests on Python 3.11 to 3.14,
  golden-file tests against Elasticsearch's own output, coverage at least 95%
  (currently 100%), zero warnings allowed
- **Security**: CodeQL, `pip-audit`, Dependabot, dependency review, Actions pinned by SHA
- **Packaging**: the wheel is built, installed in a clean environment and run

CI also runs every day. Releases are cut by pushing a tag; see
[CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).

## Layout

```
searchengine/
  analyzer.py    tokenizer, stopwords, positions
  porter.py      Porter stemmer
  index.py       inverted index, BM25F and TF-IDF search, phrases, explain
  scoring.py     the ranking formulas
  spelling.py    edit distance and typo correction
  datasets.py    JSON Lines and SciFact loaders
  cli.py         command line
  data/          25-document sample dataset
scripts/         JPL rule checker, golden-file recorder, release notes
tests/           unit, property-based and golden-file tests (Elasticsearch as the oracle)
docs/            coding standard
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues
privately, as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
