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
scores, typically to 7 significant digits: word queries over several weighted
fields (BM25F, Elasticsearch's `combined_fields`) and phrases on one field.
It even stores document lengths in one byte, the way Lucene does. A
golden-file test checks this on every run. Its tokenizer gives
Elasticsearch's terms down to Unicode casing, combining marks and letters
beyond U+FFFF, and its Porter stemmer matches Elasticsearch's on all 36,007
words tested.

> 📘 **Learning as you go?** [LEARNING.md](LEARNING.md) explains every step:
> what was built, why, the concepts and math behind it, interview questions
> and exercises.

## Features

| | From scratch (`searchengine`) | Elasticsearch (`searchengine.es_engine`) |
|---|---|---|
| Text analysis | Unicode tokenizer, Lucene stopwords, Porter stemmer | the same steps, as a custom analyzer |
| Index | positional inverted index, per field | Lucene segments in one shard, an explicit mapping |
| Ranking | BM25F with query-time field weights, TF-IDF | the same BM25F, through `combined_fields` |
| Queries | OR, AND, exact phrases (with stopword gaps) | `combined_fields` with OR or AND, `multi_match` phrases |
| Typos | Levenshtein distance, `AUTO` fuzziness, "did you mean" | `fuzziness: AUTO`, a term suggester for "did you mean" |
| Product features | | facets, highlighted snippets, autocomplete |
| Insight | `explain()` breaks every score down | the `_explain` API |

## Quick start

### From scratch

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

### On Elasticsearch

```bash
pip install -e ".[es]"                             # adds the Elasticsearch client
docker compose up -d --wait                        # Elasticsearch 9.5.5 on localhost:9200
searchengine --engine es "inverted index"          # the first run creates and fills the index
searchengine --engine es --category ranking "search"   # only one category, with facet counts
searchengine --engine es --explain "bm25 ranking"  # Elasticsearch's own score breakdown
```

```python
from searchengine.datasets import load_jsonl
from searchengine.es_engine import ElasticsearchEngine

engine = ElasticsearchEngine()  # $ES_URL, default http://localhost:9200
if engine.create_index():  # False when it already exists
    engine.index_documents(load_jsonl())  # the sample data
result = engine.search("inverted index", category="indexing")
print(result.total, result.facets)  # matches, and counts per category
print(engine.suggest("elastisearch rankng"))  # elasticsearch ranking
print(engine.autocomplete("inverted ind"))  # ['Inverted index']
```

## Project status

| Milestone | What | Status |
|---|---|---|
| 1 | Project setup, dataset loading | ✅ |
| 2 | From-scratch engine: analyzer, inverted index, BM25F, phrases, typos | ✅ |
| 2b | Professional engineering: CI/CD, quality gates, NASA/JPL rules | ✅ |
| 3 | Elasticsearch: analyzers, boosting, autocomplete, facets, parity tests | ✅ |
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
- **Integration tests**: both engines side by side on a real Elasticsearch,
  run in CI as a service container (the same image as `docker compose`,
  pinned by digest). Word queries must find the same documents with the same
  scores at any field weights; the few differences by design are
  [listed in LEARNING.md](LEARNING.md#44-where-the-two-engines-differ-on-purpose).
- **Security**: CodeQL, `pip-audit`, Dependabot (Python packages, Actions and
  the Elasticsearch image), dependency review, Actions pinned by SHA
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
  es_dsl.py      Elasticsearch settings and queries, built as plain data
  es_engine.py   the same engine on Elasticsearch
  cli.py         command line, for both engines
  data/          25-document sample dataset
scripts/         JPL rule checker, golden-file recorder, release notes
tests/           unit, property-based, golden-file and integration tests
                 (Elasticsearch as the oracle)
docs/            coding standard
docker-compose.yml   Elasticsearch for development, pinned by digest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues
privately, as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
