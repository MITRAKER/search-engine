# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The same engine on Elasticsearch, in `searchengine.es_engine`
  (`pip install -e ".[es]"`):
  - A custom analyzer with the same steps as ours, one shard and the same
    BM25 parameters, so word queries (OR and AND, any field weights, through
    `combined_fields`) get the same scores as the from-scratch engine, and
    equal scores are sorted by document id in both.
  - Bulk loading with `create` actions, so a duplicate id is an error rather
    than an overwrite, in requests of at most 500 documents or about 10 MB.
    An existing index is reused only if it was made with the same BM25
    parameters and mappings; it may be an alias of one index.
  - Facet counts per category with a `post_filter`, HTML-escaped highlighted
    snippets (also for fields over a million characters), title autocomplete
    (`search_as_you_type`, every typed word must match), "did you mean" (a
    term suggester that leaves words matching a document alone), fuzzy
    search, and Elasticsearch's `_explain`.
  - Every request is built as plain data by `searchengine.es_dsl`, so it can
    be tested without a server.
- `searchengine --engine es`, with `--index`, `--reindex` and `--category`.
  It checks the query, and with `--reindex` reads the whole data file, before
  it touches the index, and it deletes a half-built index on an error or on
  Ctrl-C. Errors, timeouts included, are one-line messages.
- Integration tests that run both engines side by side on a real
  Elasticsearch (`make integration`), and a CI job that runs them against an
  Elasticsearch service container. The release workflow runs them too.
- Dependabot updates for the Elasticsearch image.
- `LEARNING.md` chapter 4: the same engine on Elasticsearch.

### Changed

- `docker-compose.yml` pins Elasticsearch by digest. CI uses the same image,
  and a test fails if the two differ.
- `scripts/make_parity_golden.py` builds its index and queries with
  `searchengine.es_dsl`, the code the engine uses. The recorded scores did
  not change.

## [0.1.0] - 2026-11-17

### Added

- From-scratch search engine in pure Python, with no runtime dependencies:
  - Analyzer: a Unicode tokenizer that keeps combining marks inside words
    and lowercases and measures tokens the way Lucene does, Lucene's English
    stopwords with position gaps, and a Porter stemmer that matches
    Elasticsearch's `porter_stem` on every one of 36,007 test words.
  - Inverted index with per-field positional postings.
  - BM25F ranking (field weights from 1 to 1000, at query time) with the
    statistics of Lucene's `CombinedFieldQuery` and Lucene's one-byte
    document lengths, so scores match Elasticsearch's `combined_fields`;
    `SearchOptions(lucene_norms=False)` scores with exact lengths. Classic
    TF-IDF too.
  - OR and AND queries, exact phrase queries scored by phrase frequency,
    typo tolerance with Elasticsearch-style `AUTO` fuzziness that only
    corrects to words in the searched fields, and "did you mean" suggestions.
  - `explain()`, which breaks a score down term by term.
  - Save and load the index as JSON, with every loaded file validated.
  - Limits on every input: text and token length, query length and terms,
    results, fields, document ids, JSON Lines line size and index file size.
    Every public function checks its arguments.
- Command-line interface: `searchengine "query"` with `--fuzzy`, `--explain`,
  `--ranker`, `--mode` and `-k`. Bad input gives a one-line error.
- 25-document sample dataset, and a loader for the SciFact benchmark.
- Golden-file tests of Elasticsearch 9.5.5's own scores and terms
  (`tests/test_es_parity.py`), recorded by `scripts/make_parity_golden.py`.
- Engineering setup: GitHub Actions CI (lint, format, strict types, NASA/JPL
  rule checker, tests on Python 3.11 to 3.14 with coverage, dependency audit,
  package build), CodeQL, Dependabot, pre-commit hooks, release workflow with
  build provenance, issue and pull request templates.
- `LEARNING.md`, a step-by-step guide to how everything was built.

[Unreleased]: https://github.com/MITRAKER/search-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MITRAKER/search-engine/releases/tag/v0.1.0
