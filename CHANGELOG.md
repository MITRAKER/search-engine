# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-17

### Added

- From-scratch search engine in pure Python, with no runtime dependencies:
  - Analyzer: Unicode tokenizer, Lucene's English stopwords with position
    gaps, and a Porter stemmer that matches Elasticsearch's `porter_stem` on
    every one of 36,007 test words.
  - Inverted index with per-field positional postings.
  - BM25F ranking (field weights at query time), plus classic TF-IDF.
  - OR and AND queries, exact phrase queries scored by phrase frequency,
    typo tolerance with Elasticsearch-style `AUTO` fuzziness, and
    "did you mean" suggestions.
  - `explain()`, which breaks a score down term by term.
  - Save and load the index as JSON.
- Command-line interface: `searchengine "query"` with `--fuzzy`, `--explain`,
  `--ranker`, `--mode` and `-k`.
- 25-document sample dataset, and a loader for the SciFact benchmark.
- Engineering setup: GitHub Actions CI (lint, format, strict types, NASA/JPL
  rule checker, tests on Python 3.11 to 3.14 with coverage, dependency audit,
  package build), CodeQL, Dependabot, pre-commit hooks, release workflow with
  build provenance, issue and pull request templates.
- `LEARNING.md`, a step-by-step guide to how everything was built.

[Unreleased]: https://github.com/MITRAKER/search-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MITRAKER/search-engine/releases/tag/v0.1.0
