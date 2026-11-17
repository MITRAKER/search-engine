# Search Engine: from scratch and with Elasticsearch

A search engine built twice: once from first principles in pure Python
(inverted index, BM25, phrase search, typo tolerance), and once on
Elasticsearch. The two are then benchmarked against each other on ranking
quality and speed.

> 📘 Learning as you go? Read [LEARNING.md](LEARNING.md). Every milestone
> explains what was built, why, and the concepts behind it, with exercises.

## Status

| Milestone | What | Status |
|---|---|---|
| 1 | Project setup, dataset loader, Docker Compose for Elasticsearch | ✅ |
| 2 | From-scratch engine: analyzer, inverted index, TF-IDF, BM25, phrases, fuzzy | ✅ |
| 3 | Elasticsearch indexing, analyzers, boosting, autocomplete, facets | ⏳ |
| 4 | Evaluation: precision@10, nDCG, latency, from-scratch vs Elasticsearch | ⏳ |
| 5 | FastAPI + React search UI | ⏳ |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                  # run the tests
python -m searchengine.cli "inverted index"          # BM25 search
python -m searchengine.cli --fuzzy "elastisearch"    # typo tolerance
python -m searchengine.cli '"search engine"'         # exact phrase
python -m searchengine.cli --ranker tfidf "ranking"  # compare rankers
docker compose up -d                                 # Elasticsearch on :9200
```

## Layout

```
searchengine/
  analyzer.py   tokenizer, stopwords, stemmer
  index.py      inverted index, TF-IDF, BM25, phrase + fuzzy search
  datasets.py   JSON Lines loader
  cli.py        command-line search
data/sample.jsonl   25 small documents for tests and demos
tests/              pytest suite
```

## License

MIT. See [LICENSE](LICENSE).
