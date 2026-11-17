# Learning guide: how this search engine was built

This file grows with the project. Each chapter covers one milestone:
**what** was built, **why** it was designed that way, the **concepts** you
need, and **exercises** so you can rebuild it yourself.

Read the chapters in order. Keep the code open next to the text.

---

## Chapter 1: Project setup

### What was built
- A Python package `searchengine/` with no dependencies outside the standard library.
- A dataset loader (`datasets.py`) that reads **JSON Lines**.
- A small sample dataset (`data/sample.jsonl`, 25 docs) for tests and demos.
- `docker-compose.yml` that runs Elasticsearch locally (used from Chapter 3).
- A GitHub Actions workflow that runs the tests on every push.

### Why these choices
- **Pure Python first.** Using no libraries forces you to understand every
  piece. Elasticsearch comes later, so you can compare it with what you built.
- **JSON Lines** (one JSON object per line) instead of one big JSON array: you
  can read it line by line, so a 10 GB dataset never has to fit in memory.
  Elasticsearch's bulk API uses the same idea.
- **A tiny dataset you can read in one minute.** When a test fails you can
  reason about the expected answer by hand. Larger datasets come in Chapter 4.
- **CI from day one.** A green badge on GitHub tells a recruiter the project works.

### Concepts
- **Document**: the unit you search and return, here `{id, title, body}`.
- **Field**: a named part of a document. Fields can be weighted differently.
- **Docker Compose**: describes containers in a file so `docker compose up`
  starts the same setup on any machine.

### Exercises
1. Add a `url` field to a few documents. Does anything break? Why not?
2. Run `docker compose up -d`, then `curl localhost:9200`. What version is it?

---

## Chapter 2: The engine, from scratch

The whole engine is about 200 lines in `analyzer.py` and `index.py`.

### Step 1: Analysis (`analyzer.py`)

Raw text goes through a pipeline before being stored:

```
"The Runners are RUNNING!"
  tokenize   -> ["the", "runners", "are", "running"]
  stopwords  -> ["runners", "running"]
  stem       -> ["runner", "run"]
```

- **Tokenize**: split into words and lowercase. A regex `[a-z0-9]+` is enough here.
- **Stopwords**: drop very common words ("the", "and"). They match almost every
  document, so they help ranking very little and make the index bigger.
- **Stemming**: cut suffixes so "running", "runs" and "run" become the same
  term. The root doesn't need to be a real word ("query" can become "queri" in
  some stemmers). It only needs to be **consistent**.

**The key rule:** the *same* analyzer must run on documents and on queries.
If the query keeps "Running" but the index stored "run", nothing matches.
In Elasticsearch, a mismatch between index-time and search-time analyzers is a
common cause of "why doesn't this match?" bugs.

### Step 2: The inverted index (`index.py`, `add`)

A book's index says "Lucene: pages 12, 48". An inverted index does the same
for terms:

```
postings["invert"] = {"1": [0, 102, 204, 306, 317], "15": [318]}
          term          doc id: positions where the term occurs
```

To search, we only look at the docs listed under the query terms. We never
scan the whole collection. That is why search engines stay fast with
billions of documents.

We store **positions** (not only "doc contains term") because phrase search
needs to know which words are next to each other.

**Field boosting:** title terms are recorded 3 times (`field_boosts`). A word
in the title is a stronger signal than the same word in the body.
Elasticsearch does this with `"title^3"` at query time instead. We add a gap of
100 positions between fields and copies so a phrase can't match across them.

### Step 3: Ranking

Finding matching docs is easy. Putting the **best** ones first is the hard part.

**TF-IDF**

- **TF (term frequency)**: a doc that mentions "bm25" 3 times is probably more
  about BM25 than one that mentions it once.
- **IDF (inverse document frequency)**: `log(1 + (N - df + 0.5) / (df + 0.5))`.
  "search" appears in 12 of 25 docs, so it barely tells docs apart. "pagerank"
  appears in 1, so a match on it is very informative.
- Score = sum over query terms of `log(1 + tf) * idf`.

**BM25** (Elasticsearch's default) fixes two weaknesses of TF-IDF:

```
            tf * (k1 + 1)
idf  *  ---------------------------------------
        tf + k1 * (1 - b + b * docLen / avgLen)
```

- **Saturation (`k1 = 1.2`)**: the 1st occurrence of a word matters a lot, the
  20th barely matters. The tf part can never exceed `k1 + 1`. This stops
  keyword-stuffed documents from winning. See `test_bm25_saturates_term_frequency`.
- **Length normalization (`b = 0.75`)**: a match in a 20-word doc is stronger
  evidence than one match in a 5,000-word doc. With `b = 0` length is ignored;
  with `b = 1` it is fully normalized.

### Step 4: Query features

- **OR vs AND** (`mode`): OR returns docs with *any* term (better recall), AND
  requires *all* terms (better precision). We take the union or the
  intersection of the posting lists.
- **Phrase search** (`"search engine"`): first AND the terms, then check
  positions. The phrase matches if "engine" occurs at position `p + 1` for
  some position `p` of "search".
- **Typo tolerance** (`fuzzy=True`): for an unknown term, find the indexed term
  with the smallest **Levenshtein distance** (the fewest insertions, deletions
  or substitutions). `edit_distance` uses classic dynamic programming. Ties go
  to the more common word. Elasticsearch's `"fuzziness": "AUTO"` uses the same
  idea, but much faster, using automata.
- **"Did you mean"** (`suggest`): runs the same correction and shows it instead
  of applying it silently.

### Try it

```bash
python -m searchengine.cli "inverted index"
python -m searchengine.cli "elastisearch rankng"          # shows "Did you mean"
python -m searchengine.cli --fuzzy "elastisearch rankng"
python -m searchengine.cli '"search engine"'
python -m searchengine.cli --ranker tfidf "ranking"
```

### Interview questions you can now answer
- *Why is an inverted index fast?* You only touch the posting lists of the
  query terms, never every document.
- *BM25 vs TF-IDF?* Term frequency saturation and length normalization.
- *How does phrase search work?* Positional postings.
- *How would you handle typos?* Edit distance against the vocabulary, limited
  to distance 1 to 2, filtered by length first because it is expensive.

### Exercises
1. Set `b=0` in `score_bm25` and search "ranking". Which docs move, and why?
2. Remove "the" from `STOPWORDS` and search `"the inverted index"`. What changes?
3. Add a `synonyms` dict (`{"es": "elasticsearch"}`) and expand query terms with it.
4. `correct()` compares against every term in the vocabulary. How would you make
   it faster for 1 million terms? (Hint: n-gram indexes or BK-trees.)
5. Write a test showing that a doc with "cat" 100 times does not beat a doc
   whose *title* is "cat".
