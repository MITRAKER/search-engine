# Learning guide: how this search engine was built

This guide grows with the project. Each chapter covers one milestone:
**what** was built, **why** it was designed that way, the **concepts** you
need, and **exercises** so you can rebuild it yourself.

Read the chapters in order, with the code open next to the text. Every claim
here is backed by a test in `tests/`, so when in doubt, read the test.

| Chapter | Topic | Status |
|---|---|---|
| 1 | Project setup | ✅ |
| 2 | The engine, from scratch | ✅ |
| 3 | Engineering like a professional team: CI/CD, quality gates, NASA/JPL rules | ✅ |
| 4 | The same engine on Elasticsearch | ✅ |
| 5 | Measuring search quality on SciFact | ✅ |
| 6 | Web UI and API | planned |

---

## Chapter 1: Project setup

### What was built

- A Python package, `searchengine/`, that needs nothing outside the standard
  library to run.
- `pyproject.toml`: one file that describes the package (name, version,
  dependencies, the `searchengine` command) and configures every tool.
- A 25-document sample dataset in **JSON Lines**, shipped inside the package
  so `pip install` users can try the engine immediately.
- `docker-compose.yml`, which starts Elasticsearch locally (used from Chapter 4).

### Why these choices

- **Pure Python first.** Using no libraries forces you to understand every
  piece. Elasticsearch comes later, so you can compare it with what you built.
- **JSON Lines** (one JSON object per line) instead of one big JSON array: it
  can be read line by line, so a 10 GB file never has to fit in memory.
  Elasticsearch's bulk API uses the same idea.
- **A tiny dataset you can read in a minute.** When a test fails you can work
  out the right answer by hand. The real benchmark comes in Chapter 5.

### Concepts

- **Document**: the unit you search and return, here `{id, title, body}`.
- **Field**: a named part of a document. Fields can be weighted differently.
- **Virtual environment** (`python -m venv .venv`): a private set of packages
  for one project, so projects never fight over versions.
- **Editable install** (`pip install -e .`): Python imports the package
  straight from your working copy, so code changes take effect immediately.

### Exercises

1. Add a `url` field to a few sample documents. Does anything break? Why not?
   (Hint: look at `normalize_document` in `index.py`.)
2. Run `pip install -e .` in a fresh virtual environment, then
   `searchengine "stemming"`. Where does the command come from?
   (Hint: `[project.scripts]` in `pyproject.toml`.)

---

## Chapter 2: The engine, from scratch

The engine is about 1,200 lines across `analyzer.py`, `porter.py`,
`index.py`, `scoring.py` and `spelling.py`. About 700 of them are code; the
rest are the explanations in docstrings and comments.

### 2.1 Analysis: from text to terms (`analyzer.py`)

Before anything is stored, text goes through a pipeline:

```
"The Runners are RUNNING!"
  tokenize          -> the, runners, are, running      (positions 0, 1, 2, 3)
  drop stopwords    -> runners@1, running@3
  stem (Porter)     -> runner@1, run@3
```

- **Tokenize**: find runs of letters, digits and combining marks in any
  script, and lowercase them. "café", "2026", "جستجو" and "हिन्दी" are all
  tokens. Combining marks are the accents typed as separate characters, Hindi
  vowel signs and Arabic vowel marks. They are not letters, but they belong
  inside the word: a tokenizer that only kept letters would cut "हिन्दी" into
  pieces.
- **Stopwords**: drop very common words ("the", "and"). They appear in almost
  every document, so they barely help ranking. We use **Lucene's 33 English
  stopwords**, the exact list Elasticsearch uses.
- **Stemming**: cut suffixes so "running", "runs" and "run" become one term.
- **Positions keep gaps.** "index" in "inverted the index" is at position 2,
  not 1, even though "the" was dropped. Lucene does the same, and it stops the
  phrase "inverted index" from matching "inverted the index".

**The key rule:** the *same* analyzer runs on documents and on queries. If
the query keeps "Running" but the index stored "run", nothing matches. A
mismatch between index-time and search-time analysis is one of the most
common Elasticsearch bugs in real projects.

**Matching Lucene to the last character.** Chapter 4 builds the same pipeline
in Elasticsearch, and the two must produce identical terms. Three details of
Lucene, which is written in Java, needed copying:

- *Lowercasing.* Python's `str.lower()` follows Unicode's full rules: a
  capital sigma at the end of a Greek word becomes the final form "ς", and the
  Turkish dotted capital "İ" becomes "i" plus a combining dot. Java's own
  `String.toLowerCase` does exactly the same. But Lucene's `LowerCaseFilter`
  lowercases one character at a time with `Character.toLowerCase`, Unicode's
  simple mapping, so neither rule applies. `_lowercase` does the same.
- *Length.* Tokens longer than 255 characters are dropped, but Java counts
  length in UTF-16 units, where an emoji or a character such as "𝐀" counts
  twice. `java_length` counts the same way.
- *Stemming.* Lucene's stemmer counts UTF-16 units too. Section 2.2 shows
  what that changes.

`tests/test_es_parity.py` checks our terms against the ones Elasticsearch
produced for 19 tricky texts. We also compared every Unicode code point once
(1,112,032 of them, leaving out surrogates and control codes): the only
differences were characters added to Unicode after the version that Python's
own database knows.

### 2.2 The Porter stemmer (`porter.py`)

The first version of this project used a home-made suffix stripper. It had a
bug that shows why stemmers are hard: "engines" became "engin" but "engine"
stayed "engine", so a search for one never found the other. The fix was to
implement the real thing: Martin Porter's 1980 algorithm, which is also what
Elasticsearch's `porter_stem` filter runs.

The central idea: write any word as consonant groups (C) and vowel groups (V):

```
[C](VC)^m[V]        m is the "measure"

tree    -> [tr][ee]          m = 0
trouble -> [tr](ou bl)[e]    m = 1
private -> [pr](iv)(at)[e]   m = 2
```

Most rules only fire when the stem that remains is long enough, so short
words survive: "rate" keeps its "-ate" because the "r" left over is too
short (m=0), while "activate" loses it ("activ" has m=2).
The algorithm runs five steps: plurals and -ed/-ing (`caresses -> caress`,
`hopping -> hop`), y to i (`happy -> happi`), double suffixes
(`relational -> relate`), -ic/-ful/-ness, then -ance/-ment/-ive and a
final -e.

**How we know it is right:** we sent 36,007 words through Elasticsearch's own
`_analyze` API and compared. Zero differences. 3,107 of them are stored in
`tests/data/porter_golden.json`, so the test runs without Elasticsearch.
This kind of test, where a trusted system provides the expected answers, is
called an **oracle** or **golden** test.

One detail came from a later review: Lucene's stemmer works on Java's UTF-16
units, so a letter beyond U+FFFF, such as "𝐛", counts as two consonants. Our
stemmer does the same for words that contain one, so "bo𝐛ing" becomes
"bo𝐛" in both engines.

### 2.3 The inverted index (`index.py`)

A book's index says "Lucene: pages 12, 48". An inverted index does the same
for terms, and also records the field and position of every occurrence:

```
postings["invert"] = {"1":  {"title": [0], "body": [1, 18]},
                      "15": {"body": [18]}}
          term        doc id  field    positions
```

To search, we only read the posting lists of the query terms, never the whole
collection. That is why search engines stay fast with billions of documents.
We store **positions** because phrase search needs to know which words are
next to each other, and we keep **fields separate** so they can be weighted
at query time.

`add()` analyzes the whole document before changing anything. If any field is
invalid, it raises an error and the index is untouched (there is a test for
this). Databases call this property **atomicity**.

### 2.4 Ranking

Finding the matching documents is the easy part. Putting the **best** ones
first is the hard part. All formulas live in `scoring.py`.

**TF-IDF**, the classic:

- **TF (term frequency)**: a document that says "bm25" 4 times is probably
  more about BM25 than one that says it once. We use `log(1 + tf)` so the
  100th repetition does not count 100 times.
- **IDF (inverse document frequency)**: `log(N / df)`. "search" appears in 12
  of our 25 documents, so it barely tells them apart. "pagerank" appears in
  one, so matching it means a lot.
- Score = sum over the query terms of `tf part x idf`.

**BM25**, the default in Elasticsearch and Lucene, improves on TF-IDF in two ways:

```
                         tf x (k1 + 1)
score = idf  x  ---------------------------------------      idf = log(1 + (N - df + 0.5) / (df + 0.5))
                tf + k1 x (1 - b + b x docLength / avgLength)
```

- **Saturation (`k1 = 1.2`)**: the first occurrence of a word matters a lot,
  the twentieth barely matters. The tf part can never exceed `k1 + 1`. This
  stops keyword-stuffed documents from winning.
- **Length normalization (`b = 0.75`)**: one match in a 20-word document is
  stronger evidence than one match in a 5,000-word document. `b = 0` turns
  this off, `b = 1` normalizes fully.

**BM25F: one score across several fields.** A word in the title should count
more than a word in the body. BM25F multiplies each field's term frequency
and length by the field's weight (title 3, body 1 by default), adds them up
as if they formed one "virtual" field, and then applies BM25 once:

```
tf     = 3 x tf_title     + 1 x tf_body
length = 3 x length_title + 1 x length_body
```

Applying BM25 once (instead of scoring each field and adding the scores) means
saturation works across fields, so a document cannot win just by repeating a
word in both. Elasticsearch's `combined_fields` query uses exactly this model.
Like `combined_fields`, we require every field weight to be at least 1: a
weight says how many times a word counts, and a word counts at least once.

**The statistics, counted the way Lucene counts them.** BM25F needs N, df and
avgdl for the virtual field, and there is more than one sensible way to count
them across fields. We follow Lucene's `CombinedFieldQuery`, the code behind
`combined_fields`:

- **df** is the *largest per-field* df. "apache" in one document's title and
  in another document's body has df 1, not 2.
- **N** is the largest per-field number of documents with at least one word
  in that field. Two empty titles out of three documents do not make N
  smaller.
- **avgdl** is the weighted total length divided by N.

**Lengths are stored in one byte.** Lucene keeps each document's field length
(its *norm*) in a single byte to save memory. Lengths 0 to 23 are stored as
they are. Above that, Lucene keeps 4 significant bits of (length - 24), so 24
to 40 also survive exactly, 41 becomes 24 + 16 = 40, and 375 becomes
24 + 320 = 344. BM25 scores with the stored value. `lucene_length` in `scoring.py` does the same. With
several fields Lucene adds up the weighted lengths in 32-bit floats and stores
the sum in one byte again; the weighted total behind avgdl loses its fraction
after each field, because Lucene adds it into a whole number. Each detail
moves the scores of long documents by up to a few percent. If you prefer the
textbook formula with exact lengths, use `SearchOptions(lucene_norms=False)`.

**Tested against Elasticsearch's own scores.** `tests/test_es_parity.py`
replays 36 queries on 42 documents (from empty to 1,200 words long, five
field weightings, AND mode and phrases) and compares every score with the one
Elasticsearch 9.5.5 gave, recorded by `scripts/make_parity_golden.py`. They
agree to better than 2 parts in 10 million, about all the precision Lucene's
32-bit floats have. (For a document many times longer than average, Lucene's
own 32-bit arithmetic loses a few more digits: about 1 part in a million at
20 times the average length.) A second test proves the check can fail: with
exact lengths, scores move by up to 7%, and the comparison catches it.

### 2.5 Query features

- **OR vs AND** (`SearchOptions(mode=...)`): OR returns documents with *any*
  term (better recall), AND requires *all* of them (better precision).
- **Phrases** (`"search engine"` in double quotes): a document matches when,
  in the same field, every term sits at the right distance from the first.
  For the query `"inverted the index"` the distances are 0 and 2, gap included.
  Phrases are scored like Lucene does: tf is the **number of times the whole
  phrase occurs**, and idf is the sum of the terms' idfs. On one field with
  weight 1 this is exactly Elasticsearch's `match_phrase` score (the golden
  test checks it). With other weights, or several fields, Elasticsearch
  scores phrases its own way, so the scores differ.
- **Typo tolerance** (`fuzzy=True`): for each query word the index does not
  know, find the known word with the smallest **Levenshtein distance** (the
  fewest letters inserted, deleted or substituted). `edit_distance` uses
  dynamic programming with one row of memory and stops early once the
  distance must exceed the limit. Like Elasticsearch's `"fuzziness": "AUTO"`,
  we allow 0 typos in words of 1 or 2 letters, 1 in words of 3 to 5, and 2 in
  longer words. We correct the **surface** words people type ("rankng" to
  "ranking"), not the stems, because that is where typos happen. A
  correction must be a word found in the fields being searched, so a search
  of the body never "corrects" a word into one that only titles contain. Ties
  go to the more frequent word, then alphabetical order, so results never
  depend on the order documents were added.
- **"Did you mean"** (`suggest`): the same correction, shown instead of applied.
- **`explain(query, doc_id)`**: lists each clause's idf, tf and contribution.
  The parts always add up to the score (there is a test for that too).

### 2.6 How a search runs

```
query --parse--> terms --postings--> candidate docs --BM25F--> scores --heap--> top k
```

We score **term at a time**: for each query term, walk its posting list and
add its contribution to a running total per document. Then
`heapq.nsmallest(k, ...)` picks the best k in O(n log k) without sorting
everything. Scores are rounded to 32-bit floats first, as Lucene's are
(section 4.3 says why), and ties are broken by document id, so results are
reproducible.

### Try it

```bash
searchengine "inverted index"
searchengine "elastisearch rankng"            # prints "Did you mean"
searchengine --fuzzy "elastisearch rankng"
searchengine '"search engine"'
searchengine '"inverted the index"'           # no results: the gap matters
searchengine --explain "bm25 ranking"
searchengine --ranker tfidf "ranking"
searchengine --data missing.jsonl "ranking"   # one-line error, exit status 1
```

### Interview questions you can now answer

- *Why is an inverted index fast?* A query only reads the posting lists of
  its terms, never every document.
- *BM25 vs TF-IDF?* Term-frequency saturation (k1) and length normalization (b).
- *How do you search several fields?* BM25F: weight the term frequencies and
  lengths, then score once, like Elasticsearch's `combined_fields`.
- *How does phrase search work?* Positional postings, and checking that the
  terms sit at the right distances.
- *Why must index-time and query-time analysis match?* Otherwise identical
  words become different terms and never match.
- *How would you handle typos?* Edit distance against the vocabulary, limited
  by word length, with the cheap length filter applied first.
- *Why could two correct BM25 implementations disagree?* The formula leaves
  room for choices: how df and N are counted across fields, how document
  lengths are stored (Lucene rounds them into one byte), and the precision of
  the arithmetic (Lucene uses 32-bit floats).

### Exercises

1. In `scoring.py`, set `b` to 0 and search "search". Which documents move,
   and why?
2. Remove "the" from `ENGLISH_STOPWORDS` and search `"inverted the index"`.
   What changes, and why?
3. Add synonyms: expand the query term "es" to "elasticsearch" in `parse()`.
4. Elasticsearch counts swapping two neighbouring letters ("hte" for "the") as
   **one** edit (Damerau-Levenshtein). Change `edit_distance` to do the same,
   and add a test.
5. `SpellingIndex.correct` compares against every word of similar length.
   How would you make it fast for a million words? (Look up BK-trees and
   SymSpell.)
6. Change `lucene_length` to return `length` unchanged and run
   `pytest tests/test_es_parity.py`. Which documents fail? Why are they all
   longer than 40 words?

---

## Chapter 3: Engineering like a professional team

Code that works on your laptop is the start, not the finish. This chapter
covers the safety net that catches mistakes before they reach `main`.

### 3.1 The quality gates

Every pull request runs all of these in GitHub Actions. `make check` runs
most of them on your machine (lint, format, types, the coding standard, the
tests and the hygiene hooks, on one Python version), and `make audit` runs
the dependency audit. The other Python versions, the packaging test, CodeQL
and GitHub's dependency review run only in CI, and Dependabot checks for new
versions once a week. The integration tests of chapter 4 run in CI too, and on
your machine with `make integration` while Elasticsearch is running.

| Gate | Tool | Example of what it catches |
|---|---|---|
| Lint | ruff, 20 rule sets | an unused variable, a mutable default argument, `assert` on a tuple |
| Format | ruff format | inconsistent quotes and line breaks (it fixes them) |
| Types | `mypy --strict` | calling `.lower()` on something that might be `None` |
| Coding standard | `scripts/check_jpl.py` | recursion, `while True`, a 70-line function, a module-level cache |
| Tests | pytest + coverage | a broken ranking; untested branches |
| Hygiene | pre-commit hooks | trailing spaces, a merge conflict marker, a 5 MB file, a private key |
| Properties | Hypothesis | a crash on some odd Unicode input nobody thought of |
| Dependencies | pip-audit, Dependabot | a library version with a known vulnerability |
| Security | CodeQL, ruff's bandit rules | unsafe calls, injection risks, risky workflow settings |

Real examples from this project: mypy caught a variable declared twice in
`_clauses`; ruff flagged a regex with unescaped
metacharacters in a test; Hypothesis checks `edit_distance` against a
textbook implementation on hundreds of random string pairs.

### 3.2 NASA/JPL's "Power of Ten"

NASA's Jet Propulsion Laboratory writes the software that flies spacecraft,
where a crash cannot be fixed by restarting. In 2006 Gerard Holzmann
condensed their experience into ten rules. They were written for C, so
[docs/CODING_STANDARD.md](docs/CODING_STANDARD.md) translates each one into
Python and names the tool that enforces it. The short version:

1. **No recursion.** Our Porter stemmer first decided whether a "y" is a
   vowel by calling itself on the previous letter. It now computes a
   consonant mask in one loop.
2. **Every loop has a bound.** No `while True` (nor a loop over
   `itertools.count()`), and every input that drives a loop has a size
   limit: text length, token length, query length, results.
3. **No unbounded memory growth.** Data files are read one line at a time,
   and a line may be at most 10 MB. A saved index is read whole, so its file
   may be at most 500 MB. Result lists are capped (1,000 hits, 50 spelling
   candidates).
4. **Short functions.** At most 60 lines and complexity 10.
5. **Check for the impossible.** Every public function validates its inputs
   with clear errors; `assert` only guards internal invariants.
6. **Smallest scope.** No global state: every module-level name is `Final`,
   and shared tables are frozen (`tuple`, `frozenset`, `MappingProxyType`).
7. **Check every result.** `mypy --strict` makes you handle `None`; saved
   index files are validated before they are trusted; the command line turns
   bad input into a one-line error instead of a traceback.
8. **No runtime code generation.** No `eval`, `exec` or dynamic imports, and
   no monkey-patching. Tests use pytest's `monkeypatch`, which undoes every
   change when the test ends.
9. **Simple indirection.** No `getattr` with computed names; a static call graph.
10. **Zero warnings, analyzed daily.** Any warning fails the tests, and CI
    runs every day even when nobody pushes.

**How `check_jpl.py` works.** Python can parse its own source into an
**abstract syntax tree** (the `ast` module). The checker walks that tree:
function length is `end_lineno - lineno + 1`; `while True` is a `While` node
whose test is a true constant; recursion is found by building a **call graph**
(which function calls which, looking names up the way Python does, so a
nested helper that calls itself is found) and searching it for cycles. One
file cannot show everything: a call through an object passed in as an
argument, or into another module, is left to code review, and the checker's
docstring says so. Its tests show both sides for each rule: code it must
catch, and code it must leave alone, because a checker that cries wolf gets
switched off. One test runs it on this project's code.

### 3.3 CI/CD with GitHub Actions

A **workflow** (`.github/workflows/*.yml`) says *when* to run (`on:` push,
pull request, schedule, tag) and *what* to run (`jobs:`, each a list of
`steps:`). Things worth noticing in `ci.yml`:

- **Matrix**: the test job runs once per Python version (3.11 to 3.14).
- **Least privilege**: the token gets `contents: read` unless a job needs more.
- **Pinned actions**: `actions/checkout@3d3c42e5...  # v7.0.1`. A tag like
  `v7` can be moved by whoever controls it; a commit SHA cannot. Dependabot
  keeps the pins up to date.
- **Reproducible installs**: `requirements-dev.txt` pins every tool to an
  exact version, so CI behaves the same today and next month.
- **Daily schedule**: the tools are pinned, so a run on unchanged code
  installs exactly the same packages. What it catches is what changes
  outside the repository: newly published vulnerabilities, new Python patch
  releases and new runner images. New versions of the tools arrive as
  Dependabot pull requests, where CI tests them first.
- **Packaging test**: the wheel is built, installed in an empty environment
  and actually run.

**CD (continuous delivery)**: `release.yml` runs when a tag such as `v0.1.0`
is pushed. It checks the tag matches `__version__`, runs the tests, builds
the package, attaches a signed **build provenance attestation** (proof of
which workflow built which file) and publishes a GitHub Release with the
notes from `CHANGELOG.md`.

**Semantic Versioning**: `MAJOR.MINOR.PATCH`. Breaking change: bump major.
New feature: bump minor. Bug fix: bump patch.

### 3.4 Testing strategy

- **Unit tests**: one behavior each, named for the behavior
  (`test_phrase_respects_word_order`).
- **Hand calculations**: `test_bm25f_score_matches_hand_calculation` computes
  the expected score on paper, independently of `scoring.py`.
- **Golden (oracle) tests**: Elasticsearch's stems for 3,107 words, its terms
  for 19 tricky texts, and its scores for 36 queries.
- **Reference ports**: `test_scoring.py` ports Lucene's `SmallFloat` line by
  line, and checks our two-line `lucene_length` against it on every length
  up to 5,000 and on random ones up to 2^31.
- **Testing the test**: a golden test only helps if it can fail.
  `test_exact_lengths_do_not_match_elasticsearch` shows that a plausible
  mistake (exact lengths) is caught.
- **Property-based tests** with Hypothesis: instead of one example, state a
  rule ("search never crashes and every score is positive") and let the
  library generate hundreds of inputs to break it.
- **Error-path tests**: every `raise` has a test, which is how coverage reaches
  100% of lines *and* branches.

### 3.5 Your daily workflow

```bash
make install                     # once: tools and git hooks
git switch -c feat/my-change     # never work on main
# ... edit, then:
make check                       # most of the CI gates (see 3.1)
git commit -m "feat: add synonym expansion"   # Conventional Commits
git push -u origin feat/my-change             # then open a pull request
```

`pre-commit` runs the fast checks on every commit, so CI rarely fails. The
pull request template reminds you to update the CHANGELOG and this guide.
CODEOWNERS asks the maintainer to review every change.

### Interview questions you can now answer

- *What is the difference between CI and CD?* CI checks every change
  automatically; CD automates releasing a checked version.
- *Why pin GitHub Actions to a SHA?* Tags can be moved to malicious code;
  commit SHAs cannot.
- *What is property-based testing?* Testing a rule over many generated
  inputs rather than a few hand-picked examples.
- *What are NASA's Power of Ten rules and how do they apply outside C?*
  See 3.2: bounded loops and inputs, no recursion, short functions,
  validated inputs, static analysis with zero warnings.
- *Is 100% coverage enough?* No. Coverage shows which lines ran, not whether
  the assertions were right. That is why we also use oracles and properties.

### Exercises

1. Write a function with a `while True:` loop and run
   `python scripts/check_jpl.py yourfile.py`. Then rewrite the loop with a bound.
2. Add a rule to `check_jpl.py` that rejects bare `except:` clauses, with a test.
3. Break a test on purpose, push a branch and watch CI fail. Read the log,
   fix it, and watch it pass.
4. Read one Dependabot pull request when it arrives. What changed, and how do
   you know it is safe to merge?

---

## Chapter 4: The same engine on Elasticsearch

Chapter 2 built a search engine by hand. This chapter builds the same engine
on Elasticsearch, the search server most companies use, and makes the two
agree: same documents, same terms, same scores. Then chapter 5 can compare
them fairly on a real benchmark.

### What was built

- `searchengine/es_dsl.py`: every request we send, as plain Python dicts:
  the index settings, the field mappings and the queries. Elasticsearch
  speaks JSON over HTTP, and these functions only build that JSON.
- `searchengine/es_engine.py`: `ElasticsearchEngine`, which sends those
  requests with the official `elasticsearch` client. Its `search` takes the
  same arguments as `InvertedIndex.search`, and adds a category filter with
  facet counts and highlighted snippets. It also has `explain`, "did you
  mean" (`suggest`), `autocomplete` and `analyze`.
- `searchengine --engine es ...` on the command line.
- Tests in three layers (see 4.5), including a CI job that starts a real
  Elasticsearch and runs both engines side by side.

The `elasticsearch` package is optional: `pip install -e ".[es]"`. The
from-scratch engine still needs nothing outside the standard library.

### 4.1 Elasticsearch in five minutes

| Term | Meaning | In our project |
|---|---|---|
| **Index** | a collection of documents, like a database table | `searchengine` |
| **Mapping** | the type of each field: how it is stored and searched | `index_mappings()` |
| **`text` field** | analyzed into terms, for full-text search | title, body |
| **`keyword` field** | stored as one exact value, for filters and counts | category |
| **Analyzer** | tokenizer plus token filters: text to terms | `analysis_settings()` |
| **Shard** | the index is split into shards, each a Lucene index | one shard |
| **Segment** | an immutable piece of a shard; new documents go into new segments | |
| **Refresh** | makes new segments searchable, every second by default | after each bulk load |
| **Query DSL** | queries written as JSON | `build_query()` |

Inside every shard is **Lucene**, a Java library with the same core you built
in chapter 2: an inverted index with positions, BM25 and an analyzer chain.

### 4.2 Making Elasticsearch behave like our engine

**The same analyzer, step by step.** Elasticsearch's built-in analyzers would
give different terms. Its `standard` tokenizer follows Unicode's word-break
rules, so "3.14", "it's" and "snake_case" stay single tokens, while ours
splits them; its `english` analyzer also removes a trailing "'s". So we build
a **custom analyzer** from the same steps, in the same order:

```
pattern tokenizer [^\p{L}\p{M}\p{N}]+   split on anything but letters, marks, digits
lowercase                               Java's per-character lowercase
length (max 255)                        drop overlong tokens
stop (Lucene's 33 English stopwords)    positions keep their gaps
porter_stem                             the same Porter algorithm
```

`tests/test_es_parity.py` checks the terms for 19 tricky texts, and the
integration tests ask the live server too.

**One shard.** Lucene computes N, df and avgdl *per shard*. With five shards,
the same document could score differently depending on where it landed.
One shard gives the same statistics as our single index. (A big index needs
several shards; then the `dfs_query_then_fetch` search type gathers the
statistics from all shards first, at the cost of an extra round trip.)

**The same BM25.** The index's default similarity is BM25 with our `k1` and
`b`. They are *static* index settings: changing them means closing the
index, updating its settings and opening it again (no reindexing is needed,
since BM25 only uses them at search time). So the engine treats them as part
of the index: it refuses a search that asks for other values, and refuses to
reuse an index that has other ones. More generally, it stores a fingerprint
(a SHA-256 hash) of its settings and mappings in the index, in the mapping's
`_meta`, and reuses an index only if it finds its own fingerprint there. An
index made by hand, or with another analyzer, other fields or another
version of this code, is refused, and so is one with more than one shard.

**`combined_fields` for BM25F.** There are two families of multi-field
queries, and the difference is a classic interview topic:

- *Field-centric*, like `multi_match` with `best_fields`: score each field
  on its own, then keep the best (or add them up, with `most_fields`). Its
  `"operator": "and"` means "all words in the **same** field".
- *Term-centric*, like `combined_fields`: treat the fields as one big field,
  which is exactly BM25F. "and" means "all words, in any field".
  `multi_match` with `cross_fields` is term-centric too, but it still scores
  each field on its own, with blended statistics, instead of computing BM25F.

Our engine is term-centric, so the words go to `combined_fields`, with the
weights written as `title^3.0`. It runs Lucene's `CombinedFieldQuery`, whose
statistics and length rounding chapter 2.4 copied. Like it, both engines
require every weight to be at least 1.

### 4.3 What a real product adds

- **Bulk loading.** `index_documents` starts a new bulk request every 500
  documents, or sooner, once a request holds 10 MB of documents. With at most
  one more document (of up to 10 MB), a request stays far below the 100 MB
  that Elasticsearch accepts. It uses the `create` action, so a duplicate id
  is an error instead of a silent overwrite. The index must exist first;
  otherwise Elasticsearch would create one by itself, with its default
  analyzer. A bulk load is not atomic: when one fails, Elasticsearch keeps the
  requests sent before, and every good document of the failing request, even
  those after the bad one (4.4 compares this with `InvertedIndex.add_many`).
  So the engine marks the index unfinished in its `_meta` while a load runs,
  and does not reuse it until a load has finished, even after a crash. The
  command line also deletes a half-built index, on an error or on Ctrl-C, and
  with `--reindex` it checks every document of the file (`check_documents`)
  before it deletes the old index.
- **Facets.** A `terms` aggregation counts the matches per category. The
  category filter goes in `post_filter`, which runs *after* the counting, so
  the counts still show every category and the user can switch with one click.
  A filter inside the query would also shrink the counts.
- **Highlighting.** Elasticsearch returns snippets with the matches wrapped in
  `<mark>`. With `"encoder": "html"` it escapes the document text first, so a
  document containing `<script>` cannot inject code into a results page. It
  refuses to highlight past the first million characters of a field and fails
  the whole search, unless the request sets `max_analyzed_offset` to at most
  that limit (we send 999,999): then it only highlights the start of a longer
  field.
- **Autocomplete.** The title also has a `search_as_you_type` sub-field, which
  indexes word prefixes and two- and three-word shingles. A `bool_prefix`
  query matches every typed word exactly, except the last one, which may be
  a prefix: "inverted ind" finds "Inverted index". Its operator is "and", so
  every word must match; by default one of them would be enough.
- **"Did you mean".** A **term suggester** per word on a `spelling` field
  that holds every field's words, unstemmed. It allows the same number of
  typos as our `SpellingIndex` (fuzziness `AUTO`). Because that field is not
  stemmed, the same request also counts the documents each word matches in
  the searched fields: a word that matches is known and stays as it is, as in
  our engine ("analyzer" is fine when the index has "analyzers").
- **Unknown fields.** The mapping is `"dynamic": "false"`: a field we did not
  map (like `url`) is stored and returned, but not indexed. Otherwise every
  new field name in the data would add a mapping, until the index reaches
  `index.mapping.total_fields.limit` (1,000 fields) and refuses documents with
  new fields: a "mapping explosion".
- **Ties.** Lucene returns equal scores in index order, our engine by
  document id. The id is also indexed as a `keyword`, and the search sorts by
  score and then by id, so both engines keep the same documents when `k`
  cuts through a tie. That needs equal scores to *be* equal: Lucene's scores
  are 32-bit floats, while our engine adds up 64-bit ones, whose last bits
  depend on the order of the additions. Two scores that are equal in exact
  arithmetic could differ there and not tie. So our engine rounds its scores
  to 32-bit floats, like Lucene's, before it ranks them. (Rounded from slightly
  different sums, two scores could in rare cases still land on neighbouring
  32-bit floats.)

### 4.4 Where the two engines differ, on purpose

| | From scratch | Elasticsearch |
|---|---|---|
| Typos (`fuzzy`) | corrects each word to the one closest known word, then searches | expands each word to the indexed terms within the allowed edits and scores them all |
| Phrase on several fields, or with a weight other than 1 | BM25F with the phrase count | the best field's score, multiplied by its weight |
| TF-IDF | yes | no: Elasticsearch 9 has no built-in TF-IDF similarity (a `scripted` similarity could compute one) |
| Typos of two swapped letters ("rnak") | two edits | one edit (a transposition), so short words get more fixes |
| Fuzzy phrases | yes | not offered: phrase queries have no fuzziness, and intervals or span queries, which do, score differently |
| "Did you mean" with a wrong first letter and one more typo ("xlastisearch") | corrects it | no suggestion: Lucene's spell checker keeps the first letter when it looks two edits away |
| "Did you mean" between two equally close words | the one that occurs more often | the one in more documents |
| A failed load | `add_many` keeps the documents before the bad one, and none after it | the failing bulk request still indexes its good documents, even after the bad one; a document our own check refuses stops the load before its request is sent. The index is marked unfinished and not reused |

Everything else gives the same results with the same scores, to about 7
significant digits: words in OR and AND mode, any field weights, and phrases
on one field with weight 1. Typos are a good example of a design choice
rather than a right answer: correcting first is what web search engines do
("Showing results for..."); expanding is what Lucene's `FuzzyQuery` does.

### 4.5 Testing code that talks to a server

The engine is split the way many professional systems are: a **functional
core** of pure functions (`es_dsl.py`) and a thin **imperative shell** that
does the input and output (`es_engine.py`). Each layer gets its own tests:

1. **Pure functions** (`tests/test_es_dsl.py`): build a request and look at
   the dict. Fast, and no server needed.
2. **A fake client** (`tests/fake_es.py`, used by `test_es_engine.py` and
   `test_cli_es.py`): a small class with the methods the engine calls. It
   records what was sent and returns canned answers, including errors such
   as a 404 or a refused connection. This checks the shell's logic: bulk
   chunking, reading responses, error messages.
3. **The real thing** (`tests/test_es_integration.py`, marked `integration`):
   the 25 sample documents go into both engines, and 40 word queries (OR and
   AND, four weightings) must return the same documents with the same scores.
   Without a server these tests are skipped; in CI, `REQUIRE_ES=1` turns a
   missing server into a failure, so they can never be skipped silently.

The **golden file** from chapter 2 now comes from these same settings and
queries: `scripts/make_parity_golden.py` builds its index with
`index_settings()` and `index_mappings()` and its queries with
`build_query()`. Recording it again gave identical scores.

In CI, the `integration` job starts Elasticsearch as a **service container**
next to the job. It is the same image as `docker-compose.yml`, pinned by
digest, and `tests/test_repo_config.py` fails if the two ever drift apart.
The release workflow starts the same container, so a tag whose integration
tests fail is never released.

### Try it

```bash
docker compose up -d --wait                # Elasticsearch on localhost:9200
pip install -e ".[es]"
searchengine --engine es "inverted index"  # creates and fills the index once
searchengine --engine es --category ranking "bm25"
searchengine --engine es --explain "bm25 ranking"
searchengine --engine es "elastisearch"     # did you mean, from the term suggester
pytest -m integration                       # both engines side by side
```

### Interview questions you can now answer

- *Why can the same document get different scores on different shards?* Each
  shard computes N, df and avgdl from its own documents. Use one shard or
  `dfs_query_then_fetch` when exact scores matter.
- *`multi_match` or `combined_fields`?* `multi_match` with `best_fields` or
  `most_fields` is field-centric: each field scores on its own.
  `combined_fields` is term-centric and treats several fields as one (BM25F).
  `multi_match` with `cross_fields` is term-centric too: it blends each
  word's statistics across the fields, but still scores each field on its own
  and keeps the best field's score for each word.
- *`text` versus `keyword`?* `text` is analyzed into terms for full-text
  search; `keyword` is one exact value, for filters, sorting and aggregations.
- *Query, filter or `post_filter`?* A query scores; a filter only includes or
  excludes, and can be cached; `post_filter` runs after the aggregations.
- *When can a new document be found?* After a refresh makes its segment
  searchable: Elasticsearch is "near real-time".
- *How do you test code that calls a server?* Pure request builders, a fake
  client for the logic, and a few integration tests against the real server.

### Exercises

1. Create the index with `"number_of_shards": 3` and run the integration
   tests. Which ones fail, and why? Then search with
   `search_type="dfs_query_then_fetch"`.
2. Change `build_query` to use `multi_match` with `best_fields` instead of
   `combined_fields`. Which integration tests fail, and how do the top results
   change for "inverted index"?
3. Add a `synonym` token filter (for example "es" and "elasticsearch") to
   the Elasticsearch analyzer, and the same expansion to our engine. What
   must change in both so the parity tests still pass?
4. Move the category filter from `post_filter` into the query as a `bool`
   `filter`. What happens to the facet counts?
5. Run `searchengine --engine es "elastisearch"` and then with `--fuzzy`.
   Compare the results with `--engine scratch`, and explain the differences
   using 4.4.
6. Make `spelling.edit_distance` count two swapped neighbouring letters as
   one edit (the "optimal string alignment" distance), as Lucene does. Then
   "rnak" gets the same suggestion from both engines: add an integration
   test that shows it.

---

## Chapter 5: Measuring search quality on SciFact

Chapters 2 and 4 built two engines that agree with each other. Agreeing is
not the same as being good. This chapter measures how good they are, on a
benchmark that research papers use, and how fast they are. It also shows how
to tell a real improvement from luck, and how CI stops the numbers from
slipping.

### What was built

- `searchengine/evaluation.py`: judged queries, the five standard metrics
  (nDCG, MRR, precision, recall, average precision), `evaluate()` for a
  whole run, and a paired randomization test that says whether two systems
  really differ.
- `searchengine/datasets.py`: loaders for judged queries and for SciFact, and
  `fetch_scifact()`, which downloads SciFact and checks it against a pinned
  SHA-256 checksum.
- `searchengine/benchmark.py`, the `searchengine-bench` command. It runs five
  *systems* (an engine and a way of searching), scores and times each one,
  and writes the report as Markdown and JSON. With `--check` it fails when a
  quality score has moved since a recorded report.
- `ElasticsearchEngine.rank()`: a lean request that returns only ids and
  scores, with a `best_fields` query to compare against.
- `searchengine/data/sample_queries.jsonl`: 12 judged queries for the sample
  documents, so the benchmark runs without a download.
- `benchmarks/scifact.json` and `benchmarks/scifact.md`: the recorded results.
- A CI job that runs both engines on SciFact on every push, and fails if any
  quality score moves.
- `scripts/make_trec_eval_golden.py`, which records trec_eval's own scores
  for 200 random rankings in `tests/data/trec_eval.json`. Our metrics must
  agree with every one of them.

### 5.1 How search quality is measured

You cannot measure "good" without saying what the right answers are. A
**test collection** does that, with three parts:

| Part | What it is | In SciFact |
|---|---|---|
| Documents | what the engine searches | 5,183 abstracts of scientific papers |
| Queries | what users ask | 300 claims, like "0-dimensional biomaterials show inductive properties." |
| Relevance judgments ("qrels") | for each query, the documents a person judged relevant, with a grade | the abstracts each claim cites |

A **run** is what an engine returns: a ranked list of document ids for every
query. A metric compares the run with the judgments and gives a number from 0
to 1. Scoring a fixed collection offline like this is called the **Cranfield
paradigm**, after the experiments at Cranfield in the 1960s. TREC and BEIR
work this way.

Our own format for judged queries is JSON Lines:

```json
{"id": "q2", "query": "default ranking function in Elasticsearch", "relevant": {"2": 2, "3": 1, "5": 1}}
```

Grade 2 means highly relevant (document 2, "BM25 ranking function"), grade 1
relevant ("TF-IDF", "Apache Lucene"). A document that is not listed counts as
not relevant. That rule is a weakness of every test collection: a good
document that nobody judged counts as a wrong answer.

### 5.2 The metrics, by hand

Our engine's top 5 for q2:

| Rank | Document | Grade |
|---|---|---|
| 1 | 2, BM25 ranking function | 2 |
| 2 | 4, Elasticsearch | 0 |
| 3 | 5, Apache Lucene | 1 |
| 4 | 14, Faceted search | 0 |
| 5 | 13, Autocomplete | 0 |

| Metric | The question it answers | q2, top 5 |
|---|---|---|
| **Precision@k** | how much of the top k is relevant? | 2 of 5 = 0.40 |
| **Recall@k** | how many of the relevant documents are in the top k? | 2 of 3 = 0.67 |
| **Reciprocal rank** | how far down is the first relevant one? 1 / its rank | 1 / 1 = 1.00 |
| **Average precision** | the precision at each relevant document found, summed, divided by the number of relevant documents | (1/1 + 2/3) / 3 = 0.56 |
| **nDCG@k** | are the best documents at the top? It uses the grades | 2.5 / 3.13 = 0.80 |

Averaged over all the queries, reciprocal rank becomes **MRR** (mean
reciprocal rank) and average precision **MAP** (mean average precision).

**nDCG**, normalized discounted cumulative gain, is the main metric of BEIR
and of most web search work, because it is the only one of the five that
uses the grades. Each document's gain is its grade, and its rank r divides
the gain by log2(r + 1), so lower ranks count less:

```
DCG@5  = 2/log2(2) + 0/log2(3) + 1/log2(4) + 0 + 0 = 2 + 0.5        = 2.5
IDCG@5 = 2/log2(2) + 1/log2(3) + 1/log2(4)          = 2 + 0.63 + 0.5 = 3.13  (the best order)
nDCG@5 = DCG@5 / IDCG@5                                              = 0.80
```

Dividing by the ideal DCG puts every query on the same scale, from 0 to 1,
whether it has one relevant document or fifty.

Details like these decide whether your numbers can be compared with
published ones, so we follow **trec_eval**, the program TREC and BEIR score
with:

- The gain is the grade itself. Some papers use 2^grade - 1, which rewards
  highly relevant documents more. With grades 0 and 1 only, as in SciFact,
  the two agree.
- Recall and average precision divide by *all* the query's relevant
  documents, so a missed one counts as 0.
- A judged query the run has no answer for scores 0 and still counts, as with
  `trec_eval -c`. Otherwise an engine could look better by failing on the
  hard queries.

The report uses two cutoffs. **k = 10** for nDCG, MRR and precision, because
people look at the first page. **depth = 100** for recall and MAP: whether a
slower re-ranker, given our top 100, would have the right documents to work
with.

`test_scores_match_trec_eval` checks all five metrics against trec_eval's own
scores for 200 random judged queries and rankings, recorded once with
`pytrec_eval` by `scripts/make_trec_eval_golden.py`. Hypothesis tests check
the rules every metric must follow: each score is between 0 and 1, putting
the relevant documents first and the best grades first is a perfect score,
and nothing after position k changes a score cut at k.

### 5.3 SciFact and BEIR

**SciFact** (Wadden et al., EMNLP 2020) was made for checking scientific
claims. Annotators rewrote sentences that cite papers into claims, and each
claim keeps the abstracts its sentence cited: those are the documents to
find. Claims are short (12.5 words on average) and abstracts long (about 200
words), and they often use different words for the same thing. A claim about
"phosphatidylinositide 3-kinase" inhibitors cites three abstracts that only
ever say "PI3K". That is the **vocabulary mismatch** problem, and an engine
that matches words, like ours, cannot solve it: in 25 of the 339 judged
(claim, abstract) pairs, the abstract is not in our top 100.

**BEIR** (Thakur et al., NeurIPS 2021) gathers retrieval benchmarks from many
fields, so that a model can be tested on data it was not trained on, with
nDCG@10 as its main metric. SciFact is one of them. BEIR's SciFact test set is
SciFact's 300 *dev* claims, because the test claims come without published
judgments, and every one of its 339 judged pairs has grade 1. Our loader
does the same: the title stays the title, the abstract's sentences are joined
into the body, and each cited abstract is a relevant document with grade 1.

The data is **downloaded, not copied** into this repository, because SciFact
is shared under CC BY-NC 2.0. `fetch_scifact` checks the archive's SHA-256
against a pinned value before it unpacks anything, so every run uses exactly
the same data and a changed file is refused. It unpacks only the three files
it needs, under names it chooses, so an archive cannot write outside its
folder.

### 5.4 Is the difference real?

On the 300 claims, BM25F with the title counted three times scores 0.6860,
and with equal weights 0.6798. Is the title weight better, or was it lucky on
a few claims? Each claim's nDCG can be anywhere from 0 to 1, so a difference
of 0.006 in the mean could be noise.

A **paired randomization test** answers that. *Paired*, because both systems
answer the same queries, so we compare them query by query. If the two
systems were equally good, which one got a query's higher score would be luck,
and swapping the two scores would change nothing. So:

1. Take each query's difference: first system minus second.
2. Many times (10,000 here), flip the sign of each difference at random, and
   add them up.
3. The **p-value** is the share of those totals that are at least as far from
   zero as the real one.

```python
observed = abs(sum(differences))
extreme = 0
for _ in range(trials):
    flipped = sum(d if random_bit() else -d for d in differences)
    if abs(flipped) >= observed:
        extreme += 1
p = (extreme + 1) / (trials + 1)  # the real order counts as one of the orders
```

A small p, below 0.05 say, means random swaps rarely do as well, so the
difference is unlikely to be luck. Smucker, Allan and Carterette (CIKM 2007)
compared the significance tests used in information retrieval and recommend
this one: it assumes nothing about how the scores are distributed. Our seed is
fixed, so the same scores always give the same p.

Three cautions:

- **p is not the size of the effect.** With enough queries, a tiny difference
  becomes "significant". Report the difference and p together.
- **Many comparisons.** At 0.05, one comparison in 20 looks significant by
  luck alone. Our report makes four comparisons with row 1, so treat a p near
  0.05 with care. A Bonferroni correction would ask for 0.05 / 4 = 0.0125.
- **Few queries prove little.** On the sample data's 12 queries, equal weights
  differ from title x3 on one query only, and TF-IDF on three, by amounts that
  almost cancel out. The test gives p = 1.0 for both. With one differing
  query, flipping its sign cannot bring the total closer to zero, so every
  random order is as extreme as the real one.

### 5.5 Measuring speed fairly

A perfect ranking that takes ten seconds is useless, so the benchmark times
every query too, with these precautions:

- **Warm up first.** The first queries pay for things the later ones do not:
  loading code, filling caches, opening the connection to Elasticsearch, and
  the Java virtual machine compiling the code it runs most. The first 20
  queries run once, untimed.
- **Report the tail, not just the average.** The mean and the median describe
  a typical query. p95 and p99 are the times that 5% and 1% of the queries
  exceed. Users notice the slow ones, and a page that waits for 100 servers
  waits for the slowest. "The Tail at Scale" (Dean and Barroso, 2013) is the
  classic read on why.
- **Time what you compare.** `rank()` sends a lean request: no document text
  (`"_source": false`), no snippets or facet counts, and
  `"track_total_hits": false`, because counting every match costs time and a
  ranking does not need the count. Our engine is timed the same way: one call
  of `search` per query.
- **Two clocks for Elasticsearch.** The time seen from Python includes the
  client, the HTTP round trip and the JSON. Elasticsearch's own `took` covers
  only the search, in whole milliseconds. The difference is the price of the
  network and the client.
- **Do not fail CI on speed.** Times depend on the machine and on whatever
  else runs on it. The report shows them, but `--check` compares quality only.

The percentiles come from `statistics.quantiles(times, n=100,
method="inclusive")`, which interpolates between the measured times.

### 5.6 The results

| System | What it is |
|---|---|
| BM25F, title x3 | our engine as it ships: title words count three times |
| BM25F, equal weights | our engine, with title and body counted alike |
| TF-IDF, title x3 | our engine, with chapter 2's classic formula |
| combined_fields, title x3 | Elasticsearch, with the same BM25F as row 1 |
| best_fields, tie_breaker 0.5 | Elasticsearch, with the query the BEIR toolkit sends: each field scored on its own, then the best field's score plus half of the other's |

The recorded run (`benchmarks/scifact.md`), on the 300 dev claims:

| System | Engine | nDCG@10 | MRR@10 | P@10 | Recall@100 | MAP@100 | p (nDCG vs row 1) |
|---|---|---|---|---|---|---|---|
| BM25F, title x3 | scratch | 0.6860 | 0.6505 | 0.0900 | 0.9276 | 0.6450 |  |
| BM25F, equal weights | scratch | 0.6798 | 0.6442 | 0.0893 | 0.9276 | 0.6393 | 0.1305 |
| TF-IDF, title x3 | scratch | 0.6524 | 0.6138 | 0.0873 | 0.9092 | 0.6071 | 0.0003 |
| combined_fields, title x3 | es | 0.6860 | 0.6505 | 0.0900 | 0.9276 | 0.6450 | 1.0000 |
| best_fields, tie_breaker 0.5 | es | 0.6904 | 0.6564 | 0.0907 | 0.9142 | 0.6501 | 0.5130 |

| System | Engine | Mean ms | Median ms | p95 ms | p99 ms | Inside Elasticsearch, median ms |
|---|---|---|---|---|---|---|
| BM25F, title x3 | scratch | 16.81 | 15.31 | 34.63 | 42.67 |  |
| BM25F, equal weights | scratch | 17.79 | 16.14 | 38.93 | 44.91 |  |
| TF-IDF, title x3 | scratch | 6.58 | 5.91 | 14.32 | 16.92 |  |
| combined_fields, title x3 | es | 8.26 | 7.68 | 11.57 | 17.92 | 5.00 |
| best_fields, tie_breaker 0.5 | es | 7.29 | 7.01 | 9.77 | 14.73 | 5.00 |

What they say:

- **The two engines agree on a real benchmark.** Our BM25F and Elasticsearch's
  `combined_fields` get the same quality to every digit shown, and p = 1.0
  because no claim's score differs. Compared directly, the two runs are
  identical: for all 300 claims, both engines return the same 100 documents
  in the same order. Chapter 4 showed this on 25 documents; this is 5,183.
- **The title weight does not clearly help.** Equal weights score 0.006 less,
  with p = 0.13: we cannot tell that from luck.
- **BM25 beats TF-IDF.** TF-IDF scores 0.034 less, with p = 0.0003. Abstracts
  are long, and this is what BM25's saturation and length normalization
  (section 2.4) are for.
- **`best_fields` is level with BM25F.** Its nDCG@10 is 0.004 higher, with
  p = 0.51, and its recall@100 is lower: 0.914 against 0.928.
- **Published numbers need their settings.** The BEIR paper reports an
  nDCG@10 of 0.665 for BM25 on SciFact. That comes from Anserini, a toolkit
  built on Lucene, with k1 = 0.9 and b = 0.4, and the title and text as
  separate fields. The BEIR toolkit's own Elasticsearch search sends the
  `best_fields` query above, with Elasticsearch's built-in `english`
  analyzer. Our systems use k1 = 1.2 and b = 0.75, Elasticsearch's defaults,
  and chapter 2's analyzer. "BM25" is a family, not one number: the analyzer,
  k1 and b, and the way fields are combined all move the score. That is why a
  paper must state them, and why we compare our systems on the same data, in
  the same run.
- **Speed.** Elasticsearch answers in 7.7 ms at the median, and our engine in
  15.3 ms. At p99 it is 18 ms against 43 ms. About 5 ms of Elasticsearch's
  time is the search itself; the rest is the client and the round trip.
  TF-IDF is our fastest system because it ignores document length, while
  BM25F works out, for every query word, the length of every matching
  document, rounded the way Lucene stores it. Loading the 5,183 documents
  took 4.8 s into our index, and 3.3 s into Elasticsearch.

These times come from one run on one machine. Run the benchmark yourself and
yours will differ; the quality scores will not.

### 5.7 Keeping the numbers honest

A benchmark run once is a claim. A benchmark that CI runs on every push keeps
the claim true.

- `benchmarks/scifact.json` holds the recorded report. The `benchmark` job in
  CI downloads SciFact (its checksum pinned), runs both engines, and
  `--check benchmarks/scifact.json` fails if any quality score has moved by
  more than 0.0001, or if a system is not in the recording. The tables appear
  on the job's summary page, and the JSON is kept for 30 days.
- A change that is *meant* to move quality, a better analyzer for example,
  records the report again in the same pull request (CONTRIBUTING.md says
  how). The diff of `benchmarks/scifact.json` then shows the reviewer exactly
  what moved.
- The benchmark makes its own Elasticsearch index and deletes it afterwards,
  even when the run fails. It refuses an index that already exists, because
  that may hold someone's data.

### Try it

```bash
searchengine-bench                           # our engine, the sample data's 12 judged queries
docker compose up -d --wait                  # Elasticsearch, for the rest
searchengine-bench --engine both             # both engines on the sample data
searchengine-bench --scifact datasets/scifact --download --engine both   # SciFact, under a minute
make scifact                                 # the same, checked against benchmarks/scifact.json
searchengine-bench --scifact datasets/scifact --split train --queries 100
```

### Interview questions you can now answer

- *How do you measure search quality offline?* With a test collection
  (documents, queries and graded judgments), a run for each system, and
  metrics such as nDCG@10.
- *nDCG, MAP or MRR?* MRR when one good answer is enough, like a question
  with one right page. MAP when every relevant document matters and the
  judgments are yes or no. nDCG when the judgments are graded and the top
  positions matter most.
- *Why normalize DCG?* So that every query counts on the same scale, and the
  queries with many relevant documents do not dominate the mean.
- *Your change raised nDCG@10 from 0.686 to 0.690. Ship it?* Not on that
  alone. Test it with a paired test over the queries (here p = 0.51: it could
  be luck), look at the other metrics (recall@100 fell), and then test it
  online, with real users.
- *Why report p99 latency?* The mean hides the slow tail that users notice,
  and a request that fans out to many servers waits for the slowest one.
- *What are the limits of offline evaluation?* The judgments are incomplete
  (a good document nobody judged counts as wrong), the queries may not look
  like real traffic, and it says nothing about how users behave. Online A/B
  tests and click metrics measure that.
- *How do you stop quality from slipping?* Record the benchmark, fail CI when
  a score moves, and record it again on purpose, in review.

### Exercises

1. Add systems with the title weight at 2 and at 5. Which is best on the dev
   claims? Then run them on the train claims (`--split train`). Does the
   winner hold? Tuning on the data you report on is overfitting.
2. Try BEIR's k1 = 0.9 and b = 0.4 in our engine (`SearchOptions(k1=0.9,
   b=0.4)`). Does nDCG@10 move towards 0.665? Is the change significant?
3. Index SciFact in Elasticsearch with its built-in `english` analyzer, and
   run the `best_fields` query on it. How much does the analyzer alone change
   nDCG@10?
4. Write a paired t-test and compare its p-values with the randomization
   test's. When do they disagree?
5. Add a 95% bootstrap confidence interval for each system's nDCG@10:
   resample the 300 claims with replacement 10,000 times.
6. Score nDCG with the gain 2^grade - 1, and find the sample queries whose
   score changes. Why does no SciFact score change?
7. Write a test that both engines return the same top 100 for every SciFact
   claim, and run it in the CI job.
