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
| 4 | Elasticsearch | coming next |
| 5 | Measuring search quality on SciFact | planned |
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
in Elasticsearch, and the two must produce identical terms. Two details of
Java, the language Lucene is written in, needed copying:

- *Lowercasing.* Python's `str.lower()` follows Unicode's context rules: a
  capital sigma at the end of a Greek word becomes the final form "ς", and the
  Turkish dotted capital "İ" becomes "i" plus a combining dot. Java lowercases
  one character at a time, with neither rule. So `_lowercase` does the same.
- *Length.* Tokens longer than 255 characters are dropped, but Java counts
  length in UTF-16 units, where an emoji or a character such as "𝐀" counts
  twice. `java_length` counts the same way.

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
postings["invert"] = {"1":  {"title": [0], "body": [1, 12]},
                      "15": {"body": [9]}}
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
(its *norm*) in a single byte to save memory: 0 to 40 exactly, and above that
only 4 significant bits. So 41 is stored as 40 and 375 as 344, and BM25 scores
with the stored value. `lucene_length` in `scoring.py` does the same. With
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
everything. Ties are broken by document id, so results are reproducible.

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

Every pull request runs all of these in GitHub Actions. You can run them
locally with `make check`.

| Gate | Tool | Example of what it catches |
|---|---|---|
| Lint | ruff, 20 rule sets | an unused variable, a mutable default argument, `assert` on a tuple |
| Format | ruff format | inconsistent quotes and line breaks (it fixes them) |
| Types | `mypy --strict` | calling `.lower()` on something that might be `None` |
| Coding standard | `scripts/check_jpl.py` | recursion, `while True`, a 70-line function |
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
2. **Every loop has a bound.** No `while True`, and every input that drives a
   loop has a size limit: text length, token length, query terms, results.
3. **No unbounded memory growth.** Files are read one line at a time, and a
   line may be at most 10 MB; result lists are capped (1,000 hits, 50
   spelling candidates).
4. **Short functions.** At most 60 lines and complexity 10.
5. **Check for the impossible.** Every public function validates its inputs
   with clear errors; `assert` only guards internal invariants.
6. **Smallest scope.** No global state; constants are `Final` and immutable.
7. **Check every result.** `mypy --strict` makes you handle `None`; saved
   index files are validated before they are trusted; the command line turns
   bad input into a one-line error instead of a traceback.
8. **No runtime code generation.** No `eval` or `exec`.
9. **Simple indirection.** No `getattr` with computed names; a static call graph.
10. **Zero warnings, analyzed daily.** Any warning fails the tests, and CI
    runs every day even when nobody pushes.

**How `check_jpl.py` works.** Python can parse its own source into an
**abstract syntax tree** (the `ast` module). The checker walks that tree:
function length is `end_lineno - lineno + 1`; `while True` is a `While` node
whose test is a true constant; recursion is found by building a **call graph**
(which function calls which) and searching it for cycles. It has its own
tests, including one that runs it on this project's code.

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
- **Daily schedule**: catches new vulnerabilities and breaking dependency
  releases even when the code did not change.
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
make check                       # the same gates as CI
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
