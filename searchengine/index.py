"""An inverted index with TF-IDF and BM25 ranking, built from scratch.

An inverted index maps each term to the documents containing it:

    "elast" -> {doc 3: [0, 14], doc 9: [5]}
     term       doc id: positions of the term inside that doc

Searching then means: look up each query term, collect the docs, and
score them. We never scan every document, which is what makes search fast.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .analyzer import analyze


@dataclass
class Hit:
    doc_id: str
    score: float
    title: str = ""


@dataclass
class InvertedIndex:
    # term -> {doc_id -> [positions]}
    postings: dict[str, dict[str, list[int]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    doc_lengths: dict[str, int] = field(default_factory=dict)
    docs: dict[str, dict] = field(default_factory=dict)
    fields: tuple[str, ...] = ("title", "body")
    # A word in the title is a stronger signal than one in the body.
    field_boosts: dict[str, int] = field(default_factory=lambda: {"title": 3, "body": 1})

    # ---------- indexing ----------

    def add(self, doc: dict) -> None:
        """Index one document: {"id": ..., "title": ..., "body": ...}."""
        doc_id = str(doc["id"])
        if doc_id in self.docs:
            raise ValueError(f"duplicate document id: {doc_id}")
        self.docs[doc_id] = doc

        position = 0
        for name in self.fields:
            terms = analyze(doc.get(name, ""))
            # Boosting by repetition: a title term is recorded several times,
            # so it counts more in term frequency. Simple and effective.
            for _ in range(self.field_boosts.get(name, 1)):
                for term in terms:
                    self.postings[term].setdefault(doc_id, []).append(position)
                    position += 1
                position += 100  # gap so a phrase can't span two fields or copies
        # Length = number of (boosted) terms recorded for this doc.
        self.doc_lengths[doc_id] = sum(
            len(analyze(doc.get(name, ""))) * self.field_boosts.get(name, 1)
            for name in self.fields
        )

    def add_many(self, docs) -> None:
        for doc in docs:
            self.add(doc)

    # ---------- statistics used by ranking ----------

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / max(self.n_docs, 1)

    def df(self, term: str) -> int:
        """Document frequency: how many docs contain the term."""
        return len(self.postings.get(term, {}))

    def tf(self, term: str, doc_id: str) -> int:
        """Term frequency: how many times the term appears in one doc."""
        return len(self.postings.get(term, {}).get(doc_id, []))

    # ---------- scoring ----------

    def idf(self, term: str) -> float:
        """Inverse document frequency (the BM25/Lucene variant).

        Rare terms get a high weight, common terms a low one.
        The +1 inside the log keeps it positive even for very common terms.
        """
        n, df = self.n_docs, self.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score_tfidf(self, terms: list[str], doc_id: str) -> float:
        """Classic TF-IDF: sum over terms of  log(1 + tf) * idf."""
        return sum(
            math.log(1 + self.tf(t, doc_id)) * self.idf(t) for t in terms
        )

    def score_bm25(self, terms: list[str], doc_id: str, k1: float = 1.2, b: float = 0.75) -> float:
        """BM25, the default ranking function in Elasticsearch and Lucene.

        Two improvements over TF-IDF:
        * k1 (term saturation): the 10th occurrence of a word adds much less
          than the 1st. tf grows the score but levels off towards (k1 + 1).
        * b (length normalization): a match in a short doc counts more than
          the same match buried in a very long doc.
        """
        length_ratio = self.doc_lengths[doc_id] / self.avg_doc_length
        score = 0.0
        for t in terms:
            tf = self.tf(t, doc_id)
            if tf == 0:
                continue
            norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * length_ratio))
            score += self.idf(t) * norm
        return score

    # ---------- searching ----------

    def _candidates(self, terms: list[str], mode: str) -> set[str]:
        sets = [set(self.postings.get(t, {})) for t in terms]
        if not sets:
            return set()
        return set.intersection(*sets) if mode == "and" else set.union(*sets)

    def _phrase_match(self, terms: list[str], doc_id: str) -> bool:
        """True if the terms appear next to each other, in order."""
        first = self.postings.get(terms[0], {}).get(doc_id, [])
        following = [set(self.postings.get(t, {}).get(doc_id, [])) for t in terms[1:]]
        return any(
            all(start + i + 1 in positions for i, positions in enumerate(following))
            for start in first
        )

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        ranker: str = "bm25",
        mode: str = "or",
        fuzzy: bool = False,
    ) -> list[Hit]:
        """Search the index.

        query  : text; wrap it in double quotes for an exact phrase.
        ranker : "bm25" or "tfidf".
        mode   : "or" (any term) or "and" (all terms).
        fuzzy  : correct typos by mapping unknown terms to the closest known term.
        """
        phrase = len(query) > 1 and query.startswith('"') and query.endswith('"')
        terms = analyze(query.strip('"'))
        if fuzzy:
            terms = [self.correct(t) for t in terms]
        if not terms:
            return []

        candidates = self._candidates(terms, "and" if phrase else mode)
        if phrase:
            candidates = {d for d in candidates if self._phrase_match(terms, d)}

        score = self.score_bm25 if ranker == "bm25" else self.score_tfidf
        hits = [Hit(d, score(terms, d), self.docs[d].get("title", "")) for d in candidates]
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]

    # ---------- typo tolerance ----------

    def correct(self, term: str, max_distance: int = 2) -> str:
        """Return the indexed term closest to `term` (by edit distance).

        Ties are broken by document frequency: prefer the more common word.
        """
        if term in self.postings:
            return term
        best, best_key = term, (max_distance + 1, 0)
        for candidate in self.postings:
            if abs(len(candidate) - len(term)) > max_distance:
                continue  # cheap filter before the expensive computation
            d = edit_distance(term, candidate, max_distance)
            key = (d, -self.df(candidate))
            if d <= max_distance and key < best_key:
                best, best_key = candidate, key
        return best

    def suggest(self, query: str) -> str | None:
        """'Did you mean ...?' Returns a corrected query, or None if nothing changed."""
        terms = analyze(query.strip('"'))
        fixed = [self.correct(t) for t in terms]
        return " ".join(fixed) if fixed != terms else None

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "docs": self.docs,
            "field_boosts": self.field_boosts,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        data = json.loads(Path(path).read_text())
        idx = cls(field_boosts=data["field_boosts"])
        idx.postings = defaultdict(dict, data["postings"])
        idx.doc_lengths = data["doc_lengths"]
        idx.docs = data["docs"]
        return idx

    def stats(self) -> dict:
        return {
            "documents": self.n_docs,
            "unique_terms": len(self.postings),
            "avg_doc_length": round(self.avg_doc_length, 1),
            "most_common_terms": Counter({t: self.df(t) for t in self.postings}).most_common(5),
        }


def edit_distance(a: str, b: str, limit: int = 99) -> int:
    """Levenshtein distance: the fewest single-letter edits turning a into b.

    Classic dynamic programming, keeping only one row in memory.
    Stops early once every value in a row is above `limit`.
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,               # delete
                cur[j - 1] + 1,            # insert
                prev[j - 1] + (ca != cb),  # substitute
            ))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]
