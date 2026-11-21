"""A search engine built from scratch, and the same engine on Elasticsearch."""

from .analyzer import ENGLISH_STOPWORDS, Token, analyze, analyze_tokens, tokenize
from .index import ClauseScore, Hit, InvertedIndex, SearchOptions
from .porter import porter_stem
from .scoring import bm25_idf, bm25_tf, tfidf_idf, tfidf_tf
from .spelling import SpellingIndex, edit_distance, max_edits_auto

__version__ = "0.1.0"

__all__ = (
    "ENGLISH_STOPWORDS",
    "ClauseScore",
    "Hit",
    "InvertedIndex",
    "SearchOptions",
    "SpellingIndex",
    "Token",
    "__version__",
    "analyze",
    "analyze_tokens",
    "bm25_idf",
    "bm25_tf",
    "edit_distance",
    "max_edits_auto",
    "porter_stem",
    "tfidf_idf",
    "tfidf_tf",
    "tokenize",
)
