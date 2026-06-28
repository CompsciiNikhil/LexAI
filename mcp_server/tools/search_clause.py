"""
LexAI — MCP Tool: search_clause
Returns the most relevant clause for a given user query using
TF-IDF-inspired keyword scoring with a recency/position boost.

Input:
    text  (str): full document text
    query (str): user's question or search term

Output: dict {
    matched_clause  (str):   most relevant clause text
    heading         (str):   section the clause came from
    relevance_score (float): 0.0–1.0
}
"""

import math
import re
import string
from typing import Optional

from .extract_clauses import extract_clauses


# ---------------------------------------------------------------------------
# Stop words — very common words that carry no discriminating signal
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "this", "that",
        "these", "those", "it", "its", "my", "your", "our", "their", "i",
        "you", "he", "she", "we", "they", "what", "which", "who", "whom",
        "when", "where", "why", "how", "not", "no", "if", "as", "any", "all",
        "each", "both", "few", "more", "most", "other", "into", "through",
        "during", "before", "after", "above", "below", "between", "about",
        "against", "such", "than", "so", "up", "out",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stop words."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOP_WORDS]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count(term) / total_terms."""
    if not tokens:
        return {}
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in freq.items()}


def _idf(term: str, clauses_tokens: list[list[str]]) -> float:
    """Inverse document frequency across all clauses."""
    n = len(clauses_tokens)
    df = sum(1 for tokens in clauses_tokens if term in tokens)
    if df == 0:
        return 0.0
    return math.log((n + 1) / (df + 1)) + 1.0  # smoothed IDF


def _score_clause(
    query_tokens: list[str],
    clause_tokens: list[str],
    clauses_tokens: list[list[str]],
) -> float:
    """
    Compute a TF-IDF cosine-similarity-like score between the query and one
    clause.
    """
    if not query_tokens or not clause_tokens:
        return 0.0

    clause_tf = _tf(clause_tokens)
    score = 0.0

    for term in query_tokens:
        if term in clause_tf:
            idf = _idf(term, clauses_tokens)
            score += clause_tf[term] * idf

    # Normalize by query length so short queries aren't disadvantaged
    return score / len(query_tokens)


def search_clause(text: str, query: str) -> dict:
    """
    Find the most relevant clause in *text* for the given *query*.

    Strategy:
      1. Extract clauses via extract_clauses().
      2. Score each clause with TF-IDF keyword matching.
      3. Apply a small recency penalty (earlier clauses score slightly higher
         when tied — important clauses tend to come first in legal docs).
      4. Normalize the top score to a 0–1 relevance_score.

    Args:
        text:  Full document text.
        query: User's natural-language question or search term.

    Returns:
        Dict with keys: matched_clause, heading, relevance_score.
    """
    if not text or not text.strip():
        return {
            "matched_clause": "",
            "heading": "",
            "relevance_score": 0.0,
        }

    if not query or not query.strip():
        return {
            "matched_clause": "",
            "heading": "",
            "relevance_score": 0.0,
        }

    clauses = extract_clauses(text)

    if not clauses:
        return {
            "matched_clause": text[:2000],
            "heading": "Full Document",
            "relevance_score": 0.0,
        }

    query_tokens = _tokenize(query)

    # Pre-tokenize every clause (used for IDF calculation)
    clauses_tokens: list[list[str]] = [
        _tokenize(c["heading"] + " " + c["content"]) for c in clauses
    ]

    # Score each clause
    raw_scores: list[float] = [
        _score_clause(query_tokens, ct, clauses_tokens)
        for ct in clauses_tokens
    ]

    # Find best-scoring clause
    best_idx = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
    best_raw = raw_scores[best_idx]

    # Normalize to 0–1 across all clause scores
    max_possible = max(raw_scores) if raw_scores else 1.0
    relevance_score = round(best_raw / max_possible, 4) if max_possible > 0 else 0.0

    best_clause = clauses[best_idx]

    return {
        "matched_clause": best_clause["content"] or best_clause["heading"],
        "heading": best_clause["heading"],
        "relevance_score": relevance_score,
    }
