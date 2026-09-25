"""feature_extractor.py - 28-Feature Pairwise Feature Engineering.

Features computed for every (S1, S2/S3) candidate pair.
All features are scalar floats in [0, 1] or binary flags.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

try:
    from Levenshtein import ratio as lev_ratio, jaro_winkler
    HAS_LEVENSHTEIN = True
except ImportError:
    import difflib
    HAS_LEVENSHTEIN = False

try:
    from metaphone import doublemetaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False

from .preprocessor import Preprocessor


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_LEVENSHTEIN:
        return lev_ratio(a, b)
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaro_winkler(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_LEVENSHTEIN:
        return jaro_winkler(a, b)
    return difflib.SequenceMatcher(None, a, b).ratio()


def _token_sort_ratio(a: str, b: str) -> float:
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return _levenshtein_ratio(a_sorted, b_sorted)


def _token_set_ratio(a: str, b: str) -> float:
    a_toks = set(a.split())
    b_toks = set(b.split())
    inter = a_toks & b_toks
    only_a = a_toks - b_toks
    only_b = b_toks - a_toks
    s_inter = " ".join(sorted(inter))
    s_a = " ".join(sorted(inter | only_a))
    s_b = " ".join(sorted(inter | only_b))
    ratio_ab = _levenshtein_ratio(s_inter, s_a)
    ratio_ba = _levenshtein_ratio(s_inter, s_b)
    return max(ratio_ab, ratio_ba)


def _lcs_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for c in a:
        curr = [0] * (n + 1)
        for j, d in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if c == d else max(prev[j], curr[j - 1])
        prev = curr
    return (2 * prev[n]) / (m + n)


def _jaccard_tokens(a: str, b: str) -> float:
    a_toks = set(a.split())
    b_toks = set(b.split())
    if not a_toks and not b_toks:
        return 1.0
    if not a_toks or not b_toks:
        return 0.0
    return len(a_toks & b_toks) / len(a_toks | b_toks)


def _prefix_match(a: str, b: str, n: int = 4) -> float:
    n = min(n, len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def _monge_elkan(a: str, b: str) -> float:
    a_toks = a.split()
    b_toks = b.split()
    if not a_toks or not b_toks:
        return 0.0
    total = 0.0
    for tok_a in a_toks:
        best = max((_levenshtein_ratio(tok_a, tok_b) for tok_b in b_toks), default=0.0)
        total += best
    return total / len(a_toks)


def _tfidf_ngram_cosine(a: str, b: str, n: int = 3) -> float:
    a_ngrams = set(Preprocessor.ngrams(a, n))
    b_ngrams = set(Preprocessor.ngrams(b, n))
    if not a_ngrams and not b_ngrams:
        return 1.0
    if not a_ngrams or not b_ngrams:
        return 0.0
    return len(a_ngrams & b_ngrams) / (len(a_ngrams) * len(b_ngrams)) ** 0.5


def _cosine_embedding(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    if v1 is None or v2 is None:
        return 0.0
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _double_metaphone_match(a: str, b: str) -> float:
    if not HAS_METAPHONE:
        return 0.0
    a_tokens = a.split()[:3]
    b_tokens = b.split()[:3]
    if not a_tokens or not b_tokens:
        return 0.0
    a_codes: set[str] = set()
    b_codes: set[str] = set()
    for tok in a_tokens:
        codes = doublemetaphone(tok)
        a_codes.update(c for c in codes if c)
    for tok in b_tokens:
        codes = doublemetaphone(tok)
        b_codes.update(c for c in codes if c)
    if not a_codes or not b_codes:
        return 0.0
    return float(bool(a_codes & b_codes))


def _digit_jaccard(a_digits: set[str], b_digits: set[str]) -> float:
    if not a_digits and not b_digits:
        return 1.0
    if not a_digits or not b_digits:
        return 0.5
    inter = a_digits & b_digits
    union = a_digits | b_digits
    return len(inter) / len(union)


def _postal_exact(a: Optional[str], b: Optional[str]) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.5
    return 1.0 if a == b else 0.0


FEATURE_NAMES = [
    "levenshtein_name",
    "jaro_winkler_name",
    "monge_elkan_name",
    "token_sort_ratio_name",
    "token_set_ratio_name",
    "lcs_ratio_name",
    "tfidf_ngram_cosine_name",
    "jaccard_addr_tokens",
    "digit_jaccard_addr",
    "postal_exact_match",
    "double_metaphone_name",
    "word_len_diff_name",
    "prefix_match_name_4",
    "token_count_diff_name",
    "name_char_len_ratio",
    "source_is_s2",
    "source_is_s3",
    "cosine_name_emb",
    "cosine_addr_emb",
    "suffix_match_flag",
    "both_have_address",
    "shared_rare_name_token",
    "token_sort_ratio_addr",
    "token_set_ratio_addr",
    "levenshtein_addr",
    "country_encoded",
    "addr_has_digits",
    "name_exact_match",
]

assert len(FEATURE_NAMES) == 28, f"Expected 28 features, got {len(FEATURE_NAMES)}"


class FeatureExtractor:
    """Computes the 28-dimensional feature vector for a candidate pair."""

    CORPORATE_SUFFIXES = {
        "pvt", "ltd", "llc", "corp", "inc", "co", "sa", "sas",
        "sarl", "gmbh", "ag", "llp", "plc", "bv", "nv",
    }

    @classmethod
    def _suffix_match(cls, a: str, b: str) -> float:
        a_last = a.split()[-1] if a.split() else ""
        b_last = b.split()[-1] if b.split() else ""
        if not a_last or not b_last:
            return 0.5
        if a_last == b_last:
            return 1.0
        if a_last in cls.CORPORATE_SUFFIXES and b_last in cls.CORPORATE_SUFFIXES:
            return 0.7
        return 0.0

    @staticmethod
    def _shared_rare_token(a_toks: set[str], b_toks: set[str], threshold: int = 3) -> float:
        shared = a_toks & b_toks
        rare = {t for t in shared if len(t) > threshold}
        return 1.0 if rare else 0.0

    @classmethod
    def extract(
        cls,
        s1: dict,
        candidate: dict,
        s1_name_emb: Optional[np.ndarray] = None,
        s1_addr_emb: Optional[np.ndarray] = None,
        cand_name_emb: Optional[np.ndarray] = None,
        cand_addr_emb: Optional[np.ndarray] = None,
    ) -> list[float]:
        """Returns list of 28 float features for (s1, candidate) pair."""
        n1 = s1.get("cleaned_name", "")
        n2 = candidate.get("cleaned_name", "")
        a1 = s1.get("cleaned_address", "")
        a2 = candidate.get("cleaned_address", "")

        d1 = set(s1.get("addr_digits", []))
        d2 = set(candidate.get("addr_digits", []))
        p1 = s1.get("postal_code")
        p2 = candidate.get("postal_code")

        t1 = set(s1.get("name_tokens", n1.split()))
        t2 = set(candidate.get("name_tokens", n2.split()))

        cand_id = candidate.get("entity_id", "")
        source_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
        source_is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

        n1_words = n1.split()
        n2_words = n2.split()
        len1, len2 = len(n1_words), len(n2_words)
        word_len_diff = 1.0 - abs(len1 - len2) / max(len1 + len2, 1)
        tok_count_diff = 1.0 - abs(len1 - len2) / max(max(len1, len2), 1)
        char_len1, char_len2 = len(n1), len(n2)
        char_len_ratio = min(char_len1, char_len2) / max(max(char_len1, char_len2), 1)

        features = [
            _levenshtein_ratio(n1, n2),
            _jaro_winkler(n1, n2),
            _monge_elkan(n1, n2),
            _token_sort_ratio(n1, n2),
            _token_set_ratio(n1, n2),
            _lcs_ratio(n1, n2),
            _tfidf_ngram_cosine(n1, n2),
            _jaccard_tokens(a1, a2),
            _digit_jaccard(d1, d2),
            _postal_exact(p1, p2),
            _double_metaphone_match(n1, n2),
            word_len_diff,
            _prefix_match(n1, n2, 4),
            tok_count_diff,
            char_len_ratio,
            source_is_s2,
            source_is_s3,
            _cosine_embedding(s1_name_emb, cand_name_emb),
            _cosine_embedding(s1_addr_emb, cand_addr_emb),
            cls._suffix_match(n1, n2),
            1.0 if a1 and a2 else 0.0,
            cls._shared_rare_token(t1, t2),
            _token_sort_ratio(a1, a2),
            _token_set_ratio(a1, a2),
            _levenshtein_ratio(a1, a2),
            0.0,
            1.0 if (d1 or d2) else 0.0,
            1.0 if n1 == n2 and n1 else 0.0,
        ]

        assert len(features) == 28, f"Feature count mismatch: {len(features)}"
        return features

    @classmethod
    def extract_batch(
        cls,
        pairs: list[tuple[dict, dict]],
        embeddings: Optional[dict] = None,
    ) -> np.ndarray:
        """Extract features for a list of (s1_rec, cand_rec) tuples.

        Returns np.ndarray of shape (len(pairs), 28).
        """
        emb = embeddings or {}
        rows = []
        for s1, cand in pairs:
            s1_id = s1.get("entity_id", "")
            cand_id = cand.get("entity_id", "")
            s1_embs = emb.get(s1_id, {})
            cand_embs = emb.get(cand_id, {})
            row = cls.extract(
                s1, cand,
                s1_name_emb=s1_embs.get("name"),
                s1_addr_emb=s1_embs.get("addr"),
                cand_name_emb=cand_embs.get("name"),
                cand_addr_emb=cand_embs.get("addr"),
            )
            rows.append(row)
        return np.array(rows, dtype=np.float32)
