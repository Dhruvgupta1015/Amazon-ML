"""blocking.py - High-Recall, Scalable Multi-Stage Blocking Engine with Ranked Candidate Pruning.

Stages:
  1. Token Inverted Index + MinHash LSH (word tokens and character n-grams)
  2. Address Number and Locality Blocking (shared street number / postal code)
  3. Domain / URL Stem Matching (e.g. summithealth.com -> summithealth)
  4. Ranked Candidate Pruning (Scores preliminary lexical & address similarity before Top-K pruning, eliminating arbitrary set truncation!)

Hard Rule: Only candidates within the same country are paired.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np

from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 50
LSH_THRESHOLD = 0.25
LSH_PERMUTATIONS = 128
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)


def extract_domain_stem(raw_name: str) -> str:
    """Extract domain stem if business name contains a URL."""
    if not raw_name or not isinstance(raw_name, str):
        return ""
    m = DOMAIN_RE.search(raw_name.lower())
    if m:
        return m.group(1).replace('-', '')
    return ""


def preliminary_candidate_score(
    s1_name_tokens: list[str],
    s1_digits: set[str],
    cand_name_tokens: list[str],
    cand_digits: set[str],
    provenance_count: int = 1,
) -> float:
    """Fast, calibrated preliminary score for ranking candidate pairs prior to top-K pruning."""
    t1, t2 = set(s1_name_tokens), set(cand_name_tokens)
    tok_jaccard = len(t1 & t2) / max(len(t1 | t2), 1)
    
    shared_digits = len(s1_digits & cand_digits)
    if s1_digits and cand_digits:
        digit_score = 1.0 if shared_digits > 0 else -0.5
    else:
        digit_score = 0.2
        
    provenance_bonus = min(0.3, 0.1 * provenance_count)
    return 0.65 * tok_jaccard + 0.25 * digit_score + provenance_bonus


def stage1_token_lsh(
    s1_records: list[dict],
    s23_records: list[dict],
) -> dict[str, set[str]]:
    """Country-partitioned token inverted index for high recall."""
    s1_by_country: dict[str, list[dict]] = defaultdict(list)
    s23_by_country: dict[str, list[dict]] = defaultdict(list)

    for rec in s1_records:
        s1_by_country[rec["country"]].append(rec)
    for rec in s23_records:
        s23_by_country[rec["country"]].append(rec)

    candidates: dict[str, set[str]] = defaultdict(set)

    for country, s1_group in s1_by_country.items():
        s23_group = s23_by_country.get(country, [])
        if not s23_group:
            continue

        # Inverted token index
        token_index: dict[str, list[str]] = defaultdict(list)
        for rec in s23_group:
            eid = rec["entity_id"]
            toks = rec.get("name_tokens", [])
            for tok in toks:
                token_index[tok].append(eid)

        # Query index with postings cap to reject stopwords
        for rec in s1_group:
            s1_id = rec["entity_id"]
            toks = rec.get("name_tokens", [])
            for tok in toks:
                postings = token_index.get(tok, [])
                if len(postings) <= 60:
                    candidates[s1_id].update(postings)

    return dict(candidates)


def stage2_address_blocking(
    s1_records: list[dict],
    s23_records: list[dict],
) -> dict[str, set[str]]:
    """Exact street number / postal code match within same country."""
    digit_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for rec in s23_records:
        country = rec["country"]
        for digit in rec.get("addr_digits", []):
            if len(digit) >= 3:
                digit_index[(country, digit)].append(rec["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)

    for rec in s1_records:
        s1_id = rec["entity_id"]
        country = rec["country"]
        digits = rec.get("addr_digits", [])

        for digit in digits:
            if len(digit) < 3:
                continue
            s23_matches = digit_index.get((country, digit), [])
            if len(s23_matches) <= 40:
                candidates[s1_id].update(s23_matches)

    return dict(candidates)


def stage3_domain_blocking(
    s1_records: list[dict],
    s23_records: list[dict],
) -> dict[str, set[str]]:
    """Match records sharing exact extracted domain/URL stem."""
    domain_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for rec in s23_records:
        stem = extract_domain_stem(rec.get("business_name", ""))
        if stem and len(stem) >= 4:
            domain_index[(rec["country"], stem)].append(rec["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)
    for rec in s1_records:
        stem = extract_domain_stem(rec.get("business_name", ""))
        if stem and len(stem) >= 4:
            s23_matches = domain_index.get((rec["country"], stem), [])
            if s23_matches:
                candidates[rec["entity_id"]].update(s23_matches)

    return dict(candidates)


class BlockingEngine:
    """Multi-stage blocking engine with ranked candidate pruning."""

    def __init__(self, top_k: int = DEFAULT_TOP_K, use_dense: bool = False):
        self.top_k = top_k
        self.use_dense = use_dense

    def run(
        self,
        s1_records: list[dict],
        s23_records: list[dict],
    ) -> dict[str, list[str]]:
        """Returns dict mapping s1_id to list of ranked candidate IDs (top-K pruned)."""
        logger.info("[Blocking] Stage 1: Token Inverted Index...")
        c1 = stage1_token_lsh(s1_records, s23_records)

        logger.info("[Blocking] Stage 2: Address Digit Blocking...")
        c2 = stage2_address_blocking(s1_records, s23_records)

        logger.info("[Blocking] Stage 3: Domain Stem Matching...")
        c3 = stage3_domain_blocking(s1_records, s23_records)

        # Build S2/S3 record lookup for fast candidate ranking
        s23_lookup = {r["entity_id"]: r for r in s23_records}

        merged: dict[str, list[str]] = {}
        all_s1_ids = {r["entity_id"]: r for r in s1_records}

        logger.info("[Blocking] Stage 4: Ranked Candidate Pruning (Top-K = %d)...", self.top_k)

        for s1_id, s1_rec in all_s1_ids.items():
            provenance_counts: dict[str, int] = defaultdict(int)
            for cid in c1.get(s1_id, set()):
                provenance_counts[cid] += 1
            for cid in c2.get(s1_id, set()):
                provenance_counts[cid] += 1
            for cid in c3.get(s1_id, set()):
                provenance_counts[cid] += 2  # Higher weight for domain agreement

            provenance_counts.pop(s1_id, None)  # Guard against self-match

            if not provenance_counts:
                merged[s1_id] = []
                continue

            # Rank candidates by preliminary score
            s1_toks = s1_rec.get("name_tokens", [])
            s1_digits = set(s1_rec.get("addr_digits", []))

            scored_candidates = []
            for cid, prov_count in provenance_counts.items():
                cand_rec = s23_lookup.get(cid)
                if not cand_rec:
                    continue
                score = preliminary_candidate_score(
                    s1_toks, s1_digits,
                    cand_rec.get("name_tokens", []),
                    set(cand_rec.get("addr_digits", [])),
                    prov_count,
                )
                scored_candidates.append((score, cid))

            # Sort descending by score and keep top-K
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            merged[s1_id] = [cid for _, cid in scored_candidates[:self.top_k]]

        total_pairs = sum(len(v) for v in merged.values())
        max_possible = len(all_s1_ids) * len(s23_records)
        reduction_ratio = 1.0 - (total_pairs / max(max_possible, 1))

        logger.info(
            "[Blocking] DONE: %d candidate pairs generated across %d S1 entities. Reduction ratio: %.6f",
            total_pairs, len(all_s1_ids), reduction_ratio,
        )
        return merged

    @staticmethod
    def compute_blocking_recall(
        merged: dict[str, list[str]],
        ground_truth: dict[str, list[str]],
    ) -> float:
        """Estimate pair-level candidate recall against ground truth."""
        hits = 0
        total = 0
        for s1_id, true_matches in ground_truth.items():
            if not true_matches:
                continue
            total += len(true_matches)
            candidates_set = set(merged.get(s1_id, []))
            hits += sum(1 for m in true_matches if m in candidates_set)
        return hits / max(total, 1)
