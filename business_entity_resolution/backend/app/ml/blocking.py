"""blocking.py - High-Recall Multi-Stage Blocking Engine.

Stages:
  1. Token Inverted Index + MinHash LSH (char 3-grams and word tokens)
  2. Address Number and Locality Blocking (shared street number / postal code)
  3. Dense Vector Retrieval via sentence-transformers (ANN cosine search)

Hard Rule: Only candidates within the same country are paired.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)

MAX_CANDIDATES_PER_S1 = 50
LSH_THRESHOLD = 0.25
LSH_PERMUTATIONS = 128
ANN_TOP_K = 30
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _make_minhash(tokens: list[str]):
    from datasketch import MinHash
    m = MinHash(num_perm=LSH_PERMUTATIONS)
    for tok in tokens:
        m.update(tok.encode("utf-8"))
    return m


def _minhash_tokens(record: dict) -> list[str]:
    return record.get("name_ngrams", []) + record.get("name_tokens", [])


def stage1_token_lsh(
    s1_records: list[dict],
    s23_records: list[dict],
) -> dict[str, set[str]]:
    """MinHash LSH blocking on country-partitioned name tokens and ngrams."""
    from datasketch import MinHashLSH

    s1_by_country: dict[str, list[dict]] = defaultdict(list)
    s23_by_country: dict[str, list[dict]] = defaultdict(list)

    for rec in s1_records:
        s1_by_country[rec["country"]].append(rec)
    for rec in s23_records:
        s23_by_country[rec["country"]].append(rec)

    candidates: dict[str, set[str]] = defaultdict(set)

    for country in s1_by_country:
        s1_group = s1_by_country[country]
        s23_group = s23_by_country.get(country, [])
        if not s23_group:
            continue

        logger.info("[LSH] Country=%r: %d S1, %d S2/S3", country, len(s1_group), len(s23_group))

        lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_PERMUTATIONS)

        for rec in s23_group:
            eid = rec["entity_id"]
            toks = _minhash_tokens(rec)
            if not toks:
                toks = list(Preprocessor.name_tokens(eid))
            mh = _make_minhash(toks)
            try:
                lsh.insert(eid, mh)
            except ValueError:
                pass

        for rec in s1_group:
            s1_id = rec["entity_id"]
            toks = _minhash_tokens(rec)
            if not toks:
                continue
            mh = _make_minhash(toks)
            results = lsh.query(mh)
            candidates[s1_id].update(results)

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
            for eid in s23_matches:
                candidates[s1_id].add(eid)

    return dict(candidates)


class DenseBlocker:
    """Wraps sentence-transformers for GPU/CPU batched ANN retrieval."""

    def __init__(self, model_name: str = DENSE_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("[Dense] Loaded model: %s", self.model_name)
            except ImportError:
                logger.error("sentence-transformers not installed; skipping dense blocking")
                self._model = False
        return self._model

    def _encode_batch(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        model = self._load_model()
        if not model:
            return np.zeros((len(texts), 384), dtype=np.float32)
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @staticmethod
    def _record_text(rec: dict) -> str:
        name = rec.get("cleaned_name", "") or rec.get("business_name", "")
        addr = rec.get("cleaned_address", "") or rec.get("business_address", "")
        return f"{name} {addr}".strip()[:512]

    def run(
        self,
        s1_records: list[dict],
        s23_records: list[dict],
        top_k: int = ANN_TOP_K,
    ) -> dict[str, set[str]]:
        """Encode all records and retrieve top-K nearest neighbours per S1 entity."""
        s1_by_country: dict[str, list[dict]] = defaultdict(list)
        s23_by_country: dict[str, list[dict]] = defaultdict(list)
        for rec in s1_records:
            s1_by_country[rec["country"]].append(rec)
        for rec in s23_records:
            s23_by_country[rec["country"]].append(rec)

        candidates: dict[str, set[str]] = defaultdict(set)

        for country in s1_by_country:
            s1_group = s1_by_country[country]
            s23_group = s23_by_country.get(country, [])
            if not s23_group:
                continue

            logger.info("[Dense] Country=%r: encoding %d S2/S3 records", country, len(s23_group))

            s23_texts = [self._record_text(r) for r in s23_group]
            s23_ids = [r["entity_id"] for r in s23_group]
            s23_embs = self._encode_batch(s23_texts)

            s1_texts = [self._record_text(r) for r in s1_group]
            s1_ids_loc = [r["entity_id"] for r in s1_group]
            s1_embs = self._encode_batch(s1_texts)

            batch_size = 512
            for start in range(0, len(s1_embs), batch_size):
                s1_batch = s1_embs[start:start + batch_size]
                sims = s1_batch @ s23_embs.T
                k_eff = min(top_k, sims.shape[1])
                top_indices = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
                for i, s1_id in enumerate(s1_ids_loc[start:start + batch_size]):
                    for j in top_indices[i]:
                        candidates[s1_id].add(s23_ids[j])

        return dict(candidates)


class BlockingEngine:
    """Runs all three blocking stages and consolidates candidate pairs."""

    def __init__(self, use_dense: bool = True):
        self.use_dense = use_dense
        self._dense_blocker = DenseBlocker() if use_dense else None

    def run(
        self,
        s1_records: list[dict],
        s23_records: list[dict],
    ) -> dict[str, list[str]]:
        """Returns dict mapping s1_id to list of candidate IDs (capped at MAX_CANDIDATES_PER_S1)."""
        logger.info("[Blocking] Stage 1: MinHash LSH...")
        c1 = stage1_token_lsh(s1_records, s23_records)
        logger.info("[Blocking] Stage 1 done: %d pairs", sum(len(v) for v in c1.values()))

        logger.info("[Blocking] Stage 2: Address digit blocking...")
        c2 = stage2_address_blocking(s1_records, s23_records)
        logger.info("[Blocking] Stage 2 done: %d pairs", sum(len(v) for v in c2.values()))

        c3: dict[str, set[str]] = {}
        if self.use_dense and self._dense_blocker:
            logger.info("[Blocking] Stage 3: Dense ANN retrieval...")
            c3 = self._dense_blocker.run(s1_records, s23_records)
            logger.info("[Blocking] Stage 3 done: %d pairs", sum(len(v) for v in c3.values()))

        all_s1_ids = {r["entity_id"] for r in s1_records}
        merged: dict[str, list[str]] = {}

        for s1_id in all_s1_ids:
            union: set[str] = set()
            union.update(c1.get(s1_id, set()))
            union.update(c2.get(s1_id, set()))
            union.update(c3.get(s1_id, set()))
            union.discard(s1_id)
            merged[s1_id] = list(union)[:MAX_CANDIDATES_PER_S1]

        total_pairs = sum(len(v) for v in merged.values())
        max_possible = len(all_s1_ids) * len(s23_records)
        reduction_ratio = 1.0 - (total_pairs / max_possible) if max_possible > 0 else 1.0

        logger.info(
            "[Blocking] DONE: %d pairs, reduction_ratio=%.4f",
            total_pairs, reduction_ratio,
        )
        return merged

    @staticmethod
    def compute_blocking_recall(
        merged: dict[str, list[str]],
        ground_truth: dict[str, list[str]],
    ) -> float:
        """Estimate blocking recall on training data with ground truth."""
        hits = 0
        total = 0
        for s1_id, true_matches in ground_truth.items():
            if not true_matches:
                continue
            total += len(true_matches)
            candidates_set = set(merged.get(s1_id, []))
            hits += sum(1 for m in true_matches if m in candidates_set)
        return hits / max(total, 1)
