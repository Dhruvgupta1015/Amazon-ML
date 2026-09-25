"""model.py - LightGBM Match Classifier with Full-Population Singleton-Aware Threshold Tuning.

Key Features:
- Hard negative mining via inverted index and near-name candidates
- Full-population entity-level validation including zero-candidate singletons
- Calibrated street-number conflict penalty to prevent catastrophic false merges
- High-precision Macro F0.5 optimization matching official Amazon ML Challenge semantics
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed; using fallback classifier")

from .feature_extractor import FEATURE_NAMES, FeatureExtractor
from .metrics import entity_f05


def build_training_data(
    s1_records: list[dict],
    s23_records: list[dict],
    blocking_candidates: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    neg_per_pos: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct pairwise feature matrix X, label vector y, and group indices with hard negative mining."""
    s23_by_id = {r["entity_id"]: r for r in s23_records}

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    group_rows: list[int] = []
    group_idx = 0

    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        true_matches = set(ground_truth.get(s1_id, []))
        candidates = blocking_candidates.get(s1_id, [])

        positives = [m for m in true_matches if m in s23_by_id]
        hard_negs = [c for c in candidates if c not in true_matches and c in s23_by_id]

        # Balance negative ratio (e.g. 4 negatives per positive)
        n_neg_target = max(len(positives) * neg_per_pos, 2)
        hard_negs = hard_negs[:n_neg_target]

        pairs = [(s1_rec, s23_by_id[m], 1) for m in positives]
        pairs += [(s1_rec, s23_by_id[c], 0) for c in hard_negs]

        for s1_r, cand_r, label in pairs:
            feats = FeatureExtractor.extract(s1_r, cand_r)
            X_rows.append(feats)
            y_rows.append(float(label))
            group_rows.append(group_idx)

        if pairs:
            group_idx += 1

    if not X_rows:
        return np.zeros((0, 28), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        np.array(group_rows, dtype=int),
    )


class MatchClassifier:
    """LightGBM binary classifier with singleton-aware threshold optimization."""

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.optimal_threshold: float = 0.56
        self.feature_names = FEATURE_NAMES

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        val_s1_records: Optional[list[dict]] = None,
        val_s23_records: Optional[list[dict]] = None,
        val_candidates: Optional[dict[str, list[str]]] = None,
        val_ground_truth: Optional[dict[str, list[str]]] = None,
    ) -> dict:
        """Train LightGBM model and calibrate optimal threshold over complete validation population."""
        if not HAS_LGB:
            return self._train_fallback(X, y)

        logger.info("Fitting LightGBM on %d pairwise examples (Positives: %d, Negatives: %d)...",
                    len(X), int(np.sum(y == 1)), int(np.sum(y == 0)))

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)

        if val_s1_records and val_s23_records and val_candidates and val_ground_truth:
            logger.info("Conducting full-population singleton-aware threshold sweep...")
            cal_results = self.search_threshold_full_population(
                val_s1_records, val_s23_records, val_candidates, val_ground_truth
            )
            return cal_results

        return {"optimal_threshold": self.optimal_threshold, "best_f05": 0.0}

    def _train_fallback(self, X: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(random_state=42)
        self.model.fit(X, y)
        return {"optimal_threshold": 0.56, "best_f05": 0.0}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probabilities with street-number conflict penalty."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        probs = self.model.predict_proba(X)[:, 1]
        
        # Apply calibrated street number conflict penalty (feature index 27: street_number_conflict)
        if X.shape[1] > 27:
            conflict_mask = X[:, 27] > 0.5
            probs[conflict_mask] = np.maximum(0.0, probs[conflict_mask] - 0.35)
            
        return probs

    def search_threshold_full_population(
        self,
        val_s1_records: list[dict],
        val_s23_records: list[dict],
        val_candidates: dict[str, list[str]],
        val_ground_truth: dict[str, list[str]],
    ) -> dict:
        """Evaluate Macro F0.5 across ALL validation S1 entities (including 0-candidate singletons)."""
        s23_by_id = {r["entity_id"]: r for r in val_s23_records}
        all_s1_ids = [r["entity_id"] for r in val_s1_records]

        # Extract features and compute probabilities for all candidate pairs
        logger.info("Scoring validation pairs for threshold calibration...")
        candidate_scores_by_s1: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for s1_rec in val_s1_records:
            s1_id = s1_rec["entity_id"]
            cands = val_candidates.get(s1_id, [])
            valid_cands = [c for c in cands if c in s23_by_id]
            if not valid_cands:
                continue

            pairs = [(s1_rec, s23_by_id[c]) for c in valid_cands]
            X_batch = FeatureExtractor.extract_batch(pairs)
            probs = self.predict_proba(X_batch)

            for cid, prob in zip(valid_cands, probs):
                candidate_scores_by_s1[s1_id].append((cid, float(prob)))

        thresholds = [round(t, 2) for t in np.arange(0.20, 0.98, 0.02)]
        best_f05, best_tau = 0.0, 0.56
        curve = []

        for tau in thresholds:
            f05_scores = []
            for s1_id in all_s1_ids:
                true_set = set(val_ground_truth.get(s1_id, []))
                c_scores = candidate_scores_by_s1.get(s1_id, [])
                pred_set = {cid for cid, score in c_scores if score >= tau}
                f05_scores.append(entity_f05(pred_set, true_set))

            macro_f05 = float(np.mean(f05_scores))
            curve.append({"threshold": tau, "macro_f05": macro_f05})

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_tau = tau

        self.optimal_threshold = best_tau
        logger.info("Optimal Singleton-Aware Threshold: tau*=%.2f (Macro F0.5=%.4f)", best_tau, best_f05)
        return {
            "optimal_threshold": best_tau,
            "best_f05": best_f05,
            "threshold_curve": curve,
        }

    def predict_entities(
        self,
        s1_records: list[dict],
        s23_records: list[dict],
        blocking_candidates: dict[str, list[str]],
        threshold: Optional[float] = None,
    ) -> dict[str, list[str]]:
        """Run batch inference for all S1 entities with singleton and threshold guards."""
        tau = threshold if threshold is not None else self.optimal_threshold
        s23_by_id = {r["entity_id"]: r for r in s23_records}
        results: dict[str, list[str]] = {}

        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            cands = blocking_candidates.get(s1_id, [])
            valid_cands = [c for c in cands if c in s23_by_id]

            if not valid_cands:
                results[s1_id] = []
                continue

            pairs = [(s1_rec, s23_by_id[c]) for c in valid_cands]
            X_batch = FeatureExtractor.extract_batch(pairs)
            probs = self.predict_proba(X_batch)

            matched = [valid_cands[i] for i, p in enumerate(probs) if p >= tau]
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for cid in matched:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            results[s1_id] = deduped

        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.optimal_threshold}, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.optimal_threshold = data["threshold"]
        logger.info("Model loaded from %s (threshold=%.2f)", path, self.optimal_threshold)
