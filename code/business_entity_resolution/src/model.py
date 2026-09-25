"""model.py - LightGBM Match Classifier with F0.5-Optimal Threshold Tuning."""
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
    logger.warning("LightGBM not installed; model will use fallback logistic regression")

from .feature_extractor import FEATURE_NAMES, FeatureExtractor
from .metrics import entity_f05


def build_training_data(
    s1_records: list[dict],
    s23_records: list[dict],
    blocking_candidates: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    embeddings: Optional[dict] = None,
    neg_per_pos: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pairwise feature matrix X, label vector y, group array groups."""
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

        n_neg_target = max(len(positives) * neg_per_pos, 1)
        hard_negs = hard_negs[:n_neg_target]

        pairs = [(s1_rec, s23_by_id[m], 1) for m in positives if m in s23_by_id]
        pairs += [(s1_rec, s23_by_id[c], 0) for c in hard_negs]

        for s1_r, cand_r, label in pairs:
            emb = embeddings or {}
            s1_e = emb.get(s1_r["entity_id"], {})
            cand_e = emb.get(cand_r["entity_id"], {})
            feats = FeatureExtractor.extract(
                s1_r, cand_r,
                s1_name_emb=s1_e.get("name"),
                s1_addr_emb=s1_e.get("addr"),
                cand_name_emb=cand_e.get("name"),
                cand_addr_emb=cand_e.get("addr"),
            )
            X_rows.append(feats)
            y_rows.append(float(label))
            group_rows.append(group_idx)

        group_idx += 1

    if not X_rows:
        return np.zeros((0, 28), dtype=np.float32), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        np.array(group_rows, dtype=int),
    )


class MatchClassifier:
    """LightGBM binary classifier for entity match prediction."""

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 3.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.optimal_threshold: float = 0.5
        self.feature_names = FEATURE_NAMES

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
        """Train with GroupKFold CV and log F0.5 calibration results."""
        if not HAS_LGB:
            return self._train_fallback(X, y, groups)

        from sklearn.model_selection import GroupKFold

        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)

        oof_preds = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model_fold = lgb.LGBMClassifier(**self.params)
            model_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            oof_preds[val_idx] = model_fold.predict_proba(X_val)[:, 1]
            logger.info("Fold %d/%d done", fold + 1, n_splits)

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        logger.info("Final model trained on full data")

        cal_results = self._search_threshold(oof_preds, y, groups)
        return cal_results

    def _search_threshold(
        self,
        oof_preds: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> dict:
        """Search threshold in [0.50, 0.98] for max macro-F0.5."""
        thresholds = np.arange(0.50, 0.99, 0.01)
        results = []

        group_ids = np.unique(groups)
        group_true: dict[int, set[int]] = {}
        group_probs: dict[int, list[tuple[float, int]]] = {}

        for g in group_ids:
            mask = groups == g
            group_true[g] = set(np.where(mask & (y == 1))[0])
            group_probs[g] = list(zip(oof_preds[mask], np.where(mask)[0]))

        best_f05, best_tau = 0.0, 0.5

        for tau in thresholds:
            f05_scores = []
            for g in group_ids:
                true_set = group_true[g]
                predicted_set = {idx for prob, idx in group_probs[g] if prob >= tau}
                f05 = entity_f05(predicted_set, true_set)
                f05_scores.append(f05)
            macro_score = float(np.mean(f05_scores))
            results.append({"threshold": float(tau), "f05": macro_score})
            if macro_score > best_f05:
                best_f05 = macro_score
                best_tau = float(tau)

        self.optimal_threshold = best_tau
        logger.info("Optimal threshold: %.2f (F0.5=%.4f)", best_tau, best_f05)
        return {
            "threshold_curve": results,
            "optimal_threshold": best_tau,
            "best_f05": best_f05,
        }

    def _train_fallback(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
        """Fallback logistic regression if LightGBM not available."""
        from sklearn.linear_model import LogisticRegression
        logger.warning("Using fallback LogisticRegression (install lightgbm for best results)")
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.model.fit(X, y)
        return {"optimal_threshold": 0.5, "best_f05": 0.0, "threshold_curve": []}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probabilities (N,)."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]

    def predict_entities(
        self,
        s1_records: list[dict],
        s23_records: list[dict],
        blocking_candidates: dict[str, list[str]],
        embeddings: Optional[dict] = None,
        threshold: Optional[float] = None,
    ) -> dict[str, list[str]]:
        """Inference: score all candidates and apply singleton guard."""
        tau = threshold if threshold is not None else self.optimal_threshold
        s23_by_id = {r["entity_id"]: r for r in s23_records}
        results: dict[str, list[str]] = {}

        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            cands = blocking_candidates.get(s1_id, [])
            valid = [c for c in cands if c in s23_by_id]

            if not valid:
                results[s1_id] = []
                continue

            pairs = [(s1_rec, s23_by_id[c]) for c in valid]
            X_pair = FeatureExtractor.extract_batch(pairs, embeddings)
            probs = self.predict_proba(X_pair)

            if probs.max() < tau:
                results[s1_id] = []
            else:
                matched = [valid[i] for i, p in enumerate(probs) if p >= tau]
                results[s1_id] = matched

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
