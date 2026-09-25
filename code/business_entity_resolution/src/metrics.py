"""metrics.py - F0.5 Metric Computation.

Macro-averaged F0.5 per Source 1 entity (singletons included).
"""
from __future__ import annotations
import numpy as np


def entity_f05(predicted: set, truth: set) -> float:
    """Compute F0.5 for a single Source 1 entity.

    Args:
        predicted: set of predicted matched entity IDs
        truth: set of true matched entity IDs (empty = singleton)

    Returns:
        float in [0.0, 1.0]
    """
    if not predicted and not truth:
        return 1.0
    if not truth:
        return 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    beta2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> float:
    """Compute macro-averaged F0.5 over all Source 1 entities."""
    scores = []
    for s1_id, pred_list in predictions.items():
        truth_list = ground_truth.get(s1_id, [])
        score = entity_f05(set(pred_list), set(truth_list))
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def precision_recall_f05(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> dict[str, float]:
    """Compute macro P, R, F0.5 and singleton accuracy."""
    precisions, recalls, f05s = [], [], []
    singleton_correct = 0
    singleton_total = 0

    for s1_id, pred_list in predictions.items():
        truth_list = ground_truth.get(s1_id, [])
        pred_set = set(pred_list)
        truth_set = set(truth_list)

        is_singleton_truth = len(truth_set) == 0
        is_singleton_pred = len(pred_set) == 0

        if is_singleton_truth:
            singleton_total += 1
            if is_singleton_pred:
                singleton_correct += 1

        if not pred_set and not truth_set:
            precisions.append(1.0)
            recalls.append(1.0)
        elif not truth_set:
            precisions.append(0.0)
            recalls.append(1.0)
        elif not pred_set:
            precisions.append(1.0)
            recalls.append(0.0)
        else:
            tp = len(pred_set & truth_set)
            precisions.append(tp / len(pred_set))
            recalls.append(tp / len(truth_set))

        f05s.append(entity_f05(pred_set, truth_set))

    return {
        "macro_f05": float(np.mean(f05s)) if f05s else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "singleton_accuracy": (
            singleton_correct / singleton_total if singleton_total > 0 else 1.0
        ),
        "n_entities": len(predictions),
        "n_singletons_truth": singleton_total,
    }


def threshold_curve(
    s1_ids: list[str],
    s1_probs_per_entity: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, list[str]],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep threshold and return metrics curve."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.01).tolist()

    results = []
    for tau in thresholds:
        preds = {}
        for s1_id in s1_ids:
            pairs = s1_probs_per_entity.get(s1_id, [])
            preds[s1_id] = [cid for cid, prob in pairs if prob >= tau]
        metrics = precision_recall_f05(preds, ground_truth)
        metrics["threshold"] = float(tau)
        results.append(metrics)
    return results
