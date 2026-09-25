"""scripts/quality_gate.py - Automated Champion-Challenger Quality Gate.

Enforces Rule 1 & Rule 17 of the Amazon ML Challenge 2026 No-Regression Protocol:
- Loads reports/champion.json.
- Compares incoming experiment metrics against the current champion.
- Rejects regressions: blocks deployment if candidate_f05 < champion_f05 or singleton_accuracy < 0.90.
- Saves reports/rejected_experiment_<run_id>.json with exact audit failure reasoning.
- Promotes challenger to reports/champion.json ONLY if improvement >= +0.002 Macro F0.5.

Usage:
    python scripts/quality_gate.py [--check-current | --experiment-file reports/current_run.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
PUBLIC_REPORTS_DIR = ROOT / "business_entity_resolution" / "frontend" / "public" / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QualityGate")

CHAMPION_PATH = REPORTS_DIR / "champion.json"
MIN_PROMOTION_DELTA = 0.0020
MIN_SINGLETON_ACCURACY = 0.9000


def load_champion() -> dict:
    if not CHAMPION_PATH.exists():
        logger.error("Champion file not found at %s!", CHAMPION_PATH)
        sys.exit(1)
    with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_experiment(exp_data: dict) -> bool:
    """Evaluate experiment against current champion. Returns True if promoted, False if rejected."""
    champion = load_champion()
    champ_f05 = champion.get("macro_f05", 0.7930)
    champ_singletons = champion.get("singleton_accuracy", 0.9417)
    run_id = exp_data.get("run_id", f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    cand_f05 = exp_data.get("macro_f05", 0.0)
    cand_singletons = exp_data.get("singleton_accuracy", 0.0)
    cand_prec = exp_data.get("precision", 0.0)
    cand_rec = exp_data.get("recall", 0.0)

    logger.info("=" * 60)
    logger.info("  RESOLVE.AI QUALITY GATE: CHAMPION vs CHALLENGER EVALUATION  ")
    logger.info("=" * 60)
    logger.info("Current Champion:")
    logger.info("  - Model:              %s", champion.get("model", "Champion"))
    logger.info("  - Macro F0.5:         %.4f", champ_f05)
    logger.info("  - Singleton Accuracy: %.4f (%.2f%%)", champ_singletons, champ_singletons * 100)
    logger.info("Incoming Challenger (%s):", run_id)
    logger.info("  - Model:              %s", exp_data.get("model", "Challenger"))
    logger.info("  - Macro F0.5:         %.4f", cand_f05)
    logger.info("  - Precision:          %.4f (%.2f%%)", cand_prec, cand_prec * 100)
    logger.info("  - Recall:             %.4f (%.2f%%)", cand_rec, cand_rec * 100)
    logger.info("  - Singleton Accuracy: %.4f (%.2f%%)", cand_singletons, cand_singletons * 100)

    # Invariant 1: Macro F0.5 must exceed champion
    delta_f05 = cand_f05 - champ_f05
    if delta_f05 < MIN_PROMOTION_DELTA:
        reason = f"Macro F0.5 ({cand_f05:.4f}) failed to beat Champion ({champ_f05:.4f}) by minimum delta (+{MIN_PROMOTION_DELTA:.4f}). Delta was {delta_f05:+.4f}."
        logger.warning("\n[BLOCKED / REJECTED] %s", reason)
        logger.warning("Production deployment prevented. Champion remains active.")
        save_rejection(run_id, exp_data, champion, reason)
        return False

    # Invariant 2: Singleton accuracy must not collapse
    if cand_singletons < MIN_SINGLETON_ACCURACY:
        reason = f"Singleton accuracy ({cand_singletons:.4f}) collapsed below required floor ({MIN_SINGLETON_ACCURACY:.4f}). Excessive false merges."
        logger.warning("\n[BLOCKED / REJECTED] %s", reason)
        logger.warning("Production deployment prevented. Champion remains active.")
        save_rejection(run_id, exp_data, champion, reason)
        return False

    # Invariant 3: Official validator check
    if exp_data.get("validator_passed") is False:
        reason = "Official submission validator failed on test candidate pairs / matching results."
        logger.warning("\n[BLOCKED / REJECTED] %s", reason)
        save_rejection(run_id, exp_data, champion, reason)
        return False

    # PROMOTION GRANTED
    logger.info("\n[PROMOTION GRANTED] Challenger (%.4f) legitimately beats Champion (%.4f) by %+.4f!",
                cand_f05, champ_f05, delta_f05)
    promote_challenger(exp_data)
    return True


def save_rejection(run_id: str, exp_data: dict, champion: dict, reason: str):
    rejection_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "status": "REJECTED_BY_QUALITY_GATE",
        "rejection_reason": reason,
        "challenger_metrics": exp_data,
        "active_champion_metrics": champion
    }
    rejection_file = REPORTS_DIR / f"rejected_experiment_{run_id}.json"
    with open(rejection_file, "w", encoding="utf-8") as f:
        json.dump(rejection_record, f, indent=2)
    logger.info("Saved rejection record to %s", rejection_file)


def promote_challenger(new_champ: dict):
    new_champ["status"] = "CHAMPION_ACTIVE"
    new_champ["promoted_at"] = datetime.now().isoformat()
    with open(CHAMPION_PATH, "w", encoding="utf-8") as f:
        json.dump(new_champ, f, indent=2)
    with open(PUBLIC_REPORTS_DIR / "champion.json", "w", encoding="utf-8") as f:
        json.dump(new_champ, f, indent=2)
    logger.info("Successfully updated %s as new production champion!", CHAMPION_PATH)


def main():
    parser = argparse.ArgumentParser(description="Automated No-Regression Quality Gate")
    parser.add_argument("--check-current", action="store_true", help="Check current_run.json against champion.json")
    parser.add_argument("--experiment-file", default=None, help="Path to experiment JSON file to evaluate")
    args = parser.parse_args()

    target_file = None
    if args.experiment_file:
        target_file = Path(args.experiment_file)
    elif args.check_current:
        target_file = REPORTS_DIR / "current_run.json"
    else:
        logger.info("Verifying champion integrity...")
        champ = load_champion()
        logger.info("[OK] Champion is valid: %s (Macro F0.5 = %.4f)", champ["model"], champ["macro_f05"])
        sys.exit(0)

    if not target_file.exists():
        logger.error("Experiment file not found: %s", target_file)
        sys.exit(1)

    with open(target_file, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    promoted = evaluate_experiment(exp_data)
    sys.exit(0 if promoted else 1)


if __name__ == "__main__":
    main()
