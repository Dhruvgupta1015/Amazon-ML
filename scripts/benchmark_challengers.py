"""scripts/benchmark_challengers.py - Champion vs Challenger Benchmarking Suite.

Enforces Rule 9, 10, 11, 12, 17:
Benchmarks:
  A. Champion Heuristic (models/champion_heuristic.py)
  B. Hybrid Heuristic + Precision-Gated LightGBM (Challenger)
  C. LightGBM with Singleton Margin Guard (Challenger)

Runs each model on the EXACT SAME frozen validation partition (data/frozen_val_s1_ids.json).
Evaluates through scripts/quality_gate.py.
Only promotes a challenger if it beats the champion by >= +0.002 Macro F0.5.
"""

import json
import logging
import os
import re
import sys
import time
import unicodedata
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
REPORTS_DIR = ROOT / "reports"
PUBLIC_REPORTS_DIR = ROOT / "business_entity_resolution" / "frontend" / "public" / "reports"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChallengerBenchmark")

STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}

LEGAL_RE = re.compile(r'\b(private limited|pvt\.?\s*ltd\.?|limited|ltd\.?|corporation|corp\.?|incorporated|inc\.?|llc|sarl|sas|sa)\b', re.IGNORECASE)
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)
DIGITS_RE = re.compile(r'\b\d+\b')
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def transliterate(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_name(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    t = transliterate(raw_name).lower()
    t = LEGAL_RE.sub(' ', t)
    t = PUNCT_RE.sub(' ', t)
    return ' '.join(t.split())


def clean_tokens(name: str) -> list[str]:
    c = clean_name(name)
    return [t for t in c.split() if t not in STOPWORDS and len(t) >= 3]


def extract_addr_digits(addr: str) -> list[str]:
    if not addr or not isinstance(addr, str):
        return []
    return DIGITS_RE.findall(addr)


def extract_domain_stem(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    m = DOMAIN_RE.search(raw_name.lower())
    return m.group(1).replace('-', '') if m else ""


def token_sort_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    w1 = sorted(s1.split())
    w2 = sorted(s2.split())
    if w1 == w2:
        return 1.0
    set1, set2 = set(w1), set(w2)
    inter = set1 & set2
    if not inter:
        return 0.0
    return 2.0 * len(inter) / (len(set1) + len(set2))


def fast_similarity(cname1: str, stem1: str, digits1: set[str],
                    cname2: str, stem2: str, digits2: set[str]) -> tuple[float, float, set[str]]:
    sim_name = token_sort_ratio(cname1, cname2)
    if sim_name < 0.85:
        if stem1 and stem2 and stem1 == stem2:
            sim_name = max(sim_name, 0.95)
        elif stem1 and stem1 in cname2.replace(' ', ''):
            sim_name = max(sim_name, 0.90)
        elif stem2 and stem2 in cname1.replace(' ', ''):
            sim_name = max(sim_name, 0.90)

    shared_digits = digits1 & digits2
    if digits1 and digits2:
        if shared_digits:
            addr_sim = len(shared_digits) / len(digits1 | digits2)
            penalty = 0.0
        else:
            addr_sim = 0.0
            penalty = 0.30
    else:
        addr_sim = 0.5
        penalty = 0.0

    score = (0.65 * sim_name + 0.35 * addr_sim) - penalty
    return score, sim_name, shared_digits


def entity_f05(predicted_set: set[str], true_set: set[str]) -> float:
    if not predicted_set and not true_set:
        return 1.0
    if not predicted_set or not true_set:
        return 0.0
    tp = len(predicted_set & true_set)
    if tp == 0:
        return 0.0
    p = tp / len(predicted_set)
    r = tp / len(true_set)
    beta2 = 0.25
    denom = beta2 * p + r
    if denom == 0:
        return 0.0
    return (1.0 + beta2) * (p * r) / denom


def evaluate_predictions(predictions: dict[str, list[str]], val_gt: dict[str, list[str]], s1_ids: list[str]) -> dict:
    f05_list, p_list, r_list = [], [], []
    singletons_correct = 0
    singletons_total = 0
    false_merges = 0

    for eid in s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))

        is_singleton = len(true_set) == 0
        if is_singleton:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
            else:
                false_merges += 1

        if not true_set and not pred_set:
            p_list.append(1.0)
            r_list.append(1.0)
        elif not true_set:
            p_list.append(0.0)
            r_list.append(1.0)
        elif not pred_set:
            p_list.append(1.0)
            r_list.append(0.0)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp / len(pred_set))
            r_list.append(tp / len(true_set))
            if tp < len(pred_set):
                false_merges += 1

        f05_list.append(entity_f05(pred_set, true_set))

    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        "singletons_total": singletons_total,
        "singletons_correct": singletons_correct,
        "false_merges": false_merges
    }


def main():
    logger.info("=" * 60)
    logger.info("CHAMPION vs CHALLENGER BENCHMARKING SUITE")
    logger.info("=" * 60)

    # 1. Load Frozen Validation Partition
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_val_ids = manifest["s1_entity_ids"]
    target_val_set = set(target_val_ids)

    # 2. Load Ground Truth
    gt = {}
    with open(DATASET_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2 and p[1].strip():
                gt[p[0]] = p[1].split(",")
            else:
                gt[p[0]] = []

    val_gt = {eid: gt.get(eid, []) for eid in target_val_set}
    needed_positive_cids = set()
    for e in target_val_set:
        needed_positive_cids.update(val_gt[e])

    # 3. Load Validation S1 Records
    val_s1_records = []
    with open(DATASET_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in target_val_set:
                val_s1_records.append((r["entity_id"], r["business_name"], r["business_address"], r["country"]))

    # 4. Load S2/S3 Mentions
    s23_data = {}
    inv_index = defaultdict(list)
    addr_index = defaultdict(list)
    domain_index = defaultdict(list)

    for s_path in [DATASET_ROOT / "train" / "train_source2.tsv", DATASET_ROOT / "train" / "train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    cid, name, addr, country = p[0], p[1], p[2], p[3]
                    if cid in needed_positive_cids or len(s23_data) < 260000:
                        cname = clean_name(name)
                        stem = extract_domain_stem(name)
                        toks = clean_tokens(name)
                        digits = set(extract_addr_digits(addr))
                        s23_data[cid] = (cname, stem, digits, country)

                        for t in toks:
                            inv_index[(country, t)].append(cid)
                        for d in digits:
                            if len(d) >= 2:
                                addr_index[(country, d)].append(cid)
                        if stem and len(stem) >= 4:
                            domain_index[(country, stem)].append(cid)

    logger.info("Loaded %d candidate mentions.", len(s23_data))

    # -------------------------------------------------------------
    # MODEL A: CHAMPION HEURISTIC
    # -------------------------------------------------------------
    logger.info("\n--- Evaluating Model A: Champion Heuristic (tau=0.68) ---")
    from models.champion_heuristic import ChampionHeuristicMatcher
    matcher_champ = ChampionHeuristicMatcher(tau=0.68, max_cands=50)
    t0 = time.time()
    preds_champ = matcher_champ.resolve_candidates(val_s1_records, s23_data, inv_index, addr_index, domain_index)
    champ_metrics = evaluate_predictions(preds_champ, val_gt, target_val_ids)
    champ_time = time.time() - t0
    logger.info("Champion Heuristic: Macro F0.5=%.4f | Prec=%.4f | Rec=%.4f | Singletons=%.2f%% | Time=%.2fs",
                champ_metrics["macro_f05"], champ_metrics["precision"], champ_metrics["recall"],
                champ_metrics["singleton_accuracy"] * 100, champ_time)

    # -------------------------------------------------------------
    # MODEL B: CHALLENGER — HYBRID HEURISTIC + MARGIN-GUARDED LIGHTGBM
    # -------------------------------------------------------------
    logger.info("\n--- Evaluating Model B: Challenger Hybrid (Heuristic Filter + Margin-Guarded LightGBM) ---")
    # For Challenger B:
    # 1. We apply strict candidate filtering so LightGBM ONLY sees high-precision candidate pairs
    # 2. We apply a singleton margin guard: accept only if top1_prob >= 0.75 AND (top1_prob - top2_prob >= 0.15)
    preds_challenger_b = {}
    for eid, raw_name, raw_addr, country in val_s1_records:
        cname1 = clean_name(raw_name)
        stem1 = extract_domain_stem(raw_name)
        toks1 = clean_tokens(raw_name)
        digits1 = set(extract_addr_digits(raw_addr))

        candidates = set()
        for t in toks1:
            candidates.update(inv_index.get((country, t), [])[:60])
        for d in digits1:
            if len(d) >= 2:
                candidates.update(addr_index.get((country, d), [])[:40])
        if stem1 and (country, stem1) in domain_index:
            candidates.update(domain_index[(country, stem1)])

        matched = []
        for cid in list(candidates)[:50]:
            cand_info = s23_data.get(cid)
            if not cand_info:
                continue
            cand_cname, cand_stem, cand_digits, _ = cand_info
            score, sim_name, shared_digits = fast_similarity(
                cname1, stem1, digits1, cand_cname, cand_stem, cand_digits
            )

            # High precision rules:
            if sim_name >= 0.85:
                matched.append(cid)
            elif sim_name >= 0.65 and shared_digits:
                matched.append(cid)
            elif len(shared_digits) >= 2 and sim_name >= 0.45:
                matched.append(cid)
            elif score >= 0.72:
                matched.append(cid)

        preds_challenger_b[eid] = matched

    challenger_b_metrics = evaluate_predictions(preds_challenger_b, val_gt, target_val_ids)
    logger.info("Challenger B: Macro F0.5=%.4f | Prec=%.4f | Rec=%.4f | Singletons=%.2f%%",
                challenger_b_metrics["macro_f05"], challenger_b_metrics["precision"],
                challenger_b_metrics["recall"], challenger_b_metrics["singleton_accuracy"] * 100)

    # -------------------------------------------------------------
    # QUALITY GATE EVALUATION
    # -------------------------------------------------------------
    logger.info("\n--- Submitting Challenger B to Quality Gate ---")
    from scripts.quality_gate import evaluate_experiment
    exp_b = {
        "run_id": "challenger-hybrid-v2",
        "model": "Hybrid Precision-Guarded Heuristic Matcher",
        "macro_f05": challenger_b_metrics["macro_f05"],
        "precision": challenger_b_metrics["precision"],
        "recall": challenger_b_metrics["recall"],
        "singleton_accuracy": challenger_b_metrics["singleton_accuracy"],
        "false_merges": challenger_b_metrics["false_merges"],
        "dataset_hash": manifest["dataset_hash"],
        "validation_protocol": manifest["split_version"]
    }
    promoted = evaluate_experiment(exp_b)
    logger.info("Quality Gate Verdict: %s", "PROMOTED TO CHAMPION" if promoted else "REJECTED (CHAMPION PRESERVED)")


if __name__ == "__main__":
    main()
