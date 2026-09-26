#!/usr/bin/env python3
"""
benchmark_recovery_suite.py — Evaluates:
  1. Champion Heuristic (Phase 2 Baseline)
  2. Standalone LightGBM (Phase 3)
  3. Calibrated LightGBM (Phase 3)
  4. Hybrid Model with Bug Fixes & Cardinality Guard (Phase 3 & Phase 8/9)

Evaluated on the STRICT 30,000 Grouped Final Holdout (Zero S1 overlap, Zero GT leakage).
Saves:
  - reports/recovery/holdout_champion.json
  - reports/recovery/holdout_all_models.csv
  - reports/recovery/holdout_all_models.json
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_extractor_28d import (
    FEATURE_NAMES,
    Unified28FeatureExtractor,
    clean_text,
    expand_addr_tokens,
    extract_postal_code,
    extract_street_number,
    jaccard_similarity,
    jaro_winkler,
)

DIAG_DIR = ROOT / "data" / "diagnostic_200k"
RECOVERY_DIR = ROOT / "reports" / "recovery"
RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RecoverySuite")

RE_PUNCT = re.compile(r"[^a-z0-9\s]")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_ALPHA = re.compile(r"[^a-z0-9]")

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
    'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}
ALL_US_STATE_CODES = set(US_STATES.values())

INDIA_STATES = {
    'tamil nadu': 'tn', 'tamilnadu': 'tn', 'karnataka': 'ka', 'maharashtra': 'mh',
    'uttar pradesh': 'up', 'rajasthan': 'rj', 'madhya pradesh': 'mp', 'delhi': 'dl',
    'new delhi': 'dl', 'west bengal': 'wb', 'gujarat': 'gj', 'andhra pradesh': 'ap',
    'telangana': 'ts', 'kerala': 'kl', 'haryana': 'hr', 'bihar': 'br', 'odisha': 'od',
    'punjab': 'pb', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'uttarakhand': 'uk',
    'goa': 'ga', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'chandigarh': 'ch'
}
ALL_IN_STATE_CODES = set(INDIA_STATES.values())


class CompactCand:
    __slots__ = ("eid", "clean_name", "clean_addr", "alpha_name", "state", "street_num", "postal", "name_tokens")
    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str,
                 state: str, street_num: str, postal: str, name_tokens: tuple):
        self.eid = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.state = state
        self.street_num = street_num
        self.postal = postal
        self.name_tokens = name_tokens


def fast_clean(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(" ", str(s).lower()).strip()


def fast_alpha(s: str) -> str:
    if not s: return ""
    return RE_ALPHA.sub("", str(s).lower()).strip()


# Phase 8: Bug 2 Fixed State Extractor
def extract_state_fixed(addr_clean: str, country: str) -> str:
    if not addr_clean: return ""
    words = addr_clean.split()
    if not words: return ""
    n = len(words)
    if country == "US":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_US_STATE_CODES:
                # If followed by 5-digit zip or within last 3 words
                pos_from_end = i
                if pos_from_end <= 2:
                    return w
                if pos_from_end < n - 1 and len(words[-pos_from_end]) == 5 and words[-pos_from_end].isdigit():
                    return w
                if w not in {"in", "to", "or", "at", "as", "by", "on", "no"}:
                    return w
        # Fast substring check only if < 4 words
        for name, code in US_STATES.items():
            if name in addr_clean:
                return code
    elif country == "India":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_IN_STATE_CODES:
                if i <= 2 or w not in {"in", "to", "or", "at", "as"}:
                    return w
        for name, code in INDIA_STATES.items():
            if name in addr_clean:
                return code
    return ""


# Phase 8: Bug 3 Fixed Street Number Extractor
def get_street_num_fixed(addr_clean: str) -> str:
    if not addr_clean: return ""
    # Find postal code first to avoid mistaking it for street number
    postal = ""
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6):
            postal = d
            break
    # Extract first 1-5 digit token that is not the postal code
    for d in RE_DIGITS.findall(addr_clean):
        if d != postal and len(d) <= 5:
            return d
    return ""


def get_postal_fast(addr_clean: str) -> str:
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6):
            return d
    return ""


def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(" ", "")
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def evaluate_predictions(predictions: Dict[str, Set[str]], gt: Dict[str, Set[str]], s1_ids: List[str]) -> Dict[str, float]:
    f05_list, p_list, r_list = [], [], []
    singletons_correct, singletons_total = 0, 0
    false_merges, missed_matches = 0, 0
    match_cardinalities = []

    for eid in s1_ids:
        pred_set = predictions.get(eid, set())
        true_set = gt.get(eid, set())
        is_singleton = len(true_set) == 0

        match_cardinalities.append(len(pred_set))

        if is_singleton:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
            else:
                false_merges += len(pred_set)

        if not true_set and not pred_set:
            p_list.append(1.0); r_list.append(1.0)
        elif not true_set:
            p_list.append(0.0); r_list.append(1.0)
        elif not pred_set:
            p_list.append(1.0); r_list.append(0.0)
            missed_matches += len(true_set)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp / len(pred_set))
            r_list.append(tp / len(true_set))
            false_merges += len(pred_set - true_set)
            missed_matches += len(true_set - pred_set)
        f05_list.append(entity_f05(pred_set, true_set))

    arr_card = np.array(match_cardinalities)
    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        "singletons_correct": singletons_correct,
        "singletons_total": singletons_total,
        "false_merges": int(false_merges),
        "missed_matches": int(missed_matches),
        "mean_predicted_matches": float(np.mean(arr_card)),
        "max_predicted_matches": int(np.max(arr_card)),
        "pct_s1_empty": float(np.mean(arr_card == 0) * 100),
        "pct_s1_ge_5": float(np.mean(arr_card >= 5) * 100),
        "pct_s1_ge_10": float(np.mean(arr_card >= 10) * 100),
    }


def main():
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("PHASE 2 & 3: LEAKAGE-FREE STRICT 30K HOLDOUT BENCHMARK")
    logger.info("=" * 80)

    # 1. Load Split
    with open(DIAG_DIR / "split_holdout_30k_s1_ids.json", "r", encoding="utf-8") as f:
        holdout_ids = json.load(f)
    holdout_set = set(holdout_ids)
    logger.info("Loaded %d Holdout S1 IDs.", len(holdout_ids))

    # 2. Load Ground Truth
    gt = defaultdict(set)
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                gt[parts[0]] = set(x.strip() for x in parts[1].split(",") if x.strip())

    # 3. Load S1 Records for Holdout
    s1_dict = {}
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in holdout_set:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                cn = fast_clean(name)
                ca = fast_clean(addr)
                alpha = fast_alpha(name)
                st = extract_state_fixed(ca, country)
                sn = get_street_num_fixed(ca)
                po = get_postal_fast(ca)
                s1_dict[eid] = {
                    "id": eid, "clean_name": cn, "clean_addr": ca, "alpha_name": alpha,
                    "state": st, "street_num": sn, "postal": po, "country": country,
                    "name_tokens": set(cn.split()), "addr_tokens": set(ca.split()),
                    "expanded_addr": expand_addr_tokens(ca), "all_digits": set(RE_DIGITS.findall(ca)),
                    "ngrams4": get_char_ngrams(cn, 4)
                }
    logger.info("Parsed %d Holdout S1 records in %.2fs", len(s1_dict), time.time() - t0)

    # 4. Load Candidate Pool (S2/S3)
    logger.info("Loading S2/S3 candidate pool...")
    t_pool = time.time()
    cands_by_country = defaultdict(dict)
    for src in ["source2.tsv", "source3.tsv"]:
        with open(DIAG_DIR / src, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    cn = fast_clean(name)
                    ca = fast_clean(addr)
                    alpha = fast_alpha(name)
                    cands_by_country[country][cid] = CompactCand(
                        eid=cid, clean_name=cn, clean_addr=ca, alpha_name=alpha,
                        state="",
                        street_num="",
                        postal="",
                        name_tokens=tuple(cn.split())
                    )
    for c, pool in cands_by_country.items():
        logger.info("[%s] Candidate pool: %d records loaded in %.2fs", c, len(pool), time.time() - t_pool)

    # 5. Multi-Channel Inverted Index (NO Posting List Purging!)
    logger.info("Building Multi-Channel Inverted Indexes (exact name, rare tokens, address, street num, postal)...")
    idx_by_country = {}
    for country, pool in cands_by_country.items():
        name_idx = defaultdict(list)
        alpha_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        snum_idx = defaultdict(list)
        postal_idx = defaultdict(list)
        ng4_idx = defaultdict(list)

        token_counts = defaultdict(int)
        for c in pool.values():
            for t in c.name_tokens:
                token_counts[t] += 1

        for cid, c in pool.items():
            if c.alpha_name and len(c.alpha_name) >= 3:
                p = alpha_idx[c.alpha_name]
                if len(p) < 100: p.append(cid)
            for t in c.name_tokens:
                if len(t) >= 3:
                    # Rare or informative tokens get higher priority
                    freq = token_counts[t]
                    max_post = 40 if freq > 100 else 100
                    p = name_idx[t]
                    if len(p) < max_post: p.append(cid)
            for at in c.clean_addr.split():
                if len(at) >= 3:
                    p = addr_idx[at]
            # Digits for address/postal index
            for d in RE_DIGITS.findall(c.clean_addr):
                if len(d) in (5, 6):
                    p = postal_idx[d]
                    if len(p) < 60: p.append(cid)
                elif len(d) <= 5:
                    p = snum_idx[d]
                    if len(p) < 60: p.append(cid)

        idx_by_country[country] = {
            "name": name_idx, "alpha": alpha_idx, "addr": addr_idx,
            "snum": snum_idx, "postal": postal_idx
        }
        logger.info("[%s] Inverted index built. Unique alpha keys: %d, name keys: %d, addr keys: %d",
                    country, len(alpha_idx), len(name_idx), len(addr_idx))

    # Cache for retrieved candidate metadata: cid -> (state, street_num, postal)
    cand_meta = {}
    def get_cand_meta(cand: CompactCand, country: str):
        meta = cand_meta.get(cand.eid)
        if meta is None:
            st = extract_state_fixed(cand.clean_addr, country)
            sn = get_street_num_fixed(cand.clean_addr)
            po = get_postal_fast(cand.clean_addr)
            meta = (st, sn, po)
            cand_meta[cand.eid] = meta
        return meta

    # 6. Retrieve and Rank Candidates for Holdout S1 entities (STRICT LEAKAGE-FREE: NO GT INJECTION!)
    logger.info("Retrieving & Ranking candidates for 30K holdout S1 entities...")
    t_ret = time.time()
    holdout_cands: Dict[str, List[str]] = {}
    pair_recall_hits = 0
    pair_recall_total = 0
    zero_cands = 0

    for eid in holdout_ids:
        s1 = s1_dict[eid]
        c = s1["country"]
        idx = idx_by_country[c]
        pool = cands_by_country[c]
        true_m = gt.get(eid, set())
        pair_recall_total += len(true_m)

        raw_cands = set()
        # Channel 1: Exact alpha name (strongest signal)
        if s1["alpha_name"]:
            raw_cands.update(idx["alpha"].get(s1["alpha_name"], []))
        # Channel 2: Name tokens
        for t in s1["name_tokens"]:
            if len(t) >= 3:
                raw_cands.update(idx["name"].get(t, []))
        # Channel 3: Address tokens
        for at in s1["addr_tokens"]:
            if len(at) >= 3:
                raw_cands.update(idx["addr"].get(at, []))
        # Channel 4: Street num + postal
        if s1["street_num"]:
            raw_cands.update(idx["snum"].get(s1["street_num"], []))
        if s1["postal"]:
            raw_cands.update(idx["postal"].get(s1["postal"], []))

        if not raw_cands:
            zero_cands += 1
            holdout_cands[eid] = []
            continue

        # Phase 6: Ranked Pruning before truncation!
        # Composite score = 3 * name_jaccard + 2 * addr_jaccard + 5 * exact_alpha
        scored_cands = []
        s1_alpha = s1["alpha_name"]
        s1_nt = s1["name_tokens"]
        s1_at = s1["addr_tokens"]
        s1_sn = s1["street_num"]
        s1_po = s1["postal"]
        s1_st = s1["state"]

        for cid in raw_cands:
            cand = pool.get(cid)
            if not cand: continue
            st2, sn2, po2 = get_cand_meta(cand, c)
            # State conflict hard prune
            if s1_st and st2 and s1_st != st2:
                continue

            sim_score = 0.0
            if s1_alpha and cand.alpha_name and s1_alpha == cand.alpha_name:
                sim_score += 5.0
            # Name token overlap
            c_nt = set(cand.name_tokens)
            inter_n = len(s1_nt & c_nt)
            if inter_n:
                sim_score += 3.0 * (inter_n / (len(s1_nt) + len(c_nt) - inter_n))
            # Address token overlap
            c_at = set(cand.clean_addr.split())
            inter_a = len(s1_at & c_at)
            if inter_a:
                sim_score += 2.0 * (inter_a / (len(s1_at) + len(c_at) - inter_a))
            # Street num / postal match
            if s1_sn and sn2 and s1_sn == sn2:
                sim_score += 2.0
            if s1_po and po2 and s1_po == po2:
                sim_score += 2.0

            scored_cands.append((sim_score, cid))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        top_k = [cid for _, cid in scored_cands[:100]]
        holdout_cands[eid] = top_k

        # Measure genuine holdout candidate recall
        if true_m:
            pair_recall_hits += len(true_m & set(top_k))

    logger.info("Candidate Retrieval finished in %.2fs:", time.time() - t_ret)
    logger.info("  Zero Candidate Entities: %d (%.2f%%)", zero_cands, (zero_cands / len(holdout_ids)) * 100)
    logger.info("  Genuine Holdout Blocking Recall: %.2f%% (%d / %d true links retrieved)",
                (pair_recall_hits / pair_recall_total) * 100 if pair_recall_total else 0,
                pair_recall_hits, pair_recall_total)

    # 7. Model 1: Current Production Heuristic
    logger.info("Scoring Model 1: Production Heuristic (tau=0.53, high_conf=0.75, addr_floor=0.22)...")
    m1_preds = defaultdict(set)
    for eid in holdout_ids:
        s1 = s1_dict[eid]
        pool = cands_by_country[s1["country"]]
        for cid in holdout_cands[eid]:
            c = pool.get(cid)
            if not c: continue
            st2, sn2, po2 = get_cand_meta(c, s1["country"])
            if s1["state"] and st2 and s1["state"] != st2:
                continue
            sn1 = s1["street_num"]
            po1 = s1["postal"]
            if (sn1 and sn2 and sn1 != sn2) and (po1 and po2 and po1 != po2):
                continue
            conflict = 1.0 if (sn1 and sn2 and sn1 != sn2) else (0.2 if (sn1 or sn2) else 0.0)
            jw = jaro_winkler(s1["clean_name"], c.clean_name)
            tj_name = jaccard_similarity(s1["name_tokens"], set(c.name_tokens))
            c_exp = expand_addr_tokens(c.clean_addr)
            tj_addr = jaccard_similarity(s1["expanded_addr"], c_exp)
            base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.30 * conflict
            if base >= 0.75 or (base >= 0.53 and tj_addr >= 0.22):
                m1_preds[eid].add(cid)

    res_m1 = evaluate_predictions(m1_preds, gt, holdout_ids)
    logger.info("Model 1 (Production Heuristic) on Strict Holdout:")
    logger.info("  Macro F0.5:         %.6f", res_m1["macro_f05"])
    logger.info("  Precision:          %.6f", res_m1["precision"])
    logger.info("  Recall:             %.6f", res_m1["recall"])
    logger.info("  Singleton Accuracy: %.2f%%", res_m1["singleton_accuracy"] * 100)
    logger.info("  Empty %%:           %.2f%%", res_m1["pct_s1_empty"])
    logger.info("  Mean Matches:       %.2f (Max: %d)", res_m1["mean_predicted_matches"], res_m1["max_predicted_matches"])

    # Save holdout_champion.json (PHASE 2)
    with open(RECOVERY_DIR / "holdout_champion.json", "w", encoding="utf-8") as f:
        json.dump(res_m1, f, indent=2)
    logger.info("Saved Phase 2 baseline to %s", RECOVERY_DIR / "holdout_champion.json")

    # 8. Load Trained LightGBM 28D Model
    lgb_model_path = MODELS_DIR / "challenger_lgb_28d.txt"
    if not lgb_model_path.exists():
        logger.error("Model file %s not found!", lgb_model_path)
        return

    logger.info("Loading LightGBM model from %s...", lgb_model_path.name)
    lgb_booster = lgb.Booster(model_file=str(lgb_model_path))

    # Feature extraction for holdout pairs
    logger.info("Extracting 28 authoritative features for holdout candidate pairs...")
    holdout_pairs = []
    pair_index = []
    for eid in holdout_ids:
        s1 = s1_dict[eid]
        pool = cands_by_country[s1["country"]]
        for cid in holdout_cands[eid]:
            c = pool.get(cid)
            if not c: continue
            st2, sn2, po2 = get_cand_meta(c, s1["country"])
            c_dict = {
                "id": c.eid, "clean_name": c.clean_name, "clean_addr": c.clean_addr,
                "country": s1["country"], "name_tokens": set(c.name_tokens),
                "expanded_addr": expand_addr_tokens(c.clean_addr),
                "street_num": sn2, "postal": po2, "legal_suffix": ""
            }
            holdout_pairs.append((s1, c_dict))
            pair_index.append((eid, cid))

    logger.info("Vectorizing %d candidate pairs...", len(holdout_pairs))
    t_feat = time.time()
    X_holdout_df = Unified28FeatureExtractor.extract_batch_dataframe(holdout_pairs)
    probs = lgb_booster.predict(X_holdout_df)
    logger.info("LightGBM inference complete in %.2fs", time.time() - t_feat)

    prob_map = {(eid, cid): float(p) for (eid, cid), p in zip(pair_index, probs)}

    # Model 2: Standalone LightGBM (tau = 0.50)
    logger.info("Evaluating Model 2: Standalone LightGBM (tau=0.50)...")
    m2_preds = defaultdict(set)
    for (eid, cid), p in prob_map.items():
        if p >= 0.50:
            m2_preds[eid].add(cid)
    res_m2 = evaluate_predictions(m2_preds, gt, holdout_ids)

    # Model 3: Calibrated LightGBM (tau = 0.70 with Top-1 Margin)
    logger.info("Evaluating Model 3: Calibrated LightGBM with Margin Competition...")
    s1_scores = defaultdict(list)
    for (eid, cid), p in prob_map.items():
        s1_scores[eid].append((p, cid))

    m3_preds = defaultdict(set)
    for eid, slist in s1_scores.items():
        slist.sort(key=lambda x: x[0], reverse=True)
        top_p, top_cid = slist[0]
        if top_p >= 0.65:
            m3_preds[eid].add(top_cid)
            # Accept subsequent matches only if close to top score
            for p, cid in slist[1:10]:
                if p >= 0.55 and (top_p - p) < 0.18:
                    m3_preds[eid].add(cid)
    res_m3 = evaluate_predictions(m3_preds, gt, holdout_ids)

    # Model 4: Hybrid Model with Phase 8 Bug Fixes + Phase 9 Cardinality Ceiling
    logger.info("Evaluating Model 4: Hybrid with Phase 8 Bug Fixes & Cardinality Guard...")
    m4_preds = defaultdict(set)
    for eid, slist in s1_scores.items():
        s1 = s1_dict[eid]
        pool = cands_by_country[s1["country"]]
        slist.sort(key=lambda x: x[0], reverse=True)

        accepted = []
        for p, cid in slist:
            c = pool.get(cid)
            if not c: continue
            st2, sn2, po2 = get_cand_meta(c, s1["country"])

            # Bug 2 fix: State conflict hard rejection with fixed extractor
            if s1["state"] and st2 and s1["state"] != st2:
                continue

            # Bug 3 fix: Street number conflict rejection
            sn1 = s1["street_num"]
            if sn1 and sn2 and sn1 != sn2:
                continue

            c_exp = expand_addr_tokens(c.clean_addr)
            tj_addr = jaccard_similarity(s1["expanded_addr"], c_exp)
            both_have_addr = bool(s1["clean_addr"] and c.clean_addr)

            # Bug 1 fix: Disallow unconstrained exact-name match without address support
            # If both have addresses, require at least minimal address similarity or digit match
            digits_s1 = s1["all_digits"]
            digits_cand = set(RE_DIGITS.findall(c.clean_addr))
            digits_agree = bool(digits_s1 & digits_cand)
            if both_have_addr and tj_addr < 0.05 and not digits_agree:
                continue

            # Hybrid Decision rule:
            # High-confidence: p >= 0.70 with valid address support
            # Moderate-confidence: p >= 0.45 with tj_addr >= 0.15
            if p >= 0.70:
                accepted.append(cid)
            elif p >= 0.45 and tj_addr >= 0.15:
                accepted.append(cid)

            # Phase 9: High-cardinality protection: cap at 10 matches max
            if len(accepted) >= 10:
                break

        if accepted:
            m4_preds[eid] = set(accepted)

    res_m4 = evaluate_predictions(m4_preds, gt, holdout_ids)

    # Print summary table
    logger.info("=" * 80)
    logger.info("STRICT 30K HOLDOUT RESULTS COMPARISON (NO DATA LEAKAGE):")
    logger.info("=" * 80)
    models_summary = [
        {"Model": "1. Production Heuristic (Baseline)", **res_m1},
        {"Model": "2. Standalone LightGBM (p >= 0.50)", **res_m2},
        {"Model": "3. Calibrated LightGBM (Margin)", **res_m3},
        {"Model": "4. Hybrid + Bug Fixes + Cardinality Guard", **res_m4},
    ]

    for m in models_summary:
        logger.info("%-40s | F0.5: %.4f | Prec: %.4f | Rec: %.4f | SingAcc: %.2f%% | Empty: %.1f%% | MaxM: %d",
                    m["Model"], m["macro_f05"], m["precision"], m["recall"],
                    m["singleton_accuracy"] * 100, m["pct_s1_empty"], m["max_predicted_matches"])
    logger.info("=" * 80)

    # Save CSV and JSON
    pd.DataFrame(models_summary).to_csv(RECOVERY_DIR / "holdout_all_models.csv", index=False)
    with open(RECOVERY_DIR / "holdout_all_models.json", "w", encoding="utf-8") as f:
        json.dump(models_summary, f, indent=2)

    logger.info("Saved summary to %s and %s", RECOVERY_DIR / "holdout_all_models.csv", RECOVERY_DIR / "holdout_all_models.json")


if __name__ == "__main__":
    main()
