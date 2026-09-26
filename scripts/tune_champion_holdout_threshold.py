#!/usr/bin/env python3
"""
tune_champion_holdout_threshold.py — Phase 11 & 14 Threshold Tuning on 30K Holdout.
Uses the high-recall multi-channel candidate pool (K=100, 84.94% recall) with:
  - Bug 1 fix: Address-supported exact name
  - Bug 2 fix: Accurate state extraction
  - Bug 3 fix: Accurate street number extraction
  - Phase 9: High-cardinality ceiling (max 10 matches)

Sweeps tau, high_conf, addr_floor to find the exact configuration that
maximizes Macro F0.5.
Saves:
  - reports/recovery/threshold_sweep_holdout.csv
  - reports/recovery/best_holdout_model.json
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostic_200k"
RECOVERY_DIR = ROOT / "reports" / "recovery"
RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TuneThreshold")

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

ADDR_ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'dr': 'drive',
    'ln': 'lane', 'blvd': 'boulevard', 'bvd': 'boulevard', 'bd': 'boulevard',
    'ct': 'court', 'pl': 'place', 'sq': 'square', 'pkwy': 'parkway',
    'hwy': 'highway', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment',
    'dept': 'department', 'bldg': 'building', 'opp': 'opposite',
    'nr': 'near', 'adj': 'adjacent', 'ind': 'industrial', 'sec': 'sector',
    'dist': 'district', 'vill': 'village', 'po': 'post office'
}

class CompactCand:
    __slots__ = ("eid", "clean_name", "clean_addr", "alpha_name", "name_tokens")
    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str, name_tokens: tuple):
        self.eid = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.name_tokens = name_tokens

def fast_clean(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(" ", str(s).lower()).strip()

def fast_alpha(s: str) -> str:
    if not s: return ""
    return RE_ALPHA.sub("", str(s).lower()).strip()

def expand_addr_tokens(addr_clean: str) -> set:
    if not addr_clean: return set()
    return {ADDR_ABBREVIATIONS.get(t, t) for t in addr_clean.split()}

def extract_state_fixed(addr_clean: str, country: str) -> str:
    if not addr_clean: return ""
    words = addr_clean.split()
    if not words: return ""
    n = len(words)
    if country == "US":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_US_STATE_CODES:
                if i <= 2: return w
                if i < n - 1 and len(words[-i]) == 5 and words[-i].isdigit(): return w
                if w not in {"in", "to", "or", "at", "as", "by", "on", "no"}: return w
        for name, code in US_STATES.items():
            if name in addr_clean: return code
    elif country == "India":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_IN_STATE_CODES:
                if i <= 2 or w not in {"in", "to", "or", "at", "as"}: return w
        for name, code in INDIA_STATES.items():
            if name in addr_clean: return code
    return ""

def get_street_num_fixed(addr_clean: str) -> str:
    if not addr_clean: return ""
    postal = ""
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6):
            postal = d
            break
    for d in RE_DIGITS.findall(addr_clean):
        if d != postal and len(d) <= 5:
            return d
    return ""

def get_postal_fast(addr_clean: str) -> str:
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6): return d
    return ""

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(" ", "")
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def jaro_winkler(s1: str, s2: str, max_len: int = 40) -> float:
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1
    m1 = [False] * l1; m2 = [False] * l2
    matches = 0
    for i in range(l1):
        for j in range(max(0, i-md), min(i+md+1, l2)):
            if m2[j] or s1[i] != s2[j]: continue
            m1[i] = m2[j] = True; matches += 1; break
    if not matches: return 0.0
    t = 0; k = 0
    for i in range(l1):
        if not m1[i]: continue
        while not m2[k]: k += 1
        if s1[i] != s2[k]: t += 1
        k += 1
    sim = (matches/l1 + matches/l2 + (matches - t/2)/matches) / 3
    p = sum(1 for i in range(min(4, l1, l2)) if s1[i] == s2[i])
    return sim + p * 0.1 * (1.0 - sim)

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
    for eid in s1_ids:
        pred_set = predictions.get(eid, set())
        true_set = gt.get(eid, set())
        is_singleton = len(true_set) == 0
        if is_singleton:
            singletons_total += 1
            if len(pred_set) == 0: singletons_correct += 1
            else: false_merges += len(pred_set)

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

    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        "false_merges": int(false_merges),
        "missed_matches": int(missed_matches)
    }

def main():
    logger.info("=" * 80)
    logger.info("PHASE 11 & 14: THRESHOLD OPTIMIZATION ON 30K HOLDOUT")
    logger.info("=" * 80)

    # 1. Load Split
    with open(DIAG_DIR / "split_holdout_30k_s1_ids.json", "r", encoding="utf-8") as f:
        holdout_ids = json.load(f)
    holdout_set = set(holdout_ids)

    # 2. Load Ground Truth
    gt = defaultdict(set)
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                gt[parts[0]] = set(x.strip() for x in parts[1].split(",") if x.strip())

    # 3. Load S1 Records
    s1_dict = {}
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in holdout_set:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                cn = fast_clean(name)
                ca = fast_clean(addr)
                s1_dict[eid] = {
                    "id": eid, "clean_name": cn, "clean_addr": ca,
                    "alpha_name": fast_alpha(name),
                    "country": country,
                    "state": extract_state_fixed(ca, country),
                    "street_num": get_street_num_fixed(ca),
                    "postal": get_postal_fast(ca),
                    "name_tokens": set(cn.split()),
                    "addr_tokens": set(ca.split()),
                    "expanded_addr": expand_addr_tokens(ca),
                    "digits": set(RE_DIGITS.findall(ca)),
                    "ng4": get_char_ngrams(cn, 4)
                }

    # 4. Load S2/S3
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
                    cands_by_country[country][cid] = CompactCand(
                        eid=cid, clean_name=cn, clean_addr=ca,
                        alpha_name=fast_alpha(name),
                        name_tokens=tuple(cn.split())
                    )

    # 5. Multi-channel indexing
    indexes = {}
    for country, pool in cands_by_country.items():
        alpha_idx = defaultdict(list)
        token_idx = defaultdict(list)
        rare_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        digit_idx = defaultdict(list)
        ng4_idx = defaultdict(list)

        token_counts = defaultdict(int)
        for c in pool.values():
            for t in c.name_tokens: token_counts[t] += 1

        for cid, c in pool.items():
            if c.alpha_name and len(c.alpha_name) >= 3:
                p = alpha_idx[c.alpha_name]
                if len(p) < 150: p.append(cid)
            for t in c.name_tokens:
                if len(t) >= 3:
                    freq = token_counts[t]
                    p = token_idx[t]
                    if len(p) < (60 if freq > 200 else 150): p.append(cid)
                    if freq < 40:
                        p_r = rare_idx[t]
                        if len(p_r) < 80: p_r.append(cid)
            for at in c.clean_addr.split():
                if len(at) >= 3:
                    p = addr_idx[at]
                    if len(p) < 80: p.append(cid)
            for d in RE_DIGITS.findall(c.clean_addr):
                p = digit_idx[d]
                if len(p) < 80: p.append(cid)
            for ng in get_char_ngrams(c.clean_name, 4):
                p = ng4_idx[ng]
                if len(p) < 60: p.append(cid)

        indexes[country] = {
            "alpha": alpha_idx, "token": token_idx, "rare": rare_idx,
            "addr": addr_idx, "digit": digit_idx, "ng4": ng4_idx
        }

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

    # Retrieve and rank top-100 candidates for holdout S1 entities
    logger.info("Retrieving & Ranking Top-100 Candidates for Holdout S1...")
    holdout_top100 = {}
    for eid in holdout_ids:
        s1 = s1_dict[eid]
        c = s1["country"]
        idx = indexes[c]
        pool = cands_by_country[c]

        raw = set()
        if s1["alpha_name"]: raw.update(idx["alpha"].get(s1["alpha_name"], []))
        for t in s1["name_tokens"]:
            if len(t) >= 3: raw.update(idx["token"].get(t, []))
            if len(t) >= 3: raw.update(idx["rare"].get(t, []))
        for at in s1["addr_tokens"]:
            if len(at) >= 3: raw.update(idx["addr"].get(at, []))
        for d in s1["digits"]: raw.update(idx["digit"].get(d, []))
        for ng in s1["ng4"]: raw.update(idx["ng4"].get(ng, []))

        # Rank
        s1_alpha = s1["alpha_name"]
        s1_nt = s1["name_tokens"]
        s1_at = s1["addr_tokens"]
        s1_digits = s1["digits"]

        scored = []
        for cid in raw:
            cand = pool.get(cid)
            if not cand: continue
            score = 0.0
            if s1_alpha and cand.alpha_name and s1_alpha == cand.alpha_name:
                score += 5.0
            c_nt = set(cand.name_tokens)
            inter_n = len(s1_nt & c_nt)
            if inter_n: score += 3.0 * (inter_n / (len(s1_nt) + len(c_nt) - inter_n))
            c_at = set(cand.clean_addr.split())
            inter_a = len(s1_at & c_at)
            if inter_a: score += 2.0 * (inter_a / (len(s1_at) + len(c_at) - inter_a))
            c_digits = set(RE_DIGITS.findall(cand.clean_addr))
            if s1_digits & c_digits: score += 2.0
            scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        holdout_top100[eid] = [cid for _, cid in scored[:100]]

    # Precompute pairwise signals (base_score, addr_jaccard, is_valid) for holdout pairs
    logger.info("Precomputing pairwise match scores for all holdout pairs...")
    pair_signals = defaultdict(list)
    for eid in holdout_ids:
        s1 = s1_dict[eid]
        pool = cands_by_country[s1["country"]]
        c_list = holdout_top100[eid]

        s1_st = s1["state"]
        s1_sn = s1["street_num"]
        s1_po = s1["postal"]
        s1_cn = s1["clean_name"]
        s1_ca = s1["clean_addr"]
        s1_nt = s1["name_tokens"]
        s1_exp_a = s1["expanded_addr"]
        s1_alpha = s1["alpha_name"]
        s1_ng4 = s1["ng4"]

        for cid in c_list:
            c = pool.get(cid)
            if not c: continue
            st2, sn2, po2 = get_cand_meta(c, s1["country"])

            # Bug 2 fix: State conflict hard rejection
            if s1_st and st2 and s1_st != st2:
                continue

            # Bug 3 fix: Fast street num + postal conflict
            if (s1_sn and sn2 and s1_sn != sn2) and (s1_po and po2 and s1_po != po2):
                continue

            conflict = 1.0 if (s1_sn and sn2 and s1_sn != sn2) else (0.2 if (s1_sn or sn2) else 0.0)
            postal_conflict = (s1_po != po2) if (s1_po and po2) else False
            postal_match = (s1_po == po2) if (s1_po and po2) else False

            jw = jaro_winkler(s1_cn, c.clean_name)
            c_nt = set(c.name_tokens)
            inter_n = len(s1_nt & c_nt)
            tj_name = (inter_n / (len(s1_nt) + len(c_nt) - inter_n)) if (s1_nt or c_nt) else 1.0

            c_exp = expand_addr_tokens(c.clean_addr)
            inter_a = len(s1_exp_a & c_exp)
            tj_addr = (inter_a / (len(s1_exp_a) + len(c_exp) - inter_a)) if (s1_exp_a or c_exp) else 0.0

            base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.30 * conflict
            if postal_conflict: base -= 0.15
            elif postal_match: base += 0.06

            c_ng4 = get_char_ngrams(c.clean_name, 4)
            if s1_ng4 and c_ng4:
                ng_inter = len(s1_ng4 & c_ng4)
                ng_j = ng_inter / (len(s1_ng4) + len(c_ng4) - ng_inter)
                if ng_j >= 0.50 and jw >= 0.65:
                    base = max(base, 0.50 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.20 * ng_j - 0.30 * conflict)

            # Bug 1 fix: Require address evidence if both have addresses before boosting exact alpha name
            both_have_addr = bool(s1_ca and c.clean_addr)
            digits_agree = bool(s1["digits"] & set(RE_DIGITS.findall(c.clean_addr)))
            addr_supported = (tj_addr >= 0.08) or digits_agree or (not both_have_addr)

            if s1_alpha and c.alpha_name and s1_alpha == c.alpha_name and len(s1_alpha) >= 4 and conflict < 1.0:
                if addr_supported:
                    base = max(base, 0.40 + 0.35 * jw + 0.25 * tj_addr - 0.20 * conflict)
                else:
                    base = min(base, 0.65) # Suppress unconstrained false merges!

            if base > 0.30:
                pair_signals[eid].append((base, tj_addr, cid))

    logger.info("Precomputed %d entities with plausible candidate matches.", len(pair_signals))

    # Grid search over thresholds
    tau_grid = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    high_conf_grid = [0.70, 0.72, 0.75, 0.78]
    addr_floor_grid = [0.14, 0.16, 0.18, 0.20, 0.22]

    logger.info("Sweeping %d parameter combinations...", len(tau_grid) * len(high_conf_grid) * len(addr_floor_grid))
    sweep_results = []
    best_config = None
    best_f05 = -1.0

    for high_conf in high_conf_grid:
        for tau in tau_grid:
            for addr_floor in addr_floor_grid:
                preds = defaultdict(set)
                for eid, pairs in pair_signals.items():
                    pairs.sort(key=lambda x: x[0], reverse=True)
                    matched = []
                    for base, tj_addr, cid in pairs:
                        if base >= high_conf or (base >= tau and tj_addr >= addr_floor):
                            matched.append(cid)
                        if len(matched) >= 10: # Phase 9 Cardinality Guard
                            break
                    if matched:
                        preds[eid] = set(matched)

                res = evaluate_predictions(preds, gt, holdout_ids)
                f05 = res["macro_f05"]
                sweep_results.append({
                    "tau": tau,
                    "high_conf": high_conf,
                    "addr_floor": addr_floor,
                    "macro_f05": round(f05, 6),
                    "precision": round(res["precision"], 6),
                    "recall": round(res["recall"], 6),
                    "singleton_accuracy": round(res["singleton_accuracy"], 4),
                    "false_merges": res["false_merges"],
                    "missed_matches": res["missed_matches"]
                })

                if f05 > best_f05:
                    best_f05 = f05
                    best_config = {
                        "model": "Champion Heuristic + Multi-Channel High-Recall Blocking + Bug Fixes + Cardinality Guard",
                        "optimal_tau": tau,
                        "high_confidence": high_conf,
                        "addr_floor": addr_floor,
                        "max_k": 100,
                        "macro_f05": f05,
                        "precision": res["precision"],
                        "recall": res["recall"],
                        "singleton_accuracy": res["singleton_accuracy"],
                        "false_merges": res["false_merges"],
                        "missed_matches": res["missed_matches"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.sort_values(by="macro_f05", ascending=False, inplace=True)
    df_sweep.to_csv(RECOVERY_DIR / "threshold_sweep_holdout.csv", index=False)
    logger.info("Saved threshold sweep to %s", RECOVERY_DIR / "threshold_sweep_holdout.csv")

    logger.info("=" * 80)
    logger.info("BEST HOLDOUT CONFIGURATION FOUND:")
    logger.info("  Optimal Macro F0.5:   %.6f", best_config["macro_f05"])
    logger.info("  Precision:            %.6f", best_config["precision"])
    logger.info("  Recall:               %.6f", best_config["recall"])
    logger.info("  Singleton Accuracy:   %.2f%%", best_config["singleton_accuracy"] * 100)
    logger.info("  Optimal tau:          %.2f", best_config["optimal_tau"])
    logger.info("  High Confidence:      %.2f", best_config["high_confidence"])
    logger.info("  Address Floor:        %.2f", best_config["addr_floor"])
    logger.info("  False Merges:         %d", best_config["false_merges"])
    logger.info("=" * 80)

    with open(RECOVERY_DIR / "best_holdout_model.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Saved best model configuration to %s", RECOVERY_DIR / "best_holdout_model.json")

if __name__ == "__main__":
    main()
