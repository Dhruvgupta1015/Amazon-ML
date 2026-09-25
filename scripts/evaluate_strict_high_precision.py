#!/usr/bin/env python3
"""
evaluate_strict_high_precision.py - High Precision Threshold Sweep for 0.98+ Macro F0.5.
"""
import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
sys.stdout.reconfigure(encoding='utf-8')

from evaluate_optimized_matcher import clean_name, clean_address, jaro_winkler, STOPWORDS


def evaluate_pair_strict(s1: dict, s2: dict, jw_thresh: float, require_addr_anchor: bool) -> bool:
    # State check: If both specify states and they conflict, REJECT!
    st1, st2 = s1["state"], s2["state"]
    if st1 and st2 and st1 != st2:
        return False

    cname1, cname2 = s1["clean_name"], s2["clean_name"]
    alpha1, alpha2 = s1["alpha_name"], s2["alpha_name"]
    
    spec1, spec2 = s1["specific_addr"], s2["specific_addr"]
    spec_inter = spec1 & spec2
    num_spec_inter = len(spec_inter)
    
    dig1, dig2 = s1["digits"], s2["digits"]
    digits_overlap = bool(dig1 & dig2)
    has_both_digits = bool(dig1 and dig2)
    digits_conflict = bool(has_both_digits and not digits_overlap)

    # 1. Exact Name / High-Fidelity Domain Match
    if alpha1 and alpha2:
        if alpha1 == alpha2:
            if not digits_conflict:
                return True
        elif len(alpha1) >= 8 and len(alpha2) >= 8 and abs(len(alpha1) - len(alpha2)) <= 1:
            if alpha1 in alpha2 or alpha2 in alpha1:
                if not digits_conflict:
                    return True

    dom1, dom2 = s1["domain_stem"], s2["domain_stem"]
    if dom2 and len(dom2) >= 6:
        if dom2 == alpha1 or (len(dom2) >= 8 and (dom2 in alpha1 or alpha1 in dom2)):
            if not digits_conflict:
                return True
    if dom1 and len(dom1) >= 6:
        if dom1 == alpha2 or (len(dom1) >= 8 and (dom1 in alpha2 or alpha2 in dom1)):
            if not digits_conflict:
                return True

    # 2. String Similarities
    jw_n = jaro_winkler(cname1, cname2) if (cname1 and cname2) else 0.0
    ntoks1, ntoks2 = s1["name_tokens"], s2["name_tokens"]
    name_inter = ntoks1 & ntoks2
    num_name_inter = len(name_inter)
    min_tokens = min(len(ntoks1), len(ntoks2)) if (ntoks1 and ntoks2) else 0

    # CASE A: Near-identical name (typos / abbreviations)
    if jw_n >= jw_thresh:
        if not digits_conflict:
            if not spec1 or not spec2 or num_spec_inter >= 1 or digits_overlap or jw_n >= 0.98:
                return True

    if jw_n >= 0.90 and min_tokens >= 2 and num_name_inter >= min_tokens:
        if not digits_conflict:
            if not spec1 or not spec2 or num_spec_inter >= 1 or digits_overlap:
                return True

    # CASE B: Strong Address Match (Matching specific street name/city + matching digits)
    if (num_spec_inter >= 2 or (num_spec_inter >= 1 and digits_overlap)) and not digits_conflict:
        if s1["has_indic"] or s2["has_indic"]:
            return True
        if jw_n >= 0.75 or (min_tokens >= 1 and num_name_inter >= 1):
            return True

    # CASE C: Indic Script Name with Street / City Match
    if (s1["has_indic"] or s2["has_indic"]) and not digits_conflict:
        if num_spec_inter >= 2 or (num_spec_inter >= 1 and digits_overlap):
            return True

    return False


def entity_f05(pred_set: set[str], true_set: set[str]) -> float:
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def main():
    print("Evaluating Strict Precision Configurations on 5,000 validation entities...", flush=True)
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        val_ids = json.load(f)["s1_entity_ids"][:5000]
    val_set = set(val_ids)

    # Load GT
    gt = {}
    with open(DATASET_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_set:
                gt[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) >= 2 and p[1].strip() else set()

    needed_pos = set(x for s in gt.values() for x in s)

    # Load S1
    s1_records = []
    with open(DATASET_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in val_set:
                cn, alpha, dom, indic = clean_name(p[1])
                norm_a, atoks, spectoks, digs, st = clean_address(p[2], p[3])
                ntoks = set(cn.split()) - STOPWORDS
                s1_records.append({
                    "id": p[0], "clean_name": cn, "alpha_name": alpha, "domain_stem": dom,
                    "has_indic": indic, "name_tokens": ntoks,
                    "clean_addr": norm_a, "all_addr": atoks,
                    "specific_addr": spectoks, "digits": digs, "state": st, "country": p[3]
                })

    # Load S2/S3
    print("Loading S2 & S3 pool...", flush=True)
    s23_data = {}
    name_idx = defaultdict(list)
    spec_addr_idx = defaultdict(list)
    digits_idx = defaultdict(list)
    domain_idx = defaultdict(list)
    
    for s_path in [DATASET_ROOT / "train" / "train_source2.tsv", DATASET_ROOT / "train" / "train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid = p[0]
                    if cid in needed_pos or len(s23_data) < 250000:
                        cn, alpha, dom, indic = clean_name(p[1])
                        norm_a, atoks, spectoks, digs, st = clean_address(p[2], p[3])
                        ntoks = set(cn.split()) - STOPWORDS
                        rec = {
                            "id": cid, "clean_name": cn, "alpha_name": alpha, "domain_stem": dom,
                            "has_indic": indic, "name_tokens": ntoks,
                            "clean_addr": norm_a, "all_addr": atoks,
                            "specific_addr": spectoks, "digits": digs, "state": st, "country": p[3]
                        }
                        s23_data[cid] = rec
                        for t in ntoks:
                            if len(t) >= 3:
                                name_idx[(p[3], t)].append(cid)
                        for at in spectoks:
                            spec_addr_idx[(p[3], at)].append(cid)
                        for d in digs:
                            if len(d) >= 3:
                                digits_idx[(p[3], d)].append(cid)
                        if dom:
                            domain_idx[(p[3], dom)].append(cid)

    print(f"Indexed {len(s23_data)} mentions. Pre-retrieving candidates...", flush=True)
    s1_cands = {}
    for s1 in s1_records:
        cands = set()
        c = s1["country"]
        for t in s1["name_tokens"]:
            if len(t) >= 3:
                p = name_idx.get((c, t), [])
                if len(p) <= 250:
                    cands.update(p)
        for at in s1["specific_addr"]:
            p = spec_addr_idx.get((c, at), [])
            if len(p) <= 150:
                cands.update(p)
        for d in s1["digits"]:
            if len(d) >= 3:
                p = digits_idx.get((c, d), [])
                if len(p) <= 100:
                    cands.update(p)
        if s1["domain_stem"]:
            p = domain_idx.get((c, s1["domain_stem"]), [])
            cands.update(p)
        s1_cands[s1["id"]] = cands

    for jw_t in [0.92, 0.94, 0.95, 0.96]:
        f05_scores = []
        precisions = []
        recalls = []
        singleton_correct = 0
        total_singletons = 0

        for s1 in s1_records:
            s1_id = s1["id"]
            true_m = gt[s1_id]
            is_singleton = len(true_m) == 0
            if is_singleton:
                total_singletons += 1

            pred_m = set()
            for cid in s1_cands[s1_id]:
                cand = s23_data[cid]
                if evaluate_pair_strict(s1, cand, jw_thresh=jw_t, require_addr_anchor=False):
                    pred_m.add(cid)

            score = entity_f05(pred_m, true_m)
            f05_scores.append(score)
            if is_singleton and len(pred_m) == 0:
                singleton_correct += 1
            if pred_m and true_m:
                tp = len(pred_m & true_m)
                precisions.append(tp / len(pred_m))
                recalls.append(tp / len(true_m))
            elif not pred_m and not true_m:
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(0.0)

        macro_f05 = sum(f05_scores) / len(f05_scores)
        macro_prec = sum(precisions) / len(precisions)
        macro_rec = sum(recalls) / len(recalls)
        sing_acc = singleton_correct / total_singletons if total_singletons else 1.0

        print(f"[JW_Thresh={jw_t:.2f}] Macro F0.5: {macro_f05:.4f} | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f} | Sing: {sing_acc:.4f} ({singleton_correct}/{total_singletons})", flush=True)


if __name__ == "__main__":
    main()
