#!/usr/bin/env python3
"""
evaluate_argmax_matcher.py - Testing 1-to-1 Mutually Optimal Cluster Assignment.
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

# Import core preprocessors
from evaluate_optimized_matcher import clean_name, clean_address, jaro_winkler, STOPWORDS


def compute_affinity(s1: dict, s2: dict) -> float:
    """Computes a continuous matching affinity score in [0.0, 1.0]."""
    # Hard state filter
    st1, st2 = s1["state"], s2["state"]
    if st1 and st2 and st1 != st2:
        return 0.0

    cname1, cname2 = s1["clean_name"], s2["clean_name"]
    alpha1, alpha2 = s1["alpha_name"], s2["alpha_name"]
    
    spec1, spec2 = s1["specific_addr"], s2["specific_addr"]
    spec_inter = spec1 & spec2
    num_spec = len(spec_inter)
    
    dig1, dig2 = s1["digits"], s2["digits"]
    digits_overlap = bool(dig1 & dig2)
    has_both_digits = bool(dig1 and dig2)
    digits_conflict = bool(has_both_digits and not digits_overlap)
    if digits_conflict and num_spec == 0:
        return 0.0

    # Exact name / domain match
    if alpha1 and alpha2 and alpha1 == alpha2:
        return 0.99 if not digits_conflict else 0.85

    dom1, dom2 = s1["domain_stem"], s2["domain_stem"]
    if dom2 and (dom2 == alpha1 or (len(dom2) >= 8 and (dom2 in alpha1 or alpha1 in dom2))):
        return 0.98 if not digits_conflict else 0.80
    if dom1 and (dom1 == alpha2 or (len(dom1) >= 8 and (dom1 in alpha2 or alpha2 in dom1))):
        return 0.98 if not digits_conflict else 0.80

    jw_n = jaro_winkler(cname1, cname2) if (cname1 and cname2) else 0.0
    ntoks1, ntoks2 = s1["name_tokens"], s2["name_tokens"]
    name_inter = ntoks1 & ntoks2
    num_name_inter = len(name_inter)
    min_tokens = min(len(ntoks1), len(ntoks2)) if (ntoks1 and ntoks2) else 0

    # Address score
    spec_score = min(num_spec * 0.35, 0.70)
    if digits_overlap:
        spec_score += 0.25

    # Indic script handling
    if s1["has_indic"] or s2["has_indic"]:
        if num_spec >= 2 or (num_spec >= 1 and digits_overlap):
            return 0.90 + spec_score * 0.1
        return 0.0

    # English / Latin scoring
    name_score = jw_n
    if min_tokens >= 2 and num_name_inter >= min_tokens - 1:
        name_score = max(name_score, 0.90)

    # Combined affinity
    if not spec1 or not spec2:
        # Address missing in one
        return name_score * 0.95 if name_score >= 0.88 else 0.0
    else:
        if num_spec >= 1 or digits_overlap:
            return 0.5 * name_score + 0.5 * spec_score
        else:
            return name_score * 0.85 if name_score >= 0.94 else 0.0


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
    print("Testing Mutually Optimal Cluster Resolution on 5,000 entities...", flush=True)
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
                            if len(t) >= 3: name_idx[(p[3], t)].append(cid)
                        for at in spectoks: spec_addr_idx[(p[3], at)].append(cid)
                        for d in digs:
                            if len(d) >= 3: digits_idx[(p[3], d)].append(cid)
                        if dom: domain_idx[(p[3], dom)].append(cid)

    print(f"Indexed {len(s23_data)} mentions. Computing global affinities...", flush=True)
    t0 = time.time()
    
    # Candidate scores: candidate -> list of (s1_id, score)
    best_cand_for_s1 = defaultdict(dict)
    
    for s1 in s1_records:
        s1_id = s1["id"]
        cands = set()
        c = s1["country"]
        for t in s1["name_tokens"]:
            if len(t) >= 3:
                p = name_idx.get((c, t), [])
                if len(p) <= 250: cands.update(p)
        for at in s1["specific_addr"]:
            p = spec_addr_idx.get((c, at), [])
            if len(p) <= 150: cands.update(p)
        for d in s1["digits"]:
            if len(d) >= 3:
                p = digits_idx.get((c, d), [])
                if len(p) <= 100: cands.update(p)
        if s1["domain_stem"]:
            p = domain_idx.get((c, s1["domain_stem"]), [])
            cands.update(p)

        for cid in cands:
            cand = s23_data[cid]
            aff = compute_affinity(s1, cand)
            if aff >= 0.70:
                best_cand_for_s1[s1_id][cid] = aff

    # 1-to-1 optimal assignment: Each candidate S2/S3 is assigned to the single S1 with highest affinity
    cand_to_best_s1 = {}
    for s1_id, cand_scores in best_cand_for_s1.items():
        for cid, score in cand_scores.items():
            if cid not in cand_to_best_s1 or score > cand_to_best_s1[cid][1]:
                cand_to_best_s1[cid] = (s1_id, score)

    # Invert to final predictions
    final_pred = defaultdict(set)
    for cid, (s1_id, score) in cand_to_best_s1.items():
        if score >= 0.75:
            final_pred[s1_id].add(cid)

    # Evaluate
    f05_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    total_singletons = 0
    
    for s1 in s1_records:
        s1_id = s1["id"]
        true_m = gt[s1_id]
        pred_m = final_pred.get(s1_id, set())
        
        is_singleton = len(true_m) == 0
        if is_singleton:
            total_singletons += 1
            if len(pred_m) == 0:
                singleton_correct += 1

        score = entity_f05(pred_m, true_m)
        f05_scores.append(score)
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

    elapsed = time.time() - t0
    macro_f05 = sum(f05_scores) / len(f05_scores)
    macro_prec = sum(precisions) / len(precisions)
    macro_rec = sum(recalls) / len(recalls)
    sing_acc = singleton_correct / total_singletons if total_singletons else 1.0

    print("=" * 60, flush=True)
    print(f"EVALUATION RESULTS ({len(s1_records)} S1 entities in {elapsed:.2f}s):", flush=True)
    print(f"  Macro F0.5:         {macro_f05:.4f}", flush=True)
    print(f"  Precision:          {macro_prec:.4f}", flush=True)
    print(f"  Recall:             {macro_rec:.4f}", flush=True)
    print(f"  Singleton Accuracy: {sing_acc:.4f} ({singleton_correct}/{total_singletons})", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
