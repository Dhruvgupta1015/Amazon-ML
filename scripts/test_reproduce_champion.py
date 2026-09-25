#!/usr/bin/env python3
"""
Test reproducing Champion Macro F0.5 = 0.7930 on the frozen validation partition.
"""
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)
    return ' '.join(s.split())

def extract_tokens(text: str) -> set[str]:
    return set(text.split()) - STOPWORDS

def extract_digits(text: str) -> set[str]:
    if not text:
        return set()
    return set(RE_DIGITS.findall(str(text)))

def extract_street_num(text: str) -> str:
    if not text:
        return ""
    d = RE_DIGITS.findall(str(text))
    return d[0] if d else ""

def jaro_winkler_distance(s1: str, s2: str, max_len: int = 40) -> float:
    if s1 == s2:
        return 1.0
    if max_len:
        s1 = s1[:max_len]
        s2 = s2[:max_len]
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    m = matches
    sim = (m / len1 + m / len2 + (m - transpositions / 2.0) / m) / 3.0
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return sim + prefix * 0.1 * (1.0 - sim)

def token_jaccard(t1: set[str], t2: set[str]) -> float:
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0

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

def main():
    print("Loading frozen validation set...")
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    target_val_ids = manifest["s1_entity_ids"]
    target_val_set = set(target_val_ids)

    print("Loading Ground Truth...")
    gt = {}
    with open(DATASET_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in target_val_set}
    needed_pos = set()
    for e in target_val_set:
        needed_pos.update(val_gt[e])

    print("Loading S1 validation records...")
    s1_records = []
    with open(DATASET_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in target_val_set:
                cname = clean_text(r["business_name"])
                caddr = clean_text(r["business_address"])
                s1_records.append({
                    "id": r["entity_id"],
                    "name": cname,
                    "addr": caddr,
                    "name_toks": extract_tokens(cname),
                    "addr_toks": extract_tokens(caddr),
                    "street_num": extract_street_num(caddr),
                    "country": r["country"]
                })

    print(f"Loaded {len(s1_records)} S1 validation records. Loading S2/S3 candidates...")
    s23_data = {}
    name_token_index = defaultdict(list)
    addr_token_index = defaultdict(list)
    street_num_index = defaultdict(list)

    for s_path in [DATASET_ROOT / "train" / "train_source2.tsv", DATASET_ROOT / "train" / "train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    cid, raw_name, raw_addr, country = p[0], p[1], p[2], p[3]
                    if cid in needed_pos or len(s23_data) < 260000:
                        cname = clean_text(raw_name)
                        caddr = clean_text(raw_addr)
                        ntoks = extract_tokens(cname)
                        atoks = extract_tokens(caddr)
                        snum = extract_street_num(caddr)
                        s23_data[cid] = {
                            "id": cid,
                            "name": cname,
                            "addr": caddr,
                            "name_toks": ntoks,
                            "addr_toks": atoks,
                            "street_num": snum,
                            "country": country
                        }
                        for t in ntoks:
                            name_token_index[(country, t)].append(cid)
                        for at in atoks:
                            addr_token_index[(country, at)].append(cid)
                        if snum:
                            street_num_index[(country, snum)].append(cid)

    print(f"Indexed {len(s23_data)} mentions. Resolving candidates & scoring...")
    t0 = time.time()
    
    # Evaluate across thresholds
    predictions_by_tau = defaultdict(dict)
    taus = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]

    for s1 in s1_records:
        s1_id = s1["id"]
        cands = set()
        c = s1["country"]
        for t in s1["name_toks"]:
            cands.update(name_token_index.get((c, t), [])[:60])
        for at in s1["addr_toks"]:
            cands.update(addr_token_index.get((c, at), [])[:40])
        if s1["street_num"]:
            cands.update(street_num_index.get((c, s1["street_num"]), [])[:40])

        cand_list = list(cands)[:50]
        scored_cands = []
        for cid in cand_list:
            cand = s23_data.get(cid)
            if not cand:
                continue
            jw = jaro_winkler_distance(s1["name"], cand["name"])
            tj_name = token_jaccard(s1["name_toks"], cand["name_toks"])
            tj_addr = token_jaccard(s1["addr_toks"], cand["addr_toks"])
            
            # Numeric conflict
            sn1, sn2 = s1["street_num"], cand["street_num"]
            if sn1 and sn2:
                conflict = 1.0 if sn1 != sn2 else 0.0
            elif sn1 or sn2:
                conflict = 0.2
            else:
                conflict = 0.0

            score = 0.5 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.3 * conflict
            scored_cands.append((cid, score))

        for tau in taus:
            predictions_by_tau[tau][s1_id] = [cid for cid, sc in scored_cands if sc >= tau]

    elapsed = time.time() - t0
    print(f"Scoring completed in {elapsed:.2f}s.\n")

    print(f"{'Tau':<8}{'Macro F0.5':<14}{'Precision':<14}{'Recall':<14}{'Singletons':<14}{'False Merges'}")
    print("-" * 75)
    for tau in taus:
        preds = predictions_by_tau[tau]
        f05_list, p_list, r_list = [], [], []
        singletons_total = 0
        singletons_correct = 0
        false_merges = 0

        for eid in target_val_ids:
            pred_set = set(preds.get(eid, []))
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

        macro_f05 = float(np.mean(f05_list))
        prec = float(np.mean(p_list))
        rec = float(np.mean(r_list))
        sing_acc = (singletons_correct / singletons_total) * 100

        print(f"{tau:<8.2f}{macro_f05:<14.4f}{prec*100:<14.2f}%{rec*100:<14.2f}%{sing_acc:<14.2f}%{false_merges}")

if __name__ == "__main__":
    main()
