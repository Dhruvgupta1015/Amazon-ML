#!/usr/bin/env python3
"""
scripts/reproduce_champion_0793.py
Phase 1: Deterministic Reproduction of Champion Macro F0.5 = 0.7930
"""
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReproduceChampion")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'pvt ltd', 'private limited', 'co', 'company', 'sarl', 'sas', 'sa',
    'gmbh', 'ag', 'llp', 'pllc'
}

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)
    return ' '.join(s.split())

def extract_tokens(text: str) -> set[str]:
    return set(text.split())

def extract_digits(text: str) -> set[str]:
    if not text:
        return set()
    return set(RE_DIGITS.findall(str(text)))

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

def token_jaccard(toks1: set[str], toks2: set[str]) -> float:
    if not toks1 and not toks2:
        return 1.0
    if not toks1 or not toks2:
        return 0.0
    inter = len(toks1 & toks2)
    union = len(toks1 | toks2)
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

def preprocess_record(r: dict) -> dict:
    name = r.get('business_name', '')
    addr = r.get('business_address', '')
    c_name = clean_text(name)
    c_addr = clean_text(addr)
    name_toks = extract_tokens(c_name)
    addr_toks = extract_tokens(c_addr)
    digits = extract_digits(addr)
    street_num = None
    m = re.match(r'^(\d+)\b', c_addr)
    if m:
        street_num = m.group(1)
    postal = None
    for d in digits:
        if len(d) in (5, 6):
            postal = d
            break
    return {
        'id': r['entity_id'],
        'clean_name': c_name,
        'clean_addr': c_addr,
        'name_tokens': name_toks,
        'addr_tokens': addr_toks,
        'street_num': street_num,
        'postal': postal,
        'country': r.get('country', '')
    }

def main():
    logger.info("=" * 60)
    logger.info("PHASE 1: REPRODUCE CHAMPION 0.7930 ON FROZEN VALIDATION SPLIT")
    logger.info("=" * 60)

    # 1. Load Frozen Validation Partition
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]
    val_s1_set = set(val_s1_ids)

    # 2. Load Ground Truth
    gt = {}
    with open(DATA_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in val_s1_set}

    needed_positive_cids = set()
    for e in val_s1_set:
        needed_positive_cids.update(val_gt[e])
    total_true_positives = len(needed_positive_cids)
    logger.info("Validation S1: %d entities | True positive mentions: %d", len(val_s1_ids), total_true_positives)

    # 3. Load Validation S1 Records
    val_s1_records = []
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in val_s1_set:
                val_s1_records.append(preprocess_record(r))

    # 4. Load S2 & S3 Mentions (all true targets + 100k distractors per source)
    cands_processed = {}
    for s_file in [DATA_ROOT / "train" / "train_source2.tsv", DATA_ROOT / "train" / "train_source3.tsv"]:
        collected = 0
        with open(s_file, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    if cid in needed_positive_cids or collected < 100000:
                        cands_processed[cid] = preprocess_record({
                            "entity_id": cid, "business_name": name,
                            "business_address": addr, "country": country
                        })
                        if cid not in needed_positive_cids:
                            collected += 1

    logger.info("Loaded %d candidate mentions into memory.", len(cands_processed))

    # 5. Build Inverted Index
    name_token_index = defaultdict(list)
    addr_token_index = defaultdict(list)
    street_num_index = defaultdict(list)
    postal_index = defaultdict(list)

    for cid, cand in cands_processed.items():
        c = cand["country"]
        for tok in cand["name_tokens"]:
            name_token_index[(c, tok)].append(cid)
        for atok in cand["addr_tokens"]:
            addr_token_index[(c, atok)].append(cid)
        if cand["street_num"]:
            street_num_index[(c, cand["street_num"])].append(cid)
        if cand["postal"]:
            postal_index[(c, cand["postal"])].append(cid)

    # 6. Candidate Generation (Strategy D: Country-Partitioned Multi-Index)
    logger.info("Generating candidates via Strategy D (Country-Partitioned Multi-Index)...")
    val_candidates = {}
    recovered_tp = 0
    total_cand_count = 0

    for s1 in val_s1_records:
        s1_id = s1["id"]
        s1_country = s1["country"]
        true_set = set(val_gt.get(s1_id, []))
        cands = set()

        for tok in s1["name_tokens"]:
            postings = name_token_index.get((s1_country, tok), [])
            if len(postings) <= 60:
                cands.update(postings)
        for atok in s1["addr_tokens"]:
            postings = addr_token_index.get((s1_country, atok), [])
            if len(postings) <= 40:
                cands.update(postings)
        if s1["street_num"]:
            postings = street_num_index.get((s1_country, s1["street_num"]), [])
            if len(postings) <= 40:
                cands.update(postings)
        if s1["postal"]:
            postings = postal_index.get((s1_country, s1["postal"]), [])
            if len(postings) <= 30:
                cands.update(postings)

        cand_list = list(cands)[:50]
        val_candidates[s1_id] = cand_list
        total_cand_count += len(cand_list)
        recovered_tp += len(set(cand_list) & true_set)

    cand_recall = (recovered_tp / sum(len(v) for v in val_gt.values())) * 100
    avg_cands = total_cand_count / len(val_s1_records)
    logger.info("Candidate Recall: %.2f%% | Total Candidates: %d | Avg per S1: %.2f",
                cand_recall, total_cand_count, avg_cands)

    # 7. Champion Heuristic Scoring (0.5 * JW + 0.25 * NameJaccard + 0.25 * AddrJaccard - 0.3 * Conflict)
    logger.info("Scoring candidates with Champion Heuristic formula...")
    t0 = time.time()
    tau = 0.56
    predictions = {}

    for s1 in val_s1_records:
        s1_id = s1["id"]
        cand_list = val_candidates.get(s1_id, [])
        matched = []
        for cid in cand_list:
            cand = cands_processed.get(cid)
            if not cand:
                continue
            jw = jaro_winkler_distance(s1["clean_name"], cand["clean_name"])
            tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
            tj_addr = token_jaccard(s1["addr_tokens"], cand["addr_tokens"])

            sn1, sn2 = s1["street_num"], cand["street_num"]
            if sn1 and sn2:
                conflict = 1.0 if sn1 != sn2 else 0.0
            elif sn1 or sn2:
                conflict = 0.2
            else:
                conflict = 0.0

            score = 0.5 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.3 * conflict
            if score >= tau:
                matched.append(cid)
        predictions[s1_id] = matched

    elapsed = time.time() - t0

    # 8. Evaluation
    f05_list, p_list, r_list = [], [], []
    singletons_total = 0
    singletons_correct = 0
    false_merges = 0

    for eid in val_s1_ids:
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

    macro_f05 = float(np.mean(f05_list))
    prec = float(np.mean(p_list))
    rec = float(np.mean(r_list))
    sing_acc = (singletons_correct / singletons_total) * 100 if singletons_total > 0 else 100.0

    logger.info("=" * 60)
    logger.info("REPRODUCED CHAMPION METRICS (tau = %.2f):", tau)
    logger.info("  - Macro F0.5:         %.4f", macro_f05)
    logger.info("  - Macro Precision:    %.4f (%.2f%%)", prec, prec * 100)
    logger.info("  - Macro Recall:       %.4f (%.2f%%)", rec, rec * 100)
    logger.info("  - Singleton Accuracy: %.4f (%.2f%%)", sing_acc / 100, sing_acc)
    logger.info("  - False Merges:       %d", false_merges)
    logger.info("  - Runtime:            %.2fs", elapsed)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
