#!/usr/bin/env python3
"""
scripts/phase2_error_analysis.py
Phase 2: Comprehensive Error Analysis on Frozen Validation Split.

Diagnoses:
1. False merges of singleton entities (why did singletons get false matches?)
2. Missed true matches (why did non-singletons miss true links?)
3. Candidates missing from candidate generation (retrieval upper bound bottleneck)
4. High-name-similarity false positives (same name, different entity/location)
5. Shared-address-number false positives (different company at same street number)
6. Same-business-name / different-location cases
"""
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ErrorAnalysis")

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)
    return ' '.join(s.split())

def extract_tokens(text: str) -> set:
    return set(text.split())

def extract_digits(text: str) -> set:
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

def token_jaccard(toks1: set, toks2: set) -> float:
    if not toks1 and not toks2:
        return 1.0
    if not toks1 or not toks2:
        return 0.0
    inter = len(toks1 & toks2)
    union = len(toks1 | toks2)
    return inter / union if union > 0 else 0.0

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
    logger.info("PHASE 2: DETAILED ERROR ANALYSIS ON FROZEN VALIDATION SPLIT")
    logger.info("=" * 60)

    # 1. Load Split
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]
    val_s1_set = set(val_s1_ids)

    # 2. Load GT
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

    # 3. Load S1
    s1_map = {}
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in val_s1_set:
                s1_map[r["entity_id"]] = preprocess_record(r)

    # 4. Load S2/S3
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

    # Inverted Index
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

    # Generate candidates & evaluate
    tau = 0.56
    singleton_false_merges = []
    missed_true_matches = []
    missed_at_retrieval = []
    high_name_sim_fps = []
    shared_address_num_fps = []
    same_name_diff_loc_fps = []

    for eid in val_s1_ids:
        s1 = s1_map[eid]
        true_set = set(val_gt.get(eid, []))
        cands = set()
        c = s1["country"]

        for tok in s1["name_tokens"]:
            postings = name_token_index.get((c, tok), [])
            if len(postings) <= 60:
                cands.update(postings)
        for atok in s1["addr_tokens"]:
            postings = addr_token_index.get((c, atok), [])
            if len(postings) <= 40:
                cands.update(postings)
        if s1["street_num"]:
            postings = street_num_index.get((c, s1["street_num"]), [])
            if len(postings) <= 40:
                cands.update(postings)
        if s1["postal"]:
            postings = postal_index.get((c, s1["postal"]), [])
            if len(postings) <= 30:
                cands.update(postings)

        cand_list = list(cands)[:50]
        cand_set = set(cand_list)

        # Check missed at retrieval
        if true_set:
            unretrieved = true_set - cand_set
            if unretrieved:
                missed_at_retrieval.append({
                    "s1_id": eid,
                    "s1_name": s1["clean_name"],
                    "s1_addr": s1["clean_addr"],
                    "unretrieved_targets": list(unretrieved)
                })

        # Score candidates
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
                if cid not in true_set:
                    # Categorize False Positive
                    if not true_set:
                        singleton_false_merges.append((eid, cid, s1["clean_name"], cand["clean_name"], s1["clean_addr"], cand["clean_addr"], score))
                    if jw >= 0.90 and tj_addr < 0.20:
                        same_name_diff_loc_fps.append((eid, cid, s1["clean_name"], cand["clean_name"], s1["clean_addr"], cand["clean_addr"], score))
                    elif jw >= 0.85:
                        high_name_sim_fps.append((eid, cid, s1["clean_name"], cand["clean_name"], score))
                    elif sn1 and sn2 and sn1 == sn2 and jw < 0.60:
                        shared_address_num_fps.append((eid, cid, s1["clean_name"], cand["clean_name"], sn1, score))

        # Check missed true matches
        if true_set:
            unmatched = true_set - set(matched)
            if unmatched:
                missed_true_matches.append({
                    "s1_id": eid,
                    "s1_name": s1["clean_name"],
                    "s1_addr": s1["clean_addr"],
                    "missing_true_targets": list(unmatched)
                })

    logger.info("Error Analysis Statistics:")
    logger.info("  1. Singleton False Merges:              %d cases", len(singleton_false_merges))
    logger.info("  2. Missed True Matches (End-to-End):    %d entities", len(missed_true_matches))
    logger.info("  3. Missed at Candidate Generation:      %d entities (Blocking recall bottleneck)", len(missed_at_retrieval))
    logger.info("  4. Same Business Name / Diff Location:  %d cases", len(same_name_diff_loc_fps))
    logger.info("  5. High Name Similarity False Positives: %d cases", len(high_name_sim_fps))
    logger.info("  6. Shared Address Number Distractors:   %d cases", len(shared_address_num_fps))

    # Save detailed markdown report
    md_path = REPORTS_DIR / "error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Systematic Error Analysis: Active Champion vs Validation Partition\n\n")
        f.write(f"**Validation Population**: 20,000 Frozen S1 Entities (`data/frozen_val_s1_ids.json`)\n\n")
        f.write("## 1. Quantitative Breakdown of Errors\n\n")
        f.write("| Error Mode | Count | Root Cause | Architectural Mitigation |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Singleton False Merges** | {len(singleton_false_merges):,} | Loose similarity threshold matches distractor on singletons | Strict score margin guard & top-1 ambiguity check |\n")
        f.write(f"| **Retrieval Upper Bound Gap** | {len(missed_at_retrieval):,} | True targets not retrieved by simple token/street indexing | Character n-grams (3/4-gram) + rare token channels |\n")
        f.write(f"| **End-to-End Missed Matches** | {len(missed_true_matches):,} | Scoring below tau=0.56 due to heavy address variation | Hybrid ML reranker for ambiguous medium-score pairs |\n")
        f.write(f"| **Same Name / Different Location** | {len(same_name_diff_loc_fps):,} | Brand/chain stores across different cities/localities | Locality / Postal code agreement filter |\n")
        f.write(f"| **Shared Address Number False Positives** | {len(shared_address_num_fps):,} | Distinct co-located tenants at same street address | Heavier name agreement requirement when address matches |\n\n")

        f.write("## 2. Sample False Merges on Singletons (Examined)\n\n")
        for eid, cid, n1, n2, a1, a2, sc in singleton_false_merges[:5]:
            f.write(f"- **S1**: `{eid}` | Name: *{n1}* | Addr: *{a1}*\n")
            f.write(f"  **Cand**: `{cid}` | Name: *{n2}* | Addr: *{a2}*\n")
            f.write(f"  *Score*: {sc:.4f} (False merge — S1 is actually a singleton)\n\n")

        f.write("## 3. Sample Retrieval Bottlenecks (True targets not in candidates)\n\n")
        for item in missed_at_retrieval[:5]:
            f.write(f"- **S1**: `{item['s1_id']}` | Name: *{item['s1_name']}* | Addr: *{item['s1_addr']}*\n")
            f.write(f"  *Missing True Targets*: {item['unretrieved_targets']}\n\n")

    logger.info("Saved error analysis report to %s", md_path)

if __name__ == "__main__":
    main()
