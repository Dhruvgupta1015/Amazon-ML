#!/usr/bin/env python3
"""
scripts/build_200k_diagnostic_set.py - Builds the 200,000-entity Labeled Diagnostic Benchmark.
Amazon ML Challenge 2026.

Requirements:
- Input: TRAINING Source-1, Source-2, Source-3, Ground Truth
- Output: data/diagnostic_200k/ (source1.tsv, source2.tsv, source3.tsv, ground_truth.tsv, sampling_manifest.json)
- Stratified deterministic sampling with RANDOM_SEED = 20260925
- Stratification dimensions:
  1. country (US, India)
  2. singleton / non-singleton
  3. number of true matches (0, 1, 2, 3, 4, 5+)
  4. S1->S2 vs S1->S3 (none, s2_only, s3_only, both)
  5. missing business_name
  6. missing business_address
  7. name length buckets (short, medium, long)
  8. address length buckets (short, medium, long)
  9. numeric address presence
  10. normalized exact-name availability
  11. transliteration / Unicode characteristics
  12. duplicate / near-duplicate name frequency
  13. difficult ground-truth cases
- Integrity constraint: EVERY true match for selected S1 entities MUST be present in S2/S3.
"""
from __future__ import annotations
import csv
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_TRAIN = ROOT / "data_raw" / "student_resource" / "dataset" / "train"
OUTPUT_DIR = ROOT / "data" / "diagnostic_200k"

RANDOM_SEED = 20260925
TARGET_SAMPLE_SIZE = 200000

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Build200kDiagnostic")

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_ALPHA = re.compile(r'[^a-z0-9]')
RE_DIGITS = re.compile(r'\b\d+\b')

LEGAL_SUFFIXES = [
    'private limited', 'pvt limited', 'p limited', 'private ltd', 'pvt ltd',
    'corporation', 'incorporated', 'limited', 'enterprises', 'enterprise',
    'services', 'solutions', 'technologies', 'holdings', 'industries',
    'international', 'consultants', 'consultancy', 'consulting', 'center',
    'centre', 'corp', 'inc', 'llc', 'llp', 'sarl', 'sasu', 'eurl', 'gmbh',
    'gie', 'sas', 'sa', 'ag', 'bv', 'nv', 'spa', 'srl', 'ltd'
]


def clean_text(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(' ', str(s).lower()).strip()


def extract_alpha_clean(raw_name: str) -> str:
    if not raw_name: return ""
    t = clean_text(raw_name)
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
        elif t == suf:
            t = ""
    return RE_ALPHA.sub('', t)


def get_stratum_key(country: str, name: str, addr: str, matched_ids: list[str],
                    common_tokens: set[str]) -> tuple:
    num_matches = len(matched_ids)
    is_singleton = num_matches == 0
    
    # 3. match count bucket
    match_bucket = min(num_matches, 5) # 0, 1, 2, 3, 4, 5+
    
    # 4. source presence
    if is_singleton:
        source_presence = 'none'
    else:
        has_s2 = any(x.startswith('S2-') for x in matched_ids)
        has_s3 = any(x.startswith('S3-') for x in matched_ids)
        if has_s2 and has_s3: source_presence = 'both'
        elif has_s2: source_presence = 's2_only'
        else: source_presence = 's3_only'

    # 5. missing business_name
    missing_name = not bool(name.strip())
    
    # 6. missing business_address
    missing_addr = not bool(addr.strip())
    
    # 7. name length bucket
    nl = len(name.strip())
    name_len_bucket = 'short' if nl < 15 else ('med' if nl < 35 else 'long')
    
    # 8. address length bucket
    al = len(addr.strip())
    addr_len_bucket = 'short' if al < 25 else ('med' if al < 60 else 'long')
    
    # 9. numeric address presence
    has_digits = bool(RE_DIGITS.search(addr))
    
    # 10. normalized exact-name availability
    alpha = extract_alpha_clean(name)
    exact_avail = len(alpha) >= 3
    
    # 11. transliteration / Unicode characteristics
    is_unicode = any(ord(c) > 127 for c in name + addr)
    
    # 12. duplicate / near-duplicate name frequency
    cn_words = clean_text(name).split()
    first_word = cn_words[0] if cn_words else ""
    is_common = first_word in common_tokens
    
    # 13. difficult ground-truth cases
    is_difficult = (is_singleton or match_bucket >= 5 or missing_addr or missing_name or is_unicode)

    return (
        country,
        is_singleton,
        match_bucket,
        source_presence,
        missing_name,
        missing_addr,
        name_len_bucket,
        addr_len_bucket,
        has_digits,
        exact_avail,
        is_unicode,
        is_common,
        is_difficult
    )


def main():
    logger.info("=" * 80)
    logger.info("  BUILDING 200,000-ENTITY LABELED DIAGNOSTIC BENCHMARK")
    logger.info("  Deterministic Stratified Sampling (Seed = %d)", RANDOM_SEED)
    logger.info("=" * 80)
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Ground Truth
    logger.info("Loading training ground truth from %s...", DATASET_TRAIN / "train_ground_truth.tsv")
    gt_map = {}
    with open(DATASET_TRAIN / "train_ground_truth.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline() # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            m_ids = [x.strip() for x in matched_str.split(",") if x.strip()]
            gt_map[s1_id] = m_ids
    logger.info("Loaded ground truth for %d S1 entities in %.2fs", len(gt_map), time.time() - t0)

    # 2. First Pass: Token frequencies to identify common name tokens (Dimension 12)
    logger.info("Analyzing token frequencies for stratification...")
    token_counter = Counter()
    with open(DATASET_TRAIN / "train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                words = clean_text(parts[1]).split()
                if words:
                    token_counter[words[0]] += 1
    # Top 500 most frequent first words
    common_tokens = set(w for w, cnt in token_counter.most_common(500))
    logger.info("Identified %d common first-word tokens", len(common_tokens))

    # 3. Read S1 records and group into strata
    logger.info("Reading Source-1 and assigning multi-dimensional strata...")
    t_strat = time.time()
    strata = defaultdict(list)
    s1_metadata = {}
    total_s1 = 0

    with open(DATASET_TRAIN / "train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline() # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                matched_ids = gt_map.get(eid, [])
                st_key = get_stratum_key(country, name, addr, matched_ids, common_tokens)
                strata[st_key].append(eid)
                s1_metadata[eid] = (name, addr, country, matched_ids)
                total_s1 += 1

    logger.info("Grouped %d entities into %d distinct strata in %.2fs",
                total_s1, len(strata), time.time() - t_strat)

    # 4. Deterministic Stratified Sampling
    rng = random.Random(RANDOM_SEED)
    sample_ratio = TARGET_SAMPLE_SIZE / total_s1
    selected_s1_ids = []
    
    # Sort strata for 100% determinism across platforms
    sorted_strata_keys = sorted(strata.keys(), key=lambda k: str(k))
    
    # Allocate quotas per stratum
    allocated = {}
    for st_key in sorted_strata_keys:
        members = strata[st_key]
        n_members = len(members)
        # Allocate proportionally, minimum 1 if n_members > 0
        quota = max(1, round(n_members * sample_ratio))
        quota = min(quota, n_members)
        allocated[st_key] = quota

    current_total = sum(allocated.values())
    diff = TARGET_SAMPLE_SIZE - current_total
    logger.info("Initial proportional allocation: %d (difference to target: %+d)", current_total, diff)

    # Fine-adjust allocation to exactly match TARGET_SAMPLE_SIZE
    if diff > 0:
        # Need to add 'diff' samples: add to largest strata with remaining members
        eligible = [k for k in sorted_strata_keys if allocated[k] < len(strata[k])]
        eligible.sort(key=lambda k: len(strata[k]) - allocated[k], reverse=True)
        for i in range(diff):
            allocated[eligible[i % len(eligible)]] += 1
    elif diff < 0:
        # Need to remove |diff| samples: remove from strata with quota > 1
        eligible = [k for k in sorted_strata_keys if allocated[k] > 1]
        eligible.sort(key=lambda k: allocated[k], reverse=True)
        for i in range(abs(diff)):
            allocated[eligible[i % len(eligible)]] -= 1

    assert sum(allocated.values()) == TARGET_SAMPLE_SIZE, f"Total allocated {sum(allocated.values())} != {TARGET_SAMPLE_SIZE}"

    for st_key in sorted_strata_keys:
        members = strata[st_key]
        quota = allocated[st_key]
        # Sort members by ID before sampling to guarantee 100% reproducibility
        members.sort()
        sampled = rng.sample(members, quota)
        selected_s1_ids.extend(sampled)

    # Sort selected S1 IDs for canonical deterministic ordering
    selected_s1_ids.sort()
    selected_s1_set = set(selected_s1_ids)
    assert len(selected_s1_ids) == TARGET_SAMPLE_SIZE

    logger.info("Successfully sampled %d Source-1 entities across %d strata", len(selected_s1_ids), len(allocated))

    # 5. Collect ALL required true matches (S2 and S3)
    needed_s2_ids = set()
    needed_s3_ids = set()
    s1_gt_records = []
    
    country_counts = Counter()
    match_count_dist = Counter()
    singleton_count = 0

    for s1_id in selected_s1_ids:
        name, addr, country, matched_ids = s1_metadata[s1_id]
        country_counts[country] += 1
        k = len(matched_ids)
        match_count_dist[min(k, 5)] += 1
        if k == 0:
            singleton_count += 1
        for mid in matched_ids:
            if mid.startswith("S2-"):
                needed_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                needed_s3_ids.add(mid)
        s1_gt_records.append((s1_id, ",".join(matched_ids)))

    logger.info("Diagnostic Benchmark Statistics:")
    logger.info("  Total S1 entities:      %d", len(selected_s1_ids))
    logger.info("  US entities:            %d (%.1f%%)", country_counts['US'], country_counts['US'] / TARGET_SAMPLE_SIZE * 100)
    logger.info("  India entities:         %d (%.1f%%)", country_counts['India'], country_counts['India'] / TARGET_SAMPLE_SIZE * 100)
    logger.info("  Singletons (0 matches): %d (%.1f%%)", singleton_count, singleton_count / TARGET_SAMPLE_SIZE * 100)
    logger.info("  Non-singletons:         %d (%.1f%%)", TARGET_SAMPLE_SIZE - singleton_count, (TARGET_SAMPLE_SIZE - singleton_count) / TARGET_SAMPLE_SIZE * 100)
    logger.info("  True S2 matches needed: %d", len(needed_s2_ids))
    logger.info("  True S3 matches needed: %d", len(needed_s3_ids))

    # 6. Write diagnostic source1.tsv and ground_truth.tsv
    s1_out_path = OUTPUT_DIR / "source1.tsv"
    gt_out_path = OUTPUT_DIR / "ground_truth.tsv"

    logger.info("Writing %s...", s1_out_path.name)
    with open(s1_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        for s1_id in selected_s1_ids:
            name, addr, country, _ = s1_metadata[s1_id]
            writer.writerow([s1_id, name, addr, country])

    logger.info("Writing %s...", gt_out_path.name)
    with open(gt_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id, m_str in s1_gt_records:
            writer.writerow([s1_id, m_str])

    # 7. Extract S2 and S3: Include ALL true matches + stratified distractors (~1M total each)
    # Target pool: ~1.2M S2 and ~1.2M S3 records (including 100% of true matches)
    DISTRACTOR_RATIO = 0.20 # Take 20% of full S2 and S3 to provide ~1M-1.2M records each

    for src_name, src_file, needed_ids in [
        ("source2", DATASET_TRAIN / "train_source2.tsv", needed_s2_ids),
        ("source3", DATASET_TRAIN / "train_source3.tsv", needed_s3_ids)
    ]:
        out_path = OUTPUT_DIR / f"{src_name}.tsv"
        logger.info("Extracting %s -> %s (preserving 100%% of %d true matches)...",
                    src_file.name, out_path.name, len(needed_ids))
        t_src = time.time()
        
        found_true_matches = set()
        written_count = 0
        distractor_rng = random.Random(RANDOM_SEED + (2 if src_name == 'source2' else 3))

        with open(src_file, "r", encoding="utf-8", errors="ignore") as f_in, \
             open(out_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            f_in.readline() # header

            for line in f_in:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    cid, cname, caddr, ccountry = parts[0], parts[1], parts[2], parts[3]
                    is_true_match = cid in needed_ids
                    if is_true_match:
                        found_true_matches.add(cid)
                        writer.writerow([cid, cname, caddr, ccountry])
                        written_count += 1
                    else:
                        # Include as distractor with probability DISTRACTOR_RATIO
                        if distractor_rng.random() < DISTRACTOR_RATIO:
                            writer.writerow([cid, cname, caddr, ccountry])
                            written_count += 1

        logger.info("Saved %s: %d total records (%d true matches, %d distractors) in %.2fs",
                    out_path.name, written_count, len(found_true_matches),
                    written_count - len(found_true_matches), time.time() - t_src)

        # MANDATORY INTEGRITY ASSERTION (Rule Phase 2)
        missing_matches = needed_ids - found_true_matches
        if missing_matches:
            logger.error("FATAL INTEGRITY VIOLATION: %d true matches missing from %s! Examples: %s",
                         len(missing_matches), out_path.name, list(missing_matches)[:5])
            sys.exit(1)
        else:
            logger.info("INTEGRITY CHECK PASSED: 100.00%% of required %s matches are present!", src_name)

    # 8. Create Sampling Manifest
    manifest = {
        "benchmark_name": "AMAZON_ML_2026_DIAGNOSTIC_200K",
        "random_seed": RANDOM_SEED,
        "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_source1_entities": len(selected_s1_ids),
        "countries": dict(country_counts),
        "singleton_count": singleton_count,
        "singleton_rate": singleton_count / TARGET_SAMPLE_SIZE,
        "match_count_distribution": dict(match_count_dist),
        "true_matches_s2_count": len(needed_s2_ids),
        "true_matches_s3_count": len(needed_s3_ids),
        "total_true_matches": len(needed_s2_ids) + len(needed_s3_ids),
        "strata_count": len(allocated),
        "source1_file": "source1.tsv",
        "source2_file": "source2.tsv",
        "source3_file": "source3.tsv",
        "ground_truth_file": "ground_truth.tsv",
        "integrity_verified": True
    }

    manifest_path = OUTPUT_DIR / "sampling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 80)
    logger.info("200K DIAGNOSTIC BENCHMARK COMPLETE!")
    logger.info("  Source 1:      %s (%d entities)", s1_out_path, len(selected_s1_ids))
    logger.info("  Ground Truth:  %s (%d entries)", gt_out_path, len(selected_s1_ids))
    logger.info("  Source 2:      %s", OUTPUT_DIR / "source2.tsv")
    logger.info("  Source 3:      %s", OUTPUT_DIR / "source3.tsv")
    logger.info("  Manifest:      %s", manifest_path)
    logger.info("  Total Time:    %.2fs", time.time() - t0)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
