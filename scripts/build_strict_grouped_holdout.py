#!/usr/bin/env python3
"""
build_strict_grouped_holdout.py — Creates a strictly disjoint, grouped partition
of the 200K diagnostic benchmark into:
  - 140,000 TRAIN entities
  - 30,000 CALIBRATION entities
  - 30,000 FINAL HOLDOUT entities

Grouping unit: SOURCE-1 ENTITY ID.
Zero overlap across partitions. Stratified by country and match count.
Seed: 20260925
"""
import json
import random
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 20260925
ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostic_200k"
REPORT_DIR = ROOT / "reports" / "recovery"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading 200K diagnostic S1 entities and ground truth...")
    s1_path = DIAG_DIR / "source1.tsv"
    gt_path = DIAG_DIR / "ground_truth.tsv"

    # Map S1 -> matches
    s1_matches = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline() # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_matches[parts[0]] = parts[1].split(",")

    # Map S1 -> country & strata
    s1_records = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        c_idx = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[id_idx]
            country = parts[c_idx]
            n_match = len(s1_matches.get(eid, []))
            stratum = f"{country}_m{min(n_match, 6)}"
            s1_records.append((eid, stratum, country, n_match))

    assert len(s1_records) == 200000, f"Expected 200,000 entities, got {len(s1_records)}"

    # Group by stratum
    strata_groups = defaultdict(list)
    for rec in s1_records:
        strata_groups[rec[1]].append(rec[0])

    rng = random.Random(RANDOM_SEED)

    train_ids = []
    calib_ids = []
    holdout_ids = []

    # Proportions: 140K / 200K = 0.70, 30K / 200K = 0.15, 30K / 200K = 0.15
    for stratum, ids in sorted(strata_groups.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * 0.70))
        n_calib = int(round(n * 0.15))
        # remainder to holdout
        train_ids.extend(ids[:n_train])
        calib_ids.extend(ids[n_train:n_train + n_calib])
        holdout_ids.extend(ids[n_train + n_calib:])

    # Adjust exact counts if rounding difference
    all_assigned = [(eid, "train") for eid in train_ids] + \
                   [(eid, "calib") for eid in calib_ids] + \
                   [(eid, "holdout") for eid in holdout_ids]
    
    # Balance to exact 140,000 / 30,000 / 30,000
    rng.shuffle(train_ids)
    rng.shuffle(calib_ids)
    rng.shuffle(holdout_ids)

    # If calib or holdout needs trim/add
    while len(train_ids) > 140000:
        if len(calib_ids) < 30000:
            calib_ids.append(train_ids.pop())
        elif len(holdout_ids) < 30000:
            holdout_ids.append(train_ids.pop())
        else:
            break

    while len(train_ids) < 140000:
        if len(calib_ids) > 30000:
            train_ids.append(calib_ids.pop())
        elif len(holdout_ids) > 30000:
            train_ids.append(holdout_ids.pop())
        else:
            break

    while len(calib_ids) > 30000:
        holdout_ids.append(calib_ids.pop())
    while len(calib_ids) < 30000 and len(holdout_ids) > 30000:
        calib_ids.append(holdout_ids.pop())

    # STRICT ASSERTIONS
    assert len(train_ids) == 140000, f"Train count {len(train_ids)} != 140,000"
    assert len(calib_ids) == 30000, f"Calib count {len(calib_ids)} != 30,000"
    assert len(holdout_ids) == 30000, f"Holdout count {len(holdout_ids)} != 30,000"

    s_train, s_calib, s_hold = set(train_ids), set(calib_ids), set(holdout_ids)
    assert len(s_train & s_calib) == 0, "Leakage: Train and Calib overlap!"
    assert len(s_train & s_hold) == 0, "Leakage: Train and Holdout overlap!"
    assert len(s_calib & s_hold) == 0, "Leakage: Calib and Holdout overlap!"
    assert len(s_train | s_calib | s_hold) == 200000, "Incomplete partition!"

    print("Strict Grouped Holdout Partition Verified:")
    print(f"  Train:        {len(train_ids):,} entities (140,000)")
    print(f"  Calibration:  {len(calib_ids):,} entities (30,000)")
    print(f"  Final Holdout:{len(holdout_ids):,} entities (30,000)")
    print(f"  Pairwise intersection: 0 entities (100% disjoint)")

    # Save IDs
    with open(DIAG_DIR / "split_train_140k_s1_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f)
    with open(DIAG_DIR / "split_calib_30k_s1_ids.json", "w", encoding="utf-8") as f:
        json.dump(calib_ids, f)
    with open(DIAG_DIR / "split_holdout_30k_s1_ids.json", "w", encoding="utf-8") as f:
        json.dump(holdout_ids, f)

    # Save manifest
    manifest = {
        "benchmark": "AMAZON_ML_2026_STRICT_GROUPED_HOLDOUT",
        "random_seed": RANDOM_SEED,
        "train_count": len(train_ids),
        "calib_count": len(calib_ids),
        "holdout_count": len(holdout_ids),
        "total_entities": 200000,
        "grouping_unit": "source1_entity_id",
        "pairwise_overlap": 0,
        "train_file": "split_train_140k_s1_ids.json",
        "calib_file": "split_calib_30k_s1_ids.json",
        "holdout_file": "split_holdout_30k_s1_ids.json"
    }
    with open(REPORT_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {REPORT_DIR / 'split_manifest.json'}")

if __name__ == "__main__":
    main()
