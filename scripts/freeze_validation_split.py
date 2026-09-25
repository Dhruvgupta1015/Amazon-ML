"""
freeze_validation_split.py
Freezes a deterministic, leak-free validation partition of 20,000 Source 1 entities.
Saves entity IDs, split metadata, and dataset hash to data/frozen_val_s1_ids.json.
Guarantees that every model (Champion and Challengers) is evaluated on the exact same population.
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
S1_PATH = DATASET_ROOT / "train" / "train_source1.tsv"
GT_PATH = DATASET_ROOT / "train" / "train_ground_truth.tsv"

RANDOM_SEED = 42
TARGET_VAL_SIZE = 20000

def main():
    print("=" * 60)
    print("FREEZING VALIDATION SPLIT (RULE 2: NO-REGRESSION PROTOCOL)")
    print("=" * 60)

    # 1. Compute Dataset Hash
    hasher = hashlib.sha256()
    with open(S1_PATH, "rb") as f:
        hasher.update(f.read(1024 * 1024))
    dataset_hash = f"sha256_{hasher.hexdigest()[:16]}"
    print(f"Dataset Hash (train_source1.tsv prefix): {dataset_hash}")

    # 2. Load Ground Truth
    print("Loading Ground Truth...")
    gt = {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].strip():
                gt[parts[0]] = parts[1].split(",")
            else:
                gt[parts[0]] = []

    # 3. Deterministic Entity Split (hash(eid) % 100 >= 80 -> Validation)
    print(f"Sampling {TARGET_VAL_SIZE:,} Validation S1 Entities (Deterministic Hash Split)...")
    val_s1_ids = []
    singleton_count = 0
    non_singleton_count = 0
    total_true_mentions = 0

    with open(S1_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            eid = r["entity_id"]
            h = int(hashlib.md5(eid.encode("utf-8")).hexdigest()[:8], 16) % 100
            if h >= 80:  # 20% hold-out partition
                val_s1_ids.append(eid)
                true_matches = gt.get(eid, [])
                if true_matches:
                    non_singleton_count += 1
                    total_true_mentions += len(true_matches)
                else:
                    singleton_count += 1

                if len(val_s1_ids) >= TARGET_VAL_SIZE:
                    break

    print(f"Frozen Validation Set Constructed:")
    print(f"  - Total S1 Entities:     {len(val_s1_ids):,}")
    print(f"  - Singletons:            {singleton_count:,} ({singleton_count/len(val_s1_ids)*100:.1f}%)")
    print(f"  - Non-Singletons:        {non_singleton_count:,} ({non_singleton_count/len(val_s1_ids)*100:.1f}%)")
    print(f"  - Total Target Mentions: {total_true_mentions:,}")

    # 4. Save to data/frozen_val_s1_ids.json
    frozen_manifest = {
        "split_version": "v1.0-frozen-20k",
        "dataset_hash": dataset_hash,
        "random_seed": RANDOM_SEED,
        "sample_size": len(val_s1_ids),
        "singleton_count": singleton_count,
        "non_singleton_count": non_singleton_count,
        "total_true_mentions": total_true_mentions,
        "s1_entity_ids": val_s1_ids
    }

    out_path = DATA_DIR / "frozen_val_s1_ids.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frozen_manifest, f, indent=2)

    print(f"\n[OK] Validation split successfully frozen at: {out_path}")

if __name__ == "__main__":
    main()
