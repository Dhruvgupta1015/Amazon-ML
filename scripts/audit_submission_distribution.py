#!/usr/bin/env python3
"""
audit_submission_distribution.py — Audits matching_results.tsv against the
training ground truth distribution across:
  - Total S1 count
  - Empty predictions (singletons)
  - Non-empty predictions
  - Mean, Median, 95th, 99th, Max matches per S1
  - Entities with >= 5, >= 10, >= 20 matches
  - Top 500 highest-cardinality S1 entities flag
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def analyze_tsv(path: Path) -> dict:
    counts = []
    cardinality_records = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_col = header[0]
        match_col = header[1] if len(header) > 1 else "matched_entity_ids"
        for line in f:
            parts = line.strip("\r\n").split("\t")
            eid = parts[0]
            if len(parts) <= 1 or not parts[1].strip():
                counts.append(0)
            else:
                m_list = [x.strip() for x in parts[1].split(",") if x.strip()]
                c = len(m_list)
                counts.append(c)
                if c >= 10:
                    cardinality_records.append((eid, c, parts[1]))

    arr = np.array(counts)
    total = len(arr)
    empty = int((arr == 0).sum())
    non_empty = int((arr > 0).sum())

    # Sort top cardinality
    cardinality_records.sort(key=lambda x: x[1], reverse=True)
    top_500 = cardinality_records[:500]

    return {
        "file": str(path.name),
        "total_s1": total,
        "empty_predictions": empty,
        "empty_pct": float((empty / total) * 100) if total else 0.0,
        "nonempty_predictions": non_empty,
        "nonempty_pct": float((non_empty / total) * 100) if total else 0.0,
        "total_links": int(arr.sum()),
        "mean_matches": float(arr.mean()),
        "median_matches": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max_matches": int(arr.max()) if total else 0,
        "count_ge_5": int((arr >= 5).sum()),
        "pct_ge_5": float((arr >= 5).mean() * 100),
        "count_ge_10": int((arr >= 10).sum()),
        "pct_ge_10": float((arr >= 10).mean() * 100),
        "count_ge_20": int((arr >= 20).sum()),
        "pct_ge_20": float((arr >= 20).mean() * 100),
        "top_500_highest": [{"id": x[0], "count": x[1]} for x in top_500]
    }

def main():
    parser = argparse.ArgumentParser(description="Audit matching_results.tsv distribution")
    parser.add_argument("--submission", default="output/matching_results.tsv", help="Path to submission TSV")
    parser.add_argument("--output-json", default="reports/recovery/submission_audit.json", help="Output JSON path")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"File {sub_path} not found.")
        return

    print(f"Auditing {sub_path}...")
    stats = analyze_tsv(sub_path)

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("=" * 70)
    print("SUBMISSION DISTRIBUTION AUDIT:")
    print("=" * 70)
    print(f"  Total S1 Entities:     {stats['total_s1']:,}")
    print(f"  Empty (Singletons):    {stats['empty_predictions']:,} ({stats['empty_pct']:.2f}%)")
    print(f"  Non-empty Predictions: {stats['nonempty_predictions']:,} ({stats['nonempty_pct']:.2f}%)")
    print(f"  Total Predicted Links: {stats['total_links']:,}")
    print(f"  Mean Matches per S1:   {stats['mean_matches']:.4f}")
    print(f"  Median Matches:        {stats['median_matches']:.1f}")
    print(f"  95th Percentile:       {stats['p95']:.1f}")
    print(f"  99th Percentile:       {stats['p99']:.1f}")
    print(f"  Max Matches:           {stats['max_matches']}")
    print(f"  Matches >= 5:          {stats['count_ge_5']:,} ({stats['pct_ge_5']:.2f}%)")
    print(f"  Matches >= 10:         {stats['count_ge_10']:,} ({stats['pct_ge_10']:.2f}%)")
    print(f"  Matches >= 20:         {stats['count_ge_20']:,} ({stats['pct_ge_20']:.2f}%)")
    print("=" * 70)
    print(f"Audit written to {out_p}")

if __name__ == "__main__":
    main()
