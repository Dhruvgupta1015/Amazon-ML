import os
import sys
import json
import hashlib
import time
from collections import Counter
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_raw", "student_resource", "dataset")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def compute_sha256(filepath, max_bytes=64*1024*1024):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        chunk = f.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()

def json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def analyze_file(filepath):
    print(f"Profiling {os.path.basename(filepath)}...", flush=True)
    size_bytes = os.path.getsize(filepath)
    file_hash = compute_sha256(filepath)
    
    total_rows = 0
    missing_counts = {}
    country_counts = Counter()
    duplicate_ids = 0
    seen_ids = set()
    sample_rows = []
    
    chunk_size = 250000
    is_gt = "ground_truth" in filepath
    
    for chunk in pd.read_csv(filepath, sep='\t', chunksize=chunk_size, dtype=str, on_bad_lines='skip'):
        if total_rows == 0:
            missing_counts = {col: 0 for col in chunk.columns}
            sample_rows = chunk.head(3).to_dict(orient='records')
            
        total_rows += len(chunk)
        for col in chunk.columns:
            missing_counts[col] += int(chunk[col].isna().sum())
            
        id_col = 'source1_entity_id' if is_gt else 'entity_id'
        if id_col in chunk.columns:
            chunk_ids = chunk[id_col].dropna().tolist()
            for cid in chunk_ids:
                if cid in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(cid)
                    
        if 'country' in chunk.columns:
            for c, cnt in chunk['country'].value_counts().items():
                country_counts[str(c)] += int(cnt)
                
    return {
        "filename": os.path.basename(filepath),
        "path": filepath,
        "size_bytes": int(size_bytes),
        "sha256_prefix": file_hash,
        "total_records": int(total_rows),
        "unique_ids": int(len(seen_ids)),
        "duplicate_ids": int(duplicate_ids),
        "columns": list(missing_counts.keys()),
        "missing_values": {k: int(v) for k, v in missing_counts.items()},
        "country_distribution": {k: int(v) for k, v in country_counts.items()},
        "sample": sample_rows
    }

def analyze_ground_truth(filepath, s1_total_count):
    print(f"Analyzing ground truth distribution (optimized)...", flush=True)
    empty_matches = 0
    match_lengths = Counter()
    source2_mentions = 0
    source3_mentions = 0
    s1_with_matches = 0
    num_gt_rows = 0
    max_m = 0
    
    # Read line-by-line directly for maximum speed and minimal memory
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            num_gt_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) < 2 or not parts[1].strip():
                empty_matches += 1
                match_lengths[0] += 1
            else:
                m_str = parts[1].strip()
                tokens = [t.strip() for t in m_str.split(',') if t.strip()]
                cnt = len(tokens)
                if cnt > 0:
                    s1_with_matches += 1
                    match_lengths[cnt] += 1
                    if cnt > max_m:
                        max_m = cnt
                    for t in tokens:
                        if t.startswith('S2-'):
                            source2_mentions += 1
                        elif t.startswith('S3-'):
                            source3_mentions += 1
                            
    singletons_in_gt = empty_matches
    singletons_absent_from_gt = max(0, s1_total_count - num_gt_rows)
    total_singletons = singletons_in_gt + singletons_absent_from_gt
    
    return {
        "gt_rows": int(num_gt_rows),
        "s1_with_positive_matches": int(s1_with_matches),
        "gt_explicit_singletons": int(empty_matches),
        "s1_absent_from_gt_singletons": int(singletons_absent_from_gt),
        "total_effective_singletons": int(total_singletons),
        "singleton_percentage": round((total_singletons / s1_total_count) * 100, 2) if s1_total_count else 0.0,
        "match_length_histogram": {str(k): int(v) for k, v in sorted(match_lengths.items())[:15]},
        "max_matches_single_entity": int(max_m),
        "total_source2_matched_mentions": int(source2_mentions),
        "total_source3_matched_mentions": int(source3_mentions),
        "total_matched_mentions": int(source2_mentions + source3_mentions)
    }

def main():
    t0 = time.time()
    print("=" * 60)
    print("PHASE 2: COMPREHENSIVE DATASET AND SCHEMA PROFILING")
    print("=" * 60)
    
    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "test")
    
    train_files = {
        "train_source1": os.path.join(train_dir, "train_source1.tsv"),
        "train_source2": os.path.join(train_dir, "train_source2.tsv"),
        "train_source3": os.path.join(train_dir, "train_source3.tsv"),
        "train_ground_truth": os.path.join(train_dir, "train_ground_truth.tsv")
    }
    
    test_files = {
        "test_source1": os.path.join(test_dir, "test_source1.tsv"),
        "test_source2": os.path.join(test_dir, "test_source2.tsv"),
        "test_source3": os.path.join(test_dir, "test_source3.tsv")
    }
    
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_root": DATA_ROOT,
        "files": {},
        "ground_truth_profile": {}
    }
    
    for name, path in {**train_files, **test_files}.items():
        if os.path.exists(path):
            results["files"][name] = analyze_file(path)
        else:
            print(f"Warning: {path} not found.")
            
    train_s1_count = results["files"].get("train_source1", {}).get("total_records", 0)
    gt_path = train_files["train_ground_truth"]
    if os.path.exists(gt_path):
        results["ground_truth_profile"] = analyze_ground_truth(gt_path, train_s1_count)
        
    duration = time.time() - t0
    results["profiling_duration_sec"] = round(duration, 2)
    
    # Save JSON
    json_path = os.path.join(REPORTS_DIR, "dataset_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_default)
    print(f"Saved JSON profile to {json_path}")
    
    # Save Markdown
    md_path = os.path.join(REPORTS_DIR, "dataset_profile.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RESOLVE.AI: Dataset & Schema Verification Report\n\n")
        f.write(f"**Generated**: {results['timestamp']}  \n")
        f.write(f"**Profiling Runtime**: {results['profiling_duration_sec']} seconds  \n\n")
        f.write("---\n\n")
        f.write("## 1. File Integrity & Record Counts\n\n")
        f.write("| Dataset Split | File Name | Size (MB) | Total Records | Unique IDs | Duplicate IDs | SHA-256 (64MB Prefix) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for k, v in results["files"].items():
            split = "TRAIN" if "train" in k else "TEST"
            size_mb = round(v["size_bytes"] / (1024 * 1024), 2)
            f.write(f"| **{split}** | `{v['filename']}` | {size_mb} MB | {v['total_records']:,} | {v['unique_ids']:,} | {v['duplicate_ids']} | `{v['sha256_prefix'][:16]}...` |\n")
            
        f.write("\n---\n\n")
        f.write("## 2. Country Distribution\n\n")
        f.write("| Dataset Split | File Name | United States (US) | India (India) | France (France) | Other / Missing |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for k, v in results["files"].items():
            if "ground_truth" in k:
                continue
            cd = v.get("country_distribution", {})
            us_cnt = cd.get("US", 0)
            in_cnt = cd.get("India", 0)
            fr_cnt = cd.get("France", 0)
            other = sum(c for name, c in cd.items() if name not in ("US", "India", "France"))
            f.write(f"| {'TRAIN' if 'train' in k else 'TEST'} | `{v['filename']}` | {us_cnt:,} | {in_cnt:,} | {fr_cnt:,} | {other} |\n")
            
        f.write("\n---\n\n")
        f.write("## 3. Schema & Missing Value Analysis\n\n")
        for k, v in results["files"].items():
            f.write(f"### `{v['filename']}`\n")
            f.write(f"- **Columns**: `{', '.join(v['columns'])}`\n")
            f.write("- **Missing Values**:\n")
            for col, count in v["missing_values"].items():
                pct = round((count / v["total_records"]) * 100, 2) if v["total_records"] else 0
                f.write(f"  - `{col}`: {count:,} ({pct}%)\n")
            f.write("\n")
            
        f.write("---\n\n")
        f.write("## 4. Ground Truth Match & Singleton Cardinality\n\n")
        gt = results.get("ground_truth_profile", {})
        f.write(f"- **Total Rows in Ground Truth TSV**: {gt.get('gt_rows', 0):,}\n")
        f.write(f"- **Source 1 Entities with Positive Matches**: {gt.get('s1_with_positive_matches', 0):,}\n")
        f.write(f"- **Explicit Singletons in Ground Truth**: {gt.get('gt_explicit_singletons', 0):,}\n")
        f.write(f"- **Implicit Singletons (Absent from Ground Truth)**: {gt.get('s1_absent_from_gt_singletons', 0):,}\n")
        f.write(f"- **Total Effective Singletons**: {gt.get('total_effective_singletons', 0):,} ({gt.get('singleton_percentage', 0)}% of S1 entities)\n")
        f.write(f"- **Total Positive Mentions Matched**: {gt.get('total_matched_mentions', 0):,}\n")
        f.write(f"  - Source 2 Mentions: {gt.get('total_source2_matched_mentions', 0):,}\n")
        f.write(f"  - Source 3 Mentions: {gt.get('total_source3_matched_mentions', 0):,}\n")
        f.write(f"- **Max Matches for Single S1 Entity**: {gt.get('max_matches_single_entity', 0)}\n\n")
        f.write("### Match Cardinality Distribution (Positive Mentions per S1 Entity):\n\n")
        f.write("| Matches per Entity | Number of S1 Entities |\n")
        f.write("| :--- | :--- |\n")
        for k, cnt in gt.get("match_length_histogram", {}).items():
            f.write(f"| {k} | {cnt:,} |\n")
            
    print(f"Saved Markdown report to {md_path}")
    print("PHASE 2 COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
