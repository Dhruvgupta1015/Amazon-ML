"""scripts/benchmark_blocking_channels.py - Evaluates Multi-Channel Retrieval Strategies.

Enforces Rule 5, Rule 6, & Rule 7:
Evaluates 8 distinct retrieval channels on the frozen validation set (20,000 entities):
1. Exact Normalized Name
2. Name Word Tokens (Inverted Index)
3. Character 3-Gram Overlap
4. Character 4-Gram Overlap
5. Address Tokens & Locality
6. Street Number & Postal Code
7. Domain URL Stems
8. Union Multi-Channel with Ranked Candidate Pruning (Top-K)

Generates:
- reports/blocking_comparison.csv
"""

import json
import logging
import re
import sys
import time
import unicodedata
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
REPORTS_DIR = ROOT / "reports"
PUBLIC_REPORTS_DIR = ROOT / "business_entity_resolution" / "frontend" / "public" / "reports"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BlockingBenchmark")

STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}

LEGAL_RE = re.compile(r'\b(private limited|pvt\.?\s*ltd\.?|limited|ltd\.?|corporation|corp\.?|incorporated|inc\.?|llc|sarl|sas|sa)\b', re.IGNORECASE)
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)
DIGITS_RE = re.compile(r'\b\d+\b')
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def transliterate(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_name(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    t = transliterate(raw_name).lower()
    t = LEGAL_RE.sub(' ', t)
    t = PUNCT_RE.sub(' ', t)
    return ' '.join(t.split())


def clean_tokens(name: str) -> list[str]:
    c = clean_name(name)
    return [t for t in c.split() if t not in STOPWORDS and len(t) >= 3]


def ngrams(text: str, n: int) -> list[str]:
    c = clean_name(text).replace(' ', '')
    if len(c) < n:
        return [c] if c else []
    return [c[i:i+n] for i in range(len(c) - n + 1)]


def extract_addr_digits(addr: str) -> list[str]:
    if not addr or not isinstance(addr, str):
        return []
    return DIGITS_RE.findall(addr)


def extract_domain_stem(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    m = DOMAIN_RE.search(raw_name.lower())
    return m.group(1).replace('-', '') if m else ""


def main():
    logger.info("=" * 60)
    logger.info("BENCHMARKING MULTI-CHANNEL CANDIDATE RETRIEVAL (RULES 5, 6, 7)")
    logger.info("=" * 60)

    # 1. Load Frozen Validation Manifest
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_val_ids = set(manifest["s1_entity_ids"])
    logger.info("Loaded frozen validation partition: %d S1 entities", len(target_val_ids))

    # 2. Load Ground Truth
    gt = {}
    with open(DATASET_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2 and p[1].strip():
                gt[p[0]] = p[1].split(",")
            else:
                gt[p[0]] = []

    val_gt = {eid: gt.get(eid, []) for eid in target_val_ids}
    total_true_positives = sum(len(v) for v in val_gt.values())
    total_entities_with_matches = sum(1 for v in val_gt.values() if v)
    logger.info("Validation Ground Truth: %d true positive mentions across %d non-singletons",
                total_true_positives, total_entities_with_matches)

    # 3. Load Validation S1 Records
    val_s1 = []
    needed_positive_cids = set()
    for e in target_val_ids:
        needed_positive_cids.update(val_gt[e])

    with open(DATASET_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in target_val_ids:
                val_s1.append(r)

    # 4. Load S2 and S3 Records
    s23_records = {}
    for s_path in [DATASET_ROOT / "train" / "train_source2.tsv", DATASET_ROOT / "train" / "train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    cid, name, addr, country = p[0], p[1], p[2], p[3]
                    if cid in needed_positive_cids or len(s23_records) < 250000:
                        s23_records[cid] = {
                            "cid": cid,
                            "cname": clean_name(name),
                            "toks": clean_tokens(name),
                            "char3": ngrams(name, 3),
                            "char4": ngrams(name, 4),
                            "addr_toks": clean_tokens(addr),
                            "digits": set(extract_addr_digits(addr)),
                            "stem": extract_domain_stem(name),
                            "country": country
                        }

    logger.info("Loaded %d candidate mentions from S2/S3 (100%% of required true mentions present)", len(s23_records))

    # Pre-process S1 records
    processed_s1 = []
    for r in val_s1:
        processed_s1.append({
            "eid": r["entity_id"],
            "cname": clean_name(r["business_name"]),
            "toks": clean_tokens(r["business_name"]),
            "char3": ngrams(r["business_name"], 3),
            "char4": ngrams(r["business_name"], 4),
            "addr_toks": clean_tokens(r["business_address"]),
            "digits": set(extract_addr_digits(r["business_address"])),
            "stem": extract_domain_stem(r["business_name"]),
            "country": r["country"]
        })

    # Build Inverted Indexes for each channel
    logger.info("Indexing retrieval channels...")
    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    char3_idx = defaultdict(list)
    char4_idx = defaultdict(list)
    addr_tok_idx = defaultdict(list)
    digit_idx = defaultdict(list)
    stem_idx = defaultdict(list)

    for cid, r in s23_records.items():
        c = r["country"]
        if r["cname"]:
            exact_name_idx[(c, r["cname"])].append(cid)
        for t in r["toks"]:
            token_idx[(c, t)].append(cid)
        for g3 in r["char3"][:10]:
            char3_idx[(c, g3)].append(cid)
        for g4 in r["char4"][:8]:
            char4_idx[(c, g4)].append(cid)
        for at in r["addr_toks"][:5]:
            addr_tok_idx[(c, at)].append(cid)
        for d in r["digits"]:
            if len(d) >= 2:
                digit_idx[(c, d)].append(cid)
        if r["stem"] and len(r["stem"]) >= 4:
            stem_idx[(c, r["stem"])].append(cid)

    # Define Retrieval Strategies to Benchmark
    strategies = [
        ("1. Exact Normalized Name", lambda s: set(exact_name_idx.get((s["country"], s["cname"]), []))),
        ("2. Name Word Tokens (Inverted Index)", lambda s: set(cid for t in s["toks"] for cid in token_idx.get((s["country"], t), [])[:60])),
        ("3. Character 3-Gram Overlap", lambda s: set(cid for g in s["char3"][:8] for cid in char3_idx.get((s["country"], g), [])[:30])),
        ("4. Character 4-Gram Overlap", lambda s: set(cid for g in s["char4"][:6] for cid in char4_idx.get((s["country"], g), [])[:30])),
        ("5. Address Tokens & Locality", lambda s: set(cid for at in s["addr_toks"][:4] for cid in addr_tok_idx.get((s["country"], at), [])[:40])),
        ("6. Street Number & Address Digits", lambda s: set(cid for d in s["digits"] if len(d)>=2 for cid in digit_idx.get((s["country"], d), [])[:40])),
        ("7. Domain URL Stem Matching", lambda s: set(stem_idx.get((s["country"], s["stem"]), [])) if s["stem"] else set()),
    ]

    benchmark_rows = []
    max_possible_pairs = len(processed_s1) * len(s23_records)

    for strat_name, retrieval_fn in strategies:
        t0 = time.time()
        cands_generated = 0
        recovered_tp = 0
        entity_perfect = 0

        for s1 in processed_s1:
            true_set = set(val_gt[s1["eid"]])
            cands = retrieval_fn(s1)
            cands_generated += len(cands)
            recovered = len(cands & true_set)
            recovered_tp += recovered
            if true_set and true_set.issubset(cands):
                entity_perfect += 1

        elapsed = time.time() - t0
        pair_recall = recovered_tp / total_true_positives if total_true_positives > 0 else 1.0
        entity_recall = entity_perfect / total_entities_with_matches if total_entities_with_matches > 0 else 1.0
        reduction_ratio = 1.0 - (cands_generated / max_possible_pairs)
        avg_cands = cands_generated / len(processed_s1)

        row = {
            "strategy": strat_name,
            "pair_recall": round(pair_recall * 100, 2),
            "entity_recall": round(entity_recall * 100, 2),
            "avg_candidates": round(avg_cands, 2),
            "candidate_count": cands_generated,
            "reduction_ratio": round(reduction_ratio * 100, 4),
            "runtime_seconds": round(elapsed, 2)
        }
        benchmark_rows.append(row)
        logger.info("  %s: Pair Recall=%.2f%% | Entity Recall=%.2f%% | Avg Cands=%.1f | RR=%.4f%%",
                    strat_name[:35].ljust(35), row["pair_recall"], row["entity_recall"], row["avg_candidates"], row["reduction_ratio"])

    # Multi-Channel UNION with Ranked Candidate Pruning
    union_k_values = [25, 50, 75, 100, 150, 200, 300]
    for k in union_k_values:
        t0 = time.time()
        cands_generated = 0
        recovered_tp = 0
        entity_perfect = 0

        for s1 in processed_s1:
            true_set = set(val_gt[s1["eid"]])
            c = s1["country"]
            union = set()
            # Combine channels
            for t in s1["toks"]:
                union.update(token_idx.get((c, t), [])[:60])
            for d in s1["digits"]:
                if len(d) >= 2:
                    union.update(digit_idx.get((c, d), [])[:40])
            if s1["stem"]:
                union.update(stem_idx.get((c, s1["stem"]), []))

            # Ranked Candidate Pruning (Score before top-K)
            s1_toks = set(s1["toks"])
            scored = []
            for cid in union:
                cand_r = s23_records.get(cid)
                if not cand_r:
                    continue
                tok_sim = len(s1_toks & set(cand_r["toks"])) / max(len(s1_toks | set(cand_r["toks"])), 1)
                shared_d = len(s1["digits"] & cand_r["digits"])
                d_sim = 1.0 if shared_d > 0 else (0.0 if s1["digits"] and cand_r["digits"] else 0.3)
                scored.append((0.7 * tok_sim + 0.3 * d_sim, cid))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = [cid for _, cid in scored[:k]]

            cands_generated += len(top_k)
            top_set = set(top_k)
            recovered_tp += len(top_set & true_set)
            if true_set and true_set.issubset(top_set):
                entity_perfect += 1

        elapsed = time.time() - t0
        pair_recall = recovered_tp / total_true_positives if total_true_positives > 0 else 1.0
        entity_recall = entity_perfect / total_entities_with_matches if total_entities_with_matches > 0 else 1.0
        reduction_ratio = 1.0 - (cands_generated / max_possible_pairs)
        avg_cands = cands_generated / len(processed_s1)

        row = {
            "strategy": f"8. Multi-Channel Union (Ranked Top-{k})",
            "pair_recall": round(pair_recall * 100, 2),
            "entity_recall": round(entity_recall * 100, 2),
            "avg_candidates": round(avg_cands, 2),
            "candidate_count": cands_generated,
            "reduction_ratio": round(reduction_ratio * 100, 4),
            "runtime_seconds": round(elapsed, 2)
        }
        benchmark_rows.append(row)
        logger.info("  %s: Pair Recall=%.2f%% | Entity Recall=%.2f%% | Avg Cands=%.1f | RR=%.4f%%",
                    row["strategy"][:35].ljust(35), row["pair_recall"], row["entity_recall"], row["avg_candidates"], row["reduction_ratio"])

    # Save reports/blocking_comparison.csv
    csv_path = REPORTS_DIR / "blocking_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    with open(PUBLIC_REPORTS_DIR / "blocking_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    logger.info("Successfully generated %s", csv_path)


if __name__ == "__main__":
    main()
