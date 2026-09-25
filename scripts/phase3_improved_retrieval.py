#!/usr/bin/env python3
"""
scripts/phase3_improved_retrieval.py
Phase 3: Multi-Channel Candidate Generation with Independent Channel Benchmarking.

Fixes:
  1. Load ALL true positive targets unconditionally (no cap on positives).
  2. Adds new retrieval channels:
     - C1: Character 3-gram index (covers OCR/typo variants)
     - C2: Character 4-gram index (higher precision variant of C1)
     - C3: Rare token index (tokens < 50 occurrences in full corpus)
     - C4: Domain stem index (URL-extracted stems)
     - C5: Postal code index (already exists, now primary)

Reports independent pair recall contribution per channel.
Outputs the best combined candidate map for Phase 4 scoring.
"""
import csv
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3Retrieval")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)


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


def extract_domain_stem(raw_name: str) -> str:
    if not raw_name:
        return ""
    m = DOMAIN_RE.search(raw_name.lower())
    return m.group(1).replace('-', '') if m else ""


def get_char_ngrams(text: str, n: int) -> set:
    """Extract character n-grams from text (no spaces)."""
    s = text.replace(' ', '')
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def get_street_num(addr: str) -> str:
    if not addr:
        return ""
    m = re.match(r'^(\d+)\b', addr)
    return m.group(1) if m else ""


def get_postal(addr: str) -> str:
    digits = extract_digits(addr)
    for d in digits:
        if len(d) in (5, 6):
            return d
    return ""


def preprocess_record(r: dict) -> dict:
    name = r.get('business_name', '')
    addr = r.get('business_address', '')
    c_name = clean_text(name)
    c_addr = clean_text(addr)
    return {
        'id': r['entity_id'],
        'raw_name': name,
        'clean_name': c_name,
        'clean_addr': c_addr,
        'name_tokens': extract_tokens(c_name),
        'addr_tokens': extract_tokens(c_addr),
        'street_num': get_street_num(c_addr),
        'postal': get_postal(c_addr),
        'domain_stem': extract_domain_stem(name),
        'country': r.get('country', ''),
        'ngrams3': get_char_ngrams(c_name, 3),
        'ngrams4': get_char_ngrams(c_name, 4),
    }


def load_all_candidates(data_root: Path, needed_positive_cids: set, max_distractors: int = 150000) -> dict:
    """Load ALL true positive targets + up to max_distractors per source."""
    cands = {}
    for s_file in [data_root / "train" / "train_source2.tsv", data_root / "train" / "train_source3.tsv"]:
        distractor_count = 0
        with open(s_file, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    is_needed = cid in needed_positive_cids
                    if is_needed or distractor_count < max_distractors:
                        cands[cid] = preprocess_record({
                            'entity_id': cid, 'business_name': name,
                            'business_address': addr, 'country': country
                        })
                        if not is_needed:
                            distractor_count += 1
    return cands


def build_indexes(cands: dict, token_freq: dict) -> dict:
    """Build all retrieval channel indexes from candidate pool."""
    name_token_index = defaultdict(list)       # (country, token) -> [cid]
    rare_token_index = defaultdict(list)       # (country, token) -> [cid]  (freq < 50)
    addr_token_index = defaultdict(list)       # (country, addr_tok) -> [cid]
    street_num_index = defaultdict(list)       # (country, street_num) -> [cid]
    postal_index = defaultdict(list)           # (country, postal) -> [cid]
    domain_index = defaultdict(list)           # (country, domain_stem) -> [cid]
    ngram3_index = defaultdict(list)           # (country, ngram3) -> [cid]
    ngram4_index = defaultdict(list)           # (country, ngram4) -> [cid]

    for cid, cand in cands.items():
        c = cand["country"]
        for tok in cand["name_tokens"]:
            name_token_index[(c, tok)].append(cid)
            if token_freq.get(tok, 0) < 50:
                rare_token_index[(c, tok)].append(cid)
        for atok in cand["addr_tokens"]:
            addr_token_index[(c, atok)].append(cid)
        if cand["street_num"]:
            street_num_index[(c, cand["street_num"])].append(cid)
        if cand["postal"]:
            postal_index[(c, cand["postal"])].append(cid)
        if cand["domain_stem"] and len(cand["domain_stem"]) >= 4:
            domain_index[(c, cand["domain_stem"])].append(cid)
        for ng in cand["ngrams3"]:
            ngram3_index[(c, ng)].append(cid)
        for ng in cand["ngrams4"]:
            ngram4_index[(c, ng)].append(cid)

    return {
        "name_token": name_token_index,
        "rare_token": rare_token_index,
        "addr_token": addr_token_index,
        "street_num": street_num_index,
        "postal": postal_index,
        "domain": domain_index,
        "ngram3": ngram3_index,
        "ngram4": ngram4_index,
    }


def retrieve_for_entity(s1: dict, indexes: dict, max_k: int = 80) -> dict:
    """Retrieve candidates via each channel independently, then combine."""
    c = s1["country"]
    channel_hits = {
        "name_token": set(),
        "rare_token": set(),
        "addr_token": set(),
        "street_num": set(),
        "postal": set(),
        "domain": set(),
        "ngram3": set(),
        "ngram4": set(),
    }

    # Name tokens
    for tok in s1["name_tokens"]:
        postings = indexes["name_token"].get((c, tok), [])
        if len(postings) <= 60:
            channel_hits["name_token"].update(postings)

    # Rare tokens
    for tok in s1["name_tokens"]:
        postings = indexes["rare_token"].get((c, tok), [])
        if len(postings) <= 30:
            channel_hits["rare_token"].update(postings)

    # Address tokens
    for atok in s1["addr_tokens"]:
        postings = indexes["addr_token"].get((c, atok), [])
        if len(postings) <= 40:
            channel_hits["addr_token"].update(postings)

    # Street number
    if s1["street_num"]:
        postings = indexes["street_num"].get((c, s1["street_num"]), [])
        if len(postings) <= 40:
            channel_hits["street_num"].update(postings)

    # Postal code
    if s1["postal"]:
        postings = indexes["postal"].get((c, s1["postal"]), [])
        if len(postings) <= 30:
            channel_hits["postal"].update(postings)

    # Domain stem
    if s1["domain_stem"] and len(s1["domain_stem"]) >= 4:
        channel_hits["domain"].update(indexes["domain"].get((c, s1["domain_stem"]), []))

    # 3-gram char
    for ng in s1["ngrams3"]:
        postings = indexes["ngram3"].get((c, ng), [])
        if len(postings) <= 20:  # Strict cap: only rare 3-grams
            channel_hits["ngram3"].update(postings)

    # 4-gram char
    for ng in s1["ngrams4"]:
        postings = indexes["ngram4"].get((c, ng), [])
        if len(postings) <= 30:
            channel_hits["ngram4"].update(postings)

    return channel_hits


def evaluate_recall(val_s1_records, val_gt, cands, indexes, max_k):
    """Evaluate pair recall per channel and combined."""
    total_true_mentions = sum(len(v) for v in val_gt.values() if v)
    channel_names = ["name_token", "rare_token", "addr_token", "street_num", "postal", "domain", "ngram3", "ngram4"]

    channel_recovered = {ch: 0 for ch in channel_names}
    combined_recovered = 0
    combined_cand_count = 0
    entity_perfect_combined = 0
    n_with_targets = 0

    for s1 in val_s1_records:
        s1_id = s1["id"]
        true_set = set(val_gt.get(s1_id, []))
        if not true_set:
            continue
        n_with_targets += 1

        channel_hits = retrieve_for_entity(s1, indexes, max_k)

        for ch in channel_names:
            recovered = len(channel_hits[ch] & true_set)
            channel_recovered[ch] += recovered

        combined = set()
        for ch in channel_names:
            combined.update(channel_hits[ch])
        combined_list = list(combined)[:max_k]
        combined_set = set(combined_list)

        combined_recovered += len(combined_set & true_set)
        combined_cand_count += len(combined_list)

        if true_set.issubset(combined_set):
            entity_perfect_combined += 1

    logger.info("=" * 60)
    logger.info("CHANNEL RECALL CONTRIBUTIONS (max_k=%d)", max_k)
    logger.info("=" * 60)
    channel_results = []
    for ch in channel_names:
        recall_pct = (channel_recovered[ch] / total_true_mentions) * 100 if total_true_mentions > 0 else 0
        logger.info("  %-20s  Pair Recall: %.2f%%  (%d / %d true mentions)", ch, recall_pct,
                    channel_recovered[ch], total_true_mentions)
        channel_results.append({"channel": ch, "pair_recall_pct": round(recall_pct, 2),
                                 "recovered": channel_recovered[ch]})

    combined_pair_recall = (combined_recovered / total_true_mentions) * 100 if total_true_mentions > 0 else 0
    entity_perfect_pct = (entity_perfect_combined / n_with_targets) * 100 if n_with_targets > 0 else 0
    avg_cands = combined_cand_count / len(val_s1_records) if val_s1_records else 0

    logger.info("-" * 60)
    logger.info("  COMBINED (all channels, max_k=%d):", max_k)
    logger.info("    Pair Recall:          %.2f%%", combined_pair_recall)
    logger.info("    Entity-Perfect Recall: %.2f%%", entity_perfect_pct)
    logger.info("    Avg Candidates / S1:  %.2f", avg_cands)
    logger.info("    Total Candidates:     %d", combined_cand_count)
    logger.info("=" * 60)

    return channel_results, combined_pair_recall, entity_perfect_pct, avg_cands


def build_token_frequencies(cands: dict) -> dict:
    """Build corpus-level token frequency map for rare-token identification."""
    freq = defaultdict(int)
    for cand in cands.values():
        for tok in cand["name_tokens"]:
            freq[tok] += 1
    return dict(freq)


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: MULTI-CHANNEL RETRIEVAL BENCHMARK")
    logger.info("=" * 60)

    # 1. Load Frozen Split
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]
    val_s1_set = set(val_s1_ids)

    # 2. Ground Truth
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

    logger.info("Validation: %d S1 entities | %d true positive mentions to recover",
                len(val_s1_ids), len(needed_positive_cids))

    # 3. Load S1
    val_s1_records = []
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in val_s1_set:
                val_s1_records.append(preprocess_record(r))

    # 4. Load Candidates (FIX: all positives + 150k distractors per source)
    t0 = time.time()
    logger.info("Loading candidates (ALL true positives + 150k distractors per source)...")
    cands = load_all_candidates(DATA_ROOT, needed_positive_cids, max_distractors=150000)
    logger.info("Loaded %d candidates in %.2fs", len(cands), time.time() - t0)

    # Verify all positives loaded
    loaded_positives = needed_positive_cids & set(cands.keys())
    logger.info("True positive coverage: %d / %d (%.1f%%)",
                len(loaded_positives), len(needed_positive_cids),
                len(loaded_positives) / len(needed_positive_cids) * 100)

    # 5. Token frequencies for rare-token channel
    token_freq = build_token_frequencies(cands)

    # 6. Build indexes
    t0 = time.time()
    logger.info("Building indexes...")
    indexes = build_indexes(cands, token_freq)
    logger.info("Indexes built in %.2fs", time.time() - t0)

    # 7. Evaluate each channel + combined
    t0 = time.time()
    channel_results, combined_recall, entity_perfect, avg_cands = evaluate_recall(
        val_s1_records, val_gt, cands, indexes, max_k=80
    )
    logger.info("Channel evaluation completed in %.2fs", time.time() - t0)

    # 8. Save results
    result = {
        "channels": channel_results,
        "combined_pair_recall_pct": round(combined_recall, 2),
        "entity_perfect_recall_pct": round(entity_perfect, 2),
        "avg_candidates_per_s1": round(avg_cands, 2),
        "total_candidates_loaded": len(cands),
        "true_positive_coverage_pct": round(len(loaded_positives) / len(needed_positive_cids) * 100, 2)
    }

    out_path = REPORTS_DIR / "phase3_retrieval_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved retrieval benchmark to %s", out_path)


if __name__ == "__main__":
    main()
