"""
scripts/benchmark_blocking_and_pruning.py - Phase 5 Blocker Ablation & Phase 6 Candidate Pruning Benchmark.
Amazon ML Challenge 2026.

Executes:
Phase 5: Accurate, country-partitioned blocker channel ablation:
  - exact_name
  - char_3gram
  - char_4gram
  - token_overlap
  - rare_token
  - addr_token
  - postal_code
  - street_number
  - domain_stem
  - all_combined
  Outputs -> reports/diagnostic/blocker_ablation_200k.csv

Phase 6: Ranked Candidate Pruning vs Unordered Truncation:
  - Benchmark K = [25, 50, 75, 100, 150, 200, 300]
  - Ranked by composite similarity (name sim + address overlap + digit overlap)
  Outputs -> reports/diagnostic/pruning_benchmark_200k.csv
"""
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIAG_DIR = ROOT / "data" / "diagnostic_200k"
REPORTS_DIAG = ROOT / "reports" / "diagnostic"
REPORTS_DIAG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BlockingPruning")

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
RE_ALPHA = re.compile(r'[^a-z0-9]')
DOMAIN_RE = re.compile(r'\b([a-z0-9\-]+)\.(?:com|in|org|net|fr|co|io|biz|info|us)\b', re.IGNORECASE)

LEGAL_SUFFIXES = [
    'private limited', 'pvt limited', 'p limited', 'private ltd', 'pvt ltd',
    'corporation', 'incorporated', 'limited', 'enterprises', 'enterprise',
    'services', 'solutions', 'technologies', 'holdings', 'industries',
    'international', 'consultants', 'consultancy', 'consulting', 'center',
    'centre', 'corp', 'inc', 'llc', 'llp', 'sarl', 'sasu', 'eurl', 'gmbh',
    'gie', 'sas', 'sa', 'ag', 'bv', 'nv', 'spa', 'srl', 'ltd'
]

ADDR_ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'dr': 'drive', 'ln': 'lane', 'ave': 'avenue',
    'ct': 'court', 'blvd': 'boulevard', 'hwy': 'highway', 'pkwy': 'parkway',
    'apt': 'apartment', 'ste': 'suite', 'unit': 'unit', 'bldg': 'building',
    'fl': 'floor', 'flr': 'floor', 'pl': 'place', 'cir': 'circle',
    'opp': 'opposite', 'nr': 'near', 'soc': 'society', 'col': 'colony',
    'ext': 'extension', 'sec': 'sector', 'dist': 'district', 'vill': 'village',
    'po': 'postoffice', 'ps': 'policestation', 'mkt': 'market',
    'cmplx': 'complex', 'twr': 'tower'
}

STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}


class FastCand:
    __slots__ = ('id', 'clean_name', 'clean_addr', 'alpha_name', 'street_num', 'postal', 'name_tokens', 'addr_tokens', 'domain')
    def __init__(self, eid: str, name: str, addr: str):
        self.id = eid
        cname = clean_text(name)
        caddr = clean_text(addr)
        self.clean_name = cname
        self.clean_addr = caddr
        self.alpha_name = RE_ALPHA.sub('', cname)
        self.name_tokens = tuple(t for t in cname.split() if t not in STOPWORDS and len(t) >= 2)
        self.addr_tokens = tuple(expand_addr(caddr))
        # Street num
        digits = RE_DIGITS.findall(caddr)
        self.street_num = digits[0] if digits and len(digits[0]) <= 5 else ""
        self.postal = next((d for d in digits if len(d) in (5, 6)), "")
        m = DOMAIN_RE.search(name.lower())
        self.domain = m.group(1).replace('-', '') if m else ""


def clean_text(text: str) -> str:
    if not text: return ""
    t = text.lower()
    t = re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b)', '', t)
    t = RE_PUNCT.sub(' ', t)
    return ' '.join(t.split())


def expand_addr(addr_clean: str) -> List[str]:
    toks = []
    for t in addr_clean.split():
        exp = ADDR_ABBREVIATIONS.get(t, t)
        toks.append(exp)
        if exp != t: toks.append(t)
    return toks


def get_char_ngrams(text: str, n: int) -> Set[str]:
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def main():
    logger.info("=" * 80)
    logger.info("PHASE 5 & PHASE 6: BLOCKING ABLATION & CANDIDATE PRUNING BENCHMARK")
    logger.info("=" * 80)

    # 1. Load Ground Truth
    logger.info("Loading ground truth...")
    gt = {}
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            gt[parts[0]] = set(x.strip() for x in parts[1].split(",") if x.strip()) if len(parts) > 1 else set()

    # 2. Load Source 1 by country
    logger.info("Loading Source 1...")
    s1_by_country = defaultdict(list)
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_by_country[country].append(FastCand(eid, name, addr))

    # 3. Load Candidate Pool (S2/S3) by country
    logger.info("Loading Candidate Pool (S2/S3)...")
    s23_by_country = defaultdict(dict)
    for src in ["source2.tsv", "source3.tsv"]:
        with open(DIAG_DIR / src, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    s23_by_country[country][cid] = FastCand(cid, name, addr)

    for c in s1_by_country:
        logger.info("Country %-7s: %d S1 entities | %d S2/S3 records", c, len(s1_by_country[c]), len(s23_by_country[c]))

    # =========================================================================
    # PHASE 5: Comprehensive Blocker Channel Ablation
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PHASE 5: Measuring Blocker Channels Across Both US & India...")
    logger.info("=" * 80)

    # Build per-country channel indexes
    indexes_by_country = {}
    for country, pool in s23_by_country.items():
        logger.info("Building indexes for %s (%d records)...", country, len(pool))
        tf = defaultdict(int)
        for c in pool.values():
            for t in c.name_tokens: tf[t] += 1

        idx_exact = defaultdict(list)
        idx_ng3 = defaultdict(list)
        idx_ng4 = defaultdict(list)
        idx_name = defaultdict(list)
        idx_rare = defaultdict(list)
        idx_addr = defaultdict(list)
        idx_snum = defaultdict(list)
        idx_postal = defaultdict(list)
        idx_domain = defaultdict(list)

        for cid, c in pool.items():
            if c.alpha_name and len(c.alpha_name) >= 3:
                p = idx_exact[c.alpha_name]
                if len(p) <= 80: p.append(cid)
            for t in c.name_tokens:
                p = idx_name[t]
                if len(p) <= 100: p.append(cid)
                if tf.get(t, 0) < 50:
                    pr = idx_rare[t]
                    if len(pr) <= 50: pr.append(cid)
            for at in c.addr_tokens:
                p = idx_addr[at]
                if len(p) <= 60: p.append(cid)
            if c.street_num:
                p = idx_snum[c.street_num]
                if len(p) <= 60: p.append(cid)
            if c.postal:
                p = idx_postal[c.postal]
                if len(p) <= 50: p.append(cid)
            if c.domain:
                p = idx_domain[c.domain]
                if len(p) <= 50: p.append(cid)
            for ng in get_char_ngrams(c.clean_name, 3):
                p = idx_ng3[ng]
                if len(p) <= 50: p.append(cid)
            for ng in get_char_ngrams(c.clean_name, 4):
                p = idx_ng4[ng]
                if len(p) <= 50: p.append(cid)

        indexes_by_country[country] = {
            "exact_name": idx_exact,
            "char_3gram": idx_ng3,
            "char_4gram": idx_ng4,
            "token_overlap": idx_name,
            "rare_token": idx_rare,
            "addr_token": idx_addr,
            "street_number": idx_snum,
            "postal_code": idx_postal,
            "domain_stem": idx_domain,
        }

    # Evaluate channels on representative stratified evaluation sample of 25,000 non-singletons (15K US, 10K India)
    eval_sample = []
    for c in ["US", "India"]:
        non_sing = [s for s in s1_by_country[c] if len(gt[s.id]) > 0]
        n_take = 15000 if c == "US" else 10000
        eval_sample.extend([(c, s) for s in non_sing[:n_take]])

    total_eval_true_matches = sum(len(gt[s.id]) for _, s in eval_sample)
    logger.info("Evaluating on %d non-singleton entities with %d ground truth matches...",
                len(eval_sample), total_eval_true_matches)

    channel_list = [
        "exact_name", "char_3gram", "char_4gram", "token_overlap",
        "rare_token", "addr_token", "street_number", "postal_code", "domain_stem",
        "ALL_COMBINED"
    ]

    ablation_rows = []
    for ch in channel_list:
        t0 = time.time()
        retrieved_true = 0
        perfect_entities = 0
        total_candidates = 0

        for country, s1 in eval_sample:
            true_m = gt[s1.id]
            idx_dict = indexes_by_country[country]
            cands = set()

            if ch == "exact_name":
                if s1.alpha_name:
                    p = idx_dict["exact_name"].get(s1.alpha_name)
                    if p: cands.update(p)
            elif ch == "char_3gram":
                for ng in get_char_ngrams(s1.clean_name, 3):
                    p = idx_dict["char_3gram"].get(ng)
                    if p: cands.update(p)
            elif ch == "char_4gram":
                for ng in get_char_ngrams(s1.clean_name, 4):
                    p = idx_dict["char_4gram"].get(ng)
                    if p: cands.update(p)
            elif ch == "token_overlap":
                for t in s1.name_tokens:
                    p = idx_dict["token_overlap"].get(t)
                    if p: cands.update(p)
            elif ch == "rare_token":
                for t in s1.name_tokens:
                    p = idx_dict["rare_token"].get(t)
                    if p: cands.update(p)
            elif ch == "addr_token":
                for at in s1.addr_tokens:
                    p = idx_dict["addr_token"].get(at)
                    if p: cands.update(p)
            elif ch == "street_number":
                if s1.street_num:
                    p = idx_dict["street_number"].get(s1.street_num)
                    if p: cands.update(p)
            elif ch == "postal_code":
                if s1.postal:
                    p = idx_dict["postal_code"].get(s1.postal)
                    if p: cands.update(p)
            elif ch == "domain_stem":
                if s1.domain:
                    p = idx_dict["domain_stem"].get(s1.domain)
                    if p: cands.update(p)
            elif ch == "ALL_COMBINED":
                if s1.alpha_name:
                    p = idx_dict["exact_name"].get(s1.alpha_name)
                    if p: cands.update(p)
                for t in s1.name_tokens:
                    p = idx_dict["token_overlap"].get(t)
                    if p: cands.update(p)
                    pr = idx_dict["rare_token"].get(t)
                    if pr: cands.update(pr)
                for ng in get_char_ngrams(s1.clean_name, 4):
                    p = idx_dict["char_4gram"].get(ng)
                    if p: cands.update(p)
                for at in s1.addr_tokens:
                    p = idx_dict["addr_token"].get(at)
                    if p: cands.update(p)
                if s1.street_num:
                    p = idx_dict["street_number"].get(s1.street_num)
                    if p: cands.update(p)
                if s1.postal:
                    p = idx_dict["postal_code"].get(s1.postal)
                    if p: cands.update(p)
                if s1.domain:
                    p = idx_dict["domain_stem"].get(s1.domain)
                    if p: cands.update(p)

            total_candidates += len(cands)
            tp = len(cands & true_m)
            retrieved_true += tp
            if tp == len(true_m):
                perfect_entities += 1

        dt = time.time() - t0
        pair_rec = (retrieved_true / total_eval_true_matches * 100) if total_eval_true_matches else 0.0
        perf_rec = (perfect_entities / len(eval_sample) * 100) if eval_sample else 0.0
        avg_c = total_candidates / len(eval_sample)
        red_ratio = (1.0 - (total_candidates / (len(eval_sample) * 1_300_000))) * 100

        row = {
            "channel": ch,
            "pair_recall_pct": round(pair_rec, 2),
            "entity_perfect_recall_pct": round(perf_rec, 2),
            "avg_candidates": round(avg_c, 1),
            "reduction_ratio_pct": round(red_ratio, 4),
            "runtime_sec": round(dt, 2)
        }
        ablation_rows.append(row)
        logger.info("Channel %-14s -> PairRecall: %5.2f%% | PerfectRecall: %5.2f%% | AvgCands: %5.1f | Time: %4.2fs",
                    ch, pair_rec, perf_rec, avg_c, dt)

    ablation_path = REPORTS_DIAG / "blocker_ablation_200k.csv"
    with open(ablation_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    logger.info("Saved Blocker Ablation Report -> %s", ablation_path.name)

    # =========================================================================
    # PHASE 6: Candidate Pruning Benchmark (Ranked vs Unordered across K)
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PHASE 6: Candidate Pruning Benchmark across K in [25, 50, 75, 100, 150, 200, 300]")
    logger.info("Comparing: (A) Unordered Truncation vs (B) Ranked Similarity Pruning")
    logger.info("=" * 80)

    # Fast scoring function for ranking
    def rank_candidates(s1: FastCand, cands: List[str], pool: Dict[str, FastCand]) -> List[str]:
        scored = []
        s1_toks = set(s1.name_tokens)
        s1_addr = set(s1.addr_tokens)
        for cid in cands:
            c = pool.get(cid)
            if not c: continue
            # Composite lightweight ranking score
            n_inter = len(s1_toks & set(c.name_tokens))
            a_inter = len(s1_addr & set(c.addr_tokens))
            num_match = 1 if (s1.street_num and c.street_num and s1.street_num == c.street_num) else 0
            alpha_match = 2 if (s1.alpha_name and s1.alpha_name == c.alpha_name) else 0
            rank_score = n_inter * 3 + a_inter * 2 + num_match * 4 + alpha_match * 5
            scored.append((rank_score, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored]

    # Pre-retrieve raw candidates for 10,000 entities
    pruning_sample = eval_sample[:10000]
    total_pruning_true = sum(len(gt[s.id]) for _, s in pruning_sample)

    logger.info("Pre-retrieving candidate pools for 10,000 entities...")
    retrieved_map = {}
    for country, s1 in pruning_sample:
        idx_dict = indexes_by_country[country]
        cands = set()
        if s1.alpha_name:
            p = idx_dict["exact_name"].get(s1.alpha_name)
            if p: cands.update(p)
        for t in s1.name_tokens:
            p = idx_dict["token_overlap"].get(t)
            if p: cands.update(p)
            pr = idx_dict["rare_token"].get(t)
            if pr: cands.update(pr)
        for ng in get_char_ngrams(s1.clean_name, 4):
            p = idx_dict["char_4gram"].get(ng)
            if p: cands.update(p)
        for at in s1.addr_tokens:
            p = idx_dict["addr_token"].get(at)
            if p: cands.update(p)
        if s1.street_num:
            p = idx_dict["street_number"].get(s1.street_num)
            if p: cands.update(p)
        if s1.postal:
            p = idx_dict["postal_code"].get(s1.postal)
            if p: cands.update(p)
        retrieved_map[s1.id] = list(cands)

    k_values = [25, 50, 75, 100, 150, 200, 300]
    pruning_results = []

    for k in k_values:
        # 1. Unordered Truncation
        tp_unord = 0
        perf_unord = 0
        for country, s1 in pruning_sample:
            raw_c = retrieved_map[s1.id]
            trunc_c = set(raw_c[:k])
            true_m = gt[s1.id]
            hit = len(trunc_c & true_m)
            tp_unord += hit
            if hit == len(true_m): perf_unord += 1

        rec_unord = (tp_unord / total_pruning_true * 100) if total_pruning_true else 0.0
        perf_rec_unord = (perf_unord / len(pruning_sample) * 100)

        # 2. Ranked Pruning
        tp_ranked = 0
        perf_ranked = 0
        for country, s1 in pruning_sample:
            raw_c = retrieved_map[s1.id]
            ranked_c = rank_candidates(s1, raw_c, s23_by_country[country])
            trunc_c = set(ranked_c[:k])
            true_m = gt[s1.id]
            hit = len(trunc_c & true_m)
            tp_ranked += hit
            if hit == len(true_m): perf_ranked += 1

        rec_ranked = (tp_ranked / total_pruning_true * 100) if total_pruning_true else 0.0
        perf_rec_ranked = (perf_ranked / len(pruning_sample) * 100)
        gain = rec_ranked - rec_unord

        row = {
            "K": k,
            "unordered_recall_pct": round(rec_unord, 2),
            "unordered_perfect_pct": round(perf_rec_unord, 2),
            "ranked_recall_pct": round(rec_ranked, 2),
            "ranked_perfect_pct": round(perf_rec_ranked, 2),
            "gain_from_ranking_pct": round(gain, 2)
        }
        pruning_results.append(row)
        logger.info("K=%3d | Unordered Recall: %5.2f%% -> Ranked Recall: %5.2f%% (+%4.2f%%) | Perfect: %5.2f%%",
                    k, rec_unord, rec_ranked, gain, perf_rec_ranked)

    pruning_path = REPORTS_DIAG / "pruning_benchmark_200k.csv"
    with open(pruning_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pruning_results[0].keys()))
        writer.writeheader()
        writer.writerows(pruning_results)
    logger.info("Saved Pruning Benchmark Report -> %s", pruning_path.name)
    logger.info("=" * 80)
    logger.info("PHASE 5 & 6 BENCHMARKS COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
