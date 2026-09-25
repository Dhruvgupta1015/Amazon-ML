#!/usr/bin/env python3
"""
run_phase8b_inference.py - Full Test-Set Inference with Phase 8b Champion.

Decision Rule (validated on frozen 20k split, F0.5=0.8387, Singleton=94.52%):
  For each (S1, S2/S3 candidate) pair scored:
  - Accept if score >= 0.82 (high confidence, no addr check required)
  - Accept if 0.54 <= score < 0.82 AND addr_jaccard >= 0.18
  - Reject otherwise (singleton protection)

Retrieval: Phase 3 multi-channel (name_token + rare_token + addr_token +
           street_num + postal + ngram4)
"""
from __future__ import annotations
import csv
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase8bInference")

DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')

# --- Validated Decision Parameters ---
TAU       = 0.54   # Minimum score to consider any match
HIGH_CONF = 0.82   # Score >= HIGH_CONF: accept without address check
ADDR_FLOOR = 0.18  # Required addr_jaccard when score in [TAU, HIGH_CONF)
MAX_K     = 80     # Max candidates per S1 entity


def clean_text(s: str) -> str:
    if not s: return ""
    s = str(s).lower().replace('&', ' and ')
    return ' '.join(RE_PUNCT.sub(' ', s).split())

def extract_tokens(text: str) -> set:
    return set(text.split())

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def get_street_num(addr: str) -> str:
    m = re.match(r'^(\d+)\b', addr or "")
    return m.group(1) if m else ""

def get_postal(addr: str) -> str:
    for d in RE_DIGITS.findall(addr or ""):
        if len(d) in (5, 6): return d
    return ""

def preprocess(eid: str, name: str, addr: str, country: str) -> dict:
    cn = clean_text(name)
    ca = clean_text(addr)
    return {'id': eid, 'clean_name': cn, 'clean_addr': ca,
            'name_tokens': extract_tokens(cn), 'addr_tokens': extract_tokens(ca),
            'street_num': get_street_num(ca), 'postal': get_postal(ca),
            'country': country}

def jaro_winkler(s1: str, s2: str, max_len: int = 40) -> float:
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1
    m1 = [False]*l1; m2 = [False]*l2; matches = 0
    for i in range(l1):
        for j in range(max(0, i-md), min(i+md+1, l2)):
            if m2[j] or s1[i] != s2[j]: continue
            m1[i] = m2[j] = True; matches += 1; break
    if not matches: return 0.0
    t = 0; k = 0
    for i in range(l1):
        if not m1[i]: continue
        while not m2[k]: k += 1
        if s1[i] != s2[k]: t += 1
        k += 1
    sim = (matches/l1 + matches/l2 + (matches - t/2)/matches) / 3
    p = sum(1 for i in range(min(4, l1, l2)) if s1[i] == s2[i])
    return sim + p * 0.1 * (1.0 - sim)

def token_jaccard(t1: set, t2: set) -> float:
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def ngram_jaccard(ng1: set, ng2: set) -> float:
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def decide(score: float, addr_j: float) -> bool:
    """Phase 8b address-aware tiered decision."""
    if score >= HIGH_CONF: return True
    if score >= TAU and addr_j >= ADDR_FLOOR: return True
    return False

def build_indexes(s23_data: dict, token_freq: dict) -> dict:
    ni = defaultdict(list); ri = defaultdict(list)
    ai = defaultdict(list); si_idx = defaultdict(list)
    pi = defaultdict(list)
    for cid, c in s23_data.items():
        for t in c["name_tokens"]:
            ni[t].append(cid)
            if token_freq.get(t, 0) < 50: ri[t].append(cid)
        for at in c["addr_tokens"]: ai[at].append(cid)
        if c["street_num"]: si_idx[c["street_num"]].append(cid)
        if c["postal"]: pi[c["postal"]].append(cid)
    return {"name": ni, "rare": ri, "addr": ai, "snum": si_idx, "postal": pi}

def retrieve_candidates(s1: dict, idx: dict) -> list:
    cands = set()
    for t in s1["name_tokens"]:
        p = idx["name"].get(t, [])
        if len(p) <= 60: cands.update(p)
    for t in s1["name_tokens"]:
        p = idx["rare"].get(t, [])
        if len(p) <= 30: cands.update(p)
    for at in s1["addr_tokens"]:
        p = idx["addr"].get(at, [])
        if len(p) <= 40: cands.update(p)
    if s1["street_num"]:
        p = idx["snum"].get(s1["street_num"], [])
        if len(p) <= 40: cands.update(p)
    if s1["postal"]:
        p = idx["postal"].get(s1["postal"], [])
        if len(p) <= 30: cands.update(p)
    return list(cands)[:MAX_K]

def resolve_country(country: str, s1_records: list, test_s2_path: Path, test_s3_path: Path) -> dict:
    logger.info("[%s] Loading S2/S3 candidates...", country.upper())
    t_load = time.time()
    s23_data = {}
    for s_path in [test_s2_path, test_s3_path]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    cid, name, addr = parts[0], parts[1], parts[2]
                    s23_data[cid] = preprocess(cid, name, addr, country)
    logger.info("[%s] Loaded %d S2/S3 candidates in %.2fs", country.upper(), len(s23_data), time.time() - t_load)

    # Token frequencies for rare-token channel
    token_freq = defaultdict(int)
    for c in s23_data.values():
        for t in c["name_tokens"]: token_freq[t] += 1

    # Build indexes
    t_idx = time.time()
    idx = build_indexes(s23_data, token_freq)
    logger.info("[%s] Built indexes in %.2fs", country.upper(), time.time() - t_idx)

    # Resolve
    results = {}
    matched_count = 0
    t1 = time.time()
    for i, s1 in enumerate(s1_records):
        cand_list = retrieve_candidates(s1, idx)
        matched = []
        s1_ng4 = None
        for cid in cand_list:
            cand = s23_data.get(cid)
            if not cand: continue
            jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
            tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
            tj_addr = token_jaccard(s1["addr_tokens"], cand["addr_tokens"])
            sn1, sn2 = s1["street_num"], cand["street_num"]
            if sn1 and sn2: conflict = 1.0 if sn1 != sn2 else 0.0
            elif sn1 or sn2: conflict = 0.2
            else: conflict = 0.0
            base = 0.5 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.3 * conflict
            if jw >= 0.65:
                if s1_ng4 is None: s1_ng4 = get_char_ngrams(s1["clean_name"], 4)
                ng4 = ngram_jaccard(s1_ng4, get_char_ngrams(cand["clean_name"], 4))
                if ng4 >= 0.5:
                    base = max(base, 0.5*jw + 0.15*tj_name + 0.15*tj_addr + 0.2*ng4 - 0.3*conflict)
            if decide(base, tj_addr):
                matched.append(cid)
        results[s1["id"]] = (",".join(matched), ",".join(cand_list))
        if matched: matched_count += 1
        if (i + 1) % 50000 == 0 or (i + 1) == len(s1_records):
            elapsed = time.time() - t1
            logger.info("[%s] %d/%d (%.0f/s) | Matched: %d (%.1f%%)",
                        country.upper(), i+1, len(s1_records),
                        (i+1)/max(elapsed, 0.001), matched_count,
                        matched_count/(i+1)*100)
    logger.info("[%s] Done: %d/%d matched (%.1f%%)", country.upper(),
                matched_count, len(s1_records), matched_count/len(s1_records)*100)
    return results

def main():
    logger.info("=" * 70)
    logger.info("  PHASE 8b INFERENCE: AMAZON ML CHALLENGE 2026 SUBMISSION")
    logger.info("  Decision: tau=%.2f | high_conf=%.2f | addr_floor=%.2f | max_k=%d",
                TAU, HIGH_CONF, ADDR_FLOOR, MAX_K)
    logger.info("  Validated: F0.5=0.8387 | Precision=98.11%% | Singleton=94.52%%")
    logger.info("=" * 70)
    t_start = time.time()

    test_s1_path = DATASET_ROOT / "test" / "test_source1.tsv"
    test_s2_path = DATASET_ROOT / "test" / "test_source2.tsv"
    test_s3_path = DATASET_ROOT / "test" / "test_source3.tsv"

    # Read all S1 records, partition by country
    logger.info("Reading test_source1.tsv...")
    all_s1_ids = []
    country_s1 = defaultdict(list)
    with open(test_s1_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                all_s1_ids.append(eid)
                country_s1[country].append(preprocess(eid, name, addr, country))

    logger.info("Loaded %d S1 test entities across %d countries:", len(all_s1_ids), len(country_s1))
    for c, recs in country_s1.items():
        logger.info("  - %s: %d (%.1f%%)", c, len(recs), len(recs)/len(all_s1_ids)*100)

    # Resolve per country
    all_results = {}
    for country in sorted(country_s1.keys()):
        c_results = resolve_country(country, country_s1[country], test_s2_path, test_s3_path)
        all_results.update(c_results)

    # Write output TSVs
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    mr_path = output_dir / "matching_results.tsv"
    cp_path = output_dir / "candidate_pairs.tsv"

    logger.info("Writing output TSVs in exact S1 input order...")
    with open(mr_path, "w", encoding="utf-8") as f_mr, \
         open(cp_path, "w", encoding="utf-8") as f_cp:
        f_mr.write("source1_entity_id\tmatched_entity_ids\n")
        f_cp.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            matched, candidates = all_results.get(s1_id, ("", ""))
            f_mr.write(f"{s1_id}\t{matched}\n")
            f_cp.write(f"{s1_id}\t{candidates}\n")

    mr_size = mr_path.stat().st_size / (1024 * 1024)
    cp_size = cp_path.stat().st_size / (1024 * 1024)
    logger.info("Files written:")
    logger.info("  - matching_results.tsv  (%.2f MB)", mr_size)
    logger.info("  - candidate_pairs.tsv   (%.2f MB)", cp_size)

    # Sync copy to root directory as well
    import shutil
    shutil.copy2(mr_path, ROOT / "matching_results.tsv")
    shutil.copy2(cp_path, ROOT / "candidate_pairs.tsv")
    logger.info("Copied TSVs to repository root as well.")

    # Official validator
    logger.info("Running official submission validator...")
    import subprocess
    val = subprocess.run(
        ["python", "utils/validate_submission.py",
         "--matching", str(mr_path),
         "--candidate", str(cp_path),
         "--test-dir", str(DATASET_ROOT / "test")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    logger.info("Validator output:\n%s", val.stdout.strip())
    if val.returncode != 0:
        logger.error("Validator FAILED: %s", val.stderr.strip())
    else:
        logger.info("Validator: PASS — safe to submit")

    # Package submission zip
    logger.info("Packaging submission zip...")
    pkg = subprocess.run(
        ["python", "package_submission.py"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    logger.info("Packager output:\n%s", pkg.stdout.strip())

    total = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Total runtime: %.2fs (%.2f min)", total, total/60)
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

