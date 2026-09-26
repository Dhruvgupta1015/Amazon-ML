#!/usr/bin/env python3
"""
run_phase16_production_inference.py — Production Test-Set Inference Engine.
Amazon ML Challenge 2026.

Frozen Validated Configuration (from reports/recovery/best_holdout_model.json):
  - Model: Champion Heuristic + Multi-Channel High-Recall Blocking + Bug Fixes + Cardinality Guard
  - Blocking: Exact Alpha (150) + Tokens (150/60) + Rare (80) + Addr (80) + Digits (80) + 4-Grams (60)
  - Candidate Ranking: Composite Similarity (3*name + 2*addr + 4*digits + 5*exact_alpha)
  - Pruning: K = 100 (84.94% Pair Recall, 66.93% Entity-Perfect Recall, 0% Zero-Candidate Entities)
  - Scorer:
      - Bug 1 Fix: Address-Supported Exact Name (prevents unconstrained nationwide false merges)
      - Bug 2 Fix: Accurate State Extraction (proper handling of 'oh', 'me', 'wa', 'pa', 'in')
      - Bug 3 Fix: Robust Street Number Extraction (extracts number from any position in address)
      - Fast conflict checks (state conflict hard rejection, numeric conflict penalty)
  - Decision Thresholds: tau = 0.52, high_confidence = 0.78, addr_floor = 0.22
  - Cardinality Ceiling: Maximum 10 matches per S1 entity (prevents token collision explosion)
  - Output Generation:
      - Exact test_source1.tsv ordering (1,732,544 rows)
      - output/matching_results.tsv
      - output/candidate_pairs.tsv
      - Singletons cleanly output with empty string
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_raw" / "student_resource" / "dataset" / "test"
CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase16Inference")

RE_PUNCT = re.compile(r"[^a-z0-9\s]")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_ALPHA = re.compile(r"[^a-z0-9]")

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
    'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}
ALL_US_STATE_CODES = set(US_STATES.values())

INDIA_STATES = {
    'tamil nadu': 'tn', 'tamilnadu': 'tn', 'karnataka': 'ka', 'maharashtra': 'mh',
    'uttar pradesh': 'up', 'rajasthan': 'rj', 'madhya pradesh': 'mp', 'delhi': 'dl',
    'new delhi': 'dl', 'west bengal': 'wb', 'gujarat': 'gj', 'andhra pradesh': 'ap',
    'telangana': 'ts', 'kerala': 'kl', 'haryana': 'hr', 'bihar': 'br', 'odisha': 'od',
    'punjab': 'pb', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'uttarakhand': 'uk',
    'goa': 'ga', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'chandigarh': 'ch'
}
ALL_IN_STATE_CODES = set(INDIA_STATES.values())

ADDR_ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'dr': 'drive',
    'ln': 'lane', 'blvd': 'boulevard', 'bvd': 'boulevard', 'bd': 'boulevard',
    'ct': 'court', 'pl': 'place', 'sq': 'square', 'pkwy': 'parkway',
    'hwy': 'highway', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment',
    'dept': 'department', 'bldg': 'building', 'opp': 'opposite',
    'nr': 'near', 'adj': 'adjacent', 'ind': 'industrial', 'sec': 'sector',
    'dist': 'district', 'vill': 'village', 'po': 'post office',
    'rue': 'rue', 'allee': 'all', 'impasse': 'imp', 'chemin': 'che', 'route': 'rte'
}

class CompactCand:
    __slots__ = ("eid", "clean_name", "clean_addr", "alpha_name", "name_tokens", "addr_tokens", "digits")
    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str,
                 name_tokens: set, addr_tokens: set, digits: set):
        self.eid = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.name_tokens = name_tokens
        self.addr_tokens = addr_tokens
        self.digits = digits

def fast_clean(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(" ", str(s).lower()).strip()

def fast_alpha(s: str) -> str:
    if not s: return ""
    return RE_ALPHA.sub("", str(s).lower()).strip()

def expand_addr_tokens(addr_clean: str) -> set:
    if not addr_clean: return set()
    return {ADDR_ABBREVIATIONS.get(t, t) for t in addr_clean.split()}

def extract_state_fixed(addr_clean: str, country: str) -> str:
    if not addr_clean: return ""
    words = addr_clean.split()
    if not words: return ""
    n = len(words)
    if country == "US":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_US_STATE_CODES:
                if i <= 2: return w
                if i < n - 1 and len(words[-i]) == 5 and words[-i].isdigit(): return w
                if w not in {"in", "to", "or", "at", "as", "by", "on", "no"}: return w
        for name, code in US_STATES.items():
            if name in addr_clean: return code
    elif country == "India":
        for i, w in enumerate(reversed(words)):
            if len(w) == 2 and w in ALL_IN_STATE_CODES:
                if i <= 2 or w not in {"in", "to", "or", "at", "as"}: return w
        for name, code in INDIA_STATES.items():
            if name in addr_clean: return code
    return ""

def get_street_num_fixed(addr_clean: str) -> str:
    if not addr_clean: return ""
    postal = ""
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6):
            postal = d
            break
    for d in RE_DIGITS.findall(addr_clean):
        if d != postal and len(d) <= 5:
            return d
    return ""

def get_postal_fast(addr_clean: str) -> str:
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6): return d
    return ""

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(" ", "")
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def jaro_winkler(s1: str, s2: str, max_len: int = 40) -> float:
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1
    m1 = [False] * l1; m2 = [False] * l2
    matches = 0
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

def ensure_country_s23_cache(country: str, test_s2_path: Path, test_s3_path: Path) -> Path:
    cache_file = CACHE_DIR / f"test_s23_{country}.tsv"
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        logger.info("[%s] Using cached S2/S3 file: %s (%.1f MB)",
                    country.upper(), cache_file.name, cache_file.stat().st_size / (1024**2))
        return cache_file

    logger.info("[%s] Building country S2/S3 cache for %s...", country.upper(), country)
    t0 = time.time()
    count = 0
    with open(cache_file, "w", encoding="utf-8", newline="") as out_f:
        for s_path in [test_s2_path, test_s3_path]:
            with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        out_f.write(line)
                        count += 1
    logger.info("[%s] Cached %d S2/S3 lines in %.2fs -> %s", country.upper(), count, time.time() - t0, cache_file.name)
    return cache_file

def resolve_country(country: str, s1_records: list, test_s2_path: Path, test_s3_path: Path,
                    tau: float = 0.52, high_conf: float = 0.78, addr_floor: float = 0.22,
                    max_k: int = 100) -> Path:
    ckpt_file = OUTPUT_DIR / f"recovery_checkpoint_{country}.tsv"
    completed_ids = set()
    matched_count = 0
    if ckpt_file.exists() and ckpt_file.stat().st_size > 100:
        with open(ckpt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts:
                    completed_ids.add(parts[0])
                    if len(parts) > 1 and parts[1]:
                        matched_count += 1
        lines = len(completed_ids)
        if lines == len(s1_records):
            logger.info("[%s] Found completed checkpoint (%d entities). Reusing %s!",
                        country.upper(), lines, ckpt_file.name)
            return ckpt_file
        elif lines > 0:
            logger.info("[%s] Found partial checkpoint with %d entities (matched %d). Resuming...",
                        country.upper(), lines, matched_count)

    logger.info("[%s] Loading S2/S3 candidate pool...", country.upper())
    t0 = time.time()
    s23_cache = ensure_country_s23_cache(country, test_s2_path, test_s3_path)

    s23_data = {}
    with open(s23_cache, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                cid, name, addr = parts[0], parts[1], parts[2]
                cn = fast_clean(name)
                ca = fast_clean(addr)
                s23_data[cid] = CompactCand(
                    eid=cid, clean_name=cn, clean_addr=ca,
                    alpha_name=fast_alpha(name), name_tokens=set(cn.split()),
                    addr_tokens=set(ca.split()), digits=set(RE_DIGITS.findall(ca))
                )
    logger.info("[%s] Loaded %d CompactCand records in %.2fs", country.upper(), len(s23_data), time.time() - t0)

    # Build Multi-Channel Inverted Indexes
    t_idx = time.time()
    alpha_idx = defaultdict(list)
    token_idx = defaultdict(list)
    rare_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    digit_idx = defaultdict(list)
    ng4_idx = defaultdict(list)

    token_counts = defaultdict(int)
    for c in s23_data.values():
        for t in c.name_tokens: token_counts[t] += 1

    for cid, c in s23_data.items():
        if c.alpha_name and len(c.alpha_name) >= 3:
            p = alpha_idx[c.alpha_name]
            if len(p) < 150: p.append(cid)
        for t in c.name_tokens:
            if len(t) >= 3:
                freq = token_counts[t]
                p = token_idx[t]
                if len(p) < (60 if freq > 200 else 150): p.append(cid)
                if freq < 40:
                    p_r = rare_idx[t]
                    if len(p_r) < 80: p_r.append(cid)
        for at in c.clean_addr.split():
            if len(at) >= 3:
                p = addr_idx[at]
                if len(p) < 80: p.append(cid)
        for d in RE_DIGITS.findall(c.clean_addr):
            p = digit_idx[d]
            if len(p) < 80: p.append(cid)
        for ng in get_char_ngrams(c.clean_name, 4):
            p = ng4_idx[ng]
            if len(p) < 60: p.append(cid)

    logger.info("[%s] Inverted indexes built in %.2fs. Alpha: %d, Tokens: %d, Addr: %d, Digits: %d, 4-Grams: %d",
                country.upper(), time.time() - t_idx, len(alpha_idx), len(token_idx), len(addr_idx), len(digit_idx), len(ng4_idx))

    # Candidate metadata cache
    cand_meta = {}
    def get_cand_meta(cand: CompactCand):
        meta = cand_meta.get(cand.eid)
        if meta is None:
            st = extract_state_fixed(cand.clean_addr, country)
            sn = get_street_num_fixed(cand.clean_addr)
            po = get_postal_fast(cand.clean_addr)
            meta = (st, sn, po)
            cand_meta[cand.eid] = meta
        return meta

    # Stream resolved entities to checkpoint file (append mode if resuming)
    t1 = time.time()
    total_s1 = len(s1_records)
    processed_this_run = 0

    mode = "a" if completed_ids else "w"
    with open(ckpt_file, mode, encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        for i, s1 in enumerate(s1_records):
            if s1["id"] in completed_ids:
                continue

            s1_alpha = s1["alpha_name"]
            s1_nt = s1["name_tokens"]
            s1_at = s1["addr_tokens"]
            s1_digits = s1["digits"]
            s1_st = s1["state"]
            s1_sn = s1["street_num"]
            s1_po = s1["postal"]
            s1_cn = s1["clean_name"]
            s1_ca = s1["clean_addr"]
            s1_exp_a = s1["expanded_addr"]
            s1_ng4 = s1["ng4"]

            raw = set()
            if s1_alpha: raw.update(alpha_idx.get(s1_alpha, []))
            for t in s1_nt:
                if len(t) >= 3: raw.update(token_idx.get(t, []))
                if len(t) >= 3: raw.update(rare_idx.get(t, []))
            for at in s1_at:
                if len(at) >= 3: raw.update(addr_idx.get(at, []))
            for d in s1_digits: raw.update(digit_idx.get(d, []))
            for ng in s1_ng4: raw.update(ng4_idx.get(ng, []))

            # Composite similarity ranking
            scored_cands = []
            for cid in raw:
                cand = s23_data.get(cid)
                if not cand: continue
                st2, sn2, po2 = get_cand_meta(cand)
                # State conflict fast prune
                if s1_st and st2 and s1_st != st2:
                    continue

                sim_score = 0.0
                if s1_alpha and cand.alpha_name and s1_alpha == cand.alpha_name:
                    sim_score += 5.0
                c_nt = cand.name_tokens
                inter_n = len(s1_nt & c_nt)
                if inter_n: sim_score += 3.0 * (inter_n / (len(s1_nt) + len(c_nt) - inter_n))
                c_at = cand.addr_tokens
                inter_a = len(s1_at & c_at)
                if inter_a: sim_score += 2.0 * (inter_a / (len(s1_at) + len(c_at) - inter_a))
                if s1_digits & cand.digits: sim_score += 2.0
                scored_cands.append((sim_score, cid))

            scored_cands.sort(key=lambda x: x[0], reverse=True)
            cand_list = [cid for _, cid in scored_cands[:max_k]]

            # Decision Engine
            scored_matches = []
            for cid in cand_list:
                cand = s23_data.get(cid)
                if not cand: continue
                st2, sn2, po2 = get_cand_meta(cand)

                # Bug 2 fix: State conflict rejection
                if s1_st and st2 and s1_st != st2:
                    continue

                # Bug 3 fix: Fast street num + postal conflict
                if (s1_sn and sn2 and s1_sn != sn2) and (s1_po and po2 and s1_po != po2):
                    continue

                conflict = 1.0 if (s1_sn and sn2 and s1_sn != sn2) else (0.2 if (s1_sn or sn2) else 0.0)
                postal_conflict = (s1_po != po2) if (s1_po and po2) else False
                postal_match = (s1_po == po2) if (s1_po and po2) else False

                jw = jaro_winkler(s1_cn, cand.clean_name)
                c_nt = cand.name_tokens
                inter_n = len(s1_nt & c_nt)
                tj_name = (inter_n / (len(s1_nt) + len(c_nt) - inter_n)) if (s1_nt or c_nt) else 1.0

                c_exp = expand_addr_tokens(cand.clean_addr)
                inter_a = len(s1_exp_a & c_exp)
                tj_addr = (inter_a / (len(s1_exp_a) + len(c_exp) - inter_a)) if (s1_exp_a or c_exp) else 0.0

                base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.30 * conflict
                if postal_conflict: base -= 0.15
                elif postal_match: base += 0.06

                c_ng4 = get_char_ngrams(cand.clean_name, 4)
                if s1_ng4 and c_ng4:
                    ng_inter = len(s1_ng4 & c_ng4)
                    ng_j = ng_inter / (len(s1_ng4) + len(c_ng4) - ng_inter)
                    if ng_j >= 0.50 and jw >= 0.65:
                        base = max(base, 0.50 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.20 * ng_j - 0.30 * conflict)

                # Bug 1 fix: Require address support for exact alpha name boost
                both_have_addr = bool(s1_ca and cand.clean_addr)
                digits_agree = bool(s1_digits & cand.digits)
                addr_supported = (tj_addr >= 0.08) or digits_agree or (not both_have_addr)

                if s1_alpha and cand.alpha_name and s1_alpha == cand.alpha_name and len(s1_alpha) >= 4 and conflict < 1.0:
                    if addr_supported:
                        base = max(base, 0.40 + 0.35 * jw + 0.25 * tj_addr - 0.20 * conflict)
                    else:
                        base = min(base, 0.65)

                if base >= high_conf or (base >= tau and tj_addr >= addr_floor):
                    scored_matches.append((base, cid))

            # Phase 9: High-Cardinality Protection (sort descending by score, cap at 10 matches)
            scored_matches.sort(key=lambda x: x[0], reverse=True)
            matched = [cid for _, cid in scored_matches[:10]]

            if matched:
                matched_count += 1
            processed_this_run += 1

            writer.writerow([s1["id"], ",".join(matched), ",".join(cand_list)])

            if (i + 1) % 50000 == 0 or (i + 1) == total_s1:
                elapsed = max(time.time() - t1, 0.01)
                rate = processed_this_run / elapsed if processed_this_run else 0
                pct = (matched_count / (i + 1)) * 100
                logger.info("[%s] %d/%d (%.0f ent/s this run) | Matched: %d (%.1f%%)",
                            country.upper(), i + 1, total_s1, rate, matched_count, pct)

    logger.info("[%s] Completed resolution in %.2fs. Matched: %d/%d (%.1f%%)",
                country.upper(), time.time() - t1, matched_count, total_s1,
                (matched_count / total_s1) * 100 if total_s1 else 0)

    del s23_data, alpha_idx, token_idx, rare_idx, addr_idx, digit_idx, ng4_idx, cand_meta
    gc.collect()
    return ckpt_file

def main():
    logger.info("=" * 80)
    logger.info("PHASE 16: LEADERBOARD RECOVERY PRODUCTION INFERENCE ENGINE")
    logger.info("=" * 80)
    t_start = time.time()

    test_s1_path = DATA_DIR / "test_source1.tsv"
    test_s2_path = DATA_DIR / "test_source2.tsv"
    test_s3_path = DATA_DIR / "test_source3.tsv"

    logger.info("Loading test_source1.tsv...")
    s1_by_country = defaultdict(list)
    s1_order = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        country_idx = header.index("country") if "country" in header else 3

        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                eid = parts[id_idx]
                name = parts[name_idx]
                addr = parts[addr_idx]
                country = parts[country_idx]

                cn = fast_clean(name)
                ca = fast_clean(addr)
                rec = {
                    "id": eid, "clean_name": cn, "clean_addr": ca,
                    "alpha_name": fast_alpha(name), "country": country,
                    "state": extract_state_fixed(ca, country),
                    "street_num": get_street_num_fixed(ca),
                    "postal": get_postal_fast(ca),
                    "name_tokens": set(cn.split()), "addr_tokens": set(ca.split()),
                    "expanded_addr": expand_addr_tokens(ca), "digits": set(RE_DIGITS.findall(ca)),
                    "ng4": get_char_ngrams(cn, 4)
                }
                s1_by_country[country].append(rec)
                s1_order.append(eid)

    total_entities = len(s1_order)
    logger.info("Loaded %d test Source-1 entities across countries: %s",
                total_entities, {c: len(records) for c, records in s1_by_country.items()})

    # Resolve each country
    # Countries in order of size: France (259K), US (663K), India (810K)
    country_order = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))
    for c in country_order:
        resolve_country(c, s1_by_country[c], test_s2_path, test_s3_path,
                        tau=0.52, high_conf=0.78, addr_floor=0.22, max_k=100)

    # Merge checkpoints in exact original test_source1 order
    logger.info("=" * 80)
    logger.info("MERGING ALL CHECKPOINTS INTO FINAL SUBMISSION FILES IN EXACT TEST ORDER")
    logger.info("=" * 80)

    results_map = {} # eid -> (matched_str, cand_str)
    for c in s1_by_country.keys():
        ckpt_file = OUTPUT_DIR / f"recovery_checkpoint_{c}.tsv"
        logger.info("Loading %s...", ckpt_file.name)
        with open(ckpt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip("\r\n").split("\t")
                eid = parts[0]
                matched = parts[1] if len(parts) > 1 else ""
                cands = parts[2] if len(parts) > 2 else ""
                results_map[eid] = (matched, cands)

    mr_path = OUTPUT_DIR / "matching_results.tsv"
    cp_path = OUTPUT_DIR / "candidate_pairs.tsv"

    logger.info("Writing %s and %s...", mr_path.name, cp_path.name)
    total_written = 0
    empty_written = 0

    with open(mr_path, "w", encoding="utf-8", newline="") as f_mr, \
         open(cp_path, "w", encoding="utf-8", newline="") as f_cp:
        w_mr = csv.writer(f_mr, delimiter="\t")
        w_cp = csv.writer(f_cp, delimiter="\t")

        w_mr.writerow(["source1_entity_id", "matched_entity_ids"])
        w_cp.writerow(["source1_entity_id", "candidate_entity_ids"])

        for eid in s1_order:
            m_str, c_str = results_map.get(eid, ("", ""))
            w_mr.writerow([eid, m_str])
            w_cp.writerow([eid, c_str])
            total_written += 1
            if not m_str:
                empty_written += 1

    logger.info("Successfully assembled submission files:")
    logger.info("  Total S1 rows written: %d (100%% identical to test_source1.tsv)", total_written)
    logger.info("  Singletons (empty):    %d (%.2f%%)", empty_written, (empty_written / total_written) * 100)
    logger.info("  Non-empty predictions: %d (%.2f%%)", total_written - empty_written, ((total_written - empty_written) / total_written) * 100)
    logger.info("Total pipeline runtime: %.2fs", time.time() - t_start)

    # Copy output files to root directory for easy access
    logger.info("Copying final submission files to root directory...")
    shutil.copy(mr_path, ROOT / "matching_results.tsv")
    shutil.copy(mr_path, ROOT / "submission.tsv")
    shutil.copy(cp_path, ROOT / "candidate_pairs.tsv")

    # Run official packaging to generate Resolve_AI_Team_submission.zip
    logger.info("Generating Resolve_AI_Team_submission.zip...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "package_submission.py")], check=True)
    logger.info("Phase 16 full dataset production inference successfully finished!")

if __name__ == "__main__":
    main()
