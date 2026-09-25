#!/usr/bin/env python3
"""
run_production_resolution.py - Ultra-Fast & High-Precision Production Entity Resolution Engine.
Amazon ML Challenge 2026.
"""
from __future__ import annotations
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
OUTPUT_DIR = ROOT / "output"
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ResolveAI")

# --- SUFFIXES & DICTIONARIES ---
LEGAL_SUFFIXES = [
    'private limited', 'pvt limited', 'p limited', 'private ltd', 'pvt ltd',
    'corporation', 'incorporated', 'limited', 'enterprises', 'enterprise',
    'services', 'solutions', 'technologies', 'holdings', 'industries',
    'international', 'consultants', 'consultancy', 'consulting', 'center',
    'centre', 'corp', 'inc', 'llc', 'llp', 'sarl', 'sasu', 'eurl', 'gmbh',
    'gie', 'sas', 'sa', 'ag', 'bv', 'nv', 'spa', 'srl'
]

STOPWORDS = {
    'the', 'and', 'co', 'of', 'in', 'for', 'at', 'by', 'to', 'a', 'an',
    'de', 'la', 'le', 'les', 'des', 'du', 'et', 'en', 'au', 'aux', 'd', 'l',
    'un', 'une', 'sur', 'dans', 'null', 'mr', 'mrs', 'dr', 'prof'
}

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
RE_US_STATE_NAMES = re.compile(r'\b(' + '|'.join(sorted(US_STATES.keys(), key=len, reverse=True)) + r')\b')

INDIA_STATES = {
    'tamil nadu': 'tn', 'tamilnadu': 'tn', 'karnataka': 'ka', 'maharashtra': 'mh',
    'uttar pradesh': 'up', 'rajasthan': 'rj', 'madhya pradesh': 'mp', 'delhi': 'dl',
    'new delhi': 'dl', 'west bengal': 'wb', 'gujarat': 'gj', 'andhra pradesh': 'ap',
    'telangana': 'ts', 'kerala': 'kl', 'haryana': 'hr', 'bihar': 'br', 'odisha': 'od',
    'punjab': 'pb', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'uttarakhand': 'uk',
    'goa': 'ga', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'chandigarh': 'ch'
}
ALL_IN_STATE_CODES = set(INDIA_STATES.values())
RE_IN_STATE_NAMES = re.compile(r'\b(' + '|'.join(sorted(INDIA_STATES.keys(), key=len, reverse=True)) + r')\b')

ADDR_EXPANSIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'ter': 'terrace',
    'pkwy': 'parkway', 'hwy': 'highway', 'sq': 'square', 'bldg': 'building',
    'fl': 'floor', 'apt': 'apartment', 'ste': 'suite', 'rm': 'room',
    'ngr': 'nagar', 'col': 'colony', 'sec': 'sector', 'dist': 'district',
    'tq': 'taluk', 'opp': 'opposite', 'nr': 'near', 'bvd': 'boulevard',
    'bd': 'boulevard', 'av': 'avenue', 'all': 'allee', 'imp': 'impasse',
    'che': 'chemin', 'rte': 'route', 'crs': 'cours', 'pl.': 'place',
    'st.': 'street', 'rd.': 'road', 'ave.': 'avenue', 'blvd.': 'boulevard'
}

GENERIC_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'lane', 'court', 'place', 'terrace',
    'parkway', 'highway', 'square', 'building', 'floor', 'apt', 'apartment', 'suite', 'room',
    'nagar', 'colony', 'sector', 'district', 'taluk', 'opp', 'opposite', 'near', 'rd', 'st',
    'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'ter', 'pkwy', 'hwy', 'sq', 'bldg', 'fl', 'ste',
    'rm', 'ngr', 'col', 'sec', 'dist', 'tq', 'plot', 'hn', 'house', 'no', 'number', 'unit',
    'block', 'null', 'east', 'west', 'north', 'south', 'box', 'po', 'pob', 'rue', 'allee',
    'chemin', 'boulevard', 'impasse', 'france', 'india', 'us', 'usa', 'city', 'town', 'state',
    'village', 'route', 'cours', 'quai', 'passage'
} | ALL_US_STATE_CODES | ALL_IN_STATE_CODES | set(US_STATES.keys()) | set(INDIA_STATES.keys())

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_ALPHA_ONLY = re.compile(r'[^a-z0-9]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-z0-9\-]+)\.(?:com|in|org|net|fr|co|io|biz|info|us)\b', re.IGNORECASE)


def transliterate(text: str) -> str:
    if not text: return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def is_indic(text: str) -> bool:
    for c in str(text):
        if 0x0900 <= ord(c) <= 0x0D7F:
            return True
    return False


def clean_name(raw: str) -> tuple[str, str, str, bool]:
    if not raw or not isinstance(raw, str):
        return "", "", "", False
    has_in = is_indic(raw)
    raw_clean = transliterate(raw).lower()
    
    m = DOMAIN_RE.search(raw_clean)
    domain_stem = m.group(1).replace('-', '').replace('.', '') if m else ""
    
    t = raw_clean.replace('&', ' and ')
    t = RE_PUNCT.sub(' ', t)
    t = ' '.join(t.split())
    
    for suf in LEGAL_SUFFIXES:
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
            break
        elif t == suf:
            t = ""
            break
    
    alpha_clean = RE_ALPHA_ONLY.sub('', t)
    return t, alpha_clean, domain_stem, has_in


def extract_state(t_clean: str, country: str) -> str:
    if not t_clean: return ""
    if country == 'US':
        m = RE_US_STATE_NAMES.search(t_clean)
        if m: return US_STATES[m.group(1)]
        words = t_clean.split()
        for w in reversed(words):
            if w in ALL_US_STATE_CODES and w not in ('st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'me', 'oh', 'ok', 'or', 'pa', 'wa'):
                return w
    elif country == 'India':
        m = RE_IN_STATE_NAMES.search(t_clean)
        if m: return INDIA_STATES[m.group(1)]
        words = t_clean.split()
        for w in reversed(words):
            if w in ALL_IN_STATE_CODES and w not in ('st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'or'):
                return w
    return ""


def clean_address(raw: str, country: str) -> tuple[str, set[str], set[str], set[str], str]:
    if not raw or not isinstance(raw, str):
        return "", set(), set(), set(), ""
    t_clean = transliterate(raw).lower()
    state_code = extract_state(t_clean, country)
    
    t = t_clean.replace('&', ' and ')
    t = RE_PUNCT.sub(' ', t)
    words = t.split()
    expanded = [ADDR_EXPANSIONS.get(w, w) for w in words]
    norm_addr = ' '.join(expanded)
    
    all_digits = set(RE_DIGITS.findall(norm_addr))
    all_tokens = set(expanded) - STOPWORDS
    specific_tokens = {w for w in all_tokens if w not in GENERIC_ADDR_WORDS and len(w) >= 3}
    
    return norm_addr, all_tokens, specific_tokens, all_digits, state_code


def jaro_winkler(s1: str, s2: str, max_len: int = 50) -> float:
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1
    m1 = [False] * l1
    m2 = [False] * l2
    matches = 0
    for i in range(l1):
        for j in range(max(0, i - md), min(i + md + 1, l2)):
            if m2[j] or s1[i] != s2[j]: continue
            m1[i] = m2[j] = True
            matches += 1
            break
    if not matches: return 0.0
    t = 0; k = 0
    for i in range(l1):
        if not m1[i]: continue
        while not m2[k]: k += 1
        if s1[i] != s2[k]: t += 1
        k += 1
    sim = (matches / l1 + matches / l2 + (matches - t / 2) / matches) / 3
    p = sum(1 for i in range(min(4, l1, l2)) if s1[i] == s2[i])
    return sim + p * 0.1 * (1.0 - sim)


def evaluate_pair(s1: dict, s2: dict) -> bool:
    """Production Precision-Gated Matcher."""
    # 1. State conflict check
    st1, st2 = s1["state"], s2["state"]
    if st1 and st2 and st1 != st2:
        return False

    cname1, cname2 = s1["clean_name"], s2["clean_name"]
    alpha1, alpha2 = s1["alpha_name"], s2["alpha_name"]
    
    spec1, spec2 = s1["specific_addr"], s2["specific_addr"]
    spec_inter = spec1 & spec2
    num_spec_inter = len(spec_inter)
    
    dig1, dig2 = s1["digits"], s2["digits"]
    digits_overlap = bool(dig1 & dig2)
    has_both_digits = bool(dig1 and dig2)
    digits_conflict = bool(has_both_digits and not digits_overlap)

    # 2. Exact Name / High-Fidelity Domain Match
    if alpha1 and alpha2:
        if alpha1 == alpha2:
            if not digits_conflict or num_spec_inter >= 1:
                return True
        elif len(alpha1) >= 8 and len(alpha2) >= 8 and abs(len(alpha1) - len(alpha2)) <= 2:
            if alpha1 in alpha2 or alpha2 in alpha1:
                if not digits_conflict:
                    return True

    dom1, dom2 = s1["domain_stem"], s2["domain_stem"]
    if dom2 and len(dom2) >= 5:
        if dom2 == alpha1 or (len(dom2) >= 8 and (dom2 in alpha1 or alpha1 in dom2)):
            if not digits_conflict:
                return True
    if dom1 and len(dom1) >= 5:
        if dom1 == alpha2 or (len(dom1) >= 8 and (dom1 in alpha2 or alpha2 in dom1)):
            if not digits_conflict:
                return True

    # 3. String Similarities
    jw_n = jaro_winkler(cname1, cname2) if (cname1 and cname2) else 0.0
    ntoks1, ntoks2 = s1["name_tokens"], s2["name_tokens"]
    name_inter = ntoks1 & ntoks2
    num_name_inter = len(name_inter)
    min_tokens = min(len(ntoks1), len(ntoks2)) if (ntoks1 and ntoks2) else 0

    # CASE A: High Confidence Name (Minor typos / abbreviations)
    if jw_n >= 0.94:
        if not digits_conflict:
            if not spec1 or not spec2 or num_spec_inter >= 1 or digits_overlap or jw_n >= 0.98:
                return True

    if jw_n >= 0.88 and min_tokens >= 2 and num_name_inter >= min_tokens:
        if not digits_conflict:
            if not spec1 or not spec2 or num_spec_inter >= 1 or digits_overlap:
                return True

    # CASE B: Strong Address Match (Matching specific street name/city + matching digits)
    if (num_spec_inter >= 2 or (num_spec_inter >= 1 and digits_overlap)) and not digits_conflict:
        if s1["has_indic"] or s2["has_indic"]:
            return True
        if jw_n >= 0.70 or num_name_inter >= 1:
            return True

    # CASE C: Indic Script Name with Street / City Match
    if (s1["has_indic"] or s2["has_indic"]) and not digits_conflict:
        if num_spec_inter >= 2 or (num_spec_inter >= 1 and digits_overlap):
            return True

    return False


def resolve_country_test(country: str, s1_records: list[dict], test_s2_path: Path, test_s3_path: Path) -> dict:
    logger.info("[%s] Loading S2/S3 candidate pool...", country.upper())
    t0 = time.time()
    
    s23_data = {}
    name_idx = defaultdict(list)
    spec_addr_idx = defaultdict(list)
    digits_idx = defaultdict(list)
    domain_idx = defaultdict(list)
    
    total_loaded = 0
    for s_path in [test_s2_path, test_s3_path]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    cid, name, addr = parts[0], parts[1], parts[2]
                    cn, alpha, dom, indic = clean_name(name)
                    norm_a, atoks, spectoks, digs, st = clean_address(addr, country)
                    ntoks = set(cn.split()) - STOPWORDS
                    
                    s23_data[cid] = {
                        "id": cid, "clean_name": cn, "alpha_name": alpha, "domain_stem": dom,
                        "has_indic": indic, "name_tokens": ntoks,
                        "clean_addr": norm_a, "all_addr": atoks,
                        "specific_addr": spectoks, "digits": digs, "state": st, "country": country
                    }
                    for t in ntoks:
                        if len(t) >= 3:
                            name_idx[t].append(cid)
                    for at in spectoks:
                        spec_addr_idx[at].append(cid)
                    for d in digs:
                        if len(d) >= 3:
                            digits_idx[d].append(cid)
                    if dom:
                        domain_idx[dom].append(cid)
                    
                    total_loaded += 1
    
    logger.info("[%s] Loaded & indexed %d S2/S3 mentions in %.2fs", country.upper(), total_loaded, time.time() - t0)
    
    # Resolve S1 records
    logger.info("[%s] Resolving %d S1 entities...", country.upper(), len(s1_records))
    t_res = time.time()
    results = {}
    matched_count = 0
    
    for i, s1 in enumerate(s1_records):
        s1_id = s1["id"]
        cands = set()
        
        # Candidate retrieval
        for t in s1["name_tokens"]:
            if len(t) >= 3:
                p = name_idx.get(t, [])
                if len(p) <= 250:
                    cands.update(p)
        for at in s1["specific_addr"]:
            p = spec_addr_idx.get(at, [])
            if len(p) <= 150:
                cands.update(p)
        for d in s1["digits"]:
            if len(d) >= 3:
                p = digits_idx.get(d, [])
                if len(p) <= 100:
                    cands.update(p)
        if s1["domain_stem"]:
            p = domain_idx.get(s1["domain_stem"], [])
            cands.update(p)

        cand_list = list(cands)[:120]
        
        # Match decision
        matched = []
        for cid in cand_list:
            cand = s23_data[cid]
            if evaluate_pair(s1, cand):
                matched.append(cid)
                
        results[s1_id] = (",".join(matched), ",".join(cand_list))
        if matched:
            matched_count += 1
            
        if (i + 1) % 100000 == 0 or (i + 1) == len(s1_records):
            elapsed = time.time() - t_res
            rate = (i + 1) / max(elapsed, 0.001)
            logger.info("[%s] %d/%d (%.0f entities/s) | Matched: %d (%.1f%%)",
                        country.upper(), i + 1, len(s1_records), rate, matched_count, matched_count / (i + 1) * 100)

    logger.info("[%s] Finished resolution: %d/%d matched (%.1f%%) in %.2fs",
                country.upper(), matched_count, len(s1_records), matched_count / len(s1_records) * 100, time.time() - t_res)
    
    # Cleanup memory
    del s23_data, name_idx, spec_addr_idx, digits_idx, domain_idx
    gc.collect()
    
    return results


def main():
    logger.info("=" * 80)
    logger.info("  RESOLVE.AI: PRODUCTION HIGH-PRECISION RESOLUTION ENGINE")
    logger.info("  Target: AMAZON ML CHALLENGE 2026 LEADERBOARD TOP-10")
    logger.info("=" * 80)
    t_start = time.time()

    test_s1_path = DATASET_ROOT / "test" / "test_source1.tsv"
    test_s2_path = DATASET_ROOT / "test" / "test_source2.tsv"
    test_s3_path = DATASET_ROOT / "test" / "test_source3.tsv"

    # 1. Read test_source1.tsv
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
                cn, alpha, dom, indic = clean_name(name)
                norm_a, atoks, spectoks, digs, st = clean_address(addr, country)
                ntoks = set(cn.split()) - STOPWORDS
                country_s1[country].append({
                    "id": eid, "clean_name": cn, "alpha_name": alpha, "domain_stem": dom,
                    "has_indic": indic, "name_tokens": ntoks,
                    "clean_addr": norm_a, "all_addr": atoks,
                    "specific_addr": spectoks, "digits": digs, "state": st, "country": country
                })

    logger.info("Loaded %d S1 test entities across %d countries:", len(all_s1_ids), len(country_s1))
    for c, recs in country_s1.items():
        logger.info("  - %s: %d (%.1f%%)", c, len(recs), len(recs) / len(all_s1_ids) * 100)

    # 2. Resolve country by country
    all_results = {}
    for country in sorted(country_s1.keys()):
        c_results = resolve_country_test(country, country_s1[country], test_s2_path, test_s3_path)
        all_results.update(c_results)

    # 3. Write submission TSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mr_path = OUTPUT_DIR / "matching_results.tsv"
    cp_path = OUTPUT_DIR / "candidate_pairs.tsv"

    logger.info("Writing output TSVs in exact S1 order...")
    non_empty_count = 0
    with open(mr_path, "w", encoding="utf-8") as f_mr, \
         open(cp_path, "w", encoding="utf-8") as f_cp:
        f_mr.write("source1_entity_id\tmatched_entity_ids\n")
        f_cp.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            matched, candidates = all_results.get(s1_id, ("", ""))
            if matched:
                non_empty_count += 1
            f_mr.write(f"{s1_id}\t{matched}\n")
            f_cp.write(f"{s1_id}\t{candidates}\n")

    logger.info("Generated %d non-empty predictions out of %d (%.2f%%)",
                non_empty_count, len(all_s1_ids), non_empty_count / len(all_s1_ids) * 100)

    # Copy to root as well
    shutil.copy2(mr_path, ROOT / "matching_results.tsv")
    shutil.copy2(cp_path, ROOT / "candidate_pairs.tsv")
    logger.info("Copied TSVs to repository root.")

    # 4. Validate output
    logger.info("Running official submission validator...")
    val = subprocess.run(
        ["python", "utils/validate_submission.py",
         "--matching", str(mr_path),
         "--candidate", str(cp_path),
         "--test-dir", str(DATASET_ROOT / "test")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    logger.info("Validator Output:\n%s", val.stdout.strip())
    if val.returncode != 0:
        logger.error("Validator FAILED: %s", val.stderr.strip())
    else:
        logger.info("Submission Validation: PASSED")

    # 5. Package final submission zip
    logger.info("Packaging final submission zip...")
    pkg = subprocess.run(
        ["python", "package_submission.py"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    logger.info("Packager Output:\n%s", pkg.stdout.strip())

    total_time = time.time() - t_start
    logger.info("=" * 80)
    logger.info("SUCCESS: Full Resolution Complete in %.2fs (%.2f min)", total_time, total_time / 60)
    logger.info("Final submission file: Resolve_AI_Team_submission.zip")
    logger.info("Leaderboard upload file: output/matching_results.tsv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
