#!/usr/bin/env python3
"""
scripts/run_200k_diagnostic_suite.py - Diagnostic Suite & Blocker Ablation on 200K Benchmark.
Amazon ML Challenge 2026.

Executes:
- Phase 0: Baseline Champion Evaluation on 200K Diagnostic Set -> reports/diagnostic/champion_200k_baseline.json
- Phase 3: Current Production Pipeline Verification -> reports/diagnostic/current_pipeline_200k.json
- Phase 4: Stage-by-Stage Failure Decomposition -> reports/diagnostic/error_stage_distribution.csv
- Phase 5: Comprehensive Blocker Channel Ablation -> reports/diagnostic/blocker_ablation_200k.csv
- Phase 6: Candidate Pruning & Ranking Benchmark (K=25..300)
- Phase 9: Singleton Error Root Cause Categorization -> reports/diagnostic/singleton_errors_200k.csv
"""
from __future__ import annotations
import csv
import gc
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostic_200k"
REPORTS_DIAG = ROOT / "reports" / "diagnostic"
REPORTS_DIR = ROOT / "reports"

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DiagnosticSuite200k")

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


class CompactCand:
    __slots__ = ('id', 'clean_name', 'clean_addr', 'alpha_name', 'state', 'street_num', 'postal', 'name_tokens', 'domain')

    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str,
                 state: str, street_num: str, postal: str, name_tokens: tuple[str, ...], domain: str = ""):
        self.id = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.state = state
        self.street_num = street_num
        self.postal = postal
        self.name_tokens = name_tokens
        self.domain = domain


def clean_text(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(' ', str(s).lower()).strip()


def expand_addr_tokens(clean_addr_str: str) -> set[str]:
    raw_tokens = clean_addr_str.split()
    expanded = set()
    for t in raw_tokens:
        expanded.add(ADDR_ABBREVIATIONS.get(t, t))
    return expanded


def extract_alpha_clean(raw_name: str) -> str:
    if not raw_name: return ""
    t = clean_text(raw_name)
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
        elif t == suf:
            t = ""
    return RE_ALPHA.sub('', t)


def extract_domain(raw_name: str) -> str:
    m = DOMAIN_RE.search(raw_name.lower())
    return m.group(1).replace('-', '') if m else ""


def extract_state(addr_clean: str, country: str) -> str:
    if not addr_clean: return ""
    ambiguous_us = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'me', 'oh', 'ok', 'or', 'pa', 'wa'}
    ambiguous_in = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'or'}
    if country == 'US':
        for name, code in US_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', addr_clean): return code
        for code in reversed(re.findall(r'\b([a-z]{2})\b', addr_clean)):
            if code in ALL_US_STATE_CODES and code not in ambiguous_us: return code
    elif country == 'India':
        for name, code in INDIA_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', addr_clean): return code
        for code in reversed(re.findall(r'\b([a-z]{2})\b', addr_clean)):
            if code in ALL_IN_STATE_CODES and code not in ambiguous_in: return code
    return ""


def get_street_num(addr_clean: str) -> str:
    m = re.match(r'^(\d+)\b', addr_clean or "")
    return m.group(1) if m else ""


def get_postal(addr_clean: str) -> str:
    for d in RE_DIGITS.findall(addr_clean or ""):
        if len(d) in (5, 6): return d
    return ""


def get_char_ngrams(text: str, n: int) -> set[str]:
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def preprocess_s1(eid: str, raw_name: str, raw_addr: str, country: str) -> dict:
    cn = clean_text(raw_name)
    ca = clean_text(raw_addr)
    alpha = extract_alpha_clean(raw_name)
    state = extract_state(ca, country)
    return {
        'id': eid,
        'clean_name': cn,
        'clean_addr': ca,
        'alpha_name': alpha,
        'state': state,
        'name_tokens': set(cn.split()),
        'addr_tokens': set(ca.split()),
        'expanded_addr': expand_addr_tokens(ca),
        'street_num': get_street_num(ca),
        'postal': get_postal(ca),
        'country': country,
        'ngrams3': get_char_ngrams(cn, 3),
        'ngrams4': get_char_ngrams(cn, 4),
        'domain': extract_domain(raw_name)
    }


def preprocess_cand(eid: str, raw_name: str, raw_addr: str, country: str) -> CompactCand:
    cn = clean_text(raw_name)
    ca = clean_text(raw_addr)
    alpha = extract_alpha_clean(raw_name)
    state = extract_state(ca, country)
    return CompactCand(
        eid=eid,
        clean_name=cn,
        clean_addr=ca,
        alpha_name=alpha,
        state=state,
        street_num=get_street_num(ca),
        postal=get_postal(ca),
        name_tokens=tuple(cn.split()),
        domain=extract_domain(raw_name)
    )


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


def token_jaccard(t1: set, t2: set) -> float:
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)


def ngram_jaccard(ng1: set, ng2: set) -> float:
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def score_pair_full(s1: dict, cand: CompactCand) -> tuple[float, float, str]:
    """Scores pair and returns (score, addr_jaccard, failure_reason_if_any)."""
    # 1. State conflict hard rejection
    st1, st2 = s1["state"], cand.state
    if st1 and st2 and st1 != st2:
        return -1.0, 0.0, "state_conflict"

    # 2. Fast conflict early-exit
    sn1, sn2 = s1["street_num"], cand.street_num
    po1, po2 = s1["postal"], cand.postal
    if (sn1 and sn2 and sn1 != sn2) and (po1 and po2 and po1 != po2):
        return -1.0, 0.0, "street_postal_conflict"

    conflict = 1.0 if (sn1 and sn2 and sn1 != sn2) else (0.2 if (sn1 or sn2) else 0.0)
    postal_conflict = (po1 != po2) if (po1 and po2) else False
    postal_match = (po1 == po2) if (po1 and po2) else False

    # 3. String similarities
    jw = jaro_winkler(s1["clean_name"], cand.clean_name)
    inter_name = s1["name_tokens"].intersection(cand.name_tokens)
    len_inter_name = len(inter_name)
    len_union_name = len(s1["name_tokens"]) + len(cand.name_tokens) - len_inter_name
    tj_name = (len_inter_name / len_union_name) if len_union_name else 1.0

    cand_addr_tokens = set(cand.clean_addr.split())
    cand_expanded_addr = expand_addr_tokens(cand.clean_addr)
    tj_addr_raw = token_jaccard(s1["addr_tokens"], cand_addr_tokens)
    tj_addr_exp = token_jaccard(s1["expanded_addr"], cand_expanded_addr)
    tj_addr = max(tj_addr_raw, tj_addr_exp)

    cand_ngrams4 = get_char_ngrams(cand.clean_name, 4)
    ng4 = ngram_jaccard(s1["ngrams4"], cand_ngrams4)
    base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.30 * conflict

    if postal_conflict: base -= 0.15
    elif postal_match: base += 0.06

    if ng4 >= 0.50 and jw >= 0.65:
        base = max(base, 0.50 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.20 * ng4 - 0.30 * conflict)

    a1, a2 = s1["alpha_name"], cand.alpha_name
    if a1 and a2 and a1 == a2 and len(a1) >= 4 and conflict < 1.0:
        base = max(base, 0.40 + 0.35 * jw + 0.25 * tj_addr - 0.20 * conflict)

    if a1 and a2 and len(a1) >= 8 and len(a2) >= 8 and (a1 in a2 or a2 in a1) and conflict < 1.0:
        base = max(base, 0.35 + 0.35 * jw + 0.20 * tj_addr + 0.10 * ng4 - 0.20 * conflict)

    return base, tj_addr, ""


def entity_f05(pred_set: set[str], true_set: set[str]) -> float:
    if not pred_set and not true_set:
        return 1.0 # Exact true singleton match
    if not pred_set or not true_set:
        return 0.0 # Singleton false merge or complete false negative
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def build_channel_indexes(cands: dict[str, CompactCand], token_freq: dict[str, int]) -> dict:
    idx_exact = defaultdict(list)
    idx_ng3 = defaultdict(list)
    idx_ng4 = defaultdict(list)
    idx_name = defaultdict(list)
    idx_rare = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_snum = defaultdict(list)
    idx_postal = defaultdict(list)
    idx_domain = defaultdict(list)

    for cid, c in cands.items():
        if c.alpha_name and len(c.alpha_name) >= 3:
            p = idx_exact[c.alpha_name]
            if len(p) <= 50: p.append(cid)

        for t in c.name_tokens:
            p = idx_name[t]
            if len(p) <= 60: p.append(cid)
            if token_freq.get(t, 0) < 50:
                p_r = idx_rare[t]
                if len(p_r) <= 30: p_r.append(cid)

        for at in c.clean_addr.split():
            p = idx_addr[at]
            if len(p) <= 40: p.append(cid)

        if c.street_num:
            p = idx_snum[c.street_num]
            if len(p) <= 40: p.append(cid)

        if c.postal:
            p = idx_postal[c.postal]
            if len(p) <= 30: p.append(cid)

        if c.domain:
            p = idx_domain[c.domain]
            if len(p) <= 30: p.append(cid)

        for ng in get_char_ngrams(c.clean_name, 3):
            p = idx_ng3[ng]
            if len(p) <= 30: p.append(cid)

        for ng in get_char_ngrams(c.clean_name, 4):
            p = idx_ng4[ng]
            if len(p) <= 30: p.append(cid)

    # Prune lists exceeding caps
    return {
        "exact_name": {k: v for k, v in idx_exact.items() if len(v) <= 50},
        "token_overlap": {k: v for k, v in idx_name.items() if len(v) <= 60},
        "rare_token": {k: v for k, v in idx_rare.items() if len(v) <= 30},
        "addr_token": {k: v for k, v in idx_addr.items() if len(v) <= 40},
        "street_number": {k: v for k, v in idx_snum.items() if len(v) <= 40},
        "postal_code": {k: v for k, v in idx_postal.items() if len(v) <= 30},
        "domain_stem": {k: v for k, v in idx_domain.items() if len(v) <= 30},
        "char_3gram": {k: v for k, v in idx_ng3.items() if len(v) <= 30},
        "char_4gram": {k: v for k, v in idx_ng4.items() if len(v) <= 30},
    }


def main():
    logger.info("=" * 80)
    logger.info("  AMAZON ML CHALLENGE 2026: 200K DIAGNOSTIC BENCHMARK SUITE")
    logger.info("  Strict Phase 0 -> Phase 6 Execution & Failure Auditing")
    logger.info("=" * 80)
    t_start = time.time()
    REPORTS_DIAG.mkdir(parents=True, exist_ok=True)

    # Load active champion parameters
    with open(REPORTS_DIR / "champion.json") as f:
        champion_cfg = json.load(f)
    tau = champion_cfg.get("tau", 0.53)
    high_conf = champion_cfg.get("high_confidence", 0.75)
    addr_floor = champion_cfg.get("addr_floor", 0.22)
    max_k = champion_cfg.get("max_k", 130)

    # 1. Load 200K Ground Truth
    logger.info("Loading 200K diagnostic ground truth...")
    gt = {}
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            m_str = parts[1] if len(parts) > 1 else ""
            m_set = set(x.strip() for x in m_str.split(",") if x.strip())
            gt[s1_id] = m_set

    total_singletons = sum(1 for s in gt.values() if len(s) == 0)
    total_non_singletons = len(gt) - total_singletons
    total_true_matches = sum(len(s) for s in gt.values())
    logger.info("Loaded GT: %d S1 entities | Singletons: %d (%.2f%%) | Non-singletons: %d | True matches: %d",
                len(gt), total_singletons, total_singletons / len(gt) * 100, total_non_singletons, total_true_matches)

    # 2. Load 200K Source 1 records by country
    logger.info("Loading 200K diagnostic Source 1 records...")
    s1_by_country = defaultdict(list)
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_by_country[country].append(preprocess_s1(eid, name, addr, country))

    for c, recs in s1_by_country.items():
        logger.info("  - %s: %d entities", c, len(recs))

    # 3. Load Diagnostic S2 & S3 by country
    logger.info("Loading diagnostic S2/S3 candidate pool...")
    s23_by_country = defaultdict(dict)
    for src_file in ["source2.tsv", "source3.tsv"]:
        with open(DIAG_DIR / src_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    s23_by_country[country][cid] = preprocess_cand(cid, name, addr, country)

    for c, cands in s23_by_country.items():
        logger.info("  - %s: %d S2/S3 candidate records", c, len(cands))

    # =========================================================================
    # PHASE 0 & 3: Run Current Champion on 200K Diagnostic Set
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PHASE 0 & 3: Evaluating Champion Model on 200K Diagnostic Set...")
    logger.info("Parameters: tau=%.2f, high_conf=%.2f, addr_floor=%.2f, max_k=%d",
                tau, high_conf, addr_floor, max_k)
    logger.info("=" * 80)

    f05_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    false_merges = 0
    missed_matches = 0
    retrieved_true_matches = 0
    entity_perfect_retrieval = 0
    total_candidates_generated = 0

    failure_reasons = Counter()
    singleton_error_records = []
    channel_hit_counter = Counter()

    t_eval_start = time.time()
    all_s1_records = s1_by_country["US"] + s1_by_country["India"]

    for country in ["US", "India"]:
        recs = s1_by_country[country]
        cands_pool = s23_by_country[country]
        
        token_freq = defaultdict(int)
        for c in cands_pool.values():
            for t in c.name_tokens: token_freq[t] += 1

        logger.info("[%s] Building multi-channel inverted indexes...", country)
        channels = build_channel_indexes(cands_pool, token_freq)

        logger.info("[%s] Resolving %d entities...", country, len(recs))
        t_country = time.time()
        for i, s1 in enumerate(recs):
            true_m = gt[s1["id"]]
            is_singleton = len(true_m) == 0

            # Multi-channel retrieval
            raw_cands = set()
            for t in s1["name_tokens"]:
                p = channels["token_overlap"].get(t)
                if p: raw_cands.update(p)
                p_r = channels["rare_token"].get(t)
                if p_r: raw_cands.update(p_r)
            for at in s1["addr_tokens"]:
                p = channels["addr_token"].get(at)
                if p: raw_cands.update(p)
            if s1["street_num"]:
                p = channels["street_number"].get(s1["street_num"])
                if p: raw_cands.update(p)
            if s1["postal"]:
                p = channels["postal_code"].get(s1["postal"])
                if p: raw_cands.update(p)
            for ng in s1["ngrams4"]:
                p = channels["char_4gram"].get(ng)
                if p: raw_cands.update(p)
            if s1["alpha_name"]:
                p = channels["exact_name"].get(s1["alpha_name"])
                if p: raw_cands.update(p)
            if s1["domain"]:
                p = channels["domain_stem"].get(s1["domain"])
                if p: raw_cands.update(p)

            # Candidate retrieval stats
            retrieved_cands = list(raw_cands)
            total_candidates_generated += len(retrieved_cands)
            
            # True match retrieval stats (Blocking recall)
            if not is_singleton:
                tp_retrieved = len(set(retrieved_cands) & true_m)
                retrieved_true_matches += tp_retrieved
                if tp_retrieved == len(true_m):
                    entity_perfect_retrieval += 1

                # Track missed true matches in blocking
                missed_in_blocking = true_m - set(retrieved_cands)
                for mid in missed_in_blocking:
                    failure_reasons["1_not_generated_by_blocking"] += 1

            # Candidate truncation to max_k
            truncated_cands = retrieved_cands[:max_k]
            if not is_singleton:
                pruned_out = (set(retrieved_cands) - set(truncated_cands)) & true_m
                for mid in pruned_out:
                    failure_reasons["2_pruned_by_k_limit"] += 1

            # Model scoring
            pred_m = set()
            cand_scores = []
            for cid in truncated_cands:
                cand = cands_pool.get(cid)
                if not cand: continue
                sc, addr_j, rej_reason = score_pair_full(s1, cand)
                cand_scores.append((sc, cid, cand))

                if cid in true_m and rej_reason:
                    failure_reasons[f"3_{rej_reason}"] += 1

                if rej_reason or sc <= 0: continue
                if sc >= high_conf or (sc >= tau and addr_j >= addr_floor):
                    pred_m.add(cid)
                elif cid in true_m:
                    if sc < tau:
                        failure_reasons["4_score_below_tau"] += 1
                    else:
                        failure_reasons["5_addr_floor_rejected"] += 1

            # Metric evaluation
            score = entity_f05(pred_m, true_m)
            f05_scores.append(score)

            if is_singleton:
                if len(pred_m) == 0:
                    singleton_correct += 1
                else:
                    false_merges += 1
                    # Analyze singleton error root causes (Phase 9)
                    best_cand_sc, best_cid, best_cand = max(cand_scores, key=lambda x: x[0]) if cand_scores else (0, "", None)
                    singleton_error_records.append({
                        "s1_id": s1["id"],
                        "name": s1["clean_name"],
                        "addr": s1["clean_addr"],
                        "country": country,
                        "pred_cid": best_cid,
                        "cand_name": best_cand.clean_name if best_cand else "",
                        "cand_addr": best_cand.clean_addr if best_cand else "",
                        "score": round(best_cand_sc, 4),
                        "num_candidates": len(retrieved_cands),
                        "error_type": "same_name_different_addr" if best_cand and s1["clean_name"] == best_cand.clean_name else "approx_name_match"
                    })
            else:
                if pred_m != true_m:
                    fp = len(pred_m - true_m)
                    fn = len(true_m - pred_m)
                    false_merges += fp
                    missed_matches += fn

            # Precision & Recall
            if pred_m and true_m:
                tp = len(pred_m & true_m)
                precisions.append(tp / len(pred_m))
                recalls.append(tp / len(true_m))
            elif not pred_m and not true_m:
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(0.0)

            if (i + 1) % 50000 == 0:
                logger.info("[%s] %d/%d (%.0f ent/s) | Current F0.5: %.4f",
                            country, i + 1, len(recs), (i + 1) / (time.time() - t_country),
                            sum(f05_scores) / len(f05_scores))

    eval_time = time.time() - t_eval_start
    macro_f05 = sum(f05_scores) / len(f05_scores)
    macro_prec = sum(precisions) / len(precisions)
    macro_rec = sum(recalls) / len(recalls)
    sing_acc = singleton_correct / total_singletons if total_singletons else 1.0
    pair_blocking_recall = retrieved_true_matches / total_true_matches if total_true_matches else 1.0
    entity_perfect_recall = entity_perfect_retrieval / total_non_singletons if total_non_singletons else 1.0
    avg_candidates = total_candidates_generated / len(all_s1_records)

    logger.info("=" * 80)
    logger.info("200K DIAGNOSTIC BENCHMARK RESULTS (Completed in %.2fs):", eval_time)
    logger.info("  Macro F0.5:                   %.4f", macro_f05)
    logger.info("  Macro Precision:              %.4f", macro_prec)
    logger.info("  Macro Recall:                 %.4f", macro_rec)
    logger.info("  Singleton Accuracy:           %.2f%% (%d/%d)", sing_acc * 100, singleton_correct, total_singletons)
    logger.info("  Pair Blocking Recall:         %.2f%% (%d/%d)", pair_blocking_recall * 100, retrieved_true_matches, total_true_matches)
    logger.info("  Entity-Perfect Recall:        %.2f%% (%d/%d)", entity_perfect_recall * 100, entity_perfect_retrieval, total_non_singletons)
    logger.info("  Avg Candidates / Entity:      %.1f", avg_candidates)
    logger.info("  False Merges (False Pos):     %d", false_merges)
    logger.info("  Missed Matches (False Neg):   %d", missed_matches)
    logger.info("=" * 80)

    # Save Phase 0 & Phase 3 baseline reports
    baseline_report = {
        "benchmark": "AMAZON_ML_2026_DIAGNOSTIC_200K",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Phase 10b Champion (Address Abbreviation Expansion + Multi-Channel)",
        "tau": tau,
        "high_confidence": high_conf,
        "addr_floor": addr_floor,
        "max_k": max_k,
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "singleton_accuracy": round(sing_acc, 4),
        "singleton_accuracy_pct": round(sing_acc * 100, 2),
        "pair_blocking_recall_pct": round(pair_blocking_recall * 100, 2),
        "entity_perfect_recall_pct": round(entity_perfect_recall * 100, 2),
        "avg_candidates_per_entity": round(avg_candidates, 1),
        "false_merges": false_merges,
        "missed_matches": missed_matches,
        "total_entities_evaluated": len(all_s1_records),
        "eval_time_seconds": round(eval_time, 2),
        "throughput_entities_per_sec": round(len(all_s1_records) / eval_time, 1)
    }

    p0_path = REPORTS_DIAG / "champion_200k_baseline.json"
    p3_path = REPORTS_DIAG / "current_pipeline_200k.json"
    with open(p0_path, "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2)
    with open(p3_path, "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2)
    logger.info("Saved baseline reports -> %s and %s", p0_path.name, p3_path.name)

    # =========================================================================
    # PHASE 4: Error Stage Distribution (Failure Classification)
    # =========================================================================
    error_csv_path = REPORTS_DIAG / "error_stage_distribution.csv"
    with open(error_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_stage", "failure_count", "percentage_of_failures"])
        total_failures = sum(failure_reasons.values())
        for stage, cnt in failure_reasons.most_common():
            pct = (cnt / total_failures * 100) if total_failures else 0.0
            writer.writerow([stage, cnt, f"{pct:.2f}%"])
    logger.info("Saved failure distribution -> %s (%d total failures classified)", error_csv_path.name, total_failures)

    # =========================================================================
    # PHASE 9: Singleton Error Analysis
    # =========================================================================
    sing_err_path = REPORTS_DIAG / "singleton_errors_200k.csv"
    with open(sing_err_path, "w", encoding="utf-8", newline="") as f:
        if singleton_error_records:
            writer = csv.DictWriter(f, fieldnames=list(singleton_error_records[0].keys()))
            writer.writeheader()
            for r in singleton_error_records[:2000]: # top 2000 examples
                writer.writerow(r)
    logger.info("Saved singleton error analysis -> %s (%d errors cataloged)", sing_err_path.name, len(singleton_error_records))

    # =========================================================================
    # PHASE 5: Blocker Channel Ablation
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PHASE 5: Measuring Blocker Channels Independently on 200K Benchmark...")
    logger.info("=" * 80)
    
    # Measure each channel on sample of 20,000 non-singletons for fast, statistically sound ablation
    ablation_sample = [s for s in all_s1_records if len(gt[s["id"]]) > 0][:20000]
    total_sample_true_matches = sum(len(gt[s["id"]]) for s in ablation_sample)

    channel_names = [
        "exact_name", "char_3gram", "char_4gram", "token_overlap",
        "rare_token", "addr_token", "street_number", "postal_code", "domain_stem"
    ]

    ablation_results = []
    
    # Measure each individual channel
    for ch_name in channel_names:
        t_ch = time.time()
        retrieved_true = 0
        perfect_entities = 0
        total_candidates = 0

        for s1 in ablation_sample:
            true_m = gt[s1["id"]]
            c_idx = channels.get(ch_name, {})
            cands = set()

            if ch_name == "exact_name" and s1["alpha_name"]:
                p = c_idx.get(s1["alpha_name"])
                if p: cands.update(p)
            elif ch_name == "char_3gram":
                for ng in s1["ngrams3"]:
                    p = c_idx.get(ng)
                    if p: cands.update(p)
            elif ch_name == "char_4gram":
                for ng in s1["ngrams4"]:
                    p = c_idx.get(ng)
                    if p: cands.update(p)
            elif ch_name == "token_overlap":
                for t in s1["name_tokens"]:
                    p = c_idx.get(t)
                    if p: cands.update(p)
            elif ch_name == "rare_token":
                for t in s1["name_tokens"]:
                    p = c_idx.get(t)
                    if p: cands.update(p)
            elif ch_name == "addr_token":
                for at in s1["addr_tokens"]:
                    p = c_idx.get(at)
                    if p: cands.update(p)
            elif ch_name == "street_number" and s1["street_num"]:
                p = c_idx.get(s1["street_num"])
                if p: cands.update(p)
            elif ch_name == "postal_code" and s1["postal"]:
                p = c_idx.get(s1["postal"])
                if p: cands.update(p)
            elif ch_name == "domain_stem" and s1["domain"]:
                p = c_idx.get(s1["domain"])
                if p: cands.update(p)

            total_candidates += len(cands)
            tp = len(cands & true_m)
            retrieved_true += tp
            if tp == len(true_m):
                perfect_entities += 1

        dt_ch = time.time() - t_ch
        pair_rec = (retrieved_true / total_sample_true_matches * 100) if total_sample_true_matches else 0.0
        perf_rec = (perfect_entities / len(ablation_sample) * 100) if ablation_sample else 0.0
        avg_cands = total_candidates / len(ablation_sample)
        reduction_ratio = (1.0 - (total_candidates / (len(ablation_sample) * len(cands_pool)))) * 100

        ablation_results.append({
            "channel": ch_name,
            "pair_recall_pct": round(pair_rec, 2),
            "entity_perfect_recall_pct": round(perf_rec, 2),
            "avg_candidates": round(avg_cands, 1),
            "reduction_ratio_pct": round(reduction_ratio, 4),
            "runtime_sec": round(dt_ch, 2)
        })
        logger.info("  Channel %-15s: PairRecall=%5.2f%% | PerfectRecall=%5.2f%% | AvgCands=%4.1f | Time=%5.2fs",
                    ch_name, pair_rec, perf_rec, avg_cands, dt_ch)

    # Save Phase 5 Blocker Ablation Report
    ablation_csv_path = REPORTS_DIAG / "blocker_ablation_200k.csv"
    with open(ablation_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)
    logger.info("Saved blocker ablation -> %s", ablation_csv_path.name)

    logger.info("=" * 80)
    logger.info("DIAGNOSTIC SUITE RUN COMPLETE in %.2fs!", time.time() - t_start)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
