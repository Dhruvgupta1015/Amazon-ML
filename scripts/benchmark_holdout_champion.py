#!/usr/bin/env python3
"""
benchmark_holdout_champion.py — Runs the Phase 10 Champion Heuristic on the
strict 30,000 Source-1 Final Holdout.
Calculates official challenge Macro F0.5, Precision, Recall, Singleton Accuracy,
False Merges, and Missed Matches.
Saves to reports/recovery/holdout_champion.json.
"""
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostic_200k"
RECOVERY_DIR = ROOT / "reports" / "recovery"
RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HoldoutChampion")

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
RE_ALPHA = re.compile(r'[^a-z0-9]')

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
    'dist': 'district', 'vill': 'village', 'po': 'post office'
}

class CompactCand:
    __slots__ = ('eid', 'clean_name', 'clean_addr', 'alpha_name', 'state', 'street_num', 'postal', 'name_tokens')
    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str,
                 state: str, street_num: str, postal: str, name_tokens: tuple):
        self.eid = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.state = state
        self.street_num = street_num
        self.postal = postal
        self.name_tokens = name_tokens

def clean_text(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(' ', str(s).lower()).strip()

def extract_alpha_clean(s: str) -> str:
    if not s: return ""
    return RE_ALPHA.sub('', str(s).lower()).strip()

def expand_addr_tokens(addr_clean: str) -> set:
    if not addr_clean: return set()
    tokens = addr_clean.split()
    return {ADDR_ABBREVIATIONS.get(t, t) for t in tokens}

def extract_state(addr_clean: str, country: str) -> str:
    if not addr_clean: return ""
    ambiguous_us = {'in', 'oh', 'me', 'wa', 'pa', 'or', 'ok', 'id', 'la', 'de', 'hi'}
    ambiguous_in = {'as', 'or', 'in', 'is', 'to', 'at', 'an'}
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

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(' ', '')
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

def token_jaccard(t1: set, t2: set) -> float:
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def ngram_jaccard(ng1: set, ng2: set) -> float:
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def score_pair_full(s1: dict, cand: CompactCand) -> tuple:
    st1, st2 = s1["state"], cand.state
    if st1 and st2 and st1 != st2:
        return -1.0, 0.0, True

    sn1, sn2 = s1["street_num"], cand.street_num
    po1, po2 = s1["postal"], cand.postal
    if (sn1 and sn2 and sn1 != sn2) and (po1 and po2 and po1 != po2):
        return -1.0, 0.0, False

    conflict = 1.0 if (sn1 and sn2 and sn1 != sn2) else (0.2 if (sn1 or sn2) else 0.0)
    postal_conflict = (po1 != po2) if (po1 and po2) else False
    postal_match = (po1 == po2) if (po1 and po2) else False

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

    return base, tj_addr, False

def entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)

def evaluate_predictions(predictions: Dict[str, Set[str]], gt: Dict[str, Set[str]], s1_ids: List[str]) -> Dict[str, float]:
    f05_list, p_list, r_list = [], [], []
    singletons_correct, singletons_total = 0, 0
    false_merges, missed_matches = 0, 0
    for eid in s1_ids:
        pred_set = predictions.get(eid, set())
        true_set = gt.get(eid, set())
        is_singleton = len(true_set) == 0
        if is_singleton:
            singletons_total += 1
            if len(pred_set) == 0: singletons_correct += 1
            else: false_merges += len(pred_set)

        if not true_set and not pred_set:
            p_list.append(1.0); r_list.append(1.0)
        elif not true_set:
            p_list.append(0.0); r_list.append(1.0)
        elif not pred_set:
            p_list.append(1.0); r_list.append(0.0)
            missed_matches += len(true_set)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp / len(pred_set))
            r_list.append(tp / len(true_set))
            false_merges += len(pred_set - true_set)
            missed_matches += len(true_set - pred_set)
        f05_list.append(entity_f05(pred_set, true_set))

    return {
        "macro_f05": float(sum(f05_list) / len(f05_list)),
        "precision": float(sum(p_list) / len(p_list)),
        "recall": float(sum(r_list) / len(r_list)),
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        "singletons_correct": singletons_correct,
        "singletons_total": singletons_total,
        "false_merges": false_merges,
        "missed_matches": missed_matches
    }

def main():
    logger.info("Loading 30K Holdout S1 IDs...")
    with open(DIAG_DIR / "split_holdout_30k_s1_ids.json", "r", encoding="utf-8") as f:
        holdout_ids = json.load(f)
    holdout_set = set(holdout_ids)

    logger.info("Loading Ground Truth...")
    gt = defaultdict(set)
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                gt[parts[0]] = set(parts[1].split(","))

    logger.info("Loading Holdout S1 records...")
    s1_dict = {}
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in holdout_set:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                cn = clean_text(name)
                ca = clean_text(addr)
                s1_dict[eid] = {
                    'id': eid,
                    'clean_name': cn,
                    'clean_addr': ca,
                    'alpha_name': extract_alpha_clean(name),
                    'state': extract_state(ca, country),
                    'name_tokens': set(cn.split()),
                    'addr_tokens': set(ca.split()),
                    'expanded_addr': expand_addr_tokens(ca),
                    'street_num': get_street_num(ca),
                    'postal': get_postal(ca),
                    'country': country,
                    'ngrams4': get_char_ngrams(cn, 4)
                }

    logger.info("Loading Candidate pool (S2 & S3)...")
    cands_by_country = defaultdict(dict)
    for src in ["source2.tsv", "source3.tsv"]:
        with open(DIAG_DIR / src, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    cn = clean_text(name)
                    ca = clean_text(addr)
                    cands_by_country[country][cid] = CompactCand(
                        eid=cid, clean_name=cn, clean_addr=ca,
                        alpha_name=extract_alpha_clean(name),
                        state=extract_state(ca, country),
                        street_num=get_street_num(ca),
                        postal=get_postal(ca),
                        name_tokens=tuple(cn.split())
                    )

    # Build inverted index as in run_phase10_inference
    logger.info("Building inverted indexes with posting-list caps (as in Phase 10 Champion)...")
    tau, high_conf, addr_floor, max_k = 0.53, 0.75, 0.22, 130
    predictions = defaultdict(set)
    total_cands_retrieved = 0
    zero_cands_count = 0

    for country in ["US", "India"]:
        pool = cands_by_country[country]
        token_freq = defaultdict(int)
        for c in pool.values():
            for t in c.name_tokens: token_freq[t] += 1

        ni, ri, ai, si, pi, ng4i = {}, {}, {}, {}, {}, {}
        for cid, c in pool.items():
            for t in c.name_tokens:
                p = ni.get(t)
                if p is None: ni[t] = [cid]
                elif len(p) <= 60: p.append(cid)
                if token_freq.get(t, 0) < 50:
                    p_r = ri.get(t)
                    if p_r is None: ri[t] = [cid]
                    elif len(p_r) <= 30: p_r.append(cid)
            for at in c.clean_addr.split():
                p = ai.get(at)
                if p is None: ai[at] = [cid]
                elif len(p) <= 40: p.append(cid)
            if c.street_num:
                p = si.get(c.street_num)
                if p is None: si[c.street_num] = [cid]
                elif len(p) <= 40: p.append(cid)
            if c.postal:
                p = pi.get(c.postal)
                if p is None: pi[c.postal] = [cid]
                elif len(p) <= 30: p.append(cid)
            for ng in get_char_ngrams(c.clean_name, 4):
                p = ng4i.get(ng)
                if p is None: ng4i[ng] = [cid]
                elif len(p) <= 30: p.append(cid)

        idx = {
            "name": {k: v for k, v in ni.items() if len(v) <= 60},
            "rare": {k: v for k, v in ri.items() if len(v) <= 30},
            "addr": {k: v for k, v in ai.items() if len(v) <= 40},
            "snum": {k: v for k, v in si.items() if len(v) <= 40},
            "postal": {k: v for k, v in pi.items() if len(v) <= 30},
            "ng4": {k: v for k, v in ng4i.items() if len(v) <= 30}
        }

        # Resolve holdout S1 entities of this country
        country_s1 = [s1 for s1 in s1_dict.values() if s1["country"] == country]
        logger.info("[%s] Scoring %d holdout entities...", country, len(country_s1))

        for s1 in country_s1:
            cands = set()
            for t in s1["name_tokens"]:
                p = idx["name"].get(t)
                if p: cands.update(p)
            for t in s1["name_tokens"]:
                p = idx["rare"].get(t)
                if p: cands.update(p)
            for at in s1["addr_tokens"]:
                p = idx["addr"].get(at)
                if p: cands.update(p)
            if s1["street_num"]:
                p = idx["snum"].get(s1["street_num"])
                if p: cands.update(p)
            if s1["postal"]:
                p = idx["postal"].get(s1["postal"])
                if p: cands.update(p)
            for ng in s1["ngrams4"]:
                p = idx["ng4"].get(ng)
                if p: cands.update(p)

            cand_list = list(cands)[:max_k]
            total_cands_retrieved += len(cand_list)
            if len(cand_list) == 0:
                zero_cands_count += 1

            matched = set()
            for cid in cand_list:
                cand = pool.get(cid)
                if not cand: continue
                sc, addr_j, state_rej = score_pair_full(s1, cand)
                if state_rej or sc <= 0: continue
                if sc >= high_conf or (sc >= tau and addr_j >= addr_floor):
                    matched.add(cid)
            predictions[s1["id"]] = matched

    # Evaluate official metrics on holdout
    logger.info("Evaluating 30K Holdout...")
    metrics = evaluate_predictions(predictions, gt, holdout_ids)
    metrics["zero_candidates_count"] = zero_cands_count
    metrics["zero_candidates_rate"] = zero_cands_count / len(holdout_ids)
    metrics["avg_candidates_per_entity"] = total_cands_retrieved / len(holdout_ids)

    logger.info("=" * 60)
    logger.info("HOLDOUT CHAMPION BASELINE METRICS:")
    logger.info("  Macro F0.5:           %.6f", metrics["macro_f05"])
    logger.info("  Precision:            %.6f", metrics["precision"])
    logger.info("  Recall:               %.6f", metrics["recall"])
    logger.info("  Singleton Accuracy:   %.2f%%", metrics["singleton_accuracy"] * 100)
    logger.info("  Zero Candidates:      %d (%.2f%%)", zero_cands_count, metrics["zero_candidates_rate"] * 100)
    logger.info("  False Merges:         %d", metrics["false_merges"])
    logger.info("  Missed Matches:       %d", metrics["missed_matches"])
    logger.info("=" * 60)

    with open(RECOVERY_DIR / "holdout_champion.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved to %s", RECOVERY_DIR / "holdout_champion.json")

if __name__ == "__main__":
    main()
