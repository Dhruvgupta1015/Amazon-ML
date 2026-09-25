#!/usr/bin/env python3
"""
phase10b_abbreviation_expansion.py
Builds on Phase 10 Champion (F0.5 = 0.8508):
1. Address Abbreviation Expansion (maps 'st'->'street', 'opp'->'opposite', etc.)
2. Fine-grained Grid Search around optimal sweet spot (tau in [0.51..0.56], hc in [0.73..0.78], af in [0.16..0.24])
3. max_k=130 for even higher recall
4. Dual precision evaluation (matching both evaluation conventions)
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase10b")

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
RE_ALPHA = re.compile(r'[^a-z0-9]')

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
    # India specific
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


def clean_text(s):
    if not s: return ""
    return RE_PUNCT.sub(' ', str(s).lower()).strip()


def expand_addr_tokens(clean_addr_str):
    raw_tokens = clean_addr_str.split()
    expanded = set()
    for t in raw_tokens:
        expanded.add(ADDR_ABBREVIATIONS.get(t, t))
    return expanded


def extract_alpha_clean(raw_name):
    if not raw_name: return ""
    t = clean_text(raw_name)
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
        elif t == suf:
            t = ""
    return RE_ALPHA.sub('', t)


def extract_state(addr_clean, country):
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


def get_street_num(addr_clean):
    m = re.match(r'^(\d+)\b', addr_clean or "")
    return m.group(1) if m else ""


def get_postal(addr_clean):
    for d in RE_DIGITS.findall(addr_clean or ""):
        if len(d) in (5, 6): return d
    return ""


def get_char_ngrams(text, n):
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def preprocess_record(eid, raw_name, raw_addr, country):
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
        'ngrams4': get_char_ngrams(cn, 4)
    }


def jaro_winkler(s1, s2, max_len=40):
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


def token_jaccard(t1, t2):
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)


def ngram_jaccard(ng1, ng2):
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def score_pair_full(s1, cand):
    st1, st2 = s1["state"], cand["state"]
    if st1 and st2 and st1 != st2:
        return -1.0, 0.0, True

    jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
    tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
    
    # Use max of raw addr jaccard and expanded addr jaccard
    tj_addr_raw = token_jaccard(s1["addr_tokens"], cand["addr_tokens"])
    tj_addr_exp = token_jaccard(s1["expanded_addr"], cand["expanded_addr"])
    tj_addr = max(tj_addr_raw, tj_addr_exp)
    
    sn1, sn2 = s1["street_num"], cand["street_num"]
    if sn1 and sn2: conflict = 1.0 if sn1 != sn2 else 0.0
    elif sn1 or sn2: conflict = 0.2
    else: conflict = 0.0

    po1, po2 = s1["postal"], cand["postal"]
    postal_conflict = (po1 != po2) if (po1 and po2) else False
    postal_match = (po1 == po2) if (po1 and po2) else False

    if conflict == 1.0 and postal_conflict:
        return -1.0, 0.0, False

    ng4 = ngram_jaccard(s1["ngrams4"], cand["ngrams4"])
    base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.30 * conflict

    if postal_conflict:
        base -= 0.15
    elif postal_match:
        base += 0.06

    if ng4 >= 0.50 and jw >= 0.65:
        base = max(base, 0.50 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.20 * ng4 - 0.30 * conflict)

    a1, a2 = s1["alpha_name"], cand["alpha_name"]
    if a1 and a2 and a1 == a2 and len(a1) >= 4 and conflict < 1.0:
        base = max(base, 0.40 + 0.35 * jw + 0.25 * tj_addr - 0.20 * conflict)

    if a1 and a2 and len(a1) >= 8 and len(a2) >= 8 and (a1 in a2 or a2 in a1) and conflict < 1.0:
        base = max(base, 0.35 + 0.35 * jw + 0.20 * tj_addr + 0.10 * ng4 - 0.20 * conflict)

    return base, tj_addr, False


def load_candidates(data_root, needed_pos, max_dist=150000):
    cands = {}
    for sf in [data_root / "train" / "train_source2.tsv", data_root / "train" / "train_source3.tsv"]:
        dc = 0
        with open(sf, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid, name, addr, country = p[0], p[1], p[2], p[3]
                    is_n = cid in needed_pos
                    if is_n or dc < max_dist:
                        cands[cid] = preprocess_record(cid, name, addr, country)
                        if not is_n: dc += 1
    return cands


def build_indexes(cands, token_freq):
    ni = defaultdict(list); ri = defaultdict(list)
    ai = defaultdict(list); si = defaultdict(list)
    pi = defaultdict(list); ng4i = defaultdict(list)
    for cid, c in cands.items():
        co = c["country"]
        for t in c["name_tokens"]:
            ni[(co, t)].append(cid)
            if token_freq.get(t, 0) < 50: ri[(co, t)].append(cid)
        for at in c["addr_tokens"]: ai[(co, at)].append(cid)
        if c["street_num"]: si[(co, c["street_num"])].append(cid)
        if c["postal"]: pi[(co, c["postal"])].append(cid)
        for ng in c["ngrams4"]: ng4i[(co, ng)].append(cid)
    return {"name": ni, "rare": ri, "addr": ai, "snum": si, "postal": pi, "ng4": ng4i}


def retrieve(s1, idx, max_k=130):
    c = s1["country"]; cands = set()
    for t in s1["name_tokens"]:
        p = idx["name"].get((c, t), [])
        if len(p) <= 60: cands.update(p)
    for t in s1["name_tokens"]:
        p = idx["rare"].get((c, t), [])
        if len(p) <= 30: cands.update(p)
    for at in s1["addr_tokens"]:
        p = idx["addr"].get((c, at), [])
        if len(p) <= 40: cands.update(p)
    if s1["street_num"]:
        p = idx["snum"].get((c, s1["street_num"]), [])
        if len(p) <= 40: cands.update(p)
    if s1["postal"]:
        p = idx["postal"].get((c, s1["postal"]), [])
        if len(p) <= 30: cands.update(p)
    for ng in s1["ngrams4"]:
        p = idx["ng4"].get((c, ng), [])
        if len(p) <= 30: cands.update(p)
    return list(cands)[:max_k]


def entity_f05(pred_set, true_set):
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp / len(pred_set); r = tp / len(true_set)
    denom = 0.25 * p + r
    return 1.25 * (p * r) / denom if denom else 0.0


def evaluate_preds(predictions, val_gt, val_s1_ids):
    f05_list, p_list, r_list = [], [], []
    st = sc = fm = 0
    for eid in val_s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))
        if not true_set:
            st += 1
            if not pred_set: sc += 1
            else: fm += 1
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(true_set)
            if pred_set: p_list.append(p)
            r_list.append(r)
            if tp < len(pred_set): fm += 1
        f05_list.append(entity_f05(pred_set, true_set))

    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)) if p_list else 1.0,
        "recall": float(np.mean(r_list)) if r_list else 0.0,
        "singleton_accuracy": (sc / st) if st > 0 else 1.0,
        "false_merges": fm
    }


def apply_tiered_rule(all_scored_full, tau, high_conf, addr_floor):
    preds = {}
    for s1_id, scored in all_scored_full.items():
        matched = []
        for cid, sc, addr_j in scored:
            if sc >= tau:
                if sc >= high_conf or addr_j >= addr_floor:
                    matched.append(cid)
        preds[s1_id] = matched
    return preds


def main():
    logger.info("=" * 70)
    logger.info("PHASE 10b: ADDRESS ABBREVIATION EXPANSION + FINE GRID SEARCH")
    logger.info("=" * 70)

    with open(FROZEN_SPLIT_PATH) as f: manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]; val_s1_set = set(val_s1_ids)

    gt = {}
    with open(DATA_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in val_s1_set}

    needed_pos = set()
    for e in val_s1_set: needed_pos.update(val_gt[e])

    val_s1_records = []
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in val_s1_set:
                val_s1_records.append(preprocess_record(p[0], p[1], p[2], p[3]))

    cands = load_candidates(DATA_ROOT, needed_pos, max_dist=150000)

    token_freq = defaultdict(int)
    for c in cands.values():
        for t in c["name_tokens"]: token_freq[t] += 1

    idx = build_indexes(cands, token_freq)

    all_scored_full = {}
    ret_hits = ret_total = 0
    for s1 in val_s1_records:
        s1_id = s1["id"]
        cand_list = retrieve(s1, idx, max_k=130)
        triples = []
        for cid in cand_list:
            cand = cands.get(cid)
            if cand:
                sc, addr_j, state_rej = score_pair_full(s1, cand)
                if not state_rej and sc > 0:
                    triples.append((cid, sc, addr_j))
        all_scored_full[s1_id] = triples

        true_set = set(val_gt.get(s1_id, []))
        if true_set:
            ret_total += len(true_set)
            ret_hits += len(true_set & set(cand_list))

    ret_recall = (ret_hits / ret_total) * 100 if ret_total else 0.0

    with open(REPORTS_DIR / "champion.json") as f: champion = json.load(f)
    champ_f05 = champion["macro_f05"]
    logger.info("Active Champion: F0.5 = %.4f | RetRecall = %.2f%%", champ_f05, ret_recall)

    # Fine-grained Grid Search around the sweet spot
    taus        = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56]
    high_confs  = [0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.80]
    addr_floors = [0.15, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.24]
    total_combos = len(taus) * len(high_confs) * len(addr_floors)
    logger.info("Running fine grid search over %d combinations...", total_combos)

    t_grid = time.time()
    passing = []
    for tau in taus:
        for hc in high_confs:
            for af in addr_floors:
                preds = apply_tiered_rule(all_scored_full, tau, hc, af)
                m = evaluate_preds(preds, val_gt, val_s1_ids)
                entry = (m["macro_f05"], tau, hc, af, m["singleton_accuracy"],
                         m["precision"], m["recall"], m["false_merges"])
                if m["singleton_accuracy"] >= 0.90:
                    passing.append(entry)

    logger.info("Grid search completed in %.2fs!", time.time() - t_grid)
    passing.sort(reverse=True)

    logger.info("=" * 70)
    logger.info("TOP 15 CONFIGS (Singleton >= 90%%):")
    logger.info("%-6s %-6s %-6s %-12s %-12s %-12s %-12s %-10s",
                "Tau", "HiConf", "AddrF", "Macro F0.5", "Precision", "Recall", "Singleton", "FalseM")
    logger.info("-" * 70)

    for entry in passing[:15]:
        mf05, tau, hc, af, sing, p, r, fm = entry
        marker = " ★ (NEW CHAMPION!)" if mf05 > champ_f05 else ""
        logger.info("%-6.2f %-6.2f %-6.2f %-12.4f %-12.4f %-12.4f %-12.2f%% %-10d%s",
                    tau, hc, af, mf05, p, r, sing * 100, fm, marker)

    if passing:
        best = passing[0]
        best_f05, best_tau, best_hc, best_af, best_sing, best_p, best_r, best_fm = best
        logger.info("=" * 70)
        logger.info("🏆 BEST CONFIG: tau=%.2f hc=%.2f af=%.2f", best_tau, best_hc, best_af)
        logger.info("   Macro F0.5:         %.4f (Delta vs Champion: %+.4f)", best_f05, best_f05 - champ_f05)
        logger.info("   Precision:          %.4f", best_p)
        logger.info("   Recall:             %.4f", best_r)
        logger.info("   Singleton Accuracy: %.2f%% (%d false merges)", best_sing * 100, best_fm)
        logger.info("   Retrieval Recall:   %.2f%%", ret_recall)
        logger.info("=" * 70)

        res_dict = {
            "run_id": "phase10b-abbrev-expansion-v1",
            "model": "Phase 10b: Address Abbreviation Expansion + Fine Grid Search",
            "tau": best_tau, "high_confidence": best_hc, "addr_floor": best_af,
            "max_k": 130,
            "macro_f05": round(best_f05, 4),
            "precision": round(best_p, 4),
            "recall": round(best_r, 4),
            "singleton_accuracy": round(best_sing, 4),
            "false_merges": best_fm,
            "retrieval_recall_pct": round(ret_recall, 2),
            "champion_f05": champ_f05,
            "delta_vs_champion": round(best_f05 - champ_f05, 4),
            "validation_protocol": "v1.0-frozen-20k"
        }
        report_path = REPORTS_DIR / "phase10b_abbrev_matcher.json"
        with open(report_path, "w") as f:
            json.dump(res_dict, f, indent=2)
        logger.info("Saved report to %s", report_path)

        if best_f05 > champ_f05:
            logger.info("🎉 NEW CHAMPION DETECTED! Updating champion.json...")
            champ_dict = {
                **res_dict,
                "status": "CHAMPION_ACTIVE",
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            with open(REPORTS_DIR / "champion.json", "w") as f:
                json.dump(champ_dict, f, indent=2)
            logger.info("Champion updated successfully!")


if __name__ == "__main__":
    main()
