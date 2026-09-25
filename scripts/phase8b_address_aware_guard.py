#!/usr/bin/env python3
"""
scripts/phase8b_address_aware_guard.py
Phase 8b: Address-Aware Tiered Decision Rule.

Key finding from Phase 8:
  - tau=0.64 (unguarded): F0.5=0.7976, Singleton=88.69%
    (only ~15 more singleton FMs to fix to reach 90% floor)
  - Simple margin guards couldn't simultaneously satisfy both criteria

Root cause of remaining singleton FMs (from Phase 2 samples):
  - "office of aging" (Vermont) → "office of aging llc" (North Carolina) — different state
  - "shiv constructions" (Pune, Maharashtra) → "shiv constructions trading" (Chennai, Tamil Nadu)
  - Pattern: HIGH name similarity, but address tokens share NOTHING

Strategy: Tiered address-aware decision rule:
  Tier 1: score >= high_conf (0.75+) → accept (unambiguous match)
  Tier 2: score >= tau AND addr_jaccard >= addr_floor → accept (name+address agreement)
  Tier 3: reject (singleton protection — name matches but addresses diverge)

Grid search over (tau, high_conf, addr_floor).
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
logger = logging.getLogger("Phase8b")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)

def clean_text(s):
    if not s: return ""
    return ' '.join(RE_PUNCT.sub(' ', str(s).lower().replace('&', ' and ')).split())
def extract_tokens(text): return set(text.split())
def get_char_ngrams(text, n):
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}
def get_street_num(addr):
    m = re.match(r'^(\d+)\b', addr or ""); return m.group(1) if m else ""
def get_postal(addr):
    for d in RE_DIGITS.findall(addr or ""):
        if len(d) in (5, 6): return d
    return ""
def preprocess_record(r):
    name = r.get('business_name', ''); addr = r.get('business_address', '')
    cn = clean_text(name); ca = clean_text(addr)
    return {'id': r['entity_id'], 'clean_name': cn, 'clean_addr': ca,
            'name_tokens': extract_tokens(cn), 'addr_tokens': extract_tokens(ca),
            'street_num': get_street_num(ca), 'postal': get_postal(ca),
            'country': r.get('country', ''), 'ngrams4': get_char_ngrams(cn, 4)}

def jaro_winkler(s1, s2, max_len=40):
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]; l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(l1, l2) // 2 - 1; m1 = [False]*l1; m2 = [False]*l2; matches = 0
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
    p = sum(1 for i in range(min(4,l1,l2)) if s1[i]==s2[i])
    return sim + p * 0.1 * (1.0 - sim)

def token_jaccard(t1, t2):
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def ngram_jaccard(ng1, ng2):
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def entity_f05(pred_set, true_set):
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp/len(pred_set); r = tp/len(true_set)
    denom = 0.25 * p + r
    return 1.25 * (p * r) / denom if denom else 0.0

def score_pair_full(s1, cand):
    """Returns (composite_score, addr_jaccard) for tiered decision."""
    jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
    tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
    tj_addr = token_jaccard(s1["addr_tokens"], cand["addr_tokens"])
    sn1, sn2 = s1["street_num"], cand["street_num"]
    if sn1 and sn2: conflict = 1.0 if sn1 != sn2 else 0.0
    elif sn1 or sn2: conflict = 0.2
    else: conflict = 0.0
    ng4 = ngram_jaccard(s1["ngrams4"], cand["ngrams4"])
    base = 0.5 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.3 * conflict
    if ng4 >= 0.5 and jw >= 0.65:
        base = max(base, 0.5*jw + 0.15*tj_name + 0.15*tj_addr + 0.2*ng4 - 0.3*conflict)
    return base, tj_addr

def load_candidates(data_root, needed_pos, max_dist=150000):
    cands = {}
    for sf in [data_root/"train"/"train_source2.tsv", data_root/"train"/"train_source3.tsv"]:
        dc = 0
        with open(sf, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid, name, addr, country = p[0], p[1], p[2], p[3]
                    is_n = cid in needed_pos
                    if is_n or dc < max_dist:
                        cands[cid] = preprocess_record({'entity_id': cid, 'business_name': name,
                                                        'business_address': addr, 'country': country})
                        if not is_n: dc += 1
    return cands

def build_indexes(cands, token_freq):
    ni = defaultdict(list); ri = defaultdict(list)
    ai = defaultdict(list); si = defaultdict(list)
    pi = defaultdict(list); ng4i = defaultdict(list)
    for cid, c in cands.items():
        co = c["country"]
        for t in c["name_tokens"]:
            ni[(co,t)].append(cid)
            if token_freq.get(t, 0) < 50: ri[(co,t)].append(cid)
        for at in c["addr_tokens"]: ai[(co,at)].append(cid)
        if c["street_num"]: si[(co,c["street_num"])].append(cid)
        if c["postal"]: pi[(co,c["postal"])].append(cid)
        for ng in c["ngrams4"]: ng4i[(co,ng)].append(cid)
    return {"name": ni, "rare": ri, "addr": ai, "snum": si, "postal": pi, "ng4": ng4i}

def retrieve(s1, idx, max_k=80):
    c = s1["country"]; cands = set()
    for t in s1["name_tokens"]:
        p = idx["name"].get((c,t), [])
        if len(p) <= 60: cands.update(p)
    for t in s1["name_tokens"]:
        p = idx["rare"].get((c,t), [])
        if len(p) <= 30: cands.update(p)
    for at in s1["addr_tokens"]:
        p = idx["addr"].get((c,at), [])
        if len(p) <= 40: cands.update(p)
    if s1["street_num"]:
        p = idx["snum"].get((c,s1["street_num"]), [])
        if len(p) <= 40: cands.update(p)
    if s1["postal"]:
        p = idx["postal"].get((c,s1["postal"]), [])
        if len(p) <= 30: cands.update(p)
    for ng in s1["ngrams4"]:
        p = idx["ng4"].get((c,ng), [])
        if len(p) <= 30: cands.update(p)
    return list(cands)[:max_k]

def evaluate(predictions, val_gt, val_s1_ids):
    f05_list, p_list, r_list = [], [], []
    st = sc = fm = 0
    for eid in val_s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))
        if not true_set:
            st += 1
            if not pred_set: sc += 1
            else: fm += 1
        if not true_set and not pred_set: p_list.append(1.0); r_list.append(1.0)
        elif not true_set: p_list.append(0.0); r_list.append(1.0)
        elif not pred_set: p_list.append(1.0); r_list.append(0.0)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp/len(pred_set)); r_list.append(tp/len(true_set))
            if tp < len(pred_set): fm += 1
        f05_list.append(entity_f05(pred_set, true_set))
    return {"macro_f05": float(np.mean(f05_list)), "precision": float(np.mean(p_list)),
            "recall": float(np.mean(r_list)),
            "singleton_accuracy": (sc/st) if st > 0 else 1.0, "false_merges": fm}

def apply_tiered_rule(all_scored_full, tau, high_conf, addr_floor):
    """
    Tiered address-aware decision:
      Tier 1: score >= high_conf  → accept (strong evidence, bypass addr check)
      Tier 2: tau <= score < high_conf AND addr_jaccard >= addr_floor → accept
      Tier 3: reject (singleton protection)
    """
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
    logger.info("PHASE 8b: ADDRESS-AWARE TIERED DECISION RULE")
    logger.info("=" * 70)

    with open(FROZEN_SPLIT_PATH) as f: manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]; val_s1_set = set(val_s1_ids)

    gt = {}
    with open(DATA_ROOT/"train"/"train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in val_s1_set}

    needed_pos = set()
    for e in val_s1_set: needed_pos.update(val_gt[e])

    val_s1_records = []
    with open(DATA_ROOT/"train"/"train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in val_s1_set:
                val_s1_records.append(preprocess_record(r))

    logger.info("Loading candidates (100%% positives + 150k distractors)...")
    t0 = time.time()
    cands = load_candidates(DATA_ROOT, needed_pos)
    logger.info("Loaded %d candidates in %.2fs", len(cands), time.time() - t0)

    token_freq = defaultdict(int)
    for c in cands.values():
        for t in c["name_tokens"]: token_freq[t] += 1

    idx = build_indexes(cands, token_freq)

    logger.info("Scoring all candidates...")
    t0 = time.time()
    all_scored_full = {}
    for s1 in val_s1_records:
        s1_id = s1["id"]
        cand_list = retrieve(s1, idx, max_k=80)
        triples = []
        for cid in cand_list:
            cand = cands.get(cid)
            if cand:
                sc, addr_j = score_pair_full(s1, cand)
                triples.append((cid, sc, addr_j))
        all_scored_full[s1_id] = triples
    logger.info("Scoring done in %.2fs", time.time() - t0)

    with open(REPORTS_DIR/"champion.json") as f: champion = json.load(f)
    champ_f05 = champion["macro_f05"]

    # Grid search: tau x high_conf x addr_floor
    taus        = [0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    high_confs  = [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    addr_floors = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    total = len(taus) * len(high_confs) * len(addr_floors)
    logger.info("Grid search: %d x %d x %d = %d combos", len(taus), len(high_confs), len(addr_floors), total)

    passing = []
    all_results = []
    for tau in taus:
        for hc in high_confs:
            for af in addr_floors:
                preds = apply_tiered_rule(all_scored_full, tau, hc, af)
                m = evaluate(preds, val_gt, val_s1_ids)
                entry = (m["macro_f05"], tau, hc, af, m["singleton_accuracy"],
                         m["precision"], m["recall"], m["false_merges"])
                all_results.append(entry)
                if m["macro_f05"] > champ_f05 + 0.002 and m["singleton_accuracy"] >= 0.90:
                    passing.append(entry)

    all_results.sort(reverse=True)
    passing.sort(reverse=True)

    logger.info("=" * 70)
    logger.info("CHAMPION BASELINE: F0.5=%.4f | Singleton=94.17%%", champ_f05)
    logger.info("QUALITY GATE:      F0.5 > %.4f AND Singleton >= 90%%", champ_f05 + 0.002)
    logger.info("-" * 70)

    if passing:
        logger.info("✅ FOUND %d CONFIGS PASSING QUALITY GATE!", len(passing))
        logger.info("%-6s %-6s %-6s %-12s %-12s %-12s %-10s",
                    "Tau", "HiConf", "AddrF", "Macro F0.5", "Precision", "Singleton", "FalseM")
        logger.info("-" * 70)
        for f05, tau, hc, af, sing, prec, rec, fm in passing[:15]:
            logger.info("%-6.2f %-6.2f %-6.2f %-12.4f %-12.4f %-12.2f%% %-10d",
                        tau, hc, af, f05, prec, sing*100, fm)

        best_f05, best_tau, best_hc, best_af, best_sing, best_prec, best_rec, best_fm = passing[0]
        logger.info("=" * 70)
        logger.info("🏆 BEST PASSING CONFIG: tau=%.2f | high_conf=%.2f | addr_floor=%.2f", best_tau, best_hc, best_af)
        logger.info("   Macro F0.5:         %.4f  (delta: %+.4f vs champion)", best_f05, best_f05 - champ_f05)
        logger.info("   Precision:          %.4f (%.2f%%)", best_prec, best_prec*100)
        logger.info("   Recall:             %.4f (%.2f%%)", best_rec, best_rec*100)
        logger.info("   Singleton Accuracy: %.4f (%.2f%%)", best_sing, best_sing*100)
        logger.info("   False Merges:       %d", best_fm)

        result = {
            "run_id": "phase8b-addr-aware-guard-v1",
            "model": "Champion Heuristic + Phase3 Multi-Channel Retrieval + Phase8b Address-Aware Guard",
            "tau": best_tau, "high_confidence": best_hc, "addr_floor": best_af,
            "macro_f05": round(best_f05, 4), "precision": round(best_prec, 4),
            "recall": round(best_rec, 4), "singleton_accuracy": round(best_sing, 4),
            "false_merges": best_fm, "champion_f05": champ_f05,
            "delta_vs_champion": round(best_f05 - champ_f05, 4),
            "quality_gate_criteria_met": True,
            "dataset_hash": manifest["dataset_hash"],
            "validation_protocol": manifest["split_version"]
        }
        out = REPORTS_DIR / "phase8b_best_challenger.json"
        with open(out, "w") as f: json.dump(result, f, indent=2)
        logger.info("Saved to %s", out)
    else:
        logger.warning("❌ No config satisfied BOTH criteria.")
        logger.info("Top 15 configs by F0.5 (showing all):")
        logger.info("%-6s %-6s %-6s %-12s %-12s", "Tau", "HiConf", "AddrF", "Macro F0.5", "Singleton")
        for f05, tau, hc, af, sing, prec, rec, fm in all_results[:15]:
            logger.info("  %-6.2f %-6.2f %-6.2f %-12.4f %-12.2f%%", tau, hc, af, f05, sing*100)
        # Save top result regardless for analysis
        f05, tau, hc, af, sing, prec, rec, fm = all_results[0]
        result = {
            "run_id": "phase8b-addr-aware-guard-v1",
            "model": "Champion Heuristic + Phase3 Multi-Channel Retrieval + Phase8b Address-Aware Guard",
            "tau": tau, "high_confidence": hc, "addr_floor": af,
            "macro_f05": round(f05, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "singleton_accuracy": round(sing, 4),
            "false_merges": fm, "champion_f05": champ_f05,
            "delta_vs_champion": round(f05 - champ_f05, 4),
            "quality_gate_criteria_met": False,
            "dataset_hash": manifest["dataset_hash"],
            "validation_protocol": manifest["split_version"]
        }
        out = REPORTS_DIR / "phase8b_best_challenger.json"
        with open(out, "w") as f: json.dump(result, f, indent=2)
        logger.info("Saved diagnostic result to %s", out)

if __name__ == "__main__":
    main()
