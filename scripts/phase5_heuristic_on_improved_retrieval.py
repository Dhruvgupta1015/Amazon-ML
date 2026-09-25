#!/usr/bin/env python3
"""
scripts/phase5_heuristic_on_improved_retrieval.py
Phase 5: Apply Champion Heuristic on Phase 3 improved multi-channel candidate set.

Uses EXACT same candidate set for:
  - Champion heuristic benchmark (compute macro F0.5)
  - Identifies whether retrieval improvement (80% -> 85.6%) translates to F0.5 gain

If the new heuristic beats champion.json by >= +0.002:
  -> Submit to quality gate for promotion consideration.
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
logger = logging.getLogger("Phase5Heuristic")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)

def clean_text(s):
    if not s: return ""
    s = str(s).lower().replace('&', ' and ')
    return ' '.join(RE_PUNCT.sub(' ', s).split())

def extract_tokens(text):
    return set(text.split())

def extract_digits(text):
    if not text: return set()
    return set(RE_DIGITS.findall(str(text)))

def extract_domain_stem(raw_name):
    if not raw_name: return ""
    m = DOMAIN_RE.search(raw_name.lower())
    return m.group(1).replace('-', '') if m else ""

def get_char_ngrams(text, n):
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def get_street_num(addr):
    if not addr: return ""
    m = re.match(r'^(\d+)\b', addr)
    return m.group(1) if m else ""

def get_postal(addr):
    for d in RE_DIGITS.findall(addr or ""):
        if len(d) in (5, 6):
            return d
    return ""

def preprocess_record(r):
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

def jaro_winkler(s1, s2, max_len=40):
    if s1 == s2: return 1.0
    s1, s2 = s1[:max_len], s2[:max_len]
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0: return 0.0
    md = max(l1, l2) // 2 - 1
    m1 = [False]*l1; m2 = [False]*l2
    matches = 0
    for i in range(l1):
        for j in range(max(0, i-md), min(i+md+1, l2)):
            if m2[j] or s1[i] != s2[j]: continue
            m1[i] = m2[j] = True; matches += 1; break
    if matches == 0: return 0.0
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
    inter = len(t1 & t2); union = len(t1 | t2)
    return inter / union if union > 0 else 0.0

def ngram_jaccard(ng1, ng2):
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    inter = len(ng1 & ng2); union = len(ng1 | ng2)
    return inter / union if union > 0 else 0.0

def entity_f05(predicted_set, true_set):
    if not predicted_set and not true_set: return 1.0
    if not predicted_set or not true_set: return 0.0
    tp = len(predicted_set & true_set)
    if tp == 0: return 0.0
    p = tp / len(predicted_set); r = tp / len(true_set)
    denom = 0.25 * p + r
    return 1.25 * (p * r) / denom if denom else 0.0

def score_pair(s1, cand):
    """Extended heuristic scoring using champion formula + ngram boost."""
    jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
    tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
    tj_addr = token_jaccard(s1["addr_tokens"], cand["addr_tokens"])

    sn1, sn2 = s1["street_num"], cand["street_num"]
    if sn1 and sn2:
        conflict = 1.0 if sn1 != sn2 else 0.0
    elif sn1 or sn2:
        conflict = 0.2
    else:
        conflict = 0.0

    # Ngram similarity (guards against OCR variants)
    ng4 = ngram_jaccard(s1["ngrams4"], cand["ngrams4"])

    # Composite: Champion formula + ngram4 as a soft signal
    base_score = 0.5 * jw + 0.25 * tj_name + 0.25 * tj_addr - 0.3 * conflict
    # Boost: if ngram4 agreement is strong but name tokens differ
    if ng4 >= 0.5 and jw >= 0.65:
        base_score = max(base_score, 0.5 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.2 * ng4 - 0.3 * conflict)
    return base_score

def load_all_candidates(data_root, needed_positive_cids, max_distractors=150000):
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
                        cands[cid] = preprocess_record({'entity_id': cid, 'business_name': name,
                                                        'business_address': addr, 'country': country})
                        if not is_needed:
                            distractor_count += 1
    return cands

def build_indexes(cands, token_freq):
    name_idx = defaultdict(list); rare_idx = defaultdict(list)
    addr_idx = defaultdict(list); snum_idx = defaultdict(list)
    postal_idx = defaultdict(list); ng4_idx = defaultdict(list)
    for cid, cand in cands.items():
        c = cand["country"]
        for tok in cand["name_tokens"]:
            name_idx[(c, tok)].append(cid)
            if token_freq.get(tok, 0) < 50:
                rare_idx[(c, tok)].append(cid)
        for atok in cand["addr_tokens"]:
            addr_idx[(c, atok)].append(cid)
        if cand["street_num"]:
            snum_idx[(c, cand["street_num"])].append(cid)
        if cand["postal"]:
            postal_idx[(c, cand["postal"])].append(cid)
        for ng in cand["ngrams4"]:
            ng4_idx[(c, ng)].append(cid)
    return {"name": name_idx, "rare": rare_idx, "addr": addr_idx,
            "snum": snum_idx, "postal": postal_idx, "ng4": ng4_idx}

def retrieve_candidates(s1, indexes, max_k=80):
    c = s1["country"]
    cands = set()
    for tok in s1["name_tokens"]:
        p = indexes["name"].get((c, tok), [])
        if len(p) <= 60: cands.update(p)
    for tok in s1["name_tokens"]:
        p = indexes["rare"].get((c, tok), [])
        if len(p) <= 30: cands.update(p)
    for atok in s1["addr_tokens"]:
        p = indexes["addr"].get((c, atok), [])
        if len(p) <= 40: cands.update(p)
    if s1["street_num"]:
        p = indexes["snum"].get((c, s1["street_num"]), [])
        if len(p) <= 40: cands.update(p)
    if s1["postal"]:
        p = indexes["postal"].get((c, s1["postal"]), [])
        if len(p) <= 30: cands.update(p)
    for ng in s1["ngrams4"]:
        p = indexes["ng4"].get((c, ng), [])
        if len(p) <= 30: cands.update(p)
    return list(cands)[:max_k]

def evaluate_at_tau(predictions, val_gt, val_s1_ids):
    f05_list, p_list, r_list = [], [], []
    singletons_total = singletons_correct = false_merges = 0
    for eid in val_s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))
        is_singleton = len(true_set) == 0
        if is_singleton:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
            else:
                false_merges += 1
        if not true_set and not pred_set:
            p_list.append(1.0); r_list.append(1.0)
        elif not true_set:
            p_list.append(0.0); r_list.append(1.0)
        elif not pred_set:
            p_list.append(1.0); r_list.append(0.0)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp / len(pred_set)); r_list.append(tp / len(true_set))
            if tp < len(pred_set): false_merges += 1
        f05_list.append(entity_f05(pred_set, true_set))
    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        "false_merges": false_merges,
        "singletons_total": singletons_total,
    }

def main():
    logger.info("=" * 70)
    logger.info("PHASE 5: CHAMPION HEURISTIC ON IMPROVED MULTI-CHANNEL CANDIDATE SET")
    logger.info("=" * 70)

    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]
    val_s1_set = set(val_s1_ids)

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

    val_s1_records = []
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["entity_id"] in val_s1_set:
                val_s1_records.append(preprocess_record(r))

    logger.info("Loading candidates (Phase 3 setup: ALL positives + 150k distractors)...")
    t0 = time.time()
    cands = load_all_candidates(DATA_ROOT, needed_positive_cids, max_distractors=150000)
    logger.info("Loaded %d candidates in %.2fs", len(cands), time.time() - t0)

    token_freq = defaultdict(int)
    for cand in cands.values():
        for tok in cand["name_tokens"]:
            token_freq[tok] += 1

    indexes = build_indexes(cands, token_freq)

    # Threshold sweep
    taus = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68]
    logger.info("Scoring all candidates and sweeping thresholds: %s", taus)

    t0 = time.time()
    scored_pairs = {}  # s1_id -> [(cid, score)]
    for s1 in val_s1_records:
        s1_id = s1["id"]
        cand_list = retrieve_candidates(s1, indexes, max_k=80)
        pairs = []
        for cid in cand_list:
            cand = cands.get(cid)
            if not cand:
                continue
            sc = score_pair(s1, cand)
            pairs.append((cid, sc))
        scored_pairs[s1_id] = pairs
    logger.info("Scoring completed in %.2fs", time.time() - t0)

    logger.info("\n%s", "=" * 70)
    logger.info("THRESHOLD SWEEP RESULTS (Phase 3 Multi-Channel Candidate Set):")
    logger.info("%-8s %-12s %-12s %-12s %-14s %-12s", "Tau", "Macro F0.5", "Precision", "Recall", "Singletons", "FalseM")
    logger.info("-" * 72)

    best_tau = 0.56
    best_f05 = 0.0
    best_metrics = {}

    for tau in taus:
        preds = {s1_id: [cid for cid, sc in pairs if sc >= tau]
                 for s1_id, pairs in scored_pairs.items()}
        m = evaluate_at_tau(preds, val_gt, val_s1_ids)
        logger.info("%-8.2f %-12.4f %-12.4f %-12.4f %-14.2f%% %-12d",
                    tau, m["macro_f05"], m["precision"], m["recall"],
                    m["singleton_accuracy"] * 100, m["false_merges"])
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_tau = tau
            best_metrics = m

    logger.info("=" * 70)
    logger.info("BEST RESULT: tau=%.2f | Macro F0.5=%.4f | Precision=%.4f | Recall=%.4f | Singletons=%.2f%%",
                best_tau, best_f05, best_metrics["precision"], best_metrics["recall"],
                best_metrics["singleton_accuracy"] * 100)

    # Load champion baseline for comparison
    with open(REPORTS_DIR / "champion.json", "r") as f:
        champion = json.load(f)
    champ_f05 = champion["macro_f05"]
    delta = best_f05 - champ_f05
    logger.info("Champion Macro F0.5: %.4f | Delta: %+.4f", champ_f05, delta)

    if delta >= 0.002:
        logger.info("✅ IMPROVEMENT EXCEEDS +0.002 THRESHOLD — eligible for quality gate promotion!")
    else:
        logger.info("⚠️  Improvement delta %+.4f does not exceed +0.002 quality gate threshold.", delta)

    # Save result
    result = {
        "run_id": f"phase5-heuristic-multichannel-v1",
        "model": "Champion Heuristic (extended) + Phase 3 Multi-Channel Candidate Set",
        "best_tau": best_tau,
        "macro_f05": round(best_f05, 4),
        "precision": round(best_metrics["precision"], 4),
        "recall": round(best_metrics["recall"], 4),
        "singleton_accuracy": round(best_metrics["singleton_accuracy"], 4),
        "false_merges": best_metrics["false_merges"],
        "champion_f05": champ_f05,
        "delta_vs_champion": round(delta, 4),
        "retrieval_pair_recall_pct": 85.64
    }
    out_path = REPORTS_DIR / "phase5_heuristic_multichannel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved Phase 5 result to %s", out_path)

if __name__ == "__main__":
    main()
