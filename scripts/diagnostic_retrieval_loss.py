#!/usr/bin/env python3
"""
diagnostic_retrieval_loss.py
Pinpoints why retrieval recall dropped to 70.56% in Phase 9b vs 85.64% in Phase 3 / Phase 8b.
Runs on 1,000 validation entities with ground truth targets.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)

LEGAL_SUFFIXES = [
    'private limited', 'pvt limited', 'p limited', 'private ltd', 'pvt ltd',
    'corporation', 'incorporated', 'limited', 'enterprises', 'enterprise',
    'services', 'solutions', 'technologies', 'holdings', 'industries',
    'international', 'consultants', 'consultancy', 'consulting', 'center',
    'centre', 'corp', 'inc', 'llc', 'llp', 'sarl', 'sasu', 'eurl', 'gmbh',
    'gie', 'sas', 'sa', 'ag', 'bv', 'nv', 'spa', 'srl', 'ltd'
]

STOPWORDS = {
    'the', 'and', 'co', 'of', 'in', 'for', 'at', 'by', 'to', 'a', 'an',
    'de', 'la', 'le', 'les', 'des', 'du', 'et', 'en', 'au', 'aux', 'd', 'l',
    'un', 'une', 'sur', 'dans', 'null', 'mr', 'mrs', 'dr', 'prof'
}

def clean_text_p8(s):
    if not s: return ""
    return RE_PUNCT.sub(' ', str(s).lower()).strip()

def get_char_ngrams(text, n):
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def get_street_num(addr):
    m = re.match(r'^(\d+)\b', addr or "")
    return m.group(1) if m else ""

def get_postal(addr):
    for d in RE_DIGITS.findall(addr or ""):
        if len(d) in (5, 6): return d
    return ""

def clean_name_p9(raw):
    if not raw: return ""
    t = RE_PUNCT.sub(' ', str(raw).lower().replace('&', ' and '))
    t = ' '.join(t.split())
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
        elif t == suf:
            t = ""
    return t

def main():
    print("Loading validation IDs and GT...")
    with open(FROZEN_SPLIT_PATH) as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]

    gt = {}
    with open(DATA_ROOT / "train" / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []

    # Filter to first 1000 S1 records with positive targets
    s1_subset = []
    val_set = set(val_s1_ids)
    needed_pos = set()
    
    with open(DATA_ROOT / "train" / "train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in val_set:
                targets = gt.get(p[0], [])
                if targets:
                    s1_subset.append({
                        "id": p[0], "name": p[1], "addr": p[2], "country": p[3], "targets": targets
                    })
                    needed_pos.update(targets)
                    if len(s1_subset) >= 1000:
                        break

    print(f"Sampled {len(s1_subset)} S1 entities with {sum(len(x['targets']) for x in s1_subset)} total true mentions")
    print(f"Needed positive targets in pool: {len(needed_pos)}")

    # Load candidates (positives + 50k distractors)
    cands_raw = {}
    for s_path in [DATA_ROOT / "train" / "train_source2.tsv", DATA_ROOT / "train" / "train_source3.tsv"]:
        dc = 0
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid = p[0]
                    is_n = cid in needed_pos
                    if is_n or dc < 50000:
                        cands_raw[cid] = {"id": cid, "name": p[1], "addr": p[2], "country": p[3]}
                        if not is_n: dc += 1

    print(f"Loaded {len(cands_raw)} candidates")

    # ========================================================
    # STRATEGY 1: Phase 8b exact retrieval
    # ========================================================
    p8_cands = {}
    for cid, c in cands_raw.items():
        cn = clean_text_p8(c["name"])
        ca = clean_text_p8(c["addr"])
        p8_cands[cid] = {
            "id": cid, "country": c["country"],
            "name_tokens": set(cn.split()),
            "addr_tokens": set(ca.split()),
            "street_num": get_street_num(ca),
            "postal": get_postal(ca),
            "ngrams4": get_char_ngrams(cn, 4)
        }

    token_freq = defaultdict(int)
    for c in p8_cands.values():
        for t in c["name_tokens"]: token_freq[t] += 1

    p8_ni = defaultdict(list); p8_ri = defaultdict(list)
    p8_ai = defaultdict(list); p8_si = defaultdict(list)
    p8_pi = defaultdict(list); p8_ng4i = defaultdict(list)
    for cid, c in p8_cands.items():
        co = c["country"]
        for t in c["name_tokens"]:
            p8_ni[(co,t)].append(cid)
            if token_freq[t] < 50: p8_ri[(co,t)].append(cid)
        for at in c["addr_tokens"]: p8_ai[(co,at)].append(cid)
        if c["street_num"]: p8_si[(co,c["street_num"])].append(cid)
        if c["postal"]: p8_pi[(co,c["postal"])].append(cid)
        for ng in c["ngrams4"]: p8_ng4i[(co,ng)].append(cid)

    total_true = sum(len(x["targets"]) for x in s1_subset)
    
    # Test Phase 8b retrieval at max_k = 80, 120, 150
    for mk in [80, 120, 150]:
        recovered = 0
        for s1 in s1_subset:
            cn = clean_text_p8(s1["name"])
            ca = clean_text_p8(s1["addr"])
            co = s1["country"]
            nt = set(cn.split())
            at = set(ca.split())
            sn = get_street_num(ca)
            po = get_postal(ca)
            ng4 = get_char_ngrams(cn, 4)

            cset = set()
            for t in nt:
                p = p8_ni.get((co,t), [])
                if len(p) <= 60: cset.update(p)
            for t in nt:
                p = p8_ri.get((co,t), [])
                if len(p) <= 30: cset.update(p)
            for a in at:
                p = p8_ai.get((co,a), [])
                if len(p) <= 40: cset.update(p)
            if sn:
                p = p8_si.get((co,sn), [])
                if len(p) <= 40: cset.update(p)
            if po:
                p = p8_pi.get((co,po), [])
                if len(p) <= 30: cset.update(p)
            for ng in ng4:
                p = p8_ng4i.get((co,ng), [])
                if len(p) <= 30: cset.update(p)

            clist = list(cset)[:mk]
            true_set = set(s1["targets"])
            recovered += len(set(clist) & true_set)

        print(f"Strategy 1 (Phase 8b exact, max_k={mk}): Recall = {recovered/total_true*100:.2f}% ({recovered}/{total_true})")

    # ========================================================
    # STRATEGY 2: Phase 9b retrieval (with stripped suffixes on ngrams)
    # ========================================================
    for mk in [80, 120, 150]:
        recovered = 0
        for s1 in s1_subset:
            cn_stripped = clean_name_p9(s1["name"])
            cn_raw = clean_text_p8(s1["name"])
            ca = clean_text_p8(s1["addr"])
            co = s1["country"]
            nt_raw = set(cn_raw.split())
            at = set(ca.split())
            sn = get_street_num(ca)
            po = get_postal(ca)
            # Notice: Phase 9b computed ng4 on STRIPPED name!
            ng4_stripped = get_char_ngrams(cn_stripped, 4)

            cset = set()
            for t in nt_raw:
                p = p8_ni.get((co,t), [])
                if len(p) <= 80: cset.update(p)
            for t in nt_raw:
                p = p8_ri.get((co,t), [])
                if len(p) <= 40: cset.update(p)
            for a in at:
                p = p8_ai.get((co,a), [])
                if len(p) <= 50: cset.update(p)
            if sn:
                p = p8_si.get((co,sn), [])
                if len(p) <= 50: cset.update(p)
            if po:
                p = p8_pi.get((co,po), [])
                if len(p) <= 40: cset.update(p)
            for ng in ng4_stripped:
                p = p8_ng4i.get((co,ng), [])
                if len(p) <= 40: cset.update(p)

            clist = list(cset)[:mk]
            true_set = set(s1["targets"])
            recovered += len(set(clist) & true_set)

        print(f"Strategy 2 (Phase 9b stripped ng4, max_k={mk}): Recall = {recovered/total_true*100:.2f}% ({recovered}/{total_true})")

    # ========================================================
    # STRATEGY 3: High-Recall Multi-Channel (+ 3-grams, raw 4-grams, posting caps)
    # ========================================================
    p3_ng3i = defaultdict(list)
    for cid, c in p8_cands.items():
        co = c["country"]
        cn = clean_text_p8(cands_raw[cid]["name"])
        for ng in get_char_ngrams(cn, 3):
            p3_ng3i[(co,ng)].append(cid)

    for mk in [80, 120, 150, 200]:
        recovered = 0
        for s1 in s1_subset:
            cn = clean_text_p8(s1["name"])
            ca = clean_text_p8(s1["addr"])
            co = s1["country"]
            nt = set(cn.split())
            at = set(ca.split())
            sn = get_street_num(ca)
            po = get_postal(ca)
            ng3 = get_char_ngrams(cn, 3)
            ng4 = get_char_ngrams(cn, 4)

            cset = set()
            for t in nt:
                p = p8_ni.get((co,t), [])
                if len(p) <= 100: cset.update(p)
            for t in nt:
                p = p8_ri.get((co,t), [])
                if len(p) <= 50: cset.update(p)
            for a in at:
                p = p8_ai.get((co,a), [])
                if len(p) <= 60: cset.update(p)
            if sn:
                p = p8_si.get((co,sn), [])
                if len(p) <= 60: cset.update(p)
            if po:
                p = p8_pi.get((co,po), [])
                if len(p) <= 50: cset.update(p)
            for ng in ng4:
                p = p8_ng4i.get((co,ng), [])
                if len(p) <= 50: cset.update(p)
            for ng in ng3:
                p = p3_ng3i.get((co,ng), [])
                if len(p) <= 25: cset.update(p)

            clist = list(cset)[:mk]
            true_set = set(s1["targets"])
            recovered += len(set(clist) & true_set)

        print(f"Strategy 3 (High-Recall +3gram, max_k={mk}): Recall = {recovered/total_true*100:.2f}% ({recovered}/{total_true})")

if __name__ == "__main__":
    main()
