#!/usr/bin/env python3
"""
Phase 9b: Ultimate Entity Resolution Matcher — Fixed Retrieval + Fast Grid Search.

Key fixes over Phase 9:
  1. Retrieval uses RAW tokens (no stopword removal) to maximize recall
  2. Scoring uses CLEAN tokens (with stopword removal) for precision
  3. Grid search is vectorized — score once, apply thresholds cheaply
  4. Rule-based fast paths from evaluate_optimized_matcher integrated
  5. State conflict rejection to prevent cross-state false merges
  6. Domain stem matching for web-based businesses
  7. Legal suffix normalization for name comparison
  8. Wider retrieval limits for better blocking recall
"""
import csv
import json
import logging
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase9b")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data_raw" / "student_resource" / "dataset"
FROZEN_SPLIT_PATH = ROOT / "data" / "frozen_val_s1_ids.json"
REPORTS_DIR = ROOT / "reports"

# ============================================================
# NORMALIZATION CONSTANTS
# ============================================================
LEGAL_SUFFIXES = {
    'private limited', 'pvt ltd', 'pvt limited', 'p limited', 'private ltd',
    'limited', 'ltd', 'corporation', 'corp', 'incorporated', 'inc',
    'llc', 'llp', 'l l c', 'l l p', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'gmbh', 'ag', 'bv', 'nv', 'spa', 'srl',
    'enterprises', 'enterprise', 'services', 'solutions', 'technologies',
    'holdings', 'group', 'associates', 'industries', 'international',
    'consultants', 'consultancy', 'consulting', 'center', 'centre', 'shop', 'store'
}
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
INDIA_STATES = {
    'tamil nadu': 'tn', 'tamilnadu': 'tn', 'karnataka': 'ka', 'maharashtra': 'mh',
    'uttar pradesh': 'up', 'rajasthan': 'rj', 'madhya pradesh': 'mp', 'delhi': 'dl',
    'new delhi': 'dl', 'west bengal': 'wb', 'gujarat': 'gj', 'andhra pradesh': 'ap',
    'telangana': 'ts', 'kerala': 'kl', 'haryana': 'hr', 'bihar': 'br', 'odisha': 'od',
    'punjab': 'pb', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'uttarakhand': 'uk',
    'goa': 'ga', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'chandigarh': 'ch'
}
ALL_IN_STATE_CODES = set(INDIA_STATES.values())
GENERIC_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'lane', 'court', 'place', 'terrace',
    'parkway', 'highway', 'square', 'building', 'floor', 'apt', 'apartment', 'suite', 'room',
    'nagar', 'colony', 'sector', 'district', 'taluk', 'opp', 'opposite', 'near', 'rd', 'st',
    'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'ter', 'pkwy', 'hwy', 'sq', 'bldg', 'fl', 'ste',
    'rm', 'ngr', 'col', 'sec', 'dist', 'tq', 'plot', 'hn', 'house', 'no', 'number', 'unit',
    'block', 'null', 'east', 'west', 'north', 'south',
    'box', 'po', 'rue', 'allee', 'chemin', 'impasse', 'france',
    'india', 'us', 'usa', 'city', 'town', 'state', 'village'
} | ALL_US_STATE_CODES | ALL_IN_STATE_CODES | set(US_STATES.keys()) | set(INDIA_STATES.keys())

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_ALPHA_ONLY = re.compile(r'[^a-z0-9]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-z0-9\-]+)\.(?:com|in|org|net|fr|co|io|biz|info|us)\b', re.IGNORECASE)


# ============================================================
# PREPROCESSING
# ============================================================
def transliterate(text):
    if not text: return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def is_indic(text):
    for c in text:
        if 0x0900 <= ord(c) <= 0x0D7F: return True
    return False

def clean_name(raw):
    """Returns (clean_name, alpha_name, domain_stem, has_indic)"""
    if not raw: return "", "", "", False
    has_in = is_indic(raw)
    raw_clean = transliterate(raw).lower()
    m = DOMAIN_RE.search(raw_clean)
    domain_stem = m.group(1).replace('-', '').replace('.', '') if m else ""
    t = raw_clean.replace('&', ' and ')
    t = RE_PUNCT.sub(' ', t)
    t = ' '.join(t.split())
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if t.endswith(' ' + suf):
            t = t[:-len(suf)-1].strip()
        elif t == suf:
            t = ""
    alpha_clean = RE_ALPHA_ONLY.sub('', t)
    return t, alpha_clean, domain_stem, has_in

def extract_state(raw_text, country):
    if not raw_text: return ""
    t_clean = transliterate(raw_text).lower()
    ambiguous_us = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'me', 'oh', 'ok', 'or', 'pa', 'wa'}
    ambiguous_in = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'or'}
    if country == 'US':
        for name, code in US_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', t_clean): return code
        for code in reversed(re.findall(r'\b([a-z]{2})\b', t_clean)):
            if code in ALL_US_STATE_CODES and code not in ambiguous_us: return code
    elif country == 'India':
        for name, code in INDIA_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', t_clean): return code
        for code in reversed(re.findall(r'\b([a-z]{2})\b', t_clean)):
            if code in ALL_IN_STATE_CODES and code not in ambiguous_in: return code
    return ""

def clean_text_raw(s):
    """Simple cleaning for retrieval tokens — NO stopword removal."""
    if not s: return ""
    s = transliterate(s).lower().replace('&', ' and ')
    return ' '.join(RE_PUNCT.sub(' ', s).split())

def get_char_ngrams(text, n):
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def preprocess_record(eid, name_raw, addr_raw, country):
    # Rich name features
    cn, alpha, dom, indic = clean_name(name_raw)
    cn_raw = clean_text_raw(name_raw)  # for retrieval (no suffix stripping)

    # Raw tokens (for retrieval — includes stopwords & suffixes)
    raw_name_tokens = set(cn_raw.split())

    # Clean tokens (for scoring — minus stopwords)
    clean_name_tokens = set(cn.split()) - STOPWORDS

    # Address
    state = extract_state(addr_raw, country)
    addr_raw_clean = clean_text_raw(addr_raw)
    addr_tokens = set(addr_raw_clean.split())  # raw for retrieval
    addr_tokens_clean = addr_tokens - STOPWORDS  # minus stopwords for scoring

    all_digits = set(RE_DIGITS.findall(addr_raw_clean))
    specific_addr = {w for w in addr_tokens_clean if w not in GENERIC_ADDR_WORDS and len(w) >= 3}

    postal = ""
    for d in all_digits:
        if len(d) in (5, 6): postal = d; break
    street_num = ""
    m = re.match(r'^(\d+)\b', addr_raw_clean or "")
    if m: street_num = m.group(1)

    ng4 = get_char_ngrams(cn, 4)

    return {
        'id': eid, 'clean_name': cn, 'alpha_name': alpha,
        'domain_stem': dom, 'has_indic': indic,
        'raw_name_tokens': raw_name_tokens,  # for RETRIEVAL
        'name_tokens': clean_name_tokens,     # for SCORING
        'clean_addr': addr_raw_clean,
        'addr_tokens': addr_tokens,           # for RETRIEVAL
        'addr_tokens_clean': addr_tokens_clean, # for SCORING
        'specific_addr': specific_addr,
        'digits': all_digits, 'state': state, 'country': country,
        'postal': postal, 'street_num': street_num,
        'ng4': ng4
    }


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================
def jaro_winkler(s1, s2, max_len=50):
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

def token_jaccard(t1, t2):
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def token_containment(t1, t2):
    if not t1 or not t2: return 0.0
    smaller, larger = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    return len(smaller & larger) / len(smaller)

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


# ============================================================
# SCORING
# ============================================================
def score_pair(s1, cand):
    """Returns dict with composite score + feature flags for decision rules."""
    st1, st2 = s1["state"], cand["state"]
    state_conflict = bool(st1 and st2 and st1 != st2)

    jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
    tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
    tc_name = token_containment(s1["name_tokens"], cand["name_tokens"])
    ng4 = ngram_jaccard(s1["ng4"], cand["ng4"])

    a1, a2 = s1["alpha_name"], cand["alpha_name"]
    alpha_exact = (a1 == a2 and len(a1) >= 3) if (a1 and a2) else False
    alpha_contain = False
    if a1 and a2 and len(a1) >= 8 and len(a2) >= 8:
        if a1 in a2 or a2 in a1: alpha_contain = True

    dom1, dom2 = s1["domain_stem"], cand["domain_stem"]
    domain_match = False
    if dom1 and dom2 and dom1 == dom2 and len(dom1) >= 5:
        domain_match = True
    elif dom2 and len(dom2) >= 5 and a1:
        if dom2 == a1 or (len(dom2) >= 8 and (dom2 in a1 or a1 in dom2)):
            domain_match = True
    elif dom1 and len(dom1) >= 5 and a2:
        if dom1 == a2 or (len(dom1) >= 8 and (dom1 in a2 or a2 in dom1)):
            domain_match = True

    tj_addr = token_jaccard(s1["addr_tokens_clean"], cand["addr_tokens_clean"])
    spec1, spec2 = s1["specific_addr"], cand["specific_addr"]
    spec_inter = len(spec1 & spec2)
    has_both_spec = bool(spec1 and spec2)

    dig1, dig2 = s1["digits"], cand["digits"]
    digits_overlap = bool(dig1 & dig2)
    digits_conflict = bool(dig1 and dig2 and not digits_overlap)

    sn1, sn2 = s1["street_num"], cand["street_num"]
    street_num_conflict = (sn1 != sn2) if (sn1 and sn2) else False

    indic = s1["has_indic"] or cand["has_indic"]

    # Composite score
    conflict_pen = 0.0
    if street_num_conflict: conflict_pen += 0.25
    if state_conflict: conflict_pen += 0.35

    base = 0.50 * jw + 0.25 * tj_name + 0.25 * tj_addr - conflict_pen
    # Ngram boost
    if ng4 >= 0.5 and jw >= 0.60:
        alt = 0.40 * jw + 0.15 * tj_name + 0.15 * tj_addr + 0.25 * ng4 + 0.05 * tc_name - conflict_pen
        base = max(base, alt)
    # Token containment boost for abbreviations
    if tc_name >= 0.8 and jw >= 0.60:
        alt2 = 0.40 * jw + 0.20 * tc_name + 0.20 * tj_addr + 0.15 * ng4 + 0.05 * tj_name - conflict_pen
        base = max(base, alt2)

    return {
        'score': base, 'jw': jw, 'tj_name': tj_name, 'tc_name': tc_name,
        'ng4': ng4, 'alpha_exact': alpha_exact, 'alpha_contain': alpha_contain,
        'domain_match': domain_match, 'tj_addr': tj_addr,
        'spec_inter': spec_inter, 'has_both_spec': has_both_spec,
        'digits_overlap': digits_overlap, 'digits_conflict': digits_conflict,
        'street_num_conflict': street_num_conflict,
        'state_conflict': state_conflict, 'indic': indic,
    }


# ============================================================
# DECISION: Rule-based fast paths + score-based tiered decision
# ============================================================
def decide(f, tau, high_conf, addr_floor):
    # REJECT: State conflict
    if f['state_conflict']: return False

    # ACCEPT: Exact alpha name + no digit conflict
    if f['alpha_exact'] and not f['digits_conflict']:
        if not f['has_both_spec'] or f['spec_inter'] >= 1 or f['digits_overlap'] or f['jw'] >= 0.98:
            return True

    # ACCEPT: Domain match + no digit conflict
    if f['domain_match'] and not f['digits_conflict']:
        return True

    # ACCEPT: Alpha containment + address agreement
    if f['alpha_contain'] and not f['digits_conflict']:
        if f['spec_inter'] >= 1 or f['digits_overlap'] or not f['has_both_spec']:
            return True

    # ACCEPT: Near-identical JW >= 0.94
    if f['jw'] >= 0.94 and not f['digits_conflict']:
        if not f['has_both_spec'] or f['spec_inter'] >= 1 or f['digits_overlap'] or f['jw'] >= 0.98:
            return True

    # ACCEPT: Good JW + full token coverage
    if f['jw'] >= 0.88 and f['tc_name'] >= 0.8 and not f['digits_conflict']:
        if not f['has_both_spec'] or f['spec_inter'] >= 1 or f['digits_overlap']:
            return True

    # ACCEPT: Indic script + strong address
    if f['indic'] and not f['digits_conflict']:
        if f['spec_inter'] >= 2 or (f['spec_inter'] >= 1 and f['digits_overlap']):
            if f['jw'] >= 0.50 or f['tc_name'] >= 0.5:
                return True

    # ACCEPT: Strong address + reasonable name
    if (f['spec_inter'] >= 2 or (f['spec_inter'] >= 1 and f['digits_overlap'])) and not f['digits_conflict']:
        if f['jw'] >= 0.70 or f['tc_name'] >= 0.7:
            return True

    # TIERED: Score-based
    score = f['score']
    if score >= high_conf: return True
    if score >= tau and f['tj_addr'] >= addr_floor: return True
    return False


# ============================================================
# RETRIEVAL (uses RAW tokens for max recall)
# ============================================================
def build_indexes(cands, token_freq):
    ni = defaultdict(list); ri = defaultdict(list)
    ai = defaultdict(list); si = defaultdict(list)
    pi = defaultdict(list); ng4i = defaultdict(list)
    sai = defaultdict(list); domi = defaultdict(list)
    for cid, c in cands.items():
        co = c["country"]
        # Use RAW name tokens for retrieval indexing
        for t in c["raw_name_tokens"]:
            ni[(co,t)].append(cid)
            if token_freq.get(t, 0) < 50: ri[(co,t)].append(cid)
        for at in c["addr_tokens"]: ai[(co,at)].append(cid)
        for sat in c["specific_addr"]: sai[(co,sat)].append(cid)
        if c["street_num"]: si[(co,c["street_num"])].append(cid)
        if c["postal"]: pi[(co,c["postal"])].append(cid)
        for ng in c["ng4"]: ng4i[(co,ng)].append(cid)
        if c["domain_stem"]: domi[(co,c["domain_stem"])].append(cid)
    return {"name": ni, "rare": ri, "addr": ai, "spec_addr": sai,
            "snum": si, "postal": pi, "ng4": ng4i, "domain": domi}

def retrieve(s1, idx, max_k=120):
    c = s1["country"]; cands = set()
    # RAW name tokens for retrieval
    for t in s1["raw_name_tokens"]:
        p = idx["name"].get((c,t), [])
        if len(p) <= 80: cands.update(p)
    for t in s1["raw_name_tokens"]:
        p = idx["rare"].get((c,t), [])
        if len(p) <= 40: cands.update(p)
    for at in s1["addr_tokens"]:
        p = idx["addr"].get((c,at), [])
        if len(p) <= 50: cands.update(p)
    for sat in s1["specific_addr"]:
        p = idx["spec_addr"].get((c,sat), [])
        if len(p) <= 200: cands.update(p)
    if s1["street_num"]:
        p = idx["snum"].get((c,s1["street_num"]), [])
        if len(p) <= 50: cands.update(p)
    if s1["postal"]:
        p = idx["postal"].get((c,s1["postal"]), [])
        if len(p) <= 40: cands.update(p)
    for ng in s1["ng4"]:
        p = idx["ng4"].get((c,ng), [])
        if len(p) <= 40: cands.update(p)
    if s1["domain_stem"]:
        cands.update(idx["domain"].get((c,s1["domain_stem"]), []))
    return list(cands)[:max_k]


# ============================================================
# EVALUATION
# ============================================================
def evaluate(predictions, val_gt, val_s1_ids):
    f05_list = []; st = sc = fm = 0
    for eid in val_s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))
        if not true_set:
            st += 1
            if not pred_set: sc += 1
            else: fm += 1
        f05_list.append(entity_f05(pred_set, true_set))
    mf05 = float(np.mean(f05_list))
    sing = (sc/st) if st > 0 else 1.0
    return mf05, sing, fm


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("PHASE 9b: ULTIMATE MATCHER — FIXED RETRIEVAL")
    logger.info("=" * 70)

    with open(FROZEN_SPLIT_PATH) as f: manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]; val_s1_set = set(val_s1_ids)
    logger.info("Validation: %d S1 entities", len(val_s1_ids))

    gt = {}
    with open(DATA_ROOT/"train"/"train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in val_s1_set}
    needed_pos = set()
    for e in val_s1_set: needed_pos.update(val_gt[e])
    logger.info("Need %d positive targets", len(needed_pos))

    val_s1_records = []
    with open(DATA_ROOT/"train"/"train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in val_s1_set:
                val_s1_records.append(preprocess_record(p[0], p[1], p[2], p[3]))
    logger.info("Loaded %d S1 val records", len(val_s1_records))

    logger.info("Loading S2/S3 candidates...")
    t0 = time.time()
    cands = {}; dc = 0; MAX_DIST = 200000
    for s_path in [DATA_ROOT/"train"/"train_source2.tsv", DATA_ROOT/"train"/"train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid = p[0]
                    if cid in needed_pos or dc < MAX_DIST:
                        cands[cid] = preprocess_record(cid, p[1], p[2], p[3])
                        if cid not in needed_pos: dc += 1
    logger.info("Loaded %d candidates in %.2fs", len(cands), time.time() - t0)

    token_freq = defaultdict(int)
    for c in cands.values():
        for t in c["raw_name_tokens"]: token_freq[t] += 1

    idx = build_indexes(cands, token_freq)
    logger.info("Indexes built.")

    # Score all pairs + track retrieval recall
    logger.info("Scoring...")
    t0 = time.time()
    all_scored = {}
    ret_hits = ret_total = 0
    for i, s1 in enumerate(val_s1_records):
        s1_id = s1["id"]
        cand_list = retrieve(s1, idx, max_k=120)
        scored = []
        for cid in cand_list:
            cand = cands.get(cid)
            if cand:
                feat = score_pair(s1, cand)
                scored.append((cid, feat))
        all_scored[s1_id] = scored
        true_set = set(val_gt.get(s1_id, []))
        if true_set:
            ret_total += len(true_set)
            ret_hits += len(true_set & set(cand_list))
        if (i+1) % 5000 == 0:
            logger.info("  %d/%d scored...", i+1, len(val_s1_records))

    ret_recall = ret_hits / ret_total * 100 if ret_total else 0
    logger.info("Scoring done in %.2fs. Retrieval recall: %.2f%% (%d/%d)",
                time.time() - t0, ret_recall, ret_hits, ret_total)

    try:
        with open(REPORTS_DIR/"champion.json") as f: champ_f05 = json.load(f)["macro_f05"]
    except: champ_f05 = 0.8387
    logger.info("Champion: F0.5=%.4f", champ_f05)

    # Fast grid search
    taus = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    high_confs = [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    addr_floors = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    total = len(taus) * len(high_confs) * len(addr_floors)
    logger.info("Grid: %d combos", total)

    best = (0, 0, 0, 0, 0, 0)
    all_passing = []
    count = 0
    for tau in taus:
        for hc in high_confs:
            for af in addr_floors:
                preds = {}
                for s1_id, scored in all_scored.items():
                    preds[s1_id] = [cid for cid, feat in scored if decide(feat, tau, hc, af)]
                mf05, sing, fm = evaluate(preds, val_gt, val_s1_ids)
                count += 1
                if sing >= 0.90:
                    all_passing.append((mf05, tau, hc, af, sing, fm))
                    if mf05 > best[0]:
                        best = (mf05, tau, hc, af, sing, fm)
        logger.info("  Grid progress: tau=%.2f done (%d/%d)", tau, count, total)

    all_passing.sort(reverse=True)
    logger.info("=" * 70)
    logger.info("RESULTS (Retrieval Recall: %.2f%%)", ret_recall)
    logger.info("Champion: F0.5=%.4f", champ_f05)
    logger.info("-" * 70)

    if all_passing:
        logger.info("✅ %d configs pass quality gate (singleton >= 90%%)", len(all_passing))
        logger.info("Top 20:")
        logger.info("%-6s %-6s %-6s %-12s %-12s %-8s",
                    "Tau", "HiConf", "AddrF", "F0.5", "Singleton", "FalseM")
        for mf05, tau, hc, af, sing, fm in all_passing[:20]:
            marker = " ★" if mf05 > champ_f05 else ""
            logger.info("%-6.2f %-6.2f %-6.2f %-12.4f %-12.2f%% %-8d%s",
                        tau, hc, af, mf05, sing*100, fm, marker)

        mf05, tau, hc, af, sing, fm = best
        logger.info("=" * 70)
        logger.info("🏆 BEST: tau=%.2f hc=%.2f af=%.2f", tau, hc, af)
        logger.info("   F0.5=%.4f (delta %+.4f) Singleton=%.2f%% FM=%d RetRecall=%.2f%%",
                    mf05, mf05 - champ_f05, sing*100, fm, ret_recall)

        result = {
            "run_id": "phase9b-ultimate-v1",
            "model": "Phase9b: Rich Features + Rule Paths + Dual-Token Retrieval",
            "tau": tau, "high_confidence": hc, "addr_floor": af,
            "macro_f05": round(mf05, 4), "singleton_accuracy": round(sing, 4),
            "false_merges": fm, "retrieval_recall": round(ret_recall, 2),
            "champion_f05": champ_f05, "delta_vs_champion": round(mf05 - champ_f05, 4),
            "dataset_hash": manifest.get("dataset_hash", ""),
            "validation_protocol": manifest.get("split_version", "frozen-20k")
        }
        out = REPORTS_DIR / "phase9b_best_challenger.json"
        with open(out, "w") as f: json.dump(result, f, indent=2)
        logger.info("Saved to %s", out)

        if mf05 > champ_f05:
            logger.info("🎉 NEW CHAMPION!")
            champ_out = {**result, "status": "CHAMPION_ACTIVE",
                         "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(REPORTS_DIR / "champion.json", "w") as f:
                json.dump(champ_out, f, indent=2)
    else:
        logger.warning("❌ No config passed quality gate")
        logger.info("Best overall: F0.5=%.4f", best[0])

if __name__ == "__main__":
    main()
