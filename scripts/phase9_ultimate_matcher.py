#!/usr/bin/env python3
"""
Phase 9: Ultimate Entity Resolution Matcher.

Combines the best of:
  - evaluate_optimized_matcher.py: rich preprocessing (state extraction, domain stems,
    Indic detection, legal suffix normalization, specific address tokens, stopword removal)
  - phase8b: multi-channel inverted-index retrieval + tiered decision rule

New improvements:
  1. Richer scoring with 12+ signals (vs 4 in phase8b)
  2. Rule-based fast paths for exact/near-exact matches
  3. State conflict rejection (prevents cross-state false merges)
  4. Domain stem matching for web-based business names
  5. Legal suffix-agnostic name comparison
  6. Specific address token matching (excludes generic road/district terms)
  7. Token containment ratio (handles abbreviations)
  8. Indic-script aware matching
  9. Better ngram channel (3-grams + 4-grams)
  10. Wider retrieval with stricter scoring
  11. Fine-grained grid search over decision parameters
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
logger = logging.getLogger("Phase9")

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

ADDR_EXPANSIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'ter': 'terrace',
    'pkwy': 'parkway', 'hwy': 'highway', 'sq': 'square', 'bldg': 'building',
    'fl': 'floor', 'apt': 'apartment', 'ste': 'suite', 'rm': 'room',
    'ngr': 'nagar', 'col': 'colony', 'sec': 'sector', 'dist': 'district',
    'tq': 'taluk', 'opp': 'opposite', 'nr': 'near'
}

RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_ALPHA_ONLY = re.compile(r'[^a-z0-9]')
RE_DIGITS = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\b([a-z0-9\-]+)\.(?:com|in|org|net|fr|co|io|biz|info|us)\b', re.IGNORECASE)


# ============================================================
# PREPROCESSING
# ============================================================
def transliterate(text: str) -> str:
    if not text: return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def is_indic(text: str) -> bool:
    for c in text:
        if 0x0900 <= ord(c) <= 0x0D7F:
            return True
    return False

def clean_name(raw: str):
    """Returns (clean_name, alpha_name, domain_stem, has_indic)"""
    if not raw or not isinstance(raw, str):
        return "", "", "", False
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

def extract_state(raw_text: str, country: str) -> str:
    if not raw_text: return ""
    t_clean = transliterate(raw_text).lower()
    if country == 'US':
        for name, code in US_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', t_clean):
                return code
        m = re.findall(r'\b([a-z]{2})\b', t_clean)
        ambiguous = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'me', 'oh', 'ok', 'or', 'pa', 'wa'}
        for code in reversed(m):
            if code in ALL_US_STATE_CODES and code not in ambiguous:
                return code
    elif country == 'India':
        for name, code in INDIA_STATES.items():
            if re.search(r'\b' + re.escape(name) + r'\b', t_clean):
                return code
        m = re.findall(r'\b([a-z]{2})\b', t_clean)
        ambiguous = {'st', 'rd', 'dr', 'ln', 'ct', 'pl', 'fl', 'in', 'or'}
        for code in reversed(m):
            if code in ALL_IN_STATE_CODES and code not in ambiguous:
                return code
    return ""

def clean_address(raw: str, country: str):
    """Returns (norm_addr, all_tokens, specific_tokens, digits, state_code)"""
    if not raw or not isinstance(raw, str):
        return "", set(), set(), set(), ""
    state_code = extract_state(raw, country)
    t = transliterate(raw).lower().replace('&', ' and ')
    t = RE_PUNCT.sub(' ', t)
    words = t.split()
    expanded = [ADDR_EXPANSIONS.get(w, w) for w in words]
    norm_addr = ' '.join(expanded)
    all_digits = set(RE_DIGITS.findall(norm_addr))
    all_tokens = set(expanded) - STOPWORDS
    specific_tokens = {w for w in all_tokens if w not in GENERIC_ADDR_WORDS and len(w) >= 3}
    return norm_addr, all_tokens, specific_tokens, all_digits, state_code

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(' ', '')
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def get_postal(addr_digits: set) -> str:
    for d in addr_digits:
        if len(d) in (5, 6): return d
    return ""

def get_street_num(norm_addr: str) -> str:
    m = re.match(r'^(\d+)\b', norm_addr or "")
    return m.group(1) if m else ""

def preprocess_record(eid, name_raw, addr_raw, country):
    cn, alpha, dom, indic = clean_name(name_raw)
    norm_a, atoks, spectoks, digs, state = clean_address(addr_raw, country)
    ntoks = set(cn.split()) - STOPWORDS
    postal = get_postal(digs)
    street_num = get_street_num(norm_a)
    ng3 = get_char_ngrams(cn, 3)
    ng4 = get_char_ngrams(cn, 4)
    return {
        'id': eid, 'clean_name': cn, 'alpha_name': alpha,
        'domain_stem': dom, 'has_indic': indic,
        'name_tokens': ntoks, 'clean_addr': norm_a,
        'all_addr': atoks, 'specific_addr': spectoks,
        'digits': digs, 'state': state, 'country': country,
        'postal': postal, 'street_num': street_num,
        'ng3': ng3, 'ng4': ng4
    }


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================
def jaro_winkler(s1: str, s2: str, max_len: int = 50) -> float:
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

def token_containment(t1: set, t2: set) -> float:
    """Fraction of smaller set contained in larger. Good for abbreviations."""
    if not t1 or not t2: return 0.0
    smaller, larger = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    return len(smaller & larger) / len(smaller) if smaller else 0.0

def ngram_jaccard(ng1: set, ng2: set) -> float:
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def sorted_token_sim(s1: str, s2: str) -> float:
    """Sort tokens alphabetically then compute JW — order-invariant."""
    t1 = ' '.join(sorted(s1.split()))
    t2 = ' '.join(sorted(s2.split()))
    return jaro_winkler(t1, t2)

def digit_jaccard(d1: set, d2: set) -> float:
    if not d1 and not d2: return 1.0
    if not d1 or not d2: return 0.0
    return len(d1 & d2) / len(d1 | d2)


# ============================================================
# ENTITY F0.5
# ============================================================
def entity_f05(pred_set, true_set):
    if not pred_set and not true_set: return 1.0
    if not pred_set or not true_set: return 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    p = tp/len(pred_set); r = tp/len(true_set)
    denom = 0.25 * p + r
    return 1.25 * (p * r) / denom if denom else 0.0


# ============================================================
# SCORING: Rich multi-signal score
# ============================================================
def score_pair(s1: dict, cand: dict):
    """
    Returns a rich feature dict for tiered decision making.
    """
    # --- State conflict: immediate rejection signal ---
    st1, st2 = s1["state"], cand["state"]
    state_conflict = bool(st1 and st2 and st1 != st2)

    # --- Name features ---
    jw = jaro_winkler(s1["clean_name"], cand["clean_name"])
    jw_sorted = sorted_token_sim(s1["clean_name"], cand["clean_name"])
    tj_name = token_jaccard(s1["name_tokens"], cand["name_tokens"])
    tc_name = token_containment(s1["name_tokens"], cand["name_tokens"])
    ng3 = ngram_jaccard(s1["ng3"], cand["ng3"])
    ng4 = ngram_jaccard(s1["ng4"], cand["ng4"])

    # Alpha name containment
    a1, a2 = s1["alpha_name"], cand["alpha_name"]
    alpha_exact = (a1 == a2 and len(a1) >= 3) if (a1 and a2) else False
    alpha_contain = False
    if a1 and a2 and len(a1) >= 8 and len(a2) >= 8:
        if a1 in a2 or a2 in a1:
            alpha_contain = True

    # Domain stem match
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

    # --- Address features ---
    tj_addr = token_jaccard(s1["all_addr"], cand["all_addr"])
    spec1, spec2 = s1["specific_addr"], cand["specific_addr"]
    spec_inter = len(spec1 & spec2)
    has_both_spec = bool(spec1 and spec2)

    dig1, dig2 = s1["digits"], cand["digits"]
    dj = digit_jaccard(dig1, dig2)
    digits_overlap = bool(dig1 & dig2)
    digits_conflict = bool(dig1 and dig2 and not digits_overlap)

    sn1, sn2 = s1["street_num"], cand["street_num"]
    street_num_match = (sn1 == sn2) if (sn1 and sn2) else False
    street_num_conflict = (sn1 != sn2) if (sn1 and sn2) else False

    postal_match = False
    p1, p2 = s1["postal"], cand["postal"]
    if p1 and p2:
        postal_match = (p1 == p2)

    # Indic
    indic = s1["has_indic"] or cand["has_indic"]

    # --- Composite score ---
    # Start with base weighted score
    conflict_pen = 0.0
    if street_num_conflict: conflict_pen += 0.25
    if state_conflict: conflict_pen += 0.35

    # Use the best of several name similarity approaches
    name_sim = max(jw, jw_sorted)
    
    # Composite: name-heavy but with address signal
    base = 0.45 * name_sim + 0.15 * tj_name + 0.10 * tc_name + 0.15 * tj_addr + 0.10 * ng4 + 0.05 * dj - conflict_pen

    # Boost for ngram agreement when JW is moderate  
    if ng4 >= 0.5 and name_sim >= 0.60:
        alt = 0.35 * name_sim + 0.10 * tj_name + 0.10 * tc_name + 0.15 * tj_addr + 0.25 * ng4 + 0.05 * dj - conflict_pen
        base = max(base, alt)

    # Boost for high token containment (abbreviation pattern)
    if tc_name >= 0.8 and ng3 >= 0.4:
        alt2 = 0.30 * name_sim + 0.15 * tc_name + 0.15 * ng3 + 0.20 * tj_addr + 0.10 * ng4 + 0.10 * dj - conflict_pen
        base = max(base, alt2)

    return {
        'score': base,
        'jw': jw, 'jw_sorted': jw_sorted, 'tj_name': tj_name,
        'tc_name': tc_name, 'ng3': ng3, 'ng4': ng4,
        'alpha_exact': alpha_exact, 'alpha_contain': alpha_contain,
        'domain_match': domain_match,
        'tj_addr': tj_addr, 'spec_inter': spec_inter,
        'has_both_spec': has_both_spec,
        'dj': dj, 'digits_overlap': digits_overlap,
        'digits_conflict': digits_conflict,
        'street_num_match': street_num_match,
        'street_num_conflict': street_num_conflict,
        'postal_match': postal_match,
        'state_conflict': state_conflict,
        'indic': indic,
    }


# ============================================================
# DECISION RULES: Combined rule-based + score-based
# ============================================================
def decide(feat: dict, tau: float, high_conf: float, addr_floor: float) -> bool:
    """
    Multi-tier decision with rule-based fast paths:
    
    Rule 1 (REJECT): State conflict → always reject
    Rule 2 (ACCEPT): Exact alpha name + no digit conflict → accept
    Rule 3 (ACCEPT): Domain stem match + no digit conflict → accept
    Rule 4 (ACCEPT): Alpha name containment + address agreement → accept
    Rule 5 (TIERED): Score-based tiered decision with address guard
    """
    # R1: Hard reject on state conflict
    if feat['state_conflict']:
        return False

    # R2: Exact alpha name match → accept if no digit conflict
    if feat['alpha_exact'] and not feat['digits_conflict']:
        if not feat['has_both_spec'] or feat['spec_inter'] >= 1 or feat['digits_overlap'] or feat['jw'] >= 0.98:
            return True

    # R3: Domain stem match → accept if no digit conflict
    if feat['domain_match'] and not feat['digits_conflict']:
        return True

    # R4: Alpha containment + address agreement
    if feat['alpha_contain'] and not feat['digits_conflict']:
        if feat['spec_inter'] >= 1 or feat['digits_overlap'] or not feat['has_both_spec']:
            return True

    # R5: Near-identical JW (>=0.94) + no conflicts
    if feat['jw'] >= 0.94 and not feat['digits_conflict']:
        if not feat['has_both_spec'] or feat['spec_inter'] >= 1 or feat['digits_overlap'] or feat['jw'] >= 0.98:
            return True

    # R6: Strong name token overlap + good JW
    if feat['jw'] >= 0.88 and feat['tc_name'] >= 0.8 and not feat['digits_conflict']:
        if not feat['has_both_spec'] or feat['spec_inter'] >= 1 or feat['digits_overlap']:
            return True

    # R7: Indic script + strong address match
    if feat['indic'] and not feat['digits_conflict']:
        if feat['spec_inter'] >= 2 or (feat['spec_inter'] >= 1 and feat['digits_overlap']):
            if feat['jw'] >= 0.50 or feat['tc_name'] >= 0.5:
                return True

    # R8: Strong address match + reasonable name
    if (feat['spec_inter'] >= 2 or (feat['spec_inter'] >= 1 and feat['digits_overlap'])) and not feat['digits_conflict']:
        if feat['jw'] >= 0.70 or feat['tc_name'] >= 0.7:
            return True

    # T1: Score-based tiered decision
    score = feat['score']
    if score >= high_conf:
        return True
    if score >= tau and feat['tj_addr'] >= addr_floor:
        return True

    return False


# ============================================================
# RETRIEVAL
# ============================================================
def build_indexes(cands: dict, token_freq: dict) -> dict:
    ni = defaultdict(list); ri = defaultdict(list)
    ai = defaultdict(list); si = defaultdict(list)
    pi = defaultdict(list); ng4i = defaultdict(list)
    sai = defaultdict(list)  # specific addr index
    domi = defaultdict(list)  # domain index
    for cid, c in cands.items():
        co = c["country"]
        for t in c["name_tokens"]:
            ni[(co,t)].append(cid)
            if token_freq.get(t, 0) < 50: ri[(co,t)].append(cid)
        for at in c["all_addr"]: ai[(co,at)].append(cid)
        for sat in c["specific_addr"]: sai[(co,sat)].append(cid)
        if c["street_num"]: si[(co,c["street_num"])].append(cid)
        if c["postal"]: pi[(co,c["postal"])].append(cid)
        for ng in c["ng4"]: ng4i[(co,ng)].append(cid)
        if c["domain_stem"]: domi[(co,c["domain_stem"])].append(cid)
    return {"name": ni, "rare": ri, "addr": ai, "spec_addr": sai,
            "snum": si, "postal": pi, "ng4": ng4i, "domain": domi}

def retrieve(s1: dict, idx: dict, max_k: int = 120) -> list:
    c = s1["country"]
    cands = set()
    
    # Name token channel (wider threshold)
    for t in s1["name_tokens"]:
        p = idx["name"].get((c,t), [])
        if len(p) <= 100: cands.update(p)
    
    # Rare token channel
    for t in s1["name_tokens"]:
        p = idx["rare"].get((c,t), [])
        if len(p) <= 50: cands.update(p)
    
    # Address token channel
    for at in s1["all_addr"]:
        p = idx["addr"].get((c,at), [])
        if len(p) <= 60: cands.update(p)
    
    # Specific address token channel (wider, these are more discriminating)
    for sat in s1["specific_addr"]:
        p = idx["spec_addr"].get((c,sat), [])
        if len(p) <= 200: cands.update(p)
    
    # Street number channel
    if s1["street_num"]:
        p = idx["snum"].get((c,s1["street_num"]), [])
        if len(p) <= 60: cands.update(p)
    
    # Postal code channel
    if s1["postal"]:
        p = idx["postal"].get((c,s1["postal"]), [])
        if len(p) <= 50: cands.update(p)
    
    # N-gram channel
    for ng in s1["ng4"]:
        p = idx["ng4"].get((c,ng), [])
        if len(p) <= 40: cands.update(p)
    
    # Domain channel (no limit — very targeted)
    if s1["domain_stem"]:
        p = idx["domain"].get((c,s1["domain_stem"]), [])
        cands.update(p)
    
    return list(cands)[:max_k]


# ============================================================
# EVALUATION
# ============================================================
def evaluate(predictions: dict, val_gt: dict, val_s1_ids: list) -> dict:
    f05_list, p_list, r_list = [], [], []
    st = sc = fm = missed = 0
    for eid in val_s1_ids:
        pred_set = set(predictions.get(eid, []))
        true_set = set(val_gt.get(eid, []))
        if not true_set:
            st += 1
            if not pred_set: sc += 1
            else: fm += 1
        if not true_set and not pred_set:
            p_list.append(1.0); r_list.append(1.0)
        elif not true_set:
            p_list.append(0.0); r_list.append(1.0)
        elif not pred_set:
            p_list.append(1.0); r_list.append(0.0)
            missed += len(true_set)
        else:
            tp = len(pred_set & true_set)
            p_list.append(tp/len(pred_set)); r_list.append(tp/len(true_set))
            if tp < len(pred_set): fm += 1
            missed += len(true_set) - tp
        f05_list.append(entity_f05(pred_set, true_set))
    return {
        "macro_f05": float(np.mean(f05_list)),
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "singleton_accuracy": (sc/st) if st > 0 else 1.0,
        "false_merges": fm,
        "missed_matches": missed
    }


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("PHASE 9: ULTIMATE ENTITY RESOLUTION MATCHER")
    logger.info("=" * 70)
    
    # Load frozen validation split
    with open(FROZEN_SPLIT_PATH) as f:
        manifest = json.load(f)
    val_s1_ids = manifest["s1_entity_ids"]
    val_s1_set = set(val_s1_ids)
    logger.info("Validation split: %d S1 entities", len(val_s1_ids))

    # Load ground truth
    gt = {}
    with open(DATA_ROOT/"train"/"train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = p[1].split(",") if len(p) >= 2 and p[1].strip() else []
    val_gt = {eid: gt.get(eid, []) for eid in val_s1_set}

    # Collect all positive target IDs
    needed_pos = set()
    for e in val_s1_set:
        needed_pos.update(val_gt[e])
    logger.info("Need %d positive targets", len(needed_pos))

    # Load S1 validation records
    val_s1_records = []
    with open(DATA_ROOT/"train"/"train_source1.tsv", "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in val_s1_set:
                val_s1_records.append(preprocess_record(p[0], p[1], p[2], p[3]))
    logger.info("Loaded %d S1 val records", len(val_s1_records))

    # Load S2/S3 candidates (all positives + up to 250k distractors)
    logger.info("Loading S2/S3 candidates...")
    t0 = time.time()
    cands = {}
    dc = 0
    MAX_DIST = 250000
    for s_path in [DATA_ROOT/"train"/"train_source2.tsv", DATA_ROOT/"train"/"train_source3.tsv"]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) >= 4:
                    cid = p[0]
                    is_needed = cid in needed_pos
                    if is_needed or dc < MAX_DIST:
                        cands[cid] = preprocess_record(cid, p[1], p[2], p[3])
                        if not is_needed: dc += 1
    logger.info("Loaded %d candidates (%d positives + %d distractors) in %.2fs",
                len(cands), len(needed_pos), dc, time.time() - t0)

    # Token frequencies for rare-token channel
    token_freq = defaultdict(int)
    for c in cands.values():
        for t in c["name_tokens"]: token_freq[t] += 1

    # Build indexes
    t0 = time.time()
    idx = build_indexes(cands, token_freq)
    logger.info("Built indexes in %.2fs", time.time() - t0)

    # Score all pairs
    logger.info("Scoring all candidate pairs...")
    t0 = time.time()
    all_scored = {}  # s1_id -> [(cid, feat_dict), ...]
    retrieval_hits = 0
    retrieval_total = 0
    for i, s1 in enumerate(val_s1_records):
        s1_id = s1["id"]
        cand_list = retrieve(s1, idx, max_k=120)
        scored_pairs = []
        for cid in cand_list:
            cand = cands.get(cid)
            if cand:
                feat = score_pair(s1, cand)
                scored_pairs.append((cid, feat))
        all_scored[s1_id] = scored_pairs
        
        # Track retrieval recall
        true_set = set(val_gt.get(s1_id, []))
        if true_set:
            retrieval_total += len(true_set)
            retrieval_hits += len(true_set & set(cand_list))
        
        if (i+1) % 5000 == 0:
            logger.info("  Scored %d/%d...", i+1, len(val_s1_records))
    
    ret_recall = retrieval_hits / retrieval_total * 100 if retrieval_total else 0
    logger.info("Scoring done in %.2fs. Retrieval recall: %.2f%% (%d/%d)",
                time.time() - t0, ret_recall, retrieval_hits, retrieval_total)

    # Load current champion
    try:
        with open(REPORTS_DIR/"champion.json") as f:
            champion = json.load(f)
        champ_f05 = champion["macro_f05"]
    except:
        champ_f05 = 0.8387  # fallback
    logger.info("Current champion: F0.5=%.4f", champ_f05)

    # Grid search over (tau, high_conf, addr_floor)
    taus = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
    high_confs = [0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    addr_floors = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    total = len(taus) * len(high_confs) * len(addr_floors)
    logger.info("Grid search: %d x %d x %d = %d combos", len(taus), len(high_confs), len(addr_floors), total)

    passing = []
    all_results = []
    for tau in taus:
        for hc in high_confs:
            for af in addr_floors:
                preds = {}
                for s1_id, scored in all_scored.items():
                    matched = []
                    for cid, feat in scored:
                        if decide(feat, tau, hc, af):
                            matched.append(cid)
                    preds[s1_id] = matched
                m = evaluate(preds, val_gt, val_s1_ids)
                entry = (m["macro_f05"], tau, hc, af, m["singleton_accuracy"],
                         m["precision"], m["recall"], m["false_merges"], m["missed_matches"])
                all_results.append(entry)
                if m["singleton_accuracy"] >= 0.90:
                    passing.append(entry)

    all_results.sort(reverse=True)
    passing.sort(reverse=True)

    logger.info("=" * 70)
    logger.info("CHAMPION BASELINE: F0.5=%.4f", champ_f05)
    logger.info("QUALITY GATE: Singleton >= 90%%")
    logger.info("-" * 70)

    if passing:
        logger.info("✅ FOUND %d CONFIGS PASSING QUALITY GATE!", len(passing))
        logger.info("%-6s %-6s %-6s %-12s %-12s %-12s %-10s %-10s",
                    "Tau", "HiConf", "AddrF", "Macro F0.5", "Precision", "Singleton", "FalseM", "Missed")
        logger.info("-" * 80)
        for f05, tau, hc, af, sing, prec, rec, fm, missed in passing[:20]:
            marker = " ★" if f05 > champ_f05 else ""
            logger.info("%-6.2f %-6.2f %-6.2f %-12.4f %-12.4f %-12.2f%% %-10d %-10d%s",
                        tau, hc, af, f05, prec, sing*100, fm, missed, marker)

        best_f05, best_tau, best_hc, best_af, best_sing, best_prec, best_rec, best_fm, best_missed = passing[0]
        logger.info("=" * 70)
        logger.info("🏆 BEST CONFIG: tau=%.2f | high_conf=%.2f | addr_floor=%.2f", best_tau, best_hc, best_af)
        logger.info("   Macro F0.5:         %.4f  (delta: %+.4f vs champion)", best_f05, best_f05 - champ_f05)
        logger.info("   Precision:          %.4f (%.2f%%)", best_prec, best_prec*100)
        logger.info("   Recall:             %.4f (%.2f%%)", best_rec, best_rec*100)
        logger.info("   Singleton Accuracy: %.4f (%.2f%%)", best_sing, best_sing*100)
        logger.info("   False Merges:       %d", best_fm)
        logger.info("   Missed Matches:     %d", best_missed)
        logger.info("   Retrieval Recall:   %.2f%%", ret_recall)

        result = {
            "run_id": "phase9-ultimate-v1",
            "model": "Phase 9 Ultimate: Rich Features + Rule Paths + Multi-Channel Retrieval",
            "tau": best_tau, "high_confidence": best_hc, "addr_floor": best_af,
            "macro_f05": round(best_f05, 4), "precision": round(best_prec, 4),
            "recall": round(best_rec, 4), "singleton_accuracy": round(best_sing, 4),
            "false_merges": best_fm, "missed_matches": best_missed,
            "retrieval_recall": round(ret_recall, 2),
            "champion_f05": champ_f05,
            "delta_vs_champion": round(best_f05 - champ_f05, 4),
            "quality_gate_criteria_met": best_sing >= 0.90,
            "dataset_hash": manifest.get("dataset_hash", ""),
            "validation_protocol": manifest.get("split_version", "frozen-20k")
        }
        out = REPORTS_DIR / "phase9_best_challenger.json"
        with open(out, "w") as f: json.dump(result, f, indent=2)
        logger.info("Saved to %s", out)

        if best_f05 > champ_f05:
            logger.info("🎉 NEW CHAMPION! Promoting phase9...")
            champion_out = {
                **result,
                "status": "CHAMPION_ACTIVE",
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            with open(REPORTS_DIR / "champion.json", "w") as f:
                json.dump(champion_out, f, indent=2)
    else:
        logger.warning("❌ No config satisfied quality gate.")
        logger.info("Top 15 configs by F0.5:")
        for f05, tau, hc, af, sing, prec, rec, fm, missed in all_results[:15]:
            logger.info("  tau=%.2f hc=%.2f af=%.2f → F0.5=%.4f Sing=%.2f%% FM=%d",
                        tau, hc, af, f05, sing*100, fm)


if __name__ == "__main__":
    main()
