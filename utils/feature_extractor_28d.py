"""
utils/feature_extractor_28d.py - Authoritative 28-Dimensional Pairwise Feature Extractor.
Amazon ML Challenge 2026.

Guarantees 100% feature consistency across:
- Training Pipeline
- Diagnostic & Validation Benchmarks
- Full Test-Set Inference

Features (Exactly 28):
 0: levenshtein_name         - Normalized Levenshtein similarity of cleaned business names [0, 1]
 1: jaro_winkler_name        - Jaro-Winkler similarity of business names [0, 1]
 2: token_jaccard_name       - Jaccard similarity of name token sets [0, 1]
 3: token_containment_name   - Token containment ratio |T1 & T2| / min(|T1|, |T2|) [0, 1]
 4: exact_name_match         - Binary indicator of exact cleaned name match {0, 1}
 5: prefix_4_name            - Binary indicator of 4-character prefix match {0, 1}
 6: suffix_4_name            - Binary indicator of 4-character suffix match {0, 1}
 7: char_len_ratio_name      - Character length ratio min(L1, L2) / max(L1, L2) [0, 1]
 8: token_count_diff_name    - Inverse token count difference 1 / (1 + |len1 - len2|) [0, 1]
 9: legal_suffix_match       - Binary indicator of shared legal entity suffix {0, 1}
10: rare_token_shared        - Binary indicator of shared non-stopword tokens {0, 1}
11: token_jaccard_address    - Jaccard similarity of expanded address token sets [0, 1]
12: digit_jaccard_address    - Jaccard similarity of numeric digit sets in address [0, 1]
13: street_number_match      - Street number agreement score (1.0 match, 0.0 conflict, 0.5 missing)
14: postal_code_match        - Postal code agreement score (1.0 match, 0.0 conflict, 0.5 missing)
15: levenshtein_address      - Normalized Levenshtein similarity of cleaned addresses [0, 1]
16: both_have_address        - Binary indicator that both records have non-empty address {0, 1}
17: address_len_ratio        - Address character length ratio [0, 1]
18: country_is_us            - One-hot country indicator for US {0, 1}
19: country_is_india         - One-hot country indicator for India {0, 1}
20: country_is_france        - One-hot country indicator for France {0, 1}
21: source_is_s2             - One-hot candidate source indicator for S2 {0, 1}
22: source_is_s3             - One-hot candidate source indicator for S3 {0, 1}
23: name_x_addr_overlap      - Multiplicative interaction: token_jaccard_name * token_jaccard_address [0, 1]
24: min_token_len_name       - Minimum token count between the two business names >= 0
25: first_token_match        - Binary indicator of first token identity {0, 1}
26: last_token_match         - Binary indicator of last token identity {0, 1}
27: numeric_conflict_penalty - Numeric conflict penalty (1.0 hard conflict, 0.2 partial, 0.0 none)
"""
from __future__ import annotations
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    "levenshtein_name",
    "jaro_winkler_name",
    "token_jaccard_name",
    "token_containment_name",
    "exact_name_match",
    "prefix_4_name",
    "suffix_4_name",
    "char_len_ratio_name",
    "token_count_diff_name",
    "legal_suffix_match",
    "rare_token_shared",
    "token_jaccard_address",
    "digit_jaccard_address",
    "street_number_match",
    "postal_code_match",
    "levenshtein_address",
    "both_have_address",
    "address_len_ratio",
    "country_is_us",
    "country_is_india",
    "country_is_france",
    "source_is_s2",
    "source_is_s3",
    "name_x_addr_overlap",
    "min_token_len_name",
    "first_token_match",
    "last_token_match",
    "numeric_conflict_penalty"
]

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 28, f"Feature count must be exactly 28, found {NUM_FEATURES}"

# Precompiled regex patterns
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
    # India specific
    'opp': 'opposite', 'nr': 'near', 'soc': 'society', 'col': 'colony',
    'ext': 'extension', 'sec': 'sector', 'dist': 'district', 'vill': 'village',
    'po': 'postoffice', 'ps': 'policestation', 'mkt': 'market',
    'cmplx': 'complex', 'twr': 'tower'
}

STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}


def normalize_unicode(text: str) -> str:
    """Multilingual NFKD transliteration removing non-ASCII combining marks."""
    if not text or not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(raw_text: str) -> str:
    """Standardized lowercasing, acronym dot collapse, and punctuation stripping."""
    if not raw_text or not isinstance(raw_text, str):
        return ""
    t = normalize_unicode(raw_text).lower()
    # Collapse acronym dots: e.g. "l.l.c." -> "llc", "p.v.t." -> "pvt"
    t = re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b)', '', t)
    t = RE_PUNCT.sub(' ', t)
    return ' '.join(t.split())


def extract_legal_suffix(clean_name: str) -> str:
    """Extract legal suffix if present at end of name."""
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if clean_name.endswith(' ' + suf) or clean_name == suf:
            return suf
    return ""


def clean_name_tokens(clean_name_str: str) -> Set[str]:
    """Token set excluding common legal stopwords."""
    toks = clean_name_str.split()
    return {t for t in toks if t not in STOPWORDS and len(t) >= 2}


def expand_addr_tokens(addr_clean: str) -> Set[str]:
    """Expands address abbreviations to standard canonical forms."""
    if not addr_clean:
        return set()
    tokens = set()
    for t in addr_clean.split():
        exp = ADDR_ABBREVIATIONS.get(t, t)
        tokens.add(exp)
        if exp != t:
            tokens.add(t)
    return tokens


def extract_street_number(addr_clean: str) -> str:
    """Extracts first street number or house digit from address string."""
    if not addr_clean:
        return ""
    # Find all digits in address, pick the first 1-5 digit sequence
    for d in RE_DIGITS.findall(addr_clean):
        if 1 <= len(d) <= 5:
            return d
    return ""


def extract_postal_code(addr_clean: str) -> str:
    """Extracts 5 or 6 digit postal code."""
    if not addr_clean:
        return ""
    for d in RE_DIGITS.findall(addr_clean):
        if len(d) in (5, 6):
            return d
    return ""


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Computes Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0

    match_dist = max(l1, l2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * l1
    s2_matches = [False] * l2
    matches = 0
    transpositions = 0

    for i in range(l1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, l2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(l1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    sim = (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3.0

    # Prefix scale
    prefix = 0
    for i in range(min(4, min(l1, l2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return min(1.0, sim + prefix * p * (1.0 - sim))


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity [0, 1] with fast early exits."""
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    # Length filter: if length difference > 40%, similarity cannot exceed 0.6
    max_len = max(l1, l2)
    if abs(l1 - l2) / max_len > 0.40:
        return 0.0
    if l1 > 100 or l2 > 100:
        s1, s2 = s1[:100], s2[:100]
        l1, l2 = len(s1), len(s2)

    prev_row = list(range(l2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] * (l2 + 1)
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row

    dist = prev_row[l2]
    return max(0.0, 1.0 - (dist / max(l1, l2)))


class Unified28FeatureExtractor:
    """
    Authoritative single-source feature extractor for the Amazon ML entity resolution pipeline.
    Ensures exact numeric guarantees: no NaN, no Inf, bounded ranges.
    """

    @staticmethod
    def extract_pair(s1_dict: Dict[str, Any], cand_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extracts exactly 28 numeric features for a single (s1, cand) pair.
        Returns np.ndarray of shape (28,), dtype np.float32.
        """
        f = np.zeros(NUM_FEATURES, dtype=np.float32)

        # Names
        n1 = s1_dict.get('clean_name', '')
        n2 = cand_dict.get('clean_name', '')
        toks1 = s1_dict.get('name_tokens', set())
        toks2 = cand_dict.get('name_tokens', set())
        if isinstance(toks1, (list, tuple)):
            toks1 = set(toks1)
        if isinstance(toks2, (list, tuple)):
            toks2 = set(toks2)

        # Addresses
        a1 = s1_dict.get('clean_addr', '')
        a2 = cand_dict.get('clean_addr', '')
        addr_toks1 = s1_dict.get('expanded_addr', set()) or expand_addr_tokens(a1)
        addr_toks2 = cand_dict.get('expanded_addr', set()) or expand_addr_tokens(a2)
        if isinstance(addr_toks1, (list, tuple)):
            addr_toks1 = set(addr_toks1)
        if isinstance(addr_toks2, (list, tuple)):
            addr_toks2 = set(addr_toks2)

        digits1 = set(RE_DIGITS.findall(a1))
        digits2 = set(RE_DIGITS.findall(a2))

        # 0. Levenshtein name
        f[0] = levenshtein_similarity(n1, n2)

        # 1. Jaro-Winkler name
        f[1] = jaro_winkler(n1, n2)

        # 2. Token Jaccard name
        f[2] = jaccard_similarity(toks1, toks2)

        # 3. Token containment name
        if toks1 and toks2:
            min_len = min(len(toks1), len(toks2))
            f[3] = (len(toks1 & toks2) / min_len) if min_len > 0 else 0.0

        # 4. Exact name match
        f[4] = 1.0 if (n1 and n1 == n2) else 0.0

        # 5. Prefix 4 name
        f[5] = 1.0 if (len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4]) else 0.0

        # 6. Suffix 4 name
        f[6] = 1.0 if (len(n1) >= 4 and len(n2) >= 4 and n1[-4:] == n2[-4:]) else 0.0

        # 7. Char len ratio name
        l1, l2 = len(n1), len(n2)
        f[7] = (min(l1, l2) / max(l1, l2, 1)) if (l1 and l2) else 0.0

        # 8. Token count diff name
        f[8] = 1.0 / (1.0 + abs(len(toks1) - len(toks2)))

        # 9. Legal suffix match
        suf1 = s1_dict.get('legal_suffix') or extract_legal_suffix(n1)
        suf2 = cand_dict.get('legal_suffix') or extract_legal_suffix(n2)
        f[9] = 1.0 if (suf1 and suf1 == suf2) else 0.0

        # 10. Rare token shared
        f[10] = 1.0 if (toks1 & toks2) else 0.0

        # 11. Token Jaccard address
        f[11] = jaccard_similarity(addr_toks1, addr_toks2)

        # 12. Digit Jaccard address
        f[12] = jaccard_similarity(digits1, digits2)

        # 13 & 27: Street number match & numeric conflict penalty
        sn1 = s1_dict.get('street_num') or extract_street_number(a1)
        sn2 = cand_dict.get('street_num') or extract_street_number(a2)
        if sn1 and sn2:
            if sn1 == sn2:
                f[13] = 1.0
                f[27] = 0.0
            else:
                f[13] = 0.0
                f[27] = 1.0  # Hard conflict
        elif sn1 or sn2:
            f[13] = 0.5
            f[27] = 0.2  # Asymmetric partial penalty
        else:
            f[13] = 0.5
            f[27] = 0.0

        # 14. Postal code match
        po1 = s1_dict.get('postal') or extract_postal_code(a1)
        po2 = cand_dict.get('postal') or extract_postal_code(a2)
        if po1 and po2:
            f[14] = 1.0 if po1 == po2 else 0.0
        else:
            f[14] = 0.5

        # 15. Levenshtein address (fast path: require at least one shared token)
        f[15] = levenshtein_similarity(a1, a2) if (a1 and a2 and (addr_toks1 & addr_toks2)) else 0.0

        # 16. Both have address
        f[16] = 1.0 if (a1 and a2) else 0.0

        # 17. Address len ratio
        la1, la2 = len(a1), len(a2)
        f[17] = (min(la1, la2) / max(la1, la2, 1)) if (la1 and la2) else 0.0

        # 18-20: Country one-hot
        c = s1_dict.get('country', '') or cand_dict.get('country', '')
        f[18] = 1.0 if c == 'US' else 0.0
        f[19] = 1.0 if c == 'India' else 0.0
        f[20] = 1.0 if c == 'France' else 0.0

        # 21-22: Source one-hot
        cid = str(cand_dict.get('id', ''))
        f[21] = 1.0 if cid.startswith('S2-') else 0.0
        f[22] = 1.0 if cid.startswith('S3-') else 0.0

        # 23. Name x Address interaction
        f[23] = f[2] * f[11]

        # 24-26: Token position and length
        w1 = n1.split()
        w2 = n2.split()
        f[24] = float(min(len(w1), len(w2))) if (w1 and w2) else 0.0
        f[25] = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0
        f[26] = 1.0 if (w1 and w2 and w1[-1] == w2[-1]) else 0.0

        # Sanity check: ensure zero NaN / Inf
        if np.isnan(f).any() or np.isinf(f).any():
            np.nan_to_num(f, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

        return f

    @classmethod
    def extract_batch_dataframe(cls, pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> pd.DataFrame:
        """
        Extracts 28 features for a batch of pairs and returns a clean pandas DataFrame
        with exact FEATURE_NAMES column headers.
        """
        n = len(pairs)
        matrix = np.empty((n, NUM_FEATURES), dtype=np.float32)
        for i, (s1, cand) in enumerate(pairs):
            matrix[i] = cls.extract_pair(s1, cand)
        return pd.DataFrame(matrix, columns=FEATURE_NAMES)
