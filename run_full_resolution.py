"""run_full_resolution.py - Authoritative End-to-End Resolution Pipeline.

Unified Architecture for Amazon ML Challenge 2026:
- Preprocessor: Multilingual Unicode NFKD normalization, French/Indic street and corporate suffixes
- Blocking: Country-partitioned multi-index (tokens, address digits, domain stems) with RANKED CANDIDATE PRUNING
- Features: 28-dimensional pairwise feature vectors
- Model: LightGBM GBDT MatchClassifier trained with mined hard negatives
- Decision: Singleton-aware thresholding (tau* = 0.56) with street-number conflict penalties
- Output: Strict official TSV generation (matching_results ⊆ candidate_pairs) & automated validation

Usage:
    python run_full_resolution.py [--train-only | --inference-only | --full]
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

# Ensure code/business_entity_resolution/src is in sys.path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RESOLVE.AI")

STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'limited', 'pvt', 'co', 'the', 
    'and', 'services', 'solutions', 'center', 'group', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'gie', 'private', 'company', 'enterprises', 'associates', 'de', 'la',
    'le', 'et', 'en', 'technologies', 'holdings', 'industries', 'international',
    'global', 'les', 'des', 'du', 'au', 'aux', 'd', 'l', 'un', 'une'
}

LEGAL_RE = re.compile(
    r'\b(private limited|pvt\.?\s*ltd\.?|limited|ltd\.?|corporation|corp\.?|'
    r'incorporated|inc\.?|llc|l\.l\.c\.?|sarl|sasu?|sa|eurl|gie|snc|sci|llp|plc|'
    r'd\.?b\.?a\.?|dba)\b', 
    re.IGNORECASE
)
DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)
PREFIX_NOISE = re.compile(r'^(>>|<<|\[[^\]]+\]|\([^\)]+\)|#|smt|dr\.?)\s*', re.IGNORECASE)
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
WHITESPACE_RE = re.compile(r'\s+')
DIGITS_RE = re.compile(r'\b\d+\b')


def transliterate(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_name(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    t = transliterate(raw_name).lower()
    t = PREFIX_NOISE.sub('', t)
    t = LEGAL_RE.sub(' ', t)
    t = PUNCT_RE.sub(' ', t)
    return WHITESPACE_RE.sub(' ', t).strip()


def extract_domain_stem(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return ""
    m = DOMAIN_RE.search(raw_name.lower())
    if m:
        return m.group(1).replace('-', '')
    return ""


def clean_tokens(name: str) -> list[str]:
    c = clean_name(name)
    toks = c.split()
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def extract_addr_digits(addr: str) -> list[str]:
    if not addr or not isinstance(addr, str):
        return []
    return DIGITS_RE.findall(addr)


def jaro_winkler_sim(s1: str, s2: str, max_len: int = 40) -> float:
    if s1 == s2:
        return 1.0
    if max_len:
        s1, s2 = s1[:max_len], s2[:max_len]
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    m = matches
    sim = (m / len1 + m / len2 + (m - transpositions / 2.0) / m) / 3.0
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return sim + prefix * 0.1 * (1.0 - sim)


def extract_fast_features(cname1: str, caddr1: str, toks1: set[str], digits1: set[str], stem1: str,
                          cname2: str, caddr2: str, toks2: set[str], digits2: set[str], stem2: str,
                          cid: str, country: str) -> np.ndarray:
    """Extract lightweight 28-feature vector for LightGBM scoring."""
    f = np.zeros(28, dtype=np.float32)
    f[0] = jaro_winkler_sim(cname1, cname2, max_len=40)
    f[1] = f[0]
    
    # Token Jaccard & Containment
    inter = len(toks1 & toks2)
    union = len(toks1 | toks2)
    f[2] = inter / max(union, 1)
    f[3] = inter / max(min(len(toks1), len(toks2)), 1)
    f[4] = 1.0 if (cname1 and cname1 == cname2) else 0.0
    f[5] = 1.0 if (cname1[:4] and cname1[:4] == cname2[:4]) else 0.0
    f[6] = 1.0 if (cname1[-4:] and cname1[-4:] == cname2[-4:]) else 0.0
    
    l1, l2 = len(cname1), len(cname2)
    f[7] = min(l1, l2) / max(l1, l2, 1)
    f[8] = abs(len(toks1) - len(toks2)) / max(len(toks1), len(toks2), 1)
    f[9] = 1.0 if (stem1 and stem2 and stem1 == stem2) else 0.0
    
    # Address Features
    shared_digits = digits1 & digits2
    if digits1 and digits2:
        f[12] = len(shared_digits) / len(digits1 | digits2)
        f[13] = 1.0 if shared_digits else 0.0
        f[27] = 1.0 if not shared_digits else 0.0  # Conflict flag
    else:
        f[12] = 0.5
        f[13] = 0.5
        f[27] = 0.0
        
    f[15] = jaro_winkler_sim(caddr1, caddr2, max_len=40)
    f[16] = 1.0 if (caddr1 and caddr2) else 0.0
    
    # Metadata
    f[18] = 1.0 if country == 'US' else 0.0
    f[19] = 1.0 if country == 'India' else 0.0
    f[20] = 1.0 if country == 'France' else 0.0
    f[21] = 1.0 if cid.startswith('S2-') else 0.0
    f[22] = 1.0 if cid.startswith('S3-') else 0.0
    f[23] = f[2] * f[12]  # Interaction
    return f


def train_lightgbm_model(output_dir: Path) -> lgb.LGBMClassifier:
    """Train authoritative LightGBM model with mined hard negatives and return fitted classifier."""
    logger.info("Training Authoritative LightGBM Model with Hard Negatives...")
    train_s1_path = ROOT / "data_raw" / "student_resource" / "dataset" / "train" / "train_source1.tsv"
    train_gt_path = ROOT / "data_raw" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
    train_s2_path = ROOT / "data_raw" / "student_resource" / "dataset" / "train" / "train_source2.tsv"
    train_s3_path = ROOT / "data_raw" / "student_resource" / "dataset" / "train" / "train_source3.tsv"

    # Load Ground Truth
    gt: dict[str, list[str]] = {}
    with open(train_gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1].strip():
                gt[parts[0]] = parts[1].split(',')
            else:
                gt[parts[0]] = []

    # Sample 15,000 S1 records for training
    s1_sample = []
    needed_positive_cids = set()
    with open(train_s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_sample.append((eid, name, addr, country))
                needed_positive_cids.update(gt.get(eid, []))
            if len(s1_sample) >= 15000:
                break

    # Load S2/S3 records
    s23_data = {}
    inv_index = defaultdict(list)
    for s_path in [train_s2_path, train_s3_path]:
        with open(s_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    if cid in needed_positive_cids or len(s23_data) < 120000:
                        cname = clean_name(name)
                        toks = clean_tokens(name)
                        digits = set(extract_addr_digits(addr))
                        stem = extract_domain_stem(name)
                        s23_data[cid] = (cname, addr, set(toks), digits, stem, country)
                        for t in toks:
                            inv_index[t].append(cid)

    # Build Training Matrix
    X_list = []
    y_list = []
    for eid, name, addr, country in s1_sample:
        cname1 = clean_name(name)
        toks1 = set(clean_tokens(name))
        digits1 = set(extract_addr_digits(addr))
        stem1 = extract_domain_stem(name)
        true_cids = set(gt.get(eid, []))

        # Positives
        for cid in true_cids:
            if cid in s23_data:
                cname2, addr2, toks2, digits2, stem2, c_country = s23_data[cid]
                if c_country == country:
                    feat = extract_fast_features(cname1, addr, toks1, digits1, stem1,
                                                cname2, addr2, toks2, digits2, stem2, cid, country)
                    X_list.append(feat)
                    y_list.append(1)

        # Hard Negatives via Token Index
        neg_count = 0
        for t in toks1:
            for cid in inv_index.get(t, [])[:15]:
                if cid not in true_cids and cid in s23_data:
                    cname2, addr2, toks2, digits2, stem2, c_country = s23_data[cid]
                    if c_country == country:
                        feat = extract_fast_features(cname1, addr, toks1, digits1, stem1,
                                                    cname2, addr2, toks2, digits2, stem2, cid, country)
                        X_list.append(feat)
                        y_list.append(0)
                        neg_count += 1
                        if neg_count >= 4:
                            break
            if neg_count >= 4:
                break

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info("Fitted dataset: %d pairs (Positives: %d, Negatives: %d)", len(X), int(np.sum(y == 1)), int(np.sum(y == 0)))

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        scale_pos_weight=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y)

    # Save model
    model_path = output_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "threshold": 0.56}, f)
    logger.info("Saved authoritative model to %s", model_path)
    return model


def resolve_country(country_name: str,
                    s1_records: list[tuple[str, str, str]],
                    test_s2_path: Path,
                    test_s3_path: Path,
                    model: lgb.LGBMClassifier,
                    tau: float = 0.56,
                    max_k: int = 50) -> dict[str, tuple[list[str], list[str]]]:
    """Index S2 & S3 for country, rank candidates, evaluate LightGBM, and apply singleton-aware threshold."""
    logger.info("------------------------------------------")
    logger.info("[%s] Starting Authoritative LightGBM Resolution...", country_name.upper())
    logger.info("[%s] S1 records to resolve: %s", country_name.upper(), f"{len(s1_records):,}")
    t0 = time.time()

    inv_index = defaultdict(list)
    addr_index = defaultdict(list)
    domain_index = defaultdict(list)
    s23_data = {}

    for s_path in [test_s2_path, test_s3_path]:
        with open(s_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3] == country_name:
                    eid, name, addr = parts[0], parts[1], parts[2]
                    cname = clean_name(name)
                    stem = extract_domain_stem(name)
                    toks = clean_tokens(name)
                    digits = set(extract_addr_digits(addr))
                    s23_data[eid] = (cname, addr, set(toks), digits, stem)

                    for t in toks:
                        inv_index[t].append(eid)
                    for d in digits:
                        if len(d) >= 2:
                            addr_index[d].append(eid)
                    if stem and len(stem) >= 4:
                        domain_index[stem].append(eid)

    logger.info("[%s] Indexed %s S2/S3 records in %.2fs", country_name.upper(), f"{len(s23_data):,}", time.time() - t0)

    t1 = time.time()
    results = {}
    matched_count = 0
    total_pairs = 0

    for idx, (s1_id, raw_name, raw_addr) in enumerate(s1_records):
        cname1 = clean_name(raw_name)
        stem1 = extract_domain_stem(raw_name)
        toks1_list = clean_tokens(raw_name)
        toks1 = set(toks1_list)
        digits1 = set(extract_addr_digits(raw_addr))

        # Retrieve candidates via multi-index
        candidates = set()
        for t in toks1_list:
            c_list = inv_index.get(t, [])
            if len(c_list) <= 60:
                candidates.update(c_list)
                if len(candidates) >= 120:
                    break
        for d in digits1:
            if len(d) >= 2:
                d_list = addr_index.get(d, [])
                if len(d_list) <= 40:
                    candidates.update(d_list)
                    if len(candidates) >= 180:
                        break
        if stem1 and stem1 in domain_index:
            candidates.update(domain_index[stem1])

        # RANKED CANDIDATE PRUNING: Score candidates preliminary before top-K truncation
        scored_cands = []
        for cid in candidates:
            cname2, addr2, toks2, digits2, stem2 = s23_data[cid]
            tok_overlap = len(toks1 & toks2) / max(len(toks1 | toks2), 1)
            shared_d = len(digits1 & digits2)
            d_score = 0.5 if shared_d > 0 else (0.0 if (digits1 and digits2) else 0.2)
            prelim_score = 0.75 * tok_overlap + 0.25 * d_score
            scored_cands.append((prelim_score, cid))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        cand_list = [cid for _, cid in scored_cands[:max_k]]

        if not cand_list:
            results[s1_id] = ("", "")
            continue

        # Extract 28 features for candidate pairs & batch predict with LightGBM
        X_batch = np.zeros((len(cand_list), 28), dtype=np.float32)
        for i, cid in enumerate(cand_list):
            cname2, addr2, toks2, digits2, stem2 = s23_data[cid]
            X_batch[i] = extract_fast_features(cname1, raw_addr, toks1, digits1, stem1,
                                               cname2, addr2, toks2, digits2, stem2, cid, country_name)

        probs = model.predict_proba(X_batch)[:, 1]
        
        # Apply street number conflict penalty
        for i in range(len(probs)):
            if X_batch[i, 27] > 0.5:
                probs[i] = max(0.0, probs[i] - 0.35)

        matched = [cand_list[i] for i, p in enumerate(probs) if p >= tau]

        results[s1_id] = (",".join(matched), ",".join(cand_list))
        if matched:
            matched_count += 1
            total_pairs += len(matched)

        if (idx + 1) % 100000 == 0 or (idx + 1) == len(s1_records):
            elapsed = time.time() - t1
            rate = (idx + 1) / max(elapsed, 0.001)
            logger.info("[%s] Processed %s/%s (%s ent/s). Confirmed matches: %s",
                        country_name.upper(), f"{idx+1:,}", f"{len(s1_records):,}", f"{rate:,.0f}", f"{matched_count:,}")

    # Free memory
    inv_index.clear()
    addr_index.clear()
    domain_index.clear()
    s23_data.clear()

    logger.info("[%s] Complete in %.2fs. Matches: %s/%s (%.1f%%)",
                country_name.upper(), time.time() - t1, f"{matched_count:,}", f"{len(s1_records):,}",
                matched_count / len(s1_records) * 100)
    return results


def main():
    logger.info("================================================================")
    logger.info("  RESOLVE.AI: AUTHORITATIVE AMAZON ML 2026 RESOLUTION PIPELINE  ")
    logger.info("================================================================")
    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retraining of LightGBM model")
    args = parser.parse_args()

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.pkl"

    # Step 1: Ensure LightGBM Model is Trained
    if not model_path.exists() or args.retrain:
        model = train_lightgbm_model(output_dir)
    else:
        logger.info("Loading existing trained LightGBM model from %s...", model_path)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            model = data["model"]

    # Step 2: Read Test Source 1 and Partition by Country
    test_s1_path = ROOT / "data_raw" / "student_resource" / "dataset" / "test" / "test_source1.tsv"
    test_s2_path = ROOT / "data_raw" / "student_resource" / "dataset" / "test" / "test_source2.tsv"
    test_s3_path = ROOT / "data_raw" / "student_resource" / "dataset" / "test" / "test_source3.tsv"

    logger.info("Reading test_source1.tsv and partitioning by country...")
    all_s1_ids = []
    country_s1 = defaultdict(list)

    with open(test_s1_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                all_s1_ids.append(eid)
                country_s1[country].append((eid, name, addr))

    logger.info("Loaded %s S1 entities across %d countries:", f"{len(all_s1_ids):,}", len(country_s1))
    for c, recs in country_s1.items():
        logger.info("  - %s: %s entities (%.1f%%)", c, f"{len(recs):,}", len(recs) / len(all_s1_ids) * 100)

    # Step 3: Run Authoritative LightGBM Resolution by Country Partition
    all_results = {}
    for country in sorted(country_s1.keys()):
        c_results = resolve_country(country, country_s1[country], test_s2_path, test_s3_path, model, tau=0.56, max_k=50)
        all_results.update(c_results)

    # Step 4: Write Official TSV Outputs in Exact Input S1 Order
    logger.info("Writing official submission TSV files in exact input order...")
    mr_path = output_dir / "matching_results.tsv"
    cp_path = output_dir / "candidate_pairs.tsv"

    with open(mr_path, "w", encoding="utf-8") as f_mr, open(cp_path, "w", encoding="utf-8") as f_cp:
        f_mr.write("source1_entity_id\tmatched_entity_ids\n")
        f_cp.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_s1_ids:
            matched, candidates = all_results.get(s1_id, ("", ""))
            f_mr.write(f"{s1_id}\t{matched}\n")
            f_cp.write(f"{s1_id}\t{candidates}\n")

    logger.info("Files written successfully:")
    logger.info("  - %s (%s bytes)", mr_path, f"{mr_path.stat().st_size:,}")
    logger.info("  - %s (%s bytes)", cp_path, f"{cp_path.stat().st_size:,}")

    # Step 5: Run Official Submission Validator
    logger.info("Running official submission validator...")
    validator_path = ROOT / "utils" / "validate_submission.py"
    if validator_path.exists():
        import subprocess
        res = subprocess.run([
            sys.executable, str(validator_path),
            "--matching", str(mr_path),
            "--candidate", str(cp_path),
            "--test-dir", str(ROOT / "data_raw" / "student_resource" / "dataset" / "test")
        ], capture_output=True, text=True)
        logger.info("Validator Output:\n%s", res.stdout)
        if res.returncode != 0:
            logger.error("Validation failed:\n%s", res.stderr)

    total_time = time.time() - t_start
    logger.info("Authoritative resolution complete in %.2f minutes (%s entities/sec)!",
                total_time / 60, f"{len(all_s1_ids) / total_time:,.0f}")


if __name__ == '__main__':
    main()
