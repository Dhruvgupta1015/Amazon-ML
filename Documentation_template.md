# Documentation Template — Amazon ML Challenge 2026
# Business Entity Resolution

## Team Information

- **Team Name**: Resolve AI Team
- **Members**: Vivek Gupta
- **Submission Date**: September 25, 2026

---

## 1. Executive Summary

We built a multi-stage entity resolution pipeline combining high-recall blocking (MinHash LSH + address digit filtering + dense ANN retrieval with sentence-transformers) with a LightGBM pairwise classifier trained on 28 discriminative features. The pipeline is optimized for **Macro F₀.₅** via entity-level GroupKFold cross-validation and a precision-weighted threshold search.

---

## 2. Methodology

### 2.1 Preprocessing & Normalization

**Approach**: Fully open-set, country-agnostic normalization using Python's `unicodedata` library for universal diacritic stripping (covering French accents é/ê/ç/à, Indian transliteration variants, and US standard forms).

**Key transformations**:
- Corporate suffix normalization: `private limited → pvt ltd`, `société anonyme → sa`, `gmbh`, `sarl`, etc.
- Address abbreviation expansion: `st → street`, `rd → road`, `ave → avenue`, `bd/bvd → boulevard`
- Ampersand normalization: `& → and`
- Numeric signature extraction: US ZIP (5-digit), India PIN (6-digit), France CP (5-digit)
- Character 3-gram extraction for MinHash

### 2.2 Candidate Generation / Blocking Strategy

We apply three complementary blocking stages, all partitioned by country (no cross-country pairs generated):

| Stage | Method | Parameters | Rationale |
|-------|--------|-----------|-----------|
| Hard filter | Country partition | Exact string match | Businesses never cross country boundaries |
| Stage 1 | MinHash LSH | 128 permutations, threshold=0.25 | Covers typos and abbreviation variants |
| Stage 2 | Address digit blocking | ≥3-digit shared tokens | Catches co-located businesses, same ZIP |
| Stage 3 | Dense ANN (sentence-transformers) | top-30 cosine, MiniLM-L6-v2 | Handles semantic variations, transliterations |

**Candidate consolidation**: Union of all three stages, capped at 50 candidates per S1 entity.

**Estimated blocking recall**: ≥98% on training data validation.
**Reduction ratio**: >99.5% (verified on train split).

### 2.3 Feature Engineering

We compute 28 features for every (S1, S2/S3) candidate pair:

**Name similarity (7 features)**:
- Levenshtein edit distance ratio
- Jaro-Winkler similarity (prefix-aware)
- Monge-Elkan similarity (token alignment)
- Token Sort Ratio (word-order invariant)
- Token Set Ratio (set intersection comparison)
- LCS ratio (longest common subsequence)
- TF-IDF character n-gram cosine similarity

**Address similarity (3 features)**:
- Token Jaccard similarity on address tokens
- Digit Jaccard: postal code / street number exact match
- Postal code exact match (5-6 digit patterns)

**Phonetic & structural (5 features)**:
- Double Metaphone code match (English phonetic encoding)
- Word count difference ratio
- Prefix match (first 4 characters)
- Token count difference
- Character length ratio

**Cross-source prior (2 features)**:
- Is Source 2 flag
- Is Source 3 flag

**Embedding cosine similarity (2 features)**:
- Name embedding cosine (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- Address embedding cosine

**Heuristics (9 features)**:
- Suffix match (e.g., both end in `pvt ltd`)
- Both records have non-empty addresses
- Shared rare name token (>3 characters)
- Token sort/set ratios on addresses
- Levenshtein on address strings
- Address has numeric tokens flag
- Name exact match flag

### 2.4 Model Architecture

**Algorithm**: LightGBM Binary Classifier (`binary_logloss` objective)

**Configuration**:
```
num_leaves: 63
learning_rate: 0.05
n_estimators: 500 (with early stopping, patience=50)
min_child_samples: 5
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 3.0 (class imbalance correction)
```

**Training data construction**:
- Positive pairs: True matches from `train_ground_truth.tsv`
- Hard negatives: Non-matching candidates from blocking stage (5× per positive)
- GroupKFold (5 folds, grouped by S1 entity to prevent train/test entity leakage)

### 2.5 Threshold Optimization

We perform entity-level threshold search over τ ∈ [0.50, 0.98] with step 0.01:

```
For each τ:
  For each S1 entity:
    matched = {cand | P(match) ≥ τ}
    F₀.₅(entity) = entity_f05(matched, ground_truth)
  macro_F₀.₅(τ) = mean(F₀.₅ across all S1 entities)

τ* = argmax macro_F₀.₅(τ)
```

**Singleton guard**: If `max P(match for S1) < τ*`, predict empty match list (score = 1.0 for true singletons).

---

## 3. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Unseen country (France) in test set | Open-set preprocessing: unicodedata NFKD stripping covers all Latin-script languages |
| Multilingual business names | Transliteration + suffix normalization table covering French (sa/sas/sarl), German (gmbh/ag), Indian (pvt ltd) |
| Class imbalance (few positives per S1) | Hard negative mining from blocking stage + `scale_pos_weight=3.0` |
| Singleton penalty (false merges = 0.0) | Precision-weighted threshold search + singleton guard |
| Cross-country false positives | Hard country partition in all blocking stages |

---

## 4. Validation
 
Local validation F₀.₅ (frozen 20,000 entity held-out partition):
- Macro F₀.₅: 0.8539
- Precision: 0.9793
- Recall: 0.7588
- Singleton Accuracy: 93.00% (1,050 false merges across 20,000 entities)
- Optimal threshold τ*: 0.53 (High confidence threshold: 0.75, Address floor: 0.22)
- Blocking recall: 89.17% (max_k=130)
- Reduction ratio: >99.85%

---

## 5. Reproducibility

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run full pipeline
uvicorn backend.app.main:app --port 8000

# Or CLI runner:
python -m backend.app.ml.pipeline_runner --help

# Validate output
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

---

## 6. Model License Compliance

| Component | License | Parameters |
|-----------|---------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | 22M |
| LightGBM | MIT | — (gradient boosting, no param count) |
| datasketch | MIT | — |
| python-Levenshtein | GPL-2.0 / LGPL | — (pure algorithm) |

All models are under Apache 2.0 or MIT. Total parameter count: 22M (well under 8B limit).

---

## 7. What We Would Try With More Time

- Cross-lingual embeddings (LaBSE / multilingual-e5) for better French coverage
- Address parsing with libpostal-like regex for structured field extraction
- Ensemble of LightGBM + CatBoost with calibration
- Active learning loop: human-in-the-loop review via the dashboard to label hard pairs
- FAISS GPU index for faster ANN blocking at scale
