# Business Entity Resolution Pipeline — Amazon ML Challenge 2026
## Reproduction Guide & Architecture Reference

### 1. Overview
This package contains the self-contained, reproducible pipeline for the Amazon ML Challenge 2026 Business Entity Resolution competition.

- **Objective**: Match noisy external records from Source 2 and Source 3 against the deduplicated reference Source 1.
- **Evaluation Metric**: Macro-averaged $F_{0.5}$ score across all Source 1 entities (including singletons).
- **Model Constraints**: LightGBM (MIT License, ~15MB) + sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0, 22M parameters) — fully compliant with $\le 8\text{B}$ parameter limit and open-source licenses.
- **Open-Set Compliance**: Country is treated as an open set of string labels. Features and normalizers operate agnostically across US, India, and France (test set).
- **Fair Play**: 100% offline execution. Strictly zero external APIs, geocoding lookups, or commercial entity resolution services.

---

### 2. Environment Setup
```bash
# Python 3.8+ recommended
cd code/business_entity_resolution
pip install -r requirements.txt
```

---

### 3. End-to-End Pipeline Execution

To train on training data and generate predictions for the test set:

```bash
python -m src.pipeline_runner \
  --train-s1 dataset/train/train_source1.tsv \
  --train-s2 dataset/train/train_source2.tsv \
  --train-s3 dataset/train/train_source3.tsv \
  --train-gt dataset/train/train_ground_truth.tsv \
  --test-s1 dataset/test/test_source1.tsv \
  --test-s2 dataset/test/test_source2.tsv \
  --test-s3 dataset/test/test_source3.tsv \
  --output output/ \
  --use-dense
```

#### Pipeline Steps:
1. **Open-Set Preprocessing (`src/preprocessor.py`)**:
   - Universal NFKD unicode diacritic stripping (French accents é/è/ç/ê, Indian transliterations).
   - Legal suffix canonicalization (Pvt Ltd, LLC, Inc, SA, SARL).
   - Address digit and postal code signature extraction.

2. **High-Recall Multi-Stage Blocking (`src/blocking.py`)**:
   - Country partition (exact match, no cross-border leakage).
   - MinHash LSH on 3-gram character and word tokens (128 permutations).
   - Shared address street number and postal code token matching.
   - Dense vector ANN search via `sentence-transformers/all-MiniLM-L6-v2`.
   - Output: `output/candidate_pairs.tsv`.

3. **28-Feature Pairwise Vectorization (`src/feature_extractor.py`)**:
   - Name metrics: Levenshtein, Jaro-Winkler, Monge-Elkan, LCS, Token Sort/Set ratios.
   - Phonetics: Double Metaphone exact phonetic codes.
   - Address metrics: Token Jaccard, Street number digit matching, Postal code exact match.
   - Dense vector cosine similarity.

4. **Macro $F_{0.5}$ Calibrated LightGBM (`src/model.py`)**:
   - Pairwise binary ranking classifier.
   - Global threshold sweep across $\tau \in [0.05, 0.99]$ optimizing Macro $F_{0.5}$.
   - Optimal operating point $\tau^* \approx 0.68$ ensuring high precision to avoid the $2\times$ false merge penalty.
   - Singletons preserved with empty prediction ($F_{0.5} = 1.0000$).
   - Output: `output/matching_results.tsv`.

---

### 4. Local Validation

To verify that the generated outputs pass all challenge rules:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Exit code `0` confirms the submission is safe to upload.
