# RESOLVE.AI: Honest Limitations & Empirical Engineering Report

**Date**: September 25, 2026  
**Competition**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Repository**: [https://github.com/Dhruvgupta1015/Amazon-ML](https://github.com/Dhruvgupta1015/Amazon-ML)

---

## 1. Ground Truth Cardinality & French Language Domain Shift

### 1.1 Unseen Test Language (France / French)
- **Empirical Finding**: Profiling `data_raw/student_resource/dataset/` revealed that `train/` contains records from only **United States** (1,323,633 S1) and **India** (883,188 S1). **France is completely absent from the training set** (0 French records in `train_source1.tsv`).
- In `test/`, France accounts for **259,452 Source 1 records** (14.97% of the test set).
- **Limitation**: Any ML model that relies on country-specific memorized tokens or unnormalized French text would suffer severe out-of-distribution degradation.
- **Engineered Mitigation**: Our pipeline applies stateless NFKD Unicode diacritic transliteration (`café` $\to$ `cafe`), normalizes French corporate legal suffixes (`sarl`, `sas`, `sa`), and normalizes French street designations (`rue`, `boulevard`, `allee`, `impasse`, `chemin`) in zero-shot fashion.

---

## 2. False Merges vs. Singleton Preservation Dynamics

### 2.1 The Mathematical Asymmetry of Macro $F_{0.5}$
- The competition evaluates using Macro-averaged $F_{0.5}$:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Precision is weighted $2\times$ over Recall ($\beta^2 = 0.25$).
- **Singletons**: In ground truth, singletons (entities with 0 matches) represent **5.58%** of training entities and **78.9%** of test predictions.
- If a true singleton is incorrectly merged with even a single candidate mention (False Positive / False Merge):
  $$\text{Precision} = 0.0, \quad \text{Recall} = 1.0 \implies F_{0.5} = 0.0$$
  This yields a catastrophic 0.0 score for that entity.
- Conversely, predicting an empty list for a singleton yields $F_{0.5} = 1.0$.

### 2.2 Threshold Calibration Insight
- In our empirical threshold sweep (`reports/threshold_sweep.csv`), aggressive low thresholds ($\tau = 0.20$) maximized recall (81.18%) but resulted in 15,360 false merges, reducing Macro $F_{0.5}$ to 0.3616.
- A conservative, precision-calibrated threshold ($\tau^* = 0.56 - 0.68$) with a strict street number conflict penalty ($-0.35$ for conflicting street numbers in the same locality) eliminated 12,000+ false merges, yielding a verified validation score of **Macro $F_{0.5} = 0.7930$** with **0.9756 Precision**.

---

## 3. Candidate Generation (Blocking) Trade-Offs

| Strategy | Candidate Recall (%) | Reduction Ratio (%) | Average Candidates / S1 | Missing Matches / 69,256 |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy A: Name Token Inverted Index** | 45.59% | 99.9963% | 12.01 | 37,679 |
| **Strategy B: Multi-Field (Name + Address)** | 78.63% | 99.9919% | 25.89 | 14,798 |
| **Strategy C: Multi-Field + Street Signatures** | 54.89% | 99.9953% | 15.14 | 31,241 |
| **Strategy D: RESOLVE.AI Country-Partitioned Multi-Index** | **79.92%** | **99.9921%** | **25.46** | **13,909** |

- **Limitation**: Approximately 20.08% of true matches in the validation pool have extreme name abbreviations (e.g., acronyms with zero token overlap) and omitted addresses in Source 2/3.
- **Future Direction**: Dense semantic embedding retrieval using bi-encoders (e.g. `all-MiniLM-L6-v2`) fine-tuned with contrastive InfoNCE loss can recover semantic aliases without requiring token overlap, provided GPU inference budgets permit.

---

## 4. Rule Compliance & Verified Integrity

1. **Zero Internet Lookups**: Purely offline, self-contained algorithms. No Google Maps, Nominatim, or external geocoding APIs.
2. **Model Parameter Constraint**: LightGBM tree ensemble size is 2.8 MB (well below the 8B parameter rule).
3. **Format Integrity**: Outputs strictly formatted as tab-separated `.tsv` matching `utils/validate_submission.py`.
4. **Reproducibility**: Seed fixed at 42. Commands documented in README reproduce all outputs deterministically.
