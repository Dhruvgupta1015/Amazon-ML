# Phase 1 Pipeline Audit & Architecture Verification

**Project**: RESOLVE.AI — Amazon ML Challenge 2026  
**Audit Date**: September 25, 2026  
**Auditor**: Senior ML Competition Engineer & Code Auditor  

---

## 1. Executive Findings: The Dual-Pipeline Architecture Flaw

During our comprehensive code audit of the repository, we identified a critical architectural inconsistency between public claims/documentation and actual executable code. The repository currently maintains **two independent and contradictory resolution systems**:

```
                       [REPOSITORY REALITY]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
  [SYSTEM A: Packaged ML Suite]               [SYSTEM B: run_full_resolution.py]
  code/business_entity_resolution/src/       Standalone root Python script
  - MinHash LSH (datasketch)                 - Inverted Token Index
  - Dense Cosine Blocking                    - Address Digit Lookup
  - 28 Pairwise Features                     - Handcrafted fast_similarity()
  - LightGBM GBDT MatchClassifier            - Hardcoded TAU_THRESHOLD = 0.68
  - GroupKFold CV                            - Arbitrary list(union)[:50] cap
  - Official Submission Validator            - Directly outputs TSVs to output/
        │                                               │
        ▼                                               ▼
  Documented in README & Web UI              ACTUALLY GENERATED LEADERBOARD OUTPUTS!
  (Never invoked for final files)            (Zero LightGBM inference)
```

---

## 2. Answers to Specific Audit Questions

### Q1: Which script generated the current leaderboard output?
**Finding**: The files currently in `output/matching_results.tsv` and `output/candidate_pairs.tsv` were generated **exclusively by `run_full_resolution.py`**.
- `run_full_resolution.py` (lines 296–307) opens `output/matching_results.tsv` and `output/candidate_pairs.tsv` and writes them using a hand-crafted rule engine.
- `pipeline_runner.py` in `code/business_entity_resolution/src/` was **never executed on the full test set** to generate these files.

### Q2: Does the leaderboard output actually come from LightGBM?
**Finding**: **NO.**
- In `run_full_resolution.py`, lines 217–234:
  ```python
  score, sim_name, shared_digits = fast_similarity(...)
  if sim_name >= 0.82:
      matched.append(cand_id)
  elif sim_name >= 0.60 and shared_digits:
      matched.append(cand_id)
  elif len(shared_digits) >= 2 and sim_name >= 0.40:
      matched.append(cand_id)
  elif score >= TAU_THRESHOLD:
      matched.append(cand_id)
  ```
- This is a heuristic decision rule based on character Levenshtein/token-sort and shared digits. There is **no LightGBM model imported, loaded, or evaluated** anywhere in `run_full_resolution.py`.

### Q3: Does the website's displayed F0.5 correspond to a real validation run?
**Finding**: **NO (previously hardcoded / mock).**
- The website originally displayed a static `Macro F0.5 = 0.942`.
- When tested on held-out validation data (20,000 entities, Seed 42), the true baseline achieved **0.7930 Macro F0.5**.
- The `0.942` claim was an aspirational target or synthetic placeholder, not an empirically verified score.

### Q4: Does the website's blocking recall correspond to the actual candidate file?
**Finding**: **NO.**
- The website claimed `99.18%` blocking recall.
- Real empirical measurement on 20,000 training entities shows blocking recall between **72.84% and 79.92%**.
- A claim of 99.18% recall with a 98.4% reduction ratio on 10 million noisy multilingual business records is unsupported by empirical evidence.

### Q5: Are the same blocking and matching rules used in training and test inference?
**Finding**: **NO.**
- In training (`src/blocking.py`), candidates were generated via MinHash LSH + address digit matching + dense ANN.
- In test inference (`run_full_resolution.py`), candidates were generated via raw token inverted index and address digit matching.
- Training used 28 engineered features to optimize LightGBM; inference bypassed all 28 features and used hand-crafted heuristic thresholds.

---

## 3. Dangerous Flaws in Current Candidate Generation & Validation

### Flaw 1: Arbitrary Set Truncation
In `run_full_resolution.py` (line 212) and `blocking.py` (line 237):
```python
cand_list = list(candidates)[:MAX_CANDIDATES_PER_S1]
```
Because `candidates` is a Python `set`, its iteration order is arbitrary (determined by memory address hash seeds). Taking `[:50]` truncates **an arbitrary 50 candidates**, frequently discarding true matches before any classifier or similarity function evaluates them. Candidates **must be ranked by a preliminary quality score** prior to any top-K pruning.

### Flaw 2: Validation Threshold Tuning Ignores True Singletons
In `model.py` (lines 40–53):
- `build_training_data()` only creates training groups for records that have positive matches or candidate pairs.
- True singletons with 0 candidates are omitted from the training matrix.
- `_search_threshold()` subsequently computes Macro F0.5 only across these groups.
- Under official challenge rules, **every Source 1 record is evaluated**. A singleton with 0 predicted matches scores $F_{0.5} = 1.0$, while any false-positive merge drops the entity's score to $0.0$. Evaluating only candidate groups severely biases the threshold downward and ignores the devastating impact of false merges on the 80%+ singleton population.

### Flaw 3: Dense Retrieval Scalability Hazard
In `blocking.py` (line 191):
```python
sims = s1_batch @ s23_embs.T
```
Multiplying a batch of S1 embeddings by the transpose of all S2/S3 embeddings requires allocating a dense matrix of $(512 \times N)$. Across millions of records, this causes out-of-memory crashes or extreme thrashing on standard hardware.

---

## 4. Remediation Plan

1. **Unify into ONE Authoritative Pipeline**: Deprecate the heuristic branch. Refactor `run_full_resolution.py` and `code/business_entity_resolution/src/` to share the exact same model, preprocessor, blocking, ranked pruning, feature extractor, and threshold.
2. **Implement Ranked Candidate Pruning**: Score candidates before pruning; perform a systematic K sweep ($K \in [10, 20, 30, 50, 75, 100, 150, 200]$) to establish the empirical recall vs efficiency curve.
3. **Singleton-Aware Validation Framework**: Evaluate all Source 1 entities in validation, including 0-candidate singletons, matching the exact official Macro F0.5 formulation.
4. **Hard Negative Mining**: Train LightGBM with near-name distractors and same-street/different-business negatives.
5. **Generate Authentic Submissions**: Regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the unified ML pipeline and verify with the official challenge validator.
