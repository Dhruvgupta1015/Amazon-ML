# RESOLVE.AI: Empirical Candidate Recall & Blocking Audit

- **Evaluation Dataset**: Held-out Entity-Stratified (20,000 S1 Entities)
- **Pair-Level Candidate Recall**: **87.19%**
- **Entity-Level Perfect Recall**: **75.52%**
- **Reduction Ratio**: **99.9878%**
- **Total Candidate Pairs**: 558,052
- **Average Candidates / S1**: 27.9
- **True Matches Missed**: 8,873 / 69,256

### K-Sweep Optimization Table

| Top-K Cap | Candidate Recall (%) | Candidate Count | Reduction Ratio (%) | Avg Cands/S1 | Missing True Matches |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10** | **83.05%** | 174,802 | 99.9962% | 8.74 | 11,742 |
| **20** | **85.02%** | 313,441 | 99.9931% | 15.67 | 10,373 |
| **30** | **85.88%** | 421,480 | 99.9908% | 21.07 | 9,782 |
| **50** | **87.19%** | 558,052 | 99.9878% | 27.9 | 8,873 |
| **75** | **87.81%** | 627,692 | 99.9862% | 31.38 | 8,441 |
| **100** | **87.9%** | 645,905 | 99.9858% | 32.3 | 8,378 |
| **150** | **87.92%** | 650,039 | 99.9857% | 32.5 | 8,367 |
| **200** | **87.92%** | 650,048 | 99.9857% | 32.5 | 8,367 |
