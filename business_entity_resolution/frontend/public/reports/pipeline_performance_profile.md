# RESOLVE.AI: Pipeline Performance & Scalability Profile

**Profiling Date**: 2026-09-25T10:10:30Z  
**Peak RAM Consumed**: 96.74 MB  
**Benchmark Sample**: 10,000 records  

---

## 1. Stage-by-Stage Latency & Throughput

| Pipeline Stage | Metric / Units | Measured Throughput | Measured Runtime |
| :--- | :--- | :--- | :--- |
| **Data Loading** | Records / sec | **100,983.1 rec/s** | 0.099 s |
| **Normalization (NFKD)** | Records / sec | **3,875.7 rec/s** | 2.580 s |
| **Blocking & Candidate Gen** | Entities / sec | **164.4 ent/s** | 30.420 s |
| **Feature Vectorization (28D)** | Pairs / sec | **373.6 pairs/s** | 5.893 s |
| **Model Inference (LightGBM)** | Pairs / sec | **839,636.6 pairs/s** | 0.060 s |

---

## 2. Full Test Set Scalability Projection (1.73M Entities)

- **Total S1 Reference Entities**: 1,732,544
- **Projected Blocking Time**: 175.64 minutes
- **Projected Inference Time**: 0.03 minutes
- **Estimated Peak Memory**: 0.33 GB (Well within standard 16GB RAM limits)
