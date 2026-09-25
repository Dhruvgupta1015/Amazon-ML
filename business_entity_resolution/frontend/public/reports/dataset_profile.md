# RESOLVE.AI: Dataset & Schema Verification Report

**Generated**: 2026-09-25T09:42:30Z  
**Profiling Runtime**: 70.72 seconds  

---

## 1. File Integrity & Record Counts

| Dataset Split | File Name | Size (MB) | Total Records | Unique IDs | Duplicate IDs | SHA-256 (64MB Prefix) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `train_source1.tsv` | 200.34 MB | 2,206,821 | 2,206,821 | 0 | `0beab496ed90c51b...` |
| **TRAIN** | `train_source2.tsv` | 466.63 MB | 5,034,616 | 5,034,616 | 0 | `8e1b84ca44535757...` |
| **TRAIN** | `train_source3.tsv` | 480.37 MB | 5,285,603 | 5,285,603 | 0 | `30000777378ac92b...` |
| **TRAIN** | `train_ground_truth.tsv` | 121.13 MB | 2,206,821 | 2,206,821 | 0 | `d1e37a80be255805...` |
| **TEST** | `test_source1.tsv` | 166.91 MB | 1,732,544 | 1,732,544 | 0 | `29a891a9b0b8147b...` |
| **TEST** | `test_source2.tsv` | 485.86 MB | 4,887,273 | 4,887,273 | 0 | `e911b01413ceea47...` |
| **TEST** | `test_source3.tsv` | 482.56 MB | 5,082,316 | 5,082,316 | 0 | `5bea9506848e4c0f...` |

---

## 2. Country Distribution

| Dataset Split | File Name | United States (US) | India (India) | France (France) | Other / Missing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TRAIN | `train_source1.tsv` | 1,323,633 | 883,188 | 0 | 0 |
| TRAIN | `train_source2.tsv` | 3,016,817 | 2,017,799 | 0 | 0 |
| TRAIN | `train_source3.tsv` | 3,170,056 | 2,115,547 | 0 | 0 |
| TEST | `test_source1.tsv` | 663,106 | 809,986 | 259,452 | 0 |
| TEST | `test_source2.tsv` | 1,871,330 | 2,312,565 | 703,378 | 0 |
| TEST | `test_source3.tsv` | 1,945,701 | 2,405,000 | 731,615 | 0 |

---

## 3. Schema & Missing Value Analysis

### `train_source1.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 0 (0.0%)
  - `business_address`: 0 (0.0%)
  - `country`: 0 (0.0%)

### `train_source2.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 2 (0.0%)
  - `business_address`: 168,967 (3.36%)
  - `country`: 0 (0.0%)

### `train_source3.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 13 (0.0%)
  - `business_address`: 175,916 (3.33%)
  - `country`: 0 (0.0%)

### `train_ground_truth.tsv`
- **Columns**: `source1_entity_id, matched_entity_ids`
- **Missing Values**:
  - `source1_entity_id`: 0 (0.0%)
  - `matched_entity_ids`: 123,247 (5.58%)

### `test_source1.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 0 (0.0%)
  - `business_address`: 0 (0.0%)
  - `country`: 0 (0.0%)

### `test_source2.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 46 (0.0%)
  - `business_address`: 129,408 (2.65%)
  - `country`: 0 (0.0%)

### `test_source3.tsv`
- **Columns**: `entity_id, business_name, business_address, country`
- **Missing Values**:
  - `entity_id`: 0 (0.0%)
  - `business_name`: 59 (0.0%)
  - `business_address`: 136,098 (2.68%)
  - `country`: 0 (0.0%)

---

## 4. Ground Truth Match & Singleton Cardinality

- **Total Rows in Ground Truth TSV**: 2,206,821
- **Source 1 Entities with Positive Matches**: 2,083,574
- **Explicit Singletons in Ground Truth**: 123,247
- **Implicit Singletons (Absent from Ground Truth)**: 0
- **Total Effective Singletons**: 123,247 (5.58% of S1 entities)
- **Total Positive Mentions Matched**: 7,638,365
  - Source 2 Mentions: 3,693,619
  - Source 3 Mentions: 3,944,746
- **Max Matches for Single S1 Entity**: 11

### Match Cardinality Distribution (Positive Mentions per S1 Entity):

| Matches per Entity | Number of S1 Entities |
| :--- | :--- |
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |
