# Business Entity Resolution Platform — Amazon ML Challenge 2026

> **Production-grade enterprise Entity Resolution (ER) system** combining MinHash LSH blocking, dense ANN retrieval, LightGBM classification, and a Next.js interactive dashboard. Optimized for **Macro F₀.₅** with open-set multilingual support (US / India / France).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js 15 Frontend (Port 3000)              │
│  ⚡ Dashboard  ·  🔍 Match Explorer  ·  📊 F₀.₅ Lab  ·  📦 Export │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
┌──────────────────────────┴──────────────────────────────────────┐
│               FastAPI Backend Engine (Port 8000)                │
│  Ingestion → Preprocessing → Blocking → Features → LightGBM    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│           Supabase PostgreSQL + pgvector (optional)             │
│  entities · ground_truth · candidate_pairs · pipeline_runs      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Backend (Python 3.11+)

```bash
cd business_entity_resolution/backend

# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (optional — Supabase)
cp .env.example .env
# Edit .env with your Supabase URL and key

# 4. Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### Frontend (Node.js 20+)

```bash
cd business_entity_resolution/frontend

# 1. Install dependencies
npm install

# 2. Set API URL (optional)
echo "NEXT_PUBLIC_API_URL=http://localhost:8000/api" > .env.local

# 3. Start dev server
npm run dev
```

Dashboard: http://localhost:3000

### Run End-to-End Pipeline (CLI)

```bash
cd business_entity_resolution/backend

python -m app.ml.pipeline_runner \
    --train-s1  data_raw/student_resource/dataset/train/train_source1.tsv \
    --train-s2  data_raw/student_resource/dataset/train/train_source2.tsv \
    --train-s3  data_raw/student_resource/dataset/train/train_source3.tsv \
    --train-gt  data_raw/student_resource/dataset/train/train_ground_truth.tsv \
    --test-s1   data_raw/student_resource/dataset/test/test_source1.tsv \
    --test-s2   data_raw/student_resource/dataset/test/test_source2.tsv \
    --test-s3   data_raw/student_resource/dataset/test/test_source3.tsv \
    --output    output/
```

### Validate Submission

```bash
# From student_resource/ directory:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## ML Pipeline

### Stage 1: Preprocessing (`backend/app/ml/preprocessor.py`)

- Transliterates French accents (é→e, ç→c) via `unicodedata.NFKD`
- Normalizes corporate suffixes: `pvt limited → pvt ltd`, `sarl`, `gmbh`, etc.
- Expands abbreviations: `st → street`, `rd → road`, `ave → avenue`, `blvd`
- Extracts numeric signatures: street numbers, US ZIP (5-digit), India PIN (6-digit), France CP (5-digit)

### Stage 2: Blocking (`backend/app/ml/blocking.py`)

| Stage | Method | Description |
|-------|--------|-------------|
| 0 | Hard Country Partition | Never pairs across countries |
| 1 | MinHash LSH | 128-permutation MinHash on char 3-grams + word tokens; threshold 0.25 |
| 2 | Address Digit Blocking | Shared street number/postal code within country |
| 3 | Dense ANN | `sentence-transformers/all-MiniLM-L6-v2` (22M params, Apache 2.0), top-30 cosine neighbors |

Target: **≥98% recall ceiling** with **>99.5% reduction ratio**.

### Stage 3: Feature Engineering (`backend/app/ml/feature_extractor.py`)

28 discriminative features per pair:
- **Name (7)**: Levenshtein, Jaro-Winkler, Monge-Elkan, Token Sort Ratio, Token Set Ratio, LCS, TF-IDF n-gram cosine
- **Address (3)**: Token Jaccard, Digit Jaccard, Postal Exact Match
- **Phonetic (2)**: Double Metaphone match, prefix match
- **Structural (3)**: Word length difference, token count diff, char length ratio
- **Embedding (2)**: Name cosine, address cosine (384-dim sentence-transformer)
- **Cross-source (2)**: S2 flag, S3 flag
- **Heuristics (9)**: Suffix match, address presence, rare token shared, etc.

### Stage 4: Classification (`backend/app/ml/model.py`)

- **Model**: LightGBM `binary_logloss` (MIT licensed)
- **Training**: GroupKFold CV (5-fold, grouped by S1 entity to prevent leakage)
- **Threshold**: Grid search over [0.50, 0.98] step 0.01, maximizing macro F₀.₅
- **Singleton guard**: If `max P(match) < τ*`, predict empty match list

### Stage 5: Validation (`backend/app/ml/validator.py`)

Mirrors all checks from `utils/validate_submission.py`:
- Tab-separated format enforcement
- Exactly one row per S1 entity
- No self-matches, no S1-prefixed IDs
- Matches are subset of candidates

## Database Schema

See `supabase/migrations/20260325_init_schema.sql` for:
- `entities` — pgvector HNSW indexes for ANN blocking
- `ground_truth` — computed singleton flag
- `candidate_pairs`, `entity_matches`, `pipeline_runs`, `threshold_calibration`

## Key Design Decisions

1. **Open-set countries**: No hardcoded country handling. Preprocessing uses `unicodedata` for universal diacritic stripping.
2. **Precision priority**: F₀.₅ optimization means we prefer not merging over false merges. Singleton guard is critical.
3. **No external APIs**: Pure text features + pre-trained sentence-transformers. No geocoding, no business registry lookups.
4. **MIT/Apache 2.0**: LightGBM (MIT) + sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0) + datasketch (MIT).

## License

MIT License — see LICENSE file.
