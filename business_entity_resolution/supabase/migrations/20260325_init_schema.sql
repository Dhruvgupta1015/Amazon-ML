-- ============================================================
-- ML Challenge 2026 — Business Entity Resolution
-- Supabase Migration: 20260325_init_schema.sql
-- ============================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================
-- 1. Entities Table
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    entity_id        VARCHAR(64) PRIMARY KEY,
    source_prefix    VARCHAR(4)  NOT NULL,
    business_name    TEXT        NOT NULL,
    cleaned_name     TEXT,
    business_address TEXT,
    cleaned_address  TEXT,
    country          VARCHAR(64) NOT NULL,
    name_embedding   vector(384),
    address_embedding vector(384),
    dataset_split    VARCHAR(16) NOT NULL DEFAULT 'train',
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_source  ON entities(source_prefix);
CREATE INDEX IF NOT EXISTS idx_entities_country ON entities(country);
CREATE INDEX IF NOT EXISTS idx_entities_split   ON entities(dataset_split);
CREATE INDEX IF NOT EXISTS idx_entities_country_source ON entities(country, source_prefix);

CREATE INDEX IF NOT EXISTS idx_entities_name_embedding ON entities
    USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_entities_address_embedding ON entities
    USING hnsw (address_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- 2. Training Ground Truth Table
-- ============================================================
CREATE TABLE IF NOT EXISTS ground_truth (
    source1_entity_id  VARCHAR(64) PRIMARY KEY
        REFERENCES entities(entity_id) ON DELETE CASCADE,
    matched_entity_ids TEXT[],
    is_singleton       BOOLEAN GENERATED ALWAYS AS (
        cardinality(matched_entity_ids) = 0 OR matched_entity_ids IS NULL
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_gt_singleton ON ground_truth(is_singleton);

-- ============================================================
-- 3. Candidate Pairs (Blocking Phase Output)
-- ============================================================
CREATE TABLE IF NOT EXISTS candidate_pairs (
    source1_entity_id   VARCHAR(64) NOT NULL,
    candidate_entity_id VARCHAR(64) NOT NULL,
    source_dataset      VARCHAR(16) NOT NULL DEFAULT 'test',
    blocking_method     VARCHAR(32) NOT NULL,
    similarity_score    FLOAT,
    PRIMARY KEY (source1_entity_id, candidate_entity_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_s1     ON candidate_pairs(source1_entity_id);
CREATE INDEX IF NOT EXISTS idx_candidates_method ON candidate_pairs(blocking_method);
CREATE INDEX IF NOT EXISTS idx_candidates_split  ON candidate_pairs(source_dataset);

-- ============================================================
-- 4. Entity Matches (Final Model Output)
-- ============================================================
CREATE TABLE IF NOT EXISTS entity_matches (
    source1_entity_id    VARCHAR(64) NOT NULL,
    matched_entity_ids   TEXT[]  NOT NULL DEFAULT '{}',
    predicted_confidence FLOAT,
    dataset_split        VARCHAR(16) NOT NULL DEFAULT 'test',
    PRIMARY KEY (source1_entity_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_split ON entity_matches(dataset_split);

-- ============================================================
-- 5. Pipeline Job Runs & Benchmarking
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                        UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name                  TEXT    NOT NULL,
    dataset_split             VARCHAR(16) NOT NULL,
    blocking_candidates_count BIGINT,
    reduction_ratio           FLOAT,
    blocking_recall           FLOAT,
    validation_f05            FLOAT,
    validation_precision      FLOAT,
    validation_recall         FLOAT,
    optimal_threshold         FLOAT,
    status                    VARCHAR(32) DEFAULT 'pending',
    log_messages              JSONB   DEFAULT '[]'::JSONB,
    created_at                TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    finished_at               TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_split  ON pipeline_runs(dataset_split);

-- ============================================================
-- 6. Threshold Calibration Results
-- ============================================================
CREATE TABLE IF NOT EXISTS threshold_calibration (
    id            UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id        UUID    REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    threshold     FLOAT   NOT NULL,
    precision_val FLOAT,
    recall_val    FLOAT,
    f05_val       FLOAT,
    singleton_acc FLOAT,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cal_run ON threshold_calibration(run_id);
CREATE INDEX IF NOT EXISTS idx_cal_f05 ON threshold_calibration(f05_val DESC);

-- ============================================================
-- 7. Feature Store
-- ============================================================
CREATE TABLE IF NOT EXISTS pairwise_features (
    source1_entity_id   VARCHAR(64) NOT NULL,
    candidate_entity_id VARCHAR(64) NOT NULL,
    features            JSONB,
    match_prob          FLOAT,
    is_match            BOOLEAN,
    dataset_split       VARCHAR(16) NOT NULL DEFAULT 'train',
    PRIMARY KEY (source1_entity_id, candidate_entity_id)
);

-- ============================================================
-- Helper view: blocking recall estimation
-- ============================================================
CREATE OR REPLACE VIEW v_blocking_recall AS
SELECT
    COUNT(DISTINCT gt.source1_entity_id)::FLOAT /
        NULLIF((SELECT COUNT(*) FROM ground_truth WHERE cardinality(matched_entity_ids) > 0), 0)
        AS blocking_recall,
    COUNT(DISTINCT cp.source1_entity_id) AS s1_covered,
    COUNT(*) AS total_candidates
FROM ground_truth gt
JOIN candidate_pairs cp ON cp.source1_entity_id = gt.source1_entity_id
JOIN LATERAL unnest(gt.matched_entity_ids) AS gid ON cp.candidate_entity_id = gid
WHERE cardinality(gt.matched_entity_ids) > 0;
