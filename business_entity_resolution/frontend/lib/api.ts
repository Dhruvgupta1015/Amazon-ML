/**
 * lib/api.ts — Typed API client for the FastAPI backend
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Dataset ──────────────────────────────────────────────────────────────────

export interface DatasetStats {
  split: string;
  source1_count: number;
  source2_count: number;
  source3_count: number;
  countries: Record<string, number>;
  singleton_count: number | null;
  matched_count: number | null;
}

export const fetchDatasetStats = (split = "train") =>
  apiFetch<DatasetStats>(`/dataset/stats?split=${split}`);

// ── Pipeline ─────────────────────────────────────────────────────────────────

export interface PipelineRunRequest {
  run_name: string;
  dataset_split: string;
  use_dense: boolean;
  threshold?: number;
}

export interface PipelineRunResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface PipelineStatus {
  run_id: string;
  run_name: string;
  status: string;
  progress_pct: number | null;
  blocking_candidates_count: number | null;
  reduction_ratio: number | null;
  blocking_recall: number | null;
  validation_f05: number | null;
  validation_precision: number | null;
  validation_recall: number | null;
  optimal_threshold: number | null;
  log_messages: string[];
  created_at: string | null;
  finished_at: string | null;
}

export const startPipeline = (req: PipelineRunRequest) =>
  apiFetch<PipelineRunResponse>("/pipeline/run", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const getPipelineStatus = (runId: string) =>
  apiFetch<PipelineStatus>(`/pipeline/status/${runId}`);

export const listPipelines = () =>
  apiFetch<{ run_id: string; run_name: string; status: string; created_at: string }[]>(
    "/pipeline/list"
  );

// ── Benchmark ────────────────────────────────────────────────────────────────

export interface ThresholdPoint {
  threshold: number;
  macro_f05: number;
  macro_precision: number;
  macro_recall: number;
  singleton_accuracy: number;
}

export interface ThresholdTuneResponse {
  curve: ThresholdPoint[];
  optimal_threshold: number;
  best_f05: number;
}

export const tuneBenchmark = (runId: string, min = 0.05, max = 0.99, step = 0.01) =>
  apiFetch<ThresholdTuneResponse>("/benchmark/tune", {
    method: "POST",
    body: JSON.stringify({
      run_id: runId,
      threshold_min: min,
      threshold_max: max,
      threshold_step: step,
    }),
  });

// ── Validation ───────────────────────────────────────────────────────────────

export interface ValidationResponse {
  passed: boolean;
  errors: string[];
  warnings: string[];
  stats: Record<string, unknown>;
}

export const validateSubmission = (checkIds = false) =>
  apiFetch<ValidationResponse>("/validate", {
    method: "POST",
    body: JSON.stringify({ check_ids: checkIds }),
  });

// ── Export ───────────────────────────────────────────────────────────────────

export const getSubmissionZipUrl = (teamName = "team") =>
  `${BASE_URL}/export/submission?team_name=${encodeURIComponent(teamName)}`;

// ── Entity ───────────────────────────────────────────────────────────────────

export const fetchEntity = (entityId: string) =>
  apiFetch<Record<string, unknown>>(`/entity/${encodeURIComponent(entityId)}`);
