"""schemas.py - Pydantic Request/Response Models."""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PipelineRunRequest(BaseModel):
    run_name: str = Field(..., description="Human-readable name for this pipeline run")
    dataset_split: str = Field("test", description="train or test")
    use_dense: bool = Field(True, description="Enable dense ANN blocking stage")
    threshold: Optional[float] = Field(None, description="Override optimal threshold 0.0-1.0")


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    progress_pct: Optional[float] = None
    blocking_candidates_count: Optional[int] = None
    reduction_ratio: Optional[float] = None
    blocking_recall: Optional[float] = None
    validation_f05: Optional[float] = None
    validation_precision: Optional[float] = None
    validation_recall: Optional[float] = None
    optimal_threshold: Optional[float] = None
    log_messages: List[str] = []
    created_at: Optional[str] = None
    finished_at: Optional[str] = None


class ThresholdTuneRequest(BaseModel):
    run_id: str
    threshold_min: float = 0.05
    threshold_max: float = 0.99
    threshold_step: float = 0.01


class ThresholdTuneResponse(BaseModel):
    curve: List[Dict[str, float]]
    optimal_threshold: float
    best_f05: float


class ValidationRequest(BaseModel):
    matching_path: Optional[str] = None
    candidate_path: Optional[str] = None
    check_ids: bool = False


class ValidationResponse(BaseModel):
    passed: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class DatasetStatsResponse(BaseModel):
    split: str
    source1_count: int
    source2_count: int
    source3_count: int
    countries: Dict[str, int]
    singleton_count: Optional[int] = None
    matched_count: Optional[int] = None
