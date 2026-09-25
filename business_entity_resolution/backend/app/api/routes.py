"""routes.py - FastAPI Router: all REST endpoints for the ER platform."""
from __future__ import annotations

import io
import logging
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from ..core.config import settings
from .schemas import (
    DatasetStatsResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    ThresholdTuneRequest,
    ThresholdTuneResponse,
    ValidationRequest,
    ValidationResponse,
)
from ..ml.preprocessor import Preprocessor
from ..ml.blocking import BlockingEngine
from ..ml.model import MatchClassifier, build_training_data
from ..ml.metrics import precision_recall_f05
from ..ml.validator import validate_submission

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job registry
_JOBS: dict[str, dict] = {}
_MODELS: dict[str, MatchClassifier] = {}


def _load_source(path: Path | str) -> list[dict]:
    df = pd.read_csv(str(path), sep="\t", dtype=str, na_filter=False)
    records = df.to_dict(orient="records")
    preprocessor = Preprocessor()
    return [preprocessor.process_record(r) for r in records]


def _load_ground_truth(path: Path | str) -> dict[str, list[str]]:
    df = pd.read_csv(str(path), sep="\t", dtype=str, na_filter=False)
    gt: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        ids_str = row.get("matched_entity_ids", "")
        gt[row["source1_entity_id"]] = ids_str.split(",") if ids_str.strip() else []
    return gt


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a .tsv file upload, save to data directory, return info."""
    allowed = {
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
        "train_ground_truth.tsv", "test_source1.tsv", "test_source2.tsv", "test_source3.tsv",
    }
    if file.filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Unexpected file: {file.filename}")

    dest_dir = settings.DATA_DIR / ("train" if "train" in file.filename else "test")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)

    lines = content.count(b"\n")
    return {"filename": file.filename, "size_bytes": len(content), "approx_rows": lines - 1}


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
async def dataset_stats(split: str = "train"):
    base = settings.DATA_DIR / split

    def _count_and_countries(path: Path):
        if not path.exists():
            return 0, {}
        df = pd.read_csv(str(path), sep="\t", dtype=str, na_filter=False,
                         usecols=["entity_id", "country"])
        country_counts = df["country"].value_counts().to_dict()
        return len(df), country_counts

    s1_count, s1_countries = _count_and_countries(base / f"{split}_source1.tsv")
    s2_count, _ = _count_and_countries(base / f"{split}_source2.tsv")
    s3_count, _ = _count_and_countries(base / f"{split}_source3.tsv")

    singleton_count = matched_count = None
    gt_path = base / "train_ground_truth.tsv"
    if gt_path.exists():
        gt = _load_ground_truth(gt_path)
        singleton_count = sum(1 for v in gt.values() if not v)
        matched_count = sum(1 for v in gt.values() if v)

    return DatasetStatsResponse(
        split=split,
        source1_count=s1_count,
        source2_count=s2_count,
        source3_count=s3_count,
        countries=s1_countries,
        singleton_count=singleton_count,
        matched_count=matched_count,
    )


async def _run_pipeline(run_id: str, req: PipelineRunRequest):
    job = _JOBS[run_id]

    def _log(msg: str):
        logger.info(msg)
        job["logs"].append(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")

    try:
        job["status"] = "running"
        split = req.dataset_split
        base = settings.DATA_DIR / split

        _log("Loading source files...")
        s1_path = base / f"{split}_source1.tsv"
        s2_path = base / f"{split}_source2.tsv"
        s3_path = base / f"{split}_source3.tsv"

        for p in [s1_path, s2_path, s3_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        s1_records = _load_source(s1_path)
        s2_records = _load_source(s2_path)
        s3_records = _load_source(s3_path)
        s23_records = s2_records + s3_records
        job["progress_pct"] = 20

        gt: dict[str, list[str]] = {}
        if split == "train":
            gt_path = base / "train_ground_truth.tsv"
            if gt_path.exists():
                gt = _load_ground_truth(gt_path)
                _log(f"Ground truth loaded: {len(gt)} S1 entities")

        _log("Running blocking stages...")
        engine = BlockingEngine(use_dense=req.use_dense)
        candidates = engine.run(s1_records, s23_records)
        job["progress_pct"] = 50

        total_pairs = sum(len(v) for v in candidates.values())
        max_possible = len(s1_records) * len(s23_records)
        reduction_ratio = 1.0 - total_pairs / max(max_possible, 1)
        blocking_recall = engine.compute_blocking_recall(candidates, gt) if gt else None

        job.update({
            "blocking_candidates_count": total_pairs,
            "reduction_ratio": reduction_ratio,
            "blocking_recall": blocking_recall,
        })
        _log(f"Blocking: {total_pairs} pairs, rr={reduction_ratio:.4f}, recall={blocking_recall}")

        classifier = MatchClassifier()
        if split == "train" and gt:
            _log("Building training data and training classifier...")
            X, y, groups = build_training_data(s1_records, s23_records, candidates, gt)
            cal_results = classifier.train(X, y, groups)
            classifier.save(settings.MODEL_DIR / f"{run_id}_model.pkl")
            job.update({
                "optimal_threshold": cal_results["optimal_threshold"],
                "validation_f05": cal_results["best_f05"],
                "threshold_curve": cal_results.get("threshold_curve", []),
            })
            _log(f"Model trained: tau*={cal_results['optimal_threshold']:.2f}, "
                 f"F0.5={cal_results['best_f05']:.4f}")
        elif (settings.MODEL_DIR / "best_model.pkl").exists():
            _log("Loading best model from disk...")
            classifier.load(settings.MODEL_DIR / "best_model.pkl")

        if req.threshold is not None:
            classifier.optimal_threshold = req.threshold

        _log("Running inference...")
        job["progress_pct"] = 70
        predictions = classifier.predict_entities(s1_records, s23_records, candidates)
        _MODELS[run_id] = classifier

        if split == "train" and gt:
            metrics = precision_recall_f05(predictions, gt)
            job.update({
                "validation_f05": metrics["macro_f05"],
                "validation_precision": metrics["macro_precision"],
                "validation_recall": metrics["macro_recall"],
                "optimal_threshold": classifier.optimal_threshold,
            })
            _log(f"Validation: F0.5={metrics['macro_f05']:.4f}, "
                 f"P={metrics['macro_precision']:.4f}, R={metrics['macro_recall']:.4f}")

        _log("Writing output TSV files...")
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        mr_path = settings.OUTPUT_DIR / "matching_results.tsv"
        with open(mr_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_rec in s1_records:
                s1_id = s1_rec["entity_id"]
                matched = predictions.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(matched)}\n")

        cp_path = settings.OUTPUT_DIR / "candidate_pairs.tsv"
        with open(cp_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_rec in s1_records:
                s1_id = s1_rec["entity_id"]
                cands = candidates.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(cands)}\n")

        _log(f"Output files written to {settings.OUTPUT_DIR}")
        job["progress_pct"] = 100
        job["status"] = "done"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:
        logger.exception("Pipeline error")
        job["status"] = "error"
        job["logs"].append(f"ERROR: {exc}")


@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    _JOBS[run_id] = {
        "run_id": run_id,
        "run_name": req.run_name,
        "dataset_split": req.dataset_split,
        "status": "pending",
        "logs": [],
        "progress_pct": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
    background_tasks.add_task(_run_pipeline, run_id, req)
    return PipelineRunResponse(run_id=run_id, status="pending", message="Pipeline started")


@router.get("/pipeline/status/{run_id}", response_model=PipelineStatusResponse)
async def pipeline_status(run_id: str):
    job = _JOBS.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return PipelineStatusResponse(
        run_id=run_id,
        run_name=job.get("run_name", ""),
        status=job.get("status", "unknown"),
        progress_pct=job.get("progress_pct"),
        blocking_candidates_count=job.get("blocking_candidates_count"),
        reduction_ratio=job.get("reduction_ratio"),
        blocking_recall=job.get("blocking_recall"),
        validation_f05=job.get("validation_f05"),
        validation_precision=job.get("validation_precision"),
        validation_recall=job.get("validation_recall"),
        optimal_threshold=job.get("optimal_threshold"),
        log_messages=job.get("logs", []),
        created_at=job.get("created_at"),
        finished_at=job.get("finished_at"),
    )


@router.get("/pipeline/list")
async def pipeline_list():
    return [
        {
            "run_id": rid,
            "run_name": j.get("run_name"),
            "status": j.get("status"),
            "created_at": j.get("created_at"),
        }
        for rid, j in _JOBS.items()
    ]


@router.post("/benchmark/tune", response_model=ThresholdTuneResponse)
async def benchmark_tune(req: ThresholdTuneRequest):
    job = _JOBS.get(req.run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="Pipeline run not complete yet")

    cal_curve = job.get("threshold_curve", [])
    if not cal_curve:
        best_tau = job.get("optimal_threshold", 0.5)
        cal_curve = [{"threshold": best_tau, "f05": job.get("validation_f05", 0.0)}]

    best = max(cal_curve, key=lambda x: x.get("f05", 0.0), default={"threshold": 0.5, "f05": 0.0})
    return ThresholdTuneResponse(
        curve=cal_curve,
        optimal_threshold=best.get("threshold", 0.5),
        best_f05=best.get("f05", 0.0),
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_endpoint(req: ValidationRequest):
    matching_path = req.matching_path or str(settings.OUTPUT_DIR / "matching_results.tsv")
    candidate_path = req.candidate_path or str(settings.OUTPUT_DIR / "candidate_pairs.tsv")
    test_s1 = str(settings.DATA_DIR / "test" / "test_source1.tsv")
    test_s2 = str(settings.DATA_DIR / "test" / "test_source2.tsv")
    test_s3 = str(settings.DATA_DIR / "test" / "test_source3.tsv")

    result = validate_submission(
        matching_path=matching_path,
        candidate_path=candidate_path,
        test_source1_path=test_s1,
        test_source2_path=test_s2 if req.check_ids else None,
        test_source3_path=test_s3 if req.check_ids else None,
        check_ids=req.check_ids,
    )
    return ValidationResponse(
        passed=result.passed,
        errors=result.errors,
        warnings=result.warnings,
        stats=result.stats,
    )


@router.get("/export/submission")
async def export_submission(team_name: str = "team"):
    clean_team = team_name.strip().replace(" ", "_")
    mr_path = settings.OUTPUT_DIR / "matching_results.tsv"
    cp_path = settings.OUTPUT_DIR / "candidate_pairs.tsv"

    # Fallback to root output if local not populated
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    if not mr_path.exists() and (root_dir / "output" / "matching_results.tsv").exists():
        mr_path = root_dir / "output" / "matching_results.tsv"
        cp_path = root_dir / "output" / "candidate_pairs.tsv"

    if not mr_path.exists():
        raise HTTPException(
            status_code=404,
            detail="matching_results.tsv not found. Run pipeline first."
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(mr_path, arcname="output/matching_results.tsv")
        if cp_path.exists():
            zf.write(cp_path, arcname="output/candidate_pairs.tsv")

        # Code modules
        src_dir = root_dir / "code" / "business_entity_resolution" / "src"
        if src_dir.exists():
            for py_file in src_dir.glob("*.py"):
                zf.write(py_file, arcname=f"code/business_entity_resolution/src/{py_file.name}")
        else:
            backend_dir = Path(__file__).parent.parent.parent
            for py_file in backend_dir.rglob("*.py"):
                zf.write(py_file, arcname=f"code/business_entity_resolution/src/{py_file.name}")

        code_readme = root_dir / "code" / "business_entity_resolution" / "README.md"
        if code_readme.exists():
            zf.write(code_readme, arcname="code/business_entity_resolution/README.md")

        code_reqs = root_dir / "code" / "business_entity_resolution" / "requirements.txt"
        if code_reqs.exists():
            zf.write(code_reqs, arcname="code/business_entity_resolution/requirements.txt")

        doc_md = root_dir / "Documentation_template.md"
        if doc_md.exists():
            zf.write(doc_md, arcname="Documentation_template.md")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={clean_team}_submission.zip"},
    )


@router.get("/entity/{entity_id}")
async def entity_detail(entity_id: str):
    for split in ["test", "train"]:
        path = settings.DATA_DIR / split / f"{split}_source1.tsv"
        if path.exists():
            df = pd.read_csv(str(path), sep="\t", dtype=str, na_filter=False)
            row = df[df["entity_id"] == entity_id]
            if not row.empty:
                return row.to_dict(orient="records")[0]
    raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
