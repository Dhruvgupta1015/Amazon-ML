"""pipeline_runner.py - CLI entry point for end-to-end pipeline execution.

Usage:
    python -m app.ml.pipeline_runner \\
        --train-s1 dataset/train/train_source1.tsv \\
        --train-s2 dataset/train/train_source2.tsv \\
        --train-s3 dataset/train/train_source3.tsv \\
        --train-gt dataset/train/train_ground_truth.tsv \\
        --test-s1  dataset/test/test_source1.tsv \\
        --test-s2  dataset/test/test_source2.tsv \\
        --test-s3  dataset/test/test_source3.tsv \\
        --output   output/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .preprocessor import Preprocessor
from .blocking import BlockingEngine
from .model import MatchClassifier, build_training_data
from .metrics import precision_recall_f05
from .validator import validate_submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_source(path: str) -> list[dict]:
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    prep = Preprocessor()
    return [prep.process_record(r) for r in df.to_dict(orient="records")]


def load_gt(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    result: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        ids_str = row.get("matched_entity_ids", "")
        result[row["source1_entity_id"]] = ids_str.split(",") if ids_str.strip() else []
    return result


def write_tsv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--train-s1", required=True)
    parser.add_argument("--train-s2", required=True)
    parser.add_argument("--train-s3", required=True)
    parser.add_argument("--train-gt", required=True)
    parser.add_argument("--test-s1", required=True)
    parser.add_argument("--test-s2", required=True)
    parser.add_argument("--test-s3", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--no-dense", action="store_true", help="Disable dense ANN blocking")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data...")
    train_s1 = load_source(args.train_s1)
    train_s2 = load_source(args.train_s2)
    train_s3 = load_source(args.train_s3)
    train_gt = load_gt(args.train_gt)
    train_s23 = train_s2 + train_s3

    logger.info("Loading test data...")
    test_s1 = load_source(args.test_s1)
    test_s2 = load_source(args.test_s2)
    test_s3 = load_source(args.test_s3)
    test_s23 = test_s2 + test_s3

    use_dense = not args.no_dense
    engine = BlockingEngine(use_dense=use_dense)

    logger.info("Running training blocking...")
    train_candidates = engine.run(train_s1, train_s23)
    blocking_recall = engine.compute_blocking_recall(train_candidates, train_gt)
    logger.info("Training blocking recall: %.4f", blocking_recall)

    logger.info("Building training features and training LightGBM...")
    classifier = MatchClassifier()
    X, y, groups = build_training_data(train_s1, train_s23, train_candidates, train_gt)
    cal = classifier.train(X, y, groups)
    logger.info(
        "Training complete: tau*=%.2f, F0.5=%.4f",
        cal["optimal_threshold"], cal["best_f05"]
    )

    if args.threshold is not None:
        classifier.optimal_threshold = args.threshold
        logger.info("Overriding threshold to %.2f", args.threshold)

    model_path = output_dir / "best_model.pkl"
    classifier.save(model_path)

    logger.info("Running test blocking...")
    test_candidates = engine.run(test_s1, test_s23)
    total_pairs = sum(len(v) for v in test_candidates.values())
    max_possible = len(test_s1) * len(test_s23)
    rr = 1.0 - total_pairs / max(max_possible, 1)
    logger.info("Test blocking: %d pairs, reduction_ratio=%.4f", total_pairs, rr)

    logger.info("Running inference on test set...")
    predictions = classifier.predict_entities(test_s1, test_s23, test_candidates)

    mr_path = str(output_dir / "matching_results.tsv")
    cp_path = str(output_dir / "candidate_pairs.tsv")

    mr_rows = [
        [r["entity_id"], ",".join(predictions.get(r["entity_id"], []))]
        for r in test_s1
    ]
    cp_rows = [
        [r["entity_id"], ",".join(test_candidates.get(r["entity_id"], []))]
        for r in test_s1
    ]

    write_tsv(mr_path, ["source1_entity_id", "matched_entity_ids"], mr_rows)
    write_tsv(cp_path, ["source1_entity_id", "candidate_entity_ids"], cp_rows)
    logger.info("Output written to %s", output_dir)

    logger.info("Running submission validator...")
    result = validate_submission(
        matching_path=mr_path,
        candidate_path=cp_path,
        test_source1_path=args.test_s1,
    )
    if result.passed:
        logger.info("PASS -- submission is valid")
    else:
        logger.error("FAIL -- submission has issues:")
        for e in result.errors:
            logger.error("  %s", e)
        sys.exit(1)

    logger.info("Pipeline complete!")
    logger.info("  matching_results.tsv --> %s", mr_path)
    logger.info("  candidate_pairs.tsv  --> %s", cp_path)


if __name__ == "__main__":
    main()
