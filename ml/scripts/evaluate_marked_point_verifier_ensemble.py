"""Evaluate fixed, unweighted multi-seed verifier ensembles on E1 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import (
    combine_verifier_predictions,
    select_dual_pipeline_thresholds,
    suppress_overlapping_candidates,
)


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "runs/marked-point-verifier-e4/multiseed/seed-20260828/e1-candidate-predictions.json",
            "runs/marked-point-verifier-e4/multiseed/seed-20260829/e1-candidate-predictions.json",
            "runs/marked-point-verifier-e4/multiseed/seed-20260830/e1-candidate-predictions.json",
        ],
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output",
        default="runs/marked-point-verifier-e4/multiseed/ensemble-report.json",
    )
    parser.add_argument("--nms-iou", type=float, default=0.3)
    args = parser.parse_args()

    root = asset_root().resolve()
    input_paths = [(root / value).resolve() for value in args.inputs]
    formal_truth = (root / args.formal_truth).resolve()
    output = (root / args.output).resolve()
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    if len(documents) < 2:
        raise ValueError("VERIFIER_ENSEMBLE_REQUIRES_MULTIPLE_MODELS")
    if any(document.get("sealed_test_opened") is not False for document in documents):
        raise RuntimeError("SEALED_TEST_STATE_INVALID")
    if any(document.get("formal_truth_sha256") != FORMAL_TRUTH_SHA256 for document in documents):
        raise RuntimeError("VERIFIER_ENSEMBLE_TRUTH_HASH_MISMATCH")

    reports = {}
    for method in ("mean", "geometric_mean"):
        combined = combine_verifier_predictions(
            [document["predictions"] for document in documents], method=method
        )
        scored = [{**row, "verifier_score": row["score"]} for row in combined]
        image_ids = sorted({row["image_id"] for row in combined})
        dual = select_dual_pipeline_thresholds(scored, image_count=len(image_ids))
        selected = suppress_overlapping_candidates(
            combined,
            verifier_threshold=dual.verifier_threshold,
            proposal_threshold=dual.proposal_threshold,
            iou_threshold=args.nms_iou,
        )
        truth_ids = {truth_id for row in combined for truth_id in row["truth_ids"]}
        covered = {truth_id for row in selected for truth_id in row["truth_ids"]}
        per_image = [
            sum(row["image_id"] == image_id for row in selected)
            for image_id in image_ids
        ]
        reports[method] = {
            "dual_report": asdict(dual),
            "nms_iou_threshold": args.nms_iou,
            "truth_recall": len(covered) / len(truth_ids),
            "covered_truth": len(covered),
            "total_truth": len(truth_ids),
            "selected": len(selected),
            "candidates_per_image": len(selected) / len(image_ids),
            "median_candidates_per_image": statistics.median(per_image),
            "maximum_candidates_in_one_image": max(per_image),
            "passed_recall": covered == truth_ids,
            "passed_burden": len(selected) / len(image_ids) <= 20.0,
            "passed": covered == truth_ids and len(selected) / len(image_ids) <= 20.0,
        }
    result = {
        "schema_version": "marked-point-verifier-ensemble-v1",
        "policy": "fixed_equal_weight_no_val_fitted_weights",
        "input_sha256": {
            str(path): _sha256(path) for path in input_paths
        },
        "checkpoint_sha256": [document["checkpoint_sha256"] for document in documents],
        "individual_reports": [
            {
                "checkpoint_sha256": document["checkpoint_sha256"],
                "post_nms_report": document["post_nms_report"],
            }
            for document in documents
        ],
        "single_model_all_passed": all(
            document["post_nms_report"]["passed"] for document in documents
        ),
        "single_model_worst_candidates_per_image": max(
            document["post_nms_report"]["candidates_per_image"]
            for document in documents
        ),
        "formal_truth_sha256": _sha256(formal_truth),
        "sealed_test_opened": False,
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
