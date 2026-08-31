"""Evaluate complementary E1 proposal and ROI-verifier score thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import select_dual_pipeline_thresholds


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="runs/marked-point-verifier-e3/dataset-v2/manifest.json"
    )
    parser.add_argument(
        "--predictions",
        default="runs/marked-point-verifier-e3/mobilenetv3-small/best-val-predictions.json",
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output", default="runs/marked-point-verifier-e3/mobilenetv3-small/fusion-report.json"
    )
    args = parser.parse_args()

    root = asset_root().resolve()
    dataset_path = (root / args.dataset).resolve()
    predictions_path = (root / args.predictions).resolve()
    formal_truth = (root / args.formal_truth).resolve()
    output = (root / args.output).resolve()
    if _sha256(formal_truth) != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    original = {
        int(row["prediction_index"]): row
        for row in manifest["examples"]
        if row["split"] == "val"
    }
    if len(original) != manifest["counts"]["val_marked_point"] + manifest["counts"]["val_not_marked_point"]:
        raise RuntimeError("VERIFIER_VAL_INDEX_COLLISION")
    scored = []
    for row in predictions:
        source = original[int(row["prediction_index"])]
        scored.append(
            {
                "prediction_index": row["prediction_index"],
                "image_id": row["image_id"],
                "truth_ids": row["truth_ids"],
                "verifier_score": row["score"],
                "proposal_score": source["score"],
            }
        )
    image_count = len({row["image_id"] for row in scored})
    report = select_dual_pipeline_thresholds(scored, image_count=image_count)
    selected = [
        row
        for row in scored
        if row["verifier_score"] >= report.verifier_threshold
        or row["proposal_score"] >= report.proposal_threshold
    ]
    result = {
        "schema_version": "marked-point-verifier-fusion-v1",
        "selection": "verifier_score_gte_threshold_or_e1_score_gte_threshold",
        "report": asdict(report),
        "relevant_selected": sum(bool(row["truth_ids"]) for row in selected),
        "irrelevant_selected": sum(not row["truth_ids"] for row in selected),
        "dataset_sha256": _sha256(dataset_path),
        "predictions_sha256": _sha256(predictions_path),
        "formal_truth_sha256": _sha256(formal_truth),
        "sealed_test_opened": False,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
