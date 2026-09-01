"""Capture detailed phone benchmark boxes and evaluate marked-point coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from crrc_vision.android_benchmark_log import (
    parse_box_line,
    parse_complete_line,
    parse_summary_line,
)
from crrc_vision.marked_point_model_gate import is_proposal_match


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-token",
        required=True,
        help="Unique token passed to the benchmark Activity for this invocation",
    )
    args = parser.parse_args()

    if args.serial != "TPC7N18604005991":
        raise ValueError("ANDROID_BENCHMARK_PHONE_SERIAL_REQUIRED")
    truth_hash_before = _sha256(args.formal_truth)
    if truth_hash_before != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    completed = subprocess.run(
        [
            str(args.adb),
            "-s",
            args.serial,
            "logcat",
            "-d",
            "-s",
            "DetectorBenchmark:I",
            "*:S",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = completed.stdout.splitlines()
    parsed = [parse_box_line(line) for line in lines]
    boxes = [
        row
        for row in parsed
        if row is not None and row["run_token"] == args.run_token
    ]
    summaries = [parse_summary_line(line) for line in lines]
    summaries = [
        row
        for row in summaries
        if row is not None and row["run_token"] == args.run_token
    ]
    completions = [parse_complete_line(line) for line in lines]
    completions = [
        row
        for row in completions
        if row is not None and row["run_token"] == args.run_token
    ]
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    image_by_name = {Path(row["file_name"]).name: row for row in truth["images"]}
    predictions: list[dict[str, object]] = []
    for row in boxes:
        image = image_by_name.get(str(row["file_name"]))
        if image is None:
            raise RuntimeError(f"ANDROID_BENCHMARK_UNKNOWN_IMAGE:{row['file_name']}")
        predictions.append(
            {
                "image_id": image["id"],
                "category_id": 1,
                "bbox": row["bbox"],
                "score": row["score"],
                "phone_index": row["index"],
            }
        )
    expected_images = {row["id"] for row in truth["images"]}
    expected_names = set(image_by_name)
    summary_names = [str(row["file_name"]) for row in summaries]
    summary_counts = Counter(summary_names)
    contract_errors: list[str] = []
    if set(summary_names) != expected_names:
        contract_errors.append("SUMMARY_IMAGE_SET_MISMATCH")
    duplicate_summaries = sorted(
        name for name, count in summary_counts.items() if count != 1
    )
    if duplicate_summaries:
        contract_errors.append(
            "SUMMARY_COUNT_NOT_ONE:" + ",".join(duplicate_summaries)
        )
    boxes_by_name: dict[str, list[dict[str, object]]] = {
        name: [] for name in expected_names
    }
    for row in boxes:
        file_name = str(row["file_name"])
        if file_name not in boxes_by_name:
            contract_errors.append(f"BOX_UNKNOWN_IMAGE:{file_name}")
            continue
        boxes_by_name[file_name].append(row)
    for name in sorted(expected_names):
        image_summaries = [row for row in summaries if row["file_name"] == name]
        if len(image_summaries) != 1:
            continue
        image_boxes = boxes_by_name[name]
        expected_detection_count = int(image_summaries[0]["detections"])
        if len(image_boxes) != expected_detection_count:
            contract_errors.append(f"BOX_COUNT_MISMATCH:{name}")
        indices = sorted(int(row["index"]) for row in image_boxes)
        if indices != list(range(len(image_boxes))):
            contract_errors.append(f"BOX_INDEX_SET_MISMATCH:{name}")
    if len(completions) != 1:
        contract_errors.append("COMPLETE_MARKER_COUNT_NOT_ONE")
    elif int(completions[0]["images"]) != len(expected_names):
        contract_errors.append("COMPLETE_IMAGE_COUNT_MISMATCH")
    observed_images = {
        image_by_name[name]["id"] for name in set(summary_names) if name in image_by_name
    }
    annotations = truth["annotations"]
    covered_ids = {
        annotation["id"]
        for annotation in annotations
        if any(
            prediction["image_id"] == annotation["image_id"]
            and is_proposal_match(
                tuple(float(value) for value in prediction["bbox"]),
                tuple(float(value) for value in annotation["bbox"]),
            )
            for prediction in predictions
        )
    }
    expected_ids = {annotation["id"] for annotation in annotations}
    timing_report = {}
    for key in ("total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"):
        values = sorted(float(row[key]) for row in summaries)
        if values:
            timing_report[key] = {
                "p50": statistics.median(values),
                "p95": values[max(math.ceil(len(values) * 0.95) - 1, 0)],
                "mean": statistics.fmean(values),
            }
    report = {
        "schema_version": "android-marked-point-benchmark-gate-v2",
        "serial": args.serial,
        "run_token": args.run_token,
        "image_count": len(observed_images),
        "expected_image_count": len(expected_images),
        "detection_count": len(predictions),
        "covered_truth": len(covered_ids),
        "total_truth": len(expected_ids),
        "truth_recall": len(covered_ids) / len(expected_ids),
        "uncovered_truth_ids": sorted(expected_ids - covered_ids),
        "run_contract_errors": contract_errors,
        "timings": timing_report,
        "formal_truth_sha256_before": truth_hash_before,
        "formal_truth_sha256_after": _sha256(args.formal_truth),
        "passed": (
            not contract_errors
            and observed_images == expected_images
            and covered_ids == expected_ids
        ),
    }
    payload = {"report": report, "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["formal_truth_sha256_after"] != truth_hash_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
