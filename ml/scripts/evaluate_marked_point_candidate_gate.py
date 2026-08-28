"""Evaluate the frozen marked-point proposal union against reviewed truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from crrc_vision.assets import asset_root
from crrc_vision.marked_point import evaluate_candidate_recall


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="annotations/marked-point-v1")
    parser.add_argument(
        "--candidates", default="runs/marked-point-proposals-v1/union/candidates.json"
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--minimum-recall", type=float, default=0.99)
    parser.add_argument(
        "--output", default="runs/marked-point-proposals-v1/candidate-gate.json"
    )
    args = parser.parse_args()

    root = asset_root()
    dataset_root = _below(root, args.dataset)
    candidates_path = _below(root, args.candidates)
    formal_truth_path = _below(root, args.formal_truth)
    output_path = _below(root, args.output)
    if _sha256(formal_truth_path) != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    candidate_document = _load(candidates_path)
    candidate_rows = candidate_document.get("fused_candidates")
    if not isinstance(candidate_rows, list) or any(
        not isinstance(row, dict) for row in candidate_rows
    ):
        raise ValueError("INVALID_FUSED_CANDIDATES")

    partitions: dict[str, dict[str, object]] = {}
    total_truth = 0
    total_hits = 0
    misses: list[dict[str, object]] = []
    for partition in ("train", "val"):
        truth_path = dataset_root / f"instances.{partition}.json"
        report = evaluate_candidate_recall(
            _load(truth_path), candidate_rows, minimum_recall=args.minimum_recall
        )
        partitions[partition] = report
        total_truth += int(report["truth_boxes"])
        total_hits += int(report["true_positives"])
        misses.extend(
            {**row, "partition": partition} for row in report["misses"]  # type: ignore[arg-type]
        )
    recall = total_hits / total_truth if total_truth else 1.0
    output = {
        "schema_version": "marked-point-candidate-gate-v1",
        "minimum_recall": args.minimum_recall,
        "truth_boxes": total_truth,
        "true_positives": total_hits,
        "false_negatives": len(misses),
        "recall": recall,
        "passed": recall >= args.minimum_recall
        and all(bool(report["passed"]) for report in partitions.values()),
        "partitions": partitions,
        "misses": misses,
        "input_hashes": {
            "candidates_sha256": _sha256(candidates_path),
            "train_truth_sha256": _sha256(dataset_root / "instances.train.json"),
            "val_truth_sha256": _sha256(dataset_root / "instances.val.json"),
            "formal_truth_sha256": _sha256(formal_truth_path),
        },
        "old_sealed_test_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0 if output["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
