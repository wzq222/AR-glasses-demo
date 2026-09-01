"""Compare two frozen mobile-runtime prediction files and write a guarded report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.mobile_benchmark import build_parity_report, sha256_file


EXPECTED_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _load_predictions(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("PREDICTION_LIST_REQUIRED")
    return document


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--iou", type=float, default=0.95)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    args = parser.parse_args()

    root = asset_root().resolve()
    baseline_path = _below(root, args.baseline)
    candidate_path = _below(root, args.candidate)
    truth_path = _below(root, args.truth)
    run = _below(root, args.run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("PARITY_RUN_NOT_EMPTY")
    run.mkdir(parents=True, exist_ok=True)
    truth_before = sha256_file(truth_path)
    if truth_before != EXPECTED_TRUTH_SHA256:
        raise ValueError("FORMAL_TRUTH_HASH_MISMATCH")

    report = build_parity_report(
        _load_predictions(baseline_path),
        _load_predictions(candidate_path),
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        iou_threshold=args.iou,
    )
    truth_after = sha256_file(truth_path)
    report["formal_truth_sha256_before"] = truth_before
    report["formal_truth_sha256_after"] = truth_after
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    _atomic_json(run / "parity-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "parity_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
