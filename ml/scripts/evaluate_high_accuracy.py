"""Select a validation threshold or open the sealed accuracy gate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from crrc_vision.high_accuracy_gate import evaluate_at_threshold, select_threshold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _predictions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("predictions")
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"PREDICTIONS_INVALID:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("val", "sealed-test"), required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    for path in (args.truth, args.predictions, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{args.output}")

    if args.mode == "val":
        truth = _object(args.truth)
        if truth.get("info", {}).get("partition") == "sealed_test":
            raise ValueError("VALIDATION_CANNOT_READ_SEALED_TEST")
        predictions = _predictions(args.predictions)
        threshold = select_threshold(predictions, truth, minimum_precision=0.90)
        report = evaluate_at_threshold(
            predictions, truth, threshold=threshold
        )
        result = {
            "schema_version": "high-accuracy-selection-v1",
            "mode": "val",
            "threshold": threshold,
            "iou_threshold": 0.50,
            "report": asdict(report),
            "model_sha256": _sha256(args.model),
            "prediction_sha256": _sha256(args.predictions),
            "validation_sha256": _sha256(args.truth),
            "sealed_test_opened": False,
        }
        _atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.selection_manifest is None or args.audit is None:
        raise ValueError("SEALED_TEST_REQUIRES_SELECTION_AND_AUDIT")
    if not args.selection_manifest.is_file():
        raise FileNotFoundError(args.selection_manifest)
    if args.audit.exists():
        prior = _object(args.audit)
        if prior.get("sealed_test_opened") is True:
            raise RuntimeError("SEALED_TEST_ALREADY_OPENED")
        raise FileExistsError(f"AUDIT_ALREADY_EXISTS:{args.audit}")
    selection = _object(args.selection_manifest)
    if selection.get("schema_version") != "high-accuracy-selection-v1":
        raise ValueError("INVALID_SELECTION_MANIFEST")
    model_hash = _sha256(args.model)
    if model_hash != selection.get("model_sha256"):
        raise ValueError("SEALED_MODEL_DIFFERS_FROM_SELECTED_MODEL")
    audit: dict[str, object] = {
        "schema_version": "high-accuracy-sealed-audit-v1",
        "sealed_test_opened": True,
        "evaluation_status": "opened",
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "model_sha256": model_hash,
    }
    # This write deliberately occurs before loading predictions or test annotations.
    _atomic_json(args.audit, audit)
    try:
        predictions = _predictions(args.predictions)
        truth = _object(args.truth)
        if truth.get("info", {}).get("partition") != "sealed_test":
            raise ValueError("SEALED_TEST_PARTITION_REQUIRED")
        report = evaluate_at_threshold(
            predictions,
            truth,
            threshold=float(selection["threshold"]),
            enforce_sealed_minimum=True,
        )
        result = {
            **audit,
            "evaluation_status": "complete",
            "passed": report.passed,
            "report": asdict(report),
            "prediction_sha256": _sha256(args.predictions),
            "sealed_test_sha256": _sha256(args.truth),
        }
        _atomic_json(args.output, result)
        _atomic_json(args.audit, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if report.passed else 2
    except BaseException as exc:
        audit["evaluation_status"] = "failed"
        audit["error_type"] = type(exc).__name__
        _atomic_json(args.audit, audit)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
