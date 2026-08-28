"""Build a Git-external validation-only FP/FN diagnostic review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from crrc_vision.error_buckets import (
    DetectionError,
    ErrorEvidence,
    classify_error,
    detection_errors,
    threshold_from_selection,
    validate_diagnostic_truth,
)


SEALED_TRUTH_SHA256 = "63B233BFEF9C85C19039CF713A83CEBA4CB653F5996BCA06220C12EF4EA0D6E4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + box_width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + box_height))))
    return x1, y1, x2, y2


def _nearby_density(
    error: DetectionError,
    annotations: list[dict[str, object]],
    width: int,
    height: int,
) -> int:
    x, y, box_width, box_height = error.bbox
    center = np.array((x + box_width / 2, y + box_height / 2))
    radius = max(4.0 * np.hypot(box_width, box_height), 0.08 * np.hypot(width, height))
    count = 0
    for annotation in annotations:
        ax, ay, aw, ah = (float(value) for value in annotation["bbox"])
        other = np.array((ax + aw / 2, ay + ah / 2))
        if np.linalg.norm(center - other) <= radius:
            count += 1
    return max(0, count - (1 if error.truth_id is not None else 0))


def _evidence(
    error: DetectionError,
    image: np.ndarray,
    annotations: list[dict[str, object]],
    truth_by_id: dict[object, dict[str, object]],
) -> ErrorEvidence:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = _clip(error.bbox, width, height)
    crop = gray[y1:y2, x1:x2]
    annotation = truth_by_id.get(error.truth_id, {})
    reasons = " ".join(str(value).lower() for value in annotation.get("reasons", []))
    border_distance = min(x1, y1, width - x2, height - y2) / max(1, min(width, height))
    bright_fraction = float(np.mean(crop >= 245)) if crop.size else 0.0
    return ErrorEvidence(
        area_ratio=(error.bbox[2] * error.bbox[3]) / (width * height),
        border_distance_ratio=max(0.0, border_distance),
        brightness=float(np.mean(gray)),
        focus_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        local_contrast=float(np.std(crop)) if crop.size else 0.0,
        nearby_density=_nearby_density(error, annotations, width, height),
        annotation_dispute=bool(
            annotation.get("review_status") == "uncertain" or "dispute" in reasons
        ),
        occluded=bool(annotation.get("occluded") or "occlu" in reasons or "遮挡" in reasons),
        reflection=bright_fraction >= 0.20 and float(np.std(crop)) >= 45.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--threshold", type=float)
    threshold_group.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.truth, args.predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if any("sealed" in part.lower() for part in args.truth.parts):
        raise ValueError("SEALED_TRUTH_FORBIDDEN")
    if args.output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{args.output}")

    truth_hash = _sha256(args.truth)
    truth = _json(args.truth)
    if not isinstance(truth, dict):
        raise ValueError("TRUTH_ROOT_NOT_OBJECT")
    validate_diagnostic_truth(
        args.truth,
        truth,
        truth_sha256=truth_hash,
        forbidden_truth_hashes={SEALED_TRUTH_SHA256},
    )
    prediction_hash = _sha256(args.predictions)
    predictions = _json(args.predictions)
    if not isinstance(predictions, list) or any(
        not isinstance(row, dict) for row in predictions
    ):
        raise ValueError("PREDICTIONS_INVALID")

    if args.selection_manifest is not None:
        if not args.selection_manifest.is_file():
            raise FileNotFoundError(args.selection_manifest)
        selection = _json(args.selection_manifest)
        if not isinstance(selection, dict):
            raise ValueError("SELECTION_ROOT_NOT_OBJECT")
        threshold = threshold_from_selection(
            selection, prediction_sha256=prediction_hash
        )
    else:
        threshold = float(args.threshold)
    errors = detection_errors(predictions, truth, threshold=threshold)
    images = {row["id"]: row for row in truth["images"]}
    annotations_by_image: dict[object, list[dict[str, object]]] = defaultdict(list)
    truth_by_id: dict[object, dict[str, object]] = {}
    for annotation in truth["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
        truth_by_id[annotation["id"]] = annotation

    args.output.mkdir(parents=True)
    overlay_root = args.output / "overlays"
    crop_root = args.output / "crops"
    overlay_root.mkdir()
    crop_root.mkdir()
    records: list[dict[str, object]] = []
    by_image: dict[object, list[tuple[DetectionError, str]]] = defaultdict(list)
    cache: dict[object, np.ndarray] = {}
    verified_hashes: dict[object, str] = {}
    for error in sorted(
        errors,
        key=lambda row: (int(row.image_id), row.kind, row.truth_id or -1, row.prediction_index or -1),
    ):
        image_row = images[error.image_id]
        image_path = args.source_root / str(image_row["file_name"])
        if error.image_id not in cache:
            source_hash = _sha256(image_path)
            expected_hash = str(image_row.get("sha256", "")).upper()
            if expected_hash and source_hash != expected_hash:
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            cache[error.image_id] = image
            verified_hashes[error.image_id] = source_hash
        image = cache[error.image_id]
        evidence = _evidence(
            error,
            image,
            annotations_by_image[error.image_id],
            truth_by_id,
        )
        primary, secondary = classify_error(evidence)
        record_id = f"{int(error.image_id):06d}-{error.kind}-{len(records):04d}"
        x1, y1, x2, y2 = _clip(error.bbox, image.shape[1], image.shape[0])
        margin = max(8, int(0.25 * max(x2 - x1, y2 - y1)))
        crop = image[
            max(0, y1 - margin) : min(image.shape[0], y2 + margin),
            max(0, x1 - margin) : min(image.shape[1], x2 + margin),
        ]
        crop_path = crop_root / f"{record_id}.jpg"
        if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"CROP_WRITE_FAILED:{crop_path}")
        records.append(
            {
                "record_id": record_id,
                "image_id": error.image_id,
                "scene_group": image_row.get("scene_group"),
                "relative_path": image_row.get("file_name"),
                "truth_id": error.truth_id,
                "prediction_index": error.prediction_index,
                "error_kind": error.kind,
                "primary_bucket": primary,
                "secondary_tags": list(secondary),
                "bbox_xywh": list(error.bbox),
                "score": error.score,
                "evidence": evidence.__dict__,
                "source_sha256": verified_hashes[error.image_id],
                "crop": str(crop_path.relative_to(args.output)).replace("\\", "/"),
            }
        )
        by_image[error.image_id].append((error, primary))

    for image_id, rows in by_image.items():
        canvas = cache[image_id].copy()
        for error, primary in rows:
            x1, y1, x2, y2 = _clip(error.bbox, canvas.shape[1], canvas.shape[0])
            color = (0, 0, 255) if error.kind == "false_negative" else (255, 0, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                canvas,
                f"{error.kind}:{primary}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        overlay_path = overlay_root / f"{int(image_id):06d}.jpg"
        if not cv2.imwrite(str(overlay_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"OVERLAY_WRITE_FAILED:{overlay_path}")

    result = {
        "schema_version": "validation-error-review-v1",
        "partition": "val",
        "threshold": threshold,
        "iou_threshold": 0.50,
        "truth_sha256": truth_hash,
        "prediction_sha256": prediction_hash,
        "sealed_test_opened": False,
        "summary": {
            "errors": len(records),
            "by_kind": dict(Counter(row["error_kind"] for row in records)),
            "by_primary_bucket": dict(
                Counter(row["primary_bucket"] for row in records)
            ),
            "images_with_errors": len(by_image),
        },
        "records": records,
    }
    (args.output / "errors.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
