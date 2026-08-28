"""Strict review and COCO assembly contract for marked inspection points."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any


VALID_LABELS = {
    "marked_point",
    "unmarked_fastener",
    "lookalike",
    "uncertain",
}
TARGET_CATEGORY = {"id": 1, "name": "marked_point"}


def _object_rows(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def validate_review(document: Mapping[str, object]) -> list[str]:
    """Return deterministic contract errors without mutating the review."""

    errors: list[str] = []
    if document.get("schema_version") != "marked-point-review-v1":
        errors.append("INVALID_MARKED_POINT_REVIEW_SCHEMA")
    if document.get("partition") not in {"train", "val"}:
        errors.append("INVALID_MARKED_POINT_PARTITION")
    try:
        images = _object_rows(document.get("images"), "review images")
    except ValueError:
        return sorted(set(errors + ["INVALID_MARKED_POINT_REVIEW_IMAGES"]))

    for image in images:
        image_id = image.get("image_id")
        if image.get("image_status") != "complete":
            errors.append(f"IMAGE_NOT_COMPLETE:{image_id}")
        expected_value = image.get("expected_candidate_ids")
        expected = (
            set(expected_value)
            if isinstance(expected_value, list)
            and all(isinstance(value, str) and value for value in expected_value)
            else set()
        )
        try:
            decisions = _object_rows(
                image.get("candidate_decisions"), "candidate decisions"
            )
        except ValueError:
            errors.append(f"INVALID_CANDIDATE_DECISIONS:{image_id}")
            continue
        observed = [row.get("candidate_id") for row in decisions]
        if (
            set(observed) != expected
            or len(observed) != len(set(observed))
            or any(not isinstance(value, str) or not value for value in observed)
        ):
            errors.append(f"CANDIDATE_COVERAGE_MISMATCH:{image_id}")
        labels = [row.get("label") for row in decisions]
        if any(label not in VALID_LABELS for label in labels):
            errors.append(f"INVALID_MARKED_POINT_LABEL:{image_id}")
        if image.get("image_status") == "complete" and "uncertain" in labels:
            errors.append(f"UNCERTAIN_COMPLETE_CONFLICT:{image_id}")
        try:
            _object_rows(image.get("added_marked_points"), "added marked points")
        except ValueError:
            errors.append(f"INVALID_ADDED_MARKED_POINTS:{image_id}")
    return sorted(set(errors))


def _xyxy(value: object, *, width: int, height: int) -> tuple[float, float, float, float]:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and math.isfinite(float(coordinate))
            for coordinate in value
        )
    ):
        raise ValueError("INVALID_MARKED_POINT_BOX")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("INVALID_MARKED_POINT_BOX")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("BOX_OUT_OF_BOUNDS")
    return x1, y1, x2, y2


def assemble_marked_point_truth(
    review: Mapping[str, object],
    *,
    image_sizes: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    """Assemble positive marked points while retaining negative audit counts."""

    errors = validate_review(review)
    if errors:
        raise ValueError(f"MARKED_POINT_REVIEW_INVALID:{errors[0]}")
    images = _object_rows(review.get("images"), "review images")
    output_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    negative_counts: Counter[str] = Counter()
    seen_ids: set[object] = set()
    seen_paths: set[str] = set()
    seen_scenes: set[str] = set()

    for image in images:
        image_id = image.get("image_id")
        path = str(image.get("relative_path") or "")
        scene = str(image.get("scene_group") or "")
        source_sha256 = str(image.get("source_sha256") or "").upper()
        if image_id in seen_ids:
            raise ValueError(f"DUPLICATE_REVIEWED_IMAGE:{image_id}")
        if path in seen_paths:
            raise ValueError(f"DUPLICATE_REVIEWED_PATH:{path}")
        if scene in seen_scenes:
            raise ValueError(f"DUPLICATE_REVIEWED_SCENE:{scene}")
        if path not in image_sizes:
            raise ValueError(f"REVIEW_IMAGE_SIZE_MISSING:{path}")
        if not scene or len(source_sha256) != 64:
            raise ValueError(f"INVALID_REVIEWED_IDENTITY:{image_id}")
        width, height = image_sizes[path]
        if width <= 0 or height <= 0:
            raise ValueError(f"INVALID_REVIEW_IMAGE_SIZE:{path}")

        seen_ids.add(image_id)
        seen_paths.add(path)
        seen_scenes.add(scene)
        output_images.append(
            {
                "id": image_id,
                "file_name": path,
                "width": width,
                "height": height,
                "scene_group": scene,
                "sha256": source_sha256,
                "synthetic": False,
                "image_review_status": "complete",
            }
        )

        decisions = _object_rows(
            image.get("candidate_decisions"), "candidate decisions"
        )
        for decision in decisions:
            box = _xyxy(decision.get("xyxy"), width=width, height=height)
            label = str(decision["label"])
            if label != "marked_point":
                negative_counts[label] += 1
                continue
            annotations.append(
                _annotation(
                    annotation_id=len(annotations) + 1,
                    image_id=image_id,
                    box=box,
                    origin="candidate",
                    candidate_id=str(decision["candidate_id"]),
                )
            )
        for index, added in enumerate(
            _object_rows(image.get("added_marked_points"), "added marked points")
        ):
            box = _xyxy(added.get("xyxy"), width=width, height=height)
            annotations.append(
                _annotation(
                    annotation_id=len(annotations) + 1,
                    image_id=image_id,
                    box=box,
                    origin="added",
                    candidate_id=f"added-{image_id}-{index + 1}",
                )
            )

    return {
        "info": {
            "schema_version": "marked-point-coco-v1",
            "partition": review["partition"],
            "negative_counts": {
                label: int(negative_counts[label])
                for label in ("unmarked_fastener", "lookalike", "uncertain")
            },
        },
        "images": sorted(output_images, key=lambda row: int(row["id"])),
        "annotations": annotations,
        "categories": [TARGET_CATEGORY],
    }


def _annotation(
    *,
    annotation_id: int,
    image_id: object,
    box: tuple[float, float, float, float],
    origin: str,
    candidate_id: str,
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x1, y1, width, height],
        "area": width * height,
        "iscrowd": 0,
        "origin": origin,
        "candidate_id": candidate_id,
    }
