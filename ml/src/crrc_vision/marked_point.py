"""Strict review and COCO assembly contract for marked inspection points."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any


VALID_LABELS = {
    "marked_point",
    "covered_by_added_marked_point",
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
    covered_candidate_count = 0
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
            label = str(decision["label"])
            box_value = decision.get("xyxy")
            if label == "marked_point" and decision.get("geometry_adjusted") is True:
                box_value = _accepted_second_pass_box(decision)
            box = _xyxy(box_value, width=width, height=height)
            if label == "covered_by_added_marked_point":
                covered_candidate_count += 1
                continue
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
            box = _xyxy(
                _accepted_second_pass_box(added), width=width, height=height
            )
            annotations.append(
                _annotation(
                    annotation_id=len(annotations) + 1,
                    image_id=image_id,
                    box=box,
                    origin="added",
                    candidate_id=str(
                        added.get("positive_id") or f"added-{image_id}-{index + 1}"
                    ),
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
            "covered_candidate_count": covered_candidate_count,
        },
        "images": sorted(output_images, key=lambda row: int(row["id"])),
        "annotations": annotations,
        "categories": [TARGET_CATEGORY],
    }


def _accepted_second_pass_box(row: Mapping[str, object]) -> object:
    second_pass = row.get("second_pass")
    if not isinstance(second_pass, Mapping):
        raise ValueError("SECOND_PASS_REQUIRED")
    if second_pass.get("first_result_hidden") is not True:
        raise ValueError("SECOND_PASS_MUST_HIDE_FIRST_RESULT")
    if second_pass.get("decision") != "accept":
        raise ValueError("SECOND_PASS_NOT_ACCEPTED")
    return second_pass.get("final_xyxy", row.get("xyxy"))


def assemble_partition(
    review: Mapping[str, object],
    *,
    selection: Mapping[str, object],
    partition: str,
    image_sizes: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    """Bind a reviewed partition to the frozen selected identities."""

    if partition not in {"train", "val"} or review.get("partition") != partition:
        raise ValueError("REVIEW_PARTITION_MISMATCH")
    if selection.get("old_sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID")
    partition_rows: dict[str, dict[str, object]] = {}
    owners: dict[tuple[str, object], str] = {}
    for split in ("train", "val"):
        rows = _object_rows(selection.get(split), f"selection {split}")
        for source in rows:
            for field in ("scene_group", "relative_path", "sha256", "image_id"):
                value = source.get(field)
                key = (field, value)
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"SELECTION_SPLIT_LEAKAGE:{field}")
            if split == partition:
                path = str(source.get("relative_path") or "")
                partition_rows[path] = source

    forbidden = selection.get("forbidden_old_sealed")
    if not isinstance(forbidden, Mapping):
        raise ValueError("FORBIDDEN_IDENTITIES_MISSING")
    forbidden_paths = {str(value) for value in forbidden.get("paths", [])}
    forbidden_hashes = {
        str(value).upper() for value in forbidden.get("sha256", [])
    }
    for image in _object_rows(review.get("images"), "review images"):
        path = str(image.get("relative_path") or "")
        digest = str(image.get("source_sha256") or "").upper()
        if path in forbidden_paths or digest in forbidden_hashes:
            raise ValueError(f"OLD_SEALED_IMAGE_FORBIDDEN:{path}")
        selected = partition_rows.get(path)
        if selected is None:
            raise ValueError(f"REVIEW_IMAGE_OUTSIDE_PARTITION:{path}")
        if (
            image.get("image_id") != selected.get("image_id")
            or str(image.get("scene_group") or "")
            != str(selected.get("scene_group") or "")
            or digest != str(selected.get("sha256") or "").upper()
        ):
            raise ValueError(f"SELECTION_IDENTITY_MISMATCH:{path}")
    return assemble_marked_point_truth(review, image_sizes=image_sizes)


def _box_metrics(
    first: object, second: object
) -> tuple[float, float, bool, bool]:
    a = tuple(float(value) for value in first)  # type: ignore[arg-type]
    b = tuple(float(value) for value in second)  # type: ignore[arg-type]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    a_center_in_b = bx1 <= center_a[0] <= bx2 and by1 <= center_a[1] <= by2
    b_center_in_a = ax1 <= center_b[0] <= ax2 and ay1 <= center_b[1] <= ay2
    return iou, containment, a_center_in_b, b_center_in_a


def deduplicate_positive_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Suppress spatial duplicates without merging adjacent inspection points."""

    kept: list[dict[str, object]] = []
    suppressed: list[dict[str, str]] = []
    ordered = sorted(
        records,
        key=lambda row: (
            int(row.get("source_rank", 99)),
            str(row.get("relative_path") or ""),
            str(row.get("positive_id") or ""),
        ),
    )
    for row in ordered:
        duplicate_of: dict[str, object] | None = None
        for previous in kept:
            if previous.get("relative_path") != row.get("relative_path"):
                continue
            iou, containment, center_a, center_b = _box_metrics(
                row.get("dedupe_xyxy", row.get("xyxy")),
                previous.get("dedupe_xyxy", previous.get("xyxy")),
            )
            if iou >= 0.5 or (containment >= 0.6 and center_a and center_b):
                duplicate_of = previous
                break
        if duplicate_of is None:
            kept.append(row)
            continue
        suppressed.append(
            {
                "positive_id": str(row.get("positive_id") or ""),
                "kept_positive_id": str(duplicate_of.get("positive_id") or ""),
            }
        )
    return kept, suppressed


def filter_then_deduplicate_positive_records(
    records: list[dict[str, object]], rejected_ids: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Remove rejected proposals before spatial deduplication.

    A rejected high-priority proposal must not suppress a lower-priority proposal
    added by the independent full-image audit for the same physical point.
    """

    filtered = [
        row
        for row in records
        if str(row.get("positive_id") or "") not in rejected_ids
    ]
    return deduplicate_positive_records(filtered)


def unreviewed_positive_ids(
    kept_ids: set[str], accepted_ids: set[str], rejected_ids: set[str]
) -> set[str]:
    """Return only retained positives that still lack a second-pass decision."""

    return kept_ids - accepted_ids - rejected_ids


def _candidate_covered(candidate_box: object, positive_box: object) -> bool:
    iou, containment, candidate_center_in_positive, positive_center_in_candidate = (
        _box_metrics(candidate_box, positive_box)
    )
    return (
        iou >= 0.15
        or containment >= 0.5
        or candidate_center_in_positive
        or positive_center_in_candidate
    )


def build_manual_positive_records(
    selection: Mapping[str, object], manual_additions: object
) -> list[dict[str, object]]:
    """Bind full-image audit additions to frozen selected-image identities."""

    selected_by_path: dict[str, dict[str, object]] = {}
    for partition in ("train", "val"):
        for row in _object_rows(selection.get(partition), f"selection {partition}"):
            path = str(row.get("relative_path") or "")
            if not path or path in selected_by_path:
                raise ValueError(f"INVALID_SELECTED_IMAGE_IDENTITY:{path}")
            selected_by_path[path] = row

    additions = _object_rows(manual_additions, "manual additions")
    seen_ids: set[str] = set()
    output: list[dict[str, object]] = []
    for row in additions:
        manual_id = str(row.get("manual_id") or "")
        if not manual_id or manual_id in seen_ids:
            raise ValueError(f"INVALID_MANUAL_ID:{manual_id}")
        seen_ids.add(manual_id)
        path = str(row.get("relative_path") or "")
        selected = selected_by_path.get(path)
        if selected is None:
            raise ValueError(f"MANUAL_IMAGE_NOT_SELECTED:{path}")
        box = row.get("xyxy")
        if not (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in box
            )
        ):
            raise ValueError(f"INVALID_MANUAL_BOX:{manual_id}")
        xyxy = [float(value) for value in box]
        if xyxy[0] < 0 or xyxy[1] < 0 or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            raise ValueError(f"INVALID_MANUAL_BOX:{manual_id}")
        colors = row.get("mark_colors", [])
        if not isinstance(colors, list) or any(
            not isinstance(value, str) or not value for value in colors
        ):
            raise ValueError(f"INVALID_MANUAL_MARK_COLORS:{manual_id}")
        output.append(
            {
                "positive_id": f"manual-{manual_id}",
                "source_rank": 3,
                "first_pass_source": "manual",
                "first_pass_shortlist_id": manual_id,
                "relative_path": path,
                "image_id": selected.get("image_id"),
                "source_sha256": selected.get("sha256"),
                "xyxy": xyxy,
                "dedupe_xyxy": xyxy,
                "mark_colors": list(colors),
                "audit_reason": str(row.get("audit_reason") or "full_image_miss"),
            }
        )
    return output


def build_review_set(
    *,
    selection: Mapping[str, object],
    candidates: list[dict[str, object]],
    positives: list[dict[str, object]],
    uncertain_paths: Mapping[str, str],
) -> dict[str, object]:
    """Build complete train/val reviews from frozen identities and audited positives."""

    candidates_by_path: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        path = str(candidate.get("relative_path") or "")
        candidates_by_path.setdefault(path, []).append(candidate)
    positives_by_path: dict[str, list[dict[str, object]]] = {}
    for positive in positives:
        path = str(positive.get("relative_path") or "")
        positives_by_path.setdefault(path, []).append(positive)

    reviews: dict[str, dict[str, object]] = {}
    exclusions: list[dict[str, object]] = []
    for partition in ("train", "val"):
        images: list[dict[str, object]] = []
        for selected in _object_rows(selection.get(partition), f"selection {partition}"):
            path = str(selected.get("relative_path") or "")
            if path in uncertain_paths:
                exclusions.append(
                    {
                        "image_id": selected.get("image_id"),
                        "relative_path": path,
                        "scene_group": selected.get("scene_group"),
                        "source_sha256": selected.get("sha256"),
                        "reason": uncertain_paths[path],
                    }
                )
                continue
            image_candidates = sorted(
                candidates_by_path.get(path, []), key=lambda row: str(row.get("id"))
            )
            image_positives = sorted(
                positives_by_path.get(path, []),
                key=lambda row: str(row.get("positive_id")),
            )
            decisions: list[dict[str, object]] = []
            for candidate in image_candidates:
                covered = any(
                    _candidate_covered(candidate.get("xyxy"), positive.get("xyxy"))
                    for positive in image_positives
                )
                sources = candidate.get("sources")
                source_names = set(sources) if isinstance(sources, list) else set()
                label = (
                    "covered_by_added_marked_point"
                    if covered
                    else "unmarked_fastener"
                    if "fastener_v2_2" in source_names
                    else "lookalike"
                )
                decisions.append(
                    {
                        "candidate_id": candidate.get("id"),
                        "label": label,
                        "xyxy": candidate.get("xyxy"),
                        "sources": sorted(str(value) for value in source_names),
                    }
                )
            images.append(
                {
                    "image_id": selected.get("image_id"),
                    "relative_path": path,
                    "scene_group": selected.get("scene_group"),
                    "source_sha256": selected.get("sha256"),
                    "image_status": "complete",
                    "expected_candidate_ids": [
                        str(row.get("id")) for row in image_candidates
                    ],
                    "candidate_decisions": decisions,
                    "added_marked_points": image_positives,
                }
            )
        reviews[partition] = {
            "schema_version": "marked-point-review-v1",
            "partition": partition,
            "images": images,
        }
    return {
        "schema_version": "marked-point-review-set-v1",
        "reviews": reviews,
        "uncertain_exclusions": sorted(
            exclusions, key=lambda row: str(row["relative_path"])
        ),
    }


def evaluate_candidate_recall(
    truth: Mapping[str, object],
    candidates: list[dict[str, object]],
    *,
    minimum_recall: float = 0.99,
) -> dict[str, object]:
    """Measure lossless proposal coverage using mark-center or majority-GT overlap."""

    images = _object_rows(truth.get("images"), "truth images")
    annotations = _object_rows(truth.get("annotations"), "truth annotations")
    paths = {row.get("id"): str(row.get("file_name") or "") for row in images}
    by_path: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        by_path.setdefault(str(candidate.get("relative_path") or ""), []).append(
            candidate
        )
    hits = 0
    misses: list[dict[str, object]] = []
    source_hits: Counter[str] = Counter()
    for annotation in annotations:
        path = paths.get(annotation.get("image_id"), "")
        x, y, width, height = (
            float(value) for value in annotation.get("bbox", [])  # type: ignore[arg-type]
        )
        truth_box = (x, y, x + width, y + height)
        matched: list[dict[str, object]] = []
        for candidate in by_path.get(path, []):
            iou, containment, candidate_center_in_truth, _ = _box_metrics(
                candidate.get("xyxy"), truth_box
            )
            if candidate_center_in_truth or iou >= 0.1 or containment >= 0.5:
                matched.append(candidate)
        if not matched:
            misses.append(
                {
                    "annotation_id": annotation.get("id"),
                    "image_id": annotation.get("image_id"),
                    "relative_path": path,
                    "bbox": annotation.get("bbox"),
                }
            )
            continue
        hits += 1
        for source in {
            str(value)
            for candidate in matched
            for value in (
                candidate.get("sources")
                if isinstance(candidate.get("sources"), list)
                else []
            )
        }:
            source_hits[source] += 1
    total = len(annotations)
    recall = hits / total if total else 1.0
    return {
        "truth_boxes": total,
        "true_positives": hits,
        "false_negatives": len(misses),
        "recall": recall,
        "minimum_recall": minimum_recall,
        "passed": recall >= minimum_recall,
        "source_truth_hits": dict(sorted(source_hits.items())),
        "misses": misses,
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
