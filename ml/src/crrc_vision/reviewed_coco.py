"""Assemble final, whole-image-complete Codex decisions into isolated COCO."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crrc_vision.auto_labeling import VALID_CATEGORIES
from crrc_vision.codex_review import validate_first_pass_review


CATEGORY_IDS = {"fastener": 1, "pipe_joint": 2}
FINAL_PROPOSAL_DECISIONS = {"accept", "reject", "uncertain"}


@dataclass(frozen=True)
class ReviewedCocoResult:
    document: dict[str, object]
    uncertain_image_ids: tuple[int, ...]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_reviewed_coco(result: ReviewedCocoResult, output_root: Path) -> None:
    """Persist calibration output without overwriting an earlier review run."""

    output_root.mkdir(parents=True, exist_ok=True)
    instances_path = output_root / "instances.reviewed.json"
    uncertain_path = output_root / "uncertain-images.json"
    manifest_path = output_root / "assembly-manifest.json"
    for path in (instances_path, uncertain_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"reviewed COCO output exists: {path}")
    _atomic_json(instances_path, result.document)
    _atomic_json(uncertain_path, list(result.uncertain_image_ids))
    _atomic_json(
        manifest_path,
        {
            "schema_version": "safe-auto-reviewed-coco-manifest-v1",
            "instances_sha256": hashlib.sha256(instances_path.read_bytes())
            .hexdigest()
            .upper(),
            "complete_images": len(result.document["images"]),
            "accepted_annotations": len(result.document["annotations"]),
            "uncertain_images": len(result.uncertain_image_ids),
        },
    )


def _rows(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _xyxy_to_bbox(value: object) -> list[float]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(coordinate, (int, float)) for coordinate in value)
    ):
        raise ValueError(f"invalid xyxy: {value}")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty xyxy: {value}")
    return [x1, y1, x2 - x1, y2 - y1]


def _normalized_to_pixels(
    value: object,
    width: int,
    height: int,
) -> list[float]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(coordinate, (int, float)) and 0 <= coordinate <= 1
            for coordinate in value
        )
    ):
        raise ValueError(f"invalid normalized xyxy: {value}")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    return _xyxy_to_bbox([x1 * width, y1 * height, x2 * width, y2 * height])


def _second_by_image(
    document: dict[str, object] | None,
) -> dict[object, dict[str, Any]]:
    if document is None:
        return {}
    if document.get("schema_version") != "safe-auto-second-pass-review-v1":
        raise ValueError("invalid second-pass schema")
    if document.get("first_result_hidden") is not True:
        raise ValueError("second pass must hide first result")
    prompt_version = document.get("prompt_version")
    if not isinstance(prompt_version, str) or not prompt_version:
        raise ValueError("invalid second-pass prompt version")
    rows = _rows(document.get("reviews"), "second-pass reviews")
    result: dict[object, dict[str, Any]] = {}
    for row in rows:
        image_id = row.get("image_id")
        if image_id is None or image_id in result:
            raise ValueError("invalid or duplicate second-pass image")
        result[image_id] = row
    return result


def _proposal_outcomes(
    row: dict[str, Any],
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    decisions = _rows(row.get("proposal_decisions"), "proposal_decisions")
    actual_ids = [str(decision.get("proposal_id") or "") for decision in decisions]
    if "" in actual_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("invalid or duplicate proposal decision")
    missing = sorted(expected_ids - set(actual_ids))
    unknown = sorted(set(actual_ids) - expected_ids)
    if missing:
        raise ValueError(f"missing proposal decisions: {missing}")
    if unknown:
        raise ValueError(f"unknown proposal decisions: {unknown}")
    if any(
        decision.get("decision") not in FINAL_PROPOSAL_DECISIONS
        for decision in decisions
    ):
        raise ValueError("invalid proposal decision")
    if row.get("image_status") not in {"complete", "uncertain"}:
        raise ValueError("invalid second-pass image status")
    return {str(decision["proposal_id"]): decision for decision in decisions}


def assemble_reviewed_coco(
    candidates: dict[str, object],
    first_review_document: dict[str, object],
    second_review_document: dict[str, object] | None,
) -> ReviewedCocoResult:
    """Keep only images whose candidate and miss-sweep decisions are final."""

    images = _rows(candidates.get("images"), "candidate images")
    fused = _rows(candidates.get("fused_candidates"), "fused candidates")
    image_by_id = {row.get("id"): row for row in images}
    candidate_by_id = {str(row.get("id") or ""): row for row in fused}
    if None in image_by_id or len(image_by_id) != len(images):
        raise ValueError("invalid or duplicate candidate image")
    if "" in candidate_by_id or len(candidate_by_id) != len(fused):
        raise ValueError("invalid or duplicate fused candidate")
    candidates_by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
    for candidate in fused:
        candidates_by_image[candidate.get("image_id")].append(candidate)

    raw_first = first_review_document.get("reviews")
    first_reviews = _rows(
        raw_first if raw_first is not None else [first_review_document],
        "first-pass reviews",
    )
    second_by_image = _second_by_image(second_review_document)
    accepted_images: list[dict[str, Any]] = []
    accepted_annotations: list[dict[str, Any]] = []
    uncertain_ids: list[int] = []
    annotation_id = 1

    for review in first_reviews:
        errors = validate_first_pass_review(review)
        if errors:
            raise ValueError(f"invalid first-pass review: {', '.join(errors)}")
        image_id = review.get("image_id")
        image = image_by_id.get(image_id)
        if image is None:
            raise ValueError(f"review references unknown image: {image_id}")
        expected_candidates = {
            str(candidate["id"]): candidate
            for candidate in candidates_by_image.get(image_id, [])
        }
        decisions = _rows(review.get("candidate_decisions"), "candidate_decisions")
        decision_ids = [str(row.get("candidate_id") or "") for row in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("duplicate candidate decisions")
        missing = sorted(set(expected_candidates) - set(decision_ids))
        unknown = sorted(set(decision_ids) - set(expected_candidates))
        if missing:
            raise ValueError(f"missing candidate decisions: {missing}")
        if unknown:
            raise ValueError(f"unknown candidate decisions: {unknown}")

        added_boxes = _rows(review.get("added_boxes"), "added_boxes")
        proposal_boxes: dict[str, tuple[str, object]] = {}
        for decision in decisions:
            if decision.get("decision") == "needs_adjustment":
                candidate_id = str(decision["candidate_id"])
                proposal_boxes[candidate_id] = (
                    str(expected_candidates[candidate_id].get("category") or ""),
                    decision.get("corrected_xyxy"),
                )
        for index, added in enumerate(added_boxes, start=1):
            proposal_boxes[f"added-{image_id}-{index}"] = (
                str(added.get("category") or ""),
                added.get("xyxy"),
            )

        outcomes: dict[str, dict[str, Any]] = {}
        second = second_by_image.get(image_id)
        if proposal_boxes and second is not None:
            outcomes = _proposal_outcomes(second, set(proposal_boxes))

        unresolved = (
            review.get("image_status") == "uncertain"
            or any(row.get("decision") == "uncertain" for row in decisions)
            or (bool(proposal_boxes) and second is None)
            or (second is not None and second.get("image_status") != "complete")
            or any(row.get("decision") == "uncertain" for row in outcomes.values())
        )
        if unresolved:
            uncertain_ids.append(int(image_id))
            continue

        width = int(image["width"])
        height = int(image["height"])
        pending_annotations: list[tuple[str, list[float], str]] = []
        for decision in decisions:
            value = decision.get("decision")
            candidate_id = str(decision["candidate_id"])
            candidate = expected_candidates[candidate_id]
            category = str(candidate.get("category") or "")
            if value == "accept":
                pending_annotations.append(
                    (category, _xyxy_to_bbox(candidate.get("xyxy")), "candidate")
                )
            elif (
                value == "needs_adjustment"
                and outcomes[candidate_id]["decision"] == "accept"
            ):
                final_box = (
                    outcomes[candidate_id].get("final_xyxy")
                    or proposal_boxes[candidate_id][1]
                )
                pending_annotations.append(
                    (
                        category,
                        _normalized_to_pixels(final_box, width, height),
                        "adjusted",
                    )
                )
        for proposal_id, (category, proposed_box) in proposal_boxes.items():
            if not proposal_id.startswith("added-"):
                continue
            outcome = outcomes[proposal_id]
            if outcome["decision"] == "accept":
                final_box = outcome.get("final_xyxy") or proposed_box
                pending_annotations.append(
                    (
                        category,
                        _normalized_to_pixels(final_box, width, height),
                        "added",
                    )
                )

        if any(
            category not in VALID_CATEGORIES
            for category, _, _ in pending_annotations
        ):
            raise ValueError("accepted annotation has invalid category")
        output_image = dict(image)
        output_image["file_name"] = output_image.get("relative_path")
        output_image["image_review_status"] = "complete"
        accepted_images.append(output_image)
        for category, bbox, origin in pending_annotations:
            accepted_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_IDS[category],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "review_status": "accept",
                    "second_pass": (
                        "accept" if origin != "candidate" else "not_required"
                    ),
                    "origin": origin,
                }
            )
            annotation_id += 1

    return ReviewedCocoResult(
        document={
            "info": {
                "schema_version": "safe-auto-reviewed-coco-v1",
                "truth_tier": "reviewed-ai-calibration",
            },
            "images": accepted_images,
            "annotations": accepted_annotations,
            "categories": [
                {"id": 1, "name": "fastener"},
                {"id": 2, "name": "pipe_joint"},
            ],
        },
        uncertain_image_ids=tuple(sorted(uncertain_ids)),
    )
