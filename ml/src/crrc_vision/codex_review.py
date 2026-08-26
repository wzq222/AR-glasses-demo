"""Validation contract for auditable two-pass Codex visual review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from crrc_vision.auto_labeling import VALID_CATEGORIES


VALID_CANDIDATE = {"accept", "reject", "needs_adjustment", "uncertain"}
VALID_IMAGE = {"complete", "uncertain"}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _valid_normalized_box(box: object) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= value <= 1.0
            for value in box
        )
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _valid_pass(value: object) -> bool:
    row = _mapping(value)
    return (
        isinstance(row.get("prompt_version"), str)
        and bool(row["prompt_version"])
        and row.get("decision") in VALID_CANDIDATE
    )


def validate_review(review: dict[str, object]) -> tuple[str, ...]:
    """Return stable review rejection codes; no errors means it is mergeable."""

    errors: set[str] = set()
    if review.get("reviewer") != "codex-visual-auditor":
        errors.add("INVALID_REVIEWER")
    if review.get("task_version") != "safe-auto-review-v1":
        errors.add("INVALID_TASK_VERSION")

    first_pass = _mapping(review.get("first_pass"))
    if not _valid_pass(first_pass):
        errors.add("INVALID_FIRST_PASS")

    raw_decisions = review.get("candidate_decisions")
    decisions = _rows(raw_decisions)
    if not isinstance(raw_decisions, list) or len(decisions) != len(raw_decisions):
        errors.add("INVALID_CANDIDATE_DECISION")
    decision_values = {row.get("decision") for row in decisions}
    if not decision_values <= VALID_CANDIDATE:
        errors.add("INVALID_CANDIDATE_DECISION")
    candidate_ids = [row.get("candidate_id") for row in decisions]
    if (
        any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        errors.add("INVALID_CANDIDATE_REFERENCE")

    raw_added = review.get("added_boxes")
    added_boxes = _rows(raw_added)
    if not isinstance(raw_added, list) or len(added_boxes) != len(raw_added):
        errors.add("INVALID_ADDED_BOX")
    if any(
        row.get("category") not in VALID_CATEGORIES
        or not _valid_normalized_box(row.get("xyxy"))
        for row in added_boxes
    ):
        errors.add("INVALID_ADDED_BOX")

    image_status = review.get("image_status")
    if image_status not in VALID_IMAGE:
        errors.add("INVALID_IMAGE_STATUS")
    if "uncertain" in decision_values and image_status == "complete":
        errors.add("UNCERTAIN_COMPLETE_CONFLICT")

    requires_second = "needs_adjustment" in decision_values or bool(added_boxes)
    second_pass_value = review.get("second_pass")
    if requires_second and not second_pass_value:
        errors.add("SECOND_PASS_REQUIRED")
    if second_pass_value:
        second_pass = _mapping(second_pass_value)
        if not _valid_pass(second_pass):
            errors.add("INVALID_SECOND_PASS")
        if (
            second_pass.get("prompt_version") == first_pass.get("prompt_version")
            or second_pass.get("first_result_hidden") is not True
        ):
            errors.add("NON_INDEPENDENT_SECOND_PASS")

    asset_hash = review.get("asset_sha256")
    if not (
        isinstance(asset_hash, str)
        and len(asset_hash) == 64
        and all(character in "0123456789ABCDEFabcdef" for character in asset_hash)
    ):
        errors.add("INVALID_ASSET_HASH")

    reasons = review.get("reasons")
    if not (
        isinstance(reasons, list)
        and reasons
        and all(isinstance(reason, str) and reason for reason in reasons)
    ):
        errors.add("INVALID_REASONS")
    return tuple(sorted(errors))
