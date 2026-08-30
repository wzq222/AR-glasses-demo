from __future__ import annotations

from collections.abc import Mapping

import numpy as np


REVIEW_DECISIONS = frozenset({"APPROVED", "REJECTED", "UNCERTAIN"})
BLIND_SECOND_INTENTS = frozenset({"SUBTLE_DISPLACED", "DAMAGED_MARK"})


def build_review_scales(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create pixel-preserving evidence views without inventing image detail."""
    if image.ndim not in (2, 3):
        raise ValueError("review image must have two or three dimensions")
    return {
        "detail_2x": np.repeat(np.repeat(image, 2, axis=0), 2, axis=1),
        "detail_4x": np.repeat(np.repeat(image, 4, axis=0), 4, axis=1),
    }


def _records_by_id(document: Mapping[str, object] | None) -> tuple[dict[str, dict], bool]:
    if document is None:
        return {}, True
    records = document.get("records")
    if not isinstance(records, list):
        return {}, False
    result: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            return {}, False
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in result:
            return {}, False
        result[sample_id] = record
    return result, True


def required_second_review_ids(manifest: Mapping[str, object], first: Mapping[str, object]) -> set[str]:
    manifest_records, manifest_ok = _records_by_id(manifest)
    first_records, first_ok = _records_by_id(first)
    if not manifest_ok or not first_ok:
        return set()
    return {
        sample_id
        for sample_id, record in manifest_records.items()
        if record.get("intent") in BLIND_SECOND_INTENTS
        or first_records.get(sample_id, {}).get("decision") == "UNCERTAIN"
    }


def validate_h1_reviews(
    manifest: Mapping[str, object],
    first: Mapping[str, object],
    second: Mapping[str, object] | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    manifest_records, manifest_ok = _records_by_id(manifest)
    first_records, first_ok = _records_by_id(first)
    second_records, second_ok = _records_by_id(second)
    if not manifest_ok:
        errors.append("INVALID_MANIFEST_RECORDS")
        return tuple(errors)
    if not first_ok:
        errors.append("INVALID_FIRST_REVIEW_RECORDS")
        return tuple(errors)
    if not second_ok:
        errors.append("INVALID_SECOND_REVIEW_RECORDS")
        return tuple(errors)

    for sample_id, source in manifest_records.items():
        review = first_records.get(sample_id)
        if review is None:
            errors.append(f"FIRST_REVIEW_MISSING:{sample_id}")
            continue
        if review.get("decision") not in REVIEW_DECISIONS:
            errors.append(f"INVALID_FIRST_DECISION:{sample_id}")
        if review.get("image_sha256") != source.get("image_sha256"):
            errors.append(f"IMAGE_HASH_MISMATCH:{sample_id}:first")

    for sample_id in sorted(set(first_records) - set(manifest_records)):
        errors.append(f"UNKNOWN_FIRST_REVIEW_SAMPLE:{sample_id}")

    required = required_second_review_ids(manifest, first)
    for sample_id in sorted(required):
        source = manifest_records[sample_id]
        review = second_records.get(sample_id)
        if review is None:
            errors.append(f"SECOND_REVIEW_MISSING:{sample_id}")
            continue
        if review.get("decision") not in REVIEW_DECISIONS:
            errors.append(f"INVALID_SECOND_DECISION:{sample_id}")
        if review.get("image_sha256") != source.get("image_sha256"):
            errors.append(f"IMAGE_HASH_MISMATCH:{sample_id}:second")
        if any(str(key).startswith("first_") for key in review):
            errors.append(f"SECOND_REVIEW_NOT_BLIND:{sample_id}")

    return tuple(errors)


def resolve_h1_reviews(
    manifest: Mapping[str, object],
    first: Mapping[str, object],
    second: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    errors = validate_h1_reviews(manifest, first, second)
    if errors:
        raise ValueError("; ".join(errors))
    manifest_records, _ = _records_by_id(manifest)
    first_records, _ = _records_by_id(first)
    second_records, _ = _records_by_id(second)
    required = required_second_review_ids(manifest, first)
    resolved = []
    for sample_id, source in manifest_records.items():
        first_decision = first_records[sample_id]["decision"]
        final_decision = first_decision
        if sample_id in required:
            second_decision = second_records[sample_id]["decision"]
            if first_decision == "UNCERTAIN":
                final_decision = second_decision
            elif first_decision != second_decision:
                final_decision = "UNCERTAIN"
        resolved.append(
            {
                **source,
                "review_status": final_decision,
                "first_review_reason": first_records[sample_id].get("reason", ""),
                **(
                    {"second_review_reason": second_records[sample_id].get("reason", "")}
                    if sample_id in required
                    else {}
                ),
            }
        )
    return resolved


def build_review_result_document(
    manifest: Mapping[str, object],
    first: Mapping[str, object],
    second: Mapping[str, object] | None,
) -> dict[str, object]:
    records = resolve_h1_reviews(manifest, first, second)
    status_counts = {decision: 0 for decision in sorted(REVIEW_DECISIONS)}
    for record in records:
        status_counts[str(record["review_status"])] += 1
    return {
        "schema_version": "h1-imagegen-reviewed-results-v1",
        "formal_truth_sha256": manifest.get("formal_truth_sha256"),
        "count": len(records),
        "status_counts": status_counts,
        "records": records,
    }
