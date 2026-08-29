from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

from .synthetic_contract import SYNTHETIC_STATES, assert_formal_truth_unchanged, sha256_file
from .synthetic_state import validate_state


@dataclass(frozen=True)
class SyntheticAuditResult:
    passed: bool
    approved_total: int
    approved_by_state: dict[str, int]
    rejected_total: int
    uncertain_total: int
    errors: tuple[str, ...]


def audit_records(
    records: Iterable[Mapping[str, object]],
    *,
    minimum_per_state: int = 8,
    minimum_approval_rate: float = 0.75,
) -> SyntheticAuditResult:
    items = list(records)
    errors: list[str] = []
    approved_counts: Counter[str] = Counter()
    rejected_total = 0
    uncertain_total = 0
    seen_ids: set[str] = set()

    for index, record in enumerate(items):
        prefix = f"record[{index}]"
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            errors.append(f"{prefix}.sample_id missing or duplicated")
        seen_ids.add(sample_id)
        if record.get("synthetic") is not True:
            errors.append(f"{prefix}.synthetic must be true")
        if record.get("eligible_split") != "train":
            errors.append(f"{prefix}.eligible_split must be train")
        if record.get("source_split") != "train":
            errors.append(f"{prefix}.source_split must be train")
        state = str(record.get("state", ""))
        if state not in SYNTHETIC_STATES:
            errors.append(f"{prefix}.state is invalid")
        digest = str(record.get("source_reference_sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            errors.append(f"{prefix}.source_reference_sha256 is invalid")

        review = str(record.get("review_status", ""))
        if review == "APPROVED" and state in SYNTHETIC_STATES:
            approved_counts[state] += 1
        elif review == "REJECTED":
            rejected_total += 1
        elif review == "UNCERTAIN":
            uncertain_total += 1
        else:
            errors.append(f"{prefix}.review_status is invalid")

    approved_total = sum(approved_counts.values())
    for state in sorted(SYNTHETIC_STATES):
        if approved_counts[state] < minimum_per_state:
            errors.append(
                f"approved {state} {approved_counts[state]} < required {minimum_per_state}"
            )
    approval_rate = approved_total / len(items) if items else 0.0
    if approval_rate < minimum_approval_rate:
        errors.append(
            f"approval rate {approval_rate:.4f} < required {minimum_approval_rate:.4f}"
        )
    return SyntheticAuditResult(
        passed=not errors,
        approved_total=approved_total,
        approved_by_state={state: approved_counts[state] for state in sorted(SYNTHETIC_STATES)},
        rejected_total=rejected_total,
        uncertain_total=uncertain_total,
        errors=tuple(errors),
    )


def audit_manifest(
    records: Iterable[Mapping[str, object]],
    formal_truth: Path,
) -> SyntheticAuditResult:
    assert_formal_truth_unchanged(formal_truth)
    return audit_records(records)


def _valid_digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _bbox_in_bounds(value: object, width: int, height: int) -> bool:
    try:
        x, y, box_width, box_height = map(float, value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return x >= 0 and y >= 0 and box_width > 0 and box_height > 0 and x + box_width <= width and y + box_height <= height


def audit_full_dataset(
    document: Mapping[str, object],
    root: Path,
    coco_path: Path,
    formal_truth: Path,
    *,
    minimum_per_state: int = 8,
    minimum_approval_rate: float = 0.75,
    expected_formal_hash: str | None = None,
    review_pack_manifest_path: Path | None = None,
) -> SyntheticAuditResult:
    """Audit real files, COCO geometry and hash-bound visual-review evidence."""
    errors: list[str] = []
    try:
        formal_hash = (
            assert_formal_truth_unchanged(formal_truth, expected_formal_hash)
            if expected_formal_hash is not None
            else assert_formal_truth_unchanged(formal_truth)
        )
    except RuntimeError as exc:
        formal_hash = ""
        errors.append(str(exc))
    if document.get("schema_version") != "synthetic-marked-point-full-v1":
        errors.append("manifest schema_version is invalid")
    if str(document.get("formal_truth_sha256", "")).upper() != formal_hash:
        errors.append("manifest formal_truth_sha256 does not match the frozen truth")
    records = list(document.get("records", []))  # type: ignore[arg-type]
    shallow = audit_records(
        records,
        minimum_per_state=minimum_per_state,
        minimum_approval_rate=minimum_approval_rate,
    )
    errors.extend(shallow.errors)
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read COCO: {exc}")
        coco = {"images": [], "annotations": []}
    images = {int(item["id"]): item for item in coco.get("images", [])}
    if len(images) != len(coco.get("images", [])):
        errors.append("COCO image ids are duplicated")
    if len(images) != len(records):
        errors.append("COCO image count differs from manifest")
    annotations_by_image: dict[int, list[Mapping[str, object]]] = {}
    annotation_ids = [int(item.get("id", -1)) for item in coco.get("annotations", [])]
    if len(set(annotation_ids)) != len(annotation_ids):
        errors.append("COCO annotation ids are duplicated")
    review_pack_hash = ""
    review_pack_records: dict[str, Mapping[str, object]] = {}
    if review_pack_manifest_path is None:
        errors.append("review-pack manifest is required for full audit")
    else:
        try:
            review_pack_hash = sha256_file(review_pack_manifest_path)
            review_pack = json.loads(review_pack_manifest_path.read_text(encoding="utf-8"))
            pack_items = list(review_pack.get("records", []))
            review_pack_records = {str(item["sample_id"]): item for item in pack_items}
            manifest_ids = {str(item.get("sample_id", "")) for item in records}
            if len(review_pack_records) != len(pack_items) or set(review_pack_records) != manifest_ids:
                errors.append("review pack must cover every manifest record exactly")
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            errors.append(f"cannot read review pack: {exc}")
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation.get("image_id", -1)), []).append(annotation)
    hashes: list[str] = []
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        image_id = int(record.get("image_id", -1))
        image_record = images.get(image_id)
        if image_record is None:
            errors.append(f"{prefix}.image_id missing from COCO")
            continue
        image_path = root / str(record.get("image_path", ""))
        if not image_path.is_file():
            errors.append(f"{prefix}.image_path missing")
            continue
        actual_hash = sha256_file(image_path)
        hashes.append(actual_hash)
        if actual_hash != str(record.get("image_sha256", "")).upper():
            errors.append(f"{prefix}.image_sha256 mismatch")
        if actual_hash != str(image_record.get("sha256", "")).upper():
            errors.append(f"{prefix}.COCO image sha256 mismatch")
        if image_path.name != str(image_record.get("file_name", "")):
            errors.append(f"{prefix}.COCO file_name mismatch")
        if image_record.get("eligible_split") != "train" or image_record.get("synthetic") is not True:
            errors.append(f"{prefix}.COCO image lineage is invalid")
        if image_record.get("source_scene_id") != record.get("source_scene_id"):
            errors.append(f"{prefix}.COCO source scene mismatch")
        evidence = record.get("review_evidence")
        if record.get("review_status") == "APPROVED":
            if not isinstance(evidence, Mapping):
                errors.append(f"{prefix}.approved record lacks review_evidence")
            else:
                if evidence.get("image_sha256") != actual_hash:
                    errors.append(f"{prefix}.review image hash mismatch")
                if evidence.get("full_image_reviewed") is not True or evidence.get("crop_reviewed") is not True:
                    errors.append(f"{prefix}.review coverage is incomplete")
                if evidence.get("review_pack_manifest_sha256") != review_pack_hash:
                    errors.append(f"{prefix}.review-pack manifest hash mismatch")
                if not str(evidence.get("reviewer", "")).strip():
                    errors.append(f"{prefix}.reviewer is missing")
                if not str(evidence.get("reviewed_at", "")).strip():
                    errors.append(f"{prefix}.reviewed_at is missing")
                pack_record = review_pack_records.get(str(record.get("sample_id", "")))
                if pack_record is None:
                    errors.append(f"{prefix}.review-pack record is missing")
                else:
                    crop_path = review_pack_manifest_path.parent / str(pack_record.get("crop_path", ""))  # type: ignore[union-attr]
                    if not crop_path.is_file():
                        errors.append(f"{prefix}.review crop is missing")
                    else:
                        crop_hash = sha256_file(crop_path)
                        if crop_hash != str(pack_record.get("crop_sha256", "")).upper() or crop_hash != evidence.get("crop_sha256"):
                            errors.append(f"{prefix}.review crop hash mismatch")
                    if str(pack_record.get("full_image_sha256", "")).upper() != actual_hash:
                        errors.append(f"{prefix}.review-pack full image hash mismatch")
        width, height = int(image_record.get("width", 0)), int(image_record.get("height", 0))
        image_annotations = annotations_by_image.get(image_id, [])
        replacements = [item for item in image_annotations if item.get("origin") == "synthetic_replacement"]
        if len(replacements) != 1:
            errors.append(f"{prefix}.synthetic replacement count must be one")
            continue
        annotation = replacements[0]
        if annotation.get("source_sample_id") != record.get("source_sample_id"):
            errors.append(f"{prefix}.replacement source sample mismatch")
        if annotation.get("state") != record.get("state") or annotation.get("ignore_state") is not False:
            errors.append(f"{prefix}.state annotation mismatch")
        if not _bbox_in_bounds(annotation.get("bbox"), width, height):
            errors.append(f"{prefix}.replacement bbox out of bounds")
        expected_bbox = [float(value) for value in record.get("target_bbox_xywh", [])]  # type: ignore[arg-type]
        actual_bbox = [float(value) for value in annotation.get("bbox", [])]  # type: ignore[arg-type]
        if len(expected_bbox) != 4 or actual_bbox != expected_bbox:
            errors.append(f"{prefix}.replacement bbox differs from reviewed target")
        try:
            fixed = tuple(tuple(map(float, point)) for point in annotation["fixed_segment_xyxy"])  # type: ignore[index]
            moving = tuple(tuple(map(float, point)) for point in annotation["moving_segment_xyxy"])  # type: ignore[index]
            anchor = tuple(map(float, annotation["anchor_xy"]))  # type: ignore[index]
            if any(not (0 <= x < width and 0 <= y < height) for x, y in (*fixed, *moving, anchor)):
                errors.append(f"{prefix}.state geometry out of bounds")
            state_audit = validate_state(str(record.get("state", "")), fixed, moving)
            if not state_audit.accepted:
                errors.append(f"{prefix}.state geometry mismatch: {state_audit.reason}")
            if max(abs(anchor[axis] - fixed[1][axis]) for axis in (0, 1)) > 1e-4 or max(abs(anchor[axis] - moving[0][axis]) for axis in (0, 1)) > 1e-4:
                errors.append(f"{prefix}.anchor is not shared by both segments")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"{prefix}.invalid state geometry: {exc}")
        for annotation_item in image_annotations:
            if not _bbox_in_bounds(annotation_item.get("bbox"), width, height):
                errors.append(f"{prefix}.COCO annotation bbox out of bounds")
        if int(record.get("residual_original_mark_pixels", -1)) != 0:
            errors.append(f"{prefix}.residual original witness mark pixels must be zero")
        if int(image_record.get("residual_original_mark_pixels", -1)) != int(record.get("residual_original_mark_pixels", -2)):
            errors.append(f"{prefix}.COCO residual mark count mismatch")
    expected_content_hash = sha256("\n".join(hashes).encode("ascii")).hexdigest().upper()
    if expected_content_hash != str(document.get("content_sha256", "")).upper():
        errors.append("manifest content_sha256 mismatch")
    return SyntheticAuditResult(
        passed=not errors,
        approved_total=shallow.approved_total,
        approved_by_state=shallow.approved_by_state,
        rejected_total=shallow.rejected_total,
        uncertain_total=shallow.uncertain_total,
        errors=tuple(errors),
    )
