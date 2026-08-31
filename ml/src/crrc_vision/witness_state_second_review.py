"""Audit blind second-pass witness-state reviews before any training export."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

from .witness_state import measure_witness_geometry
from .witness_state_contract import MARK_ROLES, OUTPUT_STATES, TOPOLOGIES


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def _verified_task(pack_root: Path, relative: str, expected_hash: object) -> dict[str, Any]:
    path = (pack_root / relative).resolve()
    if pack_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_bytes()
    if _sha256_bytes(content) != str(expected_hash).upper():
        raise RuntimeError(f"SOURCE_TASK_HASH_MISMATCH:{relative}")
    task = json.loads(content.decode("utf-8"))
    if task.get("schema_version") != "real-witness-state-second-pass-task-v1":
        raise ValueError(f"SOURCE_TASK_SCHEMA_INVALID:{relative}")
    if task.get("blind_to_first_review") is not True:
        raise ValueError(f"SOURCE_TASK_NOT_BLIND:{relative}")
    return task


def _verify_asset(
    pack_root: Path,
    relative: object,
    expected_hash: object,
    error_identity: str,
) -> None:
    path = (pack_root / str(relative or "")).resolve()
    if pack_root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_bytes(path.read_bytes()) != str(expected_hash or "").upper():
        raise RuntimeError(error_identity)


def _as_segment(value: object, identity: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"SEGMENT_INVALID:{identity}")
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"SEGMENT_INVALID:{identity}")
        coordinates: list[float] = []
        for coordinate in point:
            if (
                not isinstance(coordinate, (int, float))
                or isinstance(coordinate, bool)
                or not isfinite(coordinate)
            ):
                raise ValueError(f"SEGMENT_INVALID:{identity}")
            coordinates.append(float(coordinate))
        points.append((coordinates[0], coordinates[1]))
    return points[0], points[1]


def _confidence(value: object, identity: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"SEGMENT_CONFIDENCE_INVALID:{identity}")
    return float(value)


def _reference_size(value: object, identity: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"REFERENCE_SIZE_INVALID:{identity}")
    return float(value)


def audit_second_pass_reviews(pack_manifest_path: Path, decisions_path: Path) -> dict[str, object]:
    """Verify review lineage, recompute geometry, and fail closed for state truth.

    A single historical observation may be reviewed and measured, but it cannot
    become ALIGNED/DISPLACED training truth.  Those states require a real,
    maintenance-confirmed controlled pair from the same inspection point.
    """
    manifest_content = pack_manifest_path.read_bytes()
    manifest_hash = _sha256_bytes(manifest_content)
    manifest = json.loads(manifest_content.decode("utf-8"))
    if manifest.get("schema_version") != "real-witness-state-second-pass-pack-v1":
        raise ValueError("SOURCE_PACK_MANIFEST_INVALID")
    if (
        manifest.get("blind_to_first_review") is not True
        or manifest.get("first_review_fields_included") is not False
    ):
        raise ValueError("SOURCE_PACK_NOT_BLIND")
    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        raise ValueError("SOURCE_PACK_MANIFEST_INVALID")

    expected_ids: list[str] = []
    seen_expected: set[str] = set()
    pack_root = pack_manifest_path.resolve().parent
    for relative, expected_hash in task_files.items():
        task = _verified_task(pack_root, str(relative), expected_hash)
        records = task.get("records")
        if not isinstance(records, list):
            raise ValueError(f"SOURCE_TASK_RECORDS_INVALID:{relative}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"SOURCE_TASK_RECORD_INVALID:{relative}")
            reference_id = str(record.get("reference_id") or "")
            key = reference_id.casefold()
            if not reference_id or key in seen_expected:
                raise ValueError(f"SOURCE_REFERENCE_ID_INVALID:{reference_id}")
            evidence_views = record.get("evidence_views")
            if not isinstance(evidence_views, dict):
                raise ValueError(f"SOURCE_EVIDENCE_INVALID:{reference_id}")
            for view_name in ("original_1x", "detail_2x", "detail_4x"):
                view = evidence_views.get(view_name)
                if not isinstance(view, dict):
                    raise ValueError(f"SOURCE_EVIDENCE_INVALID:{reference_id}:{view_name}")
                _verify_asset(
                    pack_root,
                    view.get("path"),
                    view.get("sha256"),
                    f"SOURCE_EVIDENCE_HASH_MISMATCH:{reference_id}:{view_name}",
                )
            _verify_asset(
                pack_root,
                record.get("coordinate_grid_path"),
                record.get("coordinate_grid_sha256"),
                f"SOURCE_COORDINATE_GRID_HASH_MISMATCH:{reference_id}",
            )
            seen_expected.add(key)
            expected_ids.append(reference_id)
    if manifest.get("references") != len(expected_ids):
        raise ValueError("SOURCE_PACK_REFERENCE_COUNT_MISMATCH")

    decisions_content = decisions_path.read_bytes()
    decisions = json.loads(decisions_content.decode("utf-8"))
    if decisions.get("schema_version") != "real-witness-state-second-pass-decisions-v1":
        raise ValueError("DECISIONS_SCHEMA_INVALID")
    if decisions.get("blind_to_first_review_fields") is not True:
        raise ValueError("DECISIONS_NOT_BLIND")
    if str(decisions.get("source_pack_manifest_sha256") or "").upper() != manifest_hash:
        raise RuntimeError("SOURCE_PACK_MANIFEST_HASH_MISMATCH")
    formal_truth_hash = str(manifest.get("formal_truth_sha256") or "").upper()
    if str(decisions.get("formal_truth_sha256") or "").upper() != formal_truth_hash:
        raise RuntimeError("FORMAL_TRUTH_LINEAGE_MISMATCH")
    records = decisions.get("records")
    if not isinstance(records, list) or decisions.get("count") != len(records):
        raise ValueError("DECISION_COUNT_MISMATCH")

    decision_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("DECISION_RECORD_INVALID")
        reference_id = str(record.get("reference_id") or "")
        key = reference_id.casefold()
        if not reference_id or key in decision_by_id:
            raise ValueError(f"DECISION_REFERENCE_ID_INVALID:{reference_id}")
        decision_by_id[key] = record
    if set(decision_by_id) != seen_expected:
        raise ValueError("DECISION_REFERENCE_COVERAGE_MISMATCH")

    audited_records: list[dict[str, object]] = []
    state_counts: Counter[str] = Counter()
    hint_counts: Counter[str] = Counter()
    endpoint_complete = 0
    training_eligible_count = 0
    for reference_id in expected_ids:
        record = decision_by_id[reference_id.casefold()]
        if record.get("review_status") != "REVIEWED":
            raise ValueError(f"REVIEW_INCOMPLETE:{reference_id}")
        topology = str(record.get("topology") or "")
        mark_role = str(record.get("mark_role") or "")
        output_state = str(record.get("output_state") or "")
        if topology not in TOPOLOGIES:
            raise ValueError(f"TOPOLOGY_INVALID:{reference_id}")
        if mark_role not in MARK_ROLES:
            raise ValueError(f"MARK_ROLE_INVALID:{reference_id}")
        if output_state not in OUTPUT_STATES:
            raise ValueError(f"OUTPUT_STATE_INVALID:{reference_id}")
        if not isinstance(record.get("quality_pass"), bool):
            raise ValueError(f"QUALITY_PASS_INVALID:{reference_id}")
        if record.get("damaged_mark") is not None and not isinstance(
            record.get("damaged_mark"), bool
        ):
            raise ValueError(f"DAMAGED_MARK_INVALID:{reference_id}")

        fixed_value = record.get("fixed_segment_xyxy")
        moving_value = record.get("moving_segment_xyxy")
        if (fixed_value is None) != (moving_value is None):
            raise ValueError(f"SEGMENT_PAIR_INCOMPLETE:{reference_id}")

        geometry: dict[str, float] | None = None
        minimum_confidence: float | None = None
        if fixed_value is not None:
            fixed = _as_segment(fixed_value, reference_id)
            moving = _as_segment(moving_value, reference_id)
            fixed_confidence = _confidence(record.get("fixed_segment_confidence"), reference_id)
            moving_confidence = _confidence(record.get("moving_segment_confidence"), reference_id)
            minimum_confidence = min(fixed_confidence, moving_confidence)
            metrics = measure_witness_geometry(
                fixed,
                moving,
                _reference_size(record.get("reference_size"), reference_id),
            )
            geometry = {
                "angle_degrees": metrics.angle_degrees,
                "gap_ratio": metrics.gap_ratio,
                "residual_ratio": metrics.residual_ratio,
            }
            endpoint_complete += 1
        elif any(
            record.get(key) is not None
            for key in (
                "fixed_segment_confidence",
                "moving_segment_confidence",
                "reference_size",
            )
        ):
            raise ValueError(f"SEGMENT_METADATA_WITHOUT_SEGMENTS:{reference_id}")

        truth_basis = str(record.get("truth_basis") or "")
        decidable = output_state in {"ALIGNED", "DISPLACED"}
        # This schema is derived from isolated historical reference crops.  A
        # decision field cannot turn those pixels into a controlled before/
        # after experiment.  Controlled pairs require a separate capture
        # manifest that binds both members and maintenance confirmation.
        eligible = False
        if decidable:
            raise ValueError(f"DECIDABLE_STATE_REQUIRES_CONTROLLED_PAIR:{reference_id}")
        if output_state == "DAMAGED_MARK" and record["damaged_mark"] is not True:
            raise ValueError(f"DAMAGED_STATE_REQUIRES_DAMAGE_EVIDENCE:{reference_id}")
        review_hint = record.get("review_hint")
        if review_hint is not None:
            hint_counts[str(review_hint)] += 1
        state_counts[output_state] += 1
        audited_records.append(
            {
                "reference_id": reference_id,
                "output_state": output_state,
                "review_hint": review_hint,
                "truth_basis": truth_basis,
                "training_eligible": eligible,
                "minimum_segment_confidence": minimum_confidence,
                "geometry_metrics": geometry,
            }
        )

    blocking_reasons: list[str] = []
    if training_eligible_count == 0:
        blocking_reasons.append("NO_REAL_CONTROLLED_PAIR_STATE_TRUTH")
    if manifest.get("production_thresholds_calibrated") is not True:
        blocking_reasons.append("PRODUCTION_THRESHOLDS_UNCALIBRATED")
    return {
        "schema_version": "real-witness-state-second-pass-audit-v1",
        "source_pack_manifest_sha256": manifest_hash,
        "decisions_sha256": _sha256_bytes(decisions_content),
        "formal_truth_sha256": formal_truth_hash,
        "summary": {
            "reviewed": len(audited_records),
            "endpoint_complete": endpoint_complete,
            "training_eligible": training_eligible_count,
            "state_counts": dict(sorted(state_counts.items())),
            "hint_counts": dict(sorted(hint_counts.items())),
        },
        "records": audited_records,
        "training_ready": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
