"""Build a hash-bound, train-only ROI manifest for witness-state geometry."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image

from .synthetic_contract import (
    FROZEN_FORMAL_TRUTH_SHA256,
    SYNTHETIC_STATES,
    assert_external_output,
    assert_formal_truth_unchanged,
    sha256_file,
)


def _safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return Path(*posix.parts)


def _valid_points(value: object, *, expected_points: int) -> bool:
    try:
        return len(value) == expected_points and all(  # type: ignore[arg-type]
            len(point) == 2  # type: ignore[arg-type]
            and all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and isfinite(coordinate)
                for coordinate in point
            )
            for point in value  # type: ignore[union-attr]
        )
    except (TypeError, ValueError):
        return False


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_witness_roi_manifest(
    *,
    source_manifest: Path,
    formal_truth: Path,
    output_manifest: Path,
    repository_root: Path,
    expected_formal_sha256: str = FROZEN_FORMAL_TRUTH_SHA256,
) -> dict[str, Any]:
    """Validate approved synthetic locals and write a training-only manifest.

    The output remains outside Git and explicitly refuses to represent real
    aligned/displaced truth.  It is suitable for geometry reconstruction and
    pipeline smoke tests only.
    """
    source_manifest = source_manifest.resolve()
    formal_truth = formal_truth.resolve()
    output_manifest = assert_external_output(output_manifest, repository_root)
    formal_hash = assert_formal_truth_unchanged(formal_truth, expected_formal_sha256)

    source_bytes = source_manifest.read_bytes()
    source_document = json.loads(source_bytes.decode("utf-8"))
    if str(source_document.get("formal_truth_sha256", "")).upper() != formal_hash:
        raise RuntimeError("source manifest formal truth hash mismatch")
    records = source_document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source manifest records must be a non-empty list")

    source_root = source_manifest.parent
    seen_ids: set[str] = set()
    states_by_scene: dict[str, set[str]] = defaultdict(set)
    examples: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source record must be an object")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        state = record.get("state")
        scene = record.get("source_scene_id")
        if state not in SYNTHETIC_STATES:
            raise ValueError(f"unsupported synthetic state: {state}")
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError(f"source_scene_id missing: {sample_id}")
        if (
            record.get("synthetic") is not True
            or record.get("eligible_split") != "train"
            or record.get("source_split") != "train"
            or record.get("review_status") != "APPROVED"
        ):
            raise ValueError(f"record is not approved train-only synthetic geometry: {sample_id}")

        image_relative = _safe_relative_path(record.get("image_path"), "image_path")
        mask_relative = _safe_relative_path(
            record.get("witness_mark_mask_path"), "witness_mark_mask_path"
        )
        image_path = (source_root / image_relative).resolve()
        mask_path = (source_root / mask_relative).resolve()
        if source_root not in image_path.parents or source_root not in mask_path.parents:
            raise ValueError(f"asset escaped source root: {sample_id}")
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"ROI asset missing: {sample_id}")
        image_hash = sha256_file(image_path)
        mask_hash = sha256_file(mask_path)
        if image_hash != str(record.get("image_sha256", "")).upper():
            raise RuntimeError(f"image hash mismatch: {sample_id}")
        if mask_hash != str(record.get("witness_mark_mask_sha256", "")).upper():
            raise RuntimeError(f"mask hash mismatch: {sample_id}")

        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image_size = [int(image.width), int(image.height)]
        with Image.open(mask_path) as mask:
            mask.verify()
        with Image.open(mask_path) as mask:
            if [int(mask.width), int(mask.height)] != image_size:
                raise ValueError(f"mask dimensions do not match image: {sample_id}")

        if not _valid_points(record.get("fixed_segment_xyxy"), expected_points=2):
            raise ValueError(f"fixed segment invalid: {sample_id}")
        if not _valid_points(record.get("moving_segment_xyxy"), expected_points=2):
            raise ValueError(f"moving segment invalid: {sample_id}")
        bbox = record.get("fastener_bbox_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(value)
                for value in bbox
            )
            or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
        ):
            raise ValueError(f"fastener bbox invalid: {sample_id}")
        angle = record.get("relative_angle_deg")
        if (
            not isinstance(angle, (int, float))
            or isinstance(angle, bool)
            or not isfinite(angle)
            or not 0.0 <= abs(float(angle)) <= 90.0
        ):
            raise ValueError(f"relative angle invalid: {sample_id}")

        states_by_scene[scene].add(state)
        examples.append(
            {
                "sample_id": sample_id,
                "scene_group": scene,
                "split": "train",
                "state": state,
                "synthetic_geometry_only": True,
                "image_path": image_relative.as_posix(),
                "image_sha256": image_hash,
                "witness_mark_mask_path": mask_relative.as_posix(),
                "witness_mark_mask_sha256": mask_hash,
                "image_size": image_size,
                "fastener_bbox_xyxy": [float(value) for value in bbox],
                "fixed_segment_xyxy": record["fixed_segment_xyxy"],
                "moving_segment_xyxy": record["moving_segment_xyxy"],
                "relative_angle_deg": float(angle),
                "source_reference_sha256": record.get("source_reference_sha256"),
            }
        )

    missing_state_sets = {
        scene: sorted(SYNTHETIC_STATES - states)
        for scene, states in states_by_scene.items()
        if states != SYNTHETIC_STATES
    }
    if missing_state_sets:
        raise ValueError(f"scene state trio incomplete: {missing_state_sets}")

    examples.sort(key=lambda row: str(row["sample_id"]))
    state_counts = Counter(str(row["state"]) for row in examples)
    document: dict[str, Any] = {
        "schema_version": "witness-roi-dataset-v1",
        "source_root": str(source_root),
        "input_hashes": {
            "formal_truth_sha256": formal_hash,
            "source_manifest_sha256": sha256(source_bytes).hexdigest().upper(),
        },
        "governance": {
            "synthetic_geometry_only": True,
            "real_state_truth": False,
            "sealed_test_opened": False,
        },
        "counts": {
            "examples": len(examples),
            "scene_groups": len(states_by_scene),
            "states": {
                state: state_counts[state]
                for state in ("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")
            },
        },
        "examples": examples,
    }
    _atomic_json(output_manifest, document)
    return document
