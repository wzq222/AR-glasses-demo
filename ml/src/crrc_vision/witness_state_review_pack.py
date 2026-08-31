"""Build a hash-bound human review pack for real witness-mark state crops."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .synthetic_witness_mark import extract_witness_mark_mask
from .witness_state_contract import MARK_ROLES, OUTPUT_STATES, TOPOLOGIES


@dataclass(frozen=True)
class StateReviewPackSummary:
    references: int
    geometry_proposals: int
    batches: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _decode_bgr(content: bytes, identity: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"REFERENCE_DECODE_FAILED:{identity}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG_ENCODE_FAILED:{path}")
    encoded.tofile(path)


def _evidence_views(content: bytes, root: Path, reference_id: str) -> dict[str, dict[str, object]]:
    with Image.open(BytesIO(content)) as opened:
        original = opened.convert("RGB")
    views = {
        "original_1x": (original, 1),
        "detail_2x": (
            original.resize((original.width * 2, original.height * 2), Image.Resampling.NEAREST),
            2,
        ),
        "detail_4x": (
            original.resize((original.width * 4, original.height * 4), Image.Resampling.NEAREST),
            4,
        ),
    }
    records: dict[str, dict[str, object]] = {}
    for name, (image, scale) in views.items():
        path = root / f"{reference_id}-{name}.png"
        image.save(path, format="PNG", optimize=True)
        records[name] = {
            "path": str(path.relative_to(root.parent)).replace("\\", "/"),
            "sha256": _sha256(path),
            "scale": scale,
            "source": "decoded_original_pixels",
            "interpolation": "none" if scale == 1 else "nearest",
        }
    return records


def _build_state_review_pack_unpublished(
    references_path: Path,
    reference_root: Path,
    output_root: Path,
    *,
    batch_size: int = 8,
    expected_formal_truth_sha256: str | None = None,
) -> StateReviewPackSummary:
    if output_root.exists():
        raise FileExistsError(output_root)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest_content = references_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest().upper()
    document = json.loads(manifest_content.decode("utf-8"))
    if expected_formal_truth_sha256 is not None and str(
        document.get("formal_truth_sha256") or ""
    ).upper() != expected_formal_truth_sha256.upper():
        raise RuntimeError("formal truth lineage mismatch")
    records = document.get("records")
    if not isinstance(records, list) or document.get("count") != len(records):
        raise ValueError("REFERENCE_MANIFEST_INVALID")

    prepared: list[tuple[dict[str, object], str, bytes, np.ndarray]] = []
    identities: set[str] = set()
    for source_record in records:
        if not isinstance(source_record, dict):
            raise ValueError("REFERENCE_RECORD_INVALID")
        reference_id = str(source_record.get("reference_id") or "")
        identity_key = reference_id.casefold()
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", reference_id) is None
            or identity_key in identities
        ):
            raise ValueError(f"REFERENCE_ID_INVALID:{reference_id}")
        identities.add(identity_key)
        source = _safe_path(reference_root, str(source_record.get("crop_path") or ""))
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest().upper() != str(
            source_record.get("source_reference_sha256") or ""
        ).upper():
            raise RuntimeError(f"REFERENCE_HASH_MISMATCH:{reference_id}")
        image = _decode_bgr(content, reference_id)
        try:
            with Image.open(BytesIO(content)) as opened:
                opened.verify()
        except Exception as exc:
            raise RuntimeError(f"REFERENCE_DECODE_FAILED:{reference_id}") from exc
        prepared.append((source_record, reference_id, content, image))

    output_root.mkdir(parents=True, exist_ok=False)
    evidence_root = output_root / "evidence"
    mask_root = output_root / "paint-masks"
    overlay_root = output_root / "proposal-overlays"
    task_root = output_root / "tasks"
    for path in (evidence_root, mask_root, overlay_root, task_root):
        path.mkdir()

    tasks: list[dict[str, object]] = []
    geometry_count = 0
    for source_record, reference_id, content, image in prepared:
        height, width = image.shape[:2]
        mask = extract_witness_mark_mask(
            image,
            (0.0, 0.0, float(width), float(height)),
            padding_fraction=0.0,
        )
        mask_path = mask_root / f"{reference_id}.png"
        _write_png(mask_path, mask)
        y_values, x_values = np.nonzero(mask)
        color_bbox = None
        if len(x_values):
            color_bbox = [
                int(x_values.min()),
                int(y_values.min()),
                int(x_values.max()) + 1,
                int(y_values.max()) + 1,
            ]
        # The color selector was designed for generated images with a known
        # changed-pixel baseline.  Rust, copper and painted structures in real
        # photos satisfy the same broad HSV rule, so it must not produce real
        # endpoint geometry.  Reviewers may use the pixels as a weak visual aid
        # but must bind both segments themselves.
        geometry: dict[str, object] | None = None
        reasons = ["REAL_ENDPOINTS_REQUIRE_REVIEW"]

        overlay = image.copy()
        overlay[mask > 0] = (0, 255, 0)
        overlay_path = overlay_root / f"{reference_id}.png"
        _write_png(overlay_path, overlay)

        tasks.append(
            {
                "reference_id": reference_id,
                "source_split": source_record.get("source_split"),
                "source_scene_id": source_record.get("source_scene_id"),
                "source_image": source_record.get("source_image"),
                "source_image_sha256": source_record.get("source_image_sha256"),
                "source_reference_sha256": source_record.get("source_reference_sha256"),
                "crop_box_xyxy": source_record.get("crop_box_xyxy"),
                "evidence_views": _evidence_views(content, evidence_root, reference_id),
                "paint_mask_path": str(mask_path.relative_to(output_root)).replace("\\", "/"),
                "paint_mask_sha256": _sha256(mask_path),
                "paint_mask_pixels": int(np.count_nonzero(mask)),
                "paint_color_proposal": {
                    "bbox_xyxy": color_bbox,
                    "pixel_count": int(np.count_nonzero(mask)),
                    "proposal_only": True,
                    "trusted_for_geometry": False,
                },
                "proposal_overlay_path": str(overlay_path.relative_to(output_root)).replace("\\", "/"),
                "proposal_overlay_sha256": _sha256(overlay_path),
                "geometry_proposal": geometry,
                "automatic_state": "INSUFFICIENT",
                "automatic_reason": "HUMAN_TOPOLOGY_AND_SEGMENT_BINDING_REQUIRED",
                "uncertainty_reasons": reasons,
                "review_template": {
                    "review_status": "UNREVIEWED",
                    "topology": None,
                    "mark_role": None,
                    "quality_pass": None,
                    "fixed_segment_xyxy": None,
                    "moving_segment_xyxy": None,
                    "damaged_mark": None,
                    "output_state": None,
                    "review_hint": None,
                    "notes": None,
                },
            }
        )

    task_hashes: dict[str, str] = {}
    for index in range(math.ceil(len(tasks) / batch_size)):
        path = task_root / f"task-{index + 1:03d}.json"
        payload = {
            "schema_version": "real-witness-state-review-task-v1",
            "instructions": (
                "Inspect 1x before zoom. Confirm the physical joint topology and bind one paint "
                "segment to the moving side and one to the fixed side. A geometry proposal is "
                "only a visual aid and never a state label. One-sided paint, unknown topology, "
                "blur, occlusion or conflicting evidence stays INSUFFICIENT."
            ),
            "allowed_topologies": sorted(TOPOLOGIES),
            "allowed_mark_roles": sorted(MARK_ROLES),
            "allowed_states": sorted(OUTPUT_STATES),
            "records": tasks[index * batch_size : (index + 1) * batch_size],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        task_hashes[str(path.relative_to(output_root)).replace("\\", "/")] = _sha256(path)

    summary = StateReviewPackSummary(
        references=len(tasks),
        geometry_proposals=geometry_count,
        batches=math.ceil(len(tasks) / batch_size),
    )
    manifest = {
        "schema_version": "real-witness-state-review-pack-v1",
        **asdict(summary),
        "formal_truth_sha256": document.get("formal_truth_sha256"),
        "source_manifest": str(references_path.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "state_truth_created": False,
        "production_thresholds_calibrated": False,
        "task_files": task_hashes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_state_review_pack(
    references_path: Path,
    reference_root: Path,
    output_root: Path,
    *,
    batch_size: int = 8,
    expected_formal_truth_sha256: str | None = None,
    formal_truth_path: Path | None = None,
) -> StateReviewPackSummary:
    """Build completely in a sibling staging directory, then publish atomically."""
    if output_root.exists():
        raise FileExistsError(output_root)
    if expected_formal_truth_sha256 is not None and formal_truth_path is None:
        raise ValueError("formal_truth_path is required with an expected formal hash")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    try:
        summary = _build_state_review_pack_unpublished(
            references_path,
            reference_root,
            staging_root,
            batch_size=batch_size,
            expected_formal_truth_sha256=expected_formal_truth_sha256,
        )
        if expected_formal_truth_sha256 is not None:
            assert formal_truth_path is not None
            if _sha256(formal_truth_path) != expected_formal_truth_sha256.upper():
                raise RuntimeError("FORMAL_TRUTH_CHANGED")
        staging_root.replace(output_root)
        return summary
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
