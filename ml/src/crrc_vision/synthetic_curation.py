from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .synthetic_state import validate_state
from .synthetic_witness_mark import extract_witness_mark_mask


def _read_color(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(path)


def _points(value: object) -> tuple[tuple[float, float], tuple[float, float]]:
    pairs = tuple(tuple(map(float, point)) for point in value)  # type: ignore[arg-type]
    if len(pairs) != 2 or any(len(point) != 2 for point in pairs):
        raise RuntimeError("segment must contain two xy points")
    return pairs  # type: ignore[return-value]


def _bbox(value: object, width: int, height: int, label: str) -> tuple[float, float, float, float]:
    bbox = tuple(map(float, value))  # type: ignore[arg-type]
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
        raise RuntimeError(f"{label} out of bounds")
    return bbox  # type: ignore[return-value]


def curate_candidates(
    selection: dict,
    references_path: Path,
    candidates_dir: Path,
    baselines_dir: Path,
    output: Path,
) -> dict:
    """Freeze Codex-reviewed ImageGen candidates and exact paint masks."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"curation output must be empty: {output}")
    references_document = json.loads(references_path.read_text(encoding="utf-8"))
    references = {record["reference_id"]: record for record in references_document["records"]}
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    curated: list[dict] = []
    for item in selection["records"]:
        if item.get("review_status") != "APPROVED":
            raise RuntimeError(f"candidate is not approved: {item.get('image', '<unknown>')}")
        reference_id = str(item["reference_id"])
        state = str(item["state"])
        reference = references[reference_id]
        if reference.get("source_split") != "train":
            raise RuntimeError(f"non-train reference: {reference_id}")
        source = candidates_dir / str(item["image"])
        baseline_path = baselines_dir / f"{reference_id}.png"
        image = _read_color(source)
        baseline = _read_color(baseline_path)
        if image.shape != baseline.shape:
            raise RuntimeError(f"baseline shape mismatch: {source.name}")
        height, width = image.shape[:2]
        roi = _bbox(item["mark_roi_xyxy"], width, height, "mark_roi_xyxy")
        fastener_bbox = _bbox(item["fastener_bbox_xyxy"], width, height, "fastener_bbox_xyxy")
        mark_mask = extract_witness_mark_mask(
            image,
            roi,
            padding_fraction=0.0,
            baseline_image=baseline,
        )
        if np.count_nonzero(mark_mask) < 24:
            raise RuntimeError(f"ImageGen witness mark missing: {source.name}")
        fixed = _points(item["fixed_segment_xyxy"])
        moving = _points(item["moving_segment_xyxy"])
        anchor = tuple(map(float, item["anchor_xy"]))
        audit = validate_state(state, fixed, moving)
        if not audit.accepted:
            raise RuntimeError(f"state geometry mismatch: {source.name}: {audit.reason}")

        sample_id = str(item.get("sample_id") or source.stem)
        image_destination = output / f"{sample_id}{source.suffix.lower()}"
        mask_name = f"{sample_id}.mark-mask.png"
        mask_destination = output / mask_name
        shutil.copy2(source, image_destination)
        _write_png(mask_destination, mark_mask)
        sidecar = {
            "sample_id": sample_id,
            "source_reference_sha256": reference["source_reference_sha256"],
            "source_scene_id": reference["source_scene_id"],
            "source_image": reference["source_image"],
            "source_image_sha256": reference["source_image_sha256"],
            "source_bbox_xywh": list(map(float, reference["source_bbox_xywh"])),
            "source_split": "train",
            "state": state,
            "fastener_bbox_xyxy": list(fastener_bbox),
            "fixed_segment_xyxy": [list(point) for point in fixed],
            "moving_segment_xyxy": [list(point) for point in moving],
            "anchor_xy": list(anchor),
            "review_status": "APPROVED",
            "prompt_sha256": item.get("prompt_sha256") or reference["prompt_sha256"][state],
            "witness_mark_mask_path": mask_name,
            "review_reason": item.get("review_reason", "Codex full-resolution geometry review passed"),
            "source_candidate": source.name,
        }
        sidecar_path = image_destination.with_suffix(image_destination.suffix + ".json")
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts[state] += 1
        curated.append({"sample_id": sample_id, "state": state, "image": image_destination.name})
    result = {
        "schema_version": "synthetic-marked-point-curation-v1",
        "approved_total": len(curated),
        "approved_by_state": dict(sorted(counts.items())),
        "records": curated,
    }
    (output / "curation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
