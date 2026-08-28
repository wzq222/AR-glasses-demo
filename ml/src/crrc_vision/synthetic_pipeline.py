from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .synthetic_composite import Placement, composite_sample
from .synthetic_contract import SyntheticRecord, assert_external_output, assert_formal_truth_unchanged, sha256_file
from .synthetic_state import validate_state
from .synthetic_transform import (
    TransformLimits,
    apply_homography_points,
    apply_photometric,
    sample_transform,
    transform_bbox,
    warp_image_and_mask,
)


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _points_in_bounds(points: Iterable[Iterable[float]], width: int, height: int) -> bool:
    return all(0 <= float(x) < width and 0 <= float(y) < height for x, y in points)


def ingest_local_candidates(
    image_paths: list[Path],
    output: Path,
    repository_root: Path,
    formal_truth: Path,
    expected_formal_hash: str,
) -> dict:
    output = assert_external_output(output, repository_root)
    formal_hash = assert_formal_truth_unchanged(formal_truth, expected_formal_hash)
    local_dir = output / "locals"
    local_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in image_paths:
        sidecar_path = image_path.with_suffix(image_path.suffix + ".json")
        if not sidecar_path.is_file():
            raise RuntimeError(f"missing sidecar for {image_path.name}: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode generated local image: {image_path}")
        height, width = image.shape[:2]
        bbox = tuple(map(float, sidecar["fastener_bbox_xyxy"]))
        if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            raise RuntimeError(f"bbox out of bounds: {image_path.name}")
        fixed = tuple(tuple(map(float, point)) for point in sidecar["fixed_segment_xyxy"])
        moving = tuple(tuple(map(float, point)) for point in sidecar["moving_segment_xyxy"])
        anchor = tuple(map(float, sidecar["anchor_xy"]))
        if not _points_in_bounds((*fixed, *moving, anchor), width, height):
            raise RuntimeError(f"endpoint or anchor out of bounds: {image_path.name}")
        state_audit = validate_state(sidecar["state"], fixed, moving)
        review_status = str(sidecar["review_status"])
        if review_status == "APPROVED" and not state_audit.accepted:
            raise RuntimeError(f"approved state geometry mismatch: {image_path.name}: {state_audit.reason}")
        record = SyntheticRecord(
            sample_id=sidecar["sample_id"],
            source_reference_sha256=sidecar["source_reference_sha256"],
            source_scene_id=sidecar["source_scene_id"],
            state=sidecar["state"],
            image_path=f"locals/{sidecar['sample_id']}{image_path.suffix.lower()}",
        )
        if sidecar.get("source_split") != "train":
            raise RuntimeError(f"source_split must be train: {image_path.name}")
        destination = output / record.image_path
        shutil.copy2(image_path, destination)
        records.append(
            {
                **record.__dict__,
                "source_split": "train",
                "fastener_bbox_xyxy": list(bbox),
                "fixed_segment_xyxy": [list(point) for point in fixed],
                "moving_segment_xyxy": [list(point) for point in moving],
                "anchor_xy": list(anchor),
                "relative_angle_deg": state_audit.angle_deg,
                "relative_offset_px": state_audit.relative_offset_px,
                "review_status": review_status,
                "prompt_sha256": sidecar["prompt_sha256"],
                "image_sha256": sha256_file(destination),
            }
        )
    document = {
        "schema_version": "synthetic-marked-point-local-v1",
        "formal_truth_sha256": formal_hash,
        "records": records,
    }
    _atomic_json(output / "approved-locals.json", document)
    assert_formal_truth_unchanged(formal_truth, formal_hash)
    return document


def _seed(global_seed: int, *parts: object) -> int:
    digest = sha256(":".join([str(global_seed), *map(str, parts)]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mask_for_bbox(shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox
    padding = 0.35 * max(x2 - x1, y2 - y1)
    left = max(0, int(x1 - padding))
    top = max(0, int(y1 - padding))
    right = min(width - 1, int(x2 + padding))
    bottom = min(height - 1, int(y2 + padding))
    mask = np.zeros((height, width), dtype=np.uint8)
    center = ((left + right) // 2, (top + bottom) // 2)
    axes = (max(1, (right - left) // 2), max(1, (bottom - top) // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def build_full_images(
    approved_manifest: Path,
    background_coco: Path,
    source_dir: Path,
    output: Path,
    global_seed: int,
    formal_truth: Path,
    expected_formal_hash: str,
) -> dict:
    formal_hash = assert_formal_truth_unchanged(formal_truth, expected_formal_hash)
    approved = json.loads(approved_manifest.read_text(encoding="utf-8"))["records"]
    approved = [record for record in approved if record["review_status"] == "APPROVED"]
    if not approved:
        raise RuntimeError("no approved local records")
    backgrounds = json.loads(background_coco.read_text(encoding="utf-8"))
    images = sorted(backgrounds["images"], key=lambda item: int(item["id"]))
    if not images:
        raise RuntimeError("no train backgrounds")
    annotations_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
    for annotation in backgrounds.get("annotations", []):
        x, y, width, height = map(float, annotation["bbox"])
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append((x, y, x + width, y + height))

    output_images = output / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    local_root = approved_manifest.parent
    records = []
    coco_images = []
    coco_annotations = []
    image_hashes = []
    for index, record in enumerate(approved, start=1):
        eligible_backgrounds = [item for item in images if item.get("scene_group") != record["source_scene_id"]]
        if not eligible_backgrounds:
            raise RuntimeError(f"no independent background for {record['sample_id']}")
        background_record = eligible_backgrounds[(index - 1) % len(eligible_backgrounds)]
        background = cv2.imread(str(source_dir / background_record["file_name"]), cv2.IMREAD_COLOR)
        patch = cv2.imread(str(local_root / record["image_path"]), cv2.IMREAD_COLOR)
        if background is None or patch is None:
            raise RuntimeError(f"cannot decode source for {record['sample_id']}")
        bbox = tuple(map(float, record["fastener_bbox_xyxy"]))
        segments = (
            tuple(tuple(map(float, point)) for point in record["fixed_segment_xyxy"]),
            tuple(tuple(map(float, point)) for point in record["moving_segment_xyxy"]),
        )
        sample_seed = _seed(global_seed, record["sample_id"], background_record["id"])
        rng = np.random.default_rng(sample_seed)
        mask = _mask_for_bbox(patch.shape[:2], bbox)
        perspective = sample_transform(patch.shape[1], patch.shape[0], sample_seed, TransformLimits())
        warped_patch, warped_mask = warp_image_and_mask(
            patch, mask, perspective.matrix, (patch.shape[1], patch.shape[0])
        )
        warped_patch = apply_photometric(warped_patch, sample_seed + 1)
        warped_bbox = transform_bbox(bbox, perspective.matrix)
        flat_segments = np.array([segments[0][0], segments[0][1], segments[1][0], segments[1][1]])
        warped_points = apply_homography_points(flat_segments, perspective.matrix)
        warped_segments = (
            ((float(warped_points[0, 0]), float(warped_points[0, 1])), (float(warped_points[1, 0]), float(warped_points[1, 1]))),
            ((float(warped_points[2, 0]), float(warped_points[2, 1])), (float(warped_points[3, 0]), float(warped_points[3, 1]))),
        )
        background_height, background_width = background.shape[:2]
        result = None
        for attempt in range(40):
            scale = float(rng.uniform(0.85, 1.15))
            max_x = max(1.0, background_width - patch.shape[1] * scale - 2.0)
            max_y = max(1.0, background_height - patch.shape[0] * scale - 2.0)
            placement = Placement(
                x=float(rng.uniform(1.0, max_x)),
                y=float(rng.uniform(1.0, max_y)),
                scale=scale,
                rotation_deg=float(rng.uniform(-8.0, 8.0)),
            )
            try:
                result = composite_sample(
                    background,
                    warped_patch,
                    warped_mask,
                    warped_bbox,
                    warped_segments,
                    placement,
                    existing_boxes=tuple(annotations_by_image.get(int(background_record["id"]), [])),
                )
                break
            except ValueError:
                continue
        if result is None:
            raise RuntimeError(f"no valid placement after 40 attempts: {record['sample_id']}")
        filename = f"synthetic-{index:04d}.png"
        destination = output_images / filename
        if not cv2.imwrite(str(destination), result.image):
            raise RuntimeError(f"failed to write {destination}")
        image_hash = sha256_file(destination)
        image_hashes.append(image_hash)
        image_id = index
        bbox_out = result.bbox_xyxy
        coco_images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": background_width,
                "height": background_height,
                "synthetic": True,
                "eligible_split": "train",
                "source_scene_id": record["source_scene_id"],
                "background_scene_id": background_record.get("scene_group"),
                "sha256": image_hash,
            }
        )
        coco_annotations.append(
            {
                "id": index,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [bbox_out[0], bbox_out[1], bbox_out[2] - bbox_out[0], bbox_out[3] - bbox_out[1]],
                "area": (bbox_out[2] - bbox_out[0]) * (bbox_out[3] - bbox_out[1]),
                "iscrowd": 0,
                "state": record["state"],
                "fixed_segment_xyxy": result.segments[0],
                "moving_segment_xyxy": result.segments[1],
                "anchor_xy": result.anchor_xy,
                "source_sample_id": record["sample_id"],
                "seam_score": result.seam_score,
            }
        )
        records.append(
            {
                "sample_id": f"full-{index:04d}",
                "source_sample_id": record["sample_id"],
                "state": record["state"],
                "image_path": f"images/{filename}",
                "image_sha256": image_hash,
                "synthetic": True,
                "eligible_split": "train",
                "source_split": "train",
                "source_scene_id": record["source_scene_id"],
                "source_reference_sha256": record["source_reference_sha256"],
                "review_status": "UNCERTAIN",
                "seed": sample_seed,
                "seam_score": result.seam_score,
            }
        )
    content_sha256 = sha256("\n".join(image_hashes).encode("ascii")).hexdigest().upper()
    manifest = {
        "schema_version": "synthetic-marked-point-full-v1",
        "formal_truth_sha256": formal_hash,
        "global_seed": global_seed,
        "content_sha256": content_sha256,
        "records": records,
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(
        output / "instances.synthetic-train.json",
        {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": 1, "name": "marked_point"}]},
    )
    assert_formal_truth_unchanged(formal_truth, formal_hash)
    return manifest
