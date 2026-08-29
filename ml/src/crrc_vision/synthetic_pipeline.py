from __future__ import annotations

import json
import shutil
from dataclasses import asdict
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
from .synthetic_witness_mark import extract_witness_mark_mask, remove_existing_witness_mark


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _read_mask(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> bool:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True


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
        image = _read_image(image_path)
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
        lineage_values = (
            sidecar.get("source_image"),
            sidecar.get("source_bbox_xywh"),
            sidecar.get("source_image_sha256"),
        )
        if any(value is not None for value in lineage_values) and not all(value is not None for value in lineage_values):
            raise RuntimeError(f"incomplete source image lineage: {image_path.name}")
        destination = output / record.image_path
        shutil.copy2(image_path, destination)
        mark_mask_relative = None
        mark_mask_hash = None
        if sidecar.get("witness_mark_mask_path"):
            mark_mask_source = Path(sidecar["witness_mark_mask_path"])
            if not mark_mask_source.is_absolute():
                mark_mask_source = sidecar_path.parent / mark_mask_source
            mark_mask = _read_mask(mark_mask_source)
            if mark_mask is None:
                raise RuntimeError(f"cannot decode witness mark mask: {image_path.name}")
            if mark_mask.shape != (height, width):
                raise RuntimeError(f"witness mark mask shape mismatch: {image_path.name}")
            if np.count_nonzero(mark_mask) < 12:
                raise RuntimeError(f"witness mark mask is empty: {image_path.name}")
            mark_mask_relative = f"locals/{sidecar['sample_id']}.mark-mask.png"
            mark_mask_destination = output / mark_mask_relative
            if not _write_png(mark_mask_destination, mark_mask):
                raise RuntimeError(f"failed to write witness mark mask: {image_path.name}")
            mark_mask_hash = sha256_file(mark_mask_destination)
        records.append(
            {
                **asdict(record),
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
                **(
                    {
                        "source_image": str(sidecar["source_image"]),
                        "source_bbox_xywh": list(map(float, sidecar["source_bbox_xywh"])),
                        "source_image_sha256": str(sidecar["source_image_sha256"]),
                    }
                    if sidecar.get("source_image")
                    and sidecar.get("source_bbox_xywh")
                    and sidecar.get("source_image_sha256")
                    else {}
                ),
                **(
                    {
                        "witness_mark_mask_path": mark_mask_relative,
                        "witness_mark_mask_sha256": mark_mask_hash,
                    }
                    if mark_mask_relative is not None
                    else {}
                ),
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


def _bbox_iou_xywh(first: Iterable[float], second: Iterable[float]) -> float:
    first_x, first_y, first_w, first_h = map(float, first)
    second_x, second_y, second_w, second_h = map(float, second)
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_w * first_h + second_w * second_h - intersection
    return intersection / union if union > 0 else 0.0


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
    for record in approved:
        if record.get("source_split") != "train" or record.get("eligible_split") != "train":
            raise RuntimeError(f"approved local source_split must be train: {record.get('sample_id')}")
        if record.get("synthetic") is not True:
            raise RuntimeError(f"approved local must be synthetic: {record.get('sample_id')}")
    backgrounds = json.loads(background_coco.read_text(encoding="utf-8"))
    if backgrounds.get("info", {}).get("partition") != "train":
        raise RuntimeError("background COCO partition must be train")
    images = sorted(backgrounds["images"], key=lambda item: int(item["id"]))
    if not images:
        raise RuntimeError("no train backgrounds")
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in backgrounds.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    images = [image for image in images if annotations_by_image.get(int(image["id"]))]
    if not images:
        raise RuntimeError("no annotated train backgrounds")

    output_images = output / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    local_root = approved_manifest.parent
    records = []
    coco_images = []
    coco_annotations = []
    image_hashes = []
    next_annotation_id = 1
    for index, record in enumerate(approved, start=1):
        source_lineage = (
            record.get("source_image"),
            record.get("source_bbox_xywh"),
            record.get("source_image_sha256"),
        )
        if any(value is not None for value in source_lineage) and not all(value is not None for value in source_lineage):
            raise RuntimeError(f"incomplete source image lineage for {record['sample_id']}")
        mark_only = all(value is not None for value in source_lineage)
        if mark_only:
            eligible_backgrounds = [
                item
                for item in images
                if item.get("scene_group") == record["source_scene_id"]
                and item.get("file_name") == record["source_image"]
            ]
            if not eligible_backgrounds:
                raise RuntimeError(f"source-scene background missing for {record['sample_id']}")
            background_record = eligible_backgrounds[0]
        else:
            eligible_backgrounds = [item for item in images if item.get("scene_group") != record["source_scene_id"]]
            if not eligible_backgrounds:
                raise RuntimeError(f"no independent background for {record['sample_id']}")
            background_record = eligible_backgrounds[(index - 1) % len(eligible_backgrounds)]
        background = _read_image(source_dir / background_record["file_name"])
        patch_path = local_root / record["image_path"]
        patch = _read_image(patch_path)
        if background is None or patch is None:
            raise RuntimeError(f"cannot decode source for {record['sample_id']}")
        if sha256_file(patch_path) != str(record.get("image_sha256", "")).upper():
            raise RuntimeError(f"local image SHA-256 mismatch for {record['sample_id']}")
        background_hash = sha256_file(source_dir / background_record["file_name"])
        if str(background_record.get("sha256", "")).upper() != background_hash:
            raise RuntimeError(f"background image SHA-256 mismatch for {record['sample_id']}")
        if mark_only:
            expected_source_hash = str(record.get("source_image_sha256", "")).upper()
            if not expected_source_hash or expected_source_hash != background_hash:
                raise RuntimeError(f"source image SHA-256 mismatch for {record['sample_id']}")
        bbox = tuple(map(float, record["fastener_bbox_xyxy"]))
        segments = (
            tuple(tuple(map(float, point)) for point in record["fixed_segment_xyxy"]),
            tuple(tuple(map(float, point)) for point in record["moving_segment_xyxy"]),
        )
        sample_seed = _seed(global_seed, record["sample_id"], background_record["id"])
        mask = _mask_for_bbox(patch.shape[:2], bbox)
        if record.get("witness_mark_mask_path"):
            mark_mask_path = local_root / record["witness_mark_mask_path"]
            if sha256_file(mark_mask_path) != str(record.get("witness_mark_mask_sha256", "")).upper():
                raise RuntimeError(f"witness mark mask SHA-256 mismatch for {record['sample_id']}")
            mark_mask = _read_mask(mark_mask_path)
            if mark_mask is None or mark_mask.shape != patch.shape[:2]:
                raise RuntimeError(f"invalid curated witness mark mask: {record['sample_id']}")
        else:
            mark_mask = extract_witness_mark_mask(patch, bbox)
        if np.count_nonzero(mark_mask) < 12:
            raise RuntimeError(f"ImageGen witness mark missing: {record['sample_id']}")
        perspective = sample_transform(patch.shape[1], patch.shape[0], sample_seed, TransformLimits())
        warped_patch, warped_mask = warp_image_and_mask(
            patch, mask, perspective.matrix, (patch.shape[1], patch.shape[0])
        )
        warped_mark_mask = cv2.warpPerspective(
            mark_mask,
            perspective.matrix,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
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
        background_annotations = annotations_by_image[int(background_record["id"])]
        def interior_score(annotation: dict) -> tuple[float, float]:
            x, y, width, height = map(float, annotation["bbox"])
            margin = min(x, y, background_width - x - width, background_height - y - height)
            return margin, width * height

        if mark_only:
            target_annotation = max(
                background_annotations,
                key=lambda annotation: _bbox_iou_xywh(annotation["bbox"], record["source_bbox_xywh"]),
            )
            target_iou = _bbox_iou_xywh(target_annotation["bbox"], record["source_bbox_xywh"])
            if target_iou < 0.75:
                raise RuntimeError(f"source annotation mismatch for {record['sample_id']}: IoU={target_iou:.3f}")
        else:
            target_annotation = max(background_annotations, key=interior_score)
        target_x, target_y, target_width, target_height = map(float, target_annotation["bbox"])
        target_bbox_xyxy = (
            target_x,
            target_y,
            target_x + target_width,
            target_y + target_height,
        )
        if mark_only:
            composite_background, removed_mark_mask = remove_existing_witness_mark(
                background,
                target_bbox_xyxy,
            )
            original_mark_pixels = int(np.count_nonzero(removed_mark_mask))
            residual_mark_pixels = int(
                np.count_nonzero(
                    extract_witness_mark_mask(
                        composite_background,
                        target_bbox_xyxy,
                        padding_fraction=0.15,
                    )
                )
            )
        else:
            composite_background = background
            original_mark_pixels = 0
            residual_mark_pixels = 0
        target_center_x = target_x + target_width / 2.0
        target_center_y = target_y + target_height / 2.0
        warped_width = warped_bbox[2] - warped_bbox[0]
        warped_height = warped_bbox[3] - warped_bbox[1]
        base_scale = min(target_width / warped_width, target_height / warped_height)
        warped_center_x = (warped_bbox[0] + warped_bbox[2]) / 2.0
        warped_center_y = (warped_bbox[1] + warped_bbox[3]) / 2.0
        other_boxes = []
        for existing in background_annotations:
            if existing is target_annotation:
                continue
            x, y, width, height = map(float, existing["bbox"])
            other_boxes.append((x, y, x + width, y + height))
        result = None
        for attempt in range(8):
            scale = base_scale * (1.0 - attempt * 0.025)
            placement = Placement(
                x=target_center_x - scale * warped_center_x,
                y=target_center_y - scale * warped_center_y,
                scale=scale,
                rotation_deg=0.0,
            )
            try:
                result = composite_sample(
                    composite_background,
                    warped_patch,
                    warped_mask,
                    warped_bbox,
                    warped_segments,
                    placement,
                    existing_boxes=() if mark_only else tuple(other_boxes),
                    max_overlap_iou=0.25,
                    blend_mode="mark_only" if mark_only else "seamless",
                    preserve_mask=warped_mark_mask,
                )
                break
            except (ValueError, cv2.error):
                continue
        if result is None:
            raise RuntimeError(f"no valid replacement after 8 attempts: {record['sample_id']}")
        filename = f"synthetic-{index:04d}.png"
        destination = output_images / filename
        if not _write_png(destination, result.image):
            raise RuntimeError(f"failed to write {destination}")
        image_hash = sha256_file(destination)
        image_hashes.append(image_hash)
        image_id = index
        # In mark-only mode the real fastener is never replaced.  Its detection
        # box therefore remains the reviewed source annotation exactly; only
        # the ImageGen paint pixels and state geometry are transformed.
        bbox_out = target_bbox_xyxy if mark_only else result.bbox_xyxy
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
                "removed_original_mark_pixels": original_mark_pixels,
                "residual_original_mark_pixels": residual_mark_pixels,
                "sha256": image_hash,
            }
        )
        for existing in background_annotations:
            if existing is target_annotation:
                coco_annotations.append({
                "id": next_annotation_id,
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
                "origin": "synthetic_replacement",
                "synthetic": True,
                "ignore_state": False,
                })
            else:
                preserved = {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": int(existing.get("category_id", 1)),
                    "bbox": list(map(float, existing["bbox"])),
                    "area": float(existing.get("area", existing["bbox"][2] * existing["bbox"][3])),
                    "iscrowd": int(existing.get("iscrowd", 0)),
                    "origin": "preserved_real_background",
                    "synthetic": False,
                    "state": "UNKNOWN",
                    "ignore_state": True,
                }
                coco_annotations.append(preserved)
            next_annotation_id += 1
        records.append(
            {
                "sample_id": f"full-{index:04d}",
                "image_id": image_id,
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
                "witness_mark_source": "imagegen",
                "seed": sample_seed,
                "seam_score": result.seam_score,
                "background_scene_id": background_record.get("scene_group"),
                "target_bbox_xywh": [
                    bbox_out[0],
                    bbox_out[1],
                    bbox_out[2] - bbox_out[0],
                    bbox_out[3] - bbox_out[1],
                ],
                "fixed_segment_xyxy": result.segments[0],
                "moving_segment_xyxy": result.segments[1],
                "anchor_xy": result.anchor_xy,
                "removed_original_mark_pixels": original_mark_pixels,
                "residual_original_mark_pixels": residual_mark_pixels,
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
