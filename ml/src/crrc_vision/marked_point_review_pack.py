"""Auditable full-image review assets for marked anti-loosening points."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


BUSINESS_TARGET = "marked anti-loosening inspection point"
REVIEW_RULES = {
    "positive": (
        "A fastening or pipe-joint inspection point carrying an intentional "
        "red/yellow anti-loosening mark"
    ),
    "unmarked_fastener": (
        "A real fastener or pipe joint without an intentional anti-loosening mark"
    ),
    "lookalike": (
        "Background, rust, sticker, reflection, wire, hole, or unrelated painted structure"
    ),
    "uncertain": (
        "Pixels cannot prove target identity or intentional mark ownership"
    ),
    "full_image_rule": (
        "Scan all four tiles and add every independently boundable missed marked point"
    ),
}


@dataclass(frozen=True)
class ReviewPackSummary:
    images: int
    scan_tiles: int
    candidates: int
    batches: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _safe_source(root: Path, relative: str) -> Path:
    source = (root / relative).resolve()
    if root.resolve() not in source.parents or not source.is_file():
        raise FileNotFoundError(source)
    return source


def _box(value: object, *, width: int, height: int) -> tuple[float, float, float, float]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(coordinate, (int, float)) for coordinate in value)
    ):
        raise ValueError(f"INVALID_CANDIDATE_BOX:{value}")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError(f"INVALID_CANDIDATE_BOX:{value}")
    return x1, y1, x2, y2


def _save_full(image: Image.Image, path: Path) -> None:
    rendered = image.copy()
    rendered.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    rendered.save(path, quality=92, optimize=True)


def _save_tiles(image: Image.Image, root: Path, stem: str) -> list[dict[str, object]]:
    overlap_x = max(32, round(image.width * 0.08))
    overlap_y = max(32, round(image.height * 0.08))
    center_x, center_y = image.width // 2, image.height // 2
    bounds = (
        (0, 0, min(image.width, center_x + overlap_x), min(image.height, center_y + overlap_y)),
        (max(0, center_x - overlap_x), 0, image.width, min(image.height, center_y + overlap_y)),
        (0, max(0, center_y - overlap_y), min(image.width, center_x + overlap_x), image.height),
        (max(0, center_x - overlap_x), max(0, center_y - overlap_y), image.width, image.height),
    )
    rows: list[dict[str, object]] = []
    for index, coordinates in enumerate(bounds, 1):
        tile = image.crop(coordinates)
        tile.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 8),
            f"T{index}",
            fill=(255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        path = root / f"{stem}_T{index}.jpg"
        tile.save(path, quality=92, optimize=True)
        left, top, right, bottom = coordinates
        rows.append(
            {
                "tile": f"T{index}",
                "path": str(path.relative_to(root.parent)).replace("\\", "/"),
                "sha256": _sha256(path),
                "source_xyxy": [left, top, right, bottom],
                "source_xyxy_normalized": [
                    left / image.width,
                    top / image.height,
                    right / image.width,
                    bottom / image.height,
                ],
            }
        )
    return rows


def _save_context(
    image: Image.Image, candidate: dict[str, Any], path: Path
) -> tuple[list[int], str]:
    x1, y1, x2, y2 = _box(
        candidate.get("xyxy"), width=image.width, height=image.height
    )
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_width = max(96.0, (x2 - x1) * 1.6)
    crop_height = max(96.0, (y2 - y1) * 1.6)
    left = max(0, math.floor(center_x - crop_width / 2.0))
    top = max(0, math.floor(center_y - crop_height / 2.0))
    right = min(image.width, math.ceil(center_x + crop_width / 2.0))
    bottom = min(image.height, math.ceil(center_y + crop_height / 2.0))
    context = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(context)
    draw.rectangle(
        (x1 - left, y1 - top, x2 - left, y2 - top),
        outline=(0, 255, 255),
        width=4,
    )
    context.thumbnail((384, 384), Image.Resampling.LANCZOS)
    context.save(path, quality=88, optimize=True)
    return [left, top, right, bottom], _sha256(path)


def build_review_pack(
    selection: dict[str, object],
    candidates: dict[str, object],
    source_root: Path,
    output_root: Path,
) -> ReviewPackSummary:
    """Render unoccluded full scans and exactly one crop per union candidate."""

    if output_root.exists():
        raise FileExistsError(f"REVIEW_PACK_ALREADY_EXISTS:{output_root}")
    if selection.get("old_sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID")
    forbidden = selection.get("forbidden_old_sealed")
    if not isinstance(forbidden, dict):
        raise ValueError("FORBIDDEN_IDENTITIES_MISSING")
    forbidden_paths = set(forbidden.get("paths", []))
    forbidden_hashes = {str(value).lower() for value in forbidden.get("sha256", [])}

    selected: dict[str, dict[str, object]] = {}
    for partition in ("train", "val"):
        rows = selection.get(partition)
        if not isinstance(rows, list):
            raise ValueError(f"SELECTION_INVALID:{partition}")
        for source in rows:
            if not isinstance(source, dict):
                raise ValueError("SELECTION_ROW_INVALID")
            relative = str(source.get("relative_path") or "").replace("\\", "/")
            digest = str(source.get("sha256") or "").lower()
            if relative in forbidden_paths or digest in forbidden_hashes:
                raise ValueError(f"OLD_SEALED_IMAGE_FORBIDDEN:{relative}")
            if not relative or relative in selected:
                raise ValueError(f"DUPLICATE_SELECTED_PATH:{relative}")
            selected[relative] = {**source, "partition": partition}

    image_rows = candidates.get("images")
    candidate_rows = candidates.get("fused_candidates")
    if not isinstance(image_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("CANDIDATE_DOCUMENT_INVALID")
    candidate_image_by_path = {
        str(source.get("relative_path") or "").replace("\\", "/"): source
        for source in image_rows
        if isinstance(source, dict)
    }
    if set(candidate_image_by_path) != set(selected):
        raise ValueError("CANDIDATE_IMAGE_COVERAGE_MISMATCH")

    by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for source in candidate_rows:
        if not isinstance(source, dict):
            raise ValueError("CANDIDATE_ROW_INVALID")
        candidate_id = str(source.get("id") or "")
        relative = str(source.get("relative_path") or "").replace("\\", "/")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"DUPLICATE_CANDIDATE_ID:{candidate_id}")
        if relative not in selected:
            raise ValueError(f"CANDIDATE_OUTSIDE_SELECTION:{relative}")
        image_id = source.get("image_id")
        if image_id != selected[relative].get("image_id"):
            raise ValueError(f"CANDIDATE_IMAGE_ID_MISMATCH:{candidate_id}")
        candidate_ids.add(candidate_id)
        by_image[image_id].append(source)

    output_root.mkdir(parents=True, exist_ok=False)
    full_root = output_root / "full-images"
    tile_root = output_root / "scan-tiles"
    context_root = output_root / "candidate-contexts"
    task_root = output_root / "first-pass"
    for path in (full_root, tile_root, context_root, task_root):
        path.mkdir()

    tasks: list[dict[str, object]] = []
    for relative in sorted(selected, key=lambda value: str(selected[value]["scene_group"])):
        row = selected[relative]
        image_id = row["image_id"]
        source_path = _safe_source(source_root, relative)
        if _sha256(source_path).lower() != str(row["sha256"]).lower():
            raise RuntimeError(f"SOURCE_HASH_MISMATCH:{relative}")
        image = _load_rgb(source_path)
        candidate_image = candidate_image_by_path[relative]
        expected_width = candidate_image.get("width")
        expected_height = candidate_image.get("height")
        if expected_width not in (None, image.width) or expected_height not in (None, image.height):
            raise ValueError(f"IMAGE_SIZE_MISMATCH:{relative}")

        stem = f"{int(image_id):04d}_{Path(relative).stem}"
        full_path = full_root / f"{stem}.jpg"
        _save_full(image, full_path)
        tiles = _save_tiles(image, tile_root, stem)
        task_candidates: list[dict[str, object]] = []
        for candidate in sorted(by_image[image_id], key=lambda value: str(value["id"])):
            candidate_id = str(candidate["id"])
            context_path = context_root / f"{candidate_id}.jpg"
            crop_xyxy, context_hash = _save_context(image, candidate, context_path)
            task_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "xyxy": candidate["xyxy"],
                    "sources": candidate.get("sources", []),
                    "context": str(context_path.relative_to(output_root)).replace("\\", "/"),
                    "context_sha256": context_hash,
                    "context_source_xyxy": crop_xyxy,
                }
            )
        expected_ids = [str(value["id"]) for value in sorted(by_image[image_id], key=lambda value: str(value["id"]))]
        tasks.append(
            {
                "image_id": image_id,
                "relative_path": relative,
                "scene_group": row["scene_group"],
                "partition": row["partition"],
                "source_sha256": row["sha256"],
                "business_target": BUSINESS_TARGET,
                "full_image": str(full_path.relative_to(output_root)).replace("\\", "/"),
                "full_image_sha256": _sha256(full_path),
                "scan_tiles": tiles,
                "expected_candidate_ids": expected_ids,
                "candidates": task_candidates,
            }
        )

    batches = math.ceil(len(tasks) / 8)
    for index in range(batches):
        document = {
            "schema_version": "marked-point-review-task-v1",
            "reviewer": "codex-visual-auditor",
            "first_pass": True,
            "business_target": BUSINESS_TARGET,
            "review_rules": REVIEW_RULES,
            "instructions": (
                "Inspect the unoccluded full image and every scan tile first. Add every "
                "independently boundable missed marked point. Then assign exactly one of "
                "marked_point, unmarked_fastener, lookalike, uncertain to every candidate. "
                "A complete image cannot retain uncertain. Added or adjusted positives "
                "require a blind geometry second pass."
            ),
            "images": tasks[index * 8 : (index + 1) * 8],
        }
        path = task_root / f"tasks-{index + 1:03d}.json"
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = ReviewPackSummary(
        images=len(tasks),
        scan_tiles=len(tasks) * 4,
        candidates=len(candidate_ids),
        batches=batches,
    )
    manifest = {
        "schema_version": "marked-point-review-pack-v1",
        **summary.__dict__,
        "business_target": BUSINESS_TARGET,
        "review_rules": REVIEW_RULES,
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(candidate_ids)).encode("utf-8")
        ).hexdigest().upper(),
        "old_sealed_test_opened": False,
        "task_files": {
            str(path.relative_to(output_root)).replace("\\", "/"): _sha256(path)
            for path in sorted(task_root.glob("tasks-*.json"))
        },
    }
    (output_root / "pack-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
