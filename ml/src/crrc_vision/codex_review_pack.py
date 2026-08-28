"""Render deterministic full-image and candidate-context packs for Codex review."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


CATEGORY_COLORS = {
    "fastener": (255, 196, 0),
    "pipe_joint": (0, 220, 120),
    None: (255, 60, 60),
}


@dataclass(frozen=True)
class PackSummary:
    images: int
    candidates: int
    batches: int


def _normalized_box(value: object) -> list[float]:
    box = _valid_box(value)
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in box):
        raise ValueError(f"second-pass box must be normalized: {value}")
    return list(box)


def build_second_pass_tasks(
    review_document: dict[str, object],
    output_root: Path,
    *,
    source_root: Path | None = None,
) -> int:
    """Emit only proposed geometry, never the first-pass decision text."""

    raw_reviews = review_document.get("reviews")
    reviews = raw_reviews if isinstance(raw_reviews, list) else [review_document]
    if any(not isinstance(review, dict) for review in reviews):
        raise ValueError("reviews must contain objects")
    if output_root.exists():
        raise FileExistsError(f"second-pass task directory exists: {output_root}")

    tasks: list[dict[str, object]] = []
    for review in reviews:
        proposals: list[dict[str, object]] = []
        decisions = review.get("candidate_decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("candidate_decisions must be a list")
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("candidate decision must be an object")
            if decision.get("decision") == "needs_adjustment":
                proposals.append(
                    {
                        "proposal_id": decision.get("candidate_id"),
                        "proposed_xyxy": _normalized_box(
                            decision.get("corrected_xyxy")
                        ),
                        "origin": "adjusted_candidate",
                    }
                )
        added_boxes = review.get("added_boxes", [])
        if not isinstance(added_boxes, list):
            raise ValueError("added_boxes must be a list")
        for index, added in enumerate(added_boxes):
            if not isinstance(added, dict):
                raise ValueError("added box must be an object")
            proposals.append(
                {
                    "proposal_id": f"added-{review.get('image_id')}-{index + 1}",
                    "proposed_xyxy": _normalized_box(added.get("xyxy")),
                    "category": added.get("category"),
                    "origin": "added_box",
                }
            )
        if proposals:
            tasks.append(
                {
                    "image_id": review.get("image_id"),
                    "relative_path": review.get("relative_path"),
                    "asset_sha256": review.get("asset_sha256"),
                    "proposed_boxes": proposals,
                }
            )

    output_root.mkdir(parents=True, exist_ok=False)
    if source_root is not None:
        review_image_root = output_root / "review-images"
        review_image_root.mkdir()
        for task in tasks:
            relative_path = str(task.get("relative_path") or "")
            if not relative_path:
                raise ValueError("second-pass review is missing relative_path")
            original = _load_rgb(_safe_source(source_root, relative_path))
            rendered = original.copy()
            draw = ImageDraw.Draw(rendered)
            proposed_boxes = task.get("proposed_boxes", [])
            if not isinstance(proposed_boxes, list):
                raise ValueError("proposed_boxes must be a list")
            for proposal in proposed_boxes:
                if not isinstance(proposal, dict):
                    raise ValueError("proposed box must be an object")
                x1, y1, x2, y2 = _normalized_box(
                    proposal.get("proposed_xyxy")
                )
                pixel_box = (
                    x1 * rendered.width,
                    y1 * rendered.height,
                    x2 * rendered.width,
                    y2 * rendered.height,
                )
                draw.rectangle(pixel_box, outline=(255, 196, 0), width=6)
                draw.text(
                    (pixel_box[0] + 3, max(0, pixel_box[1] - 24)),
                    str(proposal.get("proposal_id") or ""),
                    fill=(255, 196, 0),
                    stroke_width=3,
                    stroke_fill=(0, 0, 0),
                )
            rendered.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            review_image_path = review_image_root / f"{int(task['image_id']):04d}.jpg"
            rendered.save(review_image_path, quality=95)
            task["review_image"] = str(
                review_image_path.relative_to(output_root)
            ).replace("\\", "/")
            task["review_image_sha256"] = _sha256(review_image_path)
    for batch_index in range(math.ceil(len(tasks) / 8)):
        payload = {
            "schema_version": "safe-auto-second-pass-task-v1",
            "prompt_version": "second-v2",
            "first_result_hidden": True,
            "instructions": (
                "Independently judge every proposed box against visible pixels. Return one "
                "proposal_decision per proposal_id: accept, reject, or uncertain. An accepted "
                "box may include final_xyxy when its geometry needs correction. Do not infer "
                "or reproduce any first-pass decision."
            ),
            "images": tasks[batch_index * 8 : (batch_index + 1) * 8],
        }
        (output_root / f"tasks-{batch_index + 1:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return len(tasks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _safe_source(source_root: Path, relative_path: str) -> Path:
    value = (source_root / relative_path).resolve()
    if source_root.resolve() not in value.parents or not value.is_file():
        raise FileNotFoundError(value)
    return value


def _valid_box(value: object) -> tuple[float, float, float, float]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(coordinate, (int, float)) for coordinate in value)
    ):
        raise ValueError(f"invalid candidate box: {value}")
    box = tuple(float(coordinate) for coordinate in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"empty candidate box: {value}")
    return box


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for index in range(1, 4):
        x = round(width * index / 4)
        y = round(height * index / 4)
        draw.line((x, 0, x, height), fill=(150, 150, 150), width=1)
        draw.line((0, y, width, y), fill=(150, 150, 150), width=1)
    for row in range(4):
        for column in range(4):
            draw.text(
                (column * width / 4 + 5, row * height / 4 + 5),
                f"{chr(65 + row)}{column + 1}",
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )


def _render_full(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    blind: bool = False,
) -> Image.Image:
    rendered = image.copy()
    rendered.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    scale_x = rendered.width / image.width
    scale_y = rendered.height / image.height
    draw = ImageDraw.Draw(rendered)
    _draw_grid(draw, rendered.width, rendered.height)
    for candidate in candidates:
        x1, y1, x2, y2 = _valid_box(candidate.get("xyxy"))
        box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        color = (
            CATEGORY_COLORS[None]
            if blind
            else CATEGORY_COLORS.get(candidate.get("category"), CATEGORY_COLORS[None])
        )
        draw.rectangle(box, outline=color, width=4)
        draw.text(
            (box[0] + 2, max(0, box[1] - 16)),
            str(candidate.get("id", ""))[:8],
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return rendered


def _render_context(
    image: Image.Image,
    candidate: dict[str, Any],
    *,
    blind: bool = False,
) -> Image.Image:
    x1, y1, x2, y2 = _valid_box(candidate.get("xyxy"))
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(64.0, width * 2.0)
    crop_height = max(64.0, height * 2.0)
    left = max(0, math.floor(center_x - crop_width / 2))
    top = max(0, math.floor(center_y - crop_height / 2))
    right = min(image.width, math.ceil(center_x + crop_width / 2))
    bottom = min(image.height, math.ceil(center_y + crop_height / 2))
    context = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(context)
    color = (
        CATEGORY_COLORS[None]
        if blind
        else CATEGORY_COLORS.get(candidate.get("category"), CATEGORY_COLORS[None])
    )
    draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=color, width=4)
    context.thumbnail((600, 600), Image.Resampling.LANCZOS)
    return context


def _render_miss_sweep_tiles(
    image: Image.Image,
    output_root: Path,
    safe_stem: str,
) -> list[dict[str, object]]:
    """Render four overlapping, high-resolution tiles for whole-image recall review."""

    overlap_x = max(32, round(image.width * 0.08))
    overlap_y = max(32, round(image.height * 0.08))
    middle_x = image.width // 2
    middle_y = image.height // 2
    bounds = (
        (0, 0, min(image.width, middle_x + overlap_x), min(image.height, middle_y + overlap_y)),
        (max(0, middle_x - overlap_x), 0, image.width, min(image.height, middle_y + overlap_y)),
        (0, max(0, middle_y - overlap_y), min(image.width, middle_x + overlap_x), image.height),
        (max(0, middle_x - overlap_x), max(0, middle_y - overlap_y), image.width, image.height),
    )
    rendered: list[dict[str, object]] = []
    for index, (left, top, right, bottom) in enumerate(bounds, start=1):
        tile = image.crop((left, top, right, bottom))
        tile.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 8),
            f"T{index}",
            fill=(255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        path = output_root / f"{safe_stem}_T{index}.jpg"
        tile.save(path, quality=95)
        rendered.append(
            {
                "tile": f"T{index}",
                "path": str(path.relative_to(output_root.parent)).replace("\\", "/"),
                "asset_sha256": _sha256(path),
                "source_xyxy_normalized": [
                    left / image.width,
                    top / image.height,
                    right / image.width,
                    bottom / image.height,
                ],
            }
        )
    return rendered


def build_pack(
    candidates: dict[str, object],
    source_root: Path,
    output_root: Path,
    selected_relative_paths: list[str] | None = None,
    *,
    partition: str | None = None,
    partition_manifest_sha256: str | None = None,
    include_existing_decisions: bool = False,
) -> PackSummary:
    """Render all images and candidates, including zero-candidate images."""

    if partition not in {None, "train", "val", "sealed_test"}:
        raise ValueError(f"invalid high-accuracy partition: {partition}")
    if partition == "sealed_test":
        include_existing_decisions = False

    raw_images = candidates.get("images")
    raw_fused = candidates.get("fused_candidates")
    if not isinstance(raw_images, list) or any(not isinstance(row, dict) for row in raw_images):
        raise ValueError("candidate document images must be a list")
    if not isinstance(raw_fused, list) or any(not isinstance(row, dict) for row in raw_fused):
        raise ValueError("fused_candidates must be a list")
    images = raw_images
    fused = raw_fused
    if selected_relative_paths is not None:
        if len(selected_relative_paths) != len(set(selected_relative_paths)):
            raise ValueError("selected image paths must be unique")
        by_path = {str(row.get("relative_path") or ""): row for row in raw_images}
        missing = [path for path in selected_relative_paths if path not in by_path]
        if missing:
            raise ValueError(f"selected images missing from candidates: {missing}")
        images = [by_path[path] for path in selected_relative_paths]
        selected_ids = {row.get("id") for row in images}
        fused = [row for row in raw_fused if row.get("image_id") in selected_ids]
    if output_root.exists():
        raise FileExistsError(f"Codex review pack already exists: {output_root}")

    image_by_id: dict[object, dict[str, Any]] = {}
    source_by_id: dict[object, Path] = {}
    for image in images:
        image_id = image.get("id")
        relative_path = str(image.get("relative_path") or "")
        if image_id is None or image_id in image_by_id or not relative_path:
            raise ValueError("invalid or duplicate candidate image")
        image_by_id[image_id] = image
        source_by_id[image_id] = _safe_source(source_root, relative_path)

    by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for candidate in fused:
        candidate_id = str(candidate.get("id") or "")
        image_id = candidate.get("image_id")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("invalid or duplicate fused candidate ID")
        if image_id not in image_by_id:
            raise ValueError(f"candidate references unknown image: {image_id}")
        _valid_box(candidate.get("xyxy"))
        candidate_ids.add(candidate_id)
        by_image[image_id].append(candidate)

    output_root.mkdir(parents=True, exist_ok=False)
    full_root = output_root / "full-images"
    tile_root = output_root / "miss-sweep-tiles"
    context_root = output_root / "candidate-contexts"
    task_root = output_root / "first-pass"
    full_root.mkdir()
    tile_root.mkdir()
    context_root.mkdir()
    task_root.mkdir()

    task_images: list[dict[str, Any]] = []
    scene_images: dict[str, list[str]] = defaultdict(list)
    for image in images:
        scene_images[str(image.get("scene_group") or "")].append(
            str(image["relative_path"])
        )
    for image in images:
        image_id = image["id"]
        relative_path = str(image["relative_path"])
        original = _load_rgb(source_by_id[image_id])
        safe_stem = f"{int(image_id):04d}_{Path(relative_path).stem}"
        full_path = full_root / f"{safe_stem}.jpg"
        blind = partition == "sealed_test"
        _render_full(original, by_image.get(image_id, []), blind=blind).save(
            full_path, quality=93
        )
        miss_sweep_tiles = _render_miss_sweep_tiles(original, tile_root, safe_stem)
        task_candidates = []
        for candidate in sorted(
            by_image.get(image_id, []), key=lambda row: str(row["id"])
        ):
            context_path = context_root / f"{candidate['id']}.jpg"
            _render_context(original, candidate, blind=blind).save(
                context_path, quality=94
            )
            task_candidate: dict[str, object] = {
                "candidate_id": candidate["id"],
                "context": str(context_path.relative_to(output_root)).replace(
                    "\\", "/"
                ),
                "context_sha256": _sha256(context_path),
            }
            if partition != "sealed_test":
                task_candidate.update(
                    {
                        "category": candidate.get("category"),
                        "consensus_status": candidate.get("consensus_status"),
                        "supporting_families": candidate.get(
                            "supporting_families", []
                        ),
                    }
                )
                if include_existing_decisions and "decision" in candidate:
                    task_candidate["existing_decision"] = candidate["decision"]
            task_candidates.append(task_candidate)
        scene = str(image.get("scene_group") or "")
        task_images.append(
            {
                "image_id": image_id,
                "relative_path": relative_path,
                "scene_group": scene,
                "full_image": str(full_path.relative_to(output_root)).replace(
                    "\\", "/"
                ),
                "asset_sha256": _sha256(full_path),
                "miss_sweep_tiles": miss_sweep_tiles,
                "same_scene_neighbors": [
                    path for path in scene_images[scene] if path != relative_path
                ],
                "candidates": task_candidates,
            }
        )

    batches = math.ceil(len(task_images) / 8)
    for batch_index in range(batches):
        batch = {
            "schema_version": "safe-auto-review-task-v1",
            "partition": partition,
            "partition_manifest_sha256": partition_manifest_sha256,
            "reviewer": "codex-visual-auditor",
            "task_version": "safe-auto-review-v1",
            "prompt_version": "first-v1",
            "target_definitions": {
                "fastener": "A visible bolt, nut, screw, or mechanically connected fastener assembly.",
                "pipe_joint": "A visible pipe-joint connection carrying an anti-loosening mark.",
            },
            "annotation_policy": {
                "include_only_independently_boundable_connections": True,
                "anti_loosening_mark_required_for_fastener": False,
                "exclude_boundary_truncation": True,
                "exclude_incidental_tiny_background": True,
                "exclude_duplicate_subparts": True,
                "uncertain_for_blur_occlusion_or_class_ambiguity": True,
            },
            "instructions": (
                "First scan every miss_sweep_tile for targets that have no candidate. "
                "Then review every candidate context. Use accept for a correct target box, "
                "reject for background or a duplicate, needs_adjustment plus corrected_xyxy "
                "for a real target with wrong geometry, and uncertain only when pixels cannot "
                "support a decision. Put every missed target in added_boxes with category and "
                "source-image normalized xyxy. Set image_status=complete only when the tile "
                "sweep found no unresolved miss and every candidate is final; use "
                "pending_second_pass when corrected_xyxy or added_boxes need blind geometry "
                "review. Do not use uncertain merely because a second pass is pending."
            ),
            "output_contract": {
                "candidate_decisions": "one row per candidate_id",
                "added_boxes": "missed targets with category and normalized xyxy",
                "image_status": ["complete", "pending_second_pass", "uncertain"],
            },
            "images": task_images[batch_index * 8 : (batch_index + 1) * 8],
        }
        (task_root / f"tasks-{batch_index + 1:03d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = PackSummary(len(images), len(fused), batches)
    manifest = {
        "schema_version": "safe-auto-review-pack-v1",
        "partition": partition,
        "partition_manifest_sha256": partition_manifest_sha256,
        "images": summary.images,
        "candidates": summary.candidates,
        "batches": summary.batches,
        "max_images_per_batch": 8,
        "files": {
            str(path.relative_to(output_root)).replace("\\", "/"): _sha256(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        },
    }
    (output_root / "pack-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
