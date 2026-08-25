"""Minimal COCO serialization and validation for prelabels."""

from __future__ import annotations

from collections.abc import Iterable

from .prelabel import MarkedFastenerCandidate


def validate_annotation(*, width: int, height: int, bbox: list[int | float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x, y, width, height")
    x, y, box_width, box_height = bbox
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox dimensions must be positive")
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise ValueError("bbox is outside image")


def build_coco_document(
    images: Iterable[tuple[str, int, int, str, str, list[MarkedFastenerCandidate]]],
    *,
    algorithm_version: str,
) -> dict[str, object]:
    document: dict[str, object] = {
        "info": {"description": "CRRC marked-fastener prelabels", "algorithm_version": algorithm_version},
        "licenses": [],
        "categories": [{"id": 1, "name": "marked_fastener"}],
        "images": [],
        "annotations": [],
    }
    coco_images: list[dict[str, object]] = document["images"]  # type: ignore[assignment]
    annotations: list[dict[str, object]] = document["annotations"]  # type: ignore[assignment]

    annotation_id = 1
    for image_id, (file_name, width, height, split, scene_group, candidates) in enumerate(images, start=1):
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "split": split,
                "scene_group": scene_group,
            }
        )
        for candidate in candidates:
            bbox = candidate.bbox.as_coco()
            validate_annotation(width=width, height=height, bbox=bbox)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": candidate.bbox.width * candidate.bbox.height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": {
                        "algorithm_version": algorithm_version,
                        "review_status": "unreviewed",
                        "candidate_confidence": candidate.confidence,
                        "mark_color": candidate.mark_color,
                        "mark_area": candidate.mark_area,
                        "line_confidence": candidate.line.confidence,
                        "line_points": [
                            candidate.line.start.x,
                            candidate.line.start.y,
                            candidate.line.end.x,
                            candidate.line.end.y,
                        ],
                    },
                }
            )
            annotation_id += 1
    return document
