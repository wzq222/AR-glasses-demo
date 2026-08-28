from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceCandidate:
    image: dict
    annotation: dict
    crop_box_xyxy: tuple[int, int, int, int]
    brightness: float
    sharpness: float


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def context_box(
    bbox: list[float], width: int, height: int, context_scale: float = 2.6
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = map(float, bbox)
    side = max(box_width, box_height) * context_scale
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    left = max(0, int(round(center_x - side / 2.0)))
    top = max(0, int(round(center_y - side / 2.0)))
    right = min(width, int(round(center_x + side / 2.0)))
    bottom = min(height, int(round(center_y + side / 2.0)))
    return left, top, right, bottom


def crop_quality(image: np.ndarray) -> tuple[float, float]:
    if image.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


def select_reference_candidates(
    coco: dict,
    source_dir: Path,
    count: int,
    *,
    minimum_brightness: float = 50.0,
    minimum_sharpness: float = 30.0,
) -> list[ReferenceCandidate]:
    by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    candidates: list[ReferenceCandidate] = []
    seen_scenes: set[str] = set()
    for image_record in sorted(coco["images"], key=lambda item: (item["scene_group"], item["id"])):
        scene = str(image_record["scene_group"])
        if scene in seen_scenes:
            continue
        annotations = by_image.get(int(image_record["id"]), [])
        valid = [
            annotation
            for annotation in annotations
            if min(float(annotation["bbox"][2]), float(annotation["bbox"][3])) >= 36.0
            and float(annotation["bbox"][0]) > 4.0
            and float(annotation["bbox"][1]) > 4.0
            and float(annotation["bbox"][0]) + float(annotation["bbox"][2]) < float(image_record["width"]) - 4.0
            and float(annotation["bbox"][1]) + float(annotation["bbox"][3]) < float(image_record["height"]) - 4.0
        ]
        if not valid:
            continue
        annotation = max(valid, key=lambda item: float(item["bbox"][2]) * float(item["bbox"][3]))
        image = read_image(source_dir / image_record["file_name"])
        if image is None:
            continue
        crop_box = context_box(annotation["bbox"], image.shape[1], image.shape[0])
        left, top, right, bottom = crop_box
        brightness, sharpness = crop_quality(image[top:bottom, left:right])
        if brightness < minimum_brightness or sharpness < minimum_sharpness:
            continue
        seen_scenes.add(scene)
        candidates.append(
            ReferenceCandidate(image_record, annotation, crop_box, brightness, sharpness)
        )
    if len(candidates) < count:
        raise RuntimeError(
            f"quality-qualified train references {len(candidates)} < requested {count}; "
            f"brightness>={minimum_brightness}, sharpness>={minimum_sharpness}"
        )
    candidates.sort(
        key=lambda item: (item.sharpness ** 0.5) * min(item.brightness, 160.0), reverse=True
    )
    return candidates[:count]
