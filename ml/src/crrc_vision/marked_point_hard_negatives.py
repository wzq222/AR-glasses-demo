from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set

from crrc_vision.marked_point_model_gate import is_proposal_match


def _bbox(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    value = row.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("INVALID_BBOX")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or box[2] <= 0 or box[3] <= 0:
        raise ValueError("INVALID_BBOX")
    return box


def _crop_window(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    center_x = box[0] + box[2] / 2.0
    center_y = box[1] + box[3] / 2.0
    x1 = min(max(int(round(center_x - crop_width / 2.0)), 0), width - crop_width)
    y1 = min(max(int(round(center_y - crop_height / 2.0)), 0), height - crop_height)
    return x1, y1, x1 + crop_width, y1 + crop_height


def _contains_target_center(
    crop: tuple[int, int, int, int],
    targets: Sequence[tuple[float, float, float, float]],
) -> bool:
    x1, y1, x2, y2 = crop
    return any(
        x1 <= x + width / 2.0 <= x2 and y1 <= y + height / 2.0 <= y2
        for x, y, width, height in targets
    )


def select_hard_negative_crops(
    predictions: Sequence[Mapping[str, object]],
    train_truth: Mapping[str, object],
    *,
    score_threshold: float = 0.01,
    crop_size: int = 640,
    max_per_scene: int = 2,
    maximum_crops: int | None = None,
    forbidden_sha256: Set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Select high-score empty train crops without touching validation identities."""

    info = train_truth.get("info")
    if not isinstance(info, Mapping) or info.get("partition") != "train":
        raise ValueError("HARD_NEGATIVE_TRAIN_TRUTH_REQUIRED")
    if not math.isfinite(score_threshold) or score_threshold < 0.0:
        raise ValueError("INVALID_HARD_NEGATIVE_THRESHOLD")
    if crop_size <= 0 or max_per_scene <= 0 or (
        maximum_crops is not None and maximum_crops <= 0
    ):
        raise ValueError("INVALID_HARD_NEGATIVE_LIMIT")
    images = train_truth.get("images")
    annotations = train_truth.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("INVALID_TRAIN_TRUTH")
    by_id: dict[object, Mapping[str, object]] = {}
    for image in images:
        if not isinstance(image, Mapping) or image.get("id") in by_id:
            raise ValueError("INVALID_TRAIN_IMAGES")
        by_id[image.get("id")] = image
    targets_by_image: dict[object, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or annotation.get("image_id") not in by_id:
            raise ValueError("INVALID_TRAIN_ANNOTATIONS")
        targets_by_image[annotation.get("image_id")].append(_bbox(annotation))

    forbidden = {value.upper() for value in forbidden_sha256}
    ordered: list[tuple[int, Mapping[str, object]]] = []
    for index, prediction in enumerate(predictions):
        image_id = prediction.get("image_id")
        if image_id not in by_id:
            raise ValueError(f"HARD_NEGATIVE_UNKNOWN_IMAGE:{image_id}")
        score = float(prediction.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("INVALID_PREDICTION_SCORE")
        _bbox(prediction)
        if score >= score_threshold:
            ordered.append((index, prediction))
    ordered.sort(
        key=lambda item: (
            -float(item[1]["score"]),
            int(item[1]["image_id"]),
            _bbox(item[1]),
            item[0],
        )
    )

    selected: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)
    for prediction_index, prediction in ordered:
        image_id = prediction["image_id"]
        image = by_id[image_id]
        source_sha256 = str(image.get("sha256") or "").upper()
        if source_sha256 in forbidden:
            continue
        scene_group = str(image.get("scene_group") or "")
        if not scene_group or counts[scene_group] >= max_per_scene:
            continue
        box = _bbox(prediction)
        targets = targets_by_image[image_id]
        if any(is_proposal_match(box, target) for target in targets):
            continue
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("INVALID_TRAIN_IMAGE_SIZE")
        crop = _crop_window(
            box, width=width, height=height, crop_size=crop_size
        )
        if _contains_target_center(crop, targets):
            continue
        selected.append(
            {
                "prediction_index": prediction_index,
                "image_id": image_id,
                "scene_group": scene_group,
                "source_file_name": str(image.get("file_name") or ""),
                "source_sha256": source_sha256,
                "score": float(prediction["score"]),
                "prediction_bbox": list(box),
                "crop_xyxy": list(crop),
            }
        )
        counts[scene_group] += 1
        if maximum_crops is not None and len(selected) >= maximum_crops:
            break
    return selected
