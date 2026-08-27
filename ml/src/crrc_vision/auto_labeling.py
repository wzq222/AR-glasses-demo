"""Deterministic multi-source candidate fusion for safe automatic labeling."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


VALID_CATEGORIES = {"fastener", "pipe_joint"}
VALID_SOURCE_FAMILIES = {"reference_teacher", "hsv", "temporal", "student"}
DEFAULT_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_CENTER_DISTANCE_THRESHOLD = 0.25
DEFAULT_MAX_AREA_RATIO = 4.0


Box = tuple[float, float, float, float]


def verify_truth_unchanged(before_sha256: str, after_sha256: str) -> None:
    """Stop candidate generation if the formal truth changed during the run."""

    if before_sha256 != after_sha256:
        raise RuntimeError(
            f"formal truth changed during candidate generation: "
            f"{before_sha256} != {after_sha256}"
        )


def _xywh_to_xyxy(value: object) -> Box:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            for coordinate in value
        )
    ):
        raise ValueError(f"invalid xywh box: {value}")
    x, y, width, height = (float(coordinate) for coordinate in value)
    box = (x, y, x + width, y + height)
    _validate_box(box)
    return box


def normalize_teacher_payload(payload: Mapping[str, object]) -> list[Candidate]:
    """Normalize reference-teacher xywh predictions into source-aware candidates."""

    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("teacher payload predictions must be a list")
    default_pass = f"full-{payload.get('imgsz', 'legacy')}"
    output: list[Candidate] = []
    for index, value in enumerate(predictions):
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid teacher prediction at index {index}")
        pass_id = str(value.get("pass_id") or default_pass)
        prediction_id = str(value.get("id") or index)
        output.append(
            Candidate(
                relative_path=str(value.get("relative_path") or ""),
                source_id=f"teacher:{pass_id}:{prediction_id}",
                source_family="reference_teacher",
                category=str(value.get("mapped_category") or ""),
                xyxy=_xywh_to_xyxy(value.get("bbox")),
                score=float(value.get("score", -1.0)),
            )
        )
    return output


def normalize_hsv_document(document: Mapping[str, object]) -> list[Candidate]:
    """Normalize color-mark COCO candidates as fastener anchors."""

    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("HSV document requires image and annotation lists")
    image_paths: dict[object, str] = {}
    for value in images:
        if not isinstance(value, Mapping):
            raise ValueError("invalid HSV image row")
        image_id = value.get("id")
        relative_path = str(value.get("file_name") or "")
        if image_id is None or not relative_path or image_id in image_paths:
            raise ValueError("invalid or duplicate HSV image reference")
        image_paths[image_id] = relative_path

    output: list[Candidate] = []
    for index, value in enumerate(annotations):
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid HSV annotation at index {index}")
        image_id = value.get("image_id")
        if image_id not in image_paths:
            raise ValueError(f"unknown HSV image reference: {image_id}")
        raw_attributes = value.get("attributes")
        attributes: Mapping[str, Any] = (
            raw_attributes if isinstance(raw_attributes, Mapping) else {}
        )
        algorithm = str(attributes.get("algorithm_version") or "hsv-unknown")
        annotation_id = str(value.get("id") or index)
        score = float(
            attributes.get("candidate_confidence", value.get("score", 0.5))
        )
        output.append(
            Candidate(
                relative_path=image_paths[image_id],
                source_id=f"hsv:{algorithm}:{annotation_id}",
                source_family="hsv",
                category="fastener",
                xyxy=_xywh_to_xyxy(value.get("bbox")),
                score=score,
            )
        )
    return output


def _validate_box(box: Box) -> None:
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid xyxy box: {box}")


@dataclass(frozen=True)
class Candidate:
    relative_path: str
    source_id: str
    source_family: str
    category: str
    xyxy: Box
    score: float

    def __post_init__(self) -> None:
        if not self.relative_path or not self.source_id:
            raise ValueError("candidate path and source ID are required")
        if self.source_family not in VALID_SOURCE_FAMILIES:
            raise ValueError(f"unsupported source family: {self.source_family}")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"unsupported category: {self.category}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"candidate score outside [0, 1]: {self.score}")
        _validate_box(self.xyxy)

    def stable_id(self) -> str:
        raw = (
            f"{self.relative_path}|{self.source_id}|{self.source_family}|"
            f"{self.category}|{self.xyxy}|{self.score:.8f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FusedCandidate:
    relative_path: str
    category: str | None
    xyxy: Box
    member_ids: tuple[str, ...]
    supporting_families: tuple[str, ...]
    consensus_status: str

    def stable_id(self) -> str:
        raw = f"{self.relative_path}|{self.category}|{self.member_ids}|{self.xyxy}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def iou_xyxy(left: Box, right: Box) -> float:
    _validate_box(left)
    _validate_box(right)
    intersection = _intersection_area(left, right)
    left_area = _box_area(left)
    right_area = _box_area(right)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _box_area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(left: Box, right: Box) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _is_geometric_duplicate(left: Box, right: Box, iou_threshold: float) -> bool:
    if iou_xyxy(left, right) >= iou_threshold:
        return True
    left_area = _box_area(left)
    right_area = _box_area(right)
    smaller = left if left_area <= right_area else right
    smaller_area = min(left_area, right_area)
    if (
        _intersection_area(left, right) / smaller_area
        < DEFAULT_CONTAINMENT_THRESHOLD
    ):
        return False
    if max(left_area, right_area) / smaller_area > DEFAULT_MAX_AREA_RATIO:
        return False
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    center_distance = (
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    smaller_diagonal = (
        (smaller[2] - smaller[0]) ** 2 + (smaller[3] - smaller[1]) ** 2
    ) ** 0.5
    return center_distance / smaller_diagonal <= DEFAULT_CENTER_DISTANCE_THRESHOLD


def _weighted_box(cluster: list[Candidate]) -> Box:
    total = sum(row.score for row in cluster)
    weights = [row.score for row in cluster]
    if total == 0.0:
        weights = [1.0] * len(cluster)
        total = float(len(cluster))
    return tuple(
        sum(row.xyxy[index] * weight for row, weight in zip(cluster, weights)) / total
        for index in range(4)
    )  # type: ignore[return-value]


def _complete_link_cluster(
    seed: Candidate,
    pending: list[Candidate],
    iou_threshold: float,
) -> tuple[list[Candidate], list[Candidate]]:
    cluster = [seed]
    remaining: list[Candidate] = []
    for row in pending:
        matches = row.relative_path == seed.relative_path and all(
            _is_geometric_duplicate(row.xyxy, member.xyxy, iou_threshold)
            for member in cluster
        )
        if matches:
            cluster.append(row)
        else:
            remaining.append(row)
    return cluster, remaining


def fuse_candidates(
    rows: list[Candidate],
    iou_threshold: float = 0.55,
) -> list[FusedCandidate]:
    """Cluster overlapping candidates while preserving independent source families."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    pending = sorted(
        rows,
        key=lambda row: (
            row.relative_path,
            row.xyxy,
            row.category,
            row.source_id,
        ),
    )
    output: list[FusedCandidate] = []
    while pending:
        seed = pending.pop(0)
        cluster, pending = _complete_link_cluster(seed, pending, iou_threshold)
        families = tuple(sorted({row.source_family for row in cluster}))
        categories = {row.category for row in cluster}
        if len(categories) > 1:
            category = None
            status = "conflict"
        else:
            category = next(iter(categories))
            if len(families) > 1:
                status = "consensus_high"
            elif families == ("temporal",):
                status = "propagated"
            else:
                status = "single_source"
        output.append(
            FusedCandidate(
                relative_path=seed.relative_path,
                category=category,
                xyxy=_weighted_box(cluster),
                member_ids=tuple(sorted(row.stable_id() for row in cluster)),
                supporting_families=families,
                consensus_status=status,
            )
        )
    return output
