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
DEFAULT_HSV_ANCHOR_EXPANSION = 0.05
DEFAULT_ANCHOR_ASSIGNMENT_MARGIN = 0.10
DEFAULT_CLUSTER_RECONCILIATION_IOU = 0.75


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


def _candidate_sort_key(row: Candidate) -> tuple[object, ...]:
    return (row.relative_path, row.xyxy, row.category, row.source_id)


def _representative_cluster(
    seed: Candidate,
    pending: list[Candidate],
    iou_threshold: float,
) -> tuple[list[Candidate], list[Candidate]]:
    cluster = [seed]
    remaining: list[Candidate] = []
    for row in pending:
        matches = row.relative_path == seed.relative_path and (
            _is_geometric_duplicate(
                row.xyxy,
                _weighted_box(cluster),
                iou_threshold,
            )
        )
        if matches:
            cluster.append(row)
        else:
            remaining.append(row)
    return cluster, remaining


def _geometric_clusters(
    rows: list[Candidate],
    iou_threshold: float,
) -> list[list[Candidate]]:
    pending = sorted(rows, key=_candidate_sort_key)
    clusters: list[list[Candidate]] = []
    while pending:
        seed = pending.pop(0)
        cluster, pending = _representative_cluster(seed, pending, iou_threshold)
        clusters.append(cluster)
    return clusters


def _box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_diagonal(box: Box) -> float:
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def _normalized_center_distance(left: Box, right: Box) -> float:
    left_center = _box_center(left)
    right_center = _box_center(right)
    distance = (
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    return distance / _box_diagonal(right)


def _normalized_point_to_box_distance(point_box: Box, anchor_box: Box) -> float:
    x, y = _box_center(point_box)
    dx = max(anchor_box[0] - x, 0.0, x - anchor_box[2])
    dy = max(anchor_box[1] - y, 0.0, y - anchor_box[3])
    return (dx**2 + dy**2) ** 0.5 / _box_diagonal(anchor_box)


def _select_unique_anchor(
    row: Candidate,
    anchor_clusters: list[list[Candidate]],
    anchor_boxes: list[Box],
    iou_threshold: float,
) -> int | None:
    matches: list[tuple[float, int]] = []
    for index, (cluster, anchor_box) in enumerate(
        zip(anchor_clusters, anchor_boxes)
    ):
        if row.relative_path != cluster[0].relative_path:
            continue
        geometric_match = _is_geometric_duplicate(
            row.xyxy,
            anchor_box,
            iou_threshold,
        )
        marker_match = row.source_family == "hsv" and (
            _normalized_point_to_box_distance(row.xyxy, anchor_box)
            <= DEFAULT_HSV_ANCHOR_EXPANSION
        )
        if geometric_match or marker_match:
            matches.append(
                (_normalized_center_distance(row.xyxy, anchor_box), index)
            )
    matches.sort()
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][1]
    if matches[1][0] - matches[0][0] >= DEFAULT_ANCHOR_ASSIGNMENT_MARGIN:
        return matches[0][1]
    return None


def _reconcile_clusters(
    clusters: list[list[Candidate]],
) -> list[list[Candidate]]:
    pending = list(clusters)
    output: list[list[Candidate]] = []
    while pending:
        cluster = pending.pop(0)
        anchor_box = _weighted_box(cluster)
        remaining: list[list[Candidate]] = []
        for other in pending:
            matches = (
                other[0].relative_path == cluster[0].relative_path
                and iou_xyxy(anchor_box, _weighted_box(other))
                >= DEFAULT_CLUSTER_RECONCILIATION_IOU
            )
            if matches:
                cluster.extend(other)
            else:
                remaining.append(other)
        output.append(cluster)
        pending = remaining
    return output


def _clusters_for_path(
    rows: list[Candidate],
    iou_threshold: float,
) -> list[list[Candidate]]:
    teacher_rows = [
        row for row in rows if row.source_family == "reference_teacher"
    ]
    other_rows = [
        row for row in rows if row.source_family != "reference_teacher"
    ]
    anchor_clusters = _geometric_clusters(teacher_rows, iou_threshold)
    anchor_boxes = [_weighted_box(cluster) for cluster in anchor_clusters]
    residual: list[Candidate] = []
    for row in sorted(other_rows, key=_candidate_sort_key):
        anchor_index = _select_unique_anchor(
            row,
            anchor_clusters,
            anchor_boxes,
            iou_threshold,
        )
        if anchor_index is None:
            residual.append(row)
        else:
            anchor_clusters[anchor_index].append(row)
    return _reconcile_clusters(
        anchor_clusters + _geometric_clusters(residual, iou_threshold)
    )


def fuse_candidates(
    rows: list[Candidate],
    iou_threshold: float = 0.55,
) -> list[FusedCandidate]:
    """Cluster overlapping candidates while preserving independent source families."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    rows_by_path: dict[str, list[Candidate]] = {}
    for row in rows:
        rows_by_path.setdefault(row.relative_path, []).append(row)
    output: list[FusedCandidate] = []
    clusters: list[list[Candidate]] = []
    for relative_path in sorted(rows_by_path):
        clusters.extend(
            _clusters_for_path(rows_by_path[relative_path], iou_threshold)
        )
    for cluster in clusters:
        seed = cluster[0]
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


def fusion_stats(
    rows: list[Candidate],
    fused: list[FusedCandidate],
) -> dict[str, int | dict[str, int]]:
    """Summarize physical-candidate reduction for an audit manifest."""

    histogram: dict[str, int] = {}
    for item in fused:
        key = str(len(item.member_ids))
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "raw_candidates": len(rows),
        "fused_candidates": len(fused),
        "candidate_reduction": len(rows) - len(fused),
        "cluster_size_histogram": dict(
            sorted(histogram.items(), key=lambda row: int(row[0]))
        ),
    }
