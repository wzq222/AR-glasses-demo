"""Deterministic multi-source candidate fusion for safe automatic labeling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


VALID_CATEGORIES = {"fastener", "pipe_joint"}
VALID_SOURCE_FAMILIES = {"reference_teacher", "hsv", "temporal", "student"}


Box = tuple[float, float, float, float]


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
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


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


def _connected_cluster(
    seed: Candidate,
    pending: list[Candidate],
    iou_threshold: float,
) -> tuple[list[Candidate], list[Candidate]]:
    cluster = [seed]
    changed = True
    while changed:
        changed = False
        remaining: list[Candidate] = []
        for row in pending:
            matches = row.relative_path == seed.relative_path and any(
                iou_xyxy(row.xyxy, member.xyxy) >= iou_threshold
                for member in cluster
            )
            if matches:
                cluster.append(row)
                changed = True
            else:
                remaining.append(row)
        pending = remaining
    return cluster, pending


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
        cluster, pending = _connected_cluster(seed, pending, iou_threshold)
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
