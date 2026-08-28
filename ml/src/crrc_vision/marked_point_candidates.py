"""Lossless, source-preserving union for marked-point proposal branches."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Proposal:
    relative_path: str
    proposal_id: str
    source: str
    xyxy: Box
    score: float
    image_id: object | None = None
    geometry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedProposal:
    relative_path: str
    fused_id: str
    xyxy: Box
    member_ids: tuple[str, ...]
    sources: tuple[str, ...]
    score: float
    image_id: object | None
    member_geometry: tuple[dict[str, Any], ...]


def _validate(proposal: Proposal) -> None:
    if not proposal.relative_path or not proposal.proposal_id or not proposal.source:
        raise ValueError("INVALID_PROPOSAL_IDENTITY")
    if len(proposal.xyxy) != 4 or not all(
        math.isfinite(float(value)) for value in proposal.xyxy
    ):
        raise ValueError(f"INVALID_PROPOSAL_BOX:{proposal.proposal_id}")
    x1, y1, x2, y2 = proposal.xyxy
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"INVALID_PROPOSAL_BOX:{proposal.proposal_id}")
    if not math.isfinite(float(proposal.score)):
        raise ValueError(f"INVALID_PROPOSAL_SCORE:{proposal.proposal_id}")


def _iou(left: Box, right: Box) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _fuse(cluster: list[Proposal]) -> FusedProposal:
    members = sorted(cluster, key=lambda row: (row.source, row.proposal_id))
    member_ids = tuple(row.proposal_id for row in members)
    relative_path = members[0].relative_path
    fused_id = hashlib.sha256(
        f"{relative_path}|{'|'.join(member_ids)}".encode("utf-8")
    ).hexdigest()[:16]
    image_ids = {row.image_id for row in members if row.image_id is not None}
    if len(image_ids) > 1:
        raise ValueError(f"IMAGE_ID_MISMATCH:{relative_path}")
    return FusedProposal(
        relative_path=relative_path,
        fused_id=fused_id,
        xyxy=(
            min(row.xyxy[0] for row in members),
            min(row.xyxy[1] for row in members),
            max(row.xyxy[2] for row in members),
            max(row.xyxy[3] for row in members),
        ),
        member_ids=member_ids,
        sources=tuple(sorted({row.source for row in members})),
        score=max(float(row.score) for row in members),
        image_id=next(iter(image_ids), None),
        member_geometry=tuple(dict(row.geometry) for row in members),
    )


def union_proposals(
    proposals: list[Proposal], *, iou_threshold: float = 0.60
) -> list[FusedProposal]:
    """Cluster within images using complete-link IoU without score filtering."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("INVALID_IOU_THRESHOLD")
    seen_ids: set[str] = set()
    by_image: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        _validate(proposal)
        if proposal.proposal_id in seen_ids:
            raise ValueError(f"DUPLICATE_PROPOSAL_ID:{proposal.proposal_id}")
        seen_ids.add(proposal.proposal_id)
        by_image[proposal.relative_path].append(proposal)

    output: list[FusedProposal] = []
    for relative_path in sorted(by_image):
        ordered = sorted(
            by_image[relative_path],
            key=lambda row: (row.xyxy, row.source, row.proposal_id),
        )
        clusters: list[list[Proposal]] = []
        for proposal in ordered:
            destination = next(
                (
                    cluster
                    for cluster in clusters
                    if all(
                        _iou(proposal.xyxy, member.xyxy) >= iou_threshold
                        for member in cluster
                    )
                ),
                None,
            )
            if destination is None:
                clusters.append([proposal])
            else:
                destination.append(proposal)
        output.extend(_fuse(cluster) for cluster in clusters)
    return sorted(output, key=lambda row: (row.relative_path, row.xyxy, row.fused_id))
