"""Safety-first fusion for anti-loosening witness-mark state evidence.

This module evaluates relative displacement indicated by a witness mark.  It
does not claim bolt preload, remaining torque, or absolute mechanical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot, isfinite

from .witness_state_contract import MARK_ROLES, TOPOLOGIES


Point = tuple[float, float]
Segment = tuple[Point, Point]
SUPPORTED_PLANAR_TOPOLOGIES = frozenset({"bolt_head_plate", "nut_plate"})


@dataclass(frozen=True)
class StateThresholds:
    maximum_angle_degrees: float
    maximum_gap_ratio: float
    maximum_residual_ratio: float
    learned_displaced_threshold: float
    baseline_displaced_threshold: float
    damaged_mark_threshold: float
    calibrated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.calibrated, bool):
            raise ValueError("calibrated must be boolean")
        values = (
            self.maximum_angle_degrees,
            self.maximum_gap_ratio,
            self.maximum_residual_ratio,
            self.learned_displaced_threshold,
            self.baseline_displaced_threshold,
            self.damaged_mark_threshold,
        )
        if self.calibrated and (
            not all(isfinite(value) for value in values)
            or not 0.0 <= self.maximum_angle_degrees <= 90.0
            or self.maximum_gap_ratio < 0.0
            or self.maximum_residual_ratio < 0.0
            or not all(0.0 <= value <= 1.0 for value in values[3:])
        ):
            raise ValueError("state thresholds are outside valid ranges")

    @classmethod
    def uncalibrated(cls) -> "StateThresholds":
        return cls(*(float("nan"),) * 6, calibrated=False)


@dataclass(frozen=True)
class StateEvidence:
    topology: str
    mark_role: str
    quality_pass: bool
    fixed_segment_xyxy: Segment | None
    moving_segment_xyxy: Segment | None
    fixed_segment_confidence: float
    moving_segment_confidence: float
    reference_size: float
    learned_displaced_score: float | None = None
    baseline_displaced_score: float | None = None
    damaged_mark_score: float | None = None


@dataclass(frozen=True)
class WitnessStateDecision:
    state: str
    reason: str
    review_hint: str | None
    angle_degrees: float | None
    gap_ratio: float | None
    residual_ratio: float | None
    geometry_abnormal: bool | None
    supporting_sources: tuple[str, ...]


def _insufficient(reason: str, *, hint: str | None = None) -> WitnessStateDecision:
    return WitnessStateDecision(
        state="INSUFFICIENT",
        reason=reason,
        review_hint=hint,
        angle_degrees=None,
        gap_ratio=None,
        residual_ratio=None,
        geometry_abnormal=None,
        supporting_sources=(),
    )


def _distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _geometry(fixed: Segment, moving: Segment, reference_size: float) -> tuple[float, float, float]:
    fixed_vector = (fixed[1][0] - fixed[0][0], fixed[1][1] - fixed[0][1])
    moving_vector = (moving[1][0] - moving[0][0], moving[1][1] - moving[0][1])
    fixed_length = hypot(*fixed_vector)
    moving_length = hypot(*moving_vector)
    if fixed_length < 1.0e-6 or moving_length < 1.0e-6:
        raise ValueError("SEGMENT_LENGTH_INVALID")
    cosine = abs(
        (fixed_vector[0] * moving_vector[0] + fixed_vector[1] * moving_vector[1])
        / (fixed_length * moving_length)
    )
    angle = degrees(acos(max(-1.0, min(1.0, cosine))))
    gap = min(_distance(left, right) for left in fixed for right in moving) / reference_size

    points = (*fixed, *moving)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    xx = sum((point[0] - center_x) ** 2 for point in points)
    yy = sum((point[1] - center_y) ** 2 for point in points)
    xy = sum((point[0] - center_x) * (point[1] - center_y) for point in points)
    # The smaller eigenvalue is mean squared orthogonal distance to the best-fit line.
    trace = xx + yy
    discriminant = max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy) ** 0.5
    residual = max(0.0, (trace - discriminant) / (2.0 * len(points))) ** 0.5
    return angle, gap, residual / reference_size


def _valid_score(value: float | None) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_segment_coordinates(segment: Segment) -> bool:
    try:
        return len(segment) == 2 and all(
            len(point) == 2
            and all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and isfinite(coordinate)
                for coordinate in point
            )
            for point in segment
        )
    except (TypeError, ValueError):
        return False


def evaluate_witness_state(
    evidence: StateEvidence,
    thresholds: StateThresholds,
) -> WitnessStateDecision:
    """Fuse topology, geometry, learned state and optional historical evidence.

    A geometry anomaly is one evidence source, even when several correlated
    residuals exceed threshold.  `DISPLACED` therefore needs geometry plus an
    independent learned or historical-baseline source.  A lone anomaly is
    routed to human review as `POSSIBLE_DISPLACED`.
    """
    if evidence.topology not in TOPOLOGIES or evidence.topology == "unknown":
        return _insufficient("TOPOLOGY_UNKNOWN")
    if evidence.topology not in SUPPORTED_PLANAR_TOPOLOGIES:
        return _insufficient("TOPOLOGY_SOLVER_UNAVAILABLE")
    if evidence.mark_role not in MARK_ROLES or evidence.mark_role != "bridges_moving_fixed":
        return _insufficient("MARK_DOES_NOT_BRIDGE_MOVING_FIXED")
    if not isinstance(evidence.quality_pass, bool):
        return _insufficient("IMAGE_QUALITY_INVALID")
    if evidence.quality_pass is not True:
        return _insufficient("IMAGE_QUALITY_FAILED")
    if evidence.fixed_segment_xyxy is None or evidence.moving_segment_xyxy is None:
        return _insufficient("SEGMENTS_MISSING")
    if not _valid_segment_coordinates(evidence.fixed_segment_xyxy) or not _valid_segment_coordinates(
        evidence.moving_segment_xyxy
    ):
        return _insufficient("SEGMENT_COORDINATE_INVALID")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and 0.0 <= value <= 1.0
        for value in (
            evidence.fixed_segment_confidence,
            evidence.moving_segment_confidence,
        )
    ):
        return _insufficient("SEGMENT_CONFIDENCE_INVALID")
    if min(evidence.fixed_segment_confidence, evidence.moving_segment_confidence) < 0.60:
        return _insufficient("SEGMENT_CONFIDENCE_LOW")
    if (
        not isinstance(evidence.reference_size, (int, float))
        or isinstance(evidence.reference_size, bool)
        or not isfinite(evidence.reference_size)
        or evidence.reference_size <= 0.0
    ):
        return _insufficient("REFERENCE_SIZE_INVALID")
    if not all(
        _valid_score(value)
        for value in (
            evidence.learned_displaced_score,
            evidence.baseline_displaced_score,
            evidence.damaged_mark_score,
        )
    ):
        return _insufficient("EVIDENCE_SCORE_INVALID")
    if not thresholds.calibrated:
        return _insufficient("THRESHOLDS_UNCALIBRATED")

    try:
        angle, gap, residual = _geometry(
            evidence.fixed_segment_xyxy,
            evidence.moving_segment_xyxy,
            evidence.reference_size,
        )
    except (ValueError, OverflowError):
        return _insufficient("GEOMETRY_VALUE_INVALID")
    if not all(isfinite(value) for value in (angle, gap, residual)):
        return _insufficient("GEOMETRY_VALUE_INVALID")

    geometry_abnormal = (
        angle > thresholds.maximum_angle_degrees
        or gap > thresholds.maximum_gap_ratio
        or residual > thresholds.maximum_residual_ratio
    )
    sources: list[str] = []
    if geometry_abnormal:
        sources.append("geometry")
    if (
        evidence.learned_displaced_score is not None
        and evidence.learned_displaced_score >= thresholds.learned_displaced_threshold
    ):
        sources.append("learned_state")
    if (
        evidence.baseline_displaced_score is not None
        and evidence.baseline_displaced_score >= thresholds.baseline_displaced_threshold
    ):
        sources.append("historical_baseline")

    common = {
        "angle_degrees": angle,
        "gap_ratio": gap,
        "residual_ratio": residual,
        "geometry_abnormal": geometry_abnormal,
        "supporting_sources": tuple(sources),
    }
    damage_active = evidence.damaged_mark_score is not None and (
        evidence.damaged_mark_score >= thresholds.damaged_mark_threshold
    )
    if damage_active and sources:
        return WitnessStateDecision(
            "INSUFFICIENT", "DAMAGE_DISPLACEMENT_CONFLICT", None, **common
        )
    if geometry_abnormal and len(sources) >= 2:
        return WitnessStateDecision(
            "DISPLACED", "CORROBORATED_RIGID_DISPLACEMENT", None, **common
        )
    if sources:
        return WitnessStateDecision(
            "INSUFFICIENT",
            "DISPLACEMENT_REQUIRES_CORROBORATION",
            "POSSIBLE_DISPLACED",
            **common,
        )
    if damage_active:
        return WitnessStateDecision(
            "DAMAGED_MARK",
            "MARK_DAMAGE_WITHOUT_RIGID_DISPLACEMENT",
            None,
            **common,
        )
    return WitnessStateDecision(
        "ALIGNED", "GEOMETRY_ALIGNED_NO_CONFLICT", None, **common
    )
