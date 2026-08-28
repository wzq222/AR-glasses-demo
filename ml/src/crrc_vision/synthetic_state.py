from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot


Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class StateBands:
    normal: tuple[float, float] = (0.0, 3.0)
    slight: tuple[float, float] = (6.0, 12.0)
    obvious: tuple[float, float] = (18.0, 35.0)


@dataclass(frozen=True)
class StateAudit:
    accepted: bool
    declared_state: str
    measured_state: str
    angle_deg: float | None
    relative_offset_px: float | None
    reason: str


def _vector(segment: Segment) -> tuple[float, float]:
    (x1, y1), (x2, y2) = segment
    return x2 - x1, y2 - y1


def relative_angle_deg(fixed: Segment, moving: Segment) -> float:
    fixed_dx, fixed_dy = _vector(fixed)
    moving_dx, moving_dy = _vector(moving)
    if hypot(fixed_dx, fixed_dy) < 1e-6 or hypot(moving_dx, moving_dy) < 1e-6:
        raise ValueError("防松线线段退化")
    fixed_angle = degrees(atan2(fixed_dy, fixed_dx))
    moving_angle = degrees(atan2(moving_dy, moving_dx))
    difference = abs((moving_angle - fixed_angle) % 180.0)
    return min(difference, 180.0 - difference)


def relative_offset_px(fixed: Segment, moving: Segment) -> float:
    return hypot(moving[0][0] - fixed[1][0], moving[0][1] - fixed[1][1])


def classify_state(angle_deg: float, bands: StateBands = StateBands()) -> str:
    angle = abs(float(angle_deg))
    if bands.normal[0] <= angle <= bands.normal[1]:
        return "NORMAL"
    if bands.slight[0] <= angle <= bands.slight[1]:
        return "SLIGHT_LOOSE"
    if bands.obvious[0] <= angle <= bands.obvious[1]:
        return "OBVIOUS_LOOSE"
    return "UNCERTAIN"


def validate_state(
    declared_state: str,
    fixed: Segment,
    moving: Segment,
    bands: StateBands = StateBands(),
) -> StateAudit:
    try:
        angle = relative_angle_deg(fixed, moving)
        offset = relative_offset_px(fixed, moving)
    except ValueError as exc:
        return StateAudit(False, declared_state, "UNCERTAIN", None, None, str(exc))
    measured = classify_state(angle, bands)
    accepted = declared_state == measured
    reason = "状态几何一致" if accepted else f"声明{declared_state}但端点重算为{measured}"
    return StateAudit(accepted, declared_state, measured, angle, offset, reason)
