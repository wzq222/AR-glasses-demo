"""Robust per-inspection-point control limits for witness-mark angles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable


MAD_NORMAL_SCALE = 1.4826


@dataclass(frozen=True)
class BaselineControlLimit:
    ready: bool
    reason: str
    sample_count: int
    threshold_degrees: float | None
    median_abs_delta_degrees: float | None
    mad_degrees: float | None


def estimate_baseline_control_limit(
    delta_angles_degrees: Iterable[float],
    *,
    minimum_samples: int = 5,
    floor_degrees: float = 3.0,
    sigma_multiplier: float = 3.0,
) -> BaselineControlLimit:
    """Estimate ``max(floor, median(abs(delta)) + k * 1.4826 * MAD)``.

    Inputs must be repeated healthy captures of one physical inspection point.
    The function does not infer that a capture is healthy and does not create
    state truth by itself.
    """
    if not isinstance(minimum_samples, int) or isinstance(minimum_samples, bool) or minimum_samples < 1:
        raise ValueError("minimum_samples must be a positive integer")
    for name, value in (
        ("floor_degrees", floor_degrees),
        ("sigma_multiplier", sigma_multiplier),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")

    absolute_deltas: list[float] = []
    for value in delta_angles_degrees:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
        ):
            raise ValueError("baseline angle values must be finite numbers")
        absolute = abs(float(value))
        if absolute > 90.0:
            raise ValueError("baseline angle values must be within 0..90 degrees")
        absolute_deltas.append(absolute)

    count = len(absolute_deltas)
    center = median(absolute_deltas) if absolute_deltas else None
    mad = (
        median(abs(value - center) for value in absolute_deltas)
        if center is not None
        else None
    )
    if count < minimum_samples:
        return BaselineControlLimit(
            ready=False,
            reason="BASELINE_SAMPLE_COUNT_INSUFFICIENT",
            sample_count=count,
            threshold_degrees=None,
            median_abs_delta_degrees=center,
            mad_degrees=mad,
        )

    assert center is not None and mad is not None
    threshold = max(
        float(floor_degrees),
        center + float(sigma_multiplier) * MAD_NORMAL_SCALE * mad,
    )
    return BaselineControlLimit(
        ready=True,
        reason="BASELINE_READY",
        sample_count=count,
        threshold_degrees=threshold,
        median_abs_delta_degrees=center,
        mad_degrees=mad,
    )
