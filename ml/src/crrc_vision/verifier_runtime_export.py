from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence


def verifier_export_contract(
    checkpoint: Mapping[str, object],
) -> tuple[int, tuple[str, ...]]:
    """Validate the deployment-sensitive verifier checkpoint fields."""
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("VERIFIER_ARCHITECTURE_UNSUPPORTED")
    classes = tuple(str(value) for value in checkpoint.get("classes", ()))
    if (
        len(classes) != 2
        or len(set(classes)) != len(classes)
        or "marked_point" not in classes
        or "not_marked_point" not in classes
    ):
        raise ValueError("VERIFIER_CLASSES_INVALID")
    input_size = int(checkpoint.get("input_size", 0))
    if input_size <= 0:
        raise ValueError("VERIFIER_INPUT_SIZE_INVALID")
    return input_size, classes


def compare_verifier_scores(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    threshold: float,
    maximum_drift: float,
) -> dict[str, object]:
    """Compare runtime scores, including deployment-threshold decisions."""
    if len(baseline) != len(candidate):
        raise ValueError("VERIFIER_SCORE_COUNT_MISMATCH")
    drifts = [abs(float(left) - float(right)) for left, right in zip(baseline, candidate)]
    decision_mismatches = sum(
        (float(left) >= threshold) != (float(right) >= threshold)
        for left, right in zip(baseline, candidate)
    )
    observed_drift = max(drifts, default=0.0)
    return {
        "count": len(baseline),
        "maximum_score_drift": observed_drift,
        "decision_mismatches": decision_mismatches,
        "passed": observed_drift <= maximum_drift and decision_mismatches == 0,
    }
