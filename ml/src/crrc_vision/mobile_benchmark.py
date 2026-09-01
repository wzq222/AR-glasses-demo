"""Recall-first contracts for mobile detector benchmark candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkGate:
    baseline_recall: float
    baseline_complete_scene_recall: float
    minimum_precision: float
    maximum_hot_p95_ms: float
    minimum_hot_fps: float
    maximum_pss_mb: float


@dataclass(frozen=True)
class CandidateMetrics:
    recall: float
    precision: float
    complete_scene_recall: float
    hot_p95_ms: float
    hot_fps: float
    pss_mb: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_candidate(
    metrics: CandidateMetrics,
    gate: BenchmarkGate,
) -> GateResult:
    """Apply accuracy gates before mobile performance gates in stable order."""

    reasons: list[str] = []
    if metrics.recall < gate.baseline_recall - 0.01:
        reasons.append("RECALL_BELOW_BASELINE")
    if metrics.precision < gate.minimum_precision:
        reasons.append("PRECISION_BELOW_MINIMUM")
    if metrics.complete_scene_recall < gate.baseline_complete_scene_recall:
        reasons.append("COMPLETE_SCENE_RECALL_BELOW_BASELINE")
    if metrics.hot_p95_ms >= gate.maximum_hot_p95_ms:
        reasons.append("HOT_P95_TOO_SLOW")
    if metrics.hot_fps < gate.minimum_hot_fps:
        reasons.append("HOT_FPS_TOO_LOW")
    if metrics.pss_mb > gate.maximum_pss_mb:
        reasons.append("PSS_TOO_HIGH")
    return GateResult(passed=not reasons, reasons=tuple(reasons))
