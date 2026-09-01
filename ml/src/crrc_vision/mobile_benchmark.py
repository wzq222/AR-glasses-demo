"""Recall-first contracts for mobile detector benchmark candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


PINNED_NCNN_COMMIT = "2130e00c6efd910d3e926867ca94a2d96eaf9d31"
PINNED_MNN_COMMIT = "47a656efa06ba24937e800719ecbc2165806191e"
PINNED_RUNTIME_REVISIONS = {
    "ncnn": PINNED_NCNN_COMMIT,
    "mnn": PINNED_MNN_COMMIT,
}


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


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    missing: int
    unexpected: int
    max_coordinate_drift: float
    max_score_drift: float
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


def _bbox_iou(first: list[float], second: list[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("INVALID_BBOX")
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def compare_predictions(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    iou_threshold: float,
    *,
    coordinate_tolerance: float = 1.0,
    score_tolerance: float = 0.01,
) -> ParityResult:
    """Compare post-NMS detections with deterministic same-image one-to-one matching."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("INVALID_IOU_THRESHOLD")
    baseline_images = {int(item["image_id"]) for item in baseline}
    unknown_indices = {
        index
        for index, item in enumerate(candidate)
        if int(item["image_id"]) not in baseline_images
    }
    used: set[int] = set()
    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    missing = 0
    for reference in baseline:
        reference_image = int(reference["image_id"])
        reference_category = int(reference.get("category_id", 0))
        reference_box = [float(value) for value in reference["bbox"]]  # type: ignore[index]
        choices: list[tuple[float, int]] = []
        for index, proposal in enumerate(candidate):
            if index in used or index in unknown_indices:
                continue
            if int(proposal["image_id"]) != reference_image:
                continue
            if int(proposal.get("category_id", 0)) != reference_category:
                continue
            proposal_box = [float(value) for value in proposal["bbox"]]  # type: ignore[index]
            overlap = _bbox_iou(reference_box, proposal_box)
            if overlap >= iou_threshold:
                choices.append((overlap, index))
        if not choices:
            missing += 1
            continue
        _, match_index = max(choices, key=lambda item: (item[0], -item[1]))
        used.add(match_index)
        matched.append((reference, candidate[match_index]))

    unmatched_known = {
        index
        for index, proposal in enumerate(candidate)
        if index not in used and index not in unknown_indices
    }
    coordinate_drifts: list[float] = []
    score_drifts: list[float] = []
    for reference, proposal in matched:
        reference_box = [float(value) for value in reference["bbox"]]  # type: ignore[index]
        proposal_box = [float(value) for value in proposal["bbox"]]  # type: ignore[index]
        coordinate_drifts.append(
            max(abs(left - right) for left, right in zip(reference_box, proposal_box))
        )
        score_drifts.append(abs(float(reference["score"]) - float(proposal["score"])))

    max_coordinate_drift = max(coordinate_drifts, default=0.0)
    max_score_drift = max(score_drifts, default=0.0)
    reasons: list[str] = []
    if unknown_indices:
        reasons.append("UNKNOWN_IMAGE")
    if missing:
        reasons.append("MISSING_DETECTION")
    if unmatched_known:
        reasons.append("UNEXPECTED_DETECTION")
    if max_coordinate_drift > coordinate_tolerance:
        reasons.append("COORDINATE_DRIFT_TOO_HIGH")
    if max_score_drift > score_tolerance:
        reasons.append("SCORE_DRIFT_TOO_HIGH")
    return ParityResult(
        passed=not reasons,
        missing=missing,
        unexpected=len(unknown_indices) + len(unmatched_known),
        max_coordinate_drift=max_coordinate_drift,
        max_score_drift=max_score_drift,
        reasons=tuple(reasons),
    )


def build_parity_report(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    baseline_path: Path,
    candidate_path: Path,
    iou_threshold: float,
) -> dict[str, object]:
    """Build a hash-bound, fail-closed parity report."""

    result = compare_predictions(baseline, candidate, iou_threshold)
    return {
        "schema_version": "mobile-runtime-parity-v1",
        "status": "parity_passed" if result.passed else "parity_failed",
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "candidate_predictions_sha256": sha256_file(candidate_path),
        "iou_threshold": iou_threshold,
        "baseline_detection_count": len(baseline),
        "candidate_detection_count": len(candidate),
        "missing": result.missing,
        "unexpected": result.unexpected,
        "max_coordinate_drift": result.max_coordinate_drift,
        "max_score_drift": result.max_score_drift,
        "reasons": list(result.reasons),
    }


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 for a regular file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def prepare_benchmark_manifest(
    candidate: str,
    model_path: Path,
    formal_truth_path: Path,
    expected_truth_sha256: str,
    runtime_name: str,
    runtime_revision: str,
) -> dict[str, object]:
    """Bind one candidate to immutable model, runtime, and truth inputs."""

    if not candidate.strip():
        raise ValueError("CANDIDATE_REQUIRED")
    expected_revision = PINNED_RUNTIME_REVISIONS.get(runtime_name)
    if expected_revision is None:
        raise ValueError("UNKNOWN_RUNTIME")
    if runtime_revision != expected_revision:
        raise ValueError("RUNTIME_REVISION_MISMATCH")
    model_hash = sha256_file(model_path)
    truth_hash = sha256_file(formal_truth_path)
    if truth_hash != expected_truth_sha256.upper():
        raise ValueError("FORMAL_TRUTH_HASH_MISMATCH")
    return {
        "schema_version": "mobile-detector-benchmark-v1",
        "status": "prepared",
        "candidate": candidate,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_hash,
        "formal_truth_path": str(formal_truth_path.resolve()),
        "formal_truth_sha256": truth_hash,
        "runtime_name": runtime_name,
        "runtime_revision": runtime_revision,
    }
