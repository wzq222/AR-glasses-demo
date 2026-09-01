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
