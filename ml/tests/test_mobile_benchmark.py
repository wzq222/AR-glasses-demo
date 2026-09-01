import hashlib

import pytest

from crrc_vision.mobile_benchmark import (
    BenchmarkGate,
    CandidateMetrics,
    PINNED_MNN_COMMIT,
    PINNED_NCNN_COMMIT,
    build_parity_report,
    compare_predictions,
    evaluate_candidate,
    prepare_benchmark_manifest,
)


def _gate() -> BenchmarkGate:
    return BenchmarkGate(
        baseline_recall=0.584,
        baseline_complete_scene_recall=0.0,
        minimum_precision=0.60,
        maximum_hot_p95_ms=500.0,
        minimum_hot_fps=2.0,
        maximum_pss_mb=250.0,
    )


def test_rejects_fast_candidate_when_recall_drops() -> None:
    metrics = CandidateMetrics(
        recall=0.55,
        precision=0.90,
        complete_scene_recall=0.10,
        hot_p95_ms=100.0,
        hot_fps=8.0,
        pss_mb=100.0,
    )

    result = evaluate_candidate(metrics, _gate())

    assert result.passed is False
    assert result.reasons == ("RECALL_BELOW_BASELINE",)


def test_accepts_candidate_only_when_accuracy_and_hot_gates_pass() -> None:
    metrics = CandidateMetrics(
        recall=0.60,
        precision=0.65,
        complete_scene_recall=0.10,
        hot_p95_ms=420.0,
        hot_fps=2.2,
        pss_mb=220.0,
    )

    result = evaluate_candidate(metrics, _gate())

    assert result.passed is True
    assert result.reasons == ()


def test_hot_p95_limit_is_exclusive() -> None:
    metrics = CandidateMetrics(
        recall=0.60,
        precision=0.65,
        complete_scene_recall=0.10,
        hot_p95_ms=500.0,
        hot_fps=2.2,
        pss_mb=220.0,
    )

    result = evaluate_candidate(metrics, _gate())

    assert result.reasons == ("HOT_P95_TOO_SLOW",)


def test_reasons_have_stable_priority_order() -> None:
    metrics = CandidateMetrics(
        recall=0.0,
        precision=0.0,
        complete_scene_recall=-0.1,
        hot_p95_ms=900.0,
        hot_fps=0.1,
        pss_mb=400.0,
    )

    result = evaluate_candidate(metrics, _gate())

    assert result.reasons == (
        "RECALL_BELOW_BASELINE",
        "PRECISION_BELOW_MINIMUM",
        "COMPLETE_SCENE_RECALL_BELOW_BASELINE",
        "HOT_P95_TOO_SLOW",
        "HOT_FPS_TOO_LOW",
        "PSS_TOO_HIGH",
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_manifest_binds_model_truth_and_runtime(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    truth = tmp_path / "truth.json"
    model.write_bytes(b"model")
    truth.write_bytes(b"truth")

    manifest = prepare_benchmark_manifest(
        candidate="ncnn-fp16-cpu",
        model_path=model,
        formal_truth_path=truth,
        expected_truth_sha256=_sha256(truth),
        runtime_name="ncnn",
        runtime_revision=PINNED_NCNN_COMMIT,
    )

    assert manifest["schema_version"] == "mobile-detector-benchmark-v1"
    assert manifest["model_sha256"] == _sha256(model)
    assert manifest["formal_truth_sha256"] == _sha256(truth)
    assert manifest["runtime_revision"] == PINNED_NCNN_COMMIT


def test_manifest_rejects_formal_truth_hash_mismatch(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    truth = tmp_path / "truth.json"
    model.write_bytes(b"model")
    truth.write_bytes(b"truth")

    with pytest.raises(ValueError, match="FORMAL_TRUTH_HASH_MISMATCH"):
        prepare_benchmark_manifest(
            candidate="mnn-fp32-cpu",
            model_path=model,
            formal_truth_path=truth,
            expected_truth_sha256="0" * 64,
            runtime_name="mnn",
            runtime_revision=PINNED_MNN_COMMIT,
        )


@pytest.mark.parametrize(
    ("runtime_name", "runtime_revision", "error"),
    [
        ("unknown", PINNED_NCNN_COMMIT, "UNKNOWN_RUNTIME"),
        ("ncnn", "deadbeef", "RUNTIME_REVISION_MISMATCH"),
    ],
)
def test_manifest_rejects_unknown_or_unpinned_runtime(
    tmp_path,
    runtime_name: str,
    runtime_revision: str,
    error: str,
) -> None:
    model = tmp_path / "model.onnx"
    truth = tmp_path / "truth.json"
    model.write_bytes(b"model")
    truth.write_bytes(b"truth")

    with pytest.raises(ValueError, match=error):
        prepare_benchmark_manifest(
            candidate="candidate",
            model_path=model,
            formal_truth_path=truth,
            expected_truth_sha256=_sha256(truth),
            runtime_name=runtime_name,
            runtime_revision=runtime_revision,
        )


def test_parity_rejects_missing_detection() -> None:
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]

    result = compare_predictions(baseline, [], iou_threshold=0.95)

    assert result.passed is False
    assert result.missing == 1
    assert result.reasons == ("MISSING_DETECTION",)


def test_parity_accepts_small_numeric_drift() -> None:
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]
    candidate = [{"image_id": 1, "bbox": [10.2, 9.9, 20, 20], "score": 0.795}]

    result = compare_predictions(baseline, candidate, iou_threshold=0.95)

    assert result.passed is True
    assert result.max_coordinate_drift == pytest.approx(0.2)
    assert result.max_score_drift == pytest.approx(0.005)


def test_parity_rejects_unknown_image_and_large_numeric_drift() -> None:
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]
    candidate = [
        {"image_id": 1, "bbox": [11.1, 10, 20, 20], "score": 0.82},
        {"image_id": 2, "bbox": [10, 10, 20, 20], "score": 0.8},
    ]

    result = compare_predictions(baseline, candidate, iou_threshold=0.85)

    assert result.passed is False
    assert result.unexpected == 1
    assert result.reasons == (
        "UNKNOWN_IMAGE",
        "COORDINATE_DRIFT_TOO_HIGH",
        "SCORE_DRIFT_TOO_HIGH",
    )


def test_parity_report_is_fail_closed_and_binds_prediction_hashes(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]
    candidate = []
    baseline_path.write_text("baseline", encoding="utf-8")
    candidate_path.write_text("candidate", encoding="utf-8")

    report = build_parity_report(
        baseline,
        candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        iou_threshold=0.95,
    )

    assert report["status"] == "parity_failed"
    assert report["baseline_predictions_sha256"] == _sha256(baseline_path)
    assert report["candidate_predictions_sha256"] == _sha256(candidate_path)
