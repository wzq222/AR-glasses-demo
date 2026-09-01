from crrc_vision.mobile_benchmark import (
    BenchmarkGate,
    CandidateMetrics,
    evaluate_candidate,
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
