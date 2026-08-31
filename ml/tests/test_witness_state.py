import pytest

from crrc_vision.witness_state import (
    AngleInterval,
    ProvisionalTriageThresholds,
    StateEvidence,
    StateThresholds,
    evaluate_witness_state,
    measure_witness_geometry,
    triage_witness_angle,
)


CALIBRATED = StateThresholds(
    maximum_angle_degrees=8.0,
    maximum_gap_ratio=0.08,
    maximum_residual_ratio=0.05,
    learned_displaced_threshold=0.80,
    baseline_displaced_threshold=0.80,
    damaged_mark_threshold=0.75,
    calibrated=True,
)


def _evidence(**changes) -> StateEvidence:
    values = {
        "topology": "nut_plate",
        "mark_role": "bridges_moving_fixed",
        "quality_pass": True,
        "fixed_segment_xyxy": ((0.0, 0.0), (10.0, 0.0)),
        "moving_segment_xyxy": ((12.0, 0.0), (22.0, 0.0)),
        "fixed_segment_confidence": 0.95,
        "moving_segment_confidence": 0.95,
        "reference_size": 100.0,
    }
    values.update(changes)
    return StateEvidence(**values)


def test_aligned_requires_calibrated_geometry_and_valid_bridge() -> None:
    result = evaluate_witness_state(_evidence(), CALIBRATED)

    assert result.state == "ALIGNED"
    assert result.review_hint is None
    assert result.reason == "GEOMETRY_ALIGNED_NO_CONFLICT"


def test_geometry_measurement_is_available_without_state_thresholds() -> None:
    metrics = measure_witness_geometry(
        ((0.0, 0.0), (10.0, 0.0)),
        ((12.0, 0.0), (22.0, 0.0)),
        100.0,
    )

    assert metrics.angle_degrees == pytest.approx(0.0)
    assert metrics.gap_ratio == pytest.approx(0.02)
    assert metrics.residual_ratio == pytest.approx(0.0)


def test_one_abnormal_geometry_source_is_possible_not_proven_displaced() -> None:
    result = evaluate_witness_state(
        _evidence(moving_segment_xyxy=((12.0, 0.0), (12.0, 10.0))),
        CALIBRATED,
    )

    assert result.state == "INSUFFICIENT"
    assert result.review_hint == "POSSIBLE_DISPLACED"
    assert result.reason == "DISPLACEMENT_REQUIRES_CORROBORATION"
    assert result.geometry_abnormal is True


def test_geometry_and_independent_learned_evidence_can_prove_displacement() -> None:
    result = evaluate_witness_state(
        _evidence(
            moving_segment_xyxy=((12.0, 0.0), (12.0, 10.0)),
            learned_displaced_score=0.91,
        ),
        CALIBRATED,
    )

    assert result.state == "DISPLACED"
    assert result.supporting_sources == ("geometry", "learned_state")


def test_damage_is_not_silently_converted_to_displacement() -> None:
    result = evaluate_witness_state(
        _evidence(damaged_mark_score=0.92),
        CALIBRATED,
    )

    assert result.state == "DAMAGED_MARK"
    assert result.reason == "MARK_DAMAGE_WITHOUT_RIGID_DISPLACEMENT"


def test_damage_and_displacement_evidence_conflict_is_insufficient() -> None:
    result = evaluate_witness_state(
        _evidence(damaged_mark_score=0.92, learned_displaced_score=0.91),
        CALIBRATED,
    )

    assert result.state == "INSUFFICIENT"
    assert result.reason == "DAMAGE_DISPLACEMENT_CONFLICT"


def test_damage_conflicts_with_even_corroborated_displacement() -> None:
    result = evaluate_witness_state(
        _evidence(
            moving_segment_xyxy=((12.0, 0.0), (12.0, 10.0)),
            damaged_mark_score=0.92,
            learned_displaced_score=0.91,
        ),
        CALIBRATED,
    )

    assert result.state == "INSUFFICIENT"
    assert result.reason == "DAMAGE_DISPLACEMENT_CONFLICT"


def test_non_finite_confidence_or_coordinate_fails_closed() -> None:
    bad_confidence = evaluate_witness_state(
        _evidence(fixed_segment_confidence=float("nan")), CALIBRATED
    )
    bad_coordinate = evaluate_witness_state(
        _evidence(fixed_segment_xyxy=((float("nan"), 0.0), (10.0, 0.0))),
        CALIBRATED,
    )

    assert bad_confidence.state == "INSUFFICIENT"
    assert bad_confidence.reason == "SEGMENT_CONFIDENCE_INVALID"
    assert bad_coordinate.state == "INSUFFICIENT"
    assert bad_coordinate.reason == "SEGMENT_COORDINATE_INVALID"


def test_json_like_string_values_do_not_pass_typed_evidence_gate() -> None:
    bad_quality = evaluate_witness_state(_evidence(quality_pass="false"), CALIBRATED)  # type: ignore[arg-type]
    bad_coordinate = evaluate_witness_state(
        _evidence(fixed_segment_xyxy=(("0", 0.0), (10.0, 0.0))),  # type: ignore[arg-type]
        CALIBRATED,
    )

    assert bad_quality.state == "INSUFFICIENT"
    assert bad_quality.reason == "IMAGE_QUALITY_INVALID"
    assert bad_coordinate.state == "INSUFFICIENT"
    assert bad_coordinate.reason == "SEGMENT_COORDINATE_INVALID"


def test_extreme_finite_coordinates_fail_closed_instead_of_overflowing() -> None:
    result = evaluate_witness_state(
        _evidence(
            fixed_segment_xyxy=((-1.0e308, 0.0), (1.0e308, 0.0)),
            moving_segment_xyxy=((0.0, -1.0e308), (0.0, 1.0e308)),
        ),
        CALIBRATED,
    )

    assert result.state == "INSUFFICIENT"
    assert result.reason == "GEOMETRY_VALUE_INVALID"


def test_calibrated_flag_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="calibrated must be boolean"):
        StateThresholds(
            8.0,
            0.08,
            0.05,
            0.80,
            0.80,
            0.75,
            calibrated="false",  # type: ignore[arg-type]
        )


def test_unknown_topology_or_one_sided_mark_is_insufficient() -> None:
    unknown = evaluate_witness_state(_evidence(topology="unknown"), CALIBRATED)
    one_sided = evaluate_witness_state(_evidence(mark_role="moving_only"), CALIBRATED)

    assert unknown.state == "INSUFFICIENT"
    assert unknown.reason == "TOPOLOGY_UNKNOWN"
    assert one_sided.state == "INSUFFICIENT"
    assert one_sided.reason == "MARK_DOES_NOT_BRIDGE_MOVING_FIXED"


@pytest.mark.parametrize(
    "topology",
    ["nut_stud", "double_nut", "fitting_pipe", "clamp_pipe"],
)
def test_topologies_without_specific_solver_fail_closed(topology: str) -> None:
    result = evaluate_witness_state(_evidence(topology=topology), CALIBRATED)

    assert result.state == "INSUFFICIENT"
    assert result.reason == "TOPOLOGY_SOLVER_UNAVAILABLE"


def test_uncalibrated_thresholds_never_emit_aligned_or_displaced() -> None:
    result = evaluate_witness_state(_evidence(), StateThresholds.uncalibrated())

    assert result.state == "INSUFFICIENT"
    assert result.reason == "THRESHOLDS_UNCALIBRATED"


@pytest.mark.parametrize(
    ("interval", "second_view_confirms", "expected_hint", "expected_reason"),
    [
        (AngleInterval(2.5, 2.0, 3.0), False, "LIKELY_ALIGNED", "ANGLE_INTERVAL_BELOW_REVIEW_THRESHOLD"),
        (AngleInterval(3.0, 3.0, 3.0), False, "LIKELY_ALIGNED", "ANGLE_INTERVAL_BELOW_REVIEW_THRESHOLD"),
        (AngleInterval(3.0, 2.5, 3.5), False, "POSSIBLE_DISPLACED", "ANGLE_INTERVAL_CROSSES_REVIEW_THRESHOLD"),
        (AngleInterval(8.0, 3.0, 14.9), False, "POSSIBLE_DISPLACED", "ANGLE_INTERVAL_ABOVE_REVIEW_THRESHOLD"),
        (AngleInterval(14.5, 14.0, 15.0), False, "POSSIBLE_DISPLACED", "ANGLE_INTERVAL_CROSSES_HIGH_SUSPICION_THRESHOLD"),
        (AngleInterval(18.0, 15.0, 21.0), False, "POSSIBLE_DISPLACED", "SECOND_VIEW_CONFIRMATION_REQUIRED"),
        (AngleInterval(18.0, 15.0, 21.0), True, "LIKELY_DISPLACED", "ANGLE_INTERVAL_HIGH_SUSPICION_CONFIRMED"),
    ],
)
def test_provisional_triage_routes_exact_boundaries_without_formal_state(
    interval: AngleInterval,
    second_view_confirms: bool,
    expected_hint: str,
    expected_reason: str,
) -> None:
    result = triage_witness_angle(
        interval,
        ProvisionalTriageThresholds(),
        second_view_confirms=second_view_confirms,
    )

    assert result.state == "INSUFFICIENT"
    assert result.review_hint == expected_hint
    assert result.reason == expected_reason
    assert result.requires_human_confirmation is True


def test_angle_interval_rejects_invalid_or_non_finite_bounds() -> None:
    with pytest.raises(ValueError, match="angle interval"):
        AngleInterval(4.0, 5.0, 3.0)
    with pytest.raises(ValueError, match="angle interval"):
        AngleInterval(float("nan"), 0.0, 3.0)
    with pytest.raises(ValueError, match="angle interval"):
        AngleInterval(4.0, 3.0, 91.0)


def test_uncalibrated_evaluation_still_returns_measurement_and_review_hint() -> None:
    result = evaluate_witness_state(
        _evidence(
            moving_segment_xyxy=((12.0, 0.0), (22.0, 0.0)),
            angle_interval=AngleInterval(2.0, 1.0, 3.0),
        ),
        StateThresholds.uncalibrated(),
    )

    assert result.state == "INSUFFICIENT"
    assert result.reason == "THRESHOLDS_UNCALIBRATED"
    assert result.review_hint == "LIKELY_ALIGNED"
    assert result.angle_degrees == pytest.approx(0.0)
    assert result.angle_lower_degrees == pytest.approx(1.0)
    assert result.angle_upper_degrees == pytest.approx(3.0)


def test_uncalibrated_high_suspicion_requires_second_view_for_likely_hint() -> None:
    unconfirmed = evaluate_witness_state(
        _evidence(
            moving_segment_xyxy=((12.0, 0.0), (12.0, 10.0)),
            angle_interval=AngleInterval(20.0, 16.0, 24.0),
        ),
        StateThresholds.uncalibrated(),
    )
    confirmed = evaluate_witness_state(
        _evidence(
            moving_segment_xyxy=((12.0, 0.0), (12.0, 10.0)),
            angle_interval=AngleInterval(20.0, 16.0, 24.0),
            second_view_confirms=True,
        ),
        StateThresholds.uncalibrated(),
    )

    assert unconfirmed.state == confirmed.state == "INSUFFICIENT"
    assert unconfirmed.review_hint == "POSSIBLE_DISPLACED"
    assert confirmed.review_hint == "LIKELY_DISPLACED"
