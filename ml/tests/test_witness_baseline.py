import pytest

from crrc_vision.witness_baseline import estimate_baseline_control_limit


def test_baseline_uses_floor_when_healthy_variation_is_small() -> None:
    result = estimate_baseline_control_limit([0.1, -0.2, 0.3, -0.2, 20.0])

    assert result.ready is True
    assert result.threshold_degrees == pytest.approx(3.0)
    assert result.median_abs_delta_degrees == pytest.approx(0.2)
    assert result.mad_degrees == pytest.approx(0.1)
    assert result.reason == "BASELINE_READY"


def test_baseline_uses_robust_median_mad_control_limit() -> None:
    result = estimate_baseline_control_limit([5.0, 1.0, 4.0, 2.0, 3.0])

    assert result.threshold_degrees == pytest.approx(3.0 + 3.0 * 1.4826)
    assert result.sample_count == 5


def test_baseline_requires_five_healthy_repeats() -> None:
    result = estimate_baseline_control_limit([0.1, 0.2, 0.3, 0.4])

    assert result.ready is False
    assert result.threshold_degrees is None
    assert result.sample_count == 4
    assert result.reason == "BASELINE_SAMPLE_COUNT_INSUFFICIENT"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True, "1.0"])
def test_baseline_rejects_non_numeric_or_non_finite_values(bad_value: object) -> None:
    with pytest.raises(ValueError, match="baseline angle"):
        estimate_baseline_control_limit([0.0, 0.1, 0.2, 0.3, bad_value])  # type: ignore[list-item]


def test_baseline_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="minimum_samples"):
        estimate_baseline_control_limit([0.0] * 5, minimum_samples=0)
    with pytest.raises(ValueError, match="floor_degrees"):
        estimate_baseline_control_limit([0.0] * 5, floor_degrees=-1.0)
