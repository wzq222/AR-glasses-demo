import pytest

from crrc_vision.synthetic_state import classify_state, relative_angle_deg, validate_state


def test_state_bands_are_separated() -> None:
    assert classify_state(2.0) == "NORMAL"
    assert classify_state(9.0) == "SLIGHT_LOOSE"
    assert classify_state(24.0) == "OBVIOUS_LOOSE"
    assert classify_state(4.5) == "UNCERTAIN"


def test_declared_state_must_match_endpoints() -> None:
    fixed = ((10.0, 10.0), (30.0, 10.0))
    moving = ((30.0, 10.0), (50.0, 10.0))
    assert relative_angle_deg(fixed, moving) == pytest.approx(0.0)
    assert validate_state("NORMAL", fixed, moving).accepted is True
    assert validate_state("OBVIOUS_LOOSE", fixed, moving).accepted is False


def test_degenerate_segment_is_rejected() -> None:
    result = validate_state(
        "NORMAL",
        ((10.0, 10.0), (10.0, 10.0)),
        ((10.0, 10.0), (20.0, 10.0)),
    )
    assert result.accepted is False
    assert "退化" in result.reason
