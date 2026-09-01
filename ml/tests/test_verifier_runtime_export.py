from __future__ import annotations

import pytest

from crrc_vision.verifier_runtime_export import (
    compare_verifier_scores,
    verifier_export_contract,
)


def test_verifier_export_contract_preserves_128_input() -> None:
    checkpoint = {
        "architecture": "mobilenet_v3_small",
        "classes": ["marked_point", "not_marked_point"],
        "input_size": 128,
    }

    assert verifier_export_contract(checkpoint) == (
        128,
        ("marked_point", "not_marked_point"),
    )


def test_verifier_export_contract_rejects_wrong_architecture() -> None:
    with pytest.raises(ValueError, match="VERIFIER_ARCHITECTURE_UNSUPPORTED"):
        verifier_export_contract(
            {
                "architecture": "resnet18",
                "classes": ["marked_point", "not_marked_point"],
                "input_size": 128,
            }
        )


def test_verifier_export_contract_rejects_invalid_classes() -> None:
    with pytest.raises(ValueError, match="VERIFIER_CLASSES_INVALID"):
        verifier_export_contract(
            {
                "architecture": "mobilenet_v3_small",
                "classes": ["not_marked_point"],
                "input_size": 128,
            }
        )


def test_verifier_score_parity_rejects_a_threshold_flip() -> None:
    result = compare_verifier_scores(
        [0.20, 0.40], [0.20, 0.29], threshold=0.30, maximum_drift=0.01
    )

    assert result["decision_mismatches"] == 1
    assert result["passed"] is False


def test_verifier_score_parity_accepts_small_stable_drift() -> None:
    result = compare_verifier_scores(
        [0.20, 0.40], [0.201, 0.399], threshold=0.30, maximum_drift=0.01
    )

    assert result["decision_mismatches"] == 0
    assert result["maximum_score_drift"] == pytest.approx(0.001)
    assert result["passed"] is True
