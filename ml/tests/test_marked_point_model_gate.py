from __future__ import annotations

import pytest

import crrc_vision.marked_point_model_gate as gate_module
from crrc_vision.marked_point_model_gate import (
    build_proposal_gate_document,
    select_proposal_threshold,
)


def _truth() -> dict[str, object]:
    return {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "bbox": [0, 0, 10, 10]},
        ],
    }


def test_gate_selects_highest_threshold_that_keeps_required_recall() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.4},
        {"image_id": 2, "bbox": [20, 20, 10, 10], "score": 0.8},
    ]

    report = select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)

    assert report.threshold == 0.4
    assert report.coverage_recall == 1.0
    assert report.candidates_per_image == 1.5
    assert report.complete_scenes == 2
    assert (report.covered_truth, report.irrelevant_candidates, report.uncovered_truth) == (2, 1, 0)


def test_gate_rejects_unreachable_recall() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
    ]
    with pytest.raises(ValueError, match="NO_THRESHOLD_MEETS_RECALL"):
        select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)


def test_gate_counts_duplicate_predictions_as_candidate_burden() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    report = select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)
    assert report.threshold == 0.7
    assert report.candidates_per_image == 1.5
    assert report.candidate_relevance == 1.0


def test_gate_rejects_nested_box_below_localization_iou() -> None:
    truth = {
        "images": [{"id": 1}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 180, 180]}],
    }
    predictions = [
        {"image_id": 1, "bbox": [45, 45, 90, 90], "score": 0.8},
    ]

    with pytest.raises(ValueError, match="NO_THRESHOLD_MEETS_RECALL"):
        select_proposal_threshold(predictions, truth, minimum_recall=1.0)


def test_one_proposal_cannot_cover_two_adjacent_truth_boxes() -> None:
    truth = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [100, 100, 80, 80]},
            {"id": 2, "image_id": 1, "bbox": [180, 100, 80, 80]},
        ],
    }
    predictions = [
        {"image_id": 1, "bbox": [90, 90, 180, 100], "score": 0.9},
    ]

    with pytest.raises(ValueError, match="NO_THRESHOLD_MEETS_RECALL"):
        select_proposal_threshold(predictions, truth, minimum_recall=1.0)


def test_gate_uses_logarithmic_threshold_search(monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = [
        {"image_id": 1, "bbox": [20 + index, 20, 1, 1], "score": 0.99 - index * 0.02}
        for index in range(16)
    ]
    predictions.extend(
        [
            {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.4},
        ]
    )
    real_evaluate = gate_module.evaluate_at_threshold
    calls = 0

    def counted_evaluate(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(gate_module, "evaluate_at_threshold", counted_evaluate)

    report = select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)

    assert report.threshold == 0.4
    assert calls <= 6


def test_gate_document_binds_model_truth_and_predictions() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.8},
    ]
    document = build_proposal_gate_document(
        predictions,
        _truth(),
        model_sha256="A" * 64,
        truth_sha256="B" * 64,
        prediction_sha256="C" * 64,
        minimum_recall=1.0,
    )
    assert document["schema_version"] == "marked-point-model-gate-v3"
    assert document["proposal_match"] == "one_to_one_iou_gte_0.30"
    assert document["model_sha256"] == "A" * 64
    assert document["report"]["coverage_recall"] == 1.0
    assert document["passed_recall"] is True
    assert document["passed_burden"] is True
    assert document["sealed_test_opened"] is False
