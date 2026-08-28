from __future__ import annotations

import pytest

from crrc_vision.high_accuracy_gate import (
    evaluate_at_threshold,
    summarize_seed_reports,
    select_threshold,
)


def _truth(images: int = 1, boxes_per_image: int = 1) -> dict[str, object]:
    image_rows = [
        {"id": image_id, "scene_group": f"scene-{image_id}"}
        for image_id in range(1, images + 1)
    ]
    annotations = []
    annotation_id = 1
    for image in image_rows:
        for offset in range(boxes_per_image):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image["id"],
                    "category_id": 1,
                    "bbox": [offset * 30.0, 0.0, 20.0, 20.0],
                    "area": 400.0,
                }
            )
            annotation_id += 1
    return {"images": image_rows, "annotations": annotations}


def test_gate_selects_threshold_on_val_and_reuses_it_on_test() -> None:
    truth = _truth(images=2)
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.8},
        {"image_id": 2, "bbox": [0, 0, 20, 20], "score": 0.7},
        {"image_id": 2, "bbox": [100, 0, 20, 20], "score": 0.6},
    ]

    threshold = select_threshold(
        predictions, truth, minimum_precision=0.90
    )
    report = evaluate_at_threshold(predictions, truth, threshold=threshold)

    assert threshold == 0.7
    assert report.threshold == threshold
    assert report.precision == 1.0
    assert report.recall == 1.0


def test_duplicate_prediction_is_a_false_positive_and_matching_is_one_to_one() -> None:
    truth = _truth()
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.8},
    ]

    report = evaluate_at_threshold(predictions, truth, threshold=0.5)

    assert (report.true_positives, report.false_positives, report.false_negatives) == (
        1,
        1,
        0,
    )
    assert report.precision == 0.5


def test_complete_scene_rate_requires_every_target_match() -> None:
    truth = _truth(images=2, boxes_per_image=2)
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 2, "bbox": [30, 0, 20, 20], "score": 0.8},
    ]

    report = evaluate_at_threshold(predictions, truth, threshold=0.2)

    assert report.complete_scenes == 1
    assert report.complete_scene_rate == 0.5


def test_iou_boundary_counts_and_below_boundary_does_not() -> None:
    truth = _truth()
    # Equal-size boxes shifted by 20/3 have intersection 2/3 width and IoU 0.5.
    boundary = evaluate_at_threshold(
        [{"image_id": 1, "bbox": [20 / 3, 0, 20, 20], "score": 1.0}],
        truth,
        threshold=0.1,
    )
    below = evaluate_at_threshold(
        [{"image_id": 1, "bbox": [6.68, 0, 20, 20], "score": 1.0}],
        truth,
        threshold=0.1,
    )

    assert boundary.true_positives == 1
    assert below.true_positives == 0


def test_threshold_tie_prefers_higher_precision_then_higher_threshold() -> None:
    truth = _truth(images=2)
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 20, 20], "score": 0.8},
        {"image_id": 2, "bbox": [90, 0, 20, 20], "score": 0.8},
    ]

    assert select_threshold(predictions, truth, minimum_precision=0.8) == 0.9
    with pytest.raises(ValueError, match="NO_THRESHOLD_MEETS_PRECISION"):
        select_threshold(predictions, truth, minimum_precision=1.1)


def test_hard_gate_and_minimum_sealed_size() -> None:
    truth = _truth(images=30, boxes_per_image=7)
    predictions = [
        {
            "image_id": annotation["image_id"],
            "bbox": annotation["bbox"],
            "score": 0.9,
        }
        for annotation in truth["annotations"]
    ]
    report = evaluate_at_threshold(
        predictions,
        truth,
        threshold=0.5,
        enforce_sealed_minimum=True,
    )
    assert report.passed is True

    with pytest.raises(ValueError, match="SEALED_TEST_SCENE_COUNT_TOO_LOW"):
        evaluate_at_threshold(
            predictions,
            _truth(images=29, boxes_per_image=7),
            threshold=0.5,
            enforce_sealed_minimum=True,
        )


def test_three_seed_summary_reports_mean_std_range_and_worst() -> None:
    reports = [
        evaluate_at_threshold(
            [{"image_id": 1, "bbox": [0, 0, 20, 20], "score": 1.0}],
            _truth(),
            threshold=0.5,
        ),
        evaluate_at_threshold([], _truth(), threshold=0.5),
        evaluate_at_threshold(
            [{"image_id": 1, "bbox": [0, 0, 20, 20], "score": 1.0}],
            _truth(),
            threshold=0.5,
        ),
    ]
    summary = summarize_seed_reports([1, 2, 3], reports)

    assert summary.recall_mean == pytest.approx(2 / 3)
    assert summary.recall_range == 1.0
    assert summary.worst_seed == 2
