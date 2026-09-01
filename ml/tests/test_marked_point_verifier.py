from __future__ import annotations

import pytest

from crrc_vision.marked_point_verifier import (
    combine_verifier_predictions,
    select_dual_pipeline_thresholds,
    select_semantic_review_examples,
    suppress_overlapping_candidates,
    select_pipeline_threshold,
    select_verifier_examples,
    select_verifier_threshold,
    verifier_resize_size,
)


def test_verifier_resize_preserves_the_pretrained_crop_ratio() -> None:
    assert verifier_resize_size(224) == 256
    assert verifier_resize_size(128) == 146
    with pytest.raises(ValueError, match="VERIFIER_INPUT_SIZE_MUST_BE_POSITIVE"):
        verifier_resize_size(0)


def _truth(partition: str = "train") -> dict[str, object]:
    return {
        "info": {"partition": partition},
        "categories": [{"id": 1, "name": "marked_point"}],
        "images": [
            {
                "id": 1,
                "file_name": "one.jpg",
                "width": 2000,
                "height": 1500,
                "scene_group": "scene-1",
            },
            {
                "id": 2,
                "file_name": "two.jpg",
                "width": 2000,
                "height": 1500,
                "scene_group": "scene-2",
            },
        ],
        "annotations": [
            {"id": 11, "image_id": 1, "bbox": [100, 100, 180, 180]},
            {"id": 22, "image_id": 2, "bbox": [100, 100, 180, 180]},
        ],
    }


def test_verifier_examples_cover_every_truth_and_cap_negatives() -> None:
    predictions = [
        {"image_id": 1, "bbox": [140, 140, 80, 80], "score": 0.9},
        {"image_id": 1, "bbox": [145, 145, 70, 70], "score": 0.8},
        {"image_id": 1, "bbox": [1000, 900, 80, 80], "score": 0.7},
        {"image_id": 1, "bbox": [1200, 900, 80, 80], "score": 0.6},
        {"image_id": 2, "bbox": [140, 140, 80, 80], "score": 0.85},
        {"image_id": 2, "bbox": [1000, 900, 80, 80], "score": 0.65},
    ]

    examples = select_verifier_examples(
        predictions,
        _truth(),
        score_threshold=0.01,
        max_positive_per_truth=2,
        max_negative_per_scene=1,
    )

    positives = [row for row in examples if row["label"] == "marked_point"]
    negatives = [row for row in examples if row["label"] == "not_marked_point"]
    assert [row["truth_id"] for row in positives] == [11, 11, 22]
    assert [row["truth_ids"] for row in positives] == [[11], [11], [22]]
    assert all(row["truth_ids"] == [] for row in negatives)
    assert [(row["image_id"], row["score"]) for row in negatives] == [
        (1, 0.7),
        (2, 0.65),
    ]


def test_shared_proposal_can_cover_adjacent_truth_without_duplicate_crop() -> None:
    truth = _truth()
    truth["images"] = [truth["images"][0]]
    truth["annotations"] = [
        {"id": 11, "image_id": 1, "bbox": [100, 100, 80, 80]},
        {"id": 12, "image_id": 1, "bbox": [180, 100, 80, 80]},
    ]

    examples = select_verifier_examples(
        [{"image_id": 1, "bbox": [90, 90, 180, 100], "score": 0.9}],
        truth,
        score_threshold=0.01,
        max_positive_per_truth=2,
        max_negative_per_scene=1,
    )

    assert len(examples) == 1
    assert examples[0]["truth_ids"] == [11, 12]


def test_verifier_threshold_keeps_required_positive_recall() -> None:
    scored = [
        {"label": "marked_point", "score": 0.9},
        {"label": "marked_point", "score": 0.4},
        {"label": "not_marked_point", "score": 0.8},
        {"label": "not_marked_point", "score": 0.3},
    ]

    report = select_verifier_threshold(scored, minimum_recall=1.0)

    assert report.threshold == 0.4
    assert report.recall == 1.0
    assert report.precision == 2 / 3
    assert report.selected == 3


def test_pipeline_threshold_preserves_truth_coverage_before_reducing_burden() -> None:
    scored = [
        {"truth_ids": [11], "image_id": 1, "score": 0.95},
        {"truth_ids": [11], "image_id": 1, "score": 0.80},
        {"truth_ids": [22], "image_id": 2, "score": 0.55},
        {"truth_ids": [], "image_id": 1, "score": 0.90},
        {"truth_ids": [], "image_id": 2, "score": 0.50},
    ]

    report = select_pipeline_threshold(
        scored, image_count=2, minimum_truth_recall=1.0
    )

    assert report.threshold == 0.55
    assert report.truth_recall == 1.0
    assert report.selected == 4
    assert report.candidates_per_image == 2.0
    assert report.covered_truth == 2


def test_dual_thresholds_use_complementary_proposal_and_verifier_scores() -> None:
    scored = [
        {"truth_ids": [11], "verifier_score": 0.90, "proposal_score": 0.20},
        {"truth_ids": [22], "verifier_score": 0.10, "proposal_score": 0.95},
        {"truth_ids": [], "verifier_score": 0.80, "proposal_score": 0.10},
        {"truth_ids": [], "verifier_score": 0.20, "proposal_score": 0.70},
    ]

    report = select_dual_pipeline_thresholds(scored, image_count=2)

    assert report.truth_recall == 1.0
    assert report.selected == 2
    assert report.candidates_per_image == 1.0
    assert report.verifier_threshold == 0.9
    assert report.proposal_threshold == 0.95


def test_semantic_examples_keep_reviewed_negative_classes() -> None:
    review = {
        "partition": "train",
        "images": [
            {
                "image_id": 1,
                "scene_group": "scene-1",
                "relative_path": "one.jpg",
                "candidate_decisions": [
                    {
                        "candidate_id": "u",
                        "label": "unmarked_fastener",
                        "xyxy": [500, 500, 600, 600],
                    },
                    {
                        "candidate_id": "l",
                        "label": "lookalike",
                        "xyxy": [700, 700, 800, 800],
                    },
                    {
                        "candidate_id": "covered",
                        "label": "covered_by_added_marked_point",
                        "xyxy": [100, 100, 280, 280],
                    },
                ],
            },
            {
                "image_id": 2,
                "scene_group": "scene-2",
                "relative_path": "two.jpg",
                "candidate_decisions": [],
            },
        ],
    }

    examples = select_semantic_review_examples(
        _truth(), review, max_negative_per_scene_per_class=1
    )

    assert [row["label"] for row in examples] == [
        "marked_point",
        "marked_point",
        "lookalike",
        "unmarked_fastener",
    ]
    assert examples[0]["truth_ids"] == [11]
    assert examples[2]["candidate_bbox"] == [700.0, 700.0, 100.0, 100.0]


def test_semantic_nms_removes_duplicates_without_cross_image_suppression() -> None:
    rows = [
        {
            "image_id": 1,
            "candidate_bbox": [0, 0, 100, 100],
            "score": 0.9,
            "proposal_score": 0.2,
        },
        {
            "image_id": 1,
            "candidate_bbox": [10, 10, 100, 100],
            "score": 0.8,
            "proposal_score": 0.3,
        },
        {
            "image_id": 2,
            "candidate_bbox": [10, 10, 100, 100],
            "score": 0.8,
            "proposal_score": 0.3,
        },
    ]

    selected = suppress_overlapping_candidates(
        rows,
        verifier_threshold=0.5,
        proposal_threshold=0.1,
        iou_threshold=0.3,
    )

    assert [(row["image_id"], row["score"]) for row in selected] == [
        (1, 0.8),
        (2, 0.8),
    ]


def test_verifier_ensemble_combines_only_identity_aligned_predictions() -> None:
    first = [
        {"prediction_index": 7, "image_id": 1, "score": 0.25},
        {"prediction_index": 8, "image_id": 1, "score": 0.81},
    ]
    second = [
        {"prediction_index": 7, "image_id": 1, "score": 1.0},
        {"prediction_index": 8, "image_id": 1, "score": 0.01},
    ]

    mean = combine_verifier_predictions([first, second], method="mean")
    geometric = combine_verifier_predictions([first, second], method="geometric_mean")

    assert [row["score"] for row in mean] == [0.625, 0.41000000000000003]
    assert [row["score"] for row in geometric] == pytest.approx([0.5, 0.09])


def test_verifier_ensemble_rejects_misaligned_candidates() -> None:
    first = [{"prediction_index": 7, "image_id": 1, "score": 0.5}]
    second = [{"prediction_index": 8, "image_id": 1, "score": 0.5}]

    with pytest.raises(ValueError, match="VERIFIER_ENSEMBLE_IDENTITY_MISMATCH"):
        combine_verifier_predictions([first, second], method="mean")
