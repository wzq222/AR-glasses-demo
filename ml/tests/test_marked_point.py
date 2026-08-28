from __future__ import annotations

import copy

import pytest

from crrc_vision.marked_point import (
    assemble_marked_point_truth,
    validate_review,
)


def _review() -> dict[str, object]:
    return {
        "schema_version": "marked-point-review-v1",
        "partition": "val",
        "images": [
            {
                "image_id": 1,
                "relative_path": "a.jpg",
                "scene_group": "scene-a",
                "source_sha256": "A" * 64,
                "image_status": "complete",
                "expected_candidate_ids": ["m", "u", "l"],
                "candidate_decisions": [
                    {
                        "candidate_id": "m",
                        "label": "marked_point",
                        "xyxy": [1, 2, 11, 12],
                    },
                    {
                        "candidate_id": "u",
                        "label": "unmarked_fastener",
                        "xyxy": [20, 20, 30, 30],
                    },
                    {
                        "candidate_id": "l",
                        "label": "lookalike",
                        "xyxy": [40, 40, 50, 50],
                    },
                ],
                "added_marked_points": [],
            }
        ],
    }


def test_only_marked_points_enter_positive_truth() -> None:
    truth = assemble_marked_point_truth(
        _review(), image_sizes={"a.jpg": (100, 80)}
    )

    assert truth["categories"] == [{"id": 1, "name": "marked_point"}]
    assert truth["annotations"] == [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1.0, 2.0, 10.0, 10.0],
            "area": 100.0,
            "iscrowd": 0,
            "origin": "candidate",
            "candidate_id": "m",
        }
    ]
    assert truth["info"]["negative_counts"] == {
        "unmarked_fastener": 1,
        "lookalike": 1,
        "uncertain": 0,
    }


def test_complete_image_rejects_uncertain_or_missing_candidates() -> None:
    review = _review()
    image = review["images"][0]
    image["candidate_decisions"] = [
        {
            "candidate_id": "m",
            "label": "uncertain",
            "xyxy": [1, 2, 11, 12],
        }
    ]

    errors = validate_review(review)

    assert "CANDIDATE_COVERAGE_MISMATCH:1" in errors
    assert "UNCERTAIN_COMPLETE_CONFLICT:1" in errors


def test_invalid_or_duplicate_candidate_decisions_are_rejected() -> None:
    review = _review()
    image = review["images"][0]
    duplicate = copy.deepcopy(image["candidate_decisions"][0])
    duplicate["label"] = "not-a-label"
    image["candidate_decisions"].append(duplicate)

    errors = validate_review(review)

    assert "CANDIDATE_COVERAGE_MISMATCH:1" in errors
    assert "INVALID_MARKED_POINT_LABEL:1" in errors


def test_assembly_rejects_incomplete_image_and_out_of_bounds_box() -> None:
    review = _review()
    review["images"][0]["image_status"] = "uncertain"
    with pytest.raises(ValueError, match="MARKED_POINT_REVIEW_INVALID"):
        assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})

    review = _review()
    review["images"][0]["candidate_decisions"][0]["xyxy"] = [90, 2, 110, 12]
    with pytest.raises(ValueError, match="BOX_OUT_OF_BOUNDS"):
        assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})


def test_duplicate_scene_or_image_identity_is_rejected() -> None:
    review = _review()
    duplicate = copy.deepcopy(review["images"][0])
    duplicate["image_id"] = 2
    duplicate["relative_path"] = "b.jpg"
    review["images"].append(duplicate)

    with pytest.raises(ValueError, match="DUPLICATE_REVIEWED_SCENE"):
        assemble_marked_point_truth(
            review,
            image_sizes={"a.jpg": (100, 80), "b.jpg": (100, 80)},
        )
