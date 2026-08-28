from __future__ import annotations

import copy

import pytest

from crrc_vision.marked_point import (
    assemble_partition,
    assemble_marked_point_truth,
    build_review_set,
    build_manual_positive_records,
    filter_then_deduplicate_positive_records,
    deduplicate_positive_records,
    evaluate_candidate_recall,
    unreviewed_positive_ids,
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


def test_added_positive_requires_blind_geometry_second_pass() -> None:
    review = _review()
    review["images"][0]["added_marked_points"] = [
        {"xyxy": [60, 20, 80, 40]}
    ]
    with pytest.raises(ValueError, match="SECOND_PASS_REQUIRED"):
        assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})

    review["images"][0]["added_marked_points"][0]["second_pass"] = {
        "first_result_hidden": True,
        "decision": "accept",
        "final_xyxy": [61, 21, 81, 41],
    }
    truth = assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})
    assert truth["annotations"][-1]["bbox"] == [61.0, 21.0, 20.0, 20.0]


def test_adjusted_candidate_requires_blind_geometry_second_pass() -> None:
    review = _review()
    marked = review["images"][0]["candidate_decisions"][0]
    marked["geometry_adjusted"] = True
    with pytest.raises(ValueError, match="SECOND_PASS_REQUIRED"):
        assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})


def test_partition_assembly_rejects_wrong_split_hash_and_old_sealed() -> None:
    selection = {
        "train": [],
        "val": [
            {
                "image_id": 1,
                "relative_path": "a.jpg",
                "scene_group": "scene-a",
                "sha256": "A" * 64,
            }
        ],
        "forbidden_old_sealed": {"sha256": ["F" * 64], "paths": []},
        "old_sealed_test_opened": False,
    }
    truth = assemble_partition(
        _review(),
        selection=selection,
        partition="val",
        image_sizes={"a.jpg": (100, 80)},
    )
    assert len(truth["images"]) == 1

    wrong_split = copy.deepcopy(selection)
    wrong_split["train"] = wrong_split["val"]
    wrong_split["val"] = []
    with pytest.raises(ValueError, match="REVIEW_IMAGE_OUTSIDE_PARTITION"):
        assemble_partition(
            _review(),
            selection=wrong_split,
            partition="val",
            image_sizes={"a.jpg": (100, 80)},
        )

    review = _review()
    review["images"][0]["source_sha256"] = "B" * 64
    with pytest.raises(ValueError, match="SELECTION_IDENTITY_MISMATCH"):
        assemble_partition(
            review,
            selection=selection,
            partition="val",
            image_sizes={"a.jpg": (100, 80)},
        )

    review = _review()
    selection["forbidden_old_sealed"]["sha256"] = ["A" * 64]
    with pytest.raises(ValueError, match="OLD_SEALED_IMAGE_FORBIDDEN"):
        assemble_partition(
            review,
            selection=selection,
            partition="val",
            image_sizes={"a.jpg": (100, 80)},
        )


def test_covered_candidate_is_audited_without_becoming_a_negative_or_duplicate() -> None:
    review = _review()
    review["images"][0]["candidate_decisions"][0]["label"] = (
        "covered_by_added_marked_point"
    )
    review["images"][0]["added_marked_points"] = [
        {
            "positive_id": "p1",
            "xyxy": [1, 2, 11, 12],
            "second_pass": {
                "first_result_hidden": True,
                "decision": "accept",
                "final_xyxy": [1, 2, 11, 12],
            },
        }
    ]

    truth = assemble_marked_point_truth(
        review, image_sizes={"a.jpg": (100, 80)}
    )

    assert len(truth["annotations"]) == 1
    assert truth["info"]["covered_candidate_count"] == 1
    assert truth["info"]["negative_counts"] == {
        "unmarked_fastener": 1,
        "lookalike": 1,
        "uncertain": 0,
    }


def test_positive_deduplication_preserves_adjacent_points_and_source_priority() -> None:
    records = [
        {
            "positive_id": "miss-1",
            "source_rank": 1,
            "xyxy": [2, 2, 12, 12],
            "dedupe_xyxy": [2, 2, 12, 12],
        },
        {
            "positive_id": "truth-1",
            "source_rank": 0,
            "xyxy": [0, 0, 18, 10],
            "dedupe_xyxy": [0, 0, 10, 10],
        },
        {
            "positive_id": "truth-2",
            "source_rank": 0,
            "xyxy": [4, 0, 21, 10],
            "dedupe_xyxy": [11, 0, 21, 10],
        },
    ]

    kept, suppressed = deduplicate_positive_records(records)

    assert [row["positive_id"] for row in kept] == ["truth-1", "truth-2"]
    assert suppressed == [{"positive_id": "miss-1", "kept_positive_id": "truth-1"}]


def test_review_builder_covers_every_candidate_and_excludes_uncertain_scene() -> None:
    selection = {
        "train": [
            {
                "image_id": 1,
                "relative_path": "a.jpg",
                "scene_group": "scene-a",
                "sha256": "A" * 64,
            }
        ],
        "val": [
            {
                "image_id": 2,
                "relative_path": "b.jpg",
                "scene_group": "scene-b",
                "sha256": "B" * 64,
            }
        ],
    }
    candidates = [
        {
            "id": "c1",
            "relative_path": "a.jpg",
            "sources": ["fastener_v2_2"],
            "xyxy": [0, 0, 10, 10],
        },
        {
            "id": "c2",
            "relative_path": "a.jpg",
            "sources": ["a_color"],
            "xyxy": [50, 50, 60, 60],
        },
    ]
    positives = [
        {
            "positive_id": "p1",
            "relative_path": "a.jpg",
            "xyxy": [0, 0, 12, 12],
            "second_pass": {
                "first_result_hidden": True,
                "decision": "accept",
                "final_xyxy": [0, 0, 12, 12],
            },
        }
    ]

    review_set = build_review_set(
        selection=selection,
        candidates=candidates,
        positives=positives,
        uncertain_paths={"b.jpg": "motion_blur"},
    )

    train_image = review_set["reviews"]["train"]["images"][0]
    assert train_image["expected_candidate_ids"] == ["c1", "c2"]
    assert [row["label"] for row in train_image["candidate_decisions"]] == [
        "covered_by_added_marked_point",
        "lookalike",
    ]
    assert review_set["reviews"]["val"]["images"] == []
    assert review_set["uncertain_exclusions"][0]["relative_path"] == "b.jpg"


def test_candidate_recall_accepts_mark_center_hit_and_reports_misses() -> None:
    truth = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 20, 20]},
            {"id": 2, "image_id": 1, "bbox": [50, 50, 20, 20]},
        ],
    }
    candidates = [
        {
            "id": "mark-1",
            "relative_path": "a.jpg",
            "xyxy": [8, 8, 12, 12],
            "sources": ["a_color"],
        }
    ]

    report = evaluate_candidate_recall(truth, candidates, minimum_recall=0.99)

    assert report["true_positives"] == 1
    assert report["false_negatives"] == 1
    assert report["recall"] == 0.5
    assert report["passed"] is False
    assert report["misses"][0]["annotation_id"] == 2


def test_build_manual_positive_records_binds_selection_identity() -> None:
    selection = {
        "train": [
            {
                "image_id": 7,
                "relative_path": "scene.jpg",
                "sha256": "A" * 64,
            }
        ],
        "val": [],
    }

    records = build_manual_positive_records(
        selection,
        [
            {
                "manual_id": "audit-001",
                "relative_path": "scene.jpg",
                "xyxy": [10, 20, 30, 40],
                "mark_colors": ["red"],
                "audit_reason": "full_image_miss",
            }
        ],
    )

    assert records == [
        {
            "positive_id": "manual-audit-001",
            "source_rank": 3,
            "first_pass_source": "manual",
            "first_pass_shortlist_id": "audit-001",
            "relative_path": "scene.jpg",
            "image_id": 7,
            "source_sha256": "A" * 64,
            "xyxy": [10.0, 20.0, 30.0, 40.0],
            "dedupe_xyxy": [10.0, 20.0, 30.0, 40.0],
            "mark_colors": ["red"],
            "audit_reason": "full_image_miss",
        }
    ]


def test_build_manual_positive_records_rejects_unknown_image() -> None:
    selection = {"train": [], "val": []}

    with pytest.raises(ValueError, match="MANUAL_IMAGE_NOT_SELECTED"):
        build_manual_positive_records(
            selection,
            [
                {
                    "manual_id": "audit-001",
                    "relative_path": "missing.jpg",
                    "xyxy": [10, 20, 30, 40],
                }
            ],
        )


def test_rejected_primary_does_not_suppress_audited_fallback() -> None:
    records = [
        {
            "positive_id": "old-primary",
            "source_rank": 0,
            "relative_path": "scene.jpg",
            "xyxy": [10, 10, 30, 30],
        },
        {
            "positive_id": "audit-fallback",
            "source_rank": 3,
            "relative_path": "scene.jpg",
            "xyxy": [11, 11, 31, 31],
        },
    ]

    kept, suppressed = filter_then_deduplicate_positive_records(
        records, {"old-primary"}
    )

    assert [row["positive_id"] for row in kept] == ["audit-fallback"]
    assert suppressed == []


def test_removed_rejections_do_not_make_second_pass_incomplete() -> None:
    assert unreviewed_positive_ids(
        {"kept-a", "kept-b"},
        {"kept-a", "kept-b"},
        {"removed-rejection"},
    ) == set()
