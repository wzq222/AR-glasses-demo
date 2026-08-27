import pytest

from crrc_vision.auto_labeling import (
    Candidate,
    DEFAULT_ANCHOR_ASSIGNMENT_MARGIN,
    DEFAULT_CLUSTER_RECONCILIATION_IOU,
    DEFAULT_HSV_ANCHOR_EXPANSION,
    fuse_candidates,
    fusion_stats,
    normalize_hsv_document,
    normalize_teacher_payload,
    verify_truth_unchanged,
)


def candidate(
    source_id: str,
    family: str,
    category: str,
    box: tuple[float, float, float, float],
    score: float,
) -> Candidate:
    return Candidate("a.jpg", source_id, family, category, box, score)


def test_multiscale_teacher_hits_are_one_source_family() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "teacher-640",
                "reference_teacher",
                "fastener",
                (10, 10, 30, 30),
                0.8,
            ),
            candidate(
                "teacher-1280",
                "reference_teacher",
                "fastener",
                (11, 11, 31, 31),
                0.9,
            ),
        ]
    )

    assert fused[0].supporting_families == ("reference_teacher",)
    assert fused[0].consensus_status == "single_source"


def test_two_independent_families_make_high_consensus() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "teacher-640",
                "reference_teacher",
                "fastener",
                (10, 10, 30, 30),
                0.8,
            ),
            candidate("hsv-1", "hsv", "fastener", (12, 12, 32, 32), 0.7),
        ]
    )

    assert fused[0].supporting_families == ("hsv", "reference_teacher")
    assert fused[0].consensus_status == "consensus_high"


def test_nested_same_center_candidates_merge_below_iou_threshold() -> None:
    fused = fuse_candidates(
        [
            candidate("small", "hsv", "fastener", (0, 0, 20, 20), 0.8),
            candidate(
                "large",
                "reference_teacher",
                "fastener",
                (-5, -5, 25, 25),
                0.9,
            ),
        ]
    )

    assert len(fused) == 1
    assert len(fused[0].member_ids) == 2


def test_adjacent_objects_are_not_merged_by_containment_rule() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "left",
                "reference_teacher",
                "fastener",
                (0, 0, 20, 20),
                0.8,
            ),
            candidate(
                "spanning",
                "reference_teacher",
                "fastener",
                (0, 0, 40, 20),
                0.9,
            ),
        ]
    )

    assert len(fused) == 2


def test_teacher_variance_matches_cluster_representative() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "a", "reference_teacher", "fastener", (0, 0, 20, 20), 0.8
            ),
            candidate(
                "b", "reference_teacher", "fastener", (4, 0, 24, 20), 0.8
            ),
            candidate(
                "c", "reference_teacher", "fastener", (7, 0, 27, 20), 0.8
            ),
        ]
    )

    assert len(fused) == 1


def test_strict_representative_reconciliation_merges_residual_teacher_cluster() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "flat-seed",
                "reference_teacher",
                "fastener",
                (0, 0, 50, 11),
                0.8,
            ),
            candidate(
                "early-residual",
                "reference_teacher",
                "fastener",
                (3, 0, 45, 26),
                0.8,
            ),
            candidate(
                "late-a",
                "reference_teacher",
                "fastener",
                (4, -2, 48, 27),
                0.8,
            ),
            candidate(
                "late-b",
                "reference_teacher",
                "fastener",
                (5, -2, 49, 28),
                0.8,
            ),
            candidate(
                "late-c",
                "reference_teacher",
                "fastener",
                (6, -2, 50, 29),
                0.8,
            ),
        ]
    )

    assert len(fused) == 1


def test_representative_link_prevents_teacher_chain_bridge() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "a", "reference_teacher", "fastener", (0, 0, 20, 20), 0.8
            ),
            candidate(
                "b", "reference_teacher", "fastener", (5, 0, 25, 20), 0.8
            ),
            candidate(
                "c", "reference_teacher", "fastener", (10, 0, 30, 20), 0.8
            ),
        ]
    )

    assert sorted(len(item.member_ids) for item in fused) == [1, 2]


def test_hsv_marker_center_uniquely_attaches_to_teacher_anchor() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "teacher",
                "reference_teacher",
                "fastener",
                (0, 0, 20, 20),
                0.9,
            ),
            candidate("marker", "hsv", "fastener", (-80, -80, 100, 100), 0.8),
        ]
    )

    assert len(fused) == 1
    assert fused[0].supporting_families == ("hsv", "reference_teacher")


def test_hsv_marker_between_two_teacher_anchors_remains_independent() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "left",
                "reference_teacher",
                "fastener",
                (0, 0, 20, 20),
                0.9,
            ),
            candidate(
                "right",
                "reference_teacher",
                "fastener",
                (22, 0, 42, 20),
                0.9,
            ),
            candidate("marker", "hsv", "fastener", (-69, -80, 111, 100), 0.8),
        ]
    )

    assert len(fused) == 3


def test_hsv_marker_cross_category_attachment_is_conflict() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "teacher",
                "reference_teacher",
                "pipe_joint",
                (0, 0, 20, 20),
                0.9,
            ),
            candidate("marker", "hsv", "fastener", (-80, -80, 100, 100), 0.8),
        ]
    )

    assert len(fused) == 1
    assert fused[0].category is None
    assert fused[0].consensus_status == "conflict"


def test_fusion_is_deterministic_across_input_order() -> None:
    rows = [
        candidate("a", "hsv", "fastener", (0, 0, 20, 20), 0.7),
        candidate("b", "student", "fastener", (5, 0, 25, 20), 0.8),
        candidate(
            "c",
            "reference_teacher",
            "fastener",
            (10, 0, 30, 20),
            0.9,
        ),
    ]

    forward = fuse_candidates(rows)
    backward = fuse_candidates(list(reversed(rows)))

    assert [item.stable_id() for item in forward] == [
        item.stable_id() for item in backward
    ]


def test_fusion_stats_report_reduction_and_cluster_sizes() -> None:
    rows = [
        candidate("a", "hsv", "fastener", (0, 0, 20, 20), 0.8),
        candidate(
            "b",
            "reference_teacher",
            "fastener",
            (1, 1, 21, 21),
            0.9,
        ),
        candidate("c", "student", "fastener", (100, 100, 120, 120), 0.7),
    ]
    fused = fuse_candidates(rows)

    assert fusion_stats(rows, fused) == {
        "raw_candidates": 3,
        "fused_candidates": 2,
        "candidate_reduction": 1,
        "cluster_size_histogram": {"1": 1, "2": 1},
    }


def test_anchor_assignment_thresholds_are_explicit_and_conservative() -> None:
    assert DEFAULT_HSV_ANCHOR_EXPANSION == 0.05
    assert DEFAULT_ANCHOR_ASSIGNMENT_MARGIN == 0.10
    assert DEFAULT_CLUSTER_RECONCILIATION_IOU == 0.75


def test_overlapping_categories_are_conflict_not_silently_merged() -> None:
    fused = fuse_candidates(
        [
            candidate(
                "teacher-a",
                "reference_teacher",
                "fastener",
                (10, 10, 30, 30),
                0.9,
            ),
            candidate(
                "student-a",
                "student",
                "pipe_joint",
                (11, 11, 31, 31),
                0.9,
            ),
        ]
    )

    assert len(fused) == 1
    assert fused[0].category is None
    assert fused[0].consensus_status == "conflict"


def test_candidate_manifest_rejects_truth_hash_change() -> None:
    before = "A" * 64

    with pytest.raises(RuntimeError, match="formal truth changed"):
        verify_truth_unchanged(before, "B" * 64)


def test_teacher_and_hsv_sources_normalize_to_full_image_xyxy() -> None:
    teacher = normalize_teacher_payload(
        {
            "imgsz": 960,
            "predictions": [
                {
                    "id": "p1",
                    "relative_path": "a.jpg",
                    "mapped_category": "fastener",
                    "bbox": [10, 20, 30, 40],
                    "score": 0.9,
                    "pass_id": "full-960",
                }
            ],
        }
    )
    hsv = normalize_hsv_document(
        {
            "images": [{"id": 7, "file_name": "a.jpg"}],
            "annotations": [
                {
                    "id": 8,
                    "image_id": 7,
                    "bbox": [12, 22, 30, 40],
                    "attributes": {
                        "algorithm_version": "hsv-line-v2",
                        "candidate_confidence": 0.8,
                    },
                }
            ],
        }
    )

    assert teacher[0].xyxy == (10.0, 20.0, 40.0, 60.0)
    assert teacher[0].source_family == "reference_teacher"
    assert hsv[0].xyxy == (12.0, 22.0, 42.0, 62.0)
    assert hsv[0].source_family == "hsv"
