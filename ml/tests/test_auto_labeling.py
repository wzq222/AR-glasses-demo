import pytest

from crrc_vision.auto_labeling import (
    Candidate,
    fuse_candidates,
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
