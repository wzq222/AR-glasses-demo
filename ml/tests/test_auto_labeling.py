from crrc_vision.auto_labeling import Candidate, fuse_candidates


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
