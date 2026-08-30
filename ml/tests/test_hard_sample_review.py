import numpy as np

from crrc_vision.hard_sample_review import (
    build_review_scales,
    build_review_result_document,
    validate_h1_reviews,
)


def test_review_scales_preserve_pixels_at_two_and_four_times() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    scales = build_review_scales(image)

    assert set(scales) == {"detail_2x", "detail_4x"}
    assert scales["detail_2x"].shape == (4, 6, 3)
    assert scales["detail_4x"].shape == (8, 12, 3)
    assert np.all(scales["detail_4x"][4:8, 8:12] == image[1, 2])


def test_subtle_and_damaged_require_blind_second_review() -> None:
    manifest = {
        "records": [
            {
                "sample_id": "h1a-0001",
                "intent": "SUBTLE_DISPLACED",
                "image_sha256": "A" * 64,
            }
        ]
    }
    first = {
        "records": [
            {
                "sample_id": "h1a-0001",
                "decision": "APPROVED",
                "image_sha256": "A" * 64,
            }
        ]
    }
    assert "SECOND_REVIEW_MISSING:h1a-0001" in validate_h1_reviews(
        manifest, first, None
    )


def test_first_review_must_cover_and_hash_match_every_sample() -> None:
    manifest = {
        "records": [
            {"sample_id": "h1a-0001", "intent": "ALIGNED", "image_sha256": "A" * 64}
        ]
    }
    first = {
        "records": [
            {"sample_id": "h1a-0001", "decision": "APPROVED", "image_sha256": "B" * 64}
        ]
    }
    assert "IMAGE_HASH_MISMATCH:h1a-0001:first" in validate_h1_reviews(
        manifest, first, None
    )


def test_second_review_cannot_reveal_first_decision() -> None:
    manifest = {
        "records": [
            {"sample_id": "h1a-0001", "intent": "DAMAGED_MARK", "image_sha256": "A" * 64}
        ]
    }
    first = {
        "records": [
            {"sample_id": "h1a-0001", "decision": "APPROVED", "image_sha256": "A" * 64}
        ]
    }
    second = {
        "records": [
            {
                "sample_id": "h1a-0001",
                "decision": "APPROVED",
                "image_sha256": "A" * 64,
                "first_decision": "APPROVED",
            }
        ]
    }
    assert "SECOND_REVIEW_NOT_BLIND:h1a-0001" in validate_h1_reviews(
        manifest, first, second
    )


def test_result_document_counts_resolved_status_and_preserves_truth_hash() -> None:
    manifest = {
        "formal_truth_sha256": "A" * 64,
        "records": [
            {"sample_id": "h1a-0001", "intent": "ALIGNED", "image_sha256": "B" * 64},
            {
                "sample_id": "h1a-0002",
                "intent": "SUBTLE_DISPLACED",
                "image_sha256": "C" * 64,
            },
        ],
    }
    first = {
        "records": [
            {"sample_id": "h1a-0001", "decision": "APPROVED", "image_sha256": "B" * 64},
            {"sample_id": "h1a-0002", "decision": "APPROVED", "image_sha256": "C" * 64},
        ]
    }
    second = {
        "records": [
            {"sample_id": "h1a-0002", "decision": "REJECTED", "image_sha256": "C" * 64}
        ]
    }

    result = build_review_result_document(manifest, first, second)

    assert result["formal_truth_sha256"] == "A" * 64
    assert result["count"] == 2
    assert result["status_counts"] == {
        "APPROVED": 1,
        "REJECTED": 0,
        "UNCERTAIN": 1,
    }
