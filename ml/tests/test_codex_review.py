from typing import Any

from crrc_vision.codex_review import validate_review


def sample_review(
    first: str,
    second: str | None,
    image_status: str = "uncertain",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "reviewer": "codex-visual-auditor",
        "task_version": "safe-auto-review-v1",
        "asset_sha256": "A" * 64,
        "first_pass": {"prompt_version": "first-v1", "decision": first},
        "second_pass": None,
        "candidate_decisions": [{"candidate_id": "c1", "decision": first}],
        "added_boxes": [],
        "image_status": image_status,
        "reasons": ["visual-review"],
    }
    if second is not None:
        value["second_pass"] = {
            "prompt_version": "second-v1",
            "decision": second,
            "first_result_hidden": True,
        }
    return value


def test_adjusted_or_added_box_requires_blind_second_pass() -> None:
    review = sample_review(first="needs_adjustment", second=None)

    assert validate_review(review) == ("SECOND_PASS_REQUIRED",)

    review = sample_review(first="accept", second=None)
    review["added_boxes"] = [
        {"category": "fastener", "xyxy": [0.1, 0.2, 0.3, 0.4]}
    ]
    assert validate_review(review) == ("SECOND_PASS_REQUIRED",)


def test_uncertain_image_cannot_be_complete() -> None:
    review = sample_review(first="uncertain", second=None, image_status="complete")

    assert "UNCERTAIN_COMPLETE_CONFLICT" in validate_review(review)


def test_second_pass_must_use_distinct_prompt_and_hide_first_result() -> None:
    review = sample_review(first="needs_adjustment", second="accept")
    review["second_pass"]["prompt_version"] = review["first_pass"][
        "prompt_version"
    ]

    assert "NON_INDEPENDENT_SECOND_PASS" in validate_review(review)


def test_invalid_added_box_and_duplicate_candidate_are_rejected() -> None:
    review = sample_review(first="accept", second=None)
    review["candidate_decisions"].append(
        {"candidate_id": "c1", "decision": "accept"}
    )
    review["added_boxes"] = [
        {"category": "fastener", "xyxy": [0.4, 0.2, 0.3, 0.4]}
    ]

    assert validate_review(review) == (
        "INVALID_ADDED_BOX",
        "INVALID_CANDIDATE_REFERENCE",
        "SECOND_PASS_REQUIRED",
    )
