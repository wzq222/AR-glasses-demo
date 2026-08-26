import json
from typing import Any

import pytest
from PIL import Image

from crrc_vision.codex_review import merge_reviews, validate_review
from crrc_vision.codex_review_pack import build_pack, build_second_pass_tasks


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


def test_review_pack_covers_every_image_and_candidate(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 100), "gray").save(source / "a.jpg")
    Image.new("RGB", (100, 100), "gray").save(source / "b.jpg")
    candidates = {
        "images": [
            {"id": 1, "relative_path": "a.jpg", "scene_group": "g1"},
            {"id": 2, "relative_path": "b.jpg", "scene_group": "g2"},
        ],
        "fused_candidates": [
            {
                "id": "c1",
                "image_id": 1,
                "xyxy": [10, 10, 30, 30],
                "category": "fastener",
            },
            {
                "id": "c2",
                "image_id": 1,
                "xyxy": [40, 40, 60, 60],
                "category": "pipe_joint",
            },
            {
                "id": "c3",
                "image_id": 2,
                "xyxy": [20, 20, 50, 50],
                "category": "fastener",
            },
        ],
    }
    output = tmp_path / "pack"

    summary = build_pack(candidates, source, output)

    assert summary.images == 2
    assert summary.candidates == 3
    assert len(list((output / "full-images").glob("*.jpg"))) == 2
    assert len(list((output / "candidate-contexts").glob("*.jpg"))) == 3


def test_merge_refuses_missing_candidate_decision() -> None:
    candidates = {"fused_candidates": [{"id": "c1"}, {"id": "c2"}]}
    incomplete = {
        "candidate_decisions": [{"candidate_id": "c1", "decision": "accept"}]
    }

    with pytest.raises(ValueError, match="missing candidate decisions"):
        merge_reviews(candidates, incomplete)


def test_second_pass_task_hides_first_pass_decision(tmp_path) -> None:
    review = sample_review(first="needs_adjustment", second=None)
    review["image_id"] = 1
    review["candidate_decisions"][0]["corrected_xyxy"] = [0.1, 0.2, 0.4, 0.5]

    count = build_second_pass_tasks({"reviews": [review]}, tmp_path / "second")
    task = json.loads(
        next((tmp_path / "second").glob("tasks-*.json")).read_text(encoding="utf-8")
    )

    assert count == 1
    assert task["first_result_hidden"] is True
    assert "needs_adjustment" not in json.dumps(task)
