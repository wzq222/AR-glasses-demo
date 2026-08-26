import json
from typing import Any

import pytest
from PIL import Image

from crrc_vision.codex_review import (
    merge_reviews,
    validate_first_pass_review,
    validate_review,
)
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


def test_first_pass_can_queue_added_box_for_blind_second_pass() -> None:
    review = sample_review(first="accept", second=None)
    review["added_boxes"] = [
        {"category": "fastener", "xyxy": [0.1, 0.2, 0.3, 0.4]}
    ]
    review["image_status"] = "pending_second_pass"

    assert validate_first_pass_review(review) == ()
    assert validate_review(review) == ("SECOND_PASS_REQUIRED",)


def test_first_pass_cannot_claim_complete_with_pending_geometry() -> None:
    review = sample_review(first="needs_adjustment", second=None, image_status="complete")
    review["candidate_decisions"][0]["corrected_xyxy"] = [0.1, 0.2, 0.4, 0.5]

    assert validate_first_pass_review(review) == (
        "PENDING_SECOND_PASS_COMPLETE_CONFLICT",
    )


def test_pending_first_pass_cannot_be_merged_as_final_decision() -> None:
    candidates = {"fused_candidates": [{"id": "c1"}]}
    review = sample_review(first="accept", second=None)
    review["image_id"] = 1
    review["image_status"] = "pending_second_pass"

    with pytest.raises(ValueError, match="SECOND_PASS_REQUIRED"):
        merge_reviews(candidates, review)


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


def test_review_pack_includes_high_resolution_miss_sweep_tiles(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (2000, 1500), "gray").save(source / "a.jpg")
    candidates = {
        "images": [{"id": 1, "relative_path": "a.jpg", "scene_group": "g1"}],
        "fused_candidates": [],
    }
    output = tmp_path / "pack"

    build_pack(candidates, source, output)
    task = json.loads(
        (output / "first-pass" / "tasks-001.json").read_text(encoding="utf-8")
    )

    assert len(task["images"][0]["miss_sweep_tiles"]) == 4
    assert len(list((output / "miss-sweep-tiles").glob("*.jpg"))) == 4
    assert "fastener" in task["target_definitions"]
    assert "added_boxes" in task["instructions"]


def test_review_pack_can_prioritize_selected_images(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (100, 100), "gray").save(source / name)
    candidates = {
        "images": [
            {"id": 1, "relative_path": "a.jpg", "scene_group": "g1"},
            {"id": 2, "relative_path": "b.jpg", "scene_group": "g2"},
            {"id": 3, "relative_path": "c.jpg", "scene_group": "g3"},
        ],
        "fused_candidates": [
            {"id": "ca", "image_id": 1, "xyxy": [1, 1, 20, 20], "category": "fastener"},
            {"id": "cb", "image_id": 2, "xyxy": [1, 1, 20, 20], "category": "fastener"},
            {"id": "cc", "image_id": 3, "xyxy": [1, 1, 20, 20], "category": "fastener"},
        ],
    }

    summary = build_pack(
        candidates,
        source,
        tmp_path / "pack",
        selected_relative_paths=["c.jpg", "a.jpg"],
    )
    task = json.loads(
        (tmp_path / "pack" / "first-pass" / "tasks-001.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary.images == 2
    assert summary.candidates == 2
    assert [row["relative_path"] for row in task["images"]] == ["c.jpg", "a.jpg"]


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
