import json

import pytest

from crrc_vision.reviewed_coco import (
    assemble_reviewed_coco,
    merge_review_documents,
    merge_reviewed_coco,
    write_reviewed_coco,
)


def candidates_document() -> dict[str, object]:
    return {
        "images": [
            {
                "id": 1,
                "relative_path": "a.jpg",
                "width": 200,
                "height": 100,
                "scene_group": "g1",
                "split": "train",
                "sha256": "A" * 64,
                "synthetic": False,
            },
            {
                "id": 2,
                "relative_path": "b.jpg",
                "width": 100,
                "height": 100,
                "scene_group": "g2",
                "split": "val",
                "sha256": "B" * 64,
                "synthetic": False,
            },
        ],
        "fused_candidates": [
            {
                "id": "c1",
                "image_id": 1,
                "category": "fastener",
                "xyxy": [20, 10, 60, 40],
            },
            {
                "id": "c2",
                "image_id": 1,
                "category": "pipe_joint",
                "xyxy": [80, 20, 120, 60],
            },
            {
                "id": "c3",
                "image_id": 2,
                "category": "fastener",
                "xyxy": [10, 10, 30, 30],
            },
        ],
    }


def first_reviews() -> dict[str, object]:
    return {
        "reviews": [
            {
                "image_id": 1,
                "relative_path": "a.jpg",
                "reviewer": "codex-visual-auditor",
                "task_version": "safe-auto-review-v1",
                "asset_sha256": "C" * 64,
                "first_pass": {"prompt_version": "first-v2", "decision": "accept"},
                "second_pass": None,
                "candidate_decisions": [
                    {"candidate_id": "c1", "decision": "accept"},
                    {"candidate_id": "c2", "decision": "reject"},
                ],
                "added_boxes": [
                    {"category": "pipe_joint", "xyxy": [0.5, 0.5, 0.8, 0.9]}
                ],
                "image_status": "pending_second_pass",
                "reasons": ["one missed target"],
            },
            {
                "image_id": 2,
                "relative_path": "b.jpg",
                "reviewer": "codex-visual-auditor",
                "task_version": "safe-auto-review-v1",
                "asset_sha256": "D" * 64,
                "first_pass": {"prompt_version": "first-v2", "decision": "uncertain"},
                "second_pass": None,
                "candidate_decisions": [
                    {"candidate_id": "c3", "decision": "uncertain"}
                ],
                "added_boxes": [],
                "image_status": "uncertain",
                "reasons": ["motion blur"],
            },
        ]
    }


def second_reviews() -> dict[str, object]:
    return {
        "schema_version": "safe-auto-second-pass-review-v1",
        "prompt_version": "second-v2",
        "first_result_hidden": True,
        "reviews": [
            {
                "image_id": 1,
                "proposal_decisions": [
                    {"proposal_id": "added-1-1", "decision": "accept"}
                ],
                "image_status": "complete",
                "reasons": ["geometry matches visible pixels"],
            }
        ],
    }


def test_assembly_keeps_only_complete_images_and_final_boxes() -> None:
    result = assemble_reviewed_coco(
        candidates_document(),
        first_reviews(),
        second_reviews(),
    )

    assert [image["id"] for image in result.document["images"]] == [1]
    assert result.uncertain_image_ids == (2,)
    assert len(result.document["annotations"]) == 2
    assert result.document["annotations"][0]["bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert result.document["annotations"][1]["bbox"] == [100.0, 50.0, 60.0, 40.0]
    assert all(
        annotation["review_status"] == "accept"
        for annotation in result.document["annotations"]
    )


def test_assembly_refuses_incomplete_candidate_coverage() -> None:
    reviews = first_reviews()
    reviews["reviews"][0]["candidate_decisions"].pop()

    try:
        assemble_reviewed_coco(candidates_document(), reviews, second_reviews())
    except ValueError as error:
        assert "missing candidate decisions" in str(error)
    else:
        raise AssertionError("missing candidate decision was accepted")


def test_assembly_keeps_pending_image_out_when_second_pass_is_missing() -> None:
    result = assemble_reviewed_coco(candidates_document(), first_reviews(), None)

    assert result.document["images"] == []
    assert result.document["annotations"] == []
    assert result.uncertain_image_ids == (1, 2)


def test_reviewed_coco_writer_is_atomic_and_refuses_overwrite(tmp_path) -> None:
    result = assemble_reviewed_coco(
        candidates_document(),
        first_reviews(),
        second_reviews(),
    )

    write_reviewed_coco(result, tmp_path)

    written = json.loads((tmp_path / "instances.reviewed.json").read_text())
    assert [image["id"] for image in written["images"]] == [1]
    assert json.loads((tmp_path / "uncertain-images.json").read_text()) == [2]
    with pytest.raises(FileExistsError):
        write_reviewed_coco(result, tmp_path)


def test_merge_first_pass_reviews_keeps_each_completed_image_once() -> None:
    calibration = {"reviews": [first_reviews()["reviews"][0]]}
    expansion = {"reviews": [first_reviews()["reviews"][1]]}

    merged = merge_review_documents([calibration, expansion], review_kind="first")

    assert [review["image_id"] for review in merged["reviews"]] == [1, 2]


def test_merge_review_documents_refuses_duplicate_image_review() -> None:
    duplicate = {"reviews": [first_reviews()["reviews"][0]]}

    with pytest.raises(ValueError, match="duplicate review image"):
        merge_review_documents([duplicate, duplicate], review_kind="first")


def test_merge_second_pass_reviews_preserves_assembly_contract() -> None:
    review = second_reviews()

    merged = merge_review_documents([review], review_kind="second")

    assert merged["schema_version"] == "safe-auto-second-pass-review-v1"
    assert merged["first_result_hidden"] is True


def test_merge_reviewed_coco_combines_disjoint_versions_and_reassigns_ids() -> None:
    first = {
        "info": {"schema_version": "safe-auto-reviewed-coco-v1"},
        "images": [candidates_document()["images"][0]],
        "annotations": [
            {
                "id": 9,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20.0, 10.0, 40.0, 30.0],
                "area": 1200.0,
                "iscrowd": 0,
                "review_status": "accept",
            }
        ],
        "categories": [
            {"id": 1, "name": "fastener"},
            {"id": 2, "name": "pipe_joint"},
        ],
    }
    second = {
        "info": {"schema_version": "safe-auto-reviewed-coco-v1"},
        "images": [candidates_document()["images"][1]],
        "annotations": [
            {
                "id": 1,
                "image_id": 2,
                "category_id": 2,
                "bbox": [50.0, 50.0, 30.0, 40.0],
                "area": 1200.0,
                "iscrowd": 0,
                "review_status": "accept",
            }
        ],
        "categories": first["categories"],
    }

    merged = merge_reviewed_coco([first, second])

    assert [image["id"] for image in merged["images"]] == [1, 2]
    assert [annotation["id"] for annotation in merged["annotations"]] == [1, 2]
    assert [annotation["image_id"] for annotation in merged["annotations"]] == [1, 2]
    assert merged["categories"] == first["categories"]


def test_merge_reviewed_coco_refuses_duplicate_image_or_scene() -> None:
    document = assemble_reviewed_coco(
        candidates_document(), first_reviews(), second_reviews()
    ).document
    duplicate_scene = json.loads(json.dumps(document))
    duplicate_scene["images"][0]["id"] = 99
    duplicate_scene["annotations"][0]["image_id"] = 99

    with pytest.raises(ValueError, match="duplicate reviewed image"):
        merge_reviewed_coco([document, document])
    with pytest.raises(ValueError, match="duplicate reviewed scene"):
        merge_reviewed_coco([document, duplicate_scene])
