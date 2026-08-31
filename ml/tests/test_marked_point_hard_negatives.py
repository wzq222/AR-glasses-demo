from __future__ import annotations

from crrc_vision.marked_point_hard_negatives import select_hard_negative_crops


def _truth() -> dict[str, object]:
    return {
        "info": {"partition": "train"},
        "categories": [{"id": 1, "name": "marked_point"}],
        "images": [
            {
                "id": 1,
                "file_name": "one.jpg",
                "width": 2000,
                "height": 1500,
                "scene_group": "scene-1",
                "sha256": "A" * 64,
            },
            {
                "id": 2,
                "file_name": "two.jpg",
                "width": 2000,
                "height": 1500,
                "scene_group": "scene-2",
                "sha256": "B" * 64,
            },
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 80]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [100, 100, 80, 80]},
        ],
    }


def test_hard_negative_selection_is_deterministic_and_bounded_per_scene() -> None:
    predictions = [
        {"image_id": 1, "bbox": [1200, 900, 40, 40], "score": 0.9},
        {"image_id": 1, "bbox": [1700, 900, 40, 40], "score": 0.8},
        {"image_id": 2, "bbox": [1200, 900, 40, 40], "score": 0.7},
    ]

    selected = select_hard_negative_crops(
        predictions, _truth(), score_threshold=0.01, max_per_scene=1
    )

    assert [(row["image_id"], row["score"]) for row in selected] == [(1, 0.9), (2, 0.7)]
    assert all(row["crop_xyxy"][2] - row["crop_xyxy"][0] == 640 for row in selected)

    limited = select_hard_negative_crops(
        predictions,
        _truth(),
        score_threshold=0.01,
        max_per_scene=1,
        maximum_crops=1,
    )
    assert [(row["image_id"], row["score"]) for row in limited] == [(1, 0.9)]


def test_hard_negative_selection_excludes_targets_and_forbidden_hashes() -> None:
    predictions = [
        {"image_id": 1, "bbox": [110, 110, 40, 40], "score": 0.99},
        {"image_id": 1, "bbox": [300, 300, 40, 40], "score": 0.9},
        {"image_id": 1, "bbox": [1400, 900, 40, 40], "score": 0.8},
        {"image_id": 2, "bbox": [1400, 900, 40, 40], "score": 0.7},
    ]

    selected = select_hard_negative_crops(
        predictions,
        _truth(),
        score_threshold=0.01,
        max_per_scene=2,
        forbidden_sha256={"B" * 64},
    )

    assert len(selected) == 1
    assert selected[0]["image_id"] == 1
    assert selected[0]["score"] == 0.8


def test_hard_negative_selection_rejects_non_train_truth() -> None:
    truth = _truth()
    truth["info"] = {"partition": "val"}

    try:
        select_hard_negative_crops([], truth)
    except ValueError as exc:
        assert str(exc) == "HARD_NEGATIVE_TRAIN_TRUTH_REQUIRED"
    else:
        raise AssertionError("expected validation truth rejection")
