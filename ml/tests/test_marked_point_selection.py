import json

import pytest

from crrc_vision.marked_point_selection import build_marked_point_selection


def _row(
    image_id: int,
    scene: str,
    path: str,
    digest: str,
    *,
    brightness: float = 90.0,
    focus_score: float = 30.0,
    candidates: int = 2,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "scene_group": scene,
        "sha256": digest,
        "relative_path": path,
        "brightness": brightness,
        "focus_score": focus_score,
        "fused_candidate_count": candidates,
    }


def test_selection_contains_all_val_and_no_old_sealed_rows():
    result = build_marked_point_selection(
        train_rows=[_row(1, "t1", "t.jpg", "1" * 64)],
        val_rows=[_row(2, "v1", "v.jpg", "2" * 64)],
        old_sealed_hashes={"3" * 64},
        train_count=1,
        seed=20260828,
    )
    assert [row["scene_group"] for row in result["val"]] == ["v1"]
    assert not (
        {row["sha256"] for row in result["train"] + result["val"]}
        & {"3" * 64}
    )
    assert result["old_sealed_test_opened"] is False


@pytest.mark.parametrize(
    ("forbidden_key", "forbidden_value"),
    [
        ("old_sealed_hashes", "1" * 64),
        ("old_sealed_paths", "t.jpg"),
        ("old_sealed_image_ids", 1),
        ("old_sealed_scenes", "t1"),
    ],
)
def test_every_old_sealed_identity_is_forbidden(forbidden_key, forbidden_value):
    kwargs = {
        "old_sealed_hashes": set(),
        "old_sealed_paths": set(),
        "old_sealed_image_ids": set(),
        "old_sealed_scenes": set(),
    }
    kwargs[forbidden_key] = {forbidden_value}
    with pytest.raises(ValueError, match="OLD_SEALED_OVERLAP"):
        build_marked_point_selection(
            train_rows=[_row(1, "t1", "t.jpg", "1" * 64)],
            val_rows=[_row(2, "v1", "v.jpg", "2" * 64)],
            train_count=1,
            seed=20260828,
            **kwargs,
        )


def test_reordered_inputs_produce_byte_identical_json():
    train = [
        _row(
            index,
            f"t{index}",
            f"t{index}.jpg",
            f"{index:064x}",
            brightness=float(index * 10),
            focus_score=float(100 - index),
            candidates=index,
        )
        for index in range(1, 9)
    ]
    val = [
        _row(101, "v1", "v1.jpg", "a" * 64),
        _row(102, "v2", "v2.jpg", "b" * 64),
    ]
    first = build_marked_point_selection(
        train_rows=train,
        val_rows=val,
        old_sealed_hashes=set(),
        train_count=5,
        seed=20260828,
    )
    second = build_marked_point_selection(
        train_rows=list(reversed(train)),
        val_rows=list(reversed(val)),
        old_sealed_hashes=set(),
        train_count=5,
        seed=20260828,
    )
    render = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert render(first) == render(second)


def test_duplicate_scene_is_rejected():
    with pytest.raises(ValueError, match="DUPLICATE_SCENE"):
        build_marked_point_selection(
            train_rows=[
                _row(1, "same", "a.jpg", "1" * 64),
                _row(2, "same", "b.jpg", "2" * 64),
            ],
            val_rows=[],
            old_sealed_hashes=set(),
            train_count=1,
            seed=20260828,
        )


def test_too_few_rows_fails_closed():
    with pytest.raises(ValueError, match="TRAIN_SCENES_TOO_LOW"):
        build_marked_point_selection(
            train_rows=[_row(1, "t1", "t.jpg", "1" * 64)],
            val_rows=[],
            old_sealed_hashes=set(),
            train_count=2,
            seed=20260828,
        )
