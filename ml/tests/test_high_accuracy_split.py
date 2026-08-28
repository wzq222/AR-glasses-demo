from __future__ import annotations

import json
import random

import pytest

from crrc_vision.high_accuracy_split import (
    assert_partition_isolated,
    build_high_accuracy_partition,
    partition_document,
)


def _scene_ids(first: int, last: int) -> set[str]:
    return {f"scene-{index:04d}" for index in range(first, last + 1)}


def _rows_for_177_scenes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 178):
        scene = f"scene-{index:04d}"
        for frame in range(2):
            relative_path = f"{scene}-{frame}.jpg"
            rows.append(
                {
                    "scene_group": scene,
                    "relative_path": relative_path,
                    "sha256": f"{index:04x}{frame:02x}".ljust(64, "0"),
                    "image_id": index * 10 + frame,
                    "split": "train" if index % 2 else "val",
                    "captured_at": f"2024-05-29T{10 + index % 8:02d}:{index % 60:02d}:00",
                    "focus_score": float(index + frame * 1000),
                    "brightness": float((index * 7 + frame) % 256),
                    "fused_candidate_count": (index * 3 + frame) % 41,
                }
            )
    return rows


def _build(rows: list[dict[str, object]]):
    return build_high_accuracy_partition(
        manifest_rows=rows,
        existing_train_scenes=_scene_ids(1, 64),
        existing_val_scenes=_scene_ids(65, 80),
        new_train_count=52,
        new_val_count=15,
        sealed_test_count=30,
        seed=20260828,
    )


def test_partition_uses_all_177_scenes_without_overlap() -> None:
    result = _build(_rows_for_177_scenes())

    assert len(result.train_scenes) == 116
    assert len(result.val_scenes) == 31
    assert len(result.sealed_test_scenes) == 30
    assert len(
        set(result.train_scenes)
        | set(result.val_scenes)
        | set(result.sealed_test_scenes)
    ) == 177
    assert not (set(result.train_scenes) & set(result.val_scenes))
    assert not (set(result.train_scenes) & set(result.sealed_test_scenes))
    assert not (set(result.val_scenes) & set(result.sealed_test_scenes))


def test_sealed_test_never_contains_a_previously_trained_scene() -> None:
    result = _build(_rows_for_177_scenes())

    previously_trained = _scene_ids(1, 80)
    assert not (set(result.sealed_test_scenes) & previously_trained)


def test_partition_preserves_original_split_mix_in_each_new_subset() -> None:
    result = _build(_rows_for_177_scenes())
    representatives = result.scene_representatives
    new_train = set(result.train_scenes) - _scene_ids(1, 64)
    new_val = set(result.val_scenes) - _scene_ids(65, 80)

    for scenes in (new_train, new_val, set(result.sealed_test_scenes)):
        source_splits = {str(representatives[scene]["split"]) for scene in scenes}
        assert source_splits == {"train", "val"}


def test_representative_prefers_focus_then_deterministic_path() -> None:
    rows = _rows_for_177_scenes()
    rows.extend(
        [
            {
                **rows[0],
                "relative_path": "z-tie.jpg",
                "sha256": "a" * 64,
                "image_id": 9001,
                "focus_score": 5000.0,
            },
            {
                **rows[0],
                "relative_path": "a-tie.jpg",
                "sha256": "b" * 64,
                "image_id": 9002,
                "focus_score": 5000.0,
            },
        ]
    )

    result = _build(rows)

    assert result.scene_representatives["scene-0001"]["relative_path"] == "a-tie.jpg"


@pytest.mark.parametrize("field", ["scene_group", "sha256", "image_id", "relative_path"])
def test_isolation_rejects_identity_reused_across_partitions(field: str) -> None:
    train = {
        "scene_group": "scene-a",
        "sha256": "a" * 64,
        "image_id": 1,
        "relative_path": "a.jpg",
    }
    val = {
        "scene_group": "scene-b",
        "sha256": "b" * 64,
        "image_id": 2,
        "relative_path": "b.jpg",
    }
    val[field] = train[field]
    document = {"train": [train], "val": [val], "sealed_test": []}

    with pytest.raises(ValueError, match=f"HIGH_ACCURACY_SPLIT_LEAKAGE:{field}"):
        assert_partition_isolated(document)


def test_partition_json_is_byte_identical_for_reordered_inputs() -> None:
    rows = _rows_for_177_scenes()
    shuffled = list(rows)
    random.Random(71).shuffle(shuffled)

    first = json.dumps(
        partition_document(_build(rows), input_hashes={"manifest": "A"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    second = json.dumps(
        partition_document(_build(shuffled), input_hashes={"manifest": "A"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert first == second


def test_partition_refuses_existing_scene_overlap() -> None:
    with pytest.raises(ValueError, match="EXISTING_SPLIT_OVERLAP"):
        build_high_accuracy_partition(
            manifest_rows=_rows_for_177_scenes(),
            existing_train_scenes={"scene-0001"},
            existing_val_scenes={"scene-0001"},
            new_train_count=52,
            new_val_count=15,
            sealed_test_count=30,
            seed=20260828,
        )
