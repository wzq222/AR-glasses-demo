from datetime import datetime, timedelta

from crrc_vision.grouping import group_scenes, split_groups
from crrc_vision.inventory import ImageRecord


def record(name: str, seconds: int, phash: str) -> ImageRecord:
    return ImageRecord(
        relative_path=name,
        sha256=name.zfill(64),
        width=2000,
        height=1500,
        captured_at=datetime(2024, 5, 29, 11, 0, 0) + timedelta(seconds=seconds),
        phash=phash,
        focus_score=100.0,
    )


def test_time_adjacent_frames_share_scene_group() -> None:
    rows = [
        record("a.jpg", 0, "0000000000000000"),
        record("b.jpg", 2, "ffffffffffffffff"),
        record("c.jpg", 20, "aaaaaaaaaaaaaaaa"),
    ]

    groups = group_scenes(rows, max_gap_seconds=3, max_hash_distance=0)

    assert groups["a.jpg"] == groups["b.jpg"]
    assert groups["a.jpg"] != groups["c.jpg"]


def test_perceptual_near_duplicates_share_scene_group() -> None:
    rows = [
        record("a.jpg", 0, "0000000000000000"),
        record("b.jpg", 30, "0000000000000001"),
    ]

    groups = group_scenes(rows, max_gap_seconds=3, max_hash_distance=1)

    assert groups["a.jpg"] == groups["b.jpg"]


def test_group_members_never_cross_splits() -> None:
    rows = [
        record("a.jpg", 0, "0000000000000000"),
        record("b.jpg", 1, "ffffffffffffffff"),
        record("c.jpg", 40, "aaaaaaaaaaaaaaaa"),
        record("d.jpg", 80, "5555555555555555"),
    ]
    groups = group_scenes(rows, max_gap_seconds=3, max_hash_distance=0)

    splits = split_groups(groups, train_ratio=0.5, seed=7)

    assert splits["a.jpg"] == splits["b.jpg"]
    assert set(splits.values()) == {"train", "val"}
