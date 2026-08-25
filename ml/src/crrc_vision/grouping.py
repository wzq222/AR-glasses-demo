"""Scene grouping and leakage-safe dataset splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from .inventory import ImageRecord


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def group_scenes(
    records: list[ImageRecord], *, max_gap_seconds: float = 3.0, max_hash_distance: int = 4
) -> dict[str, str]:
    """Group temporally adjacent or visually near-identical images."""
    if not records:
        return {}

    union_find = _UnionFind(len(records))
    chronological = sorted(range(len(records)), key=lambda index: records[index].captured_at)
    for left, right in zip(chronological, chronological[1:]):
        gap = (records[right].captured_at - records[left].captured_at).total_seconds()
        if gap <= max_gap_seconds:
            union_find.union(left, right)

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if _hash_distance(records[left].phash, records[right].phash) <= max_hash_distance:
                union_find.union(left, right)

    members: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        members[union_find.find(index)].append(record.relative_path)

    ordered = sorted((sorted(paths) for paths in members.values()), key=lambda paths: paths[0])
    return {
        relative_path: f"scene-{group_index:04d}"
        for group_index, paths in enumerate(ordered, start=1)
        for relative_path in paths
    }


def split_groups(
    groups: dict[str, str], *, train_ratio: float = 0.8, seed: int = 20260825
) -> dict[str, str]:
    """Assign complete scene groups to train or validation partitions."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    group_ids = sorted(set(groups.values()))
    random.Random(seed).shuffle(group_ids)
    if len(group_ids) < 2:
        train_ids = set(group_ids)
    else:
        train_count = max(1, min(len(group_ids) - 1, round(len(group_ids) * train_ratio)))
        train_ids = set(group_ids[:train_count])

    return {
        relative_path: "train" if group_id in train_ids else "val"
        for relative_path, group_id in groups.items()
    }
