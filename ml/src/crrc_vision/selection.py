"""Deterministic, scene-group-safe representative-frame selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class SelectionCandidate:
    relative_path: str
    scene_group: str
    split: str
    focus_score: float
    candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _best_per_group(rows: list[SelectionCandidate]) -> list[SelectionCandidate]:
    best: dict[str, SelectionCandidate] = {}
    for row in sorted(rows, key=lambda item: item.relative_path):
        current = best.get(row.scene_group)
        score = (row.focus_score, -abs(row.candidate_count - 3), row.relative_path)
        if current is None:
            best[row.scene_group] = row
            continue
        current_score = (
            current.focus_score,
            -abs(current.candidate_count - 3),
            current.relative_path,
        )
        if score > current_score:
            best[row.scene_group] = row
    return sorted(best.values(), key=lambda item: item.scene_group)


def _density_bucket(candidate_count: int) -> int:
    if candidate_count == 0:
        return 0
    if candidate_count <= 2:
        return 1
    if candidate_count <= 7:
        return 2
    return 3


def _stratified_take(rows: list[SelectionCandidate], count: int) -> list[SelectionCandidate]:
    buckets: dict[int, list[SelectionCandidate]] = {index: [] for index in range(4)}
    for row in rows:
        buckets[_density_bucket(row.candidate_count)].append(row)
    for values in buckets.values():
        values.sort(key=lambda item: (-item.focus_score, item.scene_group, item.relative_path))

    selected: list[SelectionCandidate] = []
    while len(selected) < count:
        made_progress = False
        for bucket in range(4):
            if buckets[bucket] and len(selected) < count:
                selected.append(buckets[bucket].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def select_representatives(
    rows: list[SelectionCandidate], *, target: int = 100, val_count: int = 20
) -> list[SelectionCandidate]:
    """Choose one representative per group with split and density coverage."""
    representatives = _best_per_group(rows)
    if target < 1 or target > len(representatives):
        raise ValueError("target must fit the available scene groups")
    if val_count < 0 or val_count > target:
        raise ValueError("target validation quota is invalid")

    validation = [row for row in representatives if row.split == "val"]
    training = [row for row in representatives if row.split == "train"]
    train_count = target - val_count
    if val_count > len(validation) or train_count > len(training):
        raise ValueError("target split quota is unavailable")

    selected = _stratified_take(validation, val_count) + _stratified_take(training, train_count)
    return sorted(selected, key=lambda item: (item.split, item.scene_group, item.relative_path))

