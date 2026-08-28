"""Deterministic, scene-isolated partitions for the high-accuracy data loop."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from typing import Any


PARTITIONS = ("train", "val", "sealed_test")
IDENTITY_FIELDS = ("scene_group", "sha256", "image_id", "relative_path")


@dataclass(frozen=True)
class HighAccuracyPartition:
    train_scenes: tuple[str, ...]
    val_scenes: tuple[str, ...]
    sealed_test_scenes: tuple[str, ...]
    scene_representatives: dict[str, dict[str, object]]
    seed: int


def _as_scene(row: Mapping[str, object]) -> str:
    scene = str(row.get("scene_group") or "")
    if not scene:
        raise ValueError("MANIFEST_ROW_MISSING:scene_group")
    return scene


def _as_path(row: Mapping[str, object]) -> str:
    path = str(row.get("relative_path") or "")
    if not path:
        raise ValueError("MANIFEST_ROW_MISSING:relative_path")
    return path


def _representatives(
    rows: Sequence[Mapping[str, object]],
    fixed_representatives_by_scene: Mapping[str, str] | None,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_as_scene(row)].append(row)

    fixed = dict(fixed_representatives_by_scene or {})
    unknown = set(fixed) - set(grouped)
    if unknown:
        raise ValueError(f"FIXED_REPRESENTATIVE_UNKNOWN_SCENE:{sorted(unknown)[0]}")

    output: dict[str, dict[str, object]] = {}
    for scene in sorted(grouped):
        candidates = grouped[scene]
        fixed_path = fixed.get(scene)
        if fixed_path is not None:
            candidates = [row for row in candidates if _as_path(row) == fixed_path]
            if len(candidates) != 1:
                raise ValueError(f"FIXED_REPRESENTATIVE_NOT_FOUND:{scene}")
        selected = sorted(
            candidates,
            key=lambda row: (-float(row.get("focus_score", 0.0)), _as_path(row)),
        )[0]
        representative = dict(selected)
        for field in IDENTITY_FIELDS:
            value = representative.get(field)
            if value is None or value == "":
                raise ValueError(f"MANIFEST_ROW_MISSING:{field}")
        output[scene] = representative
    return output


def _rank_quartiles(
    representatives: Mapping[str, Mapping[str, object]],
    scenes: Iterable[str],
    field: str,
) -> dict[str, int]:
    ordered = sorted(
        scenes,
        key=lambda scene: (float(representatives[scene].get(field, 0.0)), scene),
    )
    count = len(ordered)
    if count == 0:
        return {}
    return {scene: min(3, rank * 4 // count) for rank, scene in enumerate(ordered)}


def _capture_timestamp(value: object) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise ValueError(f"INVALID_CAPTURE_TIME:{text}") from exc


def _stratified_order(
    representatives: Mapping[str, Mapping[str, object]],
    scenes: set[str],
    seed: int,
) -> list[str]:
    enriched: dict[str, dict[str, object]] = {
        scene: {
            **representatives[scene],
            "capture_timestamp": _capture_timestamp(
                representatives[scene].get("captured_at")
            ),
        }
        for scene in scenes
    }
    quartiles = {
        field: _rank_quartiles(enriched, scenes, field)
        for field in (
            "capture_timestamp",
            "focus_score",
            "brightness",
            "fused_candidate_count",
        )
    }
    split_strata: dict[str, dict[tuple[int, ...], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for scene in sorted(scenes):
        row = enriched[scene]
        source_split = str(row.get("split") or "unknown")
        key = (
            quartiles["capture_timestamp"][scene],
            quartiles["focus_score"][scene],
            quartiles["brightness"][scene],
            quartiles["fused_candidate_count"][scene],
        )
        split_strata[source_split][key].append(scene)

    split_orders: dict[str, list[str]] = {}
    for source_split, strata in split_strata.items():
        for values in strata.values():
            values.sort(
                key=lambda scene: (
                    hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(),
                    scene,
                )
            )
        keys = sorted(
            strata,
            key=lambda key: hashlib.sha256(
                f"{seed}:{source_split}:{key!r}".encode()
            ).hexdigest(),
        )
        split_order: list[str] = []
        offset = 0
        while len(split_order) < sum(len(values) for values in strata.values()):
            for key in keys:
                values = strata[key]
                if offset < len(values):
                    split_order.append(values[offset])
            offset += 1
        split_orders[source_split] = split_order

    # Capacity-weighted interleaving preserves the original split mix in every
    # prefix, so taking test first and validation second cannot exhaust one source
    # split before sampling the other.
    capacity = {key: len(values) for key, values in split_orders.items()}
    used = {key: 0 for key in split_orders}
    tie_breaker = {
        key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        for key in split_orders
    }
    ordered: list[str] = []
    while len(ordered) < len(scenes):
        available = [key for key in split_orders if used[key] < capacity[key]]
        selected = min(
            available,
            key=lambda key: (
                Fraction(used[key] + 1, capacity[key]),
                tie_breaker[key],
                key,
            ),
        )
        ordered.append(split_orders[selected][used[selected]])
        used[selected] += 1
    return ordered


def build_high_accuracy_partition(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    existing_train_scenes: set[str],
    existing_val_scenes: set[str],
    new_train_count: int,
    new_val_count: int,
    sealed_test_count: int,
    seed: int,
    fixed_representatives_by_scene: Mapping[str, str] | None = None,
) -> HighAccuracyPartition:
    """Freeze existing splits and stratify every previously unused scene."""

    overlap = existing_train_scenes & existing_val_scenes
    if overlap:
        raise ValueError(f"EXISTING_SPLIT_OVERLAP:{sorted(overlap)[0]}")
    representatives = _representatives(
        manifest_rows, fixed_representatives_by_scene
    )
    all_scenes = set(representatives)
    existing = existing_train_scenes | existing_val_scenes
    missing = existing - all_scenes
    if missing:
        raise ValueError(f"EXISTING_SCENE_NOT_IN_MANIFEST:{sorted(missing)[0]}")
    unused = all_scenes - existing
    requested = new_train_count + new_val_count + sealed_test_count
    if len(unused) != requested:
        raise ValueError(
            f"UNUSED_SCENE_QUOTA_MISMATCH:available={len(unused)}:requested={requested}"
        )

    ordered = _stratified_order(representatives, unused, seed)
    sealed = set(ordered[:sealed_test_count])
    new_val_start = sealed_test_count
    new_val = set(ordered[new_val_start : new_val_start + new_val_count])
    new_train = set(ordered[new_val_start + new_val_count :])
    if len(new_train) != new_train_count:
        raise AssertionError("internal new-train quota error")

    result = HighAccuracyPartition(
        train_scenes=tuple(sorted(existing_train_scenes | new_train)),
        val_scenes=tuple(sorted(existing_val_scenes | new_val)),
        sealed_test_scenes=tuple(sorted(sealed)),
        scene_representatives=representatives,
        seed=seed,
    )
    assert not (set(result.sealed_test_scenes) & existing)
    assert_partition_isolated(partition_document(result, input_hashes={}))
    return result


def partition_document(
    partition: HighAccuracyPartition,
    *,
    input_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Return the canonical JSON-ready partition document."""

    def rows(scenes: Sequence[str]) -> list[dict[str, object]]:
        return [dict(partition.scene_representatives[scene]) for scene in scenes]

    document: dict[str, object] = {
        "schema_version": "high-accuracy-partition-v1",
        "seed": partition.seed,
        "sealed_test_opened": False,
        "train": rows(partition.train_scenes),
        "val": rows(partition.val_scenes),
        "sealed_test": rows(partition.sealed_test_scenes),
        "input_hashes": dict(sorted(input_hashes.items())),
    }
    assert_partition_isolated(document)
    return document


def assert_partition_isolated(document: Mapping[str, object]) -> None:
    """Reject a scene or representative identity appearing in two partitions."""

    owners: dict[tuple[str, object], str] = {}
    for partition in PARTITIONS:
        rows = document.get(partition)
        if not isinstance(rows, list):
            raise ValueError(f"HIGH_ACCURACY_SPLIT_INVALID:{partition}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"HIGH_ACCURACY_SPLIT_INVALID_ROW:{partition}")
            for field in IDENTITY_FIELDS:
                if field not in row:
                    raise ValueError(f"HIGH_ACCURACY_SPLIT_MISSING:{field}")
                key = (field, row[field])
                previous = owners.setdefault(key, partition)
                if previous != partition:
                    raise ValueError(f"HIGH_ACCURACY_SPLIT_LEAKAGE:{field}")
