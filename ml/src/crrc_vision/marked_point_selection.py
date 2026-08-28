"""Deterministic development selection for marked anti-loosening points."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


IDENTITY_FIELDS = ("scene_group", "relative_path", "sha256", "image_id")
NUMERIC_STRATA = ("brightness", "focus_score", "fused_candidate_count")


def _normalized_row(source: Mapping[str, object]) -> dict[str, object]:
    row = dict(source)
    for field in IDENTITY_FIELDS:
        if row.get(field) in (None, ""):
            raise ValueError(f"SELECTION_ROW_MISSING:{field}")
    digest = str(row["sha256"]).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"INVALID_SHA256:{row['relative_path']}")
    row["sha256"] = digest
    row["scene_group"] = str(row["scene_group"])
    row["relative_path"] = str(row["relative_path"]).replace("\\", "/")
    for field in NUMERIC_STRATA:
        try:
            row[field] = float(row.get(field, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"INVALID_SELECTION_VALUE:{field}") from exc
    row["fused_candidate_count"] = int(row["fused_candidate_count"])
    row["dominant_error_bucket"] = str(
        row.get("dominant_error_bucket") or "none"
    )
    return row


def _assert_unique(rows: Sequence[Mapping[str, object]]) -> None:
    seen: dict[str, set[object]] = {field: set() for field in IDENTITY_FIELDS}
    for row in rows:
        scene = str(row["scene_group"])
        for field in IDENTITY_FIELDS:
            value = row[field]
            if value in seen[field]:
                if field == "scene_group":
                    raise ValueError(f"DUPLICATE_SCENE:{scene}")
                raise ValueError(f"DUPLICATE_IDENTITY:{field}")
            seen[field].add(value)


def _quartiles(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[field]),
            str(row["scene_group"]),
        ),
    )
    count = len(ordered)
    return {
        str(row["scene_group"]): min(3, rank * 4 // count)
        for rank, row in enumerate(ordered)
    }


def _select_train(
    rows: Sequence[dict[str, object]], *, count: int, seed: int
) -> list[dict[str, object]]:
    if len(rows) < count:
        raise ValueError(f"TRAIN_SCENES_TOO_LOW:{len(rows)}:required={count}")
    quartiles = {field: _quartiles(rows, field) for field in NUMERIC_STRATA}
    strata: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        scene = str(row["scene_group"])
        key = (
            quartiles["brightness"][scene],
            quartiles["focus_score"][scene],
            quartiles["fused_candidate_count"][scene],
            row["dominant_error_bucket"],
        )
        strata[key].append(row)
    for values in strata.values():
        values.sort(
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}|marked-point-v1|{row['scene_group']}".encode("utf-8")
                ).hexdigest(),
                str(row["scene_group"]),
            )
        )
    keys = sorted(
        strata,
        key=lambda key: hashlib.sha256(
            f"{seed}|marked-point-v1|stratum|{key!r}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, object]] = []
    offset = 0
    while len(selected) < count:
        progressed = False
        for key in keys:
            values = strata[key]
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise AssertionError("stratified selection exhausted unexpectedly")
        offset += 1
    return sorted(selected, key=lambda row: str(row["scene_group"]))


def build_marked_point_selection(
    *,
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    old_sealed_hashes: set[str],
    train_count: int,
    seed: int,
    old_sealed_paths: set[str] | None = None,
    old_sealed_image_ids: set[object] | None = None,
    old_sealed_scenes: set[str] | None = None,
) -> dict[str, Any]:
    """Select a stable train subset and preserve every validation scene.

    Callers must remove forbidden rows from a larger source pool. Passing one here is
    treated as a leak and fails closed rather than silently changing the population.
    """

    if train_count < 0:
        raise ValueError("INVALID_TRAIN_COUNT")
    train = [_normalized_row(row) for row in train_rows]
    val = [_normalized_row(row) for row in val_rows]
    _assert_unique([*train, *val])

    forbidden = {
        "sha256": {str(value).lower() for value in old_sealed_hashes},
        "relative_path": {
            str(value).replace("\\", "/") for value in (old_sealed_paths or set())
        },
        "image_id": set(old_sealed_image_ids or set()),
        "scene_group": {str(value) for value in (old_sealed_scenes or set())},
    }
    for row in [*train, *val]:
        for field, values in forbidden.items():
            if row[field] in values:
                raise ValueError(
                    f"OLD_SEALED_OVERLAP:{field}:{row['relative_path']}"
                )

    selected_train = _select_train(train, count=train_count, seed=seed)
    selected_val = sorted(val, key=lambda row: str(row["scene_group"]))
    return {
        "schema_version": "marked-point-selection-v1",
        "seed": seed,
        "train": selected_train,
        "val": selected_val,
        "forbidden_old_sealed": {
            "scenes": sorted(forbidden["scene_group"]),
            "paths": sorted(forbidden["relative_path"]),
            "sha256": sorted(forbidden["sha256"]),
            "image_ids": sorted(forbidden["image_id"], key=str),
        },
        "old_sealed_test_opened": False,
    }
