"""Freeze a 40/19 marked-point development selection outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_selection import build_marked_point_selection


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{relative}")
    return path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _identity_sets(rows: list[object]) -> dict[str, set[object]]:
    result: dict[str, set[object]] = {
        "scene_group": set(),
        "relative_path": set(),
        "sha256": set(),
        "image_id": set(),
    }
    for source in rows:
        if not isinstance(source, dict):
            raise ValueError("SEALED_ROW_NOT_OBJECT")
        for field in result:
            value = source.get(field)
            if value in (None, ""):
                raise ValueError(f"SEALED_ROW_MISSING:{field}")
            if field == "sha256":
                value = str(value).lower()
            if field == "relative_path":
                value = str(value).replace("\\", "/")
            result[field].add(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="selections/high-accuracy-v2/partition.json")
    parser.add_argument(
        "--old-partition", default="selections/high-accuracy-v1/partition.json"
    )
    parser.add_argument(
        "--error-pack", default="review-packs/high-accuracy-errors-v2/errors.json"
    )
    parser.add_argument(
        "--candidates", default="runs/safe-auto-candidates-v2.2/candidates.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output", default="selections/marked-point-v1/selection.json"
    )
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--required-val-count", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    root = asset_root()
    paths = {
        name: _below(root, relative)
        for name, relative in {
            "partition": args.partition,
            "old_partition": args.old_partition,
            "error_pack": args.error_pack,
            "candidates": args.candidates,
            "source": args.source,
            "truth": args.truth,
            "output": args.output,
        }.items()
    }
    for name in ("partition", "old_partition", "error_pack", "candidates", "truth"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if not paths["source"].is_dir():
        raise FileNotFoundError(paths["source"])
    output = paths["output"]
    if output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(paths["truth"])
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")

    partition = _load(paths["partition"])
    old_partition = _load(paths["old_partition"])
    errors = _load(paths["error_pack"])
    candidates = _load(paths["candidates"])
    if partition.get("sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID:partition")
    if old_partition.get("sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID:old_partition")

    forbidden_rows: list[object] = []
    for document in (partition, old_partition):
        rows = document.get("sealed_test")
        if not isinstance(rows, list):
            raise ValueError("SEALED_PARTITION_INVALID")
        forbidden_rows.extend(rows)
    forbidden = _identity_sets(forbidden_rows)

    image_rows = candidates.get("images")
    fused_rows = candidates.get("fused_candidates")
    if not isinstance(image_rows, list) or not isinstance(fused_rows, list):
        raise ValueError("CANDIDATES_INVALID")
    image_by_path: dict[str, dict[str, object]] = {}
    for source in image_rows:
        if not isinstance(source, dict):
            raise ValueError("CANDIDATE_IMAGE_INVALID")
        relative = str(source.get("relative_path") or "").replace("\\", "/")
        if not relative or relative in image_by_path:
            raise ValueError("CANDIDATE_IMAGE_IDENTITY_INVALID")
        image_by_path[relative] = source
    current_counts = Counter(
        source.get("image_id")
        for source in fused_rows
        if isinstance(source, dict) and source.get("image_id") is not None
    )

    def enrich(source: dict[str, object]) -> dict[str, object]:
        row = dict(source)
        relative = str(row.get("relative_path") or "").replace("\\", "/")
        current = image_by_path.get(relative)
        if current is None:
            raise ValueError(f"IMAGE_NOT_IN_CURRENT_CANDIDATES:{relative}")
        for field in ("image_id", "scene_group", "sha256"):
            left = str(row.get(field)).lower()
            right = str(current.get("id" if field == "image_id" else field)).lower()
            if left != right:
                raise ValueError(f"CURRENT_CANDIDATE_IDENTITY_MISMATCH:{field}:{relative}")
        row["fused_candidate_count"] = current_counts[current["id"]]
        return row

    partition_train = partition.get("train")
    partition_val = partition.get("val")
    if not isinstance(partition_train, list) or not isinstance(partition_val, list):
        raise ValueError("PARTITION_INVALID")
    val_scenes: set[str] = set()
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    records = errors.get("records")
    if not isinstance(records, list):
        raise ValueError("ERROR_PACK_INVALID")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("ERROR_RECORD_INVALID")
        scene = str(record.get("scene_group") or "")
        bucket = str(record.get("primary_bucket") or "none")
        if not scene:
            raise ValueError("ERROR_RECORD_SCENE_MISSING")
        val_scenes.add(scene)
        buckets[scene][bucket] += 1
    if len(val_scenes) != args.required_val_count:
        raise ValueError(
            f"VAL_SCENES_COUNT_MISMATCH:{len(val_scenes)}:required={args.required_val_count}"
        )

    def is_forbidden(row: dict[str, object]) -> bool:
        return any(
            (
                str(row.get(field)).lower()
                if field == "sha256"
                else str(row.get(field)).replace("\\", "/")
                if field == "relative_path"
                else row.get(field)
            )
            in values
            for field, values in forbidden.items()
        )

    train_rows = [
        enrich(row)
        for row in partition_train
        if isinstance(row, dict) and not is_forbidden(row)
    ]
    val_rows = []
    for source in partition_val:
        if not isinstance(source, dict):
            raise ValueError("PARTITION_ROW_INVALID")
        scene = str(source.get("scene_group") or "")
        if scene not in val_scenes:
            continue
        if is_forbidden(source):
            raise ValueError(f"OLD_SEALED_OVERLAP:val:{scene}")
        row = enrich(source)
        counts = buckets[scene]
        row["dominant_error_bucket"] = sorted(
            counts, key=lambda name: (-counts[name], name)
        )[0]
        val_rows.append(row)
    if len(val_rows) != args.required_val_count:
        raise ValueError(
            f"VAL_SCENES_NOT_FOUND:{len(val_rows)}:required={args.required_val_count}"
        )

    selection = build_marked_point_selection(
        train_rows=train_rows,
        val_rows=val_rows,
        old_sealed_hashes=set(),
        train_count=args.train_count,
        seed=args.seed,
    )
    selection["forbidden_old_sealed"] = {
        "scenes": sorted(str(value) for value in forbidden["scene_group"]),
        "paths": sorted(str(value) for value in forbidden["relative_path"]),
        "sha256": sorted(str(value) for value in forbidden["sha256"]),
        "image_ids": sorted(forbidden["image_id"], key=str),
    }
    selection["input_hashes"] = {
        "partition_sha256": _sha256(paths["partition"]),
        "old_partition_sha256": _sha256(paths["old_partition"]),
        "error_pack_sha256": _sha256(paths["error_pack"]),
        "candidates_sha256": _sha256(paths["candidates"]),
        "formal_truth_sha256": truth_before,
    }

    for split in ("train", "val"):
        for row in selection[split]:
            source = paths["source"] / str(row["relative_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            if _sha256(source).lower() != str(row["sha256"]).lower():
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{row['relative_path']}")

    truth_after = _sha256(paths["truth"])
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_SELECTION")
    selection["formal_truth_sha256"] = truth_after
    _atomic_json(output, selection)
    print(
        json.dumps(
            {
                "train_scenes": len(selection["train"]),
                "val_scenes": len(selection["val"]),
                "old_sealed_overlap": 0,
                "forbidden_old_sealed_scenes": len(forbidden["scene_group"]),
                "formal_truth_sha256": truth_after,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
