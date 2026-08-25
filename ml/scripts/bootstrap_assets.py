"""Create the auditable source manifest and leakage-safe phase-one split."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.grouping import group_scenes, split_groups
from crrc_vision.inventory import scan_images


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--max-gap-seconds", type=float, default=3.0)
    parser.add_argument("--max-hash-distance", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    root = asset_root()
    source = (root / args.source).resolve()
    records = scan_images(source)
    if not records:
        raise RuntimeError(f"No supported images found below {source}")

    groups = group_scenes(
        records,
        max_gap_seconds=args.max_gap_seconds,
        max_hash_distance=args.max_hash_distance,
    )
    splits = split_groups(groups, train_ratio=args.train_ratio, seed=args.seed)

    manifest_path = root / "manifest.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            row = record.to_dict()
            row["scene_group"] = groups[record.relative_path]
            row["split"] = splits[record.relative_path]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)

    split_payload = {
        "version": "phase1",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "grouping": {
            "max_gap_seconds": args.max_gap_seconds,
            "max_hash_distance": args.max_hash_distance,
        },
        "files": splits,
    }
    _write_json(root / "splits" / "phase1.json", split_payload)

    split_counts = Counter(splits.values())
    hashes = Counter(record.sha256 for record in records)
    focus = sorted(record.focus_score for record in records)
    audit = {
        "source": str(source),
        "image_count": len(records),
        "exact_duplicate_files": sum(count - 1 for count in hashes.values()),
        "scene_group_count": len(set(groups.values())),
        "split_counts": dict(sorted(split_counts.items())),
        "dimensions": dict(sorted(Counter(f"{row.width}x{row.height}" for row in records).items())),
        "capture_range": {
            "first": min(row.captured_at for row in records).isoformat(),
            "last": max(row.captured_at for row in records).isoformat(),
        },
        "focus_score": {
            "min": min(focus),
            "median": statistics.median(focus),
            "p95": focus[min(len(focus) - 1, round(0.95 * (len(focus) - 1)))],
            "max": max(focus),
        },
    }
    _write_json(root / "runs" / "data-audit-v1.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
