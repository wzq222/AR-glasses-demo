"""Generate COCO prelabels from the private full-image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.coco import build_coco_document
from crrc_vision.prelabel import find_marked_fasteners, read_bgr_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--output", default="annotations/prelabel-v1/instances.json")
    parser.add_argument("--min-mark-area", type=float, default=20.0)
    args = parser.parse_args()

    root = asset_root()
    manifest_path = root / args.manifest
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_rows = []
    candidate_counts: list[int] = []
    for index, row in enumerate(rows, start=1):
        path = root / args.source / row["relative_path"]
        image = read_bgr_image(path)
        candidates = find_marked_fasteners(image, min_mark_area=args.min_mark_area)
        candidate_counts.append(len(candidates))
        image_rows.append(
            (
                row["relative_path"],
                row["width"],
                row["height"],
                row["split"],
                row["scene_group"],
                candidates,
            )
        )
        if index % 50 == 0:
            print(f"processed {index}/{len(rows)}")

    document = build_coco_document(image_rows, algorithm_version="hsv-line-v2")
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)

    summary = {
        "images": len(rows),
        "annotations": len(document["annotations"]),
        "images_without_candidates": sum(count == 0 for count in candidate_counts),
        "max_candidates_per_image": max(candidate_counts, default=0),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
