"""Build an empty, auditable physical-fastener truth labeling pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.fastener_annotations import build_fastener_document, evaluate_fastener_truth


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("output must stay below CRRC_VISION_DATA_ROOT")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selections/selection-v2.json")
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--annotations", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--review-pack", default="review-packs/fastener-v2")
    args = parser.parse_args()

    root = asset_root()
    selection = json.loads((root / args.selection).read_text(encoding="utf-8"))
    manifest = {
        row["relative_path"]: row
        for row in (
            json.loads(line)
            for line in (root / args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    rows = []
    for selected in selection["items"]:
        source_row = manifest[selected["relative_path"]]
        rows.append({**selected, "width": source_row["width"], "height": source_row["height"]})

    document = build_fastener_document(rows)
    annotations_path = _below(root, args.annotations)
    review_root = _below(root, args.review_pack)
    _atomic_json(annotations_path, document)

    tasks = []
    for image in document["images"]:
        source_path = (root / args.source / image["file_name"]).resolve()
        tasks.append(
            {
                "id": image["id"],
                "data": {
                    "image": source_path.as_uri(),
                    "relative_path": image["file_name"],
                },
                "meta": {
                    "scene_group": image["scene_group"],
                    "split": image["split"],
                },
            }
        )
    _atomic_json(review_root / "label-studio-tasks.json", tasks)

    review_root.mkdir(parents=True, exist_ok=True)
    review_index = review_root / "review-index.csv"
    with review_index.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_id",
                "relative_path",
                "scene_group",
                "split",
                "image_review_status",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        for image in document["images"]:
            writer.writerow(
                {
                    "image_id": image["id"],
                    "relative_path": image["file_name"],
                    "scene_group": image["scene_group"],
                    "split": image["split"],
                    "image_review_status": "unreviewed",
                    "reviewer": "",
                    "notes": "",
                }
            )

    report = evaluate_fastener_truth(document)
    _atomic_json(root / "runs" / "fastener-truth-readiness-v2.json", report.to_dict())
    print(
        json.dumps(
            {
                "images": len(document["images"]),
                "annotations": len(document["annotations"]),
                "annotations_path": str(annotations_path),
                "review_pack": str(review_root),
                "truth_readiness": report.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
