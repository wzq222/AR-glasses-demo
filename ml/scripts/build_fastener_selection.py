"""Select representative full images for physical-fastener truth labeling."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from crrc_vision.assets import asset_root
from crrc_vision.selection import SelectionCandidate, select_representatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--candidates", default="annotations/prelabel-v1/instances.json")
    parser.add_argument("--output", default="selections/selection-v2.json")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    args = parser.parse_args()

    root = asset_root()
    manifest_rows = [
        json.loads(line)
        for line in (root / args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coco = json.loads((root / args.candidates).read_text(encoding="utf-8"))
    file_by_id = {image["id"]: image["file_name"] for image in coco["images"]}
    candidate_counts = Counter(file_by_id[row["image_id"]] for row in coco["annotations"])
    candidates = [
        SelectionCandidate(
            relative_path=row["relative_path"],
            scene_group=row["scene_group"],
            split=row["split"],
            focus_score=float(row["focus_score"]),
            candidate_count=candidate_counts[row["relative_path"]],
        )
        for row in manifest_rows
    ]
    selected = select_representatives(candidates, target=args.target, val_count=args.val_count)
    payload = {
        "version": "selection-v2",
        "target": args.target,
        "val_count": args.val_count,
        "items": [item.to_dict() for item in selected],
    }
    output = (root / args.output).resolve()
    if root.resolve() not in output.parents:
        raise ValueError("output must stay below CRRC_VISION_DATA_ROOT")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(selected),
                "scene_groups": len({item.scene_group for item in selected}),
                "splits": dict(sorted(Counter(item.split for item in selected).items())),
                "density_buckets": dict(
                    sorted(
                        Counter(
                            "zero"
                            if item.candidate_count == 0
                            else "low"
                            if item.candidate_count <= 2
                            else "medium"
                            if item.candidate_count <= 7
                            else "high"
                            for item in selected
                        ).items()
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
