"""Render full-resolution reviewed truth overlays for independent missed-point scans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from crrc_vision.assets import asset_root


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selections/marked-point-v1/selection.json")
    parser.add_argument("--dataset", default="annotations/marked-point-v1")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument(
        "--output", default="review-packs/marked-point-v1/truth-audit-v1"
    )
    args = parser.parse_args()
    root = asset_root()
    selection_path = _below(root, args.selection)
    dataset_root = _below(root, args.dataset)
    source_root = _below(root, args.source)
    output_root = _below(root, args.output)
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output_root}")
    output_root.mkdir(parents=True)

    boxes_by_path: dict[str, list[list[float]]] = {}
    truth_hashes: dict[str, str] = {}
    for partition in ("train", "val"):
        truth_path = dataset_root / f"instances.{partition}.json"
        truth = _load(truth_path)
        truth_hashes[partition] = _sha256(truth_path)
        image_paths = {
            row["id"]: str(row["file_name"]) for row in truth["images"]
        }
        for annotation in truth["annotations"]:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            boxes_by_path.setdefault(image_paths[annotation["image_id"]], []).append(
                [x, y, x + width, y + height]
            )

    selection = _load(selection_path)
    rows = sorted(
        selection["train"] + selection["val"], key=lambda row: int(row["image_id"])
    )
    records: list[dict[str, object]] = []
    for row in rows:
        relative = str(row["relative_path"])
        source_path = source_root / relative
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        boxes = boxes_by_path.get(relative, [])
        for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=5)
            draw.text(
                (x1, max(0, y1 - 20)),
                str(index),
                fill="white",
                stroke_width=3,
                stroke_fill="black",
            )
        status = "COMPLETE" if relative in boxes_by_path else "NO_POSITIVE_OR_EXCLUDED"
        draw.rectangle((0, 0, 580, 42), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            f"{status} boxes={len(boxes)} {Path(relative).stem}",
            fill=(0, 255, 0) if status == "COMPLETE" else (255, 210, 0),
        )
        output_path = output_root / f"{int(row['image_id']):04d}_{Path(relative).name}"
        image.save(output_path, quality=94)
        records.append(
            {
                "image_id": row["image_id"],
                "relative_path": relative,
                "boxes": len(boxes),
                "overlay": output_path.name,
                "overlay_sha256": _sha256(output_path),
            }
        )
    manifest = {
        "schema_version": "marked-point-truth-audit-pack-v1",
        "selection_sha256": _sha256(selection_path),
        "truth_hashes": truth_hashes,
        "records": records,
        "stats": {
            "images": len(records),
            "positive_boxes": sum(int(row["boxes"]) for row in records),
            "zero_box_images": sum(int(row["boxes"] == 0) for row in records),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["stats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
