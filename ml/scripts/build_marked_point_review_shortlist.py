"""Create a compact visual shortlist where color marks meet fastener proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from crrc_vision.assets import asset_root


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selections/marked-point-v1/selection.json")
    parser.add_argument("--color", default="runs/marked-point-proposals-v1/a-color/proposals.json")
    parser.add_argument("--fastener", default="runs/safe-auto-candidates-v2.2/candidates.json")
    parser.add_argument(
        "--high-accuracy-root",
        help="Use deduplicated broad truth boxes from this annotation directory instead of fastener candidates.",
    )
    parser.add_argument(
        "--exclude-high-accuracy-root",
        help="Skip proposals whose center is already covered by a broad-truth box.",
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--output", default="review-packs/marked-point-v1/shortlist")
    parser.add_argument("--association-padding-factor", type=float, default=0.35)
    args = parser.parse_args()

    root = asset_root()
    selection_path = _below(root, args.selection)
    color_path = _below(root, args.color)
    fastener_path = _below(root, args.fastener)
    source_root = _below(root, args.source)
    output_root = _below(root, args.output)
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output_root}")
    output_root.mkdir(parents=True)
    crop_root = output_root / "crops"
    sheet_root = output_root / "sheets"
    crop_root.mkdir()
    sheet_root.mkdir()

    selection = _load(selection_path)
    selected_rows = selection["train"] + selection["val"]  # type: ignore[operator]
    selected = {str(row["relative_path"]): row for row in selected_rows}
    color_by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _load(color_path)["proposals"]:  # type: ignore[index]
        color_by_path[str(row["relative_path"])].append(row)
    if args.high_accuracy_root:
        broad_root = _below(root, args.high_accuracy_root)
        fastener_rows = []
        for name in ("instances.train.json", "instances.val.json"):
            document = _load(broad_root / name)
            images = {
                row["id"]: row
                for row in document["images"]  # type: ignore[index]
                if str(row["file_name"]) in selected
            }
            for annotation in document["annotations"]:  # type: ignore[index]
                if annotation["image_id"] not in images:
                    continue
                image_row = images[annotation["image_id"]]
                x, y, width, height = (float(value) for value in annotation["bbox"])
                fastener_rows.append(
                    {
                        "id": f"ha-{name}-{annotation['id']}",
                        "image_id": annotation["image_id"],
                        "relative_path": image_row["file_name"],
                        "xyxy": [x, y, x + width, y + height],
                    }
                )
        fastener_input_hash = hashlib.sha256(
            "|".join(
                _sha256(broad_root / name)
                for name in ("instances.train.json", "instances.val.json")
            ).encode("utf-8")
        ).hexdigest().upper()
        association_source = "high-accuracy-v2 broad truth"
    else:
        fastener_rows = [
            row
            for row in _load(fastener_path)["fused_candidates"]  # type: ignore[index]
            if str(row["relative_path"]) in selected
        ]
        fastener_input_hash = _sha256(fastener_path)
        association_source = "safe-auto-candidates-v2.2"

    excluded_broad: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    if args.exclude_high_accuracy_root:
        broad_root = _below(root, args.exclude_high_accuracy_root)
        for name in ("instances.train.json", "instances.val.json"):
            document = _load(broad_root / name)
            images = {
                row["id"]: str(row["file_name"])
                for row in document["images"]  # type: ignore[index]
                if str(row["file_name"]) in selected
            }
            for annotation in document["annotations"]:  # type: ignore[index]
                if annotation["image_id"] not in images:
                    continue
                x, y, width, height = (float(value) for value in annotation["bbox"])
                excluded_broad[images[annotation["image_id"]]].append(
                    (x, y, x + width, y + height)
                )

    source_cache: dict[str, Image.Image] = {}
    records: list[dict[str, object]] = []
    for fastener in fastener_rows:
        relative = str(fastener["relative_path"])
        x1, y1, x2, y2 = (float(value) for value in fastener["xyxy"])
        padding = max(
            8.0, args.association_padding_factor * max(x2 - x1, y2 - y1)
        )
        hits = []
        for mark in color_by_path[relative]:
            mx1, my1, mx2, my2 = (float(value) for value in mark["mark_xyxy"])
            center_x, center_y = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0
            if x1 - padding <= center_x <= x2 + padding and y1 - padding <= center_y <= y2 + padding:
                hits.append(mark)
        if not hits:
            continue
        fastener_center_x, fastener_center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if any(
            broad_x1 - max(8.0, 0.25 * max(broad_x2 - broad_x1, broad_y2 - broad_y1))
            <= fastener_center_x
            <= broad_x2 + max(8.0, 0.25 * max(broad_x2 - broad_x1, broad_y2 - broad_y1))
            and broad_y1 - max(8.0, 0.25 * max(broad_x2 - broad_x1, broad_y2 - broad_y1))
            <= fastener_center_y
            <= broad_y2 + max(8.0, 0.25 * max(broad_x2 - broad_x1, broad_y2 - broad_y1))
            for broad_x1, broad_y1, broad_x2, broad_y2 in excluded_broad[relative]
        ):
            continue
        if relative not in source_cache:
            with Image.open(source_root / relative) as opened:
                source_cache[relative] = ImageOps.exif_transpose(opened).convert("RGB")
        image = source_cache[relative]
        union = [x1, y1, x2, y2]
        for mark in hits:
            box = [float(value) for value in mark["mark_xyxy"]]
            union = [min(union[0], box[0]), min(union[1], box[1]), max(union[2], box[2]), max(union[3], box[3])]
        center_x, center_y = (union[0] + union[2]) / 2.0, (union[1] + union[3]) / 2.0
        side = max(160.0, 2.2 * max(union[2] - union[0], union[3] - union[1]))
        left, top = max(0, math.floor(center_x - side / 2)), max(0, math.floor(center_y - side / 2))
        right, bottom = min(image.width, math.ceil(center_x + side / 2)), min(image.height, math.ceil(center_y + side / 2))
        crop = image.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(0, 255, 0), width=4)
        for mark in hits:
            mx1, my1, mx2, my2 = (float(value) for value in mark["mark_xyxy"])
            color = (255, 40, 40) if mark["color"] == "red" else (255, 220, 0)
            draw.rectangle((mx1 - left, my1 - top, mx2 - left, my2 - top), outline=color, width=3)
        shortlist_id = f"S{len(records) + 1:04d}"
        draw.text((6, 6), shortlist_id, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        crop.thumbnail((380, 330), Image.Resampling.LANCZOS)
        crop_path = crop_root / f"{shortlist_id}.jpg"
        crop.save(crop_path, quality=92)
        records.append(
            {
                "shortlist_id": shortlist_id,
                "image_id": selected[relative]["image_id"],
                "relative_path": relative,
                "scene_group": selected[relative]["scene_group"],
                "partition": "train" if selected[relative] in selection["train"] else "val",  # type: ignore[index]
                "source_sha256": selected[relative]["sha256"],
                "fastener_candidate_id": fastener["id"],
                "fastener_xyxy": fastener["xyxy"],
                "suggested_xyxy": union,
                "mark_ids": [row["id"] for row in hits],
                "mark_colors": sorted({str(row["color"]) for row in hits}),
                "crop": str(crop_path.relative_to(output_root)).replace("\\", "/"),
                "crop_sha256": _sha256(crop_path),
            }
        )

    for sheet_index in range(math.ceil(len(records) / 16)):
        page = records[sheet_index * 16 : (sheet_index + 1) * 16]
        sheet = Image.new("RGB", (1600, 1440), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            column, line = index % 4, index // 4
            with Image.open(output_root / str(row["crop"])) as opened:
                tile = opened.convert("RGB")
            x, y = column * 400, line * 360
            sheet.paste(tile, (x + (400 - tile.width) // 2, y + 25))
            draw.text((x + 5, y + 5), f"{row['shortlist_id']} {Path(str(row['relative_path'])).stem}", fill="white")
        sheet.save(sheet_root / f"sheet-{sheet_index + 1:03d}.jpg", quality=94)

    document = {
        "schema_version": "marked-point-review-shortlist-v1",
        "input_hashes": {
            "selection_sha256": _sha256(selection_path),
            "color_sha256": _sha256(color_path),
            "fastener_sha256": fastener_input_hash,
        },
        "association_source": association_source,
        "association_rule": (
            "color mark center within fastener box padded by "
            f"max(8px, {args.association_padding_factor}*long_side)"
        ),
        "records": records,
        "stats": {"records": len(records), "sheets": math.ceil(len(records) / 16)},
    }
    (output_root / "shortlist.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document["stats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
