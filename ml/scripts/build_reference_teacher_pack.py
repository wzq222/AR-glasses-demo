"""Render an auditable review pack from isolated reference-teacher predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from crrc_vision.assets import asset_root
from crrc_vision.reference_teacher import (
    build_proposal_document,
    ensure_complete_selection,
)

COLORS = {1: (0, 210, 255), 2: (0, 220, 0)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("path must stay below CRRC_VISION_DATA_ROOT")
    return path


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _manifest_rows(path: Path) -> dict[str, dict[str, object]]:
    return {
        row["relative_path"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _load_source(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _draw_overlay(
    image: Image.Image,
    annotations: list[dict[str, object]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for annotation in annotations:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        color = COLORS[int(annotation["category_id"])]
        draw.rectangle((x, y, x + width, y + height), outline=color, width=5)
        label = (
            f"{annotation['id']:03d} c{annotation['teacher_class_id']} "
            f"{float(annotation['proposal_score']):.2f}"
        )
        draw.text(
            (x + 2, max(0, y - 18)),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return overlay


def _candidate_tile(
    image: Image.Image,
    annotation: dict[str, object],
    relative_path: str,
) -> Image.Image:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    padding = max(width, height) * 0.65
    crop_box = (
        max(0, int(x - padding)),
        max(0, int(y - padding)),
        min(image.width, int(x + width + padding)),
        min(image.height, int(y + height + padding)),
    )
    crop = image.crop(crop_box)
    crop.thumbnail((280, 225), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (300, 270), "white")
    tile.paste(crop, ((300 - crop.width) // 2, 25 + (225 - crop.height) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (6, 5),
        (
            f"{annotation['id']:03d} c{annotation['teacher_class_id']} "
            f"{float(annotation['proposal_score']):.2f} {Path(relative_path).stem[-6:]}"
        ),
        fill="black",
    )
    return tile


def _write_candidate_sheets(
    output_root: Path,
    document: dict[str, object],
    source_root: Path,
) -> int:
    image_by_id = {int(row["id"]): row for row in document["images"]}
    source_cache: dict[int, Image.Image] = {}
    tiles = []
    for annotation in document["annotations"]:
        image_id = int(annotation["image_id"])
        image_row = image_by_id[image_id]
        if image_id not in source_cache:
            source_cache[image_id] = _load_source(
                source_root / str(image_row["file_name"])
            )
        tiles.append(
            _candidate_tile(
                source_cache[image_id], annotation, str(image_row["file_name"])
            )
        )
    destination = output_root / "candidate-sheets"
    destination.mkdir()
    per_sheet, columns, rows = 24, 4, 6
    for sheet_index in range(math.ceil(len(tiles) / per_sheet)):
        sheet = Image.new("RGB", (columns * 300, rows * 270), (230, 230, 230))
        for local_index, tile in enumerate(
            tiles[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        ):
            sheet.paste(
                tile,
                ((local_index % columns) * 300, (local_index // columns) * 270),
            )
        sheet.save(destination / f"candidates-{sheet_index + 1:02d}.jpg", quality=93)
    return math.ceil(len(tiles) / per_sheet)


def _write_contact_sheets(output_root: Path, overlays: list[Path]) -> int:
    destination = output_root / "contact-sheets"
    destination.mkdir()
    tiles = []
    for path in overlays:
        image = _load_source(path)
        image.thumbnail((500, 375), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (500, 415), "white")
        tile.paste(image, ((500 - image.width) // 2, (375 - image.height) // 2))
        ImageDraw.Draw(tile).text((8, 390), path.name, fill="black")
        tiles.append(tile)
    per_sheet, columns, rows = 8, 2, 4
    for sheet_index in range(math.ceil(len(tiles) / per_sheet)):
        sheet = Image.new("RGB", (columns * 500, rows * 415), (230, 230, 230))
        for local_index, tile in enumerate(
            tiles[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        ):
            sheet.paste(
                tile,
                ((local_index % columns) * 500, (local_index // columns) * 415),
            )
        sheet.save(destination / f"full-images-{sheet_index + 1:02d}.jpg", quality=92)
    return math.ceil(len(tiles) / per_sheet)


def _write_review_indexes(output_root: Path, document: dict[str, object]) -> None:
    image_by_id = {int(row["id"]): row for row in document["images"]}
    with (output_root / "review-index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [
            "annotation_id",
            "proposal_id",
            "relative_path",
            "scene_group",
            "split",
            "mapped_category",
            "teacher_class_id",
            "score",
            "review_status",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        category_by_id = {1: "fastener", 2: "pipe_joint"}
        for annotation in document["annotations"]:
            image = image_by_id[int(annotation["image_id"])]
            writer.writerow(
                {
                    "annotation_id": annotation["id"],
                    "proposal_id": annotation["proposal_id"],
                    "relative_path": image["file_name"],
                    "scene_group": image["scene_group"],
                    "split": image["split"],
                    "mapped_category": category_by_id[int(annotation["category_id"])],
                    "teacher_class_id": annotation["teacher_class_id"],
                    "score": annotation["proposal_score"],
                    "review_status": "unreviewed",
                    "reviewer": "",
                    "notes": "",
                }
            )
    with (output_root / "image-review-index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [
            "image_id",
            "relative_path",
            "scene_group",
            "split",
            "image_review_status",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", default="runs/reference-teacher-v1/raw-predictions.json"
    )
    parser.add_argument("--selection", default="selections/selection-v2.json")
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument(
        "--output", default="review-packs/fastener-v2/reference-teacher-v1"
    )
    args = parser.parse_args()

    root = asset_root()
    predictions_path = _below(root, args.predictions)
    selection_path = _below(root, args.selection)
    manifest_path = _below(root, args.manifest)
    truth_path = _below(root, args.truth)
    source_root = _below(root, args.source)
    output_root = _below(root, args.output)
    for required in (predictions_path, selection_path, manifest_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"review pack already exists: {output_root}")
    output_root.mkdir(parents=True)

    truth_before = _sha256(truth_path)
    raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_items = selection["items"]
    coverage_errors = ensure_complete_selection(
        [str(row["relative_path"]) for row in selected_items],
        [str(row["relative_path"]) for row in raw["images"]],
    )
    if coverage_errors:
        raise RuntimeError(coverage_errors[0])
    manifest = _manifest_rows(manifest_path)
    document = build_proposal_document(
        selected_items, manifest, raw["predictions"]
    )
    _atomic_json(output_root / "teacher-proposals.json", raw)
    _atomic_json(output_root / "instances.proposals.json", document)
    _write_review_indexes(output_root, document)

    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in document["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )
    overlay_root = output_root / "overlays"
    overlay_root.mkdir()
    overlay_paths = []
    for image_row in document["images"]:
        source_path = (source_root / str(image_row["file_name"])).resolve()
        if source_root.resolve() not in source_path.parents or not source_path.is_file():
            raise FileNotFoundError(source_path)
        overlay = _draw_overlay(
            _load_source(source_path),
            annotations_by_image.get(int(image_row["id"]), []),
        )
        overlay_path = overlay_root / str(image_row["file_name"])
        overlay.save(overlay_path, quality=92)
        overlay_paths.append(overlay_path)

    candidate_sheets = _write_candidate_sheets(
        output_root, document, source_root
    )
    contact_sheets = _write_contact_sheets(output_root, overlay_paths)
    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_MUTATED")

    class_counts = Counter(
        int(row["teacher_class_id"]) for row in document["annotations"]
    )
    zero_prediction_images = sum(
        not annotations_by_image.get(int(row["id"])) for row in document["images"]
    )
    output_hashes = {
        str(path.relative_to(output_root)).replace("\\", "/"): _sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    pack_manifest = {
        "schema_version": "reference-teacher-pack-v1",
        "research_only": True,
        "truth_status": "not_training_truth",
        "inputs": {
            "predictions_sha256": _sha256(predictions_path),
            "selection_sha256": _sha256(selection_path),
            "manifest_sha256": _sha256(manifest_path),
            "truth_sha256_before": truth_before,
            "truth_sha256_after": truth_after,
        },
        "counts": {
            "images": len(document["images"]),
            "predictions": len(document["annotations"]),
            "zero_prediction_images": zero_prediction_images,
            "teacher_class_counts": {
                str(key): class_counts.get(key, 0) for key in range(3)
            },
            "candidate_sheets": candidate_sheets,
            "contact_sheets": contact_sheets,
            "overlays": len(overlay_paths),
        },
        "output_hashes": output_hashes,
    }
    _atomic_json(output_root / "pack-manifest.json", pack_manifest)
    print(json.dumps(pack_manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
