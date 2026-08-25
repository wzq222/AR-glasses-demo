"""Build a deterministic, image-free-in-Git review pack in the private asset root."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from crrc_vision.assets import asset_root
from crrc_vision.prelabel import (
    BoundingBox,
    LineSegment,
    MarkedFastenerCandidate,
    VisionPoint,
    read_bgr_image,
)
from crrc_vision.review import render_overlay


def _candidate(annotation: dict[str, object]) -> MarkedFastenerCandidate:
    x, y, width, height = annotation["bbox"]  # type: ignore[misc]
    attributes = annotation["attributes"]  # type: ignore[assignment]
    x1, y1, x2, y2 = attributes["line_points"]
    return MarkedFastenerCandidate(
        BoundingBox(int(x), int(y), int(width), int(height)),
        LineSegment(VisionPoint(int(x1), int(y1)), VisionPoint(int(x2), int(y2)), float(attributes["line_confidence"])),
        str(attributes["mark_color"]),
        float(attributes["candidate_confidence"]),
        float(attributes["mark_area"]),
    )


def _select(images: list[dict[str, object]], counts: dict[int, int], limit: int) -> list[dict[str, object]]:
    ordered = sorted(
        images,
        key=lambda row: (counts.get(int(row["id"]), 0), str(row["scene_group"]), str(row["file_name"])),
    )
    if len(ordered) <= limit:
        return ordered
    selected: list[dict[str, object]] = []
    used_ids: set[int] = set()
    for offset in range(limit):
        index = round(offset * (len(ordered) - 1) / (limit - 1))
        image = ordered[index]
        image_id = int(image["id"])
        if image_id not in used_ids:
            selected.append(image)
            used_ids.add(image_id)
    if len(selected) < limit:
        selected.extend(row for row in ordered if int(row["id"]) not in used_ids and len(selected) < limit)
    return selected


def _write_jpeg(path: Path, image) -> None:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError(f"Cannot encode review image: {path}")
    path.write_bytes(encoded.tobytes())


def _candidate_tile(image, candidate: MarkedFastenerCandidate, label: str):
    box = candidate.bbox
    margin = max(box.width, box.height) // 2
    left = max(0, box.x - margin)
    top = max(0, box.y - margin)
    right = min(image.shape[1], box.x + box.width + margin)
    bottom = min(image.shape[0], box.y + box.height + margin)
    crop = render_overlay(image, [candidate])
    crop = crop[top:bottom, left:right]
    crop = cv2.resize(crop, (250, 220), interpolation=cv2.INTER_AREA)
    tile = np.zeros((250, 250, 3), dtype=np.uint8)
    tile[30:] = crop
    cv2.putText(tile, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations/prelabel-v1/instances.json")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--output", default="review-packs/prelabel-v1")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    root = asset_root()
    document = json.loads((root / args.annotations).read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in document["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    selected = _select(document["images"], {key: len(value) for key, value in by_image.items()}, args.limit)
    output = root / args.output
    originals = output / "originals"
    overlays = output / "overlays"
    originals.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    contact_tiles = []
    candidate_tiles = []
    for row in selected:
        image_id = int(row["id"])
        file_name = str(row["file_name"])
        image = read_bgr_image(root / args.source / file_name)
        image_annotations = by_image[image_id]
        candidates = [_candidate(annotation) for annotation in image_annotations]
        scale = min(1.0, 1280 / image.shape[1])
        thumbnail = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        overlay = render_overlay(image, candidates, label=f"{file_name} candidates={len(candidates)}")
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        safe_name = f"{image_id:04d}_{Path(file_name).stem}.jpg"
        _write_jpeg(originals / safe_name, thumbnail)
        _write_jpeg(overlays / safe_name, overlay)
        contact_tiles.append(cv2.resize(overlay, (420, 315), interpolation=cv2.INTER_AREA))
        if not candidates:
            index_rows.append(
                {
                    "image_id": image_id,
                    "image": file_name,
                    "scene_group": row["scene_group"],
                    "split": row["split"],
                    "candidate_count": 0,
                    "candidate_id": "",
                    "decision": "",
                    "corrected_bbox": "",
                    "comment": "",
                }
            )
        for annotation, candidate in zip(image_annotations, candidates):
            candidate_id = int(annotation["id"])
            index_rows.append(
                {
                    "image_id": image_id,
                    "image": file_name,
                    "scene_group": row["scene_group"],
                    "split": row["split"],
                    "candidate_count": len(candidates),
                    "candidate_id": candidate_id,
                    "decision": "",
                    "corrected_bbox": "",
                    "comment": "",
                }
            )
            candidate_tiles.append(_candidate_tile(image, candidate, f"image={image_id} candidate={candidate_id}"))

    with (output / "review-index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    for start in range(0, len(contact_tiles), 12):
        tiles = contact_tiles[start : start + 12]
        while len(tiles) < 12:
            tiles.append(np.zeros_like(contact_tiles[0]))
        rows = [np.hstack(tiles[row : row + 3]) for row in range(0, 12, 3)]
        _write_jpeg(output / f"contact-sheet-{start // 12 + 1:02d}.jpg", np.vstack(rows))
    for start in range(0, len(candidate_tiles), 25):
        tiles = candidate_tiles[start : start + 25]
        while len(tiles) < 25:
            tiles.append(np.zeros_like(candidate_tiles[0]))
        rows = [np.hstack(tiles[row : row + 5]) for row in range(0, 25, 5)]
        _write_jpeg(output / f"candidate-sheet-{start // 25 + 1:02d}.jpg", np.vstack(rows))
    print(json.dumps({"selected_images": len(selected), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
