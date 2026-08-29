from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import assert_external_output, sha256_file  # noqa: E402


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(path)


def _write_contact_sheets(output: Path, records: list[dict]) -> list[dict]:
    sheets = []
    for start in range(0, len(records), 8):
        tiles = []
        for record in records[start:start + 8]:
            crop = _read(output / record["crop_path"])
            scale = min(300 / crop.shape[1], 260 / crop.shape[0])
            resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            tile = np.full((300, 320, 3), 24, dtype=np.uint8)
            left = (320 - resized.shape[1]) // 2
            top = 30 + (260 - resized.shape[0]) // 2
            tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            cv2.putText(tile, f"{record['sample_id']} {record['state']}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 8:
            tiles.append(np.full((300, 320, 3), 24, dtype=np.uint8))
        sheet = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:])))
        filename = f"contact-sheet-{start // 8 + 1:02d}.png"
        _write(output / filename, sheet)
        sheets.append({"path": filename, "sha256": sha256_file(output / filename)})
    return sheets


def _write_full_image_sheets(root: Path, output: Path, records: list[dict], images: dict[int, dict]) -> list[dict]:
    sheets = []
    for start in range(0, len(records), 8):
        tiles = []
        for record in records[start:start + 8]:
            image_id = int(record["sample_id"].split("-")[-1])
            image_record = images[image_id]
            full = _read(root / "images" / image_record["file_name"])
            scale = min(300 / full.shape[1], 260 / full.shape[0])
            resized = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            tile = np.full((300, 320, 3), 24, dtype=np.uint8)
            left = (320 - resized.shape[1]) // 2
            top = 30 + (260 - resized.shape[0]) // 2
            tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            cv2.putText(tile, f"{record['sample_id']} {record['state']}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 8:
            tiles.append(np.full((300, 320, 3), 24, dtype=np.uint8))
        sheet = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:])))
        filename = f"full-contact-sheet-{start // 8 + 1:02d}.png"
        _write(output / filename, sheet)
        sheets.append({"path": filename, "sha256": sha256_file(output / filename)})
    return sheets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build raw context crops for synthetic full-image review")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_external_output(args.output, REPOSITORY_ROOT)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"review output must be empty: {args.output}")

    coco = json.loads((args.root / "instances.synthetic-train.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in coco["images"]}
    states = {record["sample_id"]: record["state"] for record in manifest["records"]}
    targets = [item for item in coco["annotations"] if item.get("origin") == "synthetic_replacement"]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for annotation in targets:
        image_record = images[int(annotation["image_id"])]
        image = _read(args.root / "images" / image_record["file_name"])
        x, y, width, height = map(float, annotation["bbox"])
        crop_side = max(160.0, 3.0 * max(width, height))
        center_x, center_y = x + width / 2.0, y + height / 2.0
        left = max(0, int(round(center_x - crop_side / 2.0)))
        top = max(0, int(round(center_y - crop_side / 2.0)))
        right = min(image.shape[1], int(round(center_x + crop_side / 2.0)))
        bottom = min(image.shape[0], int(round(center_y + crop_side / 2.0)))
        sample_id = f"full-{int(annotation['image_id']):04d}"
        state = states[sample_id]
        filename = f"{sample_id}-{state.lower()}.png"
        _write(args.output / filename, image[top:bottom, left:right])
        crop_hash = sha256_file(args.output / filename)
        records.append({
            "sample_id": sample_id,
            "state": state,
            "crop_path": filename,
            "crop_xyxy": [left, top, right, bottom],
            "target_bbox_xywh": [x, y, width, height],
            "full_image_sha256": image_record["sha256"],
            "crop_sha256": crop_hash,
        })
    contact_sheets = _write_contact_sheets(args.output, records)
    full_contact_sheets = _write_full_image_sheets(args.root, args.output, records, images)
    (args.output / "manifest.json").write_text(
        json.dumps({"schema_version": "synthetic-full-review-v1", "records": records,
                    "contact_sheets": contact_sheets,
                    "full_contact_sheets": full_contact_sheets}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
