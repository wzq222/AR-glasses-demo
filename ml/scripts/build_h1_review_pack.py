from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.hard_sample_review import (  # noqa: E402
    build_review_scales,
    required_second_review_ids,
)
from crrc_vision.synthetic_contract import assert_external_output, sha256_file  # noqa: E402


def _read(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(path)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blind H1 local review pack")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--preannotations", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--preannotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("first", "second"), default="first")
    parser.add_argument("--first-review", type=Path)
    args = parser.parse_args()

    output = assert_external_output(args.output, REPOSITORY_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"review pack output must be empty: {output}")
    jobs_doc = json.loads(args.jobs.read_text(encoding="utf-8"))
    generation_doc = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    preannotation_doc = json.loads(args.preannotations.read_text(encoding="utf-8"))
    jobs = {row["sample_id"]: row for row in jobs_doc["records"]}
    generated = {row["sample_id"]: row for row in generation_doc["records"]}
    preannotations = {row["sample_id"]: row for row in preannotation_doc["records"]}
    selected = set(jobs)
    if args.mode == "second":
        if args.first_review is None:
            raise ValueError("--first-review is required for second mode")
        first = json.loads(args.first_review.read_text(encoding="utf-8"))
        review_manifest = {
            "records": [
                {**generated[sample_id], "intent": jobs[sample_id]["intent"]}
                for sample_id in jobs
            ]
        }
        selected = required_second_review_ids(review_manifest, first)

    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    for sample_id in jobs:
        if sample_id not in selected:
            continue
        job = jobs[sample_id]
        generation = generated[sample_id]
        preannotation = preannotations[sample_id]
        source = args.generated / generation["image_path"]
        image = _read(source)
        mask = _read(args.preannotation_root / preannotation["paint_mask_path"], cv2.IMREAD_GRAYSCALE)
        overlay = image.copy()
        overlay[mask > 0] = (0.35 * overlay[mask > 0] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
        original_path = samples_dir / f"{sample_id}-original.png"
        detail_2x_path = samples_dir / f"{sample_id}-detail-2x.png"
        detail_4x_path = samples_dir / f"{sample_id}-detail-4x.png"
        overlay_path = samples_dir / f"{sample_id}-paint-overlay.png"
        shutil.copy2(source, original_path)
        details = build_review_scales(image)
        _write(detail_2x_path, details["detail_2x"])
        _write(detail_4x_path, details["detail_4x"])
        _write(overlay_path, overlay)
        left = _fit(image, 300, 260)
        right = _fit(overlay, 300, 260)
        tile = np.vstack([np.full((40, 600, 3), 24, dtype=np.uint8), np.hstack([left, right])])
        cv2.putText(tile, sample_id, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
        records.append(
            {
                "sample_id": sample_id,
                "image_sha256": generation["image_sha256"],
                "reference_id": job["reference_id"],
                "target_intent": job["intent"],
                "topology": job["topology"],
                "mark_role": job["mark_role"],
                "original_path": original_path.relative_to(output).as_posix(),
                "detail_2x_path": detail_2x_path.relative_to(output).as_posix(),
                "detail_4x_path": detail_4x_path.relative_to(output).as_posix(),
                "overlay_path": overlay_path.relative_to(output).as_posix(),
                "proposal_bbox_xyxy": preannotation["bbox_xyxy"],
                "review_template": {"decision": "", "reason": ""},
            }
        )
    contact_sheets = []
    for start in range(0, len(tiles), 4):
        batch = tiles[start : start + 4]
        while len(batch) < 4:
            batch.append(np.full_like(tiles[0], 24))
        sheet = np.vstack([np.hstack(batch[:2]), np.hstack(batch[2:])])
        path = output / f"contact-sheet-{start // 4 + 1:02d}.png"
        _write(path, sheet)
        contact_sheets.append({"path": path.name, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": f"h1-{args.mode}-review-pack-v2",
        "blind_to_first_review": args.mode == "second",
        "count": len(records),
        "records": records,
        "contact_sheets": contact_sheets,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
