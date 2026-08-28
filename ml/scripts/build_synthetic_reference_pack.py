from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import (  # noqa: E402
    assert_external_output,
    assert_formal_truth_unchanged,
    sha256_file,
)


STATES = ("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 12 train-only ImageGen reference crops")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reviewed-coco", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    return parser.parse_args()


def _select(coco: dict, count: int) -> list[tuple[dict, dict]]:
    by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    candidates: list[tuple[dict, dict]] = []
    seen_scenes: set[str] = set()
    for image in sorted(coco["images"], key=lambda item: (item["scene_group"], item["id"])):
        scene = str(image["scene_group"])
        annotations = by_image.get(int(image["id"]), [])
        valid = [
            annotation
            for annotation in annotations
            if min(float(annotation["bbox"][2]), float(annotation["bbox"][3])) >= 36.0
            and float(annotation["bbox"][0]) > 4.0
            and float(annotation["bbox"][1]) > 4.0
            and float(annotation["bbox"][0]) + float(annotation["bbox"][2]) < float(image["width"]) - 4.0
            and float(annotation["bbox"][1]) + float(annotation["bbox"][3]) < float(image["height"]) - 4.0
        ]
        if not valid or scene in seen_scenes:
            continue
        best = max(valid, key=lambda item: float(item["bbox"][2]) * float(item["bbox"][3]))
        seen_scenes.add(scene)
        candidates.append((image, best))
    if len(candidates) < count:
        raise RuntimeError(f"eligible train references {len(candidates)} < requested {count}")
    if count == 1:
        return [candidates[len(candidates) // 2]]
    positions = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    return [candidates[position] for position in positions]


def _context_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = map(float, bbox)
    side = max(box_width, box_height) * 2.6
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    left = max(0, int(round(center_x - side / 2.0)))
    top = max(0, int(round(center_y - side / 2.0)))
    right = min(width, int(round(center_x + side / 2.0)))
    bottom = min(height, int(round(center_y + side / 2.0)))
    return left, top, right, bottom


def _prompt(reference_id: str, state: str) -> str:
    state_instruction = {
        "NORMAL": "The two anti-loosening paint segments meet continuously and remain aligned (0 to 3 degrees).",
        "SLIGHT_LOOSE": "The moving-side paint segment is visibly offset by 6 to 12 degrees around the physical joint anchor.",
        "OBVIOUS_LOOSE": "The moving-side paint segment is clearly offset by 18 to 35 degrees around the physical joint anchor.",
    }[state]
    return (
        f"Create one photorealistic close-up industrial inspection photo based on reference {reference_id}. "
        "Preserve the exact fastener or connector type, camera viewpoint, metal material, cabinet environment, "
        "realistic grime, reflections, shallow phone-camera blur and lighting. Keep one clearly identifiable fastening "
        "checkpoint with red or yellow anti-loosening paint split across the fixed and moving parts. "
        f"{state_instruction} Do not rotate the whole fastener to fake the state. "
        "No illustration, CGI, text, watermark, duplicate fasteners, melted geometry or floating components. "
        "Output a realistic local photograph with enough surrounding mechanical context for later compositing."
    )


def main() -> int:
    args = _arguments()
    data_root = args.data_root.resolve()
    output = assert_external_output(args.output, REPOSITORY_ROOT)
    reviewed_coco = (args.reviewed_coco or data_root / "annotations/marked-point-v1.4/instances.train.json").resolve()
    source_dir = (args.source_dir or data_root / "source/20240529-luosi").resolve()
    formal_truth = data_root / "annotations/fastener-v2/instances.json"
    formal_hash_before = assert_formal_truth_unchanged(formal_truth)
    coco = json.loads(reviewed_coco.read_text(encoding="utf-8"))
    selected = _select(coco, args.count)
    crops_dir = output / "crops"
    prompts_dir = output / "prompts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (image_record, annotation) in enumerate(selected, start=1):
        reference_id = f"ref-{index:02d}"
        source_path = source_dir / image_record["file_name"]
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            crop_box = _context_box(annotation["bbox"], image.width, image.height)
            crop = image.crop(crop_box)
            crop_path = crops_dir / f"{reference_id}.png"
            crop.save(crop_path, format="PNG", optimize=True)
        prompt_paths = {}
        prompt_hashes = {}
        for state in STATES:
            prompt = _prompt(reference_id, state)
            prompt_path = prompts_dir / f"{reference_id}-{state.lower()}.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            prompt_paths[state] = prompt_path.relative_to(output).as_posix()
            prompt_hashes[state] = sha256(prompt.encode("utf-8")).hexdigest().upper()
        records.append(
            {
                "reference_id": reference_id,
                "source_split": "train",
                "source_scene_id": image_record["scene_group"],
                "source_image": image_record["file_name"],
                "source_image_sha256": image_record["sha256"],
                "source_reference_sha256": sha256_file(crop_path),
                "crop_path": crop_path.relative_to(output).as_posix(),
                "crop_box_xyxy": list(crop_box),
                "source_bbox_xywh": annotation["bbox"],
                "prompts": prompt_paths,
                "prompt_sha256": prompt_hashes,
            }
        )
    manifest = {
        "schema_version": "synthetic-marked-point-reference-v1",
        "count": len(records),
        "formal_truth_sha256": formal_hash_before,
        "reviewed_coco_sha256": sha256_file(reviewed_coco),
        "records": records,
    }
    (output / "references.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert_formal_truth_unchanged(formal_truth, formal_hash_before)
    print(json.dumps({"output": str(output), "references": len(records), "formal_truth_sha256": formal_hash_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
