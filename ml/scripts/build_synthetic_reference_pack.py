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
from crrc_vision.synthetic_reference import select_reference_candidates  # noqa: E402


STATES = ("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 12 train-only ImageGen reference crops")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reviewed-coco", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    return parser.parse_args()


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
    selected = select_reference_candidates(coco, source_dir, args.count)
    crops_dir = output / "crops"
    prompts_dir = output / "prompts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, candidate in enumerate(selected, start=1):
        image_record = candidate.image
        annotation = candidate.annotation
        reference_id = f"ref-{index:02d}"
        source_path = source_dir / image_record["file_name"]
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            crop_box = candidate.crop_box_xyxy
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
                "brightness": candidate.brightness,
                "sharpness": candidate.sharpness,
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
