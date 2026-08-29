from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_mark_reposition import reposition_imagegen_mark  # noqa: E402
from crrc_vision.synthetic_contract import assert_external_output  # noqa: E402
from crrc_vision.synthetic_state import validate_state  # noqa: E402
from crrc_vision.synthetic_witness_mark import extract_witness_mark_mask  # noqa: E402


STATE_ANGLES = {"NORMAL": 0.0, "SLIGHT_LOOSE": 8.0, "OBVIOUS_LOOSE": 24.0}


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


def _rotate_end(anchor: tuple[float, float], endpoint: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    dx, dy = endpoint[0] - anchor[0], endpoint[1] - anchor[1]
    return (
        anchor[0] + math.cos(radians) * dx - math.sin(radians) * dy,
        anchor[1] + math.sin(radians) * dx + math.cos(radians) * dy,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact-state locals from ImageGen bases and paint pixels")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_external_output(args.output, REPOSITORY_ROOT)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    references_document = json.loads(args.references.read_text(encoding="utf-8"))
    references = {record["reference_id"]: record for record in references_document["records"]}
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "ref-*-repositioned.png",
        "ref-*-repositioned.mark-mask.png",
        "ref-*-repositioned.png.json",
    ):
        for stale_path in args.output.glob(pattern):
            stale_path.unlink()

    donors = {}
    for donor_id, donor_spec in spec["donors"].items():
        donor_image = _read(args.candidates / donor_spec["image"])
        donor_base = _read(args.bases / f"{donor_spec['reference_id']}.png")
        mask = extract_witness_mark_mask(
            donor_image,
            tuple(map(float, donor_spec["mark_roi_xyxy"])),
            padding_fraction=0.0,
            baseline_image=donor_base,
        )
        if np.count_nonzero(mask) < 24:
            raise RuntimeError(f"empty ImageGen donor {donor_id}")
        donors[donor_id] = (
            donor_image,
            mask,
            tuple(tuple(map(float, point)) for point in donor_spec["segment_xyxy"]),
        )

    records = []
    for target in spec["targets"]:
        reference_id = target["reference_id"]
        reference = references[reference_id]
        base = _read(args.bases / f"{reference_id}.png")
        anchor = tuple(map(float, target["anchor_xy"]))
        normal_end = tuple(map(float, target["moving_normal_end_xy"]))
        fixed = tuple(tuple(map(float, point)) for point in target["fixed_segment_xyxy"])
        donor_image, donor_mask, donor_segment = donors[target["donor_id"]]
        for state, base_angle in STATE_ANGLES.items():
            signed_angle = float(target.get("rotation_sign", 1.0)) * base_angle
            moving_end = _rotate_end(anchor, normal_end, signed_angle)
            moving = (anchor, moving_end)
            result = reposition_imagegen_mark(
                base,
                donor_image,
                donor_mask,
                donor_segment_xyxy=donor_segment,
                fixed_target_xyxy=fixed,
                moving_target_xyxy=moving,
            )
            audit = validate_state(state, result.fixed_segment_xyxy, result.moving_segment_xyxy)
            if not audit.accepted:
                raise RuntimeError(f"geometry mismatch {reference_id} {state}: {audit.reason}")
            sample_id = f"{reference_id}-{state.lower()}-repositioned"
            image_name = f"{sample_id}.png"
            mask_name = f"{sample_id}.mark-mask.png"
            _write(args.output / image_name, result.image)
            _write(args.output / mask_name, result.mark_mask)
            prompt_digest = sha256(
                json.dumps(
                    {"method": "imagegen-pixel-reposition-v1", "target": target, "state": state},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest().upper()
            sidecar = {
                "sample_id": sample_id,
                "source_reference_sha256": reference["source_reference_sha256"],
                "source_scene_id": reference["source_scene_id"],
                "source_image": reference["source_image"],
                "source_image_sha256": reference["source_image_sha256"],
                "source_bbox_xywh": reference["source_bbox_xywh"],
                "source_split": "train",
                "state": state,
                "fastener_bbox_xyxy": list(map(float, target["fastener_bbox_xyxy"])),
                "fixed_segment_xyxy": [list(point) for point in result.fixed_segment_xyxy],
                "moving_segment_xyxy": [list(point) for point in result.moving_segment_xyxy],
                "anchor_xy": list(result.anchor_xy),
                "review_status": target.get("review_status", "UNCERTAIN"),
                "prompt_sha256": prompt_digest,
                "witness_mark_mask_path": mask_name,
                "witness_mark_source": "imagegen",
                "paint_transform": "affine_reposition_only",
                "donor_id": target["donor_id"],
                "relative_angle_deg": audit.angle_deg,
            }
            (args.output / f"{image_name}.json").write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            records.append({"sample_id": sample_id, "state": state, "angle_deg": audit.angle_deg})
    summary = {"schema_version": "repositioned-imagegen-batch-v1", "records": records}
    (args.output / "batch-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
