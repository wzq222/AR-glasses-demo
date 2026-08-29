from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.hard_sample_preannotation import preannotate_h1  # noqa: E402
from crrc_vision.synthetic_contract import (  # noqa: E402
    assert_external_output,
    assert_formal_truth_unchanged,
    sha256_file,
)


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode generated image: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode mask: {path}")
    encoded.tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create review-only H1 paint proposals")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path)
    args = parser.parse_args()

    output = assert_external_output(args.output, REPOSITORY_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"preannotation output must be empty: {output}")
    root = os.environ.get("CRRC_VISION_DATA_ROOT", "")
    if args.formal_truth is None and not root:
        raise RuntimeError("set CRRC_VISION_DATA_ROOT or pass --formal-truth")
    formal_truth = (
        args.formal_truth
        if args.formal_truth is not None
        else Path(root) / "annotations/fastener-v2/instances.json"
    ).resolve()
    formal_hash = assert_formal_truth_unchanged(formal_truth)

    jobs_doc = json.loads(args.jobs.read_text(encoding="utf-8"))
    generation_doc = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    if jobs_doc.get("formal_truth_sha256") != formal_hash or generation_doc.get(
        "formal_truth_sha256"
    ) != formal_hash:
        raise RuntimeError("formal truth lineage mismatch")
    jobs = {row["sample_id"]: row for row in jobs_doc["records"]}
    generated = {row["sample_id"]: row for row in generation_doc["records"]}
    if set(jobs) != set(generated):
        raise RuntimeError("jobs and generated sample identities differ")

    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id, job in jobs.items():
        generation = generated[sample_id]
        image_path = args.generated / generation["image_path"]
        if sha256_file(image_path) != generation["image_sha256"]:
            raise RuntimeError(f"generated image hash mismatch: {sample_id}")
        proposal = preannotate_h1(_read(image_path), intent=job["intent"])
        mask = proposal.pop("paint_mask")
        mask_path = masks_dir / f"{sample_id}.paint-mask.png"
        _write_png(mask_path, mask)
        records.append(
            {
                **proposal,
                "sample_id": sample_id,
                "image_path": generation["image_path"],
                "image_sha256": generation["image_sha256"],
                "prompt_sha256": generation["prompt_sha256"],
                "paint_mask_path": mask_path.relative_to(output).as_posix(),
                "paint_mask_sha256": sha256_file(mask_path),
            }
        )
    document = {
        "schema_version": "h1-preannotation-v1",
        "formal_truth_sha256": formal_hash,
        "count": len(records),
        "records": records,
    }
    destination = output / "preannotations.json"
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert_formal_truth_unchanged(formal_truth, formal_hash)
    print(json.dumps({"output": str(output), "count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
