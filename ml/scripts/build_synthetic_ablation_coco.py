from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_ablation import merge_training_documents  # noqa: E402
from crrc_vision.synthetic_audit import audit_full_dataset  # noqa: E402
from crrc_vision.synthetic_contract import assert_external_output, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build audited real+synthetic train COCO for ablation")
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-synthetic-fraction", type=float, default=0.30)
    args = parser.parse_args()
    assert_external_output(args.output, REPOSITORY_ROOT)
    if args.output.exists():
        raise FileExistsError(args.output)
    real = json.loads(args.real_train.read_text(encoding="utf-8"))
    synthetic_coco_path = args.synthetic_root / "instances.synthetic-train.json"
    synthetic_manifest_path = args.synthetic_root / "manifest.json"
    synthetic = json.loads(synthetic_coco_path.read_text(encoding="utf-8"))
    manifest = json.loads(synthetic_manifest_path.read_text(encoding="utf-8"))
    audit = audit_full_dataset(
        manifest,
        args.synthetic_root,
        synthetic_coco_path,
        args.formal_truth,
        review_pack_manifest_path=args.review_pack,
    )
    if not audit.passed:
        raise RuntimeError(f"SYNTHETIC_AUDIT_FAILED:{audit.errors}")
    merged = merge_training_documents(
        real,
        synthetic,
        synthetic_image_root=args.synthetic_root / "images",
        maximum_synthetic_fraction=args.maximum_synthetic_fraction,
    )
    merged["info"].update({
        "real_train_sha256": sha256_file(args.real_train),
        "synthetic_coco_sha256": sha256_file(synthetic_coco_path),
        "synthetic_manifest_sha256": sha256_file(synthetic_manifest_path),
        "review_pack_sha256": sha256_file(args.review_pack),
        "formal_truth_sha256": sha256_file(args.formal_truth),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(merged["info"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
