"""Freeze the Git-external high-accuracy train/val/sealed-test partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from crrc_vision.assets import asset_root
from crrc_vision.high_accuracy_split import (
    build_high_accuracy_partition,
    partition_document,
)


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"MANIFEST_ROW_NOT_OBJECT:{number}")
        rows.append(value)
    if not rows:
        raise ValueError("MANIFEST_EMPTY")
    return rows


def _brightness(path: Path) -> float:
    with Image.open(path) as image:
        gray = image.convert("L")
        return round(float(ImageStat.Stat(gray).mean[0]), 6)


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument(
        "--reviewed",
        default="annotations/silver-gate-cumulative-013/instances.silver.json",
    )
    parser.add_argument(
        "--candidates", default="runs/safe-auto-candidates-v2.2/candidates.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="selections/high-accuracy-v1")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    root = asset_root()
    manifest_path = _below(root, args.manifest)
    reviewed_path = _below(root, args.reviewed)
    candidates_path = _below(root, args.candidates)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    for required in (manifest_path, reviewed_path, candidates_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OUTPUT_DIRECTORY_NOT_EMPTY:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(truth_path)
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(
            f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}:"
            f"expected={EXPECTED_FORMAL_TRUTH_SHA256}"
        )

    manifest_rows = _load_manifest(manifest_path)
    reviewed = _load_json(reviewed_path)
    candidates = _load_json(candidates_path)
    candidate_images = candidates.get("images")
    fused_candidates = candidates.get("fused_candidates")
    reviewed_images = reviewed.get("images")
    if not isinstance(candidate_images, list) or not isinstance(fused_candidates, list):
        raise ValueError("CANDIDATE_DOCUMENT_INVALID")
    if not isinstance(reviewed_images, list):
        raise ValueError("REVIEWED_DOCUMENT_INVALID")

    image_id_by_path: dict[str, object] = {}
    for row in candidate_images:
        if not isinstance(row, dict):
            raise ValueError("CANDIDATE_IMAGE_INVALID")
        relative_path = str(row.get("relative_path") or "")
        image_id = row.get("id")
        if not relative_path or image_id is None or relative_path in image_id_by_path:
            raise ValueError("CANDIDATE_IMAGE_IDENTITY_INVALID")
        image_id_by_path[relative_path] = image_id
    candidate_counts = Counter(
        row.get("image_id")
        for row in fused_candidates
        if isinstance(row, dict) and row.get("image_id") is not None
    )

    enriched: list[dict[str, object]] = []
    for row in manifest_rows:
        relative_path = str(row.get("relative_path") or "")
        image_id = image_id_by_path.get(relative_path)
        if image_id is None:
            raise ValueError(f"MANIFEST_IMAGE_NOT_IN_CANDIDATES:{relative_path}")
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        enriched.append(
            {
                **row,
                "image_id": image_id,
                "brightness": _brightness(source_path),
                "fused_candidate_count": candidate_counts[image_id],
            }
        )

    existing_train: set[str] = set()
    existing_val: set[str] = set()
    fixed: dict[str, str] = {}
    for row in reviewed_images:
        if not isinstance(row, dict):
            raise ValueError("REVIEWED_IMAGE_INVALID")
        scene = str(row.get("scene_group") or "")
        relative_path = str(row.get("relative_path") or row.get("file_name") or "")
        split = str(row.get("split") or "")
        if not scene or not relative_path or split not in {"train", "val"}:
            raise ValueError("REVIEWED_IMAGE_IDENTITY_INVALID")
        if scene in fixed and fixed[scene] != relative_path:
            raise ValueError(f"MULTIPLE_REVIEWED_IMAGES_PER_SCENE:{scene}")
        fixed[scene] = relative_path
        (existing_train if split == "train" else existing_val).add(scene)

    partition = build_high_accuracy_partition(
        manifest_rows=enriched,
        existing_train_scenes=existing_train,
        existing_val_scenes=existing_val,
        new_train_count=52,
        new_val_count=15,
        sealed_test_count=30,
        seed=args.seed,
        fixed_representatives_by_scene=fixed,
    )
    document = partition_document(
        partition,
        input_hashes={
            "manifest_sha256": _sha256(manifest_path),
            "reviewed_sha256": _sha256(reviewed_path),
            "candidates_sha256": _sha256(candidates_path),
            "formal_truth_sha256": truth_before,
        },
    )

    # Verify every selected representative against the source bytes before sealing.
    for split in ("train", "val", "sealed_test"):
        for row in document[split]:
            if not isinstance(row, dict):
                raise AssertionError("partition row is not an object")
            source_hash = _sha256(source_root / str(row["relative_path"]))
            if source_hash != str(row["sha256"]).upper():
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{row['relative_path']}")

    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_PARTITION")
    _atomic_json(output_root / "partition.json", document)
    print(
        json.dumps(
            {
                "train_scenes": len(partition.train_scenes),
                "val_scenes": len(partition.val_scenes),
                "sealed_test_scenes": len(partition.sealed_test_scenes),
                "truth_sha256": truth_after,
                "output": str(output_root / "partition.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
