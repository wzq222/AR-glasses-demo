"""Assemble frozen high-accuracy reviewed partitions into Git-external COCO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from crrc_vision.assets import asset_root
from crrc_vision.high_accuracy_dataset import (
    assemble_high_accuracy_dataset,
    assert_uncertain_matches_exclusions,
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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
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


def _uncertain_ids(reviewed_path: Path) -> list[object]:
    uncertain_path = reviewed_path.with_name("uncertain-images.json")
    if not uncertain_path.is_file():
        raise FileNotFoundError(f"UNCERTAIN_MANIFEST_MISSING:{uncertain_path}")
    value = json.loads(uncertain_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"UNCERTAIN_MANIFEST_INVALID:{uncertain_path}")
    return value


def _source_category_counts(documents: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for document in documents:
        names = {
            row.get("id"): str(row.get("name") or "")
            for row in document.get("categories", [])
            if isinstance(row, dict)
        }
        for annotation in document.get("annotations", []):
            if isinstance(annotation, dict):
                counts[names.get(annotation.get("category_id"), "unknown")] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition", default="selections/high-accuracy-v1/partition.json"
    )
    parser.add_argument(
        "--existing-reviewed",
        default="annotations/silver-gate-cumulative-013/instances.silver.json",
    )
    parser.add_argument("--train-reviewed", required=True)
    parser.add_argument("--val-reviewed", required=True)
    parser.add_argument("--sealed-test-reviewed", required=True)
    parser.add_argument("--exclusions")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="annotations/high-accuracy-v1")
    args = parser.parse_args()

    root = asset_root()
    partition_path = _below(root, args.partition)
    existing_path = _below(root, args.existing_reviewed)
    train_path = _below(root, args.train_reviewed)
    val_path = _below(root, args.val_reviewed)
    test_path = _below(root, args.sealed_test_reviewed)
    exclusions_path = _below(root, args.exclusions) if args.exclusions else None
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    inputs = (partition_path, existing_path, train_path, val_path, test_path, truth_path)
    if exclusions_path is not None:
        inputs = (*inputs, exclusions_path)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OUTPUT_DIRECTORY_NOT_EMPTY:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(truth_path)
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    partition = _load(partition_path)
    existing = _load(existing_path)
    exclusions = _load(exclusions_path) if exclusions_path is not None else None
    new_documents = {
        "train": _load(train_path),
        "val": _load(val_path),
        "sealed_test": _load(test_path),
    }
    for split, path in (
        ("train", train_path),
        ("val", val_path),
        ("sealed_test", test_path),
    ):
        uncertain = _uncertain_ids(path)
        if exclusions is None:
            if uncertain:
                raise ValueError(
                    f"PARTITION_REVIEW_INCOMPLETE:{path}:uncertain={len(uncertain)}"
                )
        else:
            assert_uncertain_matches_exclusions(uncertain, exclusions, split)
    for split in ("train", "val", "sealed_test"):
        for row in partition[split]:
            source_path = source_root / str(row["relative_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if _sha256(source_path) != str(row["sha256"]).upper():
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{row['relative_path']}")

    result = assemble_high_accuracy_dataset(
        partition,
        existing,
        new_documents,
        exclusion_document=exclusions,
    )
    paths = {
        "train": output_root / "instances.train.json",
        "val": output_root / "instances.val.json",
        "sealed_test": output_root / "instances.sealed-test.json",
    }
    for split, path in paths.items():
        _atomic_json(path, result[split])
    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_ASSEMBLY")
    source_documents = [existing, *new_documents.values()]
    manifest = {
        "schema_version": "high-accuracy-dataset-manifest-v1",
        "counts": result["counts"],
        "boxes": {
            split: len(result[split]["annotations"])
            for split in ("train", "val", "sealed_test")
        },
        "exclusions": result["exclusions"],
        "source_category_counts_before_merge": _source_category_counts(
            source_documents
        ),
        "output_hashes": {
            path.name: _sha256(path) for path in paths.values()
        },
        "input_hashes": {path.name: _sha256(path) for path in inputs},
        "formal_truth_sha256": truth_after,
        "sealed_test_opened": False,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
