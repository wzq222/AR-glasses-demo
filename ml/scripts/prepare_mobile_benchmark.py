"""Prepare a provenance-bound, Git-external mobile benchmark run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.mobile_benchmark import prepare_benchmark_manifest


EXPECTED_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--run", required=True)
    parser.add_argument("--runtime", choices=("ncnn", "mnn"), required=True)
    parser.add_argument("--runtime-revision", required=True)
    args = parser.parse_args()

    root = asset_root().resolve()
    model = _below(root, args.model)
    truth = _below(root, args.truth)
    run = _below(root, args.run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("BENCHMARK_RUN_NOT_EMPTY")
    run.mkdir(parents=True, exist_ok=True)
    manifest = prepare_benchmark_manifest(
        candidate=args.candidate,
        model_path=model,
        formal_truth_path=truth,
        expected_truth_sha256=EXPECTED_TRUTH_SHA256,
        runtime_name=args.runtime,
        runtime_revision=args.runtime_revision,
    )
    _atomic_json(run / "benchmark-manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
