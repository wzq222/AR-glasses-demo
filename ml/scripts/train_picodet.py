"""Prepare and optionally execute a pinned PicoDet-S/M training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.picodet import (
    PINNED_PADDLEDETECTION_COMMIT,
    prepare_picodet_dataset,
    write_picodet_config,
)


EXPECTED_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("s", "m"), required=True)
    parser.add_argument(
        "--silver",
        default="annotations/silver-gate-cumulative-013/instances.silver.json",
    )
    parser.add_argument("--source-root", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--run", required=True)
    parser.add_argument("--paddledetection-root", required=True)
    parser.add_argument(
        "--runtime-data-root",
        type=Path,
        help="ASCII-only junction to CRRC_VISION_DATA_ROOT for Windows loaders",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = asset_root().resolve()
    silver = _below(root, args.silver)
    source = _below(root, args.source_root)
    truth = _below(root, args.truth)
    run_root = _below(root, args.run)
    checkout = _below(root, args.paddledetection_root)
    runtime_root = args.runtime_data_root.absolute() if args.runtime_data_root else root
    if runtime_root.resolve() != root or not runtime_root.is_dir():
        raise ValueError("runtime data root must resolve to CRRC_VISION_DATA_ROOT")
    runtime_source = runtime_root / args.source_root
    runtime_checkout = runtime_root / args.paddledetection_root
    runtime_run = runtime_root / args.run
    if not args.python.is_file():
        raise FileNotFoundError(args.python)

    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_PADDLEDETECTION_COMMIT:
        raise RuntimeError(
            f"PADDLEDETECTION_REVISION_MISMATCH:{revision} != {PINNED_PADDLEDETECTION_COMMIT}"
        )

    manifest = prepare_picodet_dataset(
        document_path=silver,
        source_root=source,
        runtime_source_root=runtime_source,
        run_root=run_root,
        formal_truth_path=truth,
        expected_truth_sha256=EXPECTED_TRUTH_SHA256,
    )
    config = write_picodet_config(
        paddledetection_root=checkout,
        runtime_paddledetection_root=runtime_checkout,
        variant=args.variant,
        run_root=run_root,
        runtime_run_root=runtime_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    command = [
        str(args.python.resolve()),
        str((checkout / "tools/train.py").resolve()),
        "-c",
        str(config),
        "--eval",
        "--amp",
    ]
    manifest.update(
        {
            "model": f"picodet-{args.variant}-416",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "paddledetection_commit": revision,
            "config_path": str(config),
            "config_sha256": _sha256(config),
            "command": command,
            "execute_requested": args.execute,
        }
    )
    _write_json(run_root / "training-manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    log_path = run_root / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=checkout,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "exit_code": completed.returncode,
        "log_path": str(log_path),
        "formal_truth_sha256_after": _sha256(truth),
    }
    _write_json(run_root / "training-result.json", result)
    if result["formal_truth_sha256_after"] != EXPECTED_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
