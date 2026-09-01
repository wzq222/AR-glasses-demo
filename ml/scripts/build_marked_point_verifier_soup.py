"""Build a single MobileNet verifier by equal-weight state-dict averaging."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.verifier_model_soup import average_state_dicts, shared_verifier_contract


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{value}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "runs/marked-point-verifier-e4/multiseed/seed-20260828/best.pt",
            "runs/marked-point-verifier-e4/multiseed/seed-20260829/best.pt",
            "runs/marked-point-verifier-e4/multiseed/seed-20260830/best.pt",
        ],
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output",
        default="runs/marked-point-verifier-e4/multiseed/model-soup-final/best.pt",
    )
    args = parser.parse_args()

    import torch

    root = asset_root().resolve()
    input_paths = [_below(root, value) for value in args.inputs]
    formal_truth = _below(root, args.formal_truth)
    output = _below(root, args.output)
    if len(input_paths) < 2:
        raise ValueError("SOUP_REQUIRES_MULTIPLE_CHECKPOINTS")
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("SOUP_DUPLICATE_CHECKPOINT")
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if output.exists():
        raise FileExistsError(f"SOUP_OUTPUT_EXISTS:{output}")
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True) for path in input_paths
    ]
    architecture, classes, dataset_sha256, input_size = shared_verifier_contract(
        checkpoints
    )
    if architecture != "mobilenet_v3_small" or "marked_point" not in classes:
        raise RuntimeError("SOUP_CHECKPOINT_ARCHITECTURE_INVALID")
    averaged = average_state_dicts(
        [checkpoint["state_dict"] for checkpoint in checkpoints]
    )
    payload = {
        "schema_version": "marked-point-verifier-model-soup-v1",
        "architecture": architecture,
        "classes": list(classes),
        "state_dict": averaged,
        "epoch": 0,
        "dataset_sha256": dataset_sha256,
        "input_size": input_size,
        "input_checkpoint_sha256": [_sha256(path) for path in input_paths],
        "formal_truth_sha256": _sha256(formal_truth),
        "sealed_test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        {
            "output": str(output),
            "sha256": _sha256(output),
            "inputs": payload["input_checkpoint_sha256"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
