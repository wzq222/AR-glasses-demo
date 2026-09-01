"""Export the marked-point verifier and prove PyTorch/ONNX numerical parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from crrc_vision.assets import asset_root
from crrc_vision.verifier_runtime_export import verifier_export_contract


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


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    import torch
    from torchvision.models import mobilenet_v3_small

    root = asset_root().resolve()
    checkpoint_path = _below(root, args.checkpoint)
    formal_truth = _below(root, args.formal_truth)
    output = _below(root, args.output)
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    input_size, classes = verifier_export_contract(checkpoint)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "marked-point-verifier-mobilenetv3-small.onnx"
    sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(20260901)
    comparison_input = rng.normal(
        size=(3, 3, input_size, input_size)
    ).astype(np.float32)
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(comparison_input)).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"images": comparison_input})[0]
    maximum_absolute_difference = float(
        np.max(np.abs(torch_output - onnx_output))
    )
    parity_passed = maximum_absolute_difference <= 1.0e-4
    report = {
        "schema_version": "marked-point-verifier-onnx-export-v1",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "onnx_sha256": _sha256(onnx_path),
        "formal_truth_sha256": _sha256(formal_truth),
        "input_size": input_size,
        "classes": list(classes),
        "input_name": "images",
        "output_name": "logits",
        "input_shape": ["batch", 3, input_size, input_size],
        "output_shape": ["batch", len(classes)],
        "opset": 17,
        "maximum_absolute_difference": maximum_absolute_difference,
        "onnx_parity_passed": parity_passed,
        "sealed_test_opened": False,
    }
    _atomic_json(output / "export-report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if not parity_passed:
        raise RuntimeError("VERIFIER_ONNX_PARITY_FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
