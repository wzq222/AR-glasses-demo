"""Export a witness ROI checkpoint and verify PyTorch/ONNX numerical parity."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import assert_external_output  # noqa: E402
from crrc_vision.witness_roi_model import MobileNetV3SmallWitnessRoi  # noqa: E402


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _quality_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        ("synthetic_witness_mask_iou", 0.50, ">="),
        ("synthetic_keypoint_error_p95_px", 3.0, "<="),
        ("synthetic_angle_error_mean_degrees", 2.0, "<="),
        ("synthetic_angle_error_p95_degrees", 3.0, "<="),
    )
    for name, threshold, direction in checks:
        value = float(metrics[name])
        passed = value >= threshold if direction == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value:.6g} requires {direction}{threshold}")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    import torch

    checkpoint = args.checkpoint.resolve()
    metrics_path = (args.metrics or checkpoint.with_name("metrics.json")).resolve()
    output = assert_external_output(args.output.resolve(), REPOSITORY_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output}")
    training_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate_passed, gate_failures = _quality_gate(training_report["metrics"])
    if not gate_passed and not args.allow_failed_gate:
        raise RuntimeError(f"WITNESS_ROI_EXPORT_GATE_FAILED:{gate_failures}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MobileNetV3SmallWitnessRoi(pretrained=False).eval()
    model.load_state_dict(checkpoint_document["model_state_dict"], strict=True)
    onnx_path = output / "witness-roi-mobilenetv3-small.onnx"
    sample = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["images"],
        output_names=["segmentation_logits", "keypoint_heatmaps", "quality_logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "segmentation_logits": {0: "batch"},
            "keypoint_heatmaps": {0: "batch"},
            "quality_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(20260901)
    comparison_input = rng.normal(size=(1, 3, 320, 320)).astype(np.float32)
    with torch.inference_mode():
        torch_outputs = [
            value.cpu().numpy() for value in model(torch.from_numpy(comparison_input))
        ]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": comparison_input})
    differences = [
        float(np.max(np.abs(torch_value - onnx_value)))
        for torch_value, onnx_value in zip(torch_outputs, onnx_outputs, strict=True)
    ]
    parity_passed = max(differences) <= 1.0e-4
    report = {
        "schema_version": "witness-roi-onnx-export-v1",
        "onnx_path": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "opset": 17,
        "inputs": {"images": ["batch", 3, 320, 320]},
        "outputs": {
            "segmentation_logits": ["batch", 4, 320, 320],
            "keypoint_heatmaps": ["batch", 4, 320, 320],
            "quality_logits": ["batch", 4],
        },
        "maximum_absolute_differences": differences,
        "onnx_parity_passed": parity_passed,
        "experimental_quality_gate_passed": gate_passed,
        "experimental_quality_gate_failures": gate_failures,
        "android_packaging_allowed": gate_passed and parity_passed,
        "real_state_accuracy_validated": False,
    }
    _atomic_json(output / "export-report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if not parity_passed:
        raise RuntimeError("WITNESS_ROI_ONNX_PARITY_FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
