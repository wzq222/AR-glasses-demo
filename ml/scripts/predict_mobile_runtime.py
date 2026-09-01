"""Run a frozen validation subset through ONNX, ncnn, or MNN on desktop."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from crrc_vision.assets import asset_root
from crrc_vision.mobile_benchmark import sha256_file
from crrc_vision.mobile_runtime_parity import (
    make_mnn_infer,
    make_ncnn_infer,
    make_onnx_infer,
    predict_image,
    resolve_below_preserving_alias,
)


EXPECTED_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"IMAGE_DECODE_FAILED:{path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("onnx", "ncnn", "mnn"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--param")
    parser.add_argument("--bin")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    args = parser.parse_args()

    verified_root = asset_root().resolve()
    root = Path(os.environ["CRRC_VISION_DATA_ROOT"]).expanduser()
    if root.resolve() != verified_root:
        raise RuntimeError("CRRC_VISION_DATA_ROOT_CHANGED")
    dataset_path = resolve_below_preserving_alias(root, args.dataset)
    image_root = resolve_below_preserving_alias(root, args.image_root)
    truth_path = resolve_below_preserving_alias(root, args.truth)
    run = resolve_below_preserving_alias(root, args.run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("PREDICTION_RUN_NOT_EMPTY")
    run.mkdir(parents=True, exist_ok=True)
    truth_before = sha256_file(truth_path)
    if truth_before != EXPECTED_TRUTH_SHA256:
        raise ValueError("FORMAL_TRUTH_HASH_MISMATCH")

    artifact_hashes: dict[str, str] = {}
    if args.runtime == "ncnn":
        if not args.param or not args.bin:
            raise ValueError("NCNN_PARAM_AND_BIN_REQUIRED")
        param_path = resolve_below_preserving_alias(root, args.param)
        bin_path = resolve_below_preserving_alias(root, args.bin)
        artifact_hashes = {
            "param": sha256_file(param_path),
            "bin": sha256_file(bin_path),
        }
        infer = make_ncnn_infer(param_path, bin_path, threads=args.threads)
        runtime_version = importlib.metadata.version("ncnn")
    else:
        if not args.model:
            raise ValueError("MODEL_REQUIRED")
        model_path = resolve_below_preserving_alias(root, args.model)
        artifact_hashes = {"model": sha256_file(model_path)}
        if args.runtime == "onnx":
            infer = make_onnx_infer(model_path, threads=args.threads)
            runtime_version = importlib.metadata.version("onnxruntime")
        else:
            infer = make_mnn_infer(model_path, threads=args.threads)
            runtime_version = importlib.metadata.version("MNN")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    images = list(dataset["images"])
    images.sort(key=lambda item: (str(item.get("scene_group", "")), int(item["id"])))
    if args.max_images > 0:
        images = images[: args.max_images]
    predictions: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    for item in images:
        image_path = image_root / str(item["file_name"])
        if sha256_file(image_path) != str(item["sha256"]).upper():
            raise ValueError(f"IMAGE_HASH_MISMATCH:{item['id']}")
        image = _read_rgb(image_path)
        started = time.perf_counter()
        detections = predict_image(infer, image, image_id=int(item["id"]))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        predictions.extend(detections)
        timings.append(
            {
                "image_id": int(item["id"]),
                "elapsed_ms": elapsed_ms,
                "detections": len(detections),
            }
        )

    predictions_path = run / "predictions.json"
    _atomic_json(predictions_path, predictions)
    truth_after = sha256_file(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    report = {
        "schema_version": "mobile-runtime-prediction-v1",
        "status": "predicted",
        "runtime": args.runtime,
        "runtime_version": runtime_version,
        "threads": args.threads,
        "artifact_hashes": artifact_hashes,
        "dataset_sha256": sha256_file(dataset_path),
        "image_count": len(images),
        "detection_count": len(predictions),
        "timings": timings,
        "predictions_sha256": sha256_file(predictions_path),
        "formal_truth_sha256_before": truth_before,
        "formal_truth_sha256_after": truth_after,
    }
    _atomic_json(run / "prediction-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
