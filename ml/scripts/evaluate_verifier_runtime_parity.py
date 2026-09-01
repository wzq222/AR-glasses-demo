"""Compare ONNX and ncnn verifier scores on real detector proposal crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from crrc_vision.marked_point_verifier import suppress_overlapping_candidates
from crrc_vision.verifier_runtime_export import compare_verifier_scores


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _crop_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    center_x, center_y = x + box_width / 2, y + box_height / 2
    side = min(max(max(box_width, box_height) * 1.6, 64.0), width, height)
    side = int(round(side))
    left = min(max(int(round(center_x - side / 2)), 0), width - side)
    top = min(max(int(round(center_y - side / 2)), 0), height - side)
    return left, top, left + side, top + side


def _prepare(image: Image.Image, bbox: list[float], input_size: int) -> np.ndarray:
    resize_size = round(input_size * 256 / 224)
    crop = image.crop(_crop_box(bbox, image.width, image.height))
    crop = crop.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
    offset = (resize_size - input_size) // 2
    crop = crop.crop((offset, offset, offset + input_size, offset + input_size))
    values = np.asarray(crop, dtype=np.float32) / 255.0
    values = (values - MEAN) / STD
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def _marked_probability(logits: np.ndarray, marked_index: int) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values -= values.max()
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return float(probabilities[marked_index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scored-proposals", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--verifier-threshold", type=float, required=True)
    parser.add_argument("--proposal-threshold", type=float, required=True)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    import ncnn
    import onnxruntime as ort

    if _sha256(args.formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    document = json.loads(args.scored_proposals.read_text(encoding="utf-8"))
    rows = document["predictions"]
    image_by_id = {row["id"]: row for row in truth["images"]}
    images: dict[int, Image.Image] = {}
    for image_id, row in image_by_id.items():
        path = Path(row["file_name"])
        if not path.is_absolute():
            path = args.source_root / path
        with Image.open(path) as image:
            images[image_id] = image.convert("RGB").copy()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    network = ncnn.Net()
    network.opt.num_threads = 4
    if network.load_param(str(args.param)) != 0 or network.load_model(str(args.bin)) != 0:
        raise RuntimeError("NCNN_VERIFIER_LOAD_FAILED")
    onnx_scores: list[float] = []
    ncnn_scores: list[float] = []
    onnx_seconds = 0.0
    ncnn_seconds = 0.0
    for row in rows:
        values = _prepare(
            images[int(row["image_id"])],
            [float(value) for value in row["candidate_bbox"]],
            args.input_size,
        )
        started = time.perf_counter()
        onnx_logits = session.run(None, {"images": values})[0]
        onnx_seconds += time.perf_counter() - started
        extractor = network.create_extractor()
        extractor.input("in0", ncnn.Mat(values[0]).clone())
        started = time.perf_counter()
        return_code, output = extractor.extract("out0")
        ncnn_seconds += time.perf_counter() - started
        if return_code != 0:
            raise RuntimeError("NCNN_VERIFIER_INFERENCE_FAILED")
        onnx_scores.append(_marked_probability(onnx_logits, 0))
        ncnn_scores.append(_marked_probability(np.asarray(output), 0))

    parity = compare_verifier_scores(
        onnx_scores,
        ncnn_scores,
        threshold=args.verifier_threshold,
        maximum_drift=0.01,
    )
    selected_indices: dict[str, list[int]] = {}
    covered_truth: dict[str, list[int]] = {}
    for runtime, scores in (("onnx", onnx_scores), ("ncnn", ncnn_scores)):
        scored = [
            {**row, "verifier_score": score, "score": score}
            for row, score in zip(rows, scores, strict=True)
        ]
        selected = suppress_overlapping_candidates(
            scored,
            verifier_threshold=args.verifier_threshold,
            proposal_threshold=args.proposal_threshold,
            iou_threshold=args.nms_iou_threshold,
        )
        selected_indices[runtime] = sorted(int(row["prediction_index"]) for row in selected)
        covered_truth[runtime] = sorted(
            {int(truth_id) for row in selected for truth_id in row["truth_ids"]}
        )
    pipeline_match = selected_indices["onnx"] == selected_indices["ncnn"]
    coverage_match = covered_truth["onnx"] == covered_truth["ncnn"]
    report = {
        "schema_version": "marked-point-verifier-runtime-parity-v1",
        "candidate_count": len(rows),
        "input_size": args.input_size,
        "verifier_threshold": args.verifier_threshold,
        "proposal_threshold": args.proposal_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "score_parity": parity,
        "onnx_selected": len(selected_indices["onnx"]),
        "ncnn_selected": len(selected_indices["ncnn"]),
        "pipeline_selection_match": pipeline_match,
        "coverage_match": coverage_match,
        "covered_truth": {
            runtime: len(values) for runtime, values in covered_truth.items()
        },
        "onnx_inference_ms_per_crop": onnx_seconds * 1000 / len(rows),
        "ncnn_inference_ms_per_crop": ncnn_seconds * 1000 / len(rows),
        "onnx_sha256": _sha256(args.onnx),
        "ncnn_param_sha256": _sha256(args.param),
        "ncnn_bin_sha256": _sha256(args.bin),
        "formal_truth_sha256": _sha256(args.formal_truth),
        "passed": bool(parity["passed"] and pipeline_match and coverage_match),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("VERIFIER_RUNTIME_PARITY_FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
