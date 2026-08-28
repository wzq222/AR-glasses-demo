"""Evaluate full-image plus sliced PicoDet inference on original COCO scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from crrc_vision.detection_fusion import (
    merge_target_classes,
    merge_target_coco_document,
    runtime_path_text,
    select_detection_mode,
    to_coco_predictions,
)


def _split_result(result: dict[str, np.ndarray]) -> list[np.ndarray]:
    boxes = result["boxes"]
    rows: list[np.ndarray] = []
    offset = 0
    for count in result["boxes_num"]:
        size = int(count)
        rows.append(boxes[offset : offset + size].copy())
        offset += size
    if offset != len(boxes):
        raise ValueError("detector box count does not match flat output")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val-coco", type=Path, required=True)
    parser.add_argument("--deploy-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--slice-height", type=int, default=840)
    parser.add_argument("--slice-width", type=int, default=1120)
    parser.add_argument("--overlap", type=float, default=0.12)
    parser.add_argument("--fusion-iou", type=float, default=0.5)
    parser.add_argument("--mode", choices=("full", "sliced", "fused"), default="fused")
    parser.add_argument("--merge-target-categories", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, runtime_path_text(args.deploy_python))
    from infer import Detector

    document = json.loads(args.val_coco.read_text(encoding="utf-8"))
    images = sorted(document["images"], key=lambda image: int(image["id"]))
    image_paths = [str(Path(image["file_name"])) for image in images]
    detector = Detector(
        runtime_path_text(args.model_dir),
        device=args.device,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.device == "CPU",
        threshold=0.01,
    )
    full = _split_result(detector.predict_image(image_paths, visual=False))
    sliced = _split_result(
        detector.predict_image_slice(
            image_paths,
            slice_size=[args.slice_height, args.slice_width],
            overlap_ratio=[args.overlap, args.overlap],
            combine_method="nms",
            match_threshold=args.fusion_iou,
            match_metric="iou",
            visual=False,
        )
    )
    if len(full) != len(images) or len(sliced) != len(images):
        raise ValueError("detector did not return one result per validation image")

    predictions: list[dict[str, object]] = []
    per_image_counts: list[int] = []
    for image, full_rows, sliced_rows in zip(images, full, sliced):
        if args.merge_target_categories:
            full_rows = merge_target_classes(full_rows)
            sliced_rows = merge_target_classes(sliced_rows)
        fused = select_detection_mode(
            full_rows,
            sliced_rows,
            mode=args.mode,
            iou_threshold=args.fusion_iou,
        )
        converted = to_coco_predictions(
            image_id=int(image["id"]),
            detections=fused,
            image_width=int(image["width"]),
            image_height=int(image["height"]),
        )
        predictions.extend(converted)
        per_image_counts.append(sum(float(row["score"]) >= 0.25 for row in converted))

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.json"
    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    ground_truth_path = args.val_coco
    if args.merge_target_categories:
        merged_truth_path = args.output / "ground-truth-merged.json"
        merged_truth_path.write_text(
            json.dumps(
                merge_target_coco_document(document),
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        ground_truth_path = merged_truth_path
    ground_truth = COCO(str(ground_truth_path))
    detections = ground_truth.loadRes(str(prediction_path))
    evaluation = COCOeval(ground_truth, detections, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    stats = evaluation.stats.tolist()
    report = {
        "images": len(images),
        "predictions": len(predictions),
        "ap": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "ap_small": stats[3],
        "ap_medium": stats[4],
        "ap_large": stats[5],
        "ar100": stats[8],
        "detections_at_0_25_per_image": sum(per_image_counts) / len(images),
        "mode": args.mode,
        "category_mode": "physical_target" if args.merge_target_categories else "original",
        "slice_size": [args.slice_height, args.slice_width],
        "overlap": args.overlap,
        "fusion_iou": args.fusion_iou,
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
