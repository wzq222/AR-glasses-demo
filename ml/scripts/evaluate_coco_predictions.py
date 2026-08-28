"""Evaluate detector predictions with an optional physical-target category merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crrc_vision.detection_fusion import (
    merge_target_coco_document,
    merge_target_coco_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-target-categories", action="store_true")
    args = parser.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    if args.merge_target_categories:
        ground_truth = merge_target_coco_document(ground_truth)
        predictions = merge_target_coco_predictions(predictions)

    args.output.mkdir(parents=True, exist_ok=True)
    ground_truth_path = args.output / "ground-truth.json"
    predictions_path = args.output / "predictions.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    coco = COCO(str(ground_truth_path))
    results = coco.loadRes(str(predictions_path))
    evaluation = COCOeval(coco, results, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    stats = evaluation.stats.tolist()
    metrics = {
        "ap": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "ap_small": stats[3],
        "ap_medium": stats[4],
        "ap_large": stats[5],
        "ar100": stats[8],
        "category_mode": (
            "physical_target" if args.merge_target_categories else "original"
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
