"""Write COCO predictions from full-image plus deterministic 2x2 YOLO inference."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import cv2
import numpy as np

from crrc_vision.detection_fusion import (
    merge_target_classes,
    nms_image_detections,
    select_detection_mode,
    to_coco_predictions,
)
from crrc_vision.reference_teacher import (
    validate_checkpoint_globals,
    validate_ultralytics_version,
)
from crrc_vision.tiles import build_tiles


def _rows(result: object) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32)[:, None]
    classes = boxes.cls.detach().cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate((classes, scores, xyxy), axis=1)


def _resolve_framework_globals(names: list[str]) -> list[object]:
    errors = validate_checkpoint_globals(names)
    if errors:
        raise RuntimeError(errors[0])
    resolved: list[object] = []
    for name in names:
        module_name, attribute = name.rsplit(".", 1)
        resolved.append(getattr(importlib.import_module(module_name), attribute))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--val-coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--mode", choices=("full", "sliced", "fused"), default="fused")
    parser.add_argument("--merge-target-categories", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    import torch
    import ultralytics
    from ultralytics import YOLO

    version_errors = validate_ultralytics_version(ultralytics.__version__)
    if version_errors:
        raise RuntimeError(version_errors[0])
    unsafe_names = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(str(args.weights))
    )
    allowed_globals = _resolve_framework_globals(unsafe_names)
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint = torch.load(
            str(args.weights), map_location="cpu", weights_only=True
        )
    checkpoint_model = checkpoint.get("model")
    if checkpoint_model is None or checkpoint.get("version") != "8.2.40":
        raise RuntimeError("INCOMPATIBLE_YOLO_CHECKPOINT")
    model = YOLO("yolov8s-p2.yaml")
    model.model = checkpoint_model.float().eval()
    model.ckpt = checkpoint
    model.ckpt_path = str(args.weights)
    model.task = "detect"
    model.model.args = checkpoint.get("train_args", {})
    document = json.loads(args.val_coco.read_text(encoding="utf-8"))
    predictions: list[dict[str, object]] = []
    for image in sorted(document["images"], key=lambda row: int(row["id"])):
        original = cv2.imread(str(Path(image["file_name"])), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(image["file_name"])
        height, width = original.shape[:2]
        tiles = build_tiles(width, height, overlap=0.12)
        inputs = [original] + [
            original[tile.y1 : tile.y2, tile.x1 : tile.x2] for tile in tiles
        ]
        results = model.predict(
            inputs,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=300,
            device=0,
            agnostic_nms=args.merge_target_categories,
            verbose=False,
        )
        full = _rows(results[0])
        sliced_rows: list[np.ndarray] = []
        for tile, result in zip(tiles, results[1:]):
            rows = _rows(result)
            if rows.size:
                rows[:, [2, 4]] += tile.x1
                rows[:, [3, 5]] += tile.y1
                sliced_rows.append(rows)
        sliced = (
            np.concatenate(sliced_rows, axis=0)
            if sliced_rows
            else np.empty((0, 6), dtype=np.float32)
        )
        if args.merge_target_categories:
            full = merge_target_classes(full)
            sliced = merge_target_classes(sliced)
        sliced = nms_image_detections(sliced, iou_threshold=args.iou)
        selected = select_detection_mode(
            full,
            sliced,
            mode=args.mode,
            iou_threshold=args.iou,
        )
        predictions.extend(
            to_coco_predictions(
                image_id=int(image["id"]),
                detections=selected,
                image_width=width,
                image_height=height,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"images": len(document["images"]), "predictions": len(predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
