"""Run a Git-external YOLO reference teacher with restricted checkpoint loading."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.reference_teacher import (
    TEACHER_CATEGORY_MAP,
    TeacherPrediction,
    build_run_manifest,
    ensure_complete_selection,
    validate_checkpoint_globals,
    validate_ultralytics_version,
    xyxy_to_xywh,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("output must stay below CRRC_VISION_DATA_ROOT")
    return path


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_framework_globals(names: list[str]) -> list[object]:
    errors = validate_checkpoint_globals(names)
    if errors:
        raise RuntimeError(errors[0])
    resolved = []
    for name in names:
        module_name, attribute = name.rsplit(".", 1)
        resolved.append(getattr(importlib.import_module(module_name), attribute))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", default="selections/selection-v2.json")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--output", default="runs/reference-teacher-v1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    root = asset_root()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    selection_path = _below(root, args.selection)
    truth_path = _below(root, args.truth)
    source_root = _below(root, args.source)
    output_root = _below(root, args.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for required in (selection_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw-predictions.json"
    manifest_path = output_root / "run-manifest.json"
    if raw_path.exists() or manifest_path.exists():
        raise FileExistsError(f"reference teacher output already exists: {output_root}")

    import torch
    import ultralytics
    from ultralytics import YOLO

    version_errors = validate_ultralytics_version(ultralytics.__version__)
    if version_errors:
        raise RuntimeError(
            f"{version_errors[0]}: expected 8.2.40, got {ultralytics.__version__}"
        )
    unsafe_names = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint))
    )
    allowed_globals = _resolve_framework_globals(unsafe_names)
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint_data = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True
        )

    model = checkpoint_data.get("model")
    if model is None or checkpoint_data.get("version") != "8.2.40":
        raise RuntimeError("INCOMPATIBLE_REFERENCE_CHECKPOINT")
    wrapper = YOLO("yolov8s.yaml")
    wrapper.model = model.float().eval()
    wrapper.ckpt = checkpoint_data
    wrapper.ckpt_path = str(checkpoint)
    wrapper.task = "detect"
    wrapper.model.args = checkpoint_data.get("train_args", {})

    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes.decode("utf-8"))
    items = selection.get("items", [])
    expected_paths = [str(item["relative_path"]) for item in items]
    if len(expected_paths) != 100:
        raise RuntimeError(f"EXPECTED_100_SELECTION_ITEMS: got {len(expected_paths)}")
    truth_sha256 = _sha256(truth_path)
    predictions: list[dict[str, object]] = []
    image_runs: list[dict[str, object]] = []

    for index, item in enumerate(items, start=1):
        relative_path = str(item["relative_path"])
        image_path = (source_root / relative_path).resolve()
        if source_root.resolve() not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(image_path)
        if str(args.device).isdigit() and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = wrapper.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        if str(args.device).isdigit() and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        height, width = (int(value) for value in result.orig_shape)
        count_before = len(predictions)
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            class_value = int(class_id)
            prediction = TeacherPrediction(
                relative_path=relative_path,
                teacher_class_id=class_value,
                teacher_class_name=str(model.names[class_value]),
                bbox=xyxy_to_xywh(tuple(float(value) for value in xyxy), width=width, height=height),
                score=float(score),
            )
            predictions.append(prediction.to_dict())
        image_runs.append(
            {
                "relative_path": relative_path,
                "scene_group": item["scene_group"],
                "split": item["split"],
                "width": width,
                "height": height,
                "predictions": len(predictions) - count_before,
                "wall_ms": round(wall_ms, 3),
                "speed_ms": {
                    key: round(float(value), 3) for key, value in result.speed.items()
                },
            }
        )
        print(
            f"processed {index}/100: {relative_path} "
            f"({image_runs[-1]['predictions']} proposals)"
        )

    coverage_errors = ensure_complete_selection(
        expected_paths, [str(row["relative_path"]) for row in image_runs]
    )
    if coverage_errors:
        raise RuntimeError(coverage_errors[0])
    truth_after = _sha256(truth_path)
    if truth_after != truth_sha256:
        raise RuntimeError("FORMAL_TRUTH_MUTATED")

    raw_payload = {
        "schema_version": "reference-teacher-v1",
        "checkpoint_version": checkpoint_data.get("version"),
        "checkpoint_date": checkpoint_data.get("date"),
        "model_names": {str(key): value for key, value in model.names.items()},
        "teacher_category_map": {
            str(key): value for key, value in TEACHER_CATEGORY_MAP.items()
        },
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "images": image_runs,
        "predictions": predictions,
    }
    run_manifest = build_run_manifest(
        checkpoint_sha256=_sha256(checkpoint),
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest().upper(),
        truth_sha256=truth_sha256,
        images=len(image_runs),
        predictions=len(predictions),
    )
    run_manifest.update(
        {
            "checkpoint_path": str(checkpoint),
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "checkpoint_globals": unsafe_names,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
        }
    )
    _atomic_json(raw_path, raw_payload)
    _atomic_json(manifest_path, run_manifest)
    print(
        json.dumps(
            {"images": len(image_runs), "predictions": len(predictions)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
