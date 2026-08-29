"""Prepare and optionally execute the guarded three-seed P2-S training run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.p2_training import (
    P2_SEEDS,
    build_resume_kwargs,
    build_train_kwargs,
    checkpoint_model,
    is_recoverable_finalization_failure,
    p2_model_yaml,
    validate_pretraining_mode,
    validate_training_checkpoint,
    validate_training_inputs,
    validate_synthetic_ablation_mode,
)
from crrc_vision.reference_teacher import (
    validate_checkpoint_globals,
    validate_ultralytics_version,
)
from crrc_vision.yolo_p2 import prepare_yolo_dataset


FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _safe_model(
    weights: Path,
    *,
    model_yaml: str,
    transfer_pretrained: bool,
    expected_sha256: str | None = None,
):
    import torch
    from ultralytics import YOLO

    unsafe_names = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(str(weights))
    )
    errors = validate_checkpoint_globals(unsafe_names)
    if errors:
        raise RuntimeError(errors[0])
    allowed = []
    for name in unsafe_names:
        module_name, attribute = name.rsplit(".", 1)
        allowed.append(getattr(importlib.import_module(module_name), attribute))
    # Ultralytics' AMP self-check loads its bundled yolov8n checkpoint after this
    # function returns. Keep the same narrowly validated framework allowlist active.
    torch.serialization.add_safe_globals(allowed)
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    try:
        validate_training_checkpoint(
            checkpoint_version=str(checkpoint.get("version", "")),
            transfer_pretrained=transfer_pretrained,
            actual_sha256=_sha256(weights),
            expected_sha256=expected_sha256,
        )
    except ValueError as exc:
        raise RuntimeError("INCOMPATIBLE_YOLO_CHECKPOINT") from exc
    try:
        selected_model = checkpoint_model(checkpoint)
    except ValueError as exc:
        raise RuntimeError("INCOMPATIBLE_YOLO_CHECKPOINT") from exc
    model = YOLO(model_yaml)
    if transfer_pretrained:
        model.model.load(selected_model.float())
    else:
        model.model = selected_model.float()
    model.ckpt = checkpoint
    model.ckpt_path = str(weights)
    model.task = "detect"
    model.model.args = checkpoint.get("train_args", {})
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-coco", default="annotations/high-accuracy-v2/instances.train.json"
    )
    parser.add_argument(
        "--val-coco", default="annotations/high-accuracy-v2/instances.val.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument(
        "--pretrained",
        default="runs/yolov8s-p2-v3-640-direct/train/weights/best.pt",
    )
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/high-accuracy-p2-s-640")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=P2_SEEDS)
    parser.add_argument("--variant", choices=("s", "m"), default="s")
    parser.add_argument("--transfer-pretrained", action="store_true")
    parser.add_argument("--expected-pretrained-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum-synthetic-fraction", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.resume and (args.seed is None or not args.execute):
        raise ValueError("RESUME_REQUIRES_ONE_SEED_AND_EXECUTE")
    if not args.resume:
        validate_pretraining_mode(
            variant=args.variant, transfer_pretrained=args.transfer_pretrained
        )
        if args.transfer_pretrained and args.expected_pretrained_sha256 is None:
            raise ValueError("TRANSFER_REQUIRES_EXPECTED_HASH")
    model_yaml = p2_model_yaml(args.variant)

    root = asset_root().resolve()
    configured_root = Path(os.environ["CRRC_VISION_DATA_ROOT"]).absolute()
    train_coco = (root / args.train_coco).resolve()
    val_coco = (root / args.val_coco).resolve()
    source_root = (root / args.source).resolve()
    pretrained = (root / args.pretrained).resolve()
    truth = (root / args.truth).resolve()
    output_root = (root / args.output).resolve()
    import torch
    import ultralytics

    version_errors = validate_ultralytics_version(ultralytics.__version__)
    if version_errors:
        raise RuntimeError(version_errors[0])
    validate_training_inputs(
        asset_root=root,
        train_coco=train_coco,
        val_coco=val_coco,
        pretrained=pretrained,
        output_root=output_root,
        ultralytics_version=ultralytics.__version__,
        allow_existing_output=args.resume,
    )
    if not source_root.is_dir() or not truth.is_file():
        raise FileNotFoundError(source_root if not source_root.is_dir() else truth)
    if _sha256(truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    train_document = json.loads(train_coco.read_text(encoding="utf-8"))
    synthetic_policy = validate_synthetic_ablation_mode(
        train_document,
        maximum_synthetic_fraction=args.maximum_synthetic_fraction,
        batch_size=args.batch_size,
    )
    trainer_class = None
    if synthetic_policy["synthetic_images"]:
        from crrc_vision.ultralytics_ablation import make_synthetic_cap_trainer

        trainer_class = make_synthetic_cap_trainer(
            maximum_synthetic_fraction=args.maximum_synthetic_fraction,
            seed=args.seed or P2_SEEDS[0],
        )
    if args.resume:
        seed_root = output_root / f"seed-{args.seed}"
        manifest_path = seed_root / "training-manifest.json"
        last = seed_root / "train" / "weights" / "last.pt"
        if not manifest_path.is_file() or not last.is_file():
            raise FileNotFoundError(
                manifest_path if not manifest_path.is_file() else last
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {"status": "resuming", "resume_checkpoint_sha256": _sha256(last)}
        )
        _atomic_json(manifest_path, manifest)
        model = _safe_model(
            last, model_yaml=model_yaml, transfer_pretrained=False
        )
        resume_kwargs = build_resume_kwargs(batch_size=args.batch_size)
        if trainer_class is None:
            model.train(**resume_kwargs)
        else:
            model.train(trainer=trainer_class, **resume_kwargs)
        best = seed_root / "train" / "weights" / "best.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"TRAINING_CHECKPOINT_MISSING:{args.seed}")
        manifest.update(
            {
                "status": "complete",
                "resumed": True,
                "best_sha256": _sha256(best),
                "last_sha256": _sha256(last),
            }
        )
        _atomic_json(manifest_path, manifest)
        print(json.dumps({"seed": args.seed, "execute": True, "resumed": True}))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = output_root / "dataset"
    runtime_dataset_root = configured_root / args.output / "dataset"
    dataset_counts = prepare_yolo_dataset(
        train_coco=train_coco,
        val_coco=val_coco,
        output_root=dataset_root,
        runtime_output_root=runtime_dataset_root,
        source_root=source_root,
        train_tiles=True,
        train_tile_views=1,
        merge_target_categories=True,
    )
    common = {
        "schema_version": "high-accuracy-p2-training-v1",
        "code_commit": _git_commit(),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "dataset_counts": dataset_counts,
        "train_sha256": _sha256(train_coco),
        "val_sha256": _sha256(val_coco),
        "pretrained_sha256": _sha256(pretrained),
        "formal_truth_sha256": _sha256(truth),
        "license_status": "AGPL-3.0; commercial deployment unresolved",
        "sealed_test_visible": False,
        "model_variant": args.variant,
        "model_yaml": model_yaml,
        "transfer_pretrained": args.transfer_pretrained,
        "expected_pretrained_sha256": args.expected_pretrained_sha256,
        "synthetic_batch_policy": {
            **synthetic_policy,
            "maximum_synthetic_fraction": args.maximum_synthetic_fraction,
        },
    }
    selected_seeds = (args.seed,) if args.seed is not None else P2_SEEDS
    for seed in selected_seeds:
        seed_root = output_root / f"seed-{seed}"
        kwargs = build_train_kwargs(
            seed=seed,
            dataset_yaml=dataset_root / "dataset.yaml",
            run_root=seed_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        manifest = {**common, "seed": seed, "train_kwargs": kwargs, "status": "ready"}
        manifest_path = seed_root / "training-manifest.json"
        _atomic_json(manifest_path, manifest)
        if not args.execute:
            continue
        model = _safe_model(
            pretrained,
            model_yaml=model_yaml,
            transfer_pretrained=args.transfer_pretrained,
            expected_sha256=args.expected_pretrained_sha256,
        )
        best = seed_root / "train" / "weights" / "best.pt"
        last = seed_root / "train" / "weights" / "last.pt"
        results_csv = seed_root / "train" / "results.csv"
        finalization_error = None
        try:
            if trainer_class is None:
                model.train(**kwargs)
            else:
                model.train(trainer=trainer_class, **kwargs)
        except BaseException as exc:
            if not is_recoverable_finalization_failure(
                error=exc,
                results_csv=results_csv,
                best=best,
                last=last,
                expected_epochs=args.epochs,
            ):
                raise
            finalization_error = f"{type(exc).__name__}:{exc}"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"TRAINING_CHECKPOINT_MISSING:{seed}")
        manifest.update(
            {
                "status": (
                    "trained_with_finalization_error"
                    if finalization_error
                    else "complete"
                ),
                "best_sha256": _sha256(best),
                "last_sha256": _sha256(last),
                "finalization_error": finalization_error,
            }
        )
        _atomic_json(manifest_path, manifest)
    print(json.dumps({**common, "seeds": selected_seeds, "execute": args.execute}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
