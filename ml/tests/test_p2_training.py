from pathlib import Path

import pytest

from crrc_vision.p2_training import (
    P2_SEEDS,
    build_resume_kwargs,
    build_train_kwargs,
    checkpoint_model,
    validate_training_inputs,
)


def test_three_seeds_and_conservative_augmentation_contract(tmp_path: Path) -> None:
    assert P2_SEEDS == (20260828, 20260829, 20260830)
    kwargs = build_train_kwargs(
        seed=P2_SEEDS[0],
        dataset_yaml=tmp_path / "dataset.yaml",
        run_root=tmp_path / "run",
        epochs=100,
        batch_size=4,
    )
    assert kwargs["imgsz"] == 640
    assert kwargs["patience"] == 15
    assert kwargs["degrees"] == 5.0
    assert kwargs["scale"] == 0.2
    assert kwargs["perspective"] == 0.0
    assert kwargs["mosaic"] == 0.0
    assert kwargs["copy_paste"] == 0.0
    assert kwargs["seed"] == 20260828
    assert "sealed" not in str(kwargs).lower()


def test_training_inputs_must_be_inside_assets_and_never_include_sealed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    train = root / "train.json"
    val = root / "val.json"
    weights = root / "best.pt"
    for path in (train, val, weights):
        path.write_bytes(b"x")
    output = root / "runs/new"

    validate_training_inputs(
        asset_root=root,
        train_coco=train,
        val_coco=val,
        pretrained=weights,
        output_root=output,
        ultralytics_version="8.2.40",
    )

    sealed = root / "instances.sealed-test.json"
    sealed.write_bytes(b"x")
    with pytest.raises(ValueError, match="SEALED_TEST_PATH_FORBIDDEN"):
        validate_training_inputs(
            asset_root=root,
            train_coco=train,
            val_coco=sealed,
            pretrained=weights,
            output_root=output,
            ultralytics_version="8.2.40",
        )
    with pytest.raises(ValueError, match="ULTRALYTICS_VERSION_MISMATCH"):
        validate_training_inputs(
            asset_root=root,
            train_coco=train,
            val_coco=val,
            pretrained=weights,
            output_root=output,
            ultralytics_version="8.3.0",
        )


def test_resume_allows_only_an_existing_nonempty_output_inside_assets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    train = root / "train.json"
    val = root / "val.json"
    weights = root / "best.pt"
    for path in (train, val, weights):
        path.write_bytes(b"x")
    output = root / "runs/existing"
    output.mkdir(parents=True)
    (output / "checkpoint.txt").write_text("ready", encoding="utf-8")

    validate_training_inputs(
        asset_root=root,
        train_coco=train,
        val_coco=val,
        pretrained=weights,
        output_root=output,
        ultralytics_version="8.2.40",
        allow_existing_output=True,
    )

    with pytest.raises(FileNotFoundError, match="RESUME_OUTPUT_MISSING"):
        validate_training_inputs(
            asset_root=root,
            train_coco=train,
            val_coco=val,
            pretrained=weights,
            output_root=root / "runs/missing",
            ultralytics_version="8.2.40",
            allow_existing_output=True,
        )


def test_checkpoint_model_accepts_training_ema_but_prefers_export_model() -> None:
    export_model = object()
    ema_model = object()

    assert checkpoint_model({"model": export_model, "ema": ema_model}) is export_model
    assert checkpoint_model({"model": None, "ema": ema_model}) is ema_model
    with pytest.raises(ValueError, match="CHECKPOINT_MODEL_MISSING"):
        checkpoint_model({"model": None, "ema": None})


def test_resume_kwargs_can_lower_batch_without_changing_the_run_contract() -> None:
    assert build_resume_kwargs(batch_size=8) == {
        "resume": True,
        "batch": 8,
        "workers": 0,
    }
    with pytest.raises(ValueError, match="INVALID_RESUME_BATCH"):
        build_resume_kwargs(batch_size=0)
