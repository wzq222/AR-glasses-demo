import hashlib
import json
from pathlib import Path

import pytest

from crrc_vision.picodet import (
    PINNED_PADDLEDETECTION_REVISION,
    build_train_command,
    prepare_picodet_dataset,
    validate_silver_dataset,
    write_picodet_config,
)


def _document(train: int = 64, val: int = 16) -> dict[str, object]:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, split in enumerate(["train"] * train + ["val"] * val, start=1):
        payload = f"image-{image_id}".encode()
        images.append(
            {
                "id": image_id,
                "file_name": f"image-{image_id}.jpg",
                "relative_path": f"image-{image_id}.jpg",
                "width": 100,
                "height": 80,
                "scene_group": f"scene-{image_id}",
                "split": split,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "synthetic": False,
                "image_review_status": "complete",
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1 if image_id % 2 else 2,
                "bbox": [5, 6, 20, 22],
                "area": 440,
                "iscrowd": 0,
                "review_status": "accept",
            }
        )
        annotation_id += 1
    return {
        "info": {
            "schema_version": "ai-silver-truth-v1",
            "truth_tier": "silver",
            "production_metrics_allowed": False,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}],
    }


def _write_images(document: dict[str, object], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for image in document["images"]:
        (root / image["relative_path"]).write_bytes(f"image-{image['id']}".encode())


def _write_jpeg_images(document: dict[str, object], root: Path) -> None:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    for image in document["images"]:
        path = root / image["relative_path"]
        Image.new("RGB", (image["width"], image["height"]), (image["id"] % 255, 20, 30)).save(
            path,
            format="JPEG",
            quality=95,
        )
        image["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_accepts_exact_64_train_16_val_real_scenes(tmp_path: Path) -> None:
    document = _document()
    _write_images(document, tmp_path)

    result = validate_silver_dataset(document, tmp_path)

    assert result.can_train is True
    assert result.train_groups == 64
    assert result.val_groups == 16
    assert result.annotations == 80
    assert result.reasons == ()


def test_gate_refuses_too_few_groups_and_scene_leakage(tmp_path: Path) -> None:
    document = _document(train=63, val=15)
    document["images"][-1]["scene_group"] = document["images"][0]["scene_group"]
    _write_images(document, tmp_path)

    result = validate_silver_dataset(document, tmp_path)

    assert result.can_train is False
    assert "TRAIN_GROUPS_BELOW_64" in result.reasons
    assert "VAL_GROUPS_BELOW_16" in result.reasons
    assert "SCENE_SPLIT_LEAKAGE" in result.reasons


def test_gate_refuses_hash_mismatch_and_invalid_category(tmp_path: Path) -> None:
    document = _document()
    document["annotations"][0]["category_id"] = 99
    _write_images(document, tmp_path)
    (tmp_path / document["images"][1]["relative_path"]).write_bytes(b"changed")

    result = validate_silver_dataset(document, tmp_path)

    assert "INVALID_CATEGORY" in result.reasons
    assert "IMAGE_HASH_MISMATCH" in result.reasons


def test_prepare_writes_deterministic_split_coco_and_manifest(tmp_path: Path) -> None:
    document = _document()
    source_root = tmp_path / "source"
    _write_images(document, source_root)
    document_path = tmp_path / "instances.silver.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    truth_path = tmp_path / "formal.json"
    truth_path.write_text("{}", encoding="utf-8")
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest().upper()

    first = prepare_picodet_dataset(
        document_path=document_path,
        source_root=source_root,
        run_root=tmp_path / "run-a",
        formal_truth_path=truth_path,
        expected_truth_sha256=truth_sha,
    )
    second = prepare_picodet_dataset(
        document_path=document_path,
        source_root=source_root,
        run_root=tmp_path / "run-b",
        formal_truth_path=truth_path,
        expected_truth_sha256=truth_sha,
    )

    assert first["status"] == "ready"
    assert first["train_groups"] == 64
    assert first["val_groups"] == 16
    assert (tmp_path / "run-a/dataset/annotations/train.json").read_bytes() == (
        tmp_path / "run-b/dataset/annotations/train.json"
    ).read_bytes()
    train = json.loads((tmp_path / "run-a/dataset/annotations/train.json").read_text())
    val = json.loads((tmp_path / "run-a/dataset/annotations/val.json").read_text())
    assert len(train["images"]) == 64
    assert len(val["images"]) == 16
    assert all(Path(image["file_name"]).is_absolute() for image in train["images"])


def test_prepare_can_emit_ascii_runtime_paths_for_windows_gbk_loader(
    tmp_path: Path,
) -> None:
    document = _document()
    source_root = tmp_path / "中文源图"
    _write_images(document, source_root)
    document_path = tmp_path / "instances.silver.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    truth_path = tmp_path / "formal.json"
    truth_path.write_text("{}", encoding="utf-8")
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest().upper()

    prepare_picodet_dataset(
        document_path=document_path,
        source_root=source_root,
        runtime_source_root=Path("E:/crrc_vision_data/source/images"),
        run_root=tmp_path / "run",
        formal_truth_path=truth_path,
        expected_truth_sha256=truth_sha,
    )

    train_bytes = (tmp_path / "run/dataset/annotations/train.json").read_bytes()
    assert all(byte < 128 for byte in train_bytes)
    train = json.loads(train_bytes)
    assert train["images"][0]["file_name"].startswith(
        "E:/crrc_vision_data/source/images/"
    )


def test_prepare_tiled_train_keeps_full_val_and_covers_every_training_box(
    tmp_path: Path,
) -> None:
    document = _document()
    source_root = tmp_path / "source"
    _write_jpeg_images(document, source_root)
    document_path = tmp_path / "instances.silver.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    truth_path = tmp_path / "formal.json"
    truth_path.write_text("{}", encoding="utf-8")
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest().upper()

    manifest = prepare_picodet_dataset(
        document_path=document_path,
        source_root=source_root,
        runtime_source_root=source_root,
        run_root=tmp_path / "run",
        formal_truth_path=truth_path,
        expected_truth_sha256=truth_sha,
        train_tiles=True,
        tile_overlap=0.12,
    )

    train = json.loads((tmp_path / "run/dataset/annotations/train.json").read_text())
    val = json.loads((tmp_path / "run/dataset/annotations/val.json").read_text())
    source_ids = {image["id"] for image in document["images"] if image["split"] == "train"}
    tiled_sources = {
        image["source_image_id"]
        for image in train["images"]
        if image.get("view") == "tile"
    }

    assert len(train["images"]) == 64 * 5
    assert len(val["images"]) == 16
    assert all(image.get("view") != "tile" for image in val["images"])
    assert tiled_sources == source_ids
    assert manifest["train_source_images"] == 64
    assert manifest["train_effective_images"] == 64 * 5
    assert manifest["tile_overlap"] == 0.12
    assert all(
        any(
            annotation.get("source_annotation_id") == source_annotation["id"]
            for annotation in train["annotations"]
        )
        for source_annotation in document["annotations"]
        if source_annotation["image_id"] in source_ids
    )
    assert all(
        (tmp_path / "run/dataset/images/train" / Path(image["file_name"]).name).is_file()
        for image in train["images"]
        if image.get("view") == "tile"
    )
    assert all(
        image["sha256"]
        == hashlib.sha256(Path(image["file_name"]).read_bytes()).hexdigest().upper()
        for image in train["images"]
        if image.get("view") == "tile"
    )


def test_prepare_refuses_changed_formal_truth(tmp_path: Path) -> None:
    document = _document()
    source_root = tmp_path / "source"
    _write_images(document, source_root)
    document_path = tmp_path / "instances.silver.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    truth_path = tmp_path / "formal.json"
    truth_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="FORMAL_TRUTH_HASH_MISMATCH"):
        prepare_picodet_dataset(
            document_path=document_path,
            source_root=source_root,
            run_root=tmp_path / "run",
            formal_truth_path=truth_path,
            expected_truth_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("variant", "config_name", "base_lr"),
    [("s", "picodet_s_416_coco_lcnet.yml", "0.04"), ("m", "picodet_m_416_coco_lcnet.yml", "0.04")],
)
def test_train_command_pins_416_contract(
    tmp_path: Path, variant: str, config_name: str, base_lr: str
) -> None:
    checkout = tmp_path / "PaddleDetection"
    (checkout / "tools").mkdir(parents=True)
    (checkout / "tools/train.py").write_text("", encoding="utf-8")
    config = checkout / "configs/picodet" / config_name
    config.parent.mkdir(parents=True)
    config.write_text("", encoding="utf-8")

    command = build_train_command(
        python=Path("python.exe"),
        paddledetection_root=checkout,
        variant=variant,
        run_root=tmp_path / f"run-{variant}",
        epochs=80,
        batch_size=8,
    )

    rendered = " ".join(str(item) for item in command)
    assert config_name in rendered
    assert "epoch=80" in command
    assert "TrainReader.batch_size=8" in command
    assert f"LearningRate.base_lr={base_lr}" in command
    assert "--eval" in command
    assert PINNED_PADDLEDETECTION_REVISION == "v2.9.0"


def test_train_command_refuses_missing_checkout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_train_command(
            python=Path("python.exe"),
            paddledetection_root=tmp_path / "missing",
            variant="s",
            run_root=tmp_path / "run",
            epochs=1,
            batch_size=1,
        )


def test_runtime_config_overrides_schedule_and_dataset_without_touching_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "PaddleDetection"
    base = checkout / "configs/picodet/picodet_s_416_coco_lcnet.yml"
    base.parent.mkdir(parents=True)
    base.write_text("epoch: 300\n", encoding="utf-8")

    output = write_picodet_config(
        paddledetection_root=checkout,
        variant="s",
        run_root=tmp_path / "run-s",
        epochs=80,
        batch_size=8,
    )

    text = output.read_text(encoding="utf-8")
    assert base.resolve().as_posix() in text
    assert "epoch: 80" in text
    assert "max_epochs: 80" in text
    assert "static_assigner_epoch: 26" in text
    assert "batch_size: 8" in text
    assert "num_classes: 2" in text
    assert "annotations/train.json" in text
    assert "annotations/val.json" in text
    assert text.count("allow_empty: true") == 2


def test_runtime_config_can_be_ascii_only_when_assets_live_below_chinese_path(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "中文" / "PaddleDetection"
    base = checkout / "configs/picodet/picodet_s_416_coco_lcnet.yml"
    base.parent.mkdir(parents=True)
    base.write_text("epoch: 300\n", encoding="utf-8")

    output = write_picodet_config(
        paddledetection_root=checkout,
        runtime_paddledetection_root=Path("E:/crrc_vision_data/runtimes/PaddleDetection-v2.9.0"),
        variant="s",
        run_root=tmp_path / "中文运行",
        runtime_run_root=Path("E:/crrc_vision_data/runs/picodet-s-v1"),
        epochs=1,
        batch_size=4,
    )

    assert all(byte < 128 for byte in output.read_bytes())


def test_runtime_config_scales_learning_rate_with_batch_size(tmp_path: Path) -> None:
    checkout = tmp_path / "PaddleDetection"
    base = checkout / "configs/picodet/picodet_s_416_coco_lcnet.yml"
    base.parent.mkdir(parents=True)
    base.write_text("epoch: 300\n", encoding="utf-8")

    output = write_picodet_config(
        paddledetection_root=checkout,
        variant="s",
        run_root=tmp_path / "run-s",
        epochs=80,
        batch_size=16,
    )

    assert "base_lr: 0.08" in output.read_text(encoding="utf-8")
