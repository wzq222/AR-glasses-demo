import json
from pathlib import Path

import pytest

from crrc_vision.yolo_p2 import prepare_yolo_dataset


def _write_coco(
    path: Path,
    split: str,
    source: Path,
    *,
    image_id: int,
    scene_group: str | None = None,
) -> None:
    document = {
        "images": [
            {
                "id": image_id,
                "file_name": str(source.absolute()),
                "width": 100,
                "height": 80,
                "scene_group": scene_group or f"{split}-scene",
            }
        ],
        "annotations": [
            {"id": 9, "image_id": image_id, "category_id": 2, "bbox": [10, 20, 30, 40]}
        ],
        "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_prepare_yolo_dataset_materializes_images_labels_and_ascii_yaml(
    tmp_path: Path,
) -> None:
    train_source = tmp_path / "train-source.jpg"
    val_source = tmp_path / "val-source.jpg"
    train_source.write_bytes(b"train")
    val_source.write_bytes(b"val")
    train_coco = tmp_path / "train.json"
    val_coco = tmp_path / "val.json"
    _write_coco(train_coco, "train", train_source, image_id=7)
    _write_coco(val_coco, "val", val_source, image_id=8)

    report = prepare_yolo_dataset(
        train_coco=train_coco,
        val_coco=val_coco,
        output_root=tmp_path / "output",
        runtime_output_root=tmp_path / "output",
    )

    output = tmp_path / "output"
    assert (output / "images/train/000007.jpg").read_bytes() == b"train"
    assert (output / "images/val/000008.jpg").read_bytes() == b"val"
    assert (output / "labels/train/000007.txt").read_text() == "1 0.250000 0.500000 0.300000 0.500000\n"
    assert all(byte < 128 for byte in (output / "dataset.yaml").read_bytes())
    assert report == {"train_images": 1, "val_images": 1, "train_annotations": 1, "val_annotations": 1}


def test_prepare_yolo_dataset_can_merge_physical_target_categories(
    tmp_path: Path,
) -> None:
    train_source = tmp_path / "train-source.jpg"
    val_source = tmp_path / "val-source.jpg"
    train_source.write_bytes(b"train")
    val_source.write_bytes(b"val")
    train_coco = tmp_path / "train.json"
    val_coco = tmp_path / "val.json"
    _write_coco(train_coco, "train", train_source, image_id=7)
    _write_coco(val_coco, "val", val_source, image_id=8)

    prepare_yolo_dataset(
        train_coco=train_coco,
        val_coco=val_coco,
        output_root=tmp_path / "merged",
        merge_target_categories=True,
    )

    output = tmp_path / "merged"
    assert (output / "labels/train/000007.txt").read_text() == (
        "0 0.250000 0.500000 0.300000 0.500000\n"
    )
    assert (output / "dataset.yaml").read_text() == (
        f"path: {output.absolute().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: fastener_target\n"
    )


def test_prepare_yolo_dataset_rejects_cross_split_scene_leakage(tmp_path: Path) -> None:
    train_source = tmp_path / "train.jpg"
    val_source = tmp_path / "val.jpg"
    train_source.write_bytes(b"train")
    val_source.write_bytes(b"val")
    train_coco = tmp_path / "train.json"
    val_coco = tmp_path / "val.json"
    _write_coco(train_coco, "train", train_source, image_id=7, scene_group="same")
    _write_coco(val_coco, "val", val_source, image_id=8, scene_group="same")

    with pytest.raises(ValueError, match="YOLO_SPLIT_SCENE_LEAKAGE"):
        prepare_yolo_dataset(
            train_coco=train_coco,
            val_coco=val_coco,
            output_root=tmp_path / "output",
        )


def test_prepare_yolo_dataset_rejects_cross_split_duplicate_content(
    tmp_path: Path,
) -> None:
    train_source = tmp_path / "train.jpg"
    val_source = tmp_path / "val.jpg"
    train_source.write_bytes(b"same")
    val_source.write_bytes(b"same")
    train_coco = tmp_path / "train.json"
    val_coco = tmp_path / "val.json"
    _write_coco(train_coco, "train", train_source, image_id=7)
    _write_coco(val_coco, "val", val_source, image_id=8)

    with pytest.raises(ValueError, match="YOLO_SPLIT_HASH_LEAKAGE"):
        prepare_yolo_dataset(
            train_coco=train_coco,
            val_coco=val_coco,
            output_root=tmp_path / "output",
        )


def test_prepare_yolo_dataset_refuses_nonempty_output_and_wrong_runtime_root(
    tmp_path: Path,
) -> None:
    train_source = tmp_path / "train.jpg"
    val_source = tmp_path / "val.jpg"
    train_source.write_bytes(b"train")
    val_source.write_bytes(b"val")
    train_coco = tmp_path / "train.json"
    val_coco = tmp_path / "val.json"
    _write_coco(train_coco, "train", train_source, image_id=7)
    _write_coco(val_coco, "val", val_source, image_id=8)
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale")

    with pytest.raises(FileExistsError, match="YOLO_OUTPUT_NOT_EMPTY"):
        prepare_yolo_dataset(
            train_coco=train_coco,
            val_coco=val_coco,
            output_root=output,
        )

    output.joinpath("stale.txt").unlink()
    wrong_runtime = tmp_path / "wrong-runtime"
    wrong_runtime.mkdir()
    with pytest.raises(ValueError, match="YOLO_RUNTIME_ROOT_MISMATCH"):
        prepare_yolo_dataset(
            train_coco=train_coco,
            val_coco=val_coco,
            output_root=output,
            runtime_output_root=wrong_runtime,
        )
