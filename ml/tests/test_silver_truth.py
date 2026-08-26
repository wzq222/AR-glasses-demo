import json

from crrc_vision.silver_truth import evaluate_dataset, evaluate_image, export_silver


def sample_image(
    status: str,
    *,
    image_id: int = 1,
    split: str = "train",
    synthetic: bool = False,
) -> dict[str, object]:
    return {
        "id": image_id,
        "scene_group": f"g{image_id}",
        "split": split,
        "image_review_status": status,
        "synthetic": synthetic,
        "width": 100,
        "height": 100,
    }


def sample_box(status: str, *, image_id: int = 1) -> dict[str, object]:
    return {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [10, 10, 20, 20],
        "review_status": status,
        "second_pass": "accept",
    }


def sample_complete_document(
    train_groups: int = 64,
    val_groups: int = 16,
    synthetic_val: bool = False,
) -> dict[str, object]:
    images = []
    boxes = []
    for index in range(train_groups + val_groups):
        split = "train" if index < train_groups else "val"
        image = sample_image(
            "complete",
            image_id=index + 1,
            split=split,
            synthetic=(
                synthetic_val and split == "val" and index == train_groups
            ),
        )
        images.append(image)
        boxes.append(sample_box("accept", image_id=index + 1))
    return {
        "images": images,
        "annotations": boxes,
        "categories": [
            {"id": 1, "name": "fastener"},
            {"id": 2, "name": "pipe_joint"},
        ],
    }


def test_candidate_accept_does_not_complete_image() -> None:
    report = evaluate_image(sample_image(status="uncertain"), [sample_box("accept")])

    assert not report.complete


def test_uncertain_box_excludes_whole_image() -> None:
    report = evaluate_image(
        sample_image(status="complete"),
        [sample_box("uncertain")],
    )

    assert report.errors == ("UNRESOLVED_CANDIDATE",)


def test_dataset_requires_64_train_and_16_val_scene_groups() -> None:
    report = evaluate_dataset(sample_complete_document(train_groups=64, val_groups=15))

    assert "INSUFFICIENT_VAL_GROUPS" in report.errors


def test_synthetic_image_is_forbidden_in_val() -> None:
    report = evaluate_dataset(sample_complete_document(synthetic_val=True))

    assert "SYNTHETIC_VALIDATION_IMAGE" in report.errors


def test_synthetic_train_images_are_capped_at_twenty_percent() -> None:
    document = sample_complete_document()
    train_images = [
        image for image in document["images"] if image["split"] == "train"
    ]
    for image in train_images[:13]:
        image["synthetic"] = True

    report = evaluate_dataset(document)

    assert "SYNTHETIC_TRAIN_RATIO_EXCEEDED" in report.errors


def test_export_writes_refusal_without_silver_coco(tmp_path) -> None:
    document = sample_complete_document(train_groups=20, val_groups=5)

    code = export_silver(document, tmp_path)

    assert code == 2
    refusal = json.loads((tmp_path / "silver-refusal.json").read_text())
    assert "INSUFFICIENT_TRAIN_GROUPS" in refusal["errors"]
    assert not (tmp_path / "instances.silver.json").exists()


def test_export_writes_isolated_silver_truth_after_gate_passes(tmp_path) -> None:
    document = sample_complete_document()

    code = export_silver(document, tmp_path)

    assert code == 0
    assert (tmp_path / "instances.silver.json").exists()
    assert (tmp_path / "silver-manifest.json").exists()
    assert not (tmp_path / "silver-refusal.json").exists()
