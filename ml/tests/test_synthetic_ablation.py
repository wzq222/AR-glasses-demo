from pathlib import Path

import pytest

from crrc_vision.synthetic_ablation import build_capped_batches, merge_training_documents
from crrc_vision.ultralytics_ablation import CappedSyntheticBatchSampler


def _document(count: int, *, synthetic: bool) -> dict:
    return {
        "info": {"partition": "train"},
        "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}],
        "images": [{
            "id": index + 1,
            "file_name": f"{'synthetic' if synthetic else 'real'}-{index + 1}.png",
            "width": 100,
            "height": 80,
            "scene_group": f"scene-{index + 1}",
            "synthetic": synthetic,
        } for index in range(count)],
        "annotations": [{
            "id": index + 1,
            "image_id": index + 1,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
        } for index in range(count)],
    }


def test_merge_training_documents_remaps_synthetic_ids_and_enforces_fraction(tmp_path: Path) -> None:
    merged = merge_training_documents(
        _document(6, synthetic=False),
        _document(2, synthetic=True),
        synthetic_image_root=tmp_path,
        maximum_synthetic_fraction=0.30,
    )
    synthetic_images = [item for item in merged["images"] if item["synthetic"]]
    assert len(merged["images"]) == 8
    assert len(synthetic_images) == 2
    assert all(item["id"] >= 1_000_000 for item in synthetic_images)
    assert all(Path(item["file_name"]).is_absolute() for item in synthetic_images)
    assert {item["image_id"] for item in merged["annotations"]} == {
        item["id"] for item in merged["images"]
    }
    with pytest.raises(ValueError, match="SYNTHETIC_FRACTION_EXCEEDED"):
        merge_training_documents(
            _document(3, synthetic=False),
            _document(2, synthetic=True),
            synthetic_image_root=tmp_path,
            maximum_synthetic_fraction=0.30,
        )


def test_capped_batches_match_control_steps_and_never_exceed_one_synthetic() -> None:
    real_indices = list(range(156))
    synthetic_indices = list(range(156, 204))
    first = build_capped_batches(
        real_indices,
        synthetic_indices,
        batch_size=4,
        maximum_synthetic_fraction=0.30,
        seed=20260829,
        epoch=0,
    )
    repeated = build_capped_batches(
        real_indices,
        synthetic_indices,
        batch_size=4,
        maximum_synthetic_fraction=0.30,
        seed=20260829,
        epoch=0,
    )
    next_epoch = build_capped_batches(
        real_indices,
        synthetic_indices,
        batch_size=4,
        maximum_synthetic_fraction=0.30,
        seed=20260829,
        epoch=1,
    )
    assert first == repeated
    assert first != next_epoch
    assert len(first) == 39
    assert all(len(batch) == 4 for batch in first)
    assert all(sum(index in synthetic_indices for index in batch) == 1 for batch in first)
    assert len({index for batch in first for index in batch if index in real_indices}) == 117


def test_batch_sampler_advances_epoch_without_changing_step_count() -> None:
    sampler = CappedSyntheticBatchSampler(
        list(range(12)), list(range(12, 16)),
        batch_size=4, maximum_synthetic_fraction=0.30, seed=20260829,
    )
    epoch_zero = list(iter(sampler))
    epoch_one = list(iter(sampler))
    assert len(sampler) == len(epoch_zero) == len(epoch_one) == 3
    assert epoch_zero != epoch_one
