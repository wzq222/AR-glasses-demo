from __future__ import annotations

from pathlib import Path

import pytest

from crrc_vision.error_buckets import (
    ErrorEvidence,
    classify_error,
    detection_errors,
    threshold_from_selection,
    validate_diagnostic_truth,
)


def test_taxonomy_uses_fixed_priority_and_keeps_secondary_tags() -> None:
    evidence = ErrorEvidence(
        area_ratio=0.0004,
        border_distance_ratio=0.0,
        brightness=25.0,
        focus_score=10.0,
        local_contrast=80.0,
        nearby_density=6,
        annotation_dispute=True,
        occluded=True,
        reflection=True,
    )

    primary, secondary = classify_error(evidence)

    assert primary == "annotation_dispute"
    assert secondary == (
        "border_truncation",
        "tiny",
        "dark",
        "blur",
        "occlusion",
        "reflection",
        "dense_pipes",
    )


def test_every_error_gets_one_primary_bucket_and_matching_is_one_to_one() -> None:
    truth = {
        "images": [{"id": 1, "scene_group": "scene-1"}],
        "annotations": [
            {"id": 10, "image_id": 1, "bbox": [0, 0, 20, 20]},
            {"id": 11, "image_id": 1, "bbox": [50, 0, 20, 20]},
        ],
    }
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 1, "bbox": [0, 0, 20, 20], "score": 0.8},
    ]

    errors = detection_errors(predictions, truth, threshold=0.5)

    assert [(row.kind, row.truth_id, row.prediction_index) for row in errors] == [
        ("false_positive", None, 1),
        ("false_negative", 11, None),
    ]


def test_diagnostic_builder_refuses_sealed_truth_by_partition_path_or_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="SEALED_TRUTH_FORBIDDEN"):
        validate_diagnostic_truth(
            tmp_path / "renamed.json",
            {"info": {"partition": "sealed_test"}},
            truth_sha256="A" * 64,
            forbidden_truth_hashes=set(),
        )
    with pytest.raises(ValueError, match="SEALED_TRUTH_FORBIDDEN"):
        validate_diagnostic_truth(
            tmp_path / "instances.sealed-test.json",
            {"info": {"partition": "val"}},
            truth_sha256="B" * 64,
            forbidden_truth_hashes=set(),
        )
    with pytest.raises(ValueError, match="SEALED_TRUTH_FORBIDDEN"):
        validate_diagnostic_truth(
            tmp_path / "renamed.json",
            {"info": {"partition": "val"}},
            truth_sha256="C" * 64,
            forbidden_truth_hashes={"C" * 64},
        )


def test_error_pack_reads_exact_threshold_from_matching_selection() -> None:
    selection = {
        "schema_version": "high-accuracy-selection-v1",
        "mode": "val",
        "threshold": 0.7443283200263977,
        "prediction_sha256": "D" * 64,
        "sealed_test_opened": False,
    }

    assert threshold_from_selection(selection, prediction_sha256="D" * 64) == 0.7443283200263977
    with pytest.raises(ValueError, match="SELECTION_PREDICTION_HASH_MISMATCH"):
        threshold_from_selection(selection, prediction_sha256="E" * 64)
