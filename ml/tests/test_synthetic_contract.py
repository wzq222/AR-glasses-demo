from pathlib import Path

import pytest

from crrc_vision.synthetic_contract import (
    FROZEN_FORMAL_TRUTH_SHA256,
    SyntheticRecord,
    assert_external_output,
    assert_formal_truth_unchanged,
)


def test_synthetic_record_is_train_only() -> None:
    record = SyntheticRecord(
        sample_id="ref-01-normal-00",
        source_reference_sha256="a" * 64,
        source_scene_id="scene-01",
        state="NORMAL",
        image_path="locals/ref-01-normal-00.png",
    )
    assert record.synthetic is True
    assert record.eligible_split == "train"


def test_synthetic_record_rejects_absolute_image_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="相对路径"):
        SyntheticRecord(
            sample_id="ref-01-normal-00",
            source_reference_sha256="a" * 64,
            source_scene_id="scene-01",
            state="NORMAL",
            image_path=str(tmp_path / "absolute.png"),
        )


def test_external_output_rejects_repo_child(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git外"):
        assert_external_output(tmp_path / "repo" / "out", tmp_path / "repo")


def test_formal_truth_hash_mismatch_is_fatal(tmp_path: Path) -> None:
    truth = tmp_path / "instances.json"
    truth.write_bytes(b"frozen")
    with pytest.raises(RuntimeError, match="formal truth"):
        assert_formal_truth_unchanged(truth, FROZEN_FORMAL_TRUTH_SHA256)
