import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import crrc_vision.witness_state_review_pack as state_pack
from crrc_vision.witness_state_review_pack import build_state_review_pack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _references(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "reference-pool"
    crops = source / "crops"
    crops.mkdir(parents=True)
    image = np.full((90, 120, 3), 80, dtype=np.uint8)
    cv2.line(image, (20, 45), (100, 45), (0, 0, 230), 6)
    crop = crops / "ref-01.png"
    assert cv2.imwrite(str(crop), image)
    document = {
        "schema_version": "test-references-v1",
        "formal_truth_sha256": "A" * 64,
        "count": 1,
        "records": [
            {
                "reference_id": "ref-01",
                "source_split": "train",
                "source_scene_id": "scene-01",
                "source_image": "a.jpg",
                "source_image_sha256": "B" * 64,
                "source_reference_sha256": _sha256(crop),
                "crop_path": "crops/ref-01.png",
                "crop_box_xyxy": [10, 20, 130, 110],
            }
        ],
    }
    manifest = source / "references.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest, source


def test_builds_real_state_pack_without_inventing_state_truth(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    output = tmp_path / "state-pack"

    summary = build_state_review_pack(references, source, output)

    assert summary.references == 1
    assert summary.geometry_proposals == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    task = json.loads((output / "tasks" / "task-001.json").read_text(encoding="utf-8"))
    record = task["records"][0]
    assert manifest["formal_truth_sha256"] == "A" * 64
    assert record["automatic_state"] == "INSUFFICIENT"
    assert record["automatic_reason"] == "HUMAN_TOPOLOGY_AND_SEGMENT_BINDING_REQUIRED"
    assert record["geometry_proposal"] is None
    assert record["paint_color_proposal"]["proposal_only"] is True
    assert "REAL_ENDPOINTS_REQUIRE_REVIEW" in record["uncertainty_reasons"]
    assert record["review_template"]["output_state"] is None
    assert set(record["evidence_views"]) == {"original_1x", "detail_2x", "detail_4x"}
    assert (output / record["paint_mask_path"]).is_file()


def test_rejects_reference_hash_mismatch(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    document = json.loads(references.read_text(encoding="utf-8"))
    document["records"][0]["source_reference_sha256"] = "F" * 64
    references.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        build_state_review_pack(references, source, tmp_path / "state-pack")
    assert not (tmp_path / "state-pack").exists()


def test_rejects_hash_matching_non_image_before_creating_output(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    crop = source / "crops" / "ref-01.png"
    crop.write_bytes(b"not an image")
    document = json.loads(references.read_text(encoding="utf-8"))
    document["records"][0]["source_reference_sha256"] = _sha256(crop)
    references.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="REFERENCE_DECODE_FAILED"):
        build_state_review_pack(references, source, tmp_path / "state-pack")
    assert not (tmp_path / "state-pack").exists()


def test_rejects_path_like_or_case_colliding_reference_ids_before_output(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    document = json.loads(references.read_text(encoding="utf-8"))
    document["records"][0]["reference_id"] = "../escaped"
    references.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="REFERENCE_ID_INVALID"):
        build_state_review_pack(references, source, tmp_path / "state-pack")
    assert not (tmp_path / "state-pack").exists()

    references, source = _references(tmp_path / "second")
    document = json.loads(references.read_text(encoding="utf-8"))
    duplicate = {**document["records"][0], "reference_id": "REF-01"}
    document["records"].append(duplicate)
    document["count"] = 2
    references.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="REFERENCE_ID_INVALID"):
        build_state_review_pack(references, source, tmp_path / "second-pack")
    assert not (tmp_path / "second-pack").exists()


def test_refuses_to_overwrite_existing_pack(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    output = tmp_path / "state-pack"
    output.mkdir()

    with pytest.raises(FileExistsError):
        build_state_review_pack(references, source, output)


def test_write_failure_does_not_publish_partial_pack(tmp_path: Path, monkeypatch) -> None:
    references, source = _references(tmp_path)
    output = tmp_path / "state-pack"

    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(state_pack, "_write_png", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        build_state_review_pack(references, source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".state-pack.staging-*"))


def test_final_formal_truth_failure_does_not_publish_pack(tmp_path: Path) -> None:
    references, source = _references(tmp_path)
    output = tmp_path / "state-pack"
    formal_truth = tmp_path / "formal.json"
    formal_truth.write_text("different truth", encoding="utf-8")

    with pytest.raises(RuntimeError, match="FORMAL_TRUTH_CHANGED"):
        build_state_review_pack(
            references,
            source,
            output,
            expected_formal_truth_sha256="A" * 64,
            formal_truth_path=formal_truth,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".state-pack.staging-*"))
