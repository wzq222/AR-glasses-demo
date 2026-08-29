import json
from hashlib import sha256
from pathlib import Path

from crrc_vision.synthetic_audit import audit_full_dataset, audit_records


def _record(index: int, state: str) -> dict:
    return {
        "sample_id": f"sample-{state.lower()}-{index}",
        "synthetic": True,
        "eligible_split": "train",
        "source_split": "train",
        "source_scene_id": f"scene-{index:02d}",
        "source_reference_sha256": f"{index + 1:064x}",
        "state": state,
        "review_status": "APPROVED",
    }


def test_audit_rejects_validation_lineage() -> None:
    record = _record(0, "NORMAL")
    record["source_split"] = "val"
    result = audit_records([record])
    assert result.passed is False
    assert any("source_split" in error for error in result.errors)


def test_audit_requires_balanced_approved_states() -> None:
    records = [_record(index, "NORMAL") for index in range(8)]
    result = audit_records(records)
    assert result.passed is False
    assert result.approved_by_state["NORMAL"] == 8
    assert result.approved_by_state["SLIGHT_LOOSE"] == 0


def test_audit_passes_balanced_pilot() -> None:
    records = []
    for state_offset, state in enumerate(("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")):
        records.extend(_record(state_offset * 20 + index, state) for index in range(8))
    result = audit_records(records)
    assert result.passed is True
    assert result.approved_total == 24


def test_full_audit_reads_image_coco_and_requires_hash_bound_review(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_bytes(b"truth")
    truth_hash = sha256(truth.read_bytes()).hexdigest().upper()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "synthetic-0001.png"
    image_path.write_bytes(b"real-image-bytes")
    image_hash = sha256(image_path.read_bytes()).hexdigest().upper()
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    crop_path = review_dir / "full-0001-normal.png"
    crop_path.write_bytes(b"reviewed-crop")
    crop_hash = sha256(crop_path.read_bytes()).hexdigest().upper()
    review_pack_path = review_dir / "manifest.json"
    review_pack_path.write_text(json.dumps({"records": [{
        "sample_id": "sample-normal-0", "crop_path": crop_path.name,
        "crop_sha256": crop_hash, "full_image_sha256": image_hash,
    }]}), encoding="utf-8")
    review_pack_hash = sha256(review_pack_path.read_bytes()).hexdigest().upper()
    record = {
        **_record(0, "NORMAL"),
        "source_sample_id": "local-0001",
        "image_id": 1,
        "image_path": "images/synthetic-0001.png",
        "image_sha256": image_hash,
        "target_bbox_xywh": [10.0, 10.0, 20.0, 20.0],
        "residual_original_mark_pixels": 0,
        "review_evidence": {
            "reviewer": "codex",
            "reviewed_at": "2026-08-29T00:00:00+08:00",
            "full_image_reviewed": True,
            "crop_reviewed": True,
            "image_sha256": image_hash,
            "crop_sha256": crop_hash,
            "review_pack_manifest_sha256": review_pack_hash,
        },
    }
    content_hash = sha256(image_hash.encode("ascii")).hexdigest().upper()
    document = {
        "schema_version": "synthetic-marked-point-full-v1",
        "formal_truth_sha256": truth_hash,
        "content_sha256": content_hash,
        "records": [record],
    }
    coco_path = tmp_path / "instances.synthetic-train.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "synthetic-0001.png", "width": 100, "height": 100,
                    "sha256": image_hash, "synthetic": True, "eligible_split": "train",
                    "source_scene_id": "scene-00", "residual_original_mark_pixels": 0}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0],
                         "origin": "synthetic_replacement", "state": "NORMAL", "ignore_state": False,
                         "source_sample_id": "local-0001",
                         "fixed_segment_xyxy": [[12.0, 20.0], [20.0, 20.0]],
                         "moving_segment_xyxy": [[20.0, 20.0], [28.0, 20.0]], "anchor_xy": [20.0, 20.0]}],
    }), encoding="utf-8")
    result = audit_full_dataset(document, tmp_path, coco_path, truth,
                                minimum_per_state=0, expected_formal_hash=truth_hash,
                                review_pack_manifest_path=review_pack_path)
    assert result.passed is True

    evidence = record.pop("review_evidence")
    result = audit_full_dataset(document, tmp_path, coco_path, truth,
                                minimum_per_state=0, expected_formal_hash=truth_hash,
                                review_pack_manifest_path=review_pack_path)
    assert result.passed is False
    assert any("review_evidence" in error for error in result.errors)
    record["review_evidence"] = evidence
    crop_path.write_bytes(b"tampered-crop")
    result = audit_full_dataset(document, tmp_path, coco_path, truth,
                                minimum_per_state=0, expected_formal_hash=truth_hash,
                                review_pack_manifest_path=review_pack_path)
    assert result.passed is False
    assert any("review crop hash mismatch" in error for error in result.errors)
