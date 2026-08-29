import json
from hashlib import sha256
from pathlib import Path

import pytest

from crrc_vision.synthetic_contract import sha256_file
from crrc_vision.synthetic_review import apply_hash_bound_review


def test_apply_review_binds_full_image_and_crop_hashes(tmp_path: Path) -> None:
    image_hash = sha256(b"image").hexdigest().upper()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"records": [{
        "sample_id": "full-0001", "image_sha256": image_hash, "review_status": "UNCERTAIN"
    }]}), encoding="utf-8")
    pack_dir = tmp_path / "review"
    pack_dir.mkdir()
    crop_path = pack_dir / "full-0001-normal.png"
    crop_path.write_bytes(b"crop")
    pack_path = pack_dir / "manifest.json"
    pack_path.write_text(json.dumps({"records": [{
        "sample_id": "full-0001", "crop_path": crop_path.name,
        "crop_sha256": sha256_file(crop_path), "full_image_sha256": image_hash,
    }]}), encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({
        "reviewer": "codex", "reviewed_at": "2026-08-29T00:00:00+08:00",
        "review_pack_manifest_sha256": sha256_file(pack_path),
        "records": [{"sample_id": "full-0001", "decision": "APPROVED",
                     "full_image_reviewed": True, "crop_reviewed": True}],
    }), encoding="utf-8")
    result = apply_hash_bound_review(manifest_path, pack_path, decisions_path)
    assert result["records"][0]["review_status"] == "APPROVED"
    assert result["records"][0]["review_evidence"]["crop_sha256"] == sha256_file(crop_path)

    crop_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="crop SHA-256 mismatch"):
        apply_hash_bound_review(manifest_path, pack_path, decisions_path)
