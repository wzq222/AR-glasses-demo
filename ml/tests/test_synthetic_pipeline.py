import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

from crrc_vision.synthetic_pipeline import build_full_images, ingest_local_candidates


def _write_truth(path: Path) -> str:
    path.write_bytes(b"unit-test-formal-truth")
    return sha256(path.read_bytes()).hexdigest().upper()


def test_ingest_rejects_missing_sidecar(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "external"
    truth = tmp_path / "truth.json"
    expected = _write_truth(truth)
    image = tmp_path / "ref-01-normal.png"
    cv2.imwrite(str(image), np.zeros((40, 40, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="sidecar"):
        ingest_local_candidates([image], output, repo, truth, expected)


def test_build_is_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    patch_path = root / "patch.png"
    background_dir = root / "backgrounds"
    background_dir.mkdir()
    background_path = background_dir / "bg.png"
    cv2.imwrite(str(patch_path), np.full((40, 40, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(background_path), np.full((160, 200, 3), 60, dtype=np.uint8))
    approved = {
        "records": [{
            "sample_id": "ref-01-normal",
            "state": "NORMAL",
            "review_status": "APPROVED",
            "source_split": "train",
            "source_scene_id": "scene-01",
            "source_reference_sha256": "a" * 64,
            "image_path": "patch.png",
            "fastener_bbox_xyxy": [8, 8, 32, 32],
            "fixed_segment_xyxy": [[10, 20], [20, 20]],
            "moving_segment_xyxy": [[20, 20], [30, 20]],
            "anchor_xy": [20, 20],
            "synthetic": True,
            "eligible_split": "train",
        }]
    }
    manifest_path = root / "approved-locals.json"
    manifest_path.write_text(json.dumps(approved), encoding="utf-8")
    backgrounds = {
        "images": [{
            "id": 1, "file_name": "bg.png", "scene_group": "scene-bg",
            "width": 200, "height": 160,
        }],
        "annotations": [],
    }
    backgrounds_path = root / "backgrounds.json"
    backgrounds_path.write_text(json.dumps(backgrounds), encoding="utf-8")
    truth = root / "truth.json"
    expected = _write_truth(truth)
    first = build_full_images(manifest_path, backgrounds_path, background_dir, root / "out-a", 20260829, truth, expected)
    second = build_full_images(manifest_path, backgrounds_path, background_dir, root / "out-b", 20260829, truth, expected)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["formal_truth_sha256"] == expected
