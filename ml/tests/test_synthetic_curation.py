import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from crrc_vision.synthetic_curation import curate_candidates


def test_curate_candidates_writes_exact_imagegen_mask_and_sidecar(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    bases = tmp_path / "bases"
    output = tmp_path / "curated"
    candidates.mkdir()
    bases.mkdir()
    baseline = np.full((100, 120, 3), 100, dtype=np.uint8)
    generated = baseline.copy()
    cv2.line(generated, (30, 50), (55, 50), (10, 220, 240), 4)
    cv2.line(generated, (55, 50), (80, 54), (10, 220, 240), 4)
    cv2.imwrite(str(bases / "ref-01.png"), baseline)
    cv2.imwrite(str(candidates / "ref-01-slight.png"), generated)
    references = {
        "records": [{
            "reference_id": "ref-01",
            "source_split": "train",
            "source_scene_id": "scene-01",
            "source_reference_sha256": "a" * 64,
            "source_image": "source.png",
            "source_image_sha256": "c" * 64,
            "source_bbox_xywh": [10, 20, 30, 40],
            "prompt_sha256": {"SLIGHT_LOOSE": "b" * 64},
        }]
    }
    references_path = tmp_path / "references.json"
    references_path.write_text(json.dumps(references), encoding="utf-8")
    selection = {
        "records": [{
            "image": "ref-01-slight.png",
            "reference_id": "ref-01",
            "state": "SLIGHT_LOOSE",
            "mark_roi_xyxy": [20, 40, 90, 65],
            "fastener_bbox_xyxy": [15, 25, 100, 80],
            "fixed_segment_xyxy": [[30, 50], [55, 50]],
            "moving_segment_xyxy": [[55, 50], [80, 54]],
            "anchor_xy": [55, 50],
            "review_status": "APPROVED",
        }]
    }
    result = curate_candidates(selection, references_path, candidates, bases, output)

    assert result["approved_by_state"] == {"SLIGHT_LOOSE": 1}
    sidecar_path = output / "ref-01-slight.png.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["witness_mark_mask_path"] == "ref-01-slight.mark-mask.png"
    assert sidecar["source_image"] == "source.png"
    assert sidecar["source_image_sha256"] == "c" * 64
    assert sidecar["source_bbox_xywh"] == [10, 20, 30, 40]
    mask = cv2.imread(str(output / sidecar["witness_mark_mask_path"]), cv2.IMREAD_GRAYSCALE)
    assert int(np.count_nonzero(mask)) > 40
    assert int(np.count_nonzero(mask[:, :15])) == 0
    selection["records"][0]["review_status"] = "UNCERTAIN"
    with pytest.raises(RuntimeError, match="not approved"):
        curate_candidates(selection, references_path, candidates, bases, tmp_path / "unreviewed")


def test_curate_candidates_rejects_state_geometry_mismatch(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    bases = tmp_path / "bases"
    candidates.mkdir()
    bases.mkdir()
    image = np.full((60, 60, 3), 100, dtype=np.uint8)
    cv2.line(image, (10, 30), (50, 30), (10, 220, 240), 3)
    cv2.imwrite(str(candidates / "candidate.png"), image)
    cv2.imwrite(str(bases / "ref-01.png"), np.full_like(image, 100))
    references_path = tmp_path / "references.json"
    references_path.write_text(json.dumps({"records": [{
        "reference_id": "ref-01", "source_split": "train", "source_scene_id": "scene-01",
        "source_reference_sha256": "a" * 64, "prompt_sha256": {"OBVIOUS_LOOSE": "b" * 64},
    }]}), encoding="utf-8")
    selection = {"records": [{
        "image": "candidate.png", "reference_id": "ref-01", "state": "OBVIOUS_LOOSE",
        "mark_roi_xyxy": [5, 20, 55, 40], "fastener_bbox_xyxy": [2, 10, 58, 50],
        "fixed_segment_xyxy": [[10, 30], [30, 30]],
        "moving_segment_xyxy": [[30, 30], [50, 30]], "anchor_xy": [30, 30],
        "review_status": "APPROVED",
    }]}
    try:
        curate_candidates(selection, references_path, candidates, bases, tmp_path / "out")
    except RuntimeError as exc:
        assert "geometry mismatch" in str(exc)
    else:
        raise AssertionError("state mismatch should be rejected")


def test_curate_candidates_rejects_unreviewed_selection_and_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "curated"
    output.mkdir()
    (output / "stale.png").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="must be empty"):
        curate_candidates({"records": []}, tmp_path / "missing.json", tmp_path, tmp_path, output)
