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


def test_ingest_serializes_fixed_train_only_fields(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "external"
    truth = tmp_path / "truth.json"
    expected = _write_truth(truth)
    image = tmp_path / "ref-01-normal.png"
    cv2.imwrite(str(image), np.zeros((40, 40, 3), dtype=np.uint8))
    mark_mask_path = tmp_path / "ref-01-normal.mark-mask.png"
    mark_mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.line(mark_mask, (8, 20), (32, 20), 255, 3)
    cv2.imwrite(str(mark_mask_path), mark_mask)
    sidecar = {
        "sample_id": "ref-01-normal",
        "source_reference_sha256": "a" * 64,
        "source_scene_id": "scene-01",
        "source_split": "train",
        "state": "NORMAL",
        "fastener_bbox_xyxy": [5, 5, 35, 35],
        "fixed_segment_xyxy": [[8, 20], [20, 20]],
        "moving_segment_xyxy": [[20, 20], [32, 20]],
        "anchor_xy": [20, 20],
        "review_status": "APPROVED",
        "prompt_sha256": "b" * 64,
        "witness_mark_mask_path": str(mark_mask_path),
    }
    image.with_suffix(".png.json").write_text(json.dumps(sidecar), encoding="utf-8")
    document = ingest_local_candidates([image], output, repo, truth, expected)
    assert document["records"][0]["synthetic"] is True
    assert document["records"][0]["eligible_split"] == "train"
    assert document["records"][0]["witness_mark_mask_path"] == "locals/ref-01-normal.mark-mask.png"
    assert (output / "locals" / "ref-01-normal.mark-mask.png").is_file()


def test_ingest_rejects_partial_source_image_lineage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    truth = tmp_path / "truth.json"
    expected = _write_truth(truth)
    image = tmp_path / "ref.png"
    cv2.imwrite(str(image), np.zeros((40, 40, 3), dtype=np.uint8))
    sidecar = {
        "sample_id": "ref", "source_reference_sha256": "a" * 64,
        "source_scene_id": "scene-01", "source_split": "train", "state": "NORMAL",
        "fastener_bbox_xyxy": [5, 5, 35, 35], "fixed_segment_xyxy": [[8, 20], [20, 20]],
        "moving_segment_xyxy": [[20, 20], [32, 20]], "anchor_xy": [20, 20],
        "review_status": "APPROVED", "prompt_sha256": "b" * 64,
        "source_image": "source.png", "source_bbox_xywh": [5, 5, 30, 30],
    }
    image.with_suffix(".png.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete source image lineage"):
        ingest_local_candidates([image], tmp_path / "external", repo, truth, expected)


def test_build_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "中文试点"
    root.mkdir()
    patch_path = root / "patch.png"
    background_dir = root / "backgrounds"
    background_dir.mkdir()
    background_path = background_dir / "bg.png"
    patch = np.full((40, 40, 3), 180, dtype=np.uint8)
    cv2.line(patch, (10, 20), (30, 20), (20, 30, 220), 4, cv2.LINE_AA)
    mark_mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.line(mark_mask, (10, 20), (30, 20), 255, 4, cv2.LINE_AA)
    for path, image in (
        (patch_path, patch),
        (background_path, np.full((160, 200, 3), 60, dtype=np.uint8)),
    ):
        success, encoded = cv2.imencode(".png", image)
        assert success
        encoded.tofile(path)
    success, encoded_mask = cv2.imencode(".png", mark_mask)
    assert success
    encoded_mask.tofile(root / "patch.mark-mask.png")
    approved = {
        "records": [{
            "sample_id": "ref-01-normal",
            "state": "NORMAL",
            "review_status": "APPROVED",
            "source_split": "train",
            "source_scene_id": "scene-01",
            "source_reference_sha256": "a" * 64,
            "image_path": "patch.png",
            "image_sha256": sha256(patch_path.read_bytes()).hexdigest().upper(),
            "witness_mark_mask_path": "patch.mark-mask.png",
            "witness_mark_mask_sha256": sha256((root / "patch.mark-mask.png").read_bytes()).hexdigest().upper(),
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
        "info": {"partition": "train"},
        "images": [{
            "id": 1, "file_name": "bg.png", "scene_group": "scene-bg",
            "width": 200, "height": 160, "sha256": sha256(background_path.read_bytes()).hexdigest().upper(),
        }],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [70, 50, 35, 35], "area": 1225, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [150, 110, 20, 20], "area": 400, "iscrowd": 0},
        ],
    }
    backgrounds_path = root / "backgrounds.json"
    backgrounds_path.write_text(json.dumps(backgrounds), encoding="utf-8")
    truth = root / "truth.json"
    expected = _write_truth(truth)
    monkeypatch.setattr(
        "crrc_vision.synthetic_pipeline.extract_witness_mark_mask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback extractor used")),
    )
    first = build_full_images(manifest_path, backgrounds_path, background_dir, root / "out-a", 20260829, truth, expected)
    second = build_full_images(manifest_path, backgrounds_path, background_dir, root / "out-b", 20260829, truth, expected)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["formal_truth_sha256"] == expected
    coco = json.loads((root / "out-a" / "instances.synthetic-train.json").read_text(encoding="utf-8"))
    assert len(coco["annotations"]) == 2
    assert sum(annotation.get("origin") == "synthetic_replacement" for annotation in coco["annotations"]) == 1
    assert sum(annotation.get("ignore_state") is True for annotation in coco["annotations"]) == 1
    rendered = cv2.imdecode(
        np.fromfile(root / "out-a" / "images" / "synthetic-0001.png", dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert np.count_nonzero(rendered[:, :, 2].astype(int) - rendered[:, :, 1].astype(int) > 50) > 0


def test_build_does_not_draw_a_mark_when_imagegen_patch_has_none(tmp_path: Path) -> None:
    root = tmp_path / "no-mark"
    root.mkdir()
    patch_path = root / "patch.png"
    background_dir = root / "backgrounds"
    background_dir.mkdir()
    background_path = background_dir / "bg.png"
    cv2.imwrite(str(patch_path), np.full((40, 40, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(background_path), np.full((160, 200, 3), 60, dtype=np.uint8))
    manifest_path = root / "approved-locals.json"
    manifest_path.write_text(json.dumps({"records": [{
        "sample_id": "ref-01-normal", "state": "NORMAL", "review_status": "APPROVED",
        "source_split": "train", "source_scene_id": "scene-01",
        "source_reference_sha256": "a" * 64, "image_path": "patch.png",
        "image_sha256": sha256(patch_path.read_bytes()).hexdigest().upper(),
        "fastener_bbox_xyxy": [8, 8, 32, 32],
        "fixed_segment_xyxy": [[10, 20], [20, 20]],
        "moving_segment_xyxy": [[20, 20], [30, 20]],
        "anchor_xy": [20, 20], "synthetic": True, "eligible_split": "train",
    }]}), encoding="utf-8")
    backgrounds_path = root / "backgrounds.json"
    backgrounds_path.write_text(json.dumps({
        "info": {"partition": "train"},
        "images": [{"id": 1, "file_name": "bg.png", "scene_group": "scene-bg", "width": 200, "height": 160,
                    "sha256": sha256(background_path.read_bytes()).hexdigest().upper()}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [70, 50, 35, 35], "area": 1225, "iscrowd": 0}],
    }), encoding="utf-8")
    truth = root / "truth.json"
    expected = _write_truth(truth)

    with pytest.raises(RuntimeError, match="ImageGen witness mark missing"):
        build_full_images(manifest_path, backgrounds_path, background_dir, root / "out", 20260829, truth, expected)


def test_build_maps_only_imagegen_paint_back_to_its_source_scene(tmp_path: Path) -> None:
    root = tmp_path / "mark-only"
    root.mkdir()
    patch_path = root / "patch.png"
    mark_mask_path = root / "patch.mark-mask.png"
    background_dir = root / "backgrounds"
    background_dir.mkdir()

    patch = np.full((40, 40, 3), (190, 190, 190), dtype=np.uint8)
    mark_mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.line(patch, (10, 20), (30, 20), (10, 20, 230), 3, cv2.LINE_AA)
    cv2.line(mark_mask, (10, 20), (30, 20), 255, 3, cv2.LINE_AA)
    cv2.imwrite(str(patch_path), patch)
    cv2.imwrite(str(mark_mask_path), mark_mask)

    own_background = np.full((160, 200, 3), 90, dtype=np.uint8)
    foreign_background = np.full((160, 200, 3), 35, dtype=np.uint8)
    cv2.imwrite(str(background_dir / "own.png"), own_background)
    cv2.imwrite(str(background_dir / "foreign.png"), foreign_background)
    manifest_path = root / "approved-locals.json"
    manifest_path.write_text(json.dumps({"records": [{
        "sample_id": "ref-01-normal", "state": "NORMAL", "review_status": "APPROVED",
        "source_split": "train", "source_scene_id": "scene-own", "source_image": "own.png",
        "source_bbox_xywh": [70, 50, 35, 35], "source_reference_sha256": "a" * 64,
        "source_image_sha256": sha256((background_dir / "own.png").read_bytes()).hexdigest().upper(),
        "image_path": "patch.png", "witness_mark_mask_path": "patch.mark-mask.png",
        "image_sha256": sha256(patch_path.read_bytes()).hexdigest().upper(),
        "witness_mark_mask_sha256": sha256(mark_mask_path.read_bytes()).hexdigest().upper(),
        "fastener_bbox_xyxy": [8, 8, 32, 32],
        "fixed_segment_xyxy": [[10, 20], [20, 20]],
        "moving_segment_xyxy": [[20, 20], [30, 20]],
        "anchor_xy": [20, 20], "synthetic": True, "eligible_split": "train",
    }]}), encoding="utf-8")
    backgrounds_path = root / "backgrounds.json"
    backgrounds_path.write_text(json.dumps({
        "info": {"partition": "train"},
        "images": [
            {"id": 1, "file_name": "foreign.png", "scene_group": "scene-foreign", "width": 200, "height": 160,
             "sha256": sha256((background_dir / "foreign.png").read_bytes()).hexdigest().upper()},
            {"id": 2, "file_name": "own.png", "scene_group": "scene-own", "width": 200, "height": 160,
             "sha256": sha256((background_dir / "own.png").read_bytes()).hexdigest().upper()},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [70, 50, 35, 35], "area": 1225, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [70, 50, 35, 35], "area": 1225, "iscrowd": 0},
        ],
    }), encoding="utf-8")
    truth = root / "truth.json"
    expected = _write_truth(truth)

    manifest = build_full_images(
        manifest_path, backgrounds_path, background_dir, root / "out", 20260829, truth, expected
    )
    rendered = cv2.imread(str(root / "out" / "images" / "synthetic-0001.png"))
    changed = np.any(rendered != own_background, axis=2)
    assert 0 < np.count_nonzero(changed) < 600
    assert np.all(rendered[0, 0] == own_background[0, 0])
    assert manifest["records"][0]["background_scene_id"] == "scene-own"
    coco = json.loads((root / "out" / "instances.synthetic-train.json").read_text(encoding="utf-8"))
    replacement = next(item for item in coco["annotations"] if item["origin"] == "synthetic_replacement")
    assert replacement["bbox"] == [70.0, 50.0, 35.0, 35.0]
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["records"][0]["image_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="local image SHA-256 mismatch"):
        build_full_images(manifest_path, backgrounds_path, background_dir, root / "tampered-local", 20260829, truth, expected)
    tampered["records"][0]["image_sha256"] = sha256(patch_path.read_bytes()).hexdigest().upper()
    tampered["records"][0]["source_image_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source image SHA-256 mismatch"):
        build_full_images(manifest_path, backgrounds_path, background_dir, root / "tampered", 20260829, truth, expected)
    tampered["records"][0]["source_image_sha256"] = sha256((background_dir / "own.png").read_bytes()).hexdigest().upper()
    tampered["records"][0]["source_split"] = "val"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source_split must be train"):
        build_full_images(manifest_path, backgrounds_path, background_dir, root / "wrong-source-split", 20260829, truth, expected)


def test_build_rejects_non_train_background_partition(tmp_path: Path) -> None:
    root = tmp_path / "wrong-partition"
    root.mkdir()
    manifest_path = root / "approved-locals.json"
    manifest_path.write_text(json.dumps({"records": [{
        "sample_id": "ref-01-normal", "state": "NORMAL", "review_status": "APPROVED",
        "source_split": "train", "source_scene_id": "scene-01", "source_reference_sha256": "a" * 64,
        "image_path": "missing.png", "fastener_bbox_xyxy": [1, 1, 4, 4],
        "fixed_segment_xyxy": [[1, 2], [2, 2]], "moving_segment_xyxy": [[2, 2], [3, 2]],
        "anchor_xy": [2, 2], "synthetic": True, "eligible_split": "train",
    }]}), encoding="utf-8")
    backgrounds_path = root / "backgrounds.json"
    backgrounds_path.write_text(json.dumps({"info": {"partition": "val"}, "images": [], "annotations": []}), encoding="utf-8")
    truth = root / "truth.json"
    expected = _write_truth(truth)
    with pytest.raises(RuntimeError, match="partition must be train"):
        build_full_images(manifest_path, backgrounds_path, root, root / "out", 1, truth, expected)
