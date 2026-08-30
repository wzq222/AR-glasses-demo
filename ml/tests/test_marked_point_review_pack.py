import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from crrc_vision.marked_point_review_pack import build_review_pack


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _sample(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (320, 240), "gray").save(source / "a.jpg")
    Image.new("RGB", (400, 300), "navy").save(source / "b.jpg")
    selection = {
        "old_sealed_test_opened": False,
        "forbidden_old_sealed": {"sha256": ["f" * 64], "paths": ["sealed.jpg"]},
        "train": [
            {
                "image_id": 1,
                "relative_path": "a.jpg",
                "scene_group": "scene-a",
                "sha256": _digest(source / "a.jpg"),
            }
        ],
        "val": [
            {
                "image_id": 2,
                "relative_path": "b.jpg",
                "scene_group": "scene-b",
                "sha256": _digest(source / "b.jpg"),
            }
        ],
    }
    candidates = {
        "images": [
            {"id": 1, "relative_path": "a.jpg"},
            {"id": 2, "relative_path": "b.jpg"},
        ],
        "fused_candidates": [
            {
                "id": "a1",
                "image_id": 1,
                "relative_path": "a.jpg",
                "xyxy": [10, 10, 40, 40],
                "sources": ["color_mark"],
            },
            {
                "id": "a2",
                "image_id": 1,
                "relative_path": "a.jpg",
                "xyxy": [100, 100, 150, 150],
                "sources": ["fastener_v2_2"],
            },
            {
                "id": "b1",
                "image_id": 2,
                "relative_path": "b.jpg",
                "xyxy": [20, 20, 80, 80],
                "sources": ["color_mark", "fastener_v2_2"],
            },
        ],
    }
    return selection, candidates, source


def _task_images(output: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((output / "first-pass").glob("tasks-*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["images"])
    return rows


def test_pack_has_full_image_four_scans_and_every_candidate(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"
    summary = build_review_pack(selection, candidates, source, output)
    tasks = _task_images(output)
    assert summary.images == 2
    assert summary.scan_tiles == 8
    assert summary.candidates == 3
    assert all(
        row["business_target"]
        == "marked anti-loosening inspection point"
        for row in tasks
    )
    assert {candidate["candidate_id"] for row in tasks for candidate in row["candidates"]} == {
        "a1",
        "a2",
        "b1",
    }
    assert all(len(row["scan_tiles"]) == 4 for row in tasks)


def test_pack_preserves_original_candidate_pixels_and_two_zoom_levels(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"

    build_review_pack(
        selection,
        candidates,
        source,
        output,
        include_zoom_evidence=True,
    )

    candidate = _task_images(output)[0]["candidates"][0]
    evidence = candidate["evidence_views"]
    assert set(evidence) == {"original_1x", "detail_2x", "detail_4x"}
    original = Image.open(output / evidence["original_1x"]["path"])
    detail_2x = Image.open(output / evidence["detail_2x"]["path"])
    detail_4x = Image.open(output / evidence["detail_4x"]["path"])
    assert detail_2x.size == (original.width * 2, original.height * 2)
    assert detail_4x.size == (original.width * 4, original.height * 4)
    assert evidence["original_1x"]["source"] == "decoded_original_pixels"
    assert evidence["detail_4x"]["interpolation"] == "nearest"


def test_pack_does_not_expand_every_raw_candidate_by_default(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"

    build_review_pack(selection, candidates, source, output)

    assert not (output / "candidate-evidence").exists()
    assert all(
        "evidence_views" not in candidate
        for image in _task_images(output)
        for candidate in image["candidates"]
    )


def test_pack_refuses_old_sealed_hash(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    selection["train"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="OLD_SEALED_IMAGE_FORBIDDEN"):
        build_review_pack(selection, candidates, source, tmp_path / "pack")


def test_pack_refuses_missing_or_duplicate_candidate_identity(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    candidates["fused_candidates"][1]["id"] = "a1"
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE_ID"):
        build_review_pack(selection, candidates, source, tmp_path / "pack")


def test_pack_refuses_existing_output(tmp_path):
    selection, candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_review_pack(selection, candidates, source, output)
