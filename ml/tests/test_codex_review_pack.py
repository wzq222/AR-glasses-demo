from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from crrc_vision.codex_review_pack import build_pack


def _sample(tmp_path: Path) -> tuple[dict[str, object], Path]:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (320, 240), "gray").save(source / "a.jpg")
    candidates: dict[str, object] = {
        "images": [
            {
                "id": 1,
                "relative_path": "a.jpg",
                "scene_group": "scene-a",
                "split": "train",
            }
        ],
        "fused_candidates": [
            {
                "id": "c1",
                "image_id": 1,
                "xyxy": [10, 10, 40, 40],
                "category": "fastener",
                "score": 0.99,
                "consensus_status": "multi_source",
                "supporting_families": ["student"],
                "decision": "accept",
            }
        ],
    }
    return candidates, source


def _task(output: Path) -> dict[str, object]:
    return json.loads(
        (output / "first-pass" / "tasks-001.json").read_text(encoding="utf-8")
    )


def test_pack_copies_partition_and_hides_sealed_model_metadata(tmp_path: Path) -> None:
    candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"

    summary = build_pack(
        candidates,
        source,
        output,
        selected_relative_paths=["a.jpg"],
        partition="sealed_test",
        partition_manifest_sha256="A" * 64,
        include_existing_decisions=False,
    )

    task = _task(output)
    candidate = task["images"][0]["candidates"][0]
    manifest = json.loads(
        (output / "pack-manifest.json").read_text(encoding="utf-8")
    )
    assert summary.images == 1
    assert summary.candidates == 1
    assert task["partition"] == "sealed_test"
    assert manifest["partition"] == "sealed_test"
    assert manifest["partition_manifest_sha256"] == "A" * 64
    assert set(candidate) == {
        "candidate_id",
        "context",
        "context_sha256",
    }
    with Image.open(output / candidate["context"]) as context:
        red, green, blue = context.convert("RGB").getpixel((10, 10))
    assert red - green > 100
    assert red - blue > 100


def test_four_overlap_tiles_cover_every_source_pixel(tmp_path: Path) -> None:
    candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"

    build_pack(candidates, source, output, partition="train")
    bounds = [
        row["source_xyxy_normalized"]
        for row in _task(output)["images"][0]["miss_sweep_tiles"]
    ]

    assert len(bounds) == 4
    for x_index in range(21):
        for y_index in range(21):
            x = x_index / 20
            y = y_index / 20
            assert any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in bounds)


def test_pack_candidate_ids_are_exhaustive(tmp_path: Path) -> None:
    candidates, source = _sample(tmp_path)
    candidates["fused_candidates"].append(
        {
            "id": "c2",
            "image_id": 1,
            "xyxy": [100, 100, 140, 140],
            "category": "pipe_joint",
        }
    )
    output = tmp_path / "pack"

    build_pack(candidates, source, output, partition="val")

    assert {
        row["candidate_id"] for row in _task(output)["images"][0]["candidates"]
    } == {"c1", "c2"}


def test_pack_refuses_nonempty_output(tmp_path: Path) -> None:
    candidates, source = _sample(tmp_path)
    output = tmp_path / "pack"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_pack(candidates, source, output, partition="train")
