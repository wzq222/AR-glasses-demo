import json
from pathlib import Path

import pytest
from PIL import Image

from crrc_vision.synthetic_contract import sha256_file
from crrc_vision.witness_roi_dataset import build_witness_roi_manifest


def _write_source(tmp_path: Path, *, duplicate_id: bool = False) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    locals_dir = source / "locals"
    locals_dir.mkdir(parents=True)
    formal_truth = tmp_path / "formal.json"
    formal_truth.write_text('{"truth":true}\n', encoding="utf-8")
    formal_hash = sha256_file(formal_truth)
    records = []
    for index, state in enumerate(("NORMAL", "SLIGHT_LOOSE", "OBVIOUS_LOOSE")):
        sample_id = "sample-normal" if duplicate_id and index == 1 else f"sample-{state.lower()}"
        image_path = locals_dir / f"{index}.png"
        mask_path = locals_dir / f"{index}.mask.png"
        Image.new("RGB", (64, 64), (20 + index, 30, 40)).save(image_path)
        Image.new("L", (64, 64), 255).save(mask_path)
        records.append(
            {
                "sample_id": sample_id,
                "source_reference_sha256": "A" * 64,
                "source_scene_id": "scene-1",
                "state": state,
                "image_path": f"locals/{index}.png",
                "synthetic": True,
                "eligible_split": "train",
                "source_split": "train",
                "fastener_bbox_xyxy": [8.0, 8.0, 56.0, 56.0],
                "fixed_segment_xyxy": [[30.0, 8.0], [30.0, 30.0]],
                "moving_segment_xyxy": [[30.0, 30.0], [30.0 + index, 55.0]],
                "relative_angle_deg": float(index * 8),
                "review_status": "APPROVED",
                "image_sha256": sha256_file(image_path),
                "witness_mark_mask_path": f"locals/{index}.mask.png",
                "witness_mark_mask_sha256": sha256_file(mask_path),
            }
        )
    manifest = source / "approved-locals.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-repositioned-v2",
                "formal_truth_sha256": formal_hash,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest, formal_truth, formal_hash


def test_build_manifest_keeps_synthetic_geometry_train_only_and_hash_bound(tmp_path: Path) -> None:
    source, formal_truth, formal_hash = _write_source(tmp_path)
    output = tmp_path / "external" / "manifest.json"

    document = build_witness_roi_manifest(
        source_manifest=source,
        formal_truth=formal_truth,
        output_manifest=output,
        repository_root=tmp_path / "repo",
        expected_formal_sha256=formal_hash,
    )

    assert document["governance"] == {
        "synthetic_geometry_only": True,
        "real_state_truth": False,
        "sealed_test_opened": False,
    }
    assert document["counts"]["examples"] == 3
    assert document["counts"]["states"] == {
        "NORMAL": 1,
        "SLIGHT_LOOSE": 1,
        "OBVIOUS_LOOSE": 1,
    }
    assert {row["split"] for row in document["examples"]} == {"train"}
    assert all(row["synthetic_geometry_only"] for row in document["examples"])
    assert document["input_hashes"]["formal_truth_sha256"] == formal_hash
    assert output.is_file()


def test_build_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    source, formal_truth, formal_hash = _write_source(tmp_path, duplicate_id=True)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_witness_roi_manifest(
            source_manifest=source,
            formal_truth=formal_truth,
            output_manifest=tmp_path / "out" / "manifest.json",
            repository_root=tmp_path / "repo",
            expected_formal_sha256=formal_hash,
        )


def test_build_manifest_rejects_asset_hash_mismatch(tmp_path: Path) -> None:
    source, formal_truth, formal_hash = _write_source(tmp_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["records"][0]["image_sha256"] = "0" * 64
    source.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="image hash mismatch"):
        build_witness_roi_manifest(
            source_manifest=source,
            formal_truth=formal_truth,
            output_manifest=tmp_path / "out" / "manifest.json",
            repository_root=tmp_path / "repo",
            expected_formal_sha256=formal_hash,
        )


def test_build_manifest_rejects_formal_truth_mismatch_without_output(tmp_path: Path) -> None:
    source, formal_truth, _ = _write_source(tmp_path)
    output = tmp_path / "out" / "manifest.json"

    with pytest.raises(RuntimeError, match="formal truth SHA-256 changed"):
        build_witness_roi_manifest(
            source_manifest=source,
            formal_truth=formal_truth,
            output_manifest=output,
            repository_root=tmp_path / "repo",
            expected_formal_sha256="F" * 64,
        )

    assert not output.exists()


def test_build_manifest_rejects_output_inside_repository(tmp_path: Path) -> None:
    source, formal_truth, formal_hash = _write_source(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(ValueError, match="Git外"):
        build_witness_roi_manifest(
            source_manifest=source,
            formal_truth=formal_truth,
            output_manifest=repository / "generated" / "manifest.json",
            repository_root=repository,
            expected_formal_sha256=formal_hash,
        )
