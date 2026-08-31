import hashlib
import json
from pathlib import Path

import pytest

from crrc_vision.witness_state_second_review import audit_second_pass_reviews


def _pack(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "pack"
    tasks = root / "tasks"
    evidence = root / "evidence"
    grids = root / "coordinate-grids"
    tasks.mkdir(parents=True)
    evidence.mkdir()
    grids.mkdir()
    evidence_views = {}
    for name in ("original_1x", "detail_2x", "detail_4x"):
        path = evidence / f"ref-01-{name}.png"
        path.write_bytes(f"verified-{name}".encode())
        evidence_views[name] = {
            "path": f"evidence/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        }
    grid_path = grids / "ref-01.png"
    grid_path.write_bytes(b"verified-grid")
    task = {
        "schema_version": "real-witness-state-second-pass-task-v1",
        "blind_to_first_review": True,
        "records": [
            {
                "reference_id": "ref-01",
                "evidence_views": evidence_views,
                "coordinate_grid_path": "coordinate-grids/ref-01.png",
                "coordinate_grid_sha256": hashlib.sha256(grid_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        ],
    }
    task_path = tasks / "task-001.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    task_hash = hashlib.sha256(task_path.read_bytes()).hexdigest().upper()
    manifest = {
        "schema_version": "real-witness-state-second-pass-pack-v1",
        "formal_truth_sha256": "A" * 64,
        "references": 1,
        "blind_to_first_review": True,
        "first_review_fields_included": False,
        "task_files": {"tasks/task-001.json": task_hash},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()


def _decisions(tmp_path: Path, manifest_hash: str) -> Path:
    document = {
        "schema_version": "real-witness-state-second-pass-decisions-v1",
        "source_pack_manifest_sha256": manifest_hash,
        "formal_truth_sha256": "A" * 64,
        "blind_to_first_review_fields": True,
        "count": 1,
        "records": [
            {
                "reference_id": "ref-01",
                "review_status": "REVIEWED",
                "topology": "nut_plate",
                "mark_role": "bridges_moving_fixed",
                "quality_pass": False,
                "fixed_segment_xyxy": [[0.0, 0.0], [10.0, 0.0]],
                "moving_segment_xyxy": [[12.0, 0.0], [22.0, 0.0]],
                "fixed_segment_confidence": 0.55,
                "moving_segment_confidence": 0.55,
                "reference_size": 100.0,
                "damaged_mark": False,
                "output_state": "INSUFFICIENT",
                "review_hint": "LOW_RESOLUTION",
                "truth_basis": "single_observation",
                "reason": "too small",
            }
        ],
    }
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_audit_measures_endpoints_but_keeps_single_observation_out_of_training(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)

    result = audit_second_pass_reviews(manifest, decisions)

    assert result["summary"]["reviewed"] == 1
    assert result["summary"]["endpoint_complete"] == 1
    assert result["summary"]["training_eligible"] == 0
    assert result["summary"]["state_counts"] == {"INSUFFICIENT": 1}
    assert result["records"][0]["geometry_metrics"]["angle_degrees"] == pytest.approx(0.0)


def test_single_observation_cannot_be_promoted_to_displaced_truth(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)
    document = json.loads(decisions.read_text(encoding="utf-8"))
    record = document["records"][0]
    record["quality_pass"] = True
    record["fixed_segment_confidence"] = 0.95
    record["moving_segment_confidence"] = 0.95
    record["output_state"] = "DISPLACED"
    decisions.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="DECIDABLE_STATE_REQUIRES_CONTROLLED_PAIR"):
        audit_second_pass_reviews(manifest, decisions)


def test_decisions_must_bind_the_exact_second_pass_manifest(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)
    document = json.loads(decisions.read_text(encoding="utf-8"))
    document["source_pack_manifest_sha256"] = "F" * 64
    decisions.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SOURCE_PACK_MANIFEST_HASH_MISMATCH"):
        audit_second_pass_reviews(manifest, decisions)


def test_audit_rejects_pack_without_explicit_blind_contract(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["blind_to_first_review"] = False
    manifest.write_text(json.dumps(document), encoding="utf-8")
    decisions = _decisions(tmp_path, digest)

    with pytest.raises(ValueError, match="SOURCE_PACK_NOT_BLIND"):
        audit_second_pass_reviews(manifest, decisions)


def test_audit_rejects_decisions_without_blind_attestation(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)
    document = json.loads(decisions.read_text(encoding="utf-8"))
    document["blind_to_first_review_fields"] = False
    decisions.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="DECISIONS_NOT_BLIND"):
        audit_second_pass_reviews(manifest, decisions)


def test_audit_rejects_evidence_changed_after_pack_publication(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)
    evidence = manifest.parent / "evidence" / "ref-01-original_1x.png"
    evidence.write_bytes(b"tampered evidence")

    with pytest.raises(RuntimeError, match="SOURCE_EVIDENCE_HASH_MISMATCH"):
        audit_second_pass_reviews(manifest, decisions)


def test_historical_second_pass_cannot_claim_controlled_pair_by_string(tmp_path: Path) -> None:
    manifest, digest = _pack(tmp_path)
    decisions = _decisions(tmp_path, digest)
    document = json.loads(decisions.read_text(encoding="utf-8"))
    record = document["records"][0]
    record["truth_basis"] = "real_controlled_pair"
    record["quality_pass"] = True
    record["fixed_segment_confidence"] = 0.95
    record["moving_segment_confidence"] = 0.95
    record["output_state"] = "DISPLACED"
    decisions.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="DECIDABLE_STATE_REQUIRES_CONTROLLED_PAIR"):
        audit_second_pass_reviews(manifest, decisions)
