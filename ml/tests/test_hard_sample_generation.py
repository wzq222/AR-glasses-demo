from pathlib import Path

import pytest

from crrc_vision.hard_sample_generation import (
    build_generation_manifest,
    build_retry_jobs,
)


def _jobs() -> dict:
    return {
        "schema_version": "h1-imagegen-jobs-v1",
        "formal_truth_sha256": "A" * 64,
        "records": [
            {
                "sample_id": "h1a-0001",
                "reference_id": "ref-01",
                "source_reference_sha256": "B" * 64,
                "prompt_sha256": "C" * 64,
                "intent": "ALIGNED",
            },
            {
                "sample_id": "h1a-0002",
                "reference_id": "ref-02",
                "source_reference_sha256": "D" * 64,
                "prompt_sha256": "E" * 64,
                "intent": "LOOKALIKE",
            },
        ],
    }


def test_generation_manifest_hash_binds_every_job(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "h1a-0001-attempt-01.png").write_bytes(b"first")
    (generated / "h1a-0002-attempt-01.png").write_bytes(b"second")
    manifest = build_generation_manifest(_jobs(), generated)
    assert manifest["count"] == 2
    assert all(row["review_status"] == "UNREVIEWED" for row in manifest["records"])
    assert all(len(row["image_sha256"]) == 64 for row in manifest["records"])


def test_generation_manifest_rejects_missing_attempt(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "h1a-0001-attempt-01.png").write_bytes(b"first")
    with pytest.raises(FileNotFoundError, match="h1a-0002"):
        build_generation_manifest(_jobs(), generated)


def test_retry_jobs_only_include_non_approved_samples() -> None:
    reviewed = {
        "formal_truth_sha256": "A" * 64,
        "records": [
            {"sample_id": "h1a-0001", "review_status": "APPROVED"},
            {"sample_id": "h1a-0002", "review_status": "UNCERTAIN"},
        ],
    }

    retry = build_retry_jobs(_jobs(), reviewed)

    assert retry["count"] == 1
    assert [row["sample_id"] for row in retry["records"]] == ["h1a-0002"]
    assert retry["formal_truth_sha256"] == "A" * 64


def test_retry_jobs_reject_review_identity_mismatch() -> None:
    reviewed = {
        "formal_truth_sha256": "A" * 64,
        "records": [{"sample_id": "unknown", "review_status": "REJECTED"}],
    }

    with pytest.raises(ValueError, match="review identities"):
        build_retry_jobs(_jobs(), reviewed)
