from __future__ import annotations

from pathlib import Path

from .synthetic_contract import sha256_file


def build_retry_jobs(jobs_document: dict, reviewed_document: dict) -> dict:
    jobs = jobs_document.get("records")
    reviewed = reviewed_document.get("records")
    if not isinstance(jobs, list) or not isinstance(reviewed, list):
        raise ValueError("jobs and reviewed records must be lists")
    if jobs_document.get("formal_truth_sha256") != reviewed_document.get(
        "formal_truth_sha256"
    ):
        raise ValueError("formal truth lineage mismatch")

    job_ids = [str(record.get("sample_id", "")) for record in jobs]
    review_by_id = {
        str(record.get("sample_id", "")): record for record in reviewed
    }
    if len(set(job_ids)) != len(job_ids) or set(review_by_id) != set(job_ids):
        raise ValueError("review identities must exactly cover job identities")

    retry_records = [
        record
        for record in jobs
        if review_by_id[str(record["sample_id"])].get("review_status") != "APPROVED"
    ]
    return {
        **{key: value for key, value in jobs_document.items() if key not in {"records", "count"}},
        "schema_version": "h1-imagegen-retry-jobs-v1",
        "count": len(retry_records),
        "records": retry_records,
    }


def build_generation_manifest(
    jobs_document: dict, generated_dir: Path, attempt: int = 1
) -> dict:
    records = jobs_document.get("records")
    if not isinstance(records, list):
        raise ValueError("jobs records must be a list")
    generated_records = []
    for job in records:
        sample_id = str(job.get("sample_id", ""))
        image_path = generated_dir / f"{sample_id}-attempt-{attempt:02d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing generated attempt for {sample_id}: {image_path}")
        generated_records.append(
            {
                "sample_id": sample_id,
                "attempt": attempt,
                "tool": "built-in-imagegen",
                "reference_id": job["reference_id"],
                "intent": job["intent"],
                "source_reference_sha256": job["source_reference_sha256"],
                "prompt_sha256": job["prompt_sha256"],
                "image_path": image_path.name,
                "image_sha256": sha256_file(image_path),
                "review_status": "UNREVIEWED",
            }
        )
    return {
        "schema_version": "h1-imagegen-generation-v1",
        "formal_truth_sha256": jobs_document.get("formal_truth_sha256"),
        "count": len(generated_records),
        "records": generated_records,
    }
