from __future__ import annotations

from pathlib import Path

from .synthetic_contract import sha256_file


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
