from __future__ import annotations

import json
from pathlib import Path

from .synthetic_contract import sha256_file


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_hash_bound_review(
    manifest_path: Path,
    review_pack_manifest_path: Path,
    decisions_path: Path,
) -> dict:
    """Apply explicit visual decisions only when their reviewed bytes still match."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_pack = json.loads(review_pack_manifest_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    pack_hash = sha256_file(review_pack_manifest_path)
    if str(decisions.get("review_pack_manifest_sha256", "")).upper() != pack_hash:
        raise RuntimeError("review-pack manifest SHA-256 mismatch")
    pack_records = {str(item["sample_id"]): item for item in review_pack.get("records", [])}
    decision_records = {str(item["sample_id"]): item for item in decisions.get("records", [])}
    manifest_ids = {str(item["sample_id"]) for item in manifest.get("records", [])}
    if set(pack_records) != manifest_ids or set(decision_records) != manifest_ids:
        raise RuntimeError("review pack and decisions must cover every manifest record exactly")
    if len(pack_records) != len(review_pack.get("records", [])) or len(decision_records) != len(decisions.get("records", [])):
        raise RuntimeError("duplicate review sample_id")
    for record in manifest["records"]:
        sample_id = str(record["sample_id"])
        pack = pack_records[sample_id]
        decision = decision_records[sample_id]
        crop_path = review_pack_manifest_path.parent / str(pack["crop_path"])
        if not crop_path.is_file() or sha256_file(crop_path) != str(pack.get("crop_sha256", "")).upper():
            raise RuntimeError(f"review crop SHA-256 mismatch: {sample_id}")
        if str(pack.get("full_image_sha256", "")).upper() != str(record.get("image_sha256", "")).upper():
            raise RuntimeError(f"reviewed full image SHA-256 mismatch: {sample_id}")
        status = str(decision.get("decision", ""))
        if status not in {"APPROVED", "REJECTED", "UNCERTAIN"}:
            raise RuntimeError(f"invalid review decision: {sample_id}")
        record["review_status"] = status
        record["review_evidence"] = {
            "reviewer": str(decisions.get("reviewer", "")),
            "reviewed_at": str(decisions.get("reviewed_at", "")),
            "full_image_reviewed": decision.get("full_image_reviewed") is True,
            "crop_reviewed": decision.get("crop_reviewed") is True,
            "image_sha256": str(record["image_sha256"]).upper(),
            "crop_sha256": str(pack["crop_sha256"]).upper(),
            "review_pack_manifest_sha256": pack_hash,
        }
        if status == "APPROVED" and (
            record["review_evidence"]["reviewer"] == ""
            or record["review_evidence"]["reviewed_at"] == ""
            or not record["review_evidence"]["full_image_reviewed"]
            or not record["review_evidence"]["crop_reviewed"]
        ):
            raise RuntimeError(f"approved decision lacks complete visual review: {sample_id}")
    _atomic_json(manifest_path, manifest)
    return manifest
