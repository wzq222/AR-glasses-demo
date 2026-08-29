from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.hard_sample_plan import build_h1a_jobs  # noqa: E402
from crrc_vision.synthetic_contract import (  # noqa: E402
    assert_external_output,
    assert_formal_truth_unchanged,
    sha256_file,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic H1a ImageGen jobs")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--topology-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--formal-truth", type=Path)
    return parser.parse_args()


def _unique_records(document: dict, key: str, label: str) -> dict[str, dict]:
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label} records must be a list")
    by_id: dict[str, dict] = {}
    for record in records:
        identifier = str(record.get(key, ""))
        if not identifier or identifier in by_id:
            raise ValueError(f"{label} has missing or duplicate {key}: {identifier}")
        by_id[identifier] = record
    return by_id


def main() -> int:
    args = _arguments()
    output = assert_external_output(args.output, REPOSITORY_ROOT)
    data_root_text = os.environ.get("CRRC_VISION_DATA_ROOT", "")
    if args.formal_truth is None and not data_root_text:
        raise RuntimeError("set CRRC_VISION_DATA_ROOT or pass --formal-truth")
    formal_truth = (
        args.formal_truth
        if args.formal_truth is not None
        else Path(data_root_text) / "annotations/fastener-v2/instances.json"
    ).resolve()
    formal_hash = assert_formal_truth_unchanged(formal_truth)

    references_doc = json.loads(args.references.read_text(encoding="utf-8"))
    topology_doc = json.loads(args.topology_decisions.read_text(encoding="utf-8"))
    references_by_id = _unique_records(references_doc, "reference_id", "references")
    topology_by_id = _unique_records(
        topology_doc, "reference_id", "topology decisions"
    )
    approved = {
        reference_id: record
        for reference_id, record in topology_by_id.items()
        if record.get("decision") == "APPROVED"
    }
    if len(approved) != 12:
        raise ValueError(f"H1a requires exactly 12 approved topology decisions, got {len(approved)}")
    missing = sorted(set(approved) - set(references_by_id))
    if missing:
        raise ValueError(f"topology decisions reference missing crops: {missing}")
    selected_references = [
        references_by_id[reference_id]
        for reference_id in references_by_id
        if reference_id in approved
    ]
    jobs = build_h1a_jobs(selected_references, approved, seed=args.seed)
    reference_root = args.references.resolve().parent
    for job in jobs:
        reference_path = (reference_root / str(job["crop_path"])).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        if sha256_file(reference_path) != job["source_reference_sha256"]:
            raise RuntimeError(f"reference hash mismatch: {reference_path}")
        job["reference_path"] = str(reference_path)

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "jobs.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing job plan: {destination}")
    document = {
        "schema_version": "h1-imagegen-jobs-v1",
        "seed": args.seed,
        "formal_truth_sha256": formal_hash,
        "references_sha256": sha256_file(args.references),
        "topology_decisions_sha256": sha256_file(args.topology_decisions),
        "count": len(jobs),
        "intent_counts": dict(Counter(str(job["intent"]) for job in jobs)),
        "records": jobs,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    assert_formal_truth_unchanged(formal_truth, formal_hash)
    print(
        json.dumps(
            {
                "output": str(destination),
                "jobs": len(jobs),
                "intent_counts": document["intent_counts"],
                "formal_truth_sha256": formal_hash,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
