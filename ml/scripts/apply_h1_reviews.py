from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.hard_sample_review import build_review_result_document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve hash-bound H1 visual reviews.")
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--first-review", type=Path, required=True)
    parser.add_argument("--second-review", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    manifest = _load(args.generation_manifest)
    formal_hash = _sha256(args.formal_truth)
    if str(manifest.get("formal_truth_sha256", "")).upper() != formal_hash:
        raise SystemExit("generation manifest does not match frozen formal truth")

    result = build_review_result_document(
        manifest,
        _load(args.first_review),
        _load(args.second_review),
    )
    result["formal_truth_sha256"] = formal_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": result["count"],
                "status_counts": result["status_counts"],
                "formal_truth_sha256": formal_hash,
            }
        )
    )


if __name__ == "__main__":
    main()
