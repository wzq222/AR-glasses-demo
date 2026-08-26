"""Export reviewed AI labels as isolated silver truth or a refusal report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import verify_truth_unchanged
from crrc_vision.silver_truth import export_silver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="annotations/silver-v1")
    args = parser.parse_args()

    root = asset_root()
    document_path = _below(root, args.document)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    for required in (document_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    document = json.loads(document_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("reviewed silver document must be an object")
    truth_before = _sha256(truth_path)
    code = export_silver(
        document,
        output_root,
        integrity={
            "source_path": args.document,
            "source_sha256": _sha256(document_path),
            "formal_truth_path": args.truth,
            "formal_truth_sha256": truth_before,
        },
    )
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    print(
        json.dumps(
            {
                "status": "exported" if code == 0 else "refused",
                "output": str(output_root),
            },
            ensure_ascii=False,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
