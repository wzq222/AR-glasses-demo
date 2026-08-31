"""Evaluate a marked-point proposal model by recall and candidate burden."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.marked_point_model_gate import build_proposal_gate_document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.99)
    args = parser.parse_args()

    for path in (args.truth, args.predictions, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{args.output}")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if truth.get("info", {}).get("partition") == "sealed_test":
        raise ValueError("MARKED_POINT_GATE_FORBIDS_SEALED_TEST")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("PREDICTIONS_INVALID")
    document = build_proposal_gate_document(
        predictions,
        truth,
        model_sha256=_sha256(args.model),
        truth_sha256=_sha256(args.truth),
        prediction_sha256=_sha256(args.predictions),
        minimum_recall=args.minimum_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

