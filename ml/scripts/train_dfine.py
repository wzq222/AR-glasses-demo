"""Guarded D-FINE-N training entrypoint; refuses dirty or partial labels."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.training import TrainingReadiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations/prelabel-v1/instances.json")
    parser.add_argument("--review-index", default="review-packs/prelabel-v2/review-index.csv")
    parser.add_argument("--dfine-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = asset_root()
    document = json.loads((root / args.annotations).read_text(encoding="utf-8"))
    with (root / args.review_index).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("candidate_id")]
    decisions = Counter(row["decision"] for row in rows)
    reviewed = decisions["accept"] + decisions["reject"]
    readiness = TrainingReadiness(
        images=len(document["images"]),
        accepted=decisions["accept"],
        rejected=decisions["reject"],
        unreviewed=len(document["annotations"]) - reviewed,
    )
    report_path = root / "runs" / "training-readiness-v1.json"
    report_path.write_text(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
    if not readiness.can_train:
        print(f"training refused; report written to {report_path}", file=sys.stderr)
        return 2
    if not args.execute:
        print("training gate passed; use --execute with a pinned D-FINE checkout and config")
        return 0
    if args.dfine_root is None or args.config is None:
        parser.error("--dfine-root and --config are required with --execute")

    train_script = args.dfine_root.resolve() / "train.py"
    config = args.config.resolve()
    if not train_script.is_file() or not config.is_file():
        raise RuntimeError("Pinned D-FINE train.py or single-class config is missing")
    output = root / "runs" / "dfine-n-v1"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(train_script),
        "-c",
        str(config),
        "--use-amp",
        "--seed=20260825",
        f"--output-dir={output}",
    ]
    subprocess.run(command, cwd=args.dfine_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
