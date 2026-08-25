"""Apply candidate-level review decisions and emit a quality summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from crrc_vision.assets import asset_root
from crrc_vision.review import apply_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-pack", default="review-packs/prelabel-v2")
    parser.add_argument("--decisions", default="review-packs/prelabel-v2/ai-decisions-v1.json")
    args = parser.parse_args()

    root = asset_root()
    index_path = root / args.review_pack / "review-index.csv"
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    raw = json.loads((root / args.decisions).read_text(encoding="utf-8"))
    decisions = {
        int(candidate_id): decision
        for decision, candidate_ids in raw.items()
        for candidate_id in candidate_ids
    }
    if len(decisions) != sum(len(candidate_ids) for candidate_ids in raw.values()):
        raise ValueError("candidate ID appears in more than one decision list")
    rows = apply_decisions(rows, decisions)
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["decision"] for row in rows if row.get("candidate_id"))
    reviewed = counts["accept"] + counts["reject"]
    summary = {
        "candidate_count": sum(counts.values()),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "needs_manual": counts["needs_manual"],
        "unreviewed": counts[""],
        "reviewed_precision": counts["accept"] / reviewed if reviewed else 0.0,
        "training_gate": "pass" if reviewed and counts["accept"] / reviewed >= 0.8 else "fail",
    }
    output = root / args.review_pack / "audit-summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
