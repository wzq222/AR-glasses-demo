"""Build an auditable marked-point review and a blind geometry-check pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from crrc_vision.assets import asset_root
from crrc_vision.marked_point import (
    build_manual_positive_records,
    build_review_set,
    filter_then_deduplicate_positive_records,
    unreviewed_positive_ids,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{relative}")
    return path


def _records(
    document: dict[str, Any], selected_ids: list[str], *, source: str, rank: int
) -> list[dict[str, object]]:
    rows = document.get("records")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"INVALID_SHORTLIST:{source}")
    by_id = {str(row.get("shortlist_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"DUPLICATE_SHORTLIST_ID:{source}")
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"UNKNOWN_SHORTLIST_ID:{source}:{missing[0]}")
    output: list[dict[str, object]] = []
    for shortlist_id in selected_ids:
        row = by_id[shortlist_id]
        output.append(
            {
                "positive_id": f"{source}-{shortlist_id}",
                "source_rank": rank,
                "first_pass_source": source,
                "first_pass_shortlist_id": shortlist_id,
                "relative_path": row["relative_path"],
                "image_id": row["image_id"],
                "source_sha256": row["source_sha256"],
                # The physical candidate is the target. The colored mask is context,
                # and may cross into a neighboring inspection point.
                "xyxy": row["fastener_xyxy"],
                "dedupe_xyxy": row["fastener_xyxy"],
                "mark_colors": row.get("mark_colors", []),
            }
        )
    return output


def _write_second_pass_pack(
    *,
    pack_root: Path,
    source_root: Path,
    positives: list[dict[str, object]],
) -> None:
    if pack_root.exists():
        raise FileExistsError(f"SECOND_PASS_PACK_ALREADY_EXISTS:{pack_root}")
    crop_root = pack_root / "crops"
    sheet_root = pack_root / "sheets"
    crop_root.mkdir(parents=True)
    sheet_root.mkdir()
    neutral_rows: list[dict[str, object]] = []
    image_cache: dict[str, Image.Image] = {}
    for index, row in enumerate(positives, 1):
        relative = str(row["relative_path"])
        if relative not in image_cache:
            with Image.open(source_root / relative) as opened:
                image_cache[relative] = ImageOps.exif_transpose(opened).convert("RGB")
        source = image_cache[relative]
        x1, y1, x2, y2 = (float(value) for value in row["xyxy"])  # type: ignore[arg-type]
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(180.0, 3.0 * max(x2 - x1, y2 - y1))
        left = max(0, math.floor(center_x - side / 2.0))
        top = max(0, math.floor(center_y - side / 2.0))
        right = min(source.width, math.ceil(center_x + side / 2.0))
        bottom = min(source.height, math.ceil(center_y + side / 2.0))
        crop = source.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(crop)
        draw.rectangle(
            (x1 - left, y1 - top, x2 - left, y2 - top),
            outline=(0, 255, 0),
            width=4,
        )
        neutral_id = f"P{index:04d}"
        draw.text(
            (6, 6), neutral_id, fill="white", stroke_width=3, stroke_fill="black"
        )
        crop.thumbnail((380, 330), Image.Resampling.LANCZOS)
        crop_path = crop_root / f"{neutral_id}.jpg"
        crop.save(crop_path, quality=94)
        neutral_rows.append(
            {
                "neutral_id": neutral_id,
                "positive_id": row["positive_id"],
                "relative_path": relative,
                "crop": str(crop_path.relative_to(pack_root)).replace("\\", "/"),
                "crop_sha256": _sha256(crop_path),
            }
        )
    for sheet_index in range(math.ceil(len(neutral_rows) / 16)):
        page = neutral_rows[sheet_index * 16 : (sheet_index + 1) * 16]
        sheet = Image.new("RGB", (1600, 1440), "#202020")
        draw = ImageDraw.Draw(sheet)
        for cell, row in enumerate(page):
            column, line = cell % 4, cell // 4
            with Image.open(pack_root / str(row["crop"])) as opened:
                tile = opened.convert("RGB")
            x, y = column * 400, line * 360
            sheet.paste(tile, (x + (400 - tile.width) // 2, y + 25))
            draw.text((x + 5, y + 5), str(row["neutral_id"]), fill="white")
        sheet.save(sheet_root / f"sheet-{sheet_index + 1:03d}.jpg", quality=94)
    manifest = {
        "schema_version": "marked-point-blind-second-pass-v1",
        "first_pass_labels_rendered": False,
        "records": neutral_rows,
        "stats": {
            "positives": len(neutral_rows),
            "sheets": math.ceil(len(neutral_rows) / 16),
        },
    }
    (pack_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selections/marked-point-v1/selection.json")
    parser.add_argument(
        "--candidates", default="runs/marked-point-proposals-v1/union/candidates.json"
    )
    parser.add_argument(
        "--pack-manifest", default="review-packs/marked-point-v1/pack-manifest.json"
    )
    parser.add_argument(
        "--truth-shortlist",
        default="review-packs/marked-point-v1/truth-shortlist/shortlist.json",
    )
    parser.add_argument(
        "--miss-shortlist",
        default="review-packs/marked-point-v1/miss-shortlist/shortlist.json",
    )
    parser.add_argument(
        "--relaxed-shortlist",
        default="review-packs/marked-point-v1/relaxed-uncovered-shortlist/shortlist.json",
    )
    parser.add_argument(
        "--audit-shortlist",
        default="review-packs/marked-point-v1/v1.1-uncovered-shortlist/shortlist.json",
    )
    parser.add_argument(
        "--decisions", default="review-packs/marked-point-v1/shortlist-decisions.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument(
        "--output", default="review-packs/marked-point-v1/review-first-pass.json"
    )
    parser.add_argument(
        "--second-pass-pack",
        default="review-packs/marked-point-v1/blind-second-pass-v1",
    )
    parser.add_argument(
        "--second-pass-manifest",
        default="review-packs/marked-point-v1/blind-second-pass-v1/manifest.json",
    )
    parser.add_argument("--require-second-pass", action="store_true")
    parser.add_argument("--ignore-second-pass-decisions", action="store_true")
    args = parser.parse_args()

    root = asset_root()
    paths = {
        name: _below(root, relative)
        for name, relative in {
            "selection": args.selection,
            "candidates": args.candidates,
            "pack_manifest": args.pack_manifest,
            "truth_shortlist": args.truth_shortlist,
            "miss_shortlist": args.miss_shortlist,
            "relaxed_shortlist": args.relaxed_shortlist,
            "audit_shortlist": args.audit_shortlist,
            "decisions": args.decisions,
            "source": args.source,
            "output": args.output,
            "second_pass_pack": args.second_pass_pack,
            "second_pass_manifest": args.second_pass_manifest,
        }.items()
    }
    if paths["output"].exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{paths['output']}")
    selection = _load(paths["selection"])
    candidates = _load(paths["candidates"])
    decisions = _load(paths["decisions"])
    selected = decisions.get("selected_positive_ids")
    if not isinstance(selected, dict):
        raise ValueError("SELECTED_POSITIVE_IDS_MISSING")
    raw_positives = _records(
        _load(paths["truth_shortlist"]),
        list(selected.get("truth", [])),
        source="truth",
        rank=0,
    ) + _records(
        _load(paths["miss_shortlist"]),
        list(selected.get("miss", [])),
        source="miss",
        rank=1,
    ) + _records(
        _load(paths["relaxed_shortlist"]),
        list(selected.get("relaxed", [])),
        source="relaxed",
        rank=2,
    ) + _records(
        _load(paths["audit_shortlist"]),
        list(selected.get("audit", [])),
        source="audit",
        rank=3,
    ) + build_manual_positive_records(
        selection,
        decisions.get("manual_additions", []),
    )
    accepted = (
        set()
        if args.ignore_second_pass_decisions
        else set(decisions.get("second_pass_accept_positive_ids", []))
    )
    rejected = (
        set()
        if args.ignore_second_pass_decisions
        else set(decisions.get("second_pass_reject_positive_ids", []))
    )
    neutral_rejects = (
        set()
        if args.ignore_second_pass_decisions
        else set(decisions.get("second_pass_reject_neutral_ids", []))
    )
    if neutral_rejects:
        second_pass_manifest = _load(paths["second_pass_manifest"])
        neutral_rows = second_pass_manifest.get("records")
        if not isinstance(neutral_rows, list):
            raise ValueError("INVALID_SECOND_PASS_MANIFEST")
        neutral_to_positive = {
            str(row.get("neutral_id")): str(row.get("positive_id"))
            for row in neutral_rows
            if isinstance(row, dict)
        }
        unknown_neutral = neutral_rejects - set(neutral_to_positive)
        if unknown_neutral:
            raise ValueError(
                f"UNKNOWN_SECOND_PASS_NEUTRAL_ID:{sorted(unknown_neutral)[0]}"
            )
        rejected.update(neutral_to_positive[value] for value in neutral_rejects)
    raw_ids = {str(row["positive_id"]) for row in raw_positives}
    unknown = (accepted | rejected) - raw_ids
    if unknown:
        raise ValueError(f"UNKNOWN_SECOND_PASS_ID:{sorted(unknown)[0]}")
    if accepted & rejected:
        raise ValueError(f"SECOND_PASS_DECISION_CONFLICT:{sorted(accepted & rejected)[0]}")
    positives, suppressed = filter_then_deduplicate_positive_records(
        raw_positives, rejected
    )
    deduplicated_count = len(positives)
    kept_ids = {str(row["positive_id"]) for row in positives}
    accepted &= kept_ids
    if (
        not args.ignore_second_pass_decisions
        and decisions.get("second_pass_accept_all_remaining") is True
    ):
        accepted = kept_ids - rejected
    unreviewed = unreviewed_positive_ids(kept_ids, accepted, rejected)
    if args.require_second_pass and unreviewed:
        raise ValueError(f"SECOND_PASS_INCOMPLETE:{len(unreviewed)}")
    for row in positives:
        if row["positive_id"] in accepted:
            row["second_pass"] = {
                "first_result_hidden": True,
                "decision": "accept",
                "final_xyxy": row["xyxy"],
            }
    uncertain = decisions.get("uncertain_paths", {})
    if not isinstance(uncertain, dict):
        raise ValueError("INVALID_UNCERTAIN_PATHS")
    fused = candidates.get("fused_candidates")
    if not isinstance(fused, list) or any(not isinstance(row, dict) for row in fused):
        raise ValueError("INVALID_FUSED_CANDIDATES")
    review = build_review_set(
        selection=selection,
        candidates=fused,
        positives=positives,
        uncertain_paths={str(key): str(value) for key, value in uncertain.items()},
    )
    review["input_hashes"] = {
        "selection_sha256": _sha256(paths["selection"]),
        "candidates_sha256": _sha256(paths["candidates"]),
        "pack_manifest_sha256": _sha256(paths["pack_manifest"]),
        "truth_shortlist_sha256": _sha256(paths["truth_shortlist"]),
        "miss_shortlist_sha256": _sha256(paths["miss_shortlist"]),
        "relaxed_shortlist_sha256": _sha256(paths["relaxed_shortlist"]),
        "audit_shortlist_sha256": _sha256(paths["audit_shortlist"]),
        "decisions_sha256": _sha256(paths["decisions"]),
    }
    if neutral_rejects:
        review["input_hashes"]["second_pass_manifest_sha256"] = _sha256(
            paths["second_pass_manifest"]
        )
    review["audit"] = {
        "raw_selected_positives": len(raw_positives),
        "pre_dedup_rejected_positives": len(rejected),
        "deduplicated_positives": deduplicated_count,
        "final_accepted_positives": len(positives),
        "suppressed_duplicates": suppressed,
        "second_pass_accepted": len(accepted),
        "second_pass_rejected": len(rejected),
        "second_pass_complete": not unreviewed,
        "candidate_negative_policy": (
            "Only uncovered candidates are negatives; candidates overlapping an "
            "added truth point use covered_by_added_marked_point."
        ),
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.require_second_pass:
        _write_second_pass_pack(
            pack_root=paths["second_pass_pack"],
            source_root=paths["source"],
            positives=positives,
        )
    print(
        json.dumps(
            {
                "positives": len(positives),
                "suppressed": len(suppressed),
                "second_pass_accepted": len(accepted),
                "second_pass_rejected": len(rejected),
                "output": str(paths["output"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
