"""Combine color-mark and generic-fastener proposals without score deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_candidates import Proposal, union_proposals


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default="selections/marked-point-v1/selection.json"
    )
    parser.add_argument(
        "--color-proposals",
        default="runs/marked-point-proposals-v1/a-color/proposals.json",
    )
    parser.add_argument(
        "--fastener-candidates", default="runs/safe-auto-candidates-v2.2/candidates.json"
    )
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output", default="runs/marked-point-proposals-v1/union/candidates.json"
    )
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    args = parser.parse_args()

    root = asset_root()
    paths = {
        name: _below(root, relative)
        for name, relative in {
            "selection": args.selection,
            "color": args.color_proposals,
            "fastener": args.fastener_candidates,
            "truth": args.truth,
            "output": args.output,
        }.items()
    }
    for name in ("selection", "color", "fastener", "truth"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    output_path = paths["output"]
    if output_path.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(paths["truth"])
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    selection = _load(paths["selection"])
    color_document = _load(paths["color"])
    fastener_document = _load(paths["fastener"])
    if selection.get("old_sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID")

    selected: dict[str, dict[str, object]] = {}
    for partition in ("train", "val"):
        rows = selection.get(partition)
        if not isinstance(rows, list):
            raise ValueError(f"SELECTION_INVALID:{partition}")
        for source in rows:
            if not isinstance(source, dict):
                raise ValueError("SELECTION_ROW_INVALID")
            relative = str(source.get("relative_path") or "").replace("\\", "/")
            if not relative or relative in selected:
                raise ValueError(f"SELECTION_IDENTITY_INVALID:{relative}")
            selected[relative] = {**source, "partition": partition}
    if len(selected) != 59:
        raise ValueError(f"SELECTION_COUNT_INVALID:{len(selected)}")

    forbidden = selection.get("forbidden_old_sealed")
    if not isinstance(forbidden, dict):
        raise ValueError("FORBIDDEN_IDENTITIES_MISSING")
    forbidden_paths = set(forbidden.get("paths", []))
    forbidden_hashes = {str(value).lower() for value in forbidden.get("sha256", [])}
    for relative, row in selected.items():
        if relative in forbidden_paths or str(row.get("sha256")).lower() in forbidden_hashes:
            raise ValueError(f"OLD_SEALED_IMAGE_FORBIDDEN:{relative}")

    proposals: list[Proposal] = []
    color_rows = color_document.get("proposals")
    if not isinstance(color_rows, list):
        raise ValueError("COLOR_PROPOSALS_INVALID")
    for source in color_rows:
        if not isinstance(source, dict):
            raise ValueError("COLOR_PROPOSAL_INVALID")
        relative = str(source.get("relative_path") or "").replace("\\", "/")
        if relative not in selected:
            raise ValueError(f"COLOR_PROPOSAL_OUTSIDE_SELECTION:{relative}")
        geometry = {
            "source": "color_mark",
            "color": source.get("color"),
            "mark_xyxy": source.get("mark_xyxy"),
            "line_xyxy": source.get("line_xyxy"),
            "area": source.get("area"),
            "elongation": source.get("elongation"),
        }
        proposals.append(
            Proposal(
                relative_path=relative,
                proposal_id=str(source.get("id") or ""),
                source="color_mark",
                xyxy=tuple(float(value) for value in source["roi_xyxy"]),
                score=float(source.get("score", 0.0)),
                image_id=source.get("image_id"),
                geometry=geometry,
            )
        )

    fastener_images = fastener_document.get("images")
    fastener_rows = fastener_document.get("fused_candidates")
    if not isinstance(fastener_images, list) or not isinstance(fastener_rows, list):
        raise ValueError("FASTENER_CANDIDATES_INVALID")
    fastener_image_by_id = {
        source.get("id"): source
        for source in fastener_images
        if isinstance(source, dict) and source.get("id") is not None
    }
    selected_fastener_paths = {
        str(source.get("relative_path") or "").replace("\\", "/")
        for source in fastener_images
        if isinstance(source, dict)
        and str(source.get("relative_path") or "").replace("\\", "/") in selected
    }
    if selected_fastener_paths != set(selected):
        missing = sorted(set(selected) - selected_fastener_paths)
        raise ValueError(f"SELECTED_IMAGE_MISSING_FROM_FASTENER_SOURCE:{missing[0]}")
    for source in fastener_rows:
        if not isinstance(source, dict):
            raise ValueError("FASTENER_PROPOSAL_INVALID")
        relative = str(source.get("relative_path") or "").replace("\\", "/")
        if relative not in selected:
            continue
        image_row = fastener_image_by_id.get(source.get("image_id"))
        if image_row is None or str(image_row.get("relative_path")) != relative:
            raise ValueError(f"FASTENER_IMAGE_IDENTITY_MISMATCH:{relative}")
        proposals.append(
            Proposal(
                relative_path=relative,
                proposal_id=f"b-{source.get('id')}",
                source="fastener_v2_2",
                xyxy=tuple(float(value) for value in source["xyxy"]),
                score=float(source.get("score", 0.0)),
                image_id=source.get("image_id"),
                geometry={
                    "source": "fastener_v2_2",
                    "category": source.get("category"),
                    "supporting_families": source.get("supporting_families", []),
                    "consensus_status": source.get("consensus_status"),
                },
            )
        )

    raw_counts = Counter(row.source for row in proposals)
    fused = union_proposals(proposals, iou_threshold=args.iou_threshold)
    union_counts = Counter(
        "both" if len(row.sources) > 1 else f"{row.sources[0]}_only"
        for row in fused
    )
    per_image = Counter(row.relative_path for row in fused)
    images: list[dict[str, object]] = []
    for relative in sorted(selected):
        row = selected[relative]
        images.append(
            {
                "id": row["image_id"],
                "relative_path": relative,
                "scene_group": row["scene_group"],
                "partition": row["partition"],
                "sha256": row["sha256"],
                "width": row.get("width"),
                "height": row.get("height"),
                "candidate_count": per_image[relative],
            }
        )
    fused_rows: list[dict[str, object]] = []
    for row in fused:
        fused_rows.append(
            {
                "id": row.fused_id,
                "image_id": row.image_id,
                "relative_path": row.relative_path,
                "xyxy": list(row.xyxy),
                "member_ids": list(row.member_ids),
                "sources": list(row.sources),
                "score": row.score,
                "member_geometry": list(row.member_geometry),
            }
        )

    truth_after = _sha256(paths["truth"])
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_UNION")
    document: dict[str, object] = {
        "schema_version": "marked-point-union-candidates-v1",
        "input_hashes": {
            "selection_sha256": _sha256(paths["selection"]),
            "color_proposals_sha256": _sha256(paths["color"]),
            "fastener_candidates_sha256": _sha256(paths["fastener"]),
            "formal_truth_sha256": truth_before,
        },
        "config": {
            "iou_threshold": args.iou_threshold,
            "clustering": "complete_link_greedy",
            "score_based_deletion": False,
        },
        "images": images,
        "fused_candidates": fused_rows,
        "stats": {
            "images": len(images),
            "raw_proposals": len(proposals),
            "raw_by_source": dict(sorted(raw_counts.items())),
            "fused_candidates": len(fused_rows),
            "fused_by_source_coverage": dict(sorted(union_counts.items())),
            "zero_candidate_images": sum(
                int(row["candidate_count"] == 0) for row in images
            ),
        },
        "formal_truth_sha256_before": truth_before,
        "formal_truth_sha256_after": truth_after,
        "old_sealed_test_opened": False,
    }
    _atomic_json(output_path, document)
    print(json.dumps({**document["stats"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
