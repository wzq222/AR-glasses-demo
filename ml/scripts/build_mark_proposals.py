"""Run the color-mark proposal branch on the frozen marked-point selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crrc_vision.assets import asset_root
from crrc_vision.mark_proposals import find_color_mark_proposals


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


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"IMAGE_DECODE_FAILED:{path}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default="selections/marked-point-v1/selection.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output", default="runs/marked-point-proposals-v1/a-color/proposals.json"
    )
    parser.add_argument("--minimum-area", type=int, default=8)
    args = parser.parse_args()

    root = asset_root()
    selection_path = _below(root, args.selection)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_path = _below(root, args.output)
    if not selection_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError("selection or formal truth missing")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_path.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(truth_path)
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    selection = _load(selection_path)
    if selection.get("old_sealed_test_opened") is not False:
        raise ValueError("SEALED_STATUS_INVALID")
    train = selection.get("train")
    val = selection.get("val")
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError("SELECTION_INVALID")
    if len(train) != 40 or len(val) != 19:
        raise ValueError(f"SELECTION_COUNT_INVALID:{len(train)}:{len(val)}")

    forbidden = selection.get("forbidden_old_sealed")
    if not isinstance(forbidden, dict):
        raise ValueError("FORBIDDEN_IDENTITIES_MISSING")
    forbidden_values = {
        "scene_group": set(forbidden.get("scenes", [])),
        "relative_path": set(forbidden.get("paths", [])),
        "sha256": {str(value).lower() for value in forbidden.get("sha256", [])},
        "image_id": set(forbidden.get("image_ids", [])),
    }

    images: list[dict[str, object]] = []
    proposals: list[dict[str, object]] = []
    color_counts: Counter[str] = Counter()
    seen_scenes: set[str] = set()
    for partition, rows in (("train", train), ("val", val)):
        for source in rows:
            if not isinstance(source, dict):
                raise ValueError("SELECTION_ROW_INVALID")
            relative = str(source.get("relative_path") or "").replace("\\", "/")
            scene = str(source.get("scene_group") or "")
            digest = str(source.get("sha256") or "").lower()
            image_id = source.get("image_id")
            values = {
                "scene_group": scene,
                "relative_path": relative,
                "sha256": digest,
                "image_id": image_id,
            }
            if any(values[field] in forbidden_values[field] for field in values):
                raise ValueError(f"OLD_SEALED_IMAGE_FORBIDDEN:{relative}")
            if scene in seen_scenes:
                raise ValueError(f"DUPLICATE_SCENE:{scene}")
            seen_scenes.add(scene)
            source_path = source_root / relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if _sha256(source_path).lower() != digest:
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{relative}")
            image = _read_image(source_path)
            found = find_color_mark_proposals(
                image, minimum_area=args.minimum_area
            )
            images.append(
                {
                    "id": image_id,
                    "relative_path": relative,
                    "scene_group": scene,
                    "partition": partition,
                    "sha256": digest,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "proposal_count": len(found),
                }
            )
            for index, proposal in enumerate(found):
                geometry = asdict(proposal)
                candidate_id = hashlib.sha256(
                    (
                        f"{relative}|{proposal.color}|{proposal.mark_xyxy}|{index}"
                    ).encode("utf-8")
                ).hexdigest()[:16]
                proposals.append(
                    {
                        "id": candidate_id,
                        "image_id": image_id,
                        "relative_path": relative,
                        **geometry,
                    }
                )
                color_counts[proposal.color] += 1

    implementation_path = Path(__file__).parents[1] / "src/crrc_vision/mark_proposals.py"
    try:
        code_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code_head = "unknown"
    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_PROPOSALS")
    document: dict[str, object] = {
        "schema_version": "marked-point-color-proposals-v1",
        "input_hashes": {
            "selection_sha256": _sha256(selection_path),
            "formal_truth_sha256": truth_before,
            "implementation_sha256": _sha256(implementation_path),
        },
        "config": {
            "minimum_area": args.minimum_area,
            "hsv_red": "H<=15 or H>=165, S>=38, V>=22",
            "hsv_yellow": "14<=H<=43, S>=32, V>=28",
            "lab_red": "L>=12, a>=143, a-b>=4",
            "lab_yellow": "L>=18, b>=145, b-a>=16",
            "roi_side": "clamp(6*mark_long_axis,96,320)",
            "color_union_before_components": True,
            "mask_closing_kernel": 3,
            "roi_overlap_deletes_candidates": False,
            "code_head": code_head,
        },
        "images": images,
        "proposals": proposals,
        "stats": {
            "images": len(images),
            "zero_proposal_images": sum(
                int(row["proposal_count"] == 0) for row in images
            ),
            "proposals": len(proposals),
            "by_color": dict(sorted(color_counts.items())),
        },
        "formal_truth_sha256_before": truth_before,
        "formal_truth_sha256_after": truth_after,
        "old_sealed_test_opened": False,
    }
    _atomic_json(output_path, document)
    print(
        json.dumps(
            {
                **document["stats"],  # type: ignore[dict-item]
                "formal_truth_sha256": truth_after,
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
