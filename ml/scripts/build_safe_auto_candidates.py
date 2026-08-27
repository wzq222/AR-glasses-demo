"""Assemble auditable multi-source candidates without modifying formal truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import (
    Candidate,
    DEFAULT_CENTER_DISTANCE_THRESHOLD,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_MAX_AREA_RATIO,
    FusedCandidate,
    fuse_candidates,
    fusion_stats,
    normalize_hsv_document,
    normalize_teacher_payload,
    verify_truth_unchanged,
)
from crrc_vision.prelabel import read_bgr_image
from crrc_vision.temporal import (
    HomographyQuality,
    propagate_between_scenes,
    validate_homography,
)


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain JSON object rows")
    paths = [str(row.get("relative_path") or "") for row in rows]
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest contains empty or duplicate relative paths")
    return rows


def _teacher_coverage(payload: dict[str, Any]) -> set[str]:
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("teacher payload requires image run records")
    return {
        str(row.get("relative_path") or "")
        for row in images
        if isinstance(row, dict)
    }


def _hsv_coverage(document: dict[str, Any]) -> set[str]:
    images = document.get("images")
    if not isinstance(images, list):
        raise ValueError("HSV document requires images")
    return {
        str(row.get("file_name") or "")
        for row in images
        if isinstance(row, dict)
    }


def _match_homography(
    source_path: Path,
    target_path: Path,
) -> tuple[np.ndarray | None, HomographyQuality | None, tuple[str, ...]]:
    source = cv2.cvtColor(read_bgr_image(source_path), cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(read_bgr_image(target_path), cv2.COLOR_BGR2GRAY)
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=2000)
    source_points, source_descriptors = orb.detectAndCompute(source, None)
    target_points, target_descriptors = orb.detectAndCompute(target, None)
    if source_descriptors is None or target_descriptors is None:
        return None, None, ("NO_TEMPORAL_DESCRIPTORS",)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(source_descriptors, target_descriptors),
        key=lambda match: (match.distance, match.queryIdx, match.trainIdx),
    )[:500]
    if len(matches) < 4:
        return None, None, ("TOO_FEW_MATCHES",)
    source_xy = np.float32(
        [source_points[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    target_xy = np.float32(
        [target_points[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(
        source_xy,
        target_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None or mask is None:
        return None, None, ("HOMOGRAPHY_NOT_FOUND",)

    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source_xy, matrix)
    errors = np.linalg.norm(projected[:, 0, :] - target_xy[:, 0, :], axis=1)
    median_error = float(np.median(errors[inlier_mask])) if inlier_mask.any() else 999.0
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    quality = HomographyQuality(
        matches=len(matches),
        inliers=int(inlier_mask.sum()),
        median_error=median_error,
        scale=scale,
    )
    return matrix, quality, validate_homography(quality)


def _score_by_member(candidates: list[Candidate]) -> dict[str, float]:
    return {candidate.stable_id(): candidate.score for candidate in candidates}


def _propagate_direction(
    *,
    source_row: dict[str, Any],
    target_row: dict[str, Any],
    matrix: np.ndarray,
    quality: HomographyQuality,
    base_by_path: dict[str, list[FusedCandidate]],
    member_scores: dict[str, float],
) -> tuple[list[Candidate], tuple[str, ...]]:
    inverse_errors = validate_homography(quality)
    if inverse_errors:
        return [], inverse_errors
    source_path = str(source_row["relative_path"])
    target_path = str(target_row["relative_path"])
    scene = str(source_row["scene_group"])
    propagated: list[Candidate] = []
    for candidate in base_by_path.get(source_path, []):
        if candidate.consensus_status != "consensus_high" or candidate.category is None:
            continue
        try:
            box = propagate_between_scenes(
                scene,
                str(target_row["scene_group"]),
                candidate.xyxy,
                matrix,
                int(target_row["width"]),
                int(target_row["height"]),
            )
        except ValueError:
            continue
        support_scores = [
            member_scores[member_id]
            for member_id in candidate.member_ids
            if member_id in member_scores
        ]
        score = min(0.95, 0.85 * min(support_scores, default=0.5))
        propagated.append(
            Candidate(
                relative_path=target_path,
                source_id=(
                    f"temporal:{source_path}:{target_path}:{candidate.stable_id()}"
                ),
                source_family="temporal",
                category=candidate.category,
                xyxy=box,
                score=score,
            )
        )
    return propagated, ()


def _temporal_candidates(
    rows: list[dict[str, Any]],
    source_root: Path,
    base_candidates: list[Candidate],
    iou_threshold: float,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    base_fused = fuse_candidates(base_candidates, iou_threshold)
    base_by_path: dict[str, list[FusedCandidate]] = defaultdict(list)
    for candidate in base_fused:
        base_by_path[candidate.relative_path].append(candidate)
    member_scores = _score_by_member(base_candidates)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene_group"])].append(row)

    output: list[Candidate] = []
    audit: list[dict[str, Any]] = []
    for scene_group in sorted(groups):
        group = groups[scene_group]
        for source_row, target_row in zip(group, group[1:]):
            source_relative = str(source_row["relative_path"])
            target_relative = str(target_row["relative_path"])
            if source_row.get("split") != target_row.get("split"):
                audit.append(
                    {
                        "source": source_relative,
                        "target": target_relative,
                        "status": "rejected",
                        "errors": ["TEMPORAL_SPLIT_LEAKAGE"],
                    }
                )
                continue
            matrix, quality, errors = _match_homography(
                source_root / source_relative,
                source_root / target_relative,
            )
            if matrix is None or quality is None or errors:
                audit.append(
                    {
                        "source": source_relative,
                        "target": target_relative,
                        "status": "rejected",
                        "errors": list(errors),
                    }
                )
                continue
            forward, forward_errors = _propagate_direction(
                source_row=source_row,
                target_row=target_row,
                matrix=matrix,
                quality=quality,
                base_by_path=base_by_path,
                member_scores=member_scores,
            )
            inverse_matrix = np.linalg.inv(matrix)
            inverse_quality = HomographyQuality(
                matches=quality.matches,
                inliers=quality.inliers,
                median_error=quality.median_error,
                scale=1.0 / quality.scale,
            )
            backward, backward_errors = _propagate_direction(
                source_row=target_row,
                target_row=source_row,
                matrix=inverse_matrix,
                quality=inverse_quality,
                base_by_path=base_by_path,
                member_scores=member_scores,
            )
            output.extend(forward)
            output.extend(backward)
            audit.append(
                {
                    "source": source_relative,
                    "target": target_relative,
                    "status": "accepted",
                    "quality": asdict(quality),
                    "forward": len(forward),
                    "backward": len(backward),
                    "errors": list(forward_errors + backward_errors),
                }
            )
    return output, audit


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.stable_id(),
        **asdict(candidate),
        "xyxy": list(candidate.xyxy),
    }


def _fused_dict(
    candidate: FusedCandidate,
    image_ids: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": candidate.stable_id(),
        "image_id": image_ids[candidate.relative_path],
        **asdict(candidate),
        "xyxy": list(candidate.xyxy),
        "member_ids": list(candidate.member_ids),
        "supporting_families": list(candidate.supporting_families),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-predictions", action="append", required=True)
    parser.add_argument("--hsv-annotations", action="append", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/safe-auto-candidates-v2")
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    root = asset_root()
    manifest_path = _below(root, args.manifest)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    teacher_paths = [_below(root, value) for value in args.teacher_predictions]
    hsv_paths = [_below(root, value) for value in args.hsv_annotations]
    for required in [manifest_path, truth_path, *teacher_paths, *hsv_paths]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"candidate output already exists: {output_root}")

    truth_before = _sha256(truth_path)
    manifest_rows = _load_manifest(manifest_path)
    expected_paths = {str(row["relative_path"]) for row in manifest_rows}
    for row in manifest_rows:
        image_path = (source_root / str(row["relative_path"])).resolve()
        if source_root.resolve() not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(image_path)

    raw_candidates: list[Candidate] = []
    input_hashes = {args.manifest: _sha256(manifest_path)}
    observed_teacher_sizes: set[int] = set()
    observed_tile_overlaps: set[float] = set()
    for path, relative in zip(teacher_paths, args.teacher_predictions):
        payload = _load_json(path)
        coverage = _teacher_coverage(payload)
        if coverage != expected_paths:
            raise RuntimeError(
                f"teacher selection coverage mismatch: {relative}: "
                f"missing={len(expected_paths - coverage)} extra={len(coverage - expected_paths)}"
            )
        raw_candidates.extend(normalize_teacher_payload(payload))
        raw_sizes = payload.get("teacher_sizes")
        if not isinstance(raw_sizes, list):
            raw_sizes = [payload.get("imgsz")]
        observed_teacher_sizes.update(
            int(size) for size in raw_sizes if isinstance(size, (int, float))
        )
        overlap = payload.get("tile_overlap")
        if isinstance(overlap, (int, float)):
            observed_tile_overlaps.add(float(overlap))
        input_hashes[relative] = _sha256(path)
    for path, relative in zip(hsv_paths, args.hsv_annotations):
        document = _load_json(path)
        coverage = _hsv_coverage(document)
        if coverage != expected_paths:
            raise RuntimeError(
                f"HSV selection coverage mismatch: {relative}: "
                f"missing={len(expected_paths - coverage)} extra={len(coverage - expected_paths)}"
            )
        raw_candidates.extend(normalize_hsv_document(document))
        input_hashes[relative] = _sha256(path)

    unexpected = {
        candidate.relative_path for candidate in raw_candidates
    } - expected_paths
    if unexpected:
        raise RuntimeError(f"candidate paths outside manifest: {sorted(unexpected)[:3]}")

    temporal: list[Candidate] = []
    temporal_audit: list[dict[str, Any]] = []
    if args.temporal:
        temporal, temporal_audit = _temporal_candidates(
            manifest_rows,
            source_root,
            raw_candidates,
            args.iou,
        )
        raw_candidates.extend(temporal)
    fused = fuse_candidates(raw_candidates, args.iou)

    image_rows: list[dict[str, Any]] = []
    image_ids: dict[str, int] = {}
    for image_id, row in enumerate(manifest_rows, start=1):
        relative_path = str(row["relative_path"])
        image_ids[relative_path] = image_id
        image_rows.append(
            {
                "id": image_id,
                "relative_path": relative_path,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "scene_group": str(row["scene_group"]),
                "split": str(row["split"]),
                "sha256": str(row["sha256"]),
                "synthetic": False,
            }
        )

    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    payload = {
        "schema_version": "safe-auto-candidates-v2",
        "input_hashes": input_hashes,
        "truth_sha256_before": truth_before,
        "truth_sha256_after": truth_after,
        "config": {
            "iou": args.iou,
            "teacher_sizes": sorted(observed_teacher_sizes),
            "tile_overlaps": sorted(observed_tile_overlaps),
            "temporal": args.temporal,
            "dedup": {
                "containment_threshold": DEFAULT_CONTAINMENT_THRESHOLD,
                "normalized_center_distance_threshold": (
                    DEFAULT_CENTER_DISTANCE_THRESHOLD
                ),
                "maximum_area_ratio": DEFAULT_MAX_AREA_RATIO,
                "linkage": "complete",
            },
        },
        "images": image_rows,
        "raw_candidates": [_candidate_dict(value) for value in raw_candidates],
        "fused_candidates": [_fused_dict(value, image_ids) for value in fused],
        "temporal_audit": temporal_audit,
        "errors": [],
        "stats": {
            "images": len(image_rows),
            "temporal_candidates": len(temporal),
            **fusion_stats(raw_candidates, fused),
        },
    }
    output_root.mkdir(parents=True, exist_ok=False)
    output_path = output_root / "candidates.json"
    temporary = output_root / "candidates.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(payload["stats"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
