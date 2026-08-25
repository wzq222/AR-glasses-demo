"""Visual review helpers for prelabel quality audits."""

from __future__ import annotations

import cv2
import numpy as np

from .prelabel import MarkedFastenerCandidate


_DECISIONS = {"accept", "reject", "needs_manual"}


def apply_decisions(rows: list[dict[str, str]], decisions: dict[int, str]) -> list[dict[str, str]]:
    candidate_ids = {int(row["candidate_id"]) for row in rows if row.get("candidate_id")}
    unknown = sorted(set(decisions) - candidate_ids)
    if unknown:
        raise ValueError(f"unknown candidate IDs: {unknown}")
    invalid = sorted(set(decisions.values()) - _DECISIONS)
    if invalid:
        raise ValueError(f"invalid decisions: {invalid}")

    updated = [row.copy() for row in rows]
    for row in updated:
        candidate_id = row.get("candidate_id")
        if candidate_id and int(candidate_id) in decisions:
            row["decision"] = decisions[int(candidate_id)]
    return updated


def render_overlay(
    image_bgr: np.ndarray,
    candidates: list[MarkedFastenerCandidate],
    *,
    label: str = "",
) -> np.ndarray:
    rendered = image_bgr.copy()
    for index, candidate in enumerate(candidates, start=1):
        box = candidate.bbox
        color = (40, 40, 255) if candidate.mark_color == "red" else (0, 220, 255)
        cv2.rectangle(rendered, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 3)
        cv2.line(
            rendered,
            (candidate.line.start.x, candidate.line.start.y),
            (candidate.line.end.x, candidate.line.end.y),
            (255, 255, 255),
            3,
        )
        cv2.circle(rendered, (candidate.line.start.x, candidate.line.start.y), 5, (255, 80, 0), -1)
        cv2.circle(rendered, (candidate.line.end.x, candidate.line.end.y), 5, (255, 80, 0), -1)
        cv2.putText(
            rendered,
            f"{index}:{candidate.confidence:.2f}",
            (box.x, max(18, box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if label:
        cv2.rectangle(rendered, (0, 0), (min(rendered.shape[1], 900), 38), (0, 0, 0), -1)
        cv2.putText(rendered, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return rendered
