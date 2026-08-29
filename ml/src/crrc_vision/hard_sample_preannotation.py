from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .synthetic_witness_mark import (
    extract_witness_mark_geometry,
    extract_witness_mark_mask,
)
from .witness_state_contract import H1_INTENTS


def preannotate_h1(image: np.ndarray, *, intent: str) -> dict[str, object]:
    if intent not in H1_INTENTS:
        raise ValueError(f"unknown H1 intent: {intent}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR three-channel array")
    height, width = image.shape[:2]
    mask = extract_witness_mark_mask(
        image,
        (0.0, 0.0, float(width), float(height)),
        padding_fraction=0.0,
    )
    mask_pixels = int(np.count_nonzero(mask))
    reasons: list[str] = []
    bbox: list[int] | None = None
    geometry: dict[str, object] | None = None

    if intent == "LOOKALIKE":
        has_marked_point = False
    else:
        has_marked_point = True
        y_values, x_values = np.nonzero(mask)
        if len(x_values) == 0:
            reasons.append("NO_PAINT_CANDIDATE")
        else:
            bbox = [
                int(x_values.min()),
                int(y_values.min()),
                int(x_values.max()) + 1,
                int(y_values.max()) + 1,
            ]
            try:
                geometry = asdict(extract_witness_mark_geometry(mask))
            except ValueError:
                reasons.append("GEOMETRY_NOT_EXTRACTABLE")

    return {
        "review_status": "UNREVIEWED",
        "intent": intent,
        "has_marked_point": has_marked_point,
        "bbox_xyxy": bbox,
        "paint_mask_pixels": mask_pixels,
        "paint_mask": mask,
        "geometry_proposal": geometry,
        "uncertainty_reasons": reasons,
    }
