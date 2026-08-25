"""Model-independent proposal records and pilot expansion policy."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
TEXT_PROMPT = "bolt. nut. screw. fastener. pipe joint."
TRANSFORMERS_VERSION = "4.40.2"


def validate_transformers_version(actual: str) -> tuple[str, ...]:
    return () if actual == TRANSFORMERS_VERSION else ("INCOMPATIBLE_TRANSFORMERS_VERSION",)


def validate_loading_info(info: dict[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    if info.get("missing_keys"):
        errors.append("MISSING_MODEL_WEIGHTS")
    if info.get("unexpected_keys"):
        errors.append("UNEXPECTED_MODEL_WEIGHTS")
    if info.get("mismatched_keys"):
        errors.append("MISMATCHED_MODEL_WEIGHTS")
    return tuple(errors)


def clip_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = box
    left, top = max(0.0, x), max(0.0, y)
    right = min(float(width), x + box_width)
    bottom = min(float(height), y + box_height)
    return left, top, max(0.0, right - left), max(0.0, bottom - top)


@dataclass(frozen=True)
class Proposal:
    relative_path: str
    category: str
    bbox: tuple[float, float, float, float]
    score: float
    source: str

    @property
    def stable_id(self) -> str:
        raw = (
            f"{self.relative_path}|{self.category}|{self.bbox}|{self.source}".encode("utf-8")
        )
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {"id": self.stable_id, **asdict(self), "bbox": list(self.bbox)}


@dataclass(frozen=True)
class PilotAudit:
    accepted: int
    rejected: int
    images_with_missed_targets: int
    images: int

    @property
    def reviewed_precision(self) -> float:
        reviewed = self.accepted + self.rejected
        return self.accepted / reviewed if reviewed else 0.0

    @property
    def missed_image_rate(self) -> float:
        return self.images_with_missed_targets / self.images if self.images else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "reviewed_precision": self.reviewed_precision,
            "missed_image_rate": self.missed_image_rate,
            "can_expand": pilot_can_expand(self),
        }


def pilot_can_expand(audit: PilotAudit) -> bool:
    return audit.reviewed_precision >= 0.50 and audit.missed_image_rate <= 0.30


def _density_bucket(candidate_count: int) -> int:
    if candidate_count == 0:
        return 0
    if candidate_count <= 2:
        return 1
    if candidate_count <= 7:
        return 2
    return 3


def select_pilot_items(items: list[dict[str, object]], *, count: int = 12) -> list[dict[str, object]]:
    if count < 1 or count > len(items):
        raise ValueError("count must fit available selection items")
    strata: dict[tuple[str, int], list[dict[str, object]]] = {}
    for item in items:
        key = (str(item["split"]), _density_bucket(int(item["candidate_count"])))
        strata.setdefault(key, []).append(item)
    for values in strata.values():
        values.sort(
            key=lambda item: (
                -float(item["focus_score"]),
                str(item["scene_group"]),
                str(item["relative_path"]),
            )
        )

    selected: list[dict[str, object]] = []
    keys = sorted(strata)
    while len(selected) < count:
        progress = False
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop(0))
                progress = True
        if not progress:
            break
    return sorted(selected, key=lambda item: (str(item["split"]), str(item["scene_group"])))
