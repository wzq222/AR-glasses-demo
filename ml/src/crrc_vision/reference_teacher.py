"""Model-independent contracts for an isolated reference labeling teacher."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

ALLOWED_GLOBAL_PREFIXES = ("torch.nn.modules.", "ultralytics.")
TEACHER_CATEGORY_MAP = {0: "fastener", 1: "pipe_joint", 2: "fastener"}
MAPPING_STATUS = "inferred_unconfirmed"
ULTRALYTICS_VERSION = "8.2.40"


def build_run_manifest(
    *,
    checkpoint_sha256: str,
    selection_sha256: str,
    truth_sha256: str,
    images: int,
    predictions: int,
) -> dict[str, object]:
    """Build the minimum integrity record for an isolated teacher run."""
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "selection_sha256": selection_sha256,
        "truth_sha256_before": truth_sha256,
        "truth_sha256_after": truth_sha256,
        "images": images,
        "predictions": predictions,
        "safe_load": "weights_only",
        "mapping_status": MAPPING_STATUS,
        "research_only": True,
    }


def validate_checkpoint_globals(names: Iterable[str]) -> tuple[str, ...]:
    """Reject checkpoint globals outside the reviewed framework namespaces."""
    return (
        ()
        if all(str(name).startswith(ALLOWED_GLOBAL_PREFIXES) for name in names)
        else ("UNSAFE_CHECKPOINT_GLOBAL",)
    )


def validate_ultralytics_version(actual: str) -> tuple[str, ...]:
    """Keep checkpoint code compatibility tied to its recorded runtime."""
    return (
        ()
        if actual == ULTRALYTICS_VERSION
        else ("INCOMPATIBLE_ULTRALYTICS_VERSION",)
    )


def xyxy_to_xywh(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Clip a corner box to the image and return COCO xywh coordinates."""
    x1, y1, x2, y2 = box
    left = min(float(width), max(0.0, x1))
    top = min(float(height), max(0.0, y1))
    right = min(float(width), max(left, x2))
    bottom = min(float(height), max(top, y2))
    return left, top, right - left, bottom - top


def map_teacher_category(class_id: int) -> tuple[str, str]:
    """Map an original teacher class while preserving its unconfirmed status."""
    if class_id not in TEACHER_CATEGORY_MAP:
        raise ValueError(f"unsupported teacher class: {class_id}")
    return TEACHER_CATEGORY_MAP[class_id], MAPPING_STATUS


def ensure_complete_selection(
    expected: Iterable[str], actual: Iterable[str]
) -> tuple[str, ...]:
    """Require one processed record for every selected image and no extras."""
    return (
        ()
        if sorted(str(value) for value in expected)
        == sorted(str(value) for value in actual)
        else ("INCOMPLETE_SELECTION_COVERAGE",)
    )


@dataclass(frozen=True)
class TeacherPrediction:
    relative_path: str
    teacher_class_id: int
    teacher_class_name: str
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def stable_id(self) -> str:
        raw = (
            f"{self.relative_path}|{self.teacher_class_id}|{self.bbox}".encode("utf-8")
        )
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        category, mapping_status = map_teacher_category(self.teacher_class_id)
        return {
            "id": self.stable_id,
            **asdict(self),
            "bbox": list(self.bbox),
            "mapped_category": category,
            "mapping_status": mapping_status,
            "review_status": "unreviewed",
        }
