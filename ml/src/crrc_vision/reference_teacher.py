"""Model-independent contracts for an isolated reference labeling teacher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

ALLOWED_GLOBAL_PREFIXES = ("torch.nn.modules.", "ultralytics.")
TEACHER_CATEGORY_MAP = {0: "fastener", 1: "pipe_joint", 2: "fastener"}
MAPPING_STATUS = "inferred_unconfirmed"
ULTRALYTICS_VERSION = "8.2.40"
PROPOSAL_CATEGORIES = {"fastener": 1, "pipe_joint": 2}


def parse_teacher_selection(data: bytes, suffix: str) -> list[dict[str, object]]:
    """Load either a curated selection JSON or the complete JSONL manifest."""

    text = data.decode("utf-8")
    if suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        document = json.loads(text)
        values = document.get("items", []) if isinstance(document, dict) else []
    if not values or any(not isinstance(row, dict) for row in values):
        raise ValueError("teacher selection is empty or invalid")
    required = {"relative_path", "scene_group", "split"}
    if any(not required <= set(row) for row in values):
        raise ValueError("teacher selection rows are missing required fields")
    return values


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


def _clip_xywh(
    box: Iterable[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = (float(value) for value in box)
    left = min(float(width), max(0.0, x))
    top = min(float(height), max(0.0, y))
    right = min(float(width), max(left, x + max(0.0, box_width)))
    bottom = min(float(height), max(top, y + max(0.0, box_height)))
    return left, top, right - left, bottom - top


def build_proposal_document(
    selection: list[dict[str, object]],
    manifest: dict[str, dict[str, object]],
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    """Convert teacher output into an explicitly non-truth COCO review document."""
    images: list[dict[str, object]] = []
    image_by_path: dict[str, dict[str, object]] = {}
    for image_id, selected in enumerate(selection, start=1):
        relative_path = str(selected["relative_path"])
        if relative_path in image_by_path:
            raise ValueError(f"duplicate selected image: {relative_path}")
        source = manifest.get(relative_path)
        if source is None:
            raise ValueError(f"selected image is missing from manifest: {relative_path}")
        image = {
            "id": image_id,
            "file_name": relative_path,
            "width": int(source["width"]),
            "height": int(source["height"]),
            "scene_group": str(selected["scene_group"]),
            "split": str(selected["split"]),
            "image_review_status": "unreviewed",
        }
        images.append(image)
        image_by_path[relative_path] = image

    annotations: list[dict[str, object]] = []
    ordered_predictions = sorted(
        predictions,
        key=lambda row: (str(row["relative_path"]), str(row.get("id", ""))),
    )
    for annotation_id, prediction in enumerate(ordered_predictions, start=1):
        relative_path = str(prediction["relative_path"])
        image = image_by_path.get(relative_path)
        if image is None:
            raise ValueError(f"prediction image is outside selection: {relative_path}")
        category = str(prediction["mapped_category"])
        if category not in PROPOSAL_CATEGORIES:
            raise ValueError(f"unsupported mapped category: {category}")
        bbox = _clip_xywh(
            prediction["bbox"],
            width=int(image["width"]),
            height=int(image["height"]),
        )
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ValueError(f"empty teacher box: {prediction.get('id', annotation_id)}")
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image["id"],
                "category_id": PROPOSAL_CATEGORIES[category],
                "bbox": list(bbox),
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "review_status": "unreviewed",
                "proposal_id": str(prediction.get("id", annotation_id)),
                "proposal_source": "reference-yolov8s",
                "proposal_score": float(prediction["score"]),
                "teacher_class_id": int(prediction["teacher_class_id"]),
                "teacher_class_name": str(prediction["teacher_class_name"]),
                "mapping_status": MAPPING_STATUS,
            }
        )
    return {
        "info": {
            "version": "reference-teacher-proposals-v1",
            "truth_status": "not_training_truth",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "fastener"},
            {"id": 2, "name": "pipe_joint"},
        ],
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
    pass_id: str = "full-640"
    tile: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.pass_id:
            raise ValueError("teacher prediction pass_id is required")
        if self.tile is not None:
            required = {"index", "x1", "y1", "x2", "y2"}
            if set(self.tile) != required:
                raise ValueError("tile metadata must contain index and xyxy bounds")

    @property
    def stable_id(self) -> str:
        raw = (
            f"{self.relative_path}|{self.pass_id}|{self.tile}|"
            f"{self.teacher_class_id}|{self.bbox}".encode("utf-8")
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
