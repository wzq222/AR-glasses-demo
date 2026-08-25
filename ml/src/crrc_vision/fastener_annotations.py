"""Physical-fastener COCO truth contract and guarded training readiness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

FASTENER_CATEGORIES = [
    {"id": 1, "name": "fastener"},
    {"id": 2, "name": "pipe_joint"},
]
VALID_IMAGE_REVIEW = {"unreviewed", "accept", "accept_empty", "needs_manual"}
VALID_ANNOTATION_REVIEW = {"unreviewed", "accept", "reject", "needs_manual"}
COMPLETE_IMAGE_REVIEW = {"accept", "accept_empty"}
COMPLETE_ANNOTATION_REVIEW = {"accept", "reject"}


@dataclass(frozen=True)
class FastenerTruthReport:
    reviewed_groups: int
    accepted_boxes: int
    error_codes: tuple[str, ...]

    @property
    def can_train(self) -> bool:
        return not self.error_codes

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "can_train": self.can_train}


def build_fastener_document(rows: list[dict[str, object]]) -> dict[str, object]:
    images = [
        {
            "id": index,
            "file_name": row["relative_path"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "scene_group": row["scene_group"],
            "split": row["split"],
            "image_review_status": "unreviewed",
        }
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "info": {
            "description": "CRRC physical-fastener truth",
            "version": "fastener-v2",
        },
        "images": images,
        "categories": FASTENER_CATEGORIES,
        "annotations": [],
    }


def _box_is_inside(annotation: dict[str, object], image: dict[str, object]) -> bool:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x, y, width, height = (float(value) for value in bbox)
    return (
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= float(image["width"])
        and y + height <= float(image["height"])
    )


def evaluate_fastener_truth(
    document: dict[str, object], *, minimum_groups: int = 80
) -> FastenerTruthReport:
    errors: set[str] = set()
    images = list(document.get("images", []))
    annotations = list(document.get("annotations", []))
    image_by_id = {row.get("id"): row for row in images}

    category_map = {row.get("id"): row.get("name") for row in document.get("categories", [])}
    if category_map != {1: "fastener", 2: "pipe_joint"}:
        errors.add("INVALID_CATEGORIES")

    image_statuses = [row.get("image_review_status") for row in images]
    if any(status not in VALID_IMAGE_REVIEW for status in image_statuses):
        errors.add("INVALID_IMAGE_REVIEW")
    if any(status not in COMPLETE_IMAGE_REVIEW for status in image_statuses):
        errors.add("UNREVIEWED_IMAGE")

    annotation_statuses = [row.get("review_status") for row in annotations]
    if any(status not in VALID_ANNOTATION_REVIEW for status in annotation_statuses):
        errors.add("INVALID_ANNOTATION_REVIEW")
    if any(status not in COMPLETE_ANNOTATION_REVIEW for status in annotation_statuses):
        errors.add("UNREVIEWED_ANNOTATION")

    accepted = [row for row in annotations if row.get("review_status") == "accept"]
    if not accepted:
        errors.add("NO_ACCEPTED_BOX")

    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            errors.add("UNKNOWN_IMAGE_REFERENCE")
            continue
        if annotation.get("category_id") not in {1, 2}:
            errors.add("INVALID_CATEGORY_REFERENCE")
        if not _box_is_inside(annotation, image):
            errors.add("BOX_OUTSIDE_IMAGE")
        if image.get("image_review_status") == "accept_empty":
            errors.add("BOX_ON_ACCEPTED_EMPTY_IMAGE")
        if annotation.get("review_status") == "accept" and image.get("image_review_status") != "accept":
            errors.add("ACCEPTED_BOX_ON_UNACCEPTED_IMAGE")

    complete_images = [
        row for row in images if row.get("image_review_status") in COMPLETE_IMAGE_REVIEW
    ]
    reviewed_groups = {row.get("scene_group") for row in complete_images}
    if len(reviewed_groups) < minimum_groups:
        errors.add("INSUFFICIENT_REVIEWED_GROUPS")

    group_splits: dict[object, set[object]] = {}
    for row in images:
        group_splits.setdefault(row.get("scene_group"), set()).add(row.get("split"))
    if any(len(splits) > 1 for splits in group_splits.values()):
        errors.add("SCENE_GROUP_LEAKAGE")
    if not any(row.get("split") == "val" for row in complete_images):
        errors.add("EMPTY_VALIDATION")

    return FastenerTruthReport(
        reviewed_groups=len(reviewed_groups),
        accepted_boxes=len(accepted),
        error_codes=tuple(sorted(errors)),
    )

