import pytest

from crrc_vision.coco import build_coco_document, validate_annotation
from crrc_vision.prelabel import BoundingBox, LineSegment, MarkedFastenerCandidate, VisionPoint


def test_coco_rejects_box_outside_image():
    with pytest.raises(ValueError, match="outside image"):
        validate_annotation(width=100, height=100, bbox=[90, 90, 20, 20])


def test_coco_document_keeps_auditable_candidate_metadata():
    candidate = MarkedFastenerCandidate(
        bbox=BoundingBox(10, 20, 40, 40),
        line=LineSegment(VisionPoint(12, 30), VisionPoint(42, 30), 0.9),
        mark_color="red",
        confidence=0.85,
        mark_area=42.0,
    )

    document = build_coco_document(
        [("image.jpg", 100, 80, "train", "scene-0001", [candidate])],
        algorithm_version="hsv-line-v1",
    )

    assert document["categories"] == [{"id": 1, "name": "marked_fastener"}]
    annotation = document["annotations"][0]
    assert annotation["bbox"] == [10, 20, 40, 40]
    assert annotation["attributes"]["algorithm_version"] == "hsv-line-v1"
    assert annotation["attributes"]["review_status"] == "unreviewed"
    assert annotation["attributes"]["mark_color"] == "red"
    assert annotation["attributes"]["line_points"] == [12, 30, 42, 30]
