import numpy as np

from crrc_vision.prelabel import BoundingBox, LineSegment, MarkedFastenerCandidate, VisionPoint
import pytest

from crrc_vision.review import apply_decisions, render_overlay


def test_render_overlay_preserves_image_size():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    candidate = MarkedFastenerCandidate(
        bbox=BoundingBox(10, 20, 30, 40),
        line=LineSegment(VisionPoint(12, 30), VisionPoint(32, 40), 0.9),
        mark_color="red",
        confidence=0.8,
        mark_area=50.0,
    )

    rendered = render_overlay(image, [candidate], label="sample.jpg")

    assert rendered.shape == image.shape
    assert np.any(rendered != image)
    assert not np.shares_memory(rendered, image)


def test_apply_decisions_rejects_unknown_candidate_id():
    rows = [{"candidate_id": "10", "decision": ""}]

    with pytest.raises(ValueError, match="unknown candidate IDs"):
        apply_decisions(rows, {11: "accept"})


def test_apply_decisions_updates_candidate_rows():
    rows = [{"candidate_id": "10", "decision": ""}, {"candidate_id": "", "decision": ""}]

    updated = apply_decisions(rows, {10: "reject"})

    assert updated[0]["decision"] == "reject"
    assert updated[1]["decision"] == ""
