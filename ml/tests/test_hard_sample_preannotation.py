import numpy as np

from crrc_vision.hard_sample_preannotation import preannotate_h1


def test_preannotation_never_self_approves() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[60:66, 30:98, 2] = 255
    result = preannotate_h1(image, intent="ALIGNED")
    assert result["review_status"] == "UNREVIEWED"
    assert result["paint_mask_pixels"] > 0


def test_lookalike_has_no_marked_point_box() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = preannotate_h1(image, intent="LOOKALIKE")
    assert result["has_marked_point"] is False
    assert result["bbox_xyxy"] is None


def test_missing_paint_is_uncertain_not_rejected() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = preannotate_h1(image, intent="SUBTLE_DISPLACED")
    assert result["review_status"] == "UNREVIEWED"
    assert "NO_PAINT_CANDIDATE" in result["uncertainty_reasons"]
