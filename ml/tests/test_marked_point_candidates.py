import pytest

from crrc_vision.marked_point_candidates import Proposal, union_proposals


def test_a_only_and_b_only_candidates_both_survive():
    rows = [
        Proposal("a.jpg", "a1", "color_mark", (10, 10, 30, 30), 0.2),
        Proposal("a.jpg", "b1", "fastener_p2", (100, 100, 130, 130), 0.01),
    ]
    fused = union_proposals(rows, iou_threshold=0.60)
    assert len(fused) == 2
    assert {tuple(row.sources) for row in fused} == {
        ("color_mark",),
        ("fastener_p2",),
    }


def test_overlap_retains_both_sources():
    rows = [
        Proposal("a.jpg", "a1", "color_mark", (10, 10, 40, 40), 0.2),
        Proposal("a.jpg", "b1", "fastener_p2", (11, 11, 41, 41), 0.01),
    ]
    fused = union_proposals(rows, iou_threshold=0.60)
    assert len(fused) == 1
    assert fused[0].sources == ("color_mark", "fastener_p2")
    assert fused[0].member_ids == ("a1", "b1")


def test_complete_link_prevents_bridge_merging_adjacent_entities():
    rows = [
        Proposal("a.jpg", "a", "color_mark", (0, 0, 20, 20), 0.1),
        Proposal("a.jpg", "b", "fastener_p2", (5, 0, 25, 20), 0.1),
        Proposal("a.jpg", "c", "color_mark", (10, 0, 30, 20), 0.1),
    ]
    fused = union_proposals(rows, iou_threshold=0.50)
    assert len(fused) == 2
    assert max(len(row.member_ids) for row in fused) == 2


def test_low_scores_and_distinct_images_are_never_deleted():
    rows = [
        Proposal("a.jpg", "a", "color_mark", (1, 1, 10, 10), 0.0),
        Proposal("b.jpg", "b", "color_mark", (1, 1, 10, 10), 0.0),
    ]
    fused = union_proposals(rows, iou_threshold=0.60)
    assert len(fused) == 2
    assert {row.relative_path for row in fused} == {"a.jpg", "b.jpg"}


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError, match="INVALID_PROPOSAL_BOX"):
        union_proposals(
            [Proposal("a.jpg", "a", "color_mark", (2, 2, 1, 3), 0.1)]
        )
